"""
Enhanced JetMamba with ParticleTransformer-like features for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from einops import rearrange

# Import optimized Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("mamba-ssm not available - using fallback")

# Import ParT utilities for pairwise features
@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2

def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2

def to_ptrapphim(x, return_mass=True, eps=1e-8):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt((energy**2 - (px**2 + py**2 + pz**2)).clamp(min=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)

def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8):
    """Compute pairwise features like ParticleTransformer"""
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        # Mass of combined system
        pt_sum = torch.sqrt((xi[:, :2] + xj[:, :2]).square().sum(dim=1, keepdim=True))
        e_sum = xi[:, 3:4] + xj[:, 3:4]
        pz_sum = xi[:, 2:3] + xj[:, 2:3]
        m2 = e_sum**2 - pt_sum**2 - pz_sum**2
        lnm2 = torch.log(m2.clamp(min=eps))
        outputs.append(lnm2)

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)

class SequenceTrimmer(nn.Module):
    """ParT-style sequence trimming for efficiency"""
    
    def __init__(self, enabled=False, target=(0.9, 1.02)):
        super().__init__()
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None):
        # x: (N, C, P), v: (N, 4, P), mask: (N, 1, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled and self.training:
            if self._counter < 5:
                self._counter += 1
            else:
                # Random trimming during training
                import random
                q = min(1, random.uniform(*self.target))
                maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                rand = torch.rand_like(mask.type_as(x))
                rand.masked_fill_(~mask, -1)
                perm = rand.argsort(dim=-1, descending=True)
                mask = torch.gather(mask, -1, perm)
                x = torch.gather(x, -1, perm.expand_as(x))
                if v is not None:
                    v = torch.gather(v, -1, perm.expand_as(v))
                
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]

        return x, v, mask

class ParTStyleEmbed(nn.Module):
    """ParticleTransformer-style embedding with normalization"""
    
    def __init__(self, input_dim, embed_dims, normalize_input=True, activation='gelu'):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in embed_dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        # x: (batch, embed_dim, seq_len)
        if self.input_bn is not None:
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()  # -> (seq_len, batch, embed_dim)
        return self.embed(x)

class PairwiseFeatureEmbedding(nn.Module):
    """
    ParT-style pairwise feature computation for Mamba context
    """
    
    def __init__(self, pair_input_dim=4, pair_embed_dims=[64, 64, 64], 
                 remove_self_pair=False, eps=1e-8):
        super().__init__()
        self.pair_input_dim = pair_input_dim
        self.remove_self_pair = remove_self_pair
        self.eps = eps
        self.out_dim = pair_embed_dims[-1]
        
        # Embedding network for pairwise features
        input_dim = pair_input_dim
        module_list = [nn.BatchNorm1d(input_dim)]
        for dim in pair_embed_dims:
            module_list.extend([
                nn.Conv1d(input_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.GELU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list[:-1])  # Remove last activation
        
    def forward(self, v):
        """
        Compute pairwise features from 4-momentum vectors
        
        Args:
            v: (B, 4, N) - 4-momentum vectors [px, py, pz, E]
            
        Returns:
            pair_features: (B, embed_dim, N, N) - pairwise feature matrix
        """
        B, _, N = v.shape
        
        # Compute pairwise features
        v_expanded = v.unsqueeze(-1).expand(-1, -1, -1, N)  # (B, 4, N, N)
        vi = v_expanded  # (B, 4, N, N)
        vj = v_expanded.transpose(-2, -1)  # (B, 4, N, N)
        
        # Reshape for pairwise computation
        vi_flat = vi.reshape(B, 4, N*N)  # (B, 4, N*N)
        vj_flat = vj.reshape(B, 4, N*N)  # (B, 4, N*N)
        
        # Compute pairwise features
        pair_fts = pairwise_lv_fts(vi_flat, vj_flat, self.pair_input_dim, self.eps)
        # pair_fts: (B, pair_input_dim, N*N)
        
        # Remove self-pairs if requested
        if self.remove_self_pair:
            mask = torch.eye(N, device=v.device).bool().flatten()
            pair_fts[:, :, mask] = 0
        
        # Embed pairwise features
        pair_embedded = self.embed(pair_fts)  # (B, embed_dim, N*N)
        
        # Reshape back to matrix form
        pair_matrix = pair_embedded.view(B, self.out_dim, N, N)
        
        return pair_matrix

class MambaWithPairwiseContext(nn.Module):
    """
    Mamba block enhanced with pairwise context (ParT-inspired)
    """
    
    def __init__(self, d_model, d_state=16, dropout=0.1, use_pairwise=True, pair_dim=64):
        super().__init__()
        
        self.use_pairwise = use_pairwise
        self.norm = nn.LayerNorm(d_model)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(d_model=d_model, d_state=d_state)
        else:
            # Fallback to multi-head attention
            self.mamba = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=dropout, batch_first=True
            )
            
        # Pairwise context integration
        if use_pairwise:
            self.pair_proj = nn.Linear(pair_dim, d_model)  # Fix: use pair_dim instead of d_model
            self.context_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, pair_context=None):
        """
        Args:
            x: (B, N, d_model) - particle sequences
            mask: (B, N) - attention mask  
            pair_context: (B, d_model, N, N) - pairwise context from 4-momentum
        """
        # Layer norm
        x_norm = self.norm(x)
        
        if MAMBA_AVAILABLE:
            # Use Mamba
            x_mamba = self.mamba(x_norm)
        else:
            # Fallback: Multi-head attention
            if mask is not None:
                attn_mask = ~mask.bool()
            else:
                attn_mask = None
            x_mamba, _ = self.mamba(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        
        # Integrate pairwise context if available
        if self.use_pairwise and pair_context is not None:
            try:
                # Pool pairwise context to particle level
                pair_pooled = pair_context.mean(dim=-1).transpose(1, 2)  # (B, N, pair_dim)
                pair_proj = self.pair_proj(pair_pooled)
                
                # Gate mechanism to control context integration
                combined = torch.cat([x_mamba, pair_proj], dim=-1)
                gate = self.context_gate(combined)
                x_mamba = x_mamba + gate * pair_proj
            except RuntimeError as e:
                # Fallback: skip pairwise context if dimension mismatch
                print(f"Warning: Skipping pairwise context due to dimension mismatch: {e}")
                pass
        
        # Residual connection
        return x + self.dropout(x_mamba)

class ClassTokenPooling(nn.Module):
    """ParT-style class token pooling"""
    
    def __init__(self, d_model, num_cls_layers=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.cls_token = Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Class attention layers (simplified version of ParT's cls_blocks)
        self.cls_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
            for _ in range(num_cls_layers)
        ])
        self.cls_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_cls_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, d_model) - particle sequences
            mask: (B, N) - validity mask
        Returns:
            cls_output: (B, d_model) - global representation
        """
        B = x.shape[0]
        
        # Expand class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        
        # Prepare mask for attention (include cls token)
        if mask is not None:
            # Add cls token position (always valid)
            cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, 1+N)
            key_padding_mask = ~full_mask.bool()
        else:
            key_padding_mask = None
        
        # Class attention layers
        for layer, norm in zip(self.cls_layers, self.cls_norms):
            # Concatenate cls token with particles
            tokens_with_cls = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, d_model)
            
            # Class attention: cls token attends to all particles
            cls_out, _ = layer(cls_tokens, tokens_with_cls, tokens_with_cls, 
                              key_padding_mask=key_padding_mask)
            cls_tokens = norm(cls_out)
        
        return self.final_norm(cls_tokens).squeeze(1)  # (B, d_model)

class EnhancedJetMamba(nn.Module):
    """
    Enhanced JetMamba with ParticleTransformer-like features for fair comparison
    """
    
    def __init__(self, num_classes=10, d_model=128, n_layers=8, d_state=16,
                 dropout=0.1, embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
                 num_cls_layers=2, use_pairwise=True, trim=True, **kwargs):
        super().__init__()
        
        # Ensure consistency: if d_model is specified, update embed_dims[-1] to match
        if d_model != embed_dims[-1]:
            embed_dims = embed_dims[:-1] + [d_model]
            print(f"  - Adjusted embed_dims to match d_model: {embed_dims}")
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_pairwise = use_pairwise
        
        print(f"Enhanced JetMamba (ParT-style features):")
        print(f"  - d_model: {d_model}, layers: {n_layers}")
        print(f"  - Embed dims: {embed_dims}")
        print(f"  - Pairwise features: {use_pairwise}")
        print(f"  - Class token layers: {num_cls_layers}")
        print(f"  - Using Mamba: {MAMBA_AVAILABLE}")
        
        # Sequence trimming (like ParT)
        self.trimmer = SequenceTrimmer(enabled=trim)
        
        # ParT-style embedding
        self.embed = ParTStyleEmbed(
            input_dim=19,  # 17 features + 2 coordinates
            embed_dims=embed_dims,
            normalize_input=True,
            activation='gelu'
        )
        
        # Pairwise feature embedding (like ParT's pair_embed)
        self.pair_embed = PairwiseFeatureEmbedding(
            pair_input_dim=4,
            pair_embed_dims=pair_embed_dims,
            remove_self_pair=False
        ) if use_pairwise else None
        
        # Mamba layers with pairwise context
        pair_dim = pair_embed_dims[-1] if use_pairwise else 64  # Get actual pair dimension
        final_dim = embed_dims[-1]  # Use final embedding dimension
        self.layers = nn.ModuleList([
            MambaWithPairwiseContext(
                d_model=final_dim,  # Use final embedding dimension 
                d_state=d_state, 
                dropout=dropout,
                use_pairwise=use_pairwise,
                pair_dim=pair_dim  # Pass correct pair dimension
            )
            for _ in range(n_layers)
        ])
        
        # Class token pooling (like ParT)
        self.pooling = ClassTokenPooling(
            d_model=final_dim,  # Use final embedding dimension
            num_cls_layers=num_cls_layers,
            dropout=0.0  # ParT uses 0 dropout for cls layers
        )
        
        # Classification head (like ParT)
        self.classifier = nn.Linear(final_dim, num_classes)  # Use final embedding dimension
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights like ParT"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass compatible with ParT interface
        """
        # Handle arguments like ParT
        if args:
            if len(args) >= 4:
                pf_points = args[0]
                pf_features = args[1]
                pf_vectors = args[2]
                pf_mask = args[3]
            else:
                raise ValueError(f"Expected at least 4 arguments, got {len(args)}")
        else:
            pf_points = kwargs['pf_points']
            pf_features = kwargs['pf_features']
            pf_vectors = kwargs['pf_vectors']
            pf_mask = kwargs['pf_mask']
        
        # Convert to ParT format: (B, C, N)
        features = pf_features  # (B, 17, N)
        v = pf_vectors          # (B, 4, N) - 4-momentum
        mask = pf_mask          # (B, 1, N)
        
        # Add coordinate features
        coords = pf_points.transpose(1, 2)  # (B, N, 2)
        features_with_coords = torch.cat([
            features.transpose(1, 2),  # (B, N, 17)
            coords                     # (B, N, 2)
        ], dim=-1)  # (B, N, 19)
        
        # Sequence trimming (like ParT)
        features_transposed = features_with_coords.transpose(1, 2)  # (B, 19, N)
        features_transposed, v, mask = self.trimmer(features_transposed, v, mask)
        
        # Update mask format for attention
        padding_mask = ~mask.squeeze(1)  # (B, N)
        
        # Embedding (like ParT)
        x = self.embed(features_transposed)  # (seq_len, batch, embed_dim)
        x = x.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        
        # Mask invalid particles
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        
        # Compute pairwise context (like ParT's pair features)
        pair_context = None
        if self.use_pairwise and self.pair_embed is not None:
            pair_context = self.pair_embed(v)  # (B, pair_embed_dim, N, N)
        
        # Apply Mamba layers with pairwise context
        for layer in self.layers:
            x = layer(x, mask=~padding_mask, pair_context=pair_context)
        
        # Global pooling with class token (like ParT)
        x_global = self.pooling(x, mask=~padding_mask)  # (B, embed_dim)
        
        # Classification
        logits = self.classifier(x_global)  # (B, num_classes)
        
        return logits


def get_model(data_config, **kwargs):
    """Model factory function for fair comparison with ParT"""
    
    print("="*70)
    print("Enhanced JetMamba - ParT-style Features")
    print(f"Available input names: {list(data_config.input_names)}")
    print(f"Available input shapes: {data_config.input_shapes}")
    print(f"Label names: {data_config.label_names}")
    print("="*70)
    
    num_classes = len(data_config.label_value)
    
    # Default ParT-like configuration
    cfg = dict(
        num_classes=num_classes,
        d_model=128,
        n_layers=8,           # Same as ParT default
        d_state=16,
        dropout=0.1,
        embed_dims=[128, 512, 128],    # Same as ParT
        pair_embed_dims=[64, 64, 64],  # Same as ParT
        num_cls_layers=2,              # Same as ParT
        use_pairwise=True,             # Enable ParT-like pairwise features
        trim=True,                     # Enable ParT-like trimming
    )
    cfg.update(**kwargs)
    
    # Ensure consistency between d_model and embed_dims
    if 'd_model' in kwargs and 'embed_dims' not in kwargs:
        # If only d_model is specified, adjust embed_dims to match
        d_model = kwargs['d_model']
        cfg['embed_dims'] = [128, 512, d_model]
        print(f"  - Auto-adjusted embed_dims to match d_model={d_model}: {cfg['embed_dims']}")
    
    print(f"Model configuration (ParT-compatible):")
    for key, value in cfg.items():
        print(f"  - {key}: {value}")
    
    model = EnhancedJetMamba(**cfg)
    
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: f'n_{k}'} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }
    
    return model, model_info


def get_loss(data_config, **kwargs):
    """Loss function"""
    return torch.nn.CrossEntropyLoss()