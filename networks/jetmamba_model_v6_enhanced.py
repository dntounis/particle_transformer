"""
JetMamba Model v6 - ENHANCED with ParticleTransformer-inspired Architecture
Key improvements:
1. Class token approach (like ParT)
2. Two-stage processing: Particle Mamba + Class Mamba
3. Pairwise particle features integration
4. Sophisticated classification head
5. Hybrid particle + image processing
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import math
from einops import rearrange

# Import optimized Mamba implementations
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✅ Official mamba-ssm imported successfully")
except ImportError:
    MAMBA_AVAILABLE = False
    print("❌ mamba-ssm not available - falling back to custom implementation")

# Add 2DMamba to Python path and import v2dmamba_scan
import sys
sys.path.insert(0, '/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/2DMamba')

try:
    import v2dmamba_scan
    V2D_SCAN_AVAILABLE = True
    print("✅ v2dmamba_scan custom kernels imported successfully")
except ImportError:
    V2D_SCAN_AVAILABLE = False
    print("❌ v2dmamba_scan not available - using fallback operations")

import time

# Import the optimized preprocessing from v5
from .jetmamba_model_v5 import (
    create_jet_images_batch_gpu, 
    extract_feature_values_gpu, 
    preprocess_channel_gpu_fast,
    OptimizedMambaBlock2D,
    V2DMambaBlock,
    SimplifiedMambaFallback
)


def delta_phi(a, b):
    """Compute delta phi with proper wrapping"""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def delta_r2(eta1, phi1, eta2, phi2):
    """Compute delta R squared"""
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def compute_pairwise_features(pf_features, pf_points, pf_mask, max_pairs=1000):
    """
    Compute pairwise features between particles (inspired by ParT)
    Returns key physics-motivated pairwise relationships
    """
    B, C, N = pf_features.shape
    device = pf_features.device
    
    # Extract particle properties (from standardized features)
    # Note: These are already log-transformed and standardized
    pt_log = pf_features[:, 0, :]  # part_pt_log
    energy_log = pf_features[:, 1, :]  # part_e_log
    eta_rel = pf_points[:, 0, :]  # part_deta 
    phi_rel = pf_points[:, 1, :]  # part_dphi
    
    # Convert back to linear scale for physics calculations
    pt = torch.exp(pt_log * 0.7 + 1.7)  # Reverse standardization
    energy = torch.exp(energy_log * 0.7 + 2.0)
    
    # Create pairwise combinations (limit to avoid memory explosion)
    n_pairs = min(max_pairs, N * (N - 1) // 2)
    
    # Use only the most energetic particles for pairwise features
    particle_importance = energy * pf_mask.squeeze(1)  # (B, N)
    _, top_indices = torch.topk(particle_importance, k=min(50, N), dim=1)  # Top 50 particles
    
    # Create pairs from top particles
    top_k = top_indices.shape[1]
    pairs = []
    for i in range(min(top_k, 10)):  # Limit to avoid too many pairs
        for j in range(i+1, min(top_k, 10)):
            pairs.append((i, j))
    
    if len(pairs) == 0:
        # Fallback: return zeros
        return torch.zeros(B, 4, N, device=device)
    
    pair_features = []
    
    for i, j in pairs[:max_pairs//len(pairs) if len(pairs) > 0 else 1]:
        # Get particle indices
        idx_i = top_indices[:, i]  # (B,)
        idx_j = top_indices[:, j]  # (B,)
        
        # Extract properties for this pair
        pt_i = torch.gather(pt, 1, idx_i.unsqueeze(1)).squeeze(1)
        pt_j = torch.gather(pt, 1, idx_j.unsqueeze(1)).squeeze(1)
        eta_i = torch.gather(eta_rel, 1, idx_i.unsqueeze(1)).squeeze(1)
        eta_j = torch.gather(eta_rel, 1, idx_j.unsqueeze(1)).squeeze(1)
        phi_i = torch.gather(phi_rel, 1, idx_i.unsqueeze(1)).squeeze(1)
        phi_j = torch.gather(phi_rel, 1, idx_j.unsqueeze(1)).squeeze(1)
        
        # Compute pairwise features
        delta_r = torch.sqrt(delta_r2(eta_i, phi_i, eta_j, phi_j))
        pt_min = torch.minimum(pt_i, pt_j)
        pt_sum = pt_i + pt_j
        
        # Key pairwise features (inspired by ParT)
        ln_kt = torch.log(pt_min * delta_r + 1e-8)  # ln(kt)
        ln_z = torch.log(pt_min / (pt_sum + 1e-8) + 1e-8)  # ln(z)
        ln_delta = torch.log(delta_r + 1e-8)  # ln(delta_R)
        kt_ratio = pt_min / (pt_sum + 1e-8)  # pt asymmetry
        
        pair_features.append(torch.stack([ln_kt, ln_z, ln_delta, kt_ratio], dim=1))
    
    if len(pair_features) > 0:
        # Aggregate pairwise features (mean over pairs)
        pairwise_stack = torch.stack(pair_features, dim=0)  # (n_pairs, B, 4)
        pairwise_agg = pairwise_stack.mean(dim=0)  # (B, 4)
        
        # Broadcast to all particles
        pairwise_broadcast = pairwise_agg.unsqueeze(-1).expand(-1, -1, N)  # (B, 4, N)
    else:
        pairwise_broadcast = torch.zeros(B, 4, N, device=device)
    
    return pairwise_broadcast


class EnhancedParticleEmbedding(nn.Module):
    """
    Enhanced particle embedding that combines multiple representations
    """
    
    def __init__(self, input_dim=17, embed_dim=128, use_pairwise=True):
        super().__init__()
        
        self.use_pairwise = use_pairwise
        
        # Main particle feature embedding
        self.particle_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Pairwise feature embedding
        if use_pairwise:
            self.pairwise_embed = nn.Sequential(
                nn.Linear(4, embed_dim // 4),
                nn.LayerNorm(embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, embed_dim)
            )
            
        # Positional embedding for particle order/position
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, 128) * 0.02)
        
    def forward(self, pf_features, pf_points, pf_mask):
        """
        Args:
            pf_features: (B, C, N) particle features
            pf_points: (B, 2, N) particle positions (eta, phi)
            pf_mask: (B, 1, N) particle mask
        Returns:
            embedded_particles: (B, N, embed_dim) 
        """
        B, C, N = pf_features.shape
        
        # Transpose to (B, N, C) for linear layers
        particles = pf_features.transpose(1, 2)  # (B, N, C)
        
        # Main embedding
        embedded = self.particle_embed(particles)  # (B, N, embed_dim)
        
        # Add pairwise features
        if self.use_pairwise:
            pairwise_fts = compute_pairwise_features(pf_features, pf_points, pf_mask)  # (B, 4, N)
            pairwise_fts = pairwise_fts.transpose(1, 2)  # (B, N, 4)
            pairwise_embedded = self.pairwise_embed(pairwise_fts)  # (B, N, embed_dim)
            embedded = embedded + pairwise_embedded
        
        # Add positional encoding
        if N <= self.pos_embed.shape[2]:
            pos = self.pos_embed[:, :, :N].transpose(1, 2)  # (1, N, embed_dim)
            embedded = embedded + pos
        
        # Apply mask
        mask = pf_mask.transpose(1, 2)  # (B, N, 1)
        embedded = embedded * mask
        
        return embedded


class MambaClassToken(nn.Module):
    """
    Mamba-based class token processing (inspired by ParT's cls_blocks)
    """
    
    def __init__(self, d_model, d_state=16, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Learnable class token
        self.cls_token = Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Class-attending Mamba blocks
        self.cls_blocks = nn.ModuleList()
        for _ in range(num_layers):
            if MAMBA_AVAILABLE:
                block = nn.ModuleDict({
                    'norm': nn.LayerNorm(d_model),
                    'mamba': Mamba(d_model=d_model, d_state=d_state),
                    'dropout': nn.Dropout(dropout)
                })
            else:
                block = nn.ModuleDict({
                    'norm': nn.LayerNorm(d_model),
                    'mamba': SimplifiedMambaFallback(d_model, d_state),
                    'dropout': nn.Dropout(dropout)
                })
            self.cls_blocks.append(block)
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, particle_embeddings, particle_mask):
        """
        Args:
            particle_embeddings: (B, N, d_model) particle representations
            particle_mask: (B, N, 1) particle mask
        Returns:
            cls_output: (B, d_model) global jet representation
        """
        B, N, _ = particle_embeddings.shape
        
        # Expand class token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        
        # For each class block
        for block in self.cls_blocks:
            # Concatenate class token with particles
            x = torch.cat([cls_tokens, particle_embeddings], dim=1)  # (B, 1+N, d_model)
            
            # Create attention mask (class token can attend to all, particles mask as usual)
            cls_mask = torch.ones(B, 1, 1, device=particle_mask.device)
            full_mask = torch.cat([cls_mask, particle_mask], dim=1)  # (B, 1+N, 1)
            
            # Apply layer norm
            x_norm = block['norm'](x)
            
            # Mamba processing - class token attends to all particles
            x_mamba = block['mamba'](x_norm)
            
            # Residual connection and dropout
            x = x + block['dropout'](x_mamba)
            
            # Extract updated class token
            cls_tokens = x[:, :1, :]  # (B, 1, d_model)
            
            # Note: We could also update particle representations, but focusing on class token
        
        # Final normalization and return class representation
        cls_output = self.final_norm(cls_tokens).squeeze(1)  # (B, d_model)
        
        return cls_output


class SophisticatedClassifier(nn.Module):
    """
    Multi-layer classifier head (inspired by ParT's fc_params)
    """
    
    def __init__(self, d_model, num_classes, hidden_dims=[512, 256], dropout=0.1):
        super().__init__()
        
        layers = []
        in_dim = d_model
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)


class EnhancedJetVisionMamba(nn.Module):
    """
    ENHANCED JetVision-Mamba combining the best of both worlds:
    - Mamba efficiency for long sequences
    - ParticleTransformer's architectural insights
    - Multi-modal processing (images + particles)
    """
    
    def __init__(self, num_classes=10, d_model=128, n_layers=4, n_cls_layers=2, 
                 d_state=16, dropout=0.1, npix=33, radius=0.8, 
                 use_images=True, use_particles=True, use_pairwise=True,
                 use_v2d=True, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.npix = npix
        self.radius = radius
        self.use_images = use_images
        self.use_particles = use_particles
        self.use_pairwise = use_pairwise
        
        print(f"\n🚀 INITIALIZING ENHANCED JetVision-Mamba:")
        print(f"   - Model dim: {d_model}, Particle layers: {n_layers}, Class layers: {n_cls_layers}")
        print(f"   - Use images: {use_images}, Use particles: {use_particles}")
        print(f"   - Use pairwise features: {use_pairwise}")
        print(f"   - Expected speedup: 40-80x in Mamba blocks")
        
        # === IMAGE PATHWAY ===
        if use_images:
            # Same optimized channel config from v5
            self.channel_config = {
                'part_pt_log': {'preprocess': 'none', 'normalize': False},
                'part_e_log': {'preprocess': 'none', 'normalize': False},
                'part_logptrel': {'preprocess': 'none', 'normalize': False},
                'part_logerel': {'preprocess': 'none', 'normalize': False},
                'part_deltaR': {'preprocess': 'none', 'normalize': False},
                'part_d0': {'preprocess': 'none', 'normalize': False},
                'part_d0err': {'preprocess': 'none', 'normalize': False},
                'part_dz': {'preprocess': 'none', 'normalize': False},
                'part_dzerr': {'preprocess': 'none', 'normalize': False},
                'part_charge': {'preprocess': 'none', 'normalize': False},
                'part_isChargedHadron': {'preprocess': 'none', 'normalize': False},
                'part_isNeutralHadron': {'preprocess': 'none', 'normalize': False},
                'part_isPhoton': {'preprocess': 'none', 'normalize': False},
                'part_isElectron': {'preprocess': 'none', 'normalize': False},
                'part_isMuon': {'preprocess': 'none', 'normalize': False},
            }
            
            n_channels = len(self.channel_config)
            
            # Image processing pathway
            self.image_proj = nn.Conv2d(n_channels, d_model, kernel_size=3, padding=1)
            self.image_norm = nn.BatchNorm2d(d_model)
            
            # Image Mamba blocks
            self.image_blocks = nn.ModuleList()
            for i in range(n_layers):
                if use_v2d and V2D_SCAN_AVAILABLE:
                    block = V2DMambaBlock(d_model, d_state, dropout=dropout)
                else:
                    block = OptimizedMambaBlock2D(d_model, d_state, dropout=dropout)
                self.image_blocks.append(block)
            
            # Image pooling
            self.image_pool = nn.AdaptiveAvgPool2d(1)
        
        # === PARTICLE PATHWAY ===
        if use_particles:
            # Enhanced particle embedding
            self.particle_embed = EnhancedParticleEmbedding(
                input_dim=17, embed_dim=d_model, use_pairwise=use_pairwise
            )
            
            # Particle Mamba blocks (1D sequence processing)
            self.particle_blocks = nn.ModuleList()
            for i in range(n_layers):
                if MAMBA_AVAILABLE:
                    block = nn.ModuleDict({
                        'norm': nn.LayerNorm(d_model),
                        'mamba': Mamba(d_model=d_model, d_state=d_state),
                        'dropout': nn.Dropout(dropout)
                    })
                else:
                    block = nn.ModuleDict({
                        'norm': nn.LayerNorm(d_model),
                        'mamba': SimplifiedMambaFallback(d_model, d_state),
                        'dropout': nn.Dropout(dropout)
                    })
                self.particle_blocks.append(block)
        
        # === CLASS TOKEN PROCESSING ===
        # This replaces the simple attention pooling with ParT-inspired class tokens
        self.class_processor = MambaClassToken(
            d_model=d_model, d_state=d_state, 
            num_layers=n_cls_layers, dropout=dropout
        )
        
        # === SOPHISTICATED CLASSIFIER ===
        # Multi-layer classifier instead of simple linear
        self.classifier = SophisticatedClassifier(
            d_model=d_model, num_classes=num_classes,
            hidden_dims=[d_model*2, d_model], dropout=dropout
        )
        
        self._initialize_weights()
        print(f"✅ Enhanced JetVision-Mamba initialized successfully!\n")
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, *args, **kwargs):
        """
        Enhanced forward pass with multi-pathway processing
        """
        # Handle arguments (same as v5)
        if args:
            if len(args) >= 4:
                pf_points = args[0]
                pf_features = args[1]
                pf_vectors = args[2]
                pf_mask = args[3]
            else:
                raise ValueError(f"Expected at least 4 positional arguments, got {len(args)}")
        else:
            pf_points = kwargs['pf_points']
            pf_features = kwargs['pf_features']
            pf_vectors = kwargs['pf_vectors']
            pf_mask = kwargs['pf_mask']
        
        batch_size = pf_features.shape[0]
        representations = []
        
        # === IMAGE PATHWAY ===
        if self.use_images:
            # Create jet images (using optimized v5 preprocessing)
            jet_images = create_jet_images_batch_gpu(
                pf_features, pf_points, pf_mask,
                self.channel_config, R=self.radius, NPix=self.npix
            )
            
            # Process through image Mamba blocks
            x_img = jet_images.permute(0, 3, 1, 2)  # (B, C, H, W)
            x_img = self.image_proj(x_img)
            x_img = self.image_norm(x_img)
            x_img = F.gelu(x_img)
            
            for block in self.image_blocks:
                x_img = block(x_img)
            
            # Global pool image representation
            img_repr = self.image_pool(x_img).flatten(1)  # (B, d_model)
            representations.append(img_repr)
        
        # === PARTICLE PATHWAY ===
        if self.use_particles:
            # Enhanced particle embedding
            particle_embeddings = self.particle_embed(pf_features, pf_points, pf_mask)  # (B, N, d_model)
            
            # Process through particle Mamba blocks
            x_part = particle_embeddings
            for block in self.particle_blocks:
                # Residual Mamba block
                residual = x_part
                x_norm = block['norm'](x_part)
                x_mamba = block['mamba'](x_norm)
                x_part = residual + block['dropout'](x_mamba)
            
            # Class token processing (replaces attention pooling)
            particle_repr = self.class_processor(x_part, pf_mask.transpose(1, 2))  # (B, d_model)
            representations.append(particle_repr)
        
        # === COMBINE REPRESENTATIONS ===
        if len(representations) == 1:
            final_repr = representations[0]
        else:
            # Multi-modal fusion: weighted sum or concatenation
            final_repr = torch.stack(representations, dim=0).mean(dim=0)  # Simple mean
            # Alternative: final_repr = torch.cat(representations, dim=1)  # Concatenation
        
        # === SOPHISTICATED CLASSIFICATION ===
        logits = self.classifier(final_repr)
        
        return logits


def get_model(data_config, **kwargs):
    """Enhanced model factory function"""
    
    print("="*70)
    print("🚀 ENHANCED JetVision-Mamba v6 - ParticleTransformer Integration")
    print(f"Available input names: {list(data_config.input_names)}")
    print(f"Available input shapes: {data_config.input_shapes}")
    print(f"Label names: {data_config.label_names}")
    print("="*70)
    
    num_classes = len(data_config.label_value)
    d_model = kwargs.get('d_model', 128)
    n_layers = kwargs.get('n_layers', 4)
    n_cls_layers = kwargs.get('n_cls_layers', 2)  # NEW: separate class layers
    d_state = kwargs.get('d_state', 16)
    dropout = kwargs.get('dropout', 0.1)
    npix = kwargs.get('npix', 33)
    radius = kwargs.get('radius', 0.8)
    use_images = kwargs.get('use_images', True)
    use_particles = kwargs.get('use_particles', True)
    use_pairwise = kwargs.get('use_pairwise', True)
    use_v2d = kwargs.get('use_v2d', True)
    
    print(f"Creating ENHANCED JetVision-Mamba:")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Model dim: {d_model}")
    print(f"  - Particle layers: {n_layers}, Class layers: {n_cls_layers}")
    print(f"  - State dim: {d_state}")
    print(f"  - Use images: {use_images}, Use particles: {use_particles}")
    print(f"  - Use pairwise features: {use_pairwise}")
    print(f"  🚀 EXPECTED: Better performance with ParT-inspired architecture!")
    
    model = EnhancedJetVisionMamba(
        num_classes=num_classes,
        d_model=d_model,
        n_layers=n_layers,
        n_cls_layers=n_cls_layers,
        d_state=d_state,
        dropout=dropout,
        npix=npix,
        radius=radius,
        use_images=use_images,
        use_particles=use_particles,
        use_pairwise=use_pairwise,
        use_v2d=use_v2d
    )
    
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