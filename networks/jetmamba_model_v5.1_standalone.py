"""
JetVision-Mamba v5.1 - Conservative Enhancements (STANDALONE)
Self-contained version with all necessary functions included
Expected: 5-12% performance gain with <20% computational overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from einops import rearrange
import time

# Import optimized Mamba implementations
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✅ Official mamba-ssm imported successfully")
except ImportError:
    MAMBA_AVAILABLE = False
    print("❌ mamba-ssm not available - falling back to custom implementation")

# ============================================================================
# CORE FUNCTIONS FROM V5 (COPIED FOR STANDALONE OPERATION)
# ============================================================================

def create_jet_images_batch_gpu(pf_features, pf_points, pf_mask, 
                               channel_config, R=0.8, NPix=33):
    """
    GPU-OPTIMIZED: Fully vectorized jet image creation
    """
    B, _, N = pf_features.shape
    n_channels = len(channel_config)
    device = pf_features.device
    
    # Transpose to (B, N, C) format
    pf_features = pf_features.transpose(1, 2)  # (B, N, 17)
    pf_points = pf_points.transpose(1, 2)      # (B, N, 2)
    pf_mask = pf_mask.transpose(1, 2).squeeze(-1)  # (B, N)
    
    # Extract coordinates - VECTORIZED
    eta_rel = pf_points[:, :, 0]  # (B, N)
    phi_rel = pf_points[:, :, 1]  # (B, N)
    
    # Create bins - GPU tensors only
    bin_width = 2 * R / NPix
    
    # Convert coordinates to bin indices - VECTORIZED ACROSS BATCH
    eta_bins = torch.floor((eta_rel + R) / bin_width).long()
    phi_bins = torch.floor((phi_rel + R) / bin_width).long()
    
    # Clamp to valid range
    eta_bins = torch.clamp(eta_bins, 0, NPix - 1)
    phi_bins = torch.clamp(phi_bins, 0, NPix - 1)
    
    # Flat indices for scatter operations
    flat_indices = eta_bins * NPix + phi_bins  # (B, N)
    
    # Pre-allocate output
    jet_images = torch.zeros(B, NPix * NPix, n_channels, device=device)
    
    # Process each channel with vectorized operations
    for ch_idx, (feature_name, ch_config) in enumerate(channel_config.items()):
        
        # Extract feature values - VECTORIZED
        values = extract_feature_values_gpu(pf_features, feature_name)
        
        # Apply preprocessing - VECTORIZED (NO CPU transfers)
        if not feature_name.startswith('part_is'):
            values = preprocess_channel_gpu_fast(
                values, 
                method=ch_config.get('preprocess', 'none')
            )
        
        # Apply mask
        values = values * pf_mask
        
        # VECTORIZED histogram accumulation using scatter_add
        batch_indices = torch.arange(B, device=device)[:, None].expand(B, N)
        
        # Flatten for scatter operation
        batch_flat = batch_indices.reshape(-1)
        spatial_flat = flat_indices.reshape(-1) 
        values_flat = values.reshape(-1)
        
        # Combined indices for batch + spatial
        combined_indices = batch_flat * (NPix * NPix) + spatial_flat
        
        # Scatter accumulation - MUCH faster than Python loops
        temp_hist = torch.zeros(B * NPix * NPix, device=device)
        temp_hist.scatter_add_(0, combined_indices, values_flat)
        
        # Reshape and normalize
        channel_hist = temp_hist.view(B, NPix * NPix)
        
        if ch_config.get('normalize', True):
            channel_sums = channel_hist.sum(dim=1, keepdim=True)
            channel_sums = torch.clamp(channel_sums, min=1e-8)
            channel_hist = channel_hist / channel_sums
        
        jet_images[:, :, ch_idx] = channel_hist
    
    # Reshape to final format
    return jet_images.view(B, NPix, NPix, n_channels)


def extract_feature_values_gpu(pf_features, feature_name):
    """Extract feature values using EXACT names from JetClass_full.yaml"""
    
    # CORRECTED: Use exact names from YAML pf_features section
    feature_indices = {
        # Energy features (exact YAML names)
        'part_pt_log': 0,           # part_pt_log, standardized
        'part_e_log': 1,            # part_e_log, standardized
        'part_logptrel': 2,         # part_logptrel, standardized  
        'part_logerel': 3,          # part_logerel, standardized
        
        # Geometric features
        'part_deltaR': 4,           # part_deltaR, standardized
        
        # Track parameters  
        'part_charge': 5,           # part_charge (no standardization)
        'part_isChargedHadron': 6,
        'part_isNeutralHadron': 7,
        'part_isPhoton': 8,
        'part_isElectron': 9,
        'part_isMuon': 10,
        'part_d0': 11,              # part_d0 (already tanh-transformed)
        'part_d0err': 12,           # part_d0err (clipped [0,1])
        'part_dz': 13,              # part_dz (already tanh-transformed)
        'part_dzerr': 14,           # part_dzerr (clipped [0,1])
        # Note: part_deta (15), part_dphi (16) available but not used for jet images
    }
    
    if feature_name in feature_indices:
        idx = feature_indices[feature_name]
        return pf_features[:, :, idx]  # Use directly - no transformation!
    else:
        return pf_features[:, :, 0]  # fallback to first feature


def preprocess_channel_gpu_fast(values, method='log', epsilon=1e-8):
    """
    FAST GPU preprocessing - NO CPU transfers
    """
    # Clamp to reasonable ranges (no percentile computation)
    values = torch.clamp(values, min=0, max=1000.0)  # Fixed maximum
    
    # All GPU operations
    if method == 'log':
        return torch.log1p(values)
    elif method == 'log_epsilon':
        return torch.log(values + epsilon)
    elif method == 'sqrt':
        return torch.sqrt(values)
    elif method == 'tanh':
        return torch.tanh(values)
    else:
        return values


class OptimizedMambaBlock2D(nn.Module):
    """
    OPTIMIZED 2D Mamba block using official mamba-ssm implementation
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        
        if MAMBA_AVAILABLE:
            # Use official optimized Mamba implementation
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            print(f"✅ Using optimized official Mamba (d_model={d_model})")
        else:
            # Fallback to simplified implementation if official not available
            self.mamba = SimplifiedMambaFallback(d_model, d_state)
            print(f"⚠️ Using fallback Mamba implementation")
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass with 2D to 1D conversion for Mamba processing
        """
        B, C, H, W = x.shape
        
        # Convert 2D to sequence for Mamba: (B, C, H, W) -> (B, H*W, C)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        
        # Apply layer norm
        x_norm = self.norm(x_seq)
        
        # Apply optimized Mamba
        x_mamba = self.mamba(x_norm)
        
        # Convert back to 2D: (B, H*W, C) -> (B, C, H, W)
        x_mamba = rearrange(x_mamba, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Residual connection and dropout
        return x + self.dropout(x_mamba)


class SimplifiedMambaFallback(nn.Module):
    """
    Simplified fallback if official Mamba not available
    """
    
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Simple transformer-like fallback
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        # Self-attention (transformer fallback)
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        
        # Feed forward
        ffn_out = self.ffn(x)
        x = x + ffn_out
        
        return x


# ============================================================================
# V5.1 ENHANCEMENTS
# ============================================================================

class EnhancedMultiHeadAttentionPooling(nn.Module):
    """
    Improved attention pooling with multiple heads and better capacity
    Low risk, proven effective improvement over single-head attention
    """
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Learnable queries (multiple instead of single)
        self.queries = Parameter(torch.randn(num_heads, 1, self.head_dim) * 0.02)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=False
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature maps from Mamba blocks
        Returns:
            pooled: (B, d_model) global representation
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: (B, C, H, W) -> (H*W, B, C)
        x_flat = rearrange(x, 'b c h w -> (h w) b c')
        
        # Expand queries for batch: (num_heads, 1, head_dim) -> (1, B, d_model)
        queries = rearrange(self.queries, 'h 1 d -> 1 (h d)').expand(1, B, -1)
        
        # Apply multi-head attention - FIX: Use positional arguments for FLOPs counter compatibility
        pooled, attn_weights = self.attention(
            queries,        # query: (1, B, d_model)
            x_flat,         # key: (H*W, B, d_model)  
            x_flat          # value: (H*W, B, d_model)
        )
        
        # Remove sequence dimension and apply output projection
        pooled = pooled.squeeze(0)  # (B, d_model)
        pooled = self.output_proj(pooled)
        
        return pooled


class SophisticatedClassificationHead(nn.Module):
    """
    Multi-layer classification head with proper regularization
    """
    
    def __init__(self, d_model, num_classes, hidden_dims=[256, 128], 
                 dropout=0.1, use_batch_norm=False):
        super().__init__()
        
        layers = []
        in_dim = d_model
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            # Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(nn.GELU())
            
            # Dropout (higher for earlier layers)
            dropout_rate = dropout * (1.5 - 0.5 * i / len(hidden_dims))
            layers.append(nn.Dropout(dropout_rate))
            
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)


class OptimizedInputProjection(nn.Module):
    """
    Enhanced input projection with better feature processing
    """
    
    def __init__(self, in_channels, d_model, kernel_size=3):
        super().__init__()
        
        # Two-stage projection for better feature extraction
        self.proj1 = nn.Conv2d(in_channels, d_model//2, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm1 = nn.BatchNorm2d(d_model//2)
        
        self.proj2 = nn.Conv2d(d_model//2, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm2 = nn.BatchNorm2d(d_model)
        
        # Residual connection if dimensions match
        self.residual = nn.Conv2d(in_channels, d_model, kernel_size=1) if in_channels != d_model else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.proj1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        
        x = self.proj2(x)
        x = self.norm2(x)
        
        # Residual connection
        x = x + residual
        x = F.gelu(x)
        
        return x


class ConservativeJetVisionMamba(nn.Module):
    """
    Conservative enhancement of JetVision-Mamba v5
    Focus on proven improvements with minimal risk
    Expected: 5-12% performance gain with <20% computational overhead
    """
    
    def __init__(self, num_classes=10, d_model=128, n_layers=4, d_state=16, 
                 dropout=0.1, npix=33, radius=0.8, use_v2d=True, 
                 enhanced_pooling=True, sophisticated_classifier=True, 
                 num_attention_heads=8, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.npix = npix
        self.radius = radius
        
        print(f"\n🚀 INITIALIZING CONSERVATIVE JetVision-Mamba v5.1 (STANDALONE):")
        print(f"   - Model dim: {d_model}, Layers: {n_layers}")
        print(f"   - Enhanced pooling: {enhanced_pooling}")
        print(f"   - Attention heads: {num_attention_heads}")
        print(f"   - Sophisticated classifier: {sophisticated_classifier}")
        print(f"   - Expected: 5-12% gain with <20% overhead")
        
        # Same optimized channel config from v5 (proven working)
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
        
        # IMPROVEMENT 1: Enhanced input projection
        self.input_proj = OptimizedInputProjection(n_channels, d_model)
        
        # Mamba blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            block = OptimizedMambaBlock2D(d_model, d_state, dropout=dropout)
            self.blocks.append(block)
        
        # IMPROVEMENT 2: Enhanced attention pooling
        if enhanced_pooling:
            self.attention_pool = EnhancedMultiHeadAttentionPooling(
                d_model, num_heads=num_attention_heads, dropout=dropout
            )
        else:
            # Simple pooling fallback
            self.attention_pool = SimpleAttentionPooling(d_model)
        
        # IMPROVEMENT 3: Sophisticated classification head
        if sophisticated_classifier:
            self.classifier = SophisticatedClassificationHead(
                d_model, num_classes, 
                hidden_dims=[d_model*2, d_model], 
                dropout=dropout
            )
        else:
            # Simple classifier fallback
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )
        
        self._initialize_weights()
        print(f"✅ Conservative JetVision-Mamba v5.1 (STANDALONE) initialized successfully!\n")
    
    def _initialize_weights(self):
        """Conservative weight initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, *args, **kwargs):
        """Forward pass - keep same interface as v5"""
        
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
        
        # Create jet images (using proven v5 preprocessing)
        jet_images = create_jet_images_batch_gpu(
            pf_features, pf_points, pf_mask,
            self.channel_config, R=self.radius, NPix=self.npix
        )
        
        # Enhanced input projection
        x = jet_images.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.input_proj(x)
        
        # Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Enhanced attention pooling 
        x_pooled = self.attention_pool(x)
        
        # Sophisticated classification
        logits = self.classifier(x_pooled)
        
        return logits


class SimpleAttentionPooling(nn.Module):
    """Simple attention pooling fallback"""
    
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.query = Parameter(torch.randn(1, 1, d_model) * 0.1)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.attention(query, x_flat, x_flat)
        return pooled.squeeze(1)


def get_model(data_config, **kwargs):
    """Conservative model factory function"""
    
    print("="*70)
    print("🚀 CONSERVATIVE JetVision-Mamba v5.1 (STANDALONE) - Proven Enhancements Only")
    print(f"Available input names: {list(data_config.input_names)}")
    print(f"Available input shapes: {data_config.input_shapes}")
    print(f"Label names: {data_config.label_names}")
    print("="*70)
    
    num_classes = len(data_config.label_value)
    d_model = kwargs.get('d_model', 128)
    n_layers = kwargs.get('n_layers', 4)
    d_state = kwargs.get('d_state', 16)
    dropout = kwargs.get('dropout', 0.1)
    npix = kwargs.get('npix', 33)
    radius = kwargs.get('radius', 0.8)
    enhanced_pooling = kwargs.get('enhanced_pooling', True)
    sophisticated_classifier = kwargs.get('sophisticated_classifier', True)
    num_attention_heads = kwargs.get('num_attention_heads', 8)
    
    print(f"Creating CONSERVATIVE JetVision-Mamba v5.1 (STANDALONE):")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Model dim: {d_model}, Layers: {n_layers}")
    print(f"  - Enhanced pooling: {enhanced_pooling}")
    print(f"  - Attention heads: {num_attention_heads}")
    print(f"  - Sophisticated classifier: {sophisticated_classifier}")
    print(f"  🎯 TARGET: 5-12% performance gain with minimal risk")
    
    model = ConservativeJetVisionMamba(
        num_classes=num_classes,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        dropout=dropout,
        npix=npix,
        radius=radius,
        enhanced_pooling=enhanced_pooling,
        sophisticated_classifier=sophisticated_classifier,
        num_attention_heads=num_attention_heads
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