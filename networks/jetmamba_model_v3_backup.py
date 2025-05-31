"""
JetMamba Model v3 - Full 2D Mamba Implementation
Converts particle features to jet images and applies 2D Mamba for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
from einops import rearrange


def preprocess_channel(values, method='log', clip_percentile=None, epsilon=1e-8):
    """Preprocess channel values to handle wide dynamic range"""
    # Handle zero/negative values
    values = torch.clamp(values, min=0)
    
    # Optional percentile clipping
    if clip_percentile is not None and values.max() > 0:
        # Use torch operations for differentiability
        valid_values = values[values > 0]
        if len(valid_values) > 0:
            clip_value = torch.quantile(valid_values, clip_percentile / 100.0)
            values = torch.clamp(values, max=clip_value)
    
    # Apply transformation
    if method == 'log':
        transformed = torch.log1p(values)
    elif method == 'log_epsilon':
        transformed = torch.log(values + epsilon)
    elif method == 'sqrt':
        transformed = torch.sqrt(values)
    elif method == 'tanh':
        transformed = torch.tanh(values)
    else:  # 'none'
        transformed = values
    
    return transformed


def create_jet_images_batch(pf_features, pf_points, pf_mask, 
                           channel_config, R=0.8, NPix=33):
    """
    Create multi-channel jet images from batch of particle data
    
    Args:
        pf_features: (B, C=17, N=128) - particle features
        pf_points: (B, C=2, N=128) - particle eta, phi
        pf_mask: (B, 1, N=128) - particle mask
        channel_config: dict of channel configurations
        R: jet radius
        NPix: pixels per dimension
    
    Returns:
        jet_images: (B, NPix, NPix, n_channels)
    """
    B, _, N = pf_features.shape
    n_channels = len(channel_config)
    device = pf_features.device
    
    # Transpose to (B, N, C) format
    pf_features = pf_features.transpose(1, 2)  # (B, N=128, C=17)
    pf_points = pf_points.transpose(1, 2)      # (B, N=128, C=2)
    pf_mask = pf_mask.transpose(1, 2).squeeze(-1)  # (B, N=128)
    
    # Extract coordinates - assuming pf_points contains [deta, dphi]
    eta_rel = pf_points[:, :, 0]  # (B, N) - relative eta
    phi_rel = pf_points[:, :, 1]  # (B, N) - relative phi
    
    # Extract pT from pf_features (assume it's the first processed feature)
    # This is a simplification - you'd need to map to actual pT from features
    pt = torch.exp(pf_features[:, :, 0] * 0.7 + 1.7)  # Reverse the log transform from config
    
    # Initialize output
    jet_images = torch.zeros(B, NPix, NPix, n_channels, device=device)
    
    # Create bins
    bins = torch.linspace(-R, R, NPix + 1, device=device)
    
    # Process each sample in the batch
    for b in range(B):
        # Get valid particles for this jet
        valid = pf_mask[b] > 0.5
        if not valid.any():
            continue
            
        eta_b = eta_rel[b][valid]  # (n_valid,)
        phi_b = phi_rel[b][valid]  # (n_valid,)
        features_b = pf_features[b][valid]  # (n_valid, 17)
        pt_b = pt[b][valid]  # (n_valid,)
        
        # Process each channel
        for ch_idx, (feature_name, ch_config) in enumerate(channel_config.items()):
            # Get feature values based on feature name
            if feature_name == 'part_pt':
                values = pt_b
            elif feature_name == 'part_energy':
                values = torch.exp(features_b[:, 1] * 0.7 + 2.0)  # Reverse log transform
            elif feature_name.startswith('part_is'):
                # Binary features - find the right index
                binary_idx = ['part_isChargedHadron', 'part_isNeutralHadron', 
                             'part_isPhoton', 'part_isElectron', 'part_isMuon'].index(feature_name)
                values = features_b[:, 6 + binary_idx]  # Binary features start at index 6
            elif feature_name == 'part_charge':
                values = features_b[:, 5]  # Charge is at index 5
            elif feature_name == 'part_d0val':
                values = features_b[:, 11]  # d0 is at index 11
            else:
                # Default to pT for unknown features
                values = pt_b
            
            # Apply preprocessing
            if not feature_name.startswith('part_is'):
                values = preprocess_channel(
                    values,
                    method=ch_config.get('preprocess', 'none'),
                    clip_percentile=ch_config.get('clip_percentile', None)
                )
            
            # Create 2D histogram using torch operations
            # Simple binning (you could use torch.histogramdd if available)
            eta_indices = torch.bucketize(eta_b, bins[:-1], right=False)
            phi_indices = torch.bucketize(phi_b, bins[:-1], right=False)
            
            # Ensure indices are within bounds
            eta_indices = torch.clamp(eta_indices, 0, NPix - 1)
            phi_indices = torch.clamp(phi_indices, 0, NPix - 1)
            
            # Accumulate values in histogram
            for i in range(len(eta_indices)):
                eta_idx = eta_indices[i]
                phi_idx = phi_indices[i]
                jet_images[b, eta_idx, phi_idx, ch_idx] += values[i]
            
            # Normalize if requested
            if ch_config.get('normalize', True) and jet_images[b, :, :, ch_idx].sum() > 0:
                jet_images[b, :, :, ch_idx] /= jet_images[b, :, :, ch_idx].sum()
    
    return jet_images


class SelectiveSSM2D(nn.Module):
    """2D Selective State Space Model layer - Fixed version"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, scan_type='raster'):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.scan_type = scan_type
        
        d_inner = self.expand * d_model
        self.d_inner = d_inner
        
        # Linear projections
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.out_proj = nn.Linear(d_inner, d_model)
        
        # 1D convolution for temporal mixing
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, 
            kernel_size=d_conv, 
            padding='same',  # Use 'same' padding to ensure length preservation
            groups=d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)  # B, C, dt
        
        # Initialize A parameter (diagonal state matrix)
        # FIXED: Create A for each dimension of d_inner
        A = torch.arange(1, d_state + 1, dtype=torch.float32)[None, :].repeat(d_inner, 1)
        self.A_log = Parameter(torch.log(A))
        
        # Initialize D parameter (skip connection)
        self.D = Parameter(torch.randn(d_inner) * 0.1)
        
    def scan_2d(self, x, scan_type='raster'):
        """Convert 2D feature map to 1D sequence"""
        B, C, H, W = x.shape
        
        if scan_type == 'raster':
            x_seq = rearrange(x, 'b c h w -> b (h w) c')
        elif scan_type == 'zigzag':
            x_seq = []
            for h in range(H):
                if h % 2 == 0:
                    x_seq.append(x[:, :, h, :])
                else:
                    x_seq.append(x[:, :, h, :].flip(-1))
            x_seq = torch.cat([rearrange(xs, 'b c w -> b w c') for xs in x_seq], dim=1)
        else:
            x_seq = rearrange(x, 'b c h w -> b (h w) c')
            
        return x_seq
    
    def unscan_2d(self, x_seq, H, W, scan_type='raster'):
        """Convert 1D sequence back to 2D feature map"""
        if scan_type == 'raster':
            return rearrange(x_seq, 'b (h w) c -> b c h w', h=H, w=W)
        elif scan_type == 'zigzag':
            B, _, C = x_seq.shape
            x_2d = torch.zeros(B, C, H, W, device=x_seq.device, dtype=x_seq.dtype)
            idx = 0
            for h in range(H):
                if h % 2 == 0:
                    x_2d[:, :, h, :] = x_seq[:, idx:idx+W, :].permute(0, 2, 1)
                else:
                    x_2d[:, :, h, :] = x_seq[:, idx:idx+W, :].flip(1).permute(0, 2, 1)
                idx += W
            return x_2d
        else:
            return rearrange(x_seq, 'b (h w) c -> b c h w', h=H, w=W)
    
    def selective_scan(self, x, dt, A, B, C, D):
        """Core selective scan operation - FIXED version"""
        B_batch, L, D_in = x.shape
        
        # Ensure dt is positive and stable
        dt = F.softplus(dt) + 1e-6
        
        # FIXED: Proper dimension handling for A matrix
        # A should be (D_in, d_state), we need to expand correctly
        # dt: (B_batch, L), A: (D_in, d_state)
        # We want: (B_batch, L, D_in, d_state)
        
        dt_expanded = dt.unsqueeze(-1).unsqueeze(-1)  # (B_batch, L, 1, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)      # (1, 1, D_in, d_state)
        
        # Discretization - now dimensions match
        A_discrete = torch.exp(dt_expanded * A_expanded)  # (B_batch, L, D_in, d_state)
        
        # B and C projections
        B_discrete = dt.unsqueeze(-1) * B  # (B_batch, L, d_state)
        
        # Initialize hidden state
        h = torch.zeros(B_batch, D_in, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for i in range(L):
            # State update: h = A * h + B * x
            # A_discrete[i]: (B_batch, D_in, d_state)
            # h: (B_batch, D_in, d_state)
            # B_discrete[i]: (B_batch, d_state)
            # x[i]: (B_batch, D_in)
            
            h = A_discrete[:, i] * h + B_discrete[:, i].unsqueeze(1) * x[:, i].unsqueeze(-1)
            
            # Output: y = C * h + D * x
            # C[:, i]: (B_batch, d_state)
            # h: (B_batch, D_in, d_state)
            # D: (D_in,)
            # x[:, i]: (B_batch, D_in)
            
            y = torch.sum(C[:, i].unsqueeze(1) * h, dim=-1) + D.unsqueeze(0) * x[:, i]
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)
    
    def forward(self, x):
        """Forward pass of 2D SSM"""
        B, C, H, W = x.shape
        
        # Convert to sequence
        x_seq = self.scan_2d(x, self.scan_type)  # (B, L, C)
        
        # Input projection and gating
        xz = self.in_proj(x_seq)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Apply activation to gate
        z = F.silu(z)
        
        # 1D convolution (reshape for conv1d)
        x_conv = rearrange(x_proj, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        
        # Apply activation
        x_conv = F.silu(x_conv)
        
        # SSM parameters
        ssm_params = self.x_proj(x_conv)  # (B, L, d_state*2 + 1)
        B_ssm, C_ssm, dt = torch.split(ssm_params, [self.d_state, self.d_state, 1], dim=-1)
        
        # Prepare A matrix - FIXED
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Apply selective scan
        y = self.selective_scan(x_conv, dt.squeeze(-1), A, B_ssm, C_ssm, self.D)
        
        # Apply gate
        y = y * z
        
        # Output projection
        y = self.out_proj(y)
        
        # Convert back to 2D
        y_2d = self.unscan_2d(y, H, W, self.scan_type)
        
        return y_2d


class MambaBlock2D(nn.Module):
    """2D Mamba block with residual connection and normalization"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM2D(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Forward pass with residual connection"""
        B, C, H, W = x.shape
        
        # Reshape for layer norm
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.norm(x_flat)
        x_norm = rearrange(x_norm, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Apply SSM
        x_ssm = self.ssm(x_norm)
        
        # Dropout and residual
        x_out = x + self.dropout(x_ssm)
        
        return x_out


class AttentionPooling2D(nn.Module):
    """Attention-based pooling for 2D feature maps"""
    
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.query = Parameter(torch.randn(1, 1, d_model) * 0.1)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
    def forward(self, x):
        """Pool 2D features using attention"""
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # Expand query for batch
        query = self.query.expand(B, -1, -1)
        
        # Apply attention pooling
        pooled, _ = self.attention(query, x_flat, x_flat)
        
        return pooled.squeeze(1)  # (B, C)


class JetVisionMamba(nn.Module):
    """Complete JetVision-Mamba model: Particles -> Jet Images -> 2D Mamba -> Classification"""
    
    def __init__(self, num_classes=10, d_model=128, n_layers=4, d_state=16, 
                 dropout=0.1, npix=33, radius=0.8, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.npix = npix
        self.radius = radius
        
        # Channel configuration for jet images
        self.channel_config = {
            'part_pt': {'preprocess': 'log', 'clip_percentile': 99.5, 'normalize': True},
            'part_energy': {'preprocess': 'log', 'clip_percentile': 99.5, 'normalize': True},
            'part_charge': {'preprocess': 'none', 'normalize': False},
            'part_d0val': {'preprocess': 'tanh', 'normalize': True},
            'part_isChargedHadron': {'preprocess': 'none', 'normalize': False},
            'part_isNeutralHadron': {'preprocess': 'none', 'normalize': False},
            'part_isPhoton': {'preprocess': 'none', 'normalize': False},
            'part_isElectron': {'preprocess': 'none', 'normalize': False},
            'part_isMuon': {'preprocess': 'none', 'normalize': False},
        }
        
        n_channels = len(self.channel_config)
        
        # Input projection from multi-channel image to d_model
        self.input_proj = nn.Conv2d(n_channels, d_model, kernel_size=3, padding=1)
        self.input_norm = nn.BatchNorm2d(d_model)
        
        # Stack of 2D Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock2D(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Global pooling
        self.attention_pool = AttentionPooling2D(d_model, num_heads=4)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, *args, **kwargs):
        """Forward pass: Particles -> Jet Images -> 2D Mamba -> Classification"""
        
        # Handle both positional and keyword arguments
        if args:
            # When called with positional arguments (from FLOPS counter)
            if len(args) >= 4:
                pf_points = args[0]    # (B, 2, N)
                pf_features = args[1]  # (B, 17, N)
                pf_vectors = args[2]   # (B, 4, N)
                pf_mask = args[3]      # (B, 1, N)
            else:
                raise ValueError(f"Expected at least 4 positional arguments, got {len(args)}")
        else:
            # When called with keyword arguments (normal case)
            pf_points = kwargs['pf_points']      # (B, 2, N)
            pf_features = kwargs['pf_features']  # (B, 17, N)
            pf_vectors = kwargs['pf_vectors']    # (B, 4, N)
            pf_mask = kwargs['pf_mask']          # (B, 1, N)
        
        # Create jet images from particle data
        jet_images = create_jet_images_batch(
            pf_features, pf_points, pf_mask,
            self.channel_config, R=self.radius, NPix=self.npix
        )  # (B, NPix, NPix, n_channels)
        
        # Convert to PyTorch tensor format: (B, C, H, W)
        x = jet_images.permute(0, 3, 1, 2)  # (B, n_channels, NPix, NPix)
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # Apply 2D Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Global pooling
        x_pooled = self.attention_pool(x)
        
        # Classification
        logits = self.classifier(x_pooled)
        
        return logits


def get_model(data_config, **kwargs):
    """Model factory function for Weaver"""
    
    print("="*50)
    print("DEBUG: JetVision-Mamba v3 FIXED Data Config")
    print(f"Available input names: {list(data_config.input_names)}")
    print(f"Available input shapes: {data_config.input_shapes}")
    print(f"Label names: {data_config.label_names}")
    print("="*50)
    
    # Model parameters
    num_classes = len(data_config.label_value)
    d_model = kwargs.get('d_model', 128)
    n_layers = kwargs.get('n_layers', 4)
    d_state = kwargs.get('d_state', 16)
    dropout = kwargs.get('dropout', 0.1)
    npix = kwargs.get('npix', 33)
    radius = kwargs.get('radius', 0.8)
    
    print(f"Creating JetVision-Mamba FIXED:")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Model dim: {d_model}")
    print(f"  - Layers: {n_layers}")
    print(f"  - State dim: {d_state}")
    print(f"  - Image size: {npix}x{npix}")
    print(f"  - Radius: {radius}")
    
    model = JetVisionMamba(
        num_classes=num_classes,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        dropout=dropout,
        npix=npix,
        radius=radius
    )
    
    # Model info for Weaver
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: f'n_{k}'} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }
    
    return model, model_info


def get_loss(data_config, **kwargs):
    """Loss function for classification"""
    return torch.nn.CrossEntropyLoss()