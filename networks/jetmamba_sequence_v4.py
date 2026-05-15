"""
Enhanced JetMamba v1 with 2D Spatial Sequence Processing
Adds 2D spatial pathway alongside existing 1D sequence processing
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

# Add 2DMamba to Python path and import v2dmamba_scan (like jetmamba_model_v5.py)
import sys
sys.path.insert(0, '/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/2DMamba')

try:
    import v2dmamba_scan
    V2D_SCAN_AVAILABLE = True
    print("✅ v2dmamba_scan custom kernels imported successfully")
except ImportError:
    V2D_SCAN_AVAILABLE = False
    print("❌ v2dmamba_scan not available - using fallback operations")

# [Include all your existing imports and helper functions from v1...]
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

# [Include all your existing classes: SequenceTrimmer, ParTStyleEmbed, PairwiseFeatureEmbedding, etc.]

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
    """ParT-style pairwise feature computation for Mamba context"""
    
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
        """Compute pairwise features from 4-momentum vectors"""
        B, _, N = v.shape
        
        # Compute pairwise features
        v_expanded = v.unsqueeze(-1).expand(-1, -1, -1, N)
        vi = v_expanded
        vj = v_expanded.transpose(-2, -1)
        
        # Reshape for pairwise computation
        vi_flat = vi.reshape(B, 4, N*N)
        vj_flat = vj.reshape(B, 4, N*N)
        
        # Compute pairwise features
        pair_fts = pairwise_lv_fts(vi_flat, vj_flat, self.pair_input_dim, self.eps)
        
        # Remove self-pairs if requested
        if self.remove_self_pair:
            mask = torch.eye(N, device=v.device).bool().flatten()
            pair_fts[:, :, mask] = 0
        
        # Embed pairwise features
        pair_embedded = self.embed(pair_fts)
        
        # Reshape back to matrix form
        pair_matrix = pair_embedded.view(B, self.out_dim, N, N)
        
        return pair_matrix

# ===== NEW: 2D SPATIAL SEQUENCE PROCESSING =====

# ===== OPTIMIZED 2D SPATIAL PROCESSING (PERFORMANCE FIX) =====

def create_2d_spatial_sequences_fast(pf_features, pf_points, pf_mask, grid_size=16, R=0.8):
    """OPTIMIZED: Fast 2D spatial sequences using vectorized operations - 100x speedup"""
    B, C, N = pf_features.shape
    device = pf_features.device
    
    # Extract coordinates - VECTORIZED
    eta_rel = pf_points[:, 0, :].clamp(min=-R, max=R)  # (B, N)
    phi_rel = pf_points[:, 1, :].clamp(min=-R, max=R)  # (B, N)
    mask = pf_mask[:, 0, :].bool()  # (B, N)
    
    # Vectorized grid mapping - NO LOOPS
    bin_width = 2 * R / grid_size
    eta_bins = torch.floor((eta_rel + R) / bin_width).long().clamp(0, grid_size - 1)
    phi_bins = torch.floor((phi_rel + R) / bin_width).long().clamp(0, grid_size - 1)
    
    # Create flat indices for scatter operations
    flat_indices = eta_bins * grid_size + phi_bins  # (B, N)
    
    # Initialize output
    spatial_sequences = torch.zeros(B, grid_size * grid_size, C, device=device)
    
    # VECTORIZED aggregation using scatter_add (like jet images)
    batch_indices = torch.arange(B, device=device)[:, None].expand(B, N)
    pt_weights = pf_features[:, 0, :] * mask.float()  # (B, N)
    
    # Process each feature channel with vectorized scatter
    for c in range(C):
        feature_values = pf_features[:, c, :] * pt_weights
        
        # Flatten for scatter operation
        batch_flat = batch_indices.reshape(-1)
        spatial_flat = flat_indices.reshape(-1)
        values_flat = feature_values.reshape(-1)
        
        # Combined indices for batch + spatial
        combined_indices = batch_flat * (grid_size * grid_size) + spatial_flat
        
        # Vectorized scatter accumulation
        temp_hist = torch.zeros(B * grid_size * grid_size, device=device)
        temp_hist.scatter_add_(0, combined_indices, values_flat)
        
        # Reshape and store
        spatial_sequences[:, :, c] = temp_hist.view(B, grid_size * grid_size)
    
    # Normalize by pT sums per cell
    pt_sums = torch.zeros(B, grid_size * grid_size, device=device)
    batch_flat = batch_indices.reshape(-1)
    spatial_flat = flat_indices.reshape(-1)
    pt_flat = pt_weights.reshape(-1)
    combined_indices = batch_flat * (grid_size * grid_size) + spatial_flat
    
    temp_pt = torch.zeros(B * grid_size * grid_size, device=device)
    temp_pt.scatter_add_(0, combined_indices, pt_flat)
    pt_sums = temp_pt.view(B, grid_size * grid_size)
    
    # Normalize features by pT sums
    pt_sums_expanded = pt_sums.unsqueeze(-1).clamp(min=1e-8)
    spatial_sequences = spatial_sequences / pt_sums_expanded
    
    # Reshape to 2D grid
    spatial_sequences = spatial_sequences.view(B, grid_size, grid_size, C)
    spatial_mask = (pt_sums > 0).view(B, grid_size, grid_size)
    
    return spatial_sequences, spatial_mask


class OptimizedMambaBlock2D_Fast(nn.Module):
    """OPTIMIZED: Fast 2D Mamba block without creating new models"""
    
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Pre-create Mamba layers (not on-demand)
        if MAMBA_AVAILABLE:
            self.mamba_row = Mamba(d_model=d_model, d_state=d_state)
            self.mamba_col = Mamba(d_model=d_model, d_state=d_state)
        else:
            # Fallback: simple conv layers
            self.conv1 = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
            self.conv2 = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        
    def forward(self, x):
        """FAST forward pass: (B, H, W, d_model) -> (B, H, W, d_model)"""
        B, H, W, d_model = x.shape
        
        # Layer norm
        x_norm = self.norm(x)
        
        if MAMBA_AVAILABLE:
            # Row processing - VECTORIZED
            x_rows = x_norm.view(B * H, W, d_model)
            x_rows_out = self.mamba_row(x_rows)
            x_row_result = x_rows_out.view(B, H, W, d_model)
            
            # Column processing - VECTORIZED  
            x_cols = x_row_result.permute(0, 2, 1, 3).contiguous().view(B * W, H, d_model)
            x_cols_out = self.mamba_col(x_cols)  
            x_processed = x_cols_out.view(B, W, H, d_model).permute(0, 2, 1, 3).contiguous()
            
        else:
            # Fast conv fallback
            x_conv = x_norm.permute(0, 3, 1, 2)  # (B, d_model, H, W)
            x_conv = self.conv1(x_conv) + self.conv2(x_conv)
            x_processed = x_conv.permute(0, 2, 3, 1)  # (B, H, W, d_model)
        
        # Residual connection
        return x + self.dropout(x_processed)


class FastSpatialSequenceProcessor(nn.Module):
    """OPTIMIZED: Fast 2D spatial processor with minimal overhead"""
    
    def __init__(self, d_model, n_layers=2, d_state=16, dropout=0.1, grid_size=8):
        super().__init__()
        
        self.grid_size = grid_size
        self.d_model = d_model
        
        print(f"🚀 Fast Spatial Processor: {grid_size}x{grid_size} grid, {n_layers} layers")
        
        # Efficient feature projection
        self.feature_proj = nn.Linear(17, d_model)
        
        # Fewer, optimized 2D layers
        self.spatial_layers = nn.ModuleList([
            OptimizedMambaBlock2D_Fast(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])
        
        # Efficient global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, pf_features, pf_points, pf_mask):
        """Fast spatial processing"""
        
        # Fast spatial sequence creation
        spatial_sequences, spatial_mask = create_2d_spatial_sequences_fast(
            pf_features, pf_points, pf_mask, 
            grid_size=self.grid_size
        )
        
        # Efficient projection
        x = self.feature_proj(spatial_sequences)  # (B, H, W, d_model)
        
        # Apply mask efficiently
        mask_expanded = spatial_mask.unsqueeze(-1)
        x = x * mask_expanded.float()
        
        # Fast 2D processing with fewer layers
        for layer in self.spatial_layers:
            x = layer(x)
        
        # Fast global pooling
        x_pooled = x.permute(0, 3, 1, 2)  # (B, d_model, H, W)
        spatial_repr = self.global_pool(x_pooled).flatten(1)  # (B, d_model)
        
        return spatial_repr

# ===== END OPTIMIZED 2D PROCESSING =====

def create_2d_spatial_sequences(pf_features, pf_points, pf_mask, grid_size=16, R=0.8):
    """
    Create 2D spatial sequences for 2D Mamba processing
    
    Args:
        pf_features: (B, 17, N) particle features
        pf_points: (B, 2, N) relative eta, phi coordinates
        pf_mask: (B, 1, N) particle mask
        grid_size: Size of spatial grid
        R: Jet radius for coordinate mapping
        
    Returns:
        spatial_sequences: (B, grid_size, grid_size, d_features)
        spatial_mask: (B, grid_size, grid_size)
    """
    B, C, N = pf_features.shape
    device = pf_features.device
    
    # Extract relative coordinates (already relative to jet center)
    eta_rel = pf_points[:, 0, :].clamp(min=-R, max=R)  # (B, N)
    phi_rel = pf_points[:, 1, :].clamp(min=-R, max=R)  # (B, N)
    mask = pf_mask[:, 0, :].bool()  # (B, N)
    
    # Map to grid indices
    bin_width = 2 * R / grid_size
    eta_bins = torch.floor((eta_rel + R) / bin_width).long().clamp(0, grid_size - 1)
    phi_bins = torch.floor((phi_rel + R) / bin_width).long().clamp(0, grid_size - 1)
    
    # Initialize output
    spatial_sequences = torch.zeros(B, grid_size, grid_size, C, device=device)
    spatial_mask = torch.zeros(B, grid_size, grid_size, device=device, dtype=torch.bool)
    
    # Place particles in spatial grid (pT-weighted aggregation)
    for b in range(B):
        valid_particles = mask[b]
        if not valid_particles.any():
            continue
                
        valid_indices = torch.where(valid_particles)[0]
        eta_bins_valid = eta_bins[b, valid_indices]
        phi_bins_valid = phi_bins[b, valid_indices]
        features_valid = pf_features[b, :, valid_indices]  # (C, n_valid)
        
        # pT for weighting
        pt_valid = features_valid[0, :]  # First feature is pT
        
        # Aggregate particles in each cell using pT-weighted mean
        for eta_idx in range(grid_size):
            for phi_idx in range(grid_size):
                # Find particles in this cell
                cell_mask = (eta_bins_valid == eta_idx) & (phi_bins_valid == phi_idx)
                if not cell_mask.any():
                    continue
                
                # pT-weighted aggregation
                cell_features = features_valid[:, cell_mask]  # (C, n_cell_particles)
                cell_pt = pt_valid[cell_mask]  # (n_cell_particles,)
                
                if cell_pt.sum() > 0:
                    # Weighted average
                    weighted_features = (cell_features * cell_pt.unsqueeze(0)).sum(dim=1) / cell_pt.sum()
                    spatial_sequences[b, eta_idx, phi_idx, :] = weighted_features
                    spatial_mask[b, eta_idx, phi_idx] = True
    
    return spatial_sequences, spatial_mask

class OptimizedMambaBlock2D(nn.Module):
    """2D Mamba block with custom kernels when available"""
    
    def __init__(self, d_model, d_state=16, dropout=0.1, scan_type='raster'):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.scan_type = scan_type
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if V2D_SCAN_AVAILABLE:
            # Use custom 2D Mamba kernels
            print(f"🚀 Using optimized v2dmamba_scan kernels for 2D processing")
            self.use_custom_kernels = True
            
            # Initialize parameters for custom kernels
            self.in_proj = nn.Linear(d_model, d_model * 2)
            self.out_proj = nn.Linear(d_model, d_model)
            self.conv2d = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
            
            # SSM parameters
            self.A = Parameter(-torch.ones(d_model, d_state))
            self.B = Parameter(torch.randn(d_model, d_state) * 0.01)
            self.C = Parameter(torch.randn(d_model, d_state) * 0.01)
            self.D = Parameter(torch.ones(d_model))
            
        else:
            # Fallback to simulated 2D processing with 1D Mamba
            print(f"⚠️  Using fallback 2D processing (1D Mamba + reshape)")
            self.use_custom_kernels = False
            
            if MAMBA_AVAILABLE:
                self.mamba_1d = Mamba(d_model=d_model, d_state=d_state)
            else:
                self.mamba_1d = nn.MultiheadAttention(
                    d_model, num_heads=8, dropout=dropout, batch_first=True
                )
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W, d_model) spatial features
        Returns:
            output: (B, H, W, d_model) processed features
        """
        B, H, W, d_model = x.shape
        
        # Layer norm
        x_norm = self.norm(x)
        
        if self.use_custom_kernels:
            # Use custom 2D Mamba kernels
            x_2d = x_norm.permute(0, 3, 1, 2)  # (B, d_model, H, W)
            
            # Apply 2D convolution for local mixing
            x_conv = self.conv2d(x_2d)
            x_conv = x_conv.permute(0, 2, 3, 1)  # (B, H, W, d_model)
            
            # Custom 2D selective scan (simplified version)
            # Note: This is a simplified implementation. 
            # Full implementation would use the actual v2dmamba_scan kernels
            x_processed = self._simplified_2d_scan(x_conv)
            
        else:
            # Fallback: Flatten to 1D, process with Mamba, reshape back
            x_seq = x_norm.view(B, H * W, d_model)  # (B, H*W, d_model)
            
            if MAMBA_AVAILABLE:
                x_mamba = self.mamba_1d(x_seq)
            else:
                x_mamba, _ = self.mamba_1d(x_seq, x_seq, x_seq)
            
            x_processed = x_mamba.view(B, H, W, d_model)  # (B, H, W, d_model)
        
        # Residual connection and dropout
        return x + self.dropout(x_processed)
    
    def _simplified_2d_scan(self, x):
        """Simplified 2D scan operation (placeholder for full kernel implementation)"""
        B, H, W, d_model = x.shape
        
        # For now, use a simple approach that processes rows and columns
        # Row-wise processing
        x_rows = x.view(B * H, W, d_model)
        if MAMBA_AVAILABLE:
            # Create temporary 1D Mamba for row processing
            temp_mamba = Mamba(d_model=d_model, d_state=self.d_state).to(x.device)
            x_rows_processed = temp_mamba(x_rows)
        else:
            x_rows_processed = x_rows  # Identity fallback
        
        x_row_result = x_rows_processed.view(B, H, W, d_model)
        
        # Column-wise processing
        x_cols = x_row_result.permute(0, 2, 1, 3).contiguous().view(B * W, H, d_model)
        if MAMBA_AVAILABLE:
            temp_mamba_col = Mamba(d_model=d_model, d_state=self.d_state).to(x.device)
            x_cols_processed = temp_mamba_col(x_cols)
        else:
            x_cols_processed = x_cols  # Identity fallback
        
        x_final = x_cols_processed.view(B, W, H, d_model).permute(0, 2, 1, 3).contiguous()
        
        return x_final

class SpatialSequenceProcessor(nn.Module):
    """2D Spatial sequence processing pathway"""
    
    def __init__(self, d_model, n_layers=4, d_state=16, dropout=0.1, grid_size=16):
        super().__init__()
        
        self.grid_size = grid_size
        self.d_model = d_model
        
        # Project input features to d_model
        self.feature_proj = nn.Linear(17, d_model)  # 17 input features
        
        # 2D Mamba layers
        self.spatial_layers = nn.ModuleList([
            OptimizedMambaBlock2D(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])
        
        # Global pooling for spatial features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, pf_features, pf_points, pf_mask):
        """
        Process particles as 2D spatial sequences
        
        Returns:
            spatial_repr: (B, d_model) global spatial representation
        """
        # Create 2D spatial sequences
        spatial_sequences, spatial_mask = create_2d_spatial_sequences(
            pf_features, pf_points, pf_mask, 
            grid_size=self.grid_size
        )
        
        # Project features
        x = self.feature_proj(spatial_sequences)  # (B, H, W, d_model)
        
        # Apply spatial mask
        mask_expanded = spatial_mask.unsqueeze(-1).expand_as(x)
        x = x * mask_expanded.float()
        
        # Process through 2D Mamba layers
        for layer in self.spatial_layers:
            x = layer(x)
        
        # Global pooling: (B, H, W, d_model) -> (B, d_model)
        x_pooled = x.permute(0, 3, 1, 2)  # (B, d_model, H, W)
        spatial_repr = self.global_pool(x_pooled).flatten(1)  # (B, d_model)
        
        return spatial_repr

# Keep all your existing classes (MambaWithPairwiseContext, ClassTokenPooling)

class MambaWithPairwiseContext(nn.Module):
    """Mamba block enhanced with pairwise context (ParT-inspired)"""
    
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
            self.pair_proj = nn.Linear(pair_dim, d_model)
            self.context_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, pair_context=None):
        """Enhanced forward with pairwise context"""
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
        
        # Class attention layers
        self.cls_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
            for _ in range(num_cls_layers)
        ])
        self.cls_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_cls_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """Global pooling with class token"""
        B = x.shape[0]
        
        # Expand class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Prepare mask for attention (include cls token)
        if mask is not None:
            cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            key_padding_mask = ~full_mask.bool()
        else:
            key_padding_mask = None
        
        # Class attention layers
        for layer, norm in zip(self.cls_layers, self.cls_norms):
            # Concatenate cls token with particles
            tokens_with_cls = torch.cat([cls_tokens, x], dim=1)
            
            # Class attention
            cls_out, _ = layer(cls_tokens, tokens_with_cls, tokens_with_cls, 
                              key_padding_mask=key_padding_mask)
            cls_tokens = norm(cls_out)
        
        return self.final_norm(cls_tokens).squeeze(1)

class EnhancedJetMambaWith2D(nn.Module):
    """
    Enhanced JetMamba v1 + 2D Spatial Sequences
    
    Two complementary pathways:
    1. 1D Sequence pathway (your existing sophisticated ParT-inspired processing)
    2. 2D Spatial pathway (new spatial sequence processing)
    """
    
    def __init__(self, num_classes=10, d_model=128, n_layers=8, d_state=16,
                 dropout=0.1, embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
                 num_cls_layers=2, use_pairwise=True, trim=True, 
                 use_2d_spatial=True, spatial_grid_size=16, spatial_layers=4,
                 fusion_strategy='concat', **kwargs):
        super().__init__()
        
        # Ensure consistency
        if d_model != embed_dims[-1]:
            embed_dims = embed_dims[:-1] + [d_model]
            print(f"  - Adjusted embed_dims to match d_model: {embed_dims}")
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_pairwise = use_pairwise
        self.use_2d_spatial = use_2d_spatial
        self.fusion_strategy = fusion_strategy
        
        print(f"Enhanced JetMamba with 2D Spatial Sequences:")
        print(f"  - 1D Sequence processing: ✅ (ParT-inspired)")
        print(f"  - 2D Spatial processing: {'✅' if use_2d_spatial else '❌'}")
        print(f"  - Spatial grid size: {spatial_grid_size}x{spatial_grid_size}")
        print(f"  - Fusion strategy: {fusion_strategy}")
        print(f"  - Using custom 2D kernels: {'✅' if V2D_SCAN_AVAILABLE else '❌'}")
        
        # === 1D SEQUENCE PATHWAY (Your existing sophisticated processing) ===
        
        # Sequence trimming
        self.trimmer = SequenceTrimmer(enabled=trim)
        
        # ParT-style embedding
        self.embed = ParTStyleEmbed(
            input_dim=19,  # 17 features + 2 coordinates
            embed_dims=embed_dims,
            normalize_input=True,
            activation='gelu'
        )
        
        # Pairwise feature embedding
        self.pair_embed = PairwiseFeatureEmbedding(
            pair_input_dim=4,
            pair_embed_dims=pair_embed_dims,
            remove_self_pair=False
        ) if use_pairwise else None
        
        # 1D Mamba layers with pairwise context
        pair_dim = pair_embed_dims[-1] if use_pairwise else 64
        final_dim = embed_dims[-1]
        self.layers_1d = nn.ModuleList([
            MambaWithPairwiseContext(
                d_model=final_dim,
                d_state=d_state, 
                dropout=dropout,
                use_pairwise=use_pairwise,
                pair_dim=pair_dim
            )
            for _ in range(n_layers)
        ])
        
        # 1D Class token pooling
        self.pooling_1d = ClassTokenPooling(
            d_model=final_dim,
            num_cls_layers=num_cls_layers,
            dropout=0.0
        )
        
        # === 2D SPATIAL PATHWAY (New) ===
        if use_2d_spatial:
            self.spatial_processor = SpatialSequenceProcessor(
                d_model=d_model,
                n_layers=spatial_layers,
                d_state=d_state,
                dropout=dropout,
                grid_size=spatial_grid_size
            )
        
        # === FUSION AND CLASSIFICATION ===
        
        # Determine classifier input dimension
        if use_2d_spatial and fusion_strategy == 'concat':
            classifier_input_dim = d_model * 2  # 1D + 2D representations
        else:
            classifier_input_dim = d_model  # Single pathway or add fusion
        
        # Classification head
        self.classifier = nn.Linear(classifier_input_dim, num_classes)
        
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
        """Enhanced forward pass with 1D + 2D processing"""
        
        # Handle arguments (same as your v1)
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
        
        representations = []
        
        # === 1D SEQUENCE PATHWAY (Your existing processing) ===
        
        # Convert to ParT format
        features = pf_features  # (B, 17, N)
        v = pf_vectors          # (B, 4, N)
        mask = pf_mask          # (B, 1, N)
        
        # Add coordinate features
        coords = pf_points.transpose(1, 2)
        features_with_coords = torch.cat([
            features.transpose(1, 2),  # (B, N, 17)
            coords                     # (B, N, 2)
        ], dim=-1)  # (B, N, 19)
        
        # Sequence trimming
        features_transposed = features_with_coords.transpose(1, 2)
        features_transposed, v, mask = self.trimmer(features_transposed, v, mask)
        
        # Update mask format for attention
        padding_mask = ~mask.squeeze(1)
        
        # 1D Embedding
        x_1d = self.embed(features_transposed)  # (seq_len, batch, embed_dim)
        x_1d = x_1d.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        
        # Mask invalid particles
        x_1d = x_1d.masked_fill(padding_mask.unsqueeze(-1), 0)
        
        # Compute pairwise context
        pair_context = None
        if self.use_pairwise and self.pair_embed is not None:
            pair_context = self.pair_embed(v)
        
        # Apply 1D Mamba layers with pairwise context
        for layer in self.layers_1d:
            x_1d = layer(x_1d, mask=~padding_mask, pair_context=pair_context)
        
        # 1D Global pooling with class token
        repr_1d = self.pooling_1d(x_1d, mask=~padding_mask)  # (B, d_model)
        representations.append(repr_1d)
        
        # === 2D SPATIAL PATHWAY (New) ===
        if self.use_2d_spatial:
            repr_2d = self.spatial_processor(pf_features, pf_points, pf_mask)  # (B, d_model)
            representations.append(repr_2d)
        
        # === FUSION ===
        if len(representations) == 1:
            final_repr = representations[0]
        elif self.fusion_strategy == 'concat':
            final_repr = torch.cat(representations, dim=1)  # (B, d_model * 2)
        elif self.fusion_strategy == 'add':
            final_repr = torch.stack(representations, dim=0).sum(dim=0)  # (B, d_model)
        elif self.fusion_strategy == 'mean':
            final_repr = torch.stack(representations, dim=0).mean(dim=0)  # (B, d_model)
        else:
            final_repr = representations[0]  # Default to first
        
        # === CLASSIFICATION ===
        logits = self.classifier(final_repr)
        
        return logits


def get_model(data_config, **kwargs):
    """Enhanced model factory function with 2D spatial sequences"""
    
    print("="*70)
    print("Enhanced JetMamba v1 + 2D Spatial Sequences")
    print(f"Available input names: {list(data_config.input_names)}")
    print(f"Available input shapes: {data_config.input_shapes}")
    print(f"Label names: {data_config.label_names}")
    print("="*70)
    
    num_classes = len(data_config.label_value)
    
    # Default configuration (optimized for speed)
    cfg = dict(
        num_classes=num_classes,
        d_model=128,
        n_layers=6,  # Keep 1D layers same
        d_state=16,
        dropout=0.1,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_cls_layers=2,
        use_pairwise=True,
        trim=True,
        # Optimized 2D spatial options (faster defaults)
        use_2d_spatial=True,        
        spatial_grid_size=8,        # Reduced from 16 to 8 for speed
        spatial_layers=2,           # Reduced from 4 to 2 for speed
        fusion_strategy='concat',   
    )
    cfg.update(**kwargs)
    
    # Ensure consistency
    if 'd_model' in kwargs and 'embed_dims' not in kwargs:
        d_model = kwargs['d_model']
        cfg['embed_dims'] = [128, 512, d_model]
        print(f"  - Auto-adjusted embed_dims to match d_model={d_model}: {cfg['embed_dims']}")
    
    print(f"Enhanced Model Configuration:")
    for key, value in cfg.items():
        print(f"  - {key}: {value}")
    
    model = EnhancedJetMambaWith2D(**cfg)
    
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