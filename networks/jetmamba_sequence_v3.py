"""
Enhanced JetMamba with Spatial-Aware Sequential Mamba (v3)
=========================================================

This version replaces the standard 1D Mamba blocks with spatial-aware Mamba blocks
that explicitly track 2D spatial relationships in η-φ space. This addresses the core
limitation that "Mamba's Markovian state evolution assumption fundamentally conflicts
with the non-local, multi-particle correlations essential for complex jet classification."

Key changes from v1:
- Replaced MambaWithPairwiseContext with SpatialAwareMambaBlock  
- Added explicit 2D spatial coordinate tracking in state vectors
- Added spatial positional encoding for better spatial awareness
- Enhanced state evolution to preserve spatial correlations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from einops import rearrange
import numpy as np

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

# ============================================================================
# SPATIAL-AWARE MAMBA IMPLEMENTATION (NEW in v3)
# ============================================================================

class SpatialPositionalEncoding(nn.Module):
    """Add 2D spatial position encoding to particle sequences"""
    
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.d_model = d_model
        
        # Create learnable spatial embeddings
        self.eta_embedding = nn.Embedding(max_len, d_model // 8)
        self.phi_embedding = nn.Embedding(max_len, d_model // 8)
        self.dr_embedding = nn.Embedding(max_len, d_model // 8)
        self.pt_rank_embedding = nn.Embedding(max_len, d_model // 8)
        self.angle_embedding = nn.Embedding(max_len, d_model // 8)
        self.zone_embedding = nn.Embedding(9, d_model // 8)  # 3x3 spatial zones
        
        # Radial and angular encodings
        self.radial_proj = nn.Linear(2, d_model // 8)  # For r, log_r
        self.angular_proj = nn.Linear(2, d_model // 8)  # For sin(phi), cos(phi)
        
    def compute_spatial_bins(self, eta_rel, phi_rel, dr, pt_rank, angle):
        """Convert continuous spatial coordinates to discrete bins"""
        # Quantize to spatial bins (0-199 range)
        max_val = 199
        
        eta_bins = torch.clamp(((eta_rel + 1.0) / 2.0 * max_val).long(), 0, max_val)
        phi_bins = torch.clamp(((phi_rel + math.pi) / (2 * math.pi) * max_val).long(), 0, max_val)
        dr_bins = torch.clamp((dr / 1.0 * max_val).long(), 0, max_val)  # Assume max dr = 1.0
        pt_bins = torch.clamp(pt_rank.long(), 0, max_val)
        angle_bins = torch.clamp(((angle + math.pi) / (2 * math.pi) * max_val).long(), 0, max_val)
        
        # 3x3 spatial zones
        eta_zone = torch.clamp(((eta_rel + 1.0) / 2.0 * 3).long(), 0, 2)
        phi_zone = torch.clamp(((phi_rel + math.pi) / (2 * math.pi) * 3).long(), 0, 2)
        zone_bins = eta_zone * 3 + phi_zone
        
        return eta_bins, phi_bins, dr_bins, pt_bins, angle_bins, zone_bins
        
    def forward(self, x, spatial_coords):
        """
        Add spatial position encodings
        
        Args:
            x: (B, N, d_model) - particle features
            spatial_coords: (B, N, 6) - [eta_rel, phi_rel, dr, pt_rank, angle, log_dr]
        """
        B, N, _ = x.shape
        
        eta_rel = spatial_coords[:, :, 0]
        phi_rel = spatial_coords[:, :, 1]
        dr = spatial_coords[:, :, 2]
        pt_rank = spatial_coords[:, :, 3]
        angle = spatial_coords[:, :, 4]
        log_dr = spatial_coords[:, :, 5]
        
        # Get discrete bins
        eta_bins, phi_bins, dr_bins, pt_bins, angle_bins, zone_bins = self.compute_spatial_bins(
            eta_rel, phi_rel, dr, pt_rank, angle
        )
        
        # Get embeddings
        eta_emb = self.eta_embedding(eta_bins)
        phi_emb = self.phi_embedding(phi_bins)
        dr_emb = self.dr_embedding(dr_bins)
        pt_emb = self.pt_rank_embedding(pt_bins)
        angle_emb = self.angle_embedding(angle_bins)
        zone_emb = self.zone_embedding(zone_bins)
        
        # Continuous spatial features
        radial_features = torch.stack([dr, log_dr], dim=-1)
        angular_features = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        
        radial_emb = self.radial_proj(radial_features)
        angular_emb = self.angular_proj(angular_features)
        
        # Combine all spatial encodings
        spatial_encoding = torch.cat([
            eta_emb, phi_emb, dr_emb, pt_emb, angle_emb, zone_emb, radial_emb, angular_emb
        ], dim=-1)  # (B, N, d_model)
        
        return x + spatial_encoding

class SpatialAwareMambaBlock(nn.Module):
    """
    Mamba block that maintains explicit awareness of 2D spatial relationships
    
    This addresses the core limitation that Mamba's sequential processing destroys
    spatial correlations essential for jet physics.
    """
    
    def __init__(self, d_model, d_state=16, spatial_state_dim=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.spatial_state_dim = spatial_state_dim
        
        # Standard components
        self.norm = nn.LayerNorm(d_model)
        
        if MAMBA_AVAILABLE:
            # Use standard Mamba for sequence processing
            self.mamba = Mamba(d_model=d_model, d_state=d_state)
        else:
            # Fallback to multi-head attention
            self.mamba = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
            
        # Spatial context integration (NEW in v3)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(6, d_model // 4),  # [eta, phi, dr, pt_rank, angle, log_dr]
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
        # Spatial state evolution (NEW in v3)
        self.spatial_state_processor = nn.Sequential(
            nn.Linear(spatial_state_dim + d_model // 4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, spatial_state_dim)
        )
        
        # Spatial-sequential fusion (NEW in v3)
        self.spatial_fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 4 + spatial_state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Spatial attention for non-local correlations (NEW in v3)
        self.spatial_attention = nn.MultiheadAttention(
            d_model // 4, num_heads=4, dropout=dropout, batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, spatial_coords, mask=None, spatial_state=None):
        """
        Args:
            x: (B, N, d_model) - particle features  
            spatial_coords: (B, N, 6) - [eta_rel, phi_rel, dr, pt_rank, angle, log_dr]
            mask: (B, N) - padding mask
            spatial_state: (B, N, spatial_state_dim) - previous spatial state
        """
        B, N, _ = x.shape
        
        # Initialize spatial state if not provided
        if spatial_state is None:
            spatial_state = torch.zeros(B, N, self.spatial_state_dim, device=x.device)
        
        # Standard Mamba processing
        x_norm = self.norm(x)
        
        if MAMBA_AVAILABLE:
            x_mamba = self.mamba(x_norm)
        else:
            if mask is not None:
                attn_mask = ~mask.bool()
            else:
                attn_mask = None
            x_mamba, _ = self.mamba(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        
        # Encode spatial information (NEW in v3)
        spatial_features = self.spatial_encoder(spatial_coords)  # (B, N, d_model//4)
        
        # Spatial attention for non-local correlations (NEW in v3)
        if mask is not None:
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
            
        spatial_context, _ = self.spatial_attention(
            spatial_features, spatial_features, spatial_features,
            key_padding_mask=attn_mask
        )
        
        # Update spatial state (NEW in v3)
        spatial_input = torch.cat([spatial_state, spatial_features], dim=-1)
        new_spatial_state = self.spatial_state_processor(spatial_input)
        
        # Apply mask to spatial state
        if mask is not None:
            new_spatial_state = new_spatial_state * mask.unsqueeze(-1).float()
        
        # Fuse sequential and spatial information (NEW in v3)
        combined_features = torch.cat([x_mamba, spatial_context, new_spatial_state], dim=-1)
        fused_output = self.spatial_fusion(combined_features)
        
        # Residual connection
        output = x + self.dropout(fused_output)
        
        return output, new_spatial_state

class SpatialAwareJetMamba(nn.Module):
    """
    Enhanced JetMamba with spatial-aware sequential processing
    
    This version explicitly tracks 2D spatial relationships in the state vectors
    while maintaining the efficiency benefits of sequential processing.
    """
    
    def __init__(self, num_classes=10, d_model=128, n_layers=6, d_state=16,
                 spatial_state_dim=8, dropout=0.1, use_spatial_encoding=True, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_spatial_encoding = use_spatial_encoding
        
        print(f"Enhanced JetMamba v3 (Spatial-Aware Sequential):")
        print(f"  - d_model: {d_model}, layers: {n_layers}")
        print(f"  - Spatial state dim: {spatial_state_dim}")
        print(f"  - Spatial encoding: {use_spatial_encoding}")
        print(f"  - Using Mamba: {MAMBA_AVAILABLE}")
        
        # Input embedding
        self.feature_embedding = nn.Linear(19, d_model)  # 17 features + 2 coordinates
        
        # Spatial position encoding (NEW in v3)
        if use_spatial_encoding:
            self.spatial_encoder = SpatialPositionalEncoding(d_model)
        
        # Spatial-aware Mamba blocks (NEW in v3)
        self.layers = nn.ModuleList([
            SpatialAwareMambaBlock(
                d_model=d_model,
                d_state=d_state,
                spatial_state_dim=spatial_state_dim,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Global pooling with class token
        self.attention_pool = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def compute_spatial_coordinates(self, pf_points, pf_features, pf_mask):
        """
        Compute comprehensive spatial relationship features (NEW in v3)
        
        Returns 6 spatial coordinates per particle:
        [eta_rel, phi_rel, dr, pt_rank, angle, log_dr]
        """
        B, _, N = pf_features.shape
        
        # Extract coordinates and features
        eta = pf_points[:, 0, :]  # (B, N)
        phi = pf_points[:, 1, :]  # (B, N)
        pt = pf_features[:, 0, :]  # (B, N) - assuming first feature is pT
        mask = pf_mask[:, 0, :].bool()  # (B, N)
        
        # Compute jet center (pT-weighted)
        pt_masked = pt * mask.float()
        total_pt = pt_masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        eta_center = (eta * pt_masked).sum(dim=1, keepdim=True) / total_pt
        phi_center = torch.atan2(
            (torch.sin(phi) * pt_masked).sum(dim=1, keepdim=True),
            (torch.cos(phi) * pt_masked).sum(dim=1, keepdim=True)
        )
        
        # Relative coordinates
        eta_rel = eta - eta_center
        phi_rel = phi - phi_center
        
        # Handle phi wraparound
        phi_rel = torch.where(phi_rel > math.pi, phi_rel - 2*math.pi, phi_rel)
        phi_rel = torch.where(phi_rel < -math.pi, phi_rel + 2*math.pi, phi_rel)
        
        # Distance from center
        dr = torch.sqrt(eta_rel**2 + phi_rel**2)
        log_dr = torch.log(dr.clamp(min=1e-8))
        
        # Angle in jet frame
        angle = torch.atan2(phi_rel, eta_rel)
        
        # pT ranking (0 = highest pT)
        pt_ranks = torch.argsort(torch.argsort(pt, dim=1, descending=True), dim=1).float()
        
        # Stack spatial coordinates
        spatial_coords = torch.stack([eta_rel, phi_rel, dr, pt_ranks, angle, log_dr], dim=-1)  # (B, N, 6)
        
        return spatial_coords
        
    def forward(self, pf_points, pf_features, pf_vectors, pf_mask, **kwargs):
        """Forward pass with spatial-aware processing"""
        B, C, N = pf_features.shape
        
        # Compute spatial coordinates (NEW in v3)
        spatial_coords = self.compute_spatial_coordinates(pf_points, pf_features, pf_mask)
        
        # Combine features and coordinates
        features_with_coords = torch.cat([
            pf_features.transpose(1, 2),  # (B, N, C)
            pf_points.transpose(1, 2)     # (B, N, 2)
        ], dim=-1)  # (B, N, C+2)
        
        # Embed features
        x = self.feature_embedding(features_with_coords)
        
        # Apply spatial positional encoding (NEW in v3)
        if self.use_spatial_encoding:
            x = self.spatial_encoder(x, spatial_coords)
        
        # Prepare mask
        mask = pf_mask.squeeze(1)  # (B, N)
        
        # Apply spatial-aware Mamba blocks (NEW in v3)
        spatial_state = None
        for layer in self.layers:
            x, spatial_state = layer(x, spatial_coords, mask, spatial_state)
        
        # Global pooling with class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_out, _ = self.attention_pool(cls_tokens, x, x, key_padding_mask=~mask.bool())
        
        return self.classifier(cls_out.squeeze(1))

class StableSpatialPositionalEncoding(nn.Module):
    """FIXED: Numerically stable spatial encoding"""
    
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.d_model = d_model
        
        # Simplified embeddings to avoid complexity
        self.spatial_proj = nn.Linear(4, d_model // 2)  # [eta_rel, phi_rel, dr, pt_rank]
        self.learned_encoding = nn.Parameter(torch.randn(max_len, d_model // 2) * 0.01)
        
    def forward(self, x, spatial_coords):
        """Stable spatial encoding"""
        B, N, _ = x.shape
        
        # Take only first 4 stable coordinates
        stable_coords = spatial_coords[:, :, :4]  # [eta_rel, phi_rel, dr, pt_rank]
        
        # Clip to prevent extreme values
        stable_coords = torch.clamp(stable_coords, min=-10, max=10)
        
        # Project spatial coordinates
        spatial_emb = self.spatial_proj(stable_coords)
        
        # Add learned positional encoding
        seq_len = min(N, len(self.learned_encoding))
        pos_emb = self.learned_encoding[:seq_len].unsqueeze(0).expand(B, -1, -1)
        if N > seq_len:
            # Pad if sequence is longer
            padding = torch.zeros(B, N - seq_len, self.learned_encoding.size(-1), device=x.device)
            pos_emb = torch.cat([pos_emb, padding], dim=1)
        
        # Combine
        spatial_encoding = torch.cat([spatial_emb, pos_emb], dim=-1)
        
        return x + spatial_encoding

class StableSpatialAwareMambaBlock(nn.Module):
    """FIXED: Numerically stable spatial-aware Mamba block"""
    
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(d_model=d_model, d_state=d_state)
        else:
            self.mamba = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
            
        # Simplified spatial integration (avoid complex state evolution)
        self.spatial_mixer = nn.Sequential(
            nn.Linear(d_model + 4, d_model),  # d_model + 4 spatial coords
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, spatial_coords, mask=None):
        """Stable forward pass"""
        # Standard Mamba processing
        x_norm = self.norm(x)
        
        if MAMBA_AVAILABLE:
            x_mamba = self.mamba(x_norm)
        else:
            if mask is not None:
                attn_mask = ~mask.bool()
            else:
                attn_mask = None
            x_mamba, _ = self.mamba(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        
        # Simple spatial integration (STABLE)
        stable_coords = spatial_coords[:, :, :4].clamp(min=-10, max=10)
        combined = torch.cat([x_mamba, stable_coords], dim=-1)
        spatial_enhanced = self.spatial_mixer(combined)
        
        return x + self.dropout(spatial_enhanced)

class StableSpatialAwareJetMamba(nn.Module):
    """FIXED: Stable spatial-aware JetMamba"""
    
    def __init__(self, num_classes=10, d_model=128, n_layers=6, d_state=16, dropout=0.1, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        
        print(f"Enhanced JetMamba v3 (STABLE Spatial-Aware):")
        print(f"  - d_model: {d_model}, layers: {n_layers}")
        print(f"  - Using Mamba: {MAMBA_AVAILABLE}")
        
        # Input embedding
        self.feature_embedding = nn.Linear(19, d_model)
        
        # Spatial encoding
        self.spatial_encoder = StableSpatialPositionalEncoding(d_model)
        
        # Stable spatial-aware Mamba blocks
        self.layers = nn.ModuleList([
            StableSpatialAwareMambaBlock(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Conservative initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain for stability
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def compute_stable_spatial_coordinates(self, pf_points, pf_features, pf_mask):
        """FIXED: Numerically stable spatial coordinate computation"""
        B, _, N = pf_features.shape
        
        eta = pf_points[:, 0, :].clamp(min=-5, max=5)  # Clip extreme values
        phi = pf_points[:, 1, :].clamp(min=-math.pi, max=math.pi)
        pt = pf_features[:, 0, :].clamp(min=1e-8, max=1e6)  # Prevent zero/inf
        mask = pf_mask[:, 0, :].bool()
        
        # Stable jet center computation
        pt_masked = pt * mask.float()
        total_pt = pt_masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        eta_center = (eta * pt_masked).sum(dim=1, keepdim=True) / total_pt
        phi_center = torch.atan2(
            (torch.sin(phi) * pt_masked).sum(dim=1, keepdim=True),
            (torch.cos(phi) * pt_masked).sum(dim=1, keepdim=True)
        )
        
        # Relative coordinates with clipping
        eta_rel = (eta - eta_center).clamp(min=-2, max=2)
        phi_rel = phi - phi_center
        phi_rel = torch.where(phi_rel > math.pi, phi_rel - 2*math.pi, phi_rel)
        phi_rel = torch.where(phi_rel < -math.pi, phi_rel + 2*math.pi, phi_rel)
        phi_rel = phi_rel.clamp(min=-math.pi, max=math.pi)
        
        # Stable distance computation
        dr = torch.sqrt(eta_rel**2 + phi_rel**2).clamp(min=1e-8, max=10)
        
        # pT ranking (stable)
        pt_ranks = torch.argsort(torch.argsort(pt, dim=1, descending=True), dim=1).float()
        pt_ranks = pt_ranks / (N - 1)  # Normalize to [0, 1]
        
        # Return stable coordinates [eta_rel, phi_rel, dr, pt_ranks]
        spatial_coords = torch.stack([eta_rel, phi_rel, dr, pt_ranks], dim=-1)
        
        return spatial_coords
        
    def forward(self, pf_points, pf_features, pf_vectors, pf_mask, **kwargs):
        """Stable forward pass"""
        B, C, N = pf_features.shape
        
        # Stable spatial coordinates
        spatial_coords = self.compute_stable_spatial_coordinates(pf_points, pf_features, pf_mask)
        
        # Combine features and coordinates
        features_with_coords = torch.cat([
            pf_features.transpose(1, 2),
            pf_points.transpose(1, 2)
        ], dim=-1)
        
        # Embed features
        x = self.feature_embedding(features_with_coords)
        
        # Apply spatial encoding
        x = self.spatial_encoder(x, spatial_coords)
        
        # Mask
        mask = pf_mask.squeeze(1)
        
        # Apply stable spatial-aware Mamba blocks
        for layer in self.layers:
            x = layer(x, spatial_coords, mask)
        
        # Global pooling
        x = x.transpose(1, 2)  # (B, C, N)
        x = self.global_pool(x).squeeze(-1)  # (B, C)
        
        return self.classifier(x)








# ============================================================================
# ORIGINAL ENHANCED JETMAMBA COMPONENTS (from v1) - kept for compatibility
# ============================================================================

class SequenceTrimmer(nn.Module):
    """ParT-style sequence trimming for efficiency"""
    
    def __init__(self, enabled=False, target=(0.9, 1.02)):
        super().__init__()
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None):
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled and self.training:
            if self._counter < 5:
                self._counter += 1
            else:
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
        if self.input_bn is not None:
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        return self.embed(x)

class PairwiseFeatureEmbedding(nn.Module):
    """ParT-style pairwise feature computation"""
    
    def __init__(self, pair_input_dim=4, pair_embed_dims=[64, 64, 64], 
                 remove_self_pair=False, eps=1e-8):
        super().__init__()
        self.pair_input_dim = pair_input_dim
        self.remove_self_pair = remove_self_pair
        self.eps = eps
        self.out_dim = pair_embed_dims[-1]
        
        input_dim = pair_input_dim
        module_list = [nn.BatchNorm1d(input_dim)]
        for dim in pair_embed_dims:
            module_list.extend([
                nn.Conv1d(input_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.GELU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list[:-1])
        
    def forward(self, v):
        B, _, N = v.shape
        
        v_expanded = v.unsqueeze(-1).expand(-1, -1, -1, N)
        vi = v_expanded
        vj = v_expanded.transpose(-2, -1)
        
        vi_flat = vi.reshape(B, 4, N*N)
        vj_flat = vj.reshape(B, 4, N*N)
        
        pair_fts = pairwise_lv_fts(vi_flat, vj_flat, self.pair_input_dim, self.eps)
        
        if self.remove_self_pair:
            mask = torch.eye(N, device=v.device).bool().flatten()
            pair_fts[:, :, mask] = 0
        
        pair_embedded = self.embed(pair_fts)
        pair_matrix = pair_embedded.view(B, self.out_dim, N, N)
        
        return pair_matrix

class ClassTokenPooling(nn.Module):
    """ParT-style class token pooling"""
    
    def __init__(self, d_model, num_cls_layers=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.cls_token = Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.cls_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
            for _ in range(num_cls_layers)
        ])
        self.cls_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_cls_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        B = x.shape[0]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if mask is not None:
            cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            key_padding_mask = ~full_mask.bool()
        else:
            key_padding_mask = None
        
        for layer, norm in zip(self.cls_layers, self.cls_norms):
            tokens_with_cls = torch.cat([cls_tokens, x], dim=1)
            cls_out, _ = layer(cls_tokens, tokens_with_cls, tokens_with_cls, 
                              key_padding_mask=key_padding_mask)
            cls_tokens = norm(cls_out)
        
        return self.final_norm(cls_tokens).squeeze(1)

# ============================================================================
# MAIN MODEL CLASS (v3 with Spatial-Aware Mamba)
# ============================================================================

class EnhancedJetMambaV3(nn.Module):
    """
    Enhanced JetMamba with Spatial-Aware Sequential Mamba (v3)
    
    This version replaces standard Mamba blocks with spatial-aware blocks that
    explicitly track 2D spatial relationships, addressing the core limitation
    that sequential processing destroys spatial correlations essential for jets.
    """
    
    def __init__(self, num_classes=10, d_model=128, n_layers=8, d_state=16,
                 dropout=0.1, embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
                 num_cls_layers=2, use_pairwise=True, trim=True, 
                 spatial_state_dim=8, use_spatial_encoding=True, **kwargs):
        super().__init__()
        
        if d_model != embed_dims[-1]:
            embed_dims = embed_dims[:-1] + [d_model]
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_pairwise = use_pairwise
        self.use_spatial_encoding = use_spatial_encoding
        
        print(f"Enhanced JetMamba v3 (Spatial-Aware Sequential):")
        print(f"  - d_model: {d_model}, layers: {n_layers}")
        print(f"  - Spatial state dim: {spatial_state_dim}")
        print(f"  - Spatial encoding: {use_spatial_encoding}")
        print(f"  - Using Mamba: {MAMBA_AVAILABLE}")
        
        # Sequence trimming
        self.trimmer = SequenceTrimmer(enabled=trim)
        
        # ParT-style embedding
        self.embed = ParTStyleEmbed(
            input_dim=19,  # 17 features + 2 coordinates
            embed_dims=embed_dims,
            normalize_input=True,
            activation='gelu'
        )
        
        # Pairwise feature embedding (optional)
        self.pair_embed = PairwiseFeatureEmbedding(
            pair_input_dim=4,
            pair_embed_dims=pair_embed_dims,
            remove_self_pair=False
        ) if use_pairwise else None
        
        # Spatial-aware processing core (NEW in v3)
        # self.spatial_processor = SpatialAwareJetMamba(
        #     num_classes=num_classes,
        #     d_model=embed_dims[-1],
        #     n_layers=n_layers,
        #     d_state=d_state,
        #     spatial_state_dim=spatial_state_dim,
        #     dropout=dropout,
        #     use_spatial_encoding=use_spatial_encoding
        # )

        #Jim
        self.spatial_processor = StableSpatialAwareJetMamba(
            num_classes=num_classes,
            d_model=embed_dims[-1],
            n_layers=n_layers,
            d_state=d_state,
            spatial_state_dim=spatial_state_dim,
            use_spatial_encoding=use_spatial_encoding,
            dropout=dropout
        )



        
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
        """Forward pass with spatial-aware processing"""
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
        
        # Use spatial-aware processor directly (NEW in v3)
        #print(f"🧭 Using spatial-aware sequential Mamba...")
        return self.spatial_processor(pf_points, pf_features, pf_vectors, pf_mask)

def get_model(data_config, **kwargs):
    """Model factory function for Spatial-Aware JetMamba v3"""
    
    print("="*70)
    print("Enhanced JetMamba v3 - Spatial-Aware Sequential Mamba")
    print(f"Available input names: {list(data_config.input_names)}")
    print(f"Available input shapes: {data_config.input_shapes}")
    print(f"Label names: {data_config.label_names}")
    print("="*70)
    
    num_classes = len(data_config.label_value)
    
    cfg = dict(
        num_classes=num_classes,
        d_model=128,
        n_layers=8,
        d_state=16,
        dropout=0.1,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_cls_layers=2,
        use_pairwise=True,
        trim=True,
        spatial_state_dim=8,        # NEW: Spatial state dimension
        use_spatial_encoding=True,  # NEW: Enable spatial encoding
    )
    cfg.update(**kwargs)
    
    if 'd_model' in kwargs and 'embed_dims' not in kwargs:
        d_model = kwargs['d_model']
        cfg['embed_dims'] = [128, 512, d_model]
    
    print(f"Model configuration (Spatial-Aware v3):")
    for key, value in cfg.items():
        print(f"  - {key}: {value}")
    
    model = EnhancedJetMambaV3(**cfg)
    
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