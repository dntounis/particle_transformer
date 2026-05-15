"""
SIMPLE 2D Enhancement - Minimal computational overhead
Instead of complex spatial grids, just add spatial features to existing 1D processing
"""

import torch
import torch.nn as nn

def compute_simple_spatial_features(pf_features, pf_points, pf_mask):
    """
    Compute simple spatial features with minimal overhead
    
    Instead of complex 2D grids, just add spatial statistics
    """
    B, C, N = pf_features.shape
    device = pf_features.device
    
    # Extract coordinates
    eta_rel = pf_points[:, 0, :]  # (B, N)
    phi_rel = pf_points[:, 1, :]  # (B, N)
    mask = pf_mask[:, 0, :].bool()  # (B, N)
    
    # Get valid particles
    pt = pf_features[:, 0, :] * mask.float()  # (B, N) - pT with mask
    
    # Simple spatial statistics (very fast to compute)
    spatial_features = []
    
    # 1. pT-weighted spatial moments
    total_pt = pt.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
    
    # Weighted mean positions (already computed in preprocessing, but recalculate)
    eta_mean = (eta_rel * pt).sum(dim=1, keepdim=True) / total_pt  # (B, 1)
    phi_mean = (phi_rel * pt).sum(dim=1, keepdim=True) / total_pt  # (B, 1)
    
    # 2. Spatial spread (like jet width)
    eta_var = ((eta_rel - eta_mean)**2 * pt).sum(dim=1, keepdim=True) / total_pt  # (B, 1)
    phi_var = ((phi_rel - phi_mean)**2 * pt).sum(dim=1, keepdim=True) / total_pt  # (B, 1)
    
    # 3. Radial statistics
    r_particles = torch.sqrt(eta_rel**2 + phi_rel**2)  # (B, N)
    r_mean = (r_particles * pt).sum(dim=1, keepdim=True) / total_pt  # (B, 1)
    r_var = ((r_particles - r_mean)**2 * pt).sum(dim=1, keepdim=True) / total_pt  # (B, 1)
    
    # 4. Asymmetry measures
    eta_skew = ((eta_rel - eta_mean)**3 * pt).sum(dim=1, keepdim=True) / total_pt  # (B, 1)
    phi_skew = ((phi_rel - phi_mean)**3 * pt).sum(dim=1, keepdim=True) / total_pt  # (B, 1)
    
    # Stack all spatial features
    spatial_stats = torch.cat([
        eta_mean, phi_mean,    # Position
        eta_var, phi_var,      # Spread  
        r_mean, r_var,         # Radial properties
        eta_skew, phi_skew     # Asymmetry
    ], dim=1)  # (B, 8)
    
    return spatial_stats


class SimpleSpatialEnhancement(nn.Module):
    """
    Simple spatial enhancement with minimal overhead
    Just adds spatial statistics to your existing 1D processing
    """
    
    def __init__(self, d_model, spatial_dim=8):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        
        # Simple projection for spatial features
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Gate to control spatial contribution
        self.spatial_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        print(f"✅ Simple Spatial Enhancement: {spatial_dim} spatial features -> {d_model} dim")
        
    def forward(self, particle_repr, pf_features, pf_points, pf_mask):
        """
        Add spatial enhancement to existing particle representation
        
        Args:
            particle_repr: (B, d_model) - output from your 1D processing
            pf_features, pf_points, pf_mask: raw inputs
            
        Returns:
            enhanced_repr: (B, d_model) - spatially enhanced representation
        """
        # Compute simple spatial features (very fast)
        spatial_stats = compute_simple_spatial_features(pf_features, pf_points, pf_mask)  # (B, 8)
        
        # Project to same dimension
        spatial_repr = self.spatial_proj(spatial_stats)  # (B, d_model)
        
        # Gated fusion
        combined = torch.cat([particle_repr, spatial_repr], dim=1)  # (B, 2*d_model)
        gate = self.spatial_gate(combined)  # (B, d_model)
        
        # Enhanced representation
        enhanced_repr = particle_repr + gate * spatial_repr
        
        return enhanced_repr


# Minimal modification to your existing v1 model
class EnhancedJetMambaSimple(nn.Module):
    """
    Your existing v1 model + simple spatial enhancement
    Minimal changes, minimal overhead
    """
    
    def __init__(self, num_classes=10, d_model=128, n_layers=8, d_state=16,
                 dropout=0.1, embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
                 num_cls_layers=2, use_pairwise=True, trim=True, 
                 use_simple_spatial=True, **kwargs):
        super().__init__()
        
        # ... [All your existing v1 initialization] ...
        




        
        # NEW: Simple spatial enhancement (minimal overhead)
        if use_simple_spatial:
            self.spatial_enhancer = SimpleSpatialEnhancement(d_model=d_model)
            print("✅ Added simple spatial enhancement")
        else:
            self.spatial_enhancer = None
        
        # ... [Rest of your v1 initialization] ...
    
    def forward(self, *args, **kwargs):
        """
        Your existing v1 forward pass + simple spatial enhancement
        """
        # ... [Your existing v1 forward pass until you get particle_repr] ...
        
        # Get 1D representation (your existing code)
        particle_repr = self.pooling_1d(x_1d, mask=~padding_mask)  # (B, d_model)
        
        # NEW: Add simple spatial enhancement (minimal overhead)
        if self.spatial_enhancer is not None:
            particle_repr = self.spatial_enhancer(
                particle_repr, pf_features, pf_points, pf_mask
            )
        
        # Classification (your existing code)
        logits = self.classifier(particle_repr)
        
        return logits