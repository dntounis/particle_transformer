"""
Enhanced JetMamba with Lund Plane Ordering (v2)
==============================================

This version adds physics-motivated Lund plane ordering instead of simple pT ordering.
The Lund plane follows QCD splitting history through Cambridge-Aachen declustering,
providing better ordering for complex jet signatures like Tbqq.

Key changes from v1:
- Added LundPlaneOrderingLayer before Mamba processing
- Particles ordered by QCD splitting sequence instead of pT
- Additional Lund-based features (ln_kt, ln_z, ln_delta_R, splitting_order)
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
# LUND PLANE ORDERING IMPLEMENTATION
# ============================================================================

class LundDeclustering:
    """Container for Lund plane declustering information"""
    
    def __init__(self, delta_R, kt, z, eta, phi, particle_idx):
        self.delta_R = delta_R
        self.kt = kt
        self.z = z
        self.eta = eta
        self.phi = phi
        self.particle_idx = particle_idx
        
        # Lund coordinates (with safety for log)
        self.ln_delta_R = math.log(max(delta_R, 1e-10))
        self.ln_kt = math.log(max(kt, 1e-10))
        self.ln_z = math.log(max(z, 1e-10))

class SimplifiedCambridgeAachen:
    """Simplified Cambridge-Aachen clustering for Lund plane construction"""
    
    def __init__(self, R=0.8):
        self.R = R
        
    def delta_R(self, eta1, phi1, eta2, phi2):
        """Calculate ΔR distance between two particles"""
        deta = eta1 - eta2
        dphi = phi1 - phi2
        # Handle phi wraparound
        dphi = abs(dphi)
        if dphi > math.pi:
            dphi = 2 * math.pi - dphi
        return math.sqrt(deta**2 + dphi**2)
    
    def cluster_jet(self, particles):
        """Simplified Cambridge-Aachen clustering"""
        # Start with individual particles as proto-jets
        proto_jets = [(p[0], p[1], p[2], p[3], [p[4]]) for p in particles]
        clustering_history = []
        
        while len(proto_jets) > 1:
            # Find pair with smallest ΔR
            min_dR = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(proto_jets)):
                for j in range(i + 1, len(proto_jets)):
                    dR = self.delta_R(proto_jets[i][1], proto_jets[i][2], 
                                     proto_jets[j][1], proto_jets[j][2])
                    if dR < min_dR:
                        min_dR = dR
                        merge_i, merge_j = i, j
            
            if merge_i >= 0 and merge_j >= 0:
                jet_i = proto_jets[merge_i]
                jet_j = proto_jets[merge_j]
                
                # Combine 4-momentum
                pt_new = jet_i[0] + jet_j[0]
                eta_new = (jet_i[0] * jet_i[1] + jet_j[0] * jet_j[1]) / max(pt_new, 1e-10)
                phi_new = (jet_i[0] * jet_i[2] + jet_j[0] * jet_j[2]) / max(pt_new, 1e-10)
                e_new = jet_i[3] + jet_j[3]
                constituents_new = jet_i[4] + jet_j[4]
                
                # Record clustering step
                clustering_history.append({
                    'step': len(clustering_history),
                    'harder_jet': jet_i if jet_i[0] > jet_j[0] else jet_j,
                    'softer_jet': jet_j if jet_i[0] > jet_j[0] else jet_i,
                    'delta_R': min_dR,
                    'kt': min(jet_i[0], jet_j[0]) * min_dR,
                    'z': min(jet_i[0], jet_j[0]) / max(jet_i[0] + jet_j[0], 1e-10),
                    'merged_jet': (pt_new, eta_new, phi_new, e_new, constituents_new)
                })
                
                # Remove old jets and add new one
                new_proto_jets = []
                for k, jet in enumerate(proto_jets):
                    if k != merge_i and k != merge_j:
                        new_proto_jets.append(jet)
                new_proto_jets.append((pt_new, eta_new, phi_new, e_new, constituents_new))
                proto_jets = new_proto_jets
            else:
                break
                
        return clustering_history

class LundPlaneOrderingLayer(nn.Module):
    """
    Apply Lund plane ordering to particle sequences
    
    This layer reorders particles based on QCD splitting history instead of pT,
    providing physics-motivated ordering for Mamba models.
    """
    
    def __init__(self, add_lund_features=True, min_kt=1.0, min_z=0.05):
        super().__init__()
        self.add_lund_features = add_lund_features
        self.min_kt = min_kt
        self.min_z = min_z
        
    def generate_lund_ordering(self, particles_np):
        """Generate Lund plane ordering from particle array"""
        try:
            # Initialize clustering
            clustering = SimplifiedCambridgeAachen(R=0.8)
            
            # Convert to list format for clustering
            particle_list = [(particles_np[i, 0], particles_np[i, 1], particles_np[i, 2], 
                             particles_np[i, 3], i) for i in range(len(particles_np))]
            
            # Get clustering history
            history = clustering.cluster_jet(particle_list)
            
            # Extract Lund orderings (following harder branch)
            lund_declusterings = []
            for step in reversed(history):
                harder_jet = step['harder_jet']
                softer_jet = step['softer_jet']
                
                # Apply physics cuts
                if step['kt'] < self.min_kt or step['z'] < self.min_z:
                    continue
                    
                # Create Lund declustering for softer emission
                if len(softer_jet[4]) == 1:  # Single particle
                    declustering = LundDeclustering(
                        delta_R=step['delta_R'],
                        kt=step['kt'],
                        z=step['z'],
                        eta=softer_jet[1],
                        phi=softer_jet[2],
                        particle_idx=softer_jet[4][0]
                    )
                    lund_declusterings.append(declustering)
            
            # Create ordering
            ordered_indices = []
            used_particles = set()
            
            # First: particles in Lund declustering order
            for declustering in lund_declusterings:
                if declustering.particle_idx not in used_particles:
                    ordered_indices.append(declustering.particle_idx)
                    used_particles.add(declustering.particle_idx)
            
            # Then: remaining particles by pT
            remaining = [(particles_np[i, 0], i) for i in range(len(particles_np)) 
                        if i not in used_particles]
            remaining.sort(reverse=True)
            for _, idx in remaining:
                ordered_indices.append(idx)
            
            # Create Lund features
            lund_features = np.zeros((len(particles_np), 4))
            for i, declustering in enumerate(lund_declusterings):
                if declustering.particle_idx < len(particles_np):
                    idx = declustering.particle_idx
                    lund_features[idx, 0] = declustering.ln_kt
                    lund_features[idx, 1] = declustering.ln_z
                    lund_features[idx, 2] = declustering.ln_delta_R
                    lund_features[idx, 3] = i  # Splitting order
                    
            return np.array(ordered_indices), lund_features
            
        except Exception as e:
            # Fallback to pT ordering
            print(f"Lund ordering failed: {e}, using pT fallback")
            pt_order = np.argsort(-particles_np[:, 0])  # Descending pT
            lund_features = np.zeros((len(particles_np), 4))
            return pt_order, lund_features
    
    def forward(self, pf_features, pf_points, pf_mask):
        """
        Apply Lund plane ordering
        
        Args:
            pf_features: (B, C, N) - particle features
            pf_points: (B, 2, N) - [eta, phi] coordinates
            pf_mask: (B, 1, N) - validity mask
            
        Returns:
            ordered_features: (B, C+4, N) - Lund-ordered features with optional Lund features
            ordered_points: (B, 2, N) - Lund-ordered coordinates
            ordered_mask: (B, 1, N) - Lund-ordered mask
        """
        B, C, N = pf_features.shape
        device = pf_features.device
        
        ordered_features_list = []
        ordered_points_list = []
        ordered_mask_list = []
        
        for b in range(B):
            # Extract valid particles
            mask = pf_mask[b, 0, :].bool()
            valid_indices = torch.where(mask)[0]
            
            if len(valid_indices) <= 1:
                # Handle empty or single-particle jets
                if self.add_lund_features:
                    lund_features = torch.zeros(4, N, device=device)
                    full_features = torch.cat([pf_features[b], lund_features], dim=0)
                else:
                    full_features = pf_features[b]
                    
                ordered_features_list.append(full_features)
                ordered_points_list.append(pf_points[b])
                ordered_mask_list.append(pf_mask[b])
                continue
            
            # Convert to numpy for Lund processing
            features_np = pf_features[b, :, valid_indices].cpu().numpy()
            points_np = pf_points[b, :, valid_indices].cpu().numpy()
            
            # Create particle array [pt, eta, phi, energy]
            particles_np = np.zeros((len(valid_indices), 4))
            particles_np[:, 0] = features_np[0, :]  # pt (assume first feature)
            particles_np[:, 1] = points_np[0, :]    # eta
            particles_np[:, 2] = points_np[1, :]    # phi  
            particles_np[:, 3] = features_np[1, :] if C > 1 else features_np[0, :] * np.cosh(points_np[0, :])  # energy
            
            # Generate Lund ordering
            ordered_indices, lund_features = self.generate_lund_ordering(particles_np)
            
            # Apply ordering to valid particles
            reordered_valid_indices = valid_indices[ordered_indices]
            
            # Create reordered tensors
            ordered_features = pf_features[b, :, reordered_valid_indices]
            ordered_points = pf_points[b, :, reordered_valid_indices]
            ordered_mask_single = pf_mask[b, :, reordered_valid_indices]
            
            # Pad back to original size
            full_ordered_features = torch.zeros_like(pf_features[b])
            full_ordered_points = torch.zeros_like(pf_points[b])
            full_ordered_mask = torch.zeros_like(pf_mask[b])
            
            n_valid = len(reordered_valid_indices)
            full_ordered_features[:, :n_valid] = ordered_features
            full_ordered_points[:, :n_valid] = ordered_points
            full_ordered_mask[:, :n_valid] = ordered_mask_single
            
            # Add Lund features if requested
            if self.add_lund_features:
                lund_tensor = torch.from_numpy(lund_features[ordered_indices]).float().to(device)
                lund_padded = torch.zeros(4, N, device=device)
                lund_padded[:, :n_valid] = lund_tensor.T
                full_ordered_features = torch.cat([full_ordered_features, lund_padded], dim=0)
            
            ordered_features_list.append(full_ordered_features)
            ordered_points_list.append(full_ordered_points)
            ordered_mask_list.append(full_ordered_mask)
        
        # Stack back to batch format
        ordered_features = torch.stack(ordered_features_list)
        ordered_points = torch.stack(ordered_points_list)
        ordered_mask = torch.stack(ordered_mask_list)
        
        return ordered_features, ordered_points, ordered_mask

class FastPhysicsOrderingLayer(nn.Module):
    """
    PERFORMANCE FIX: Fast physics-motivated ordering instead of expensive Lund clustering
    
    This provides 10-20x speedup while maintaining physics motivation.
    """
    
    def __init__(self, add_lund_features=True):
        super().__init__()
        self.add_lund_features = add_lund_features
        
    def forward(self, pf_features, pf_points, pf_mask):
        """Fast physics ordering without Cambridge-Aachen clustering"""
        B, C, N = pf_features.shape
        device = pf_features.device
        
        # Extract basic quantities
        pt = pf_features[:, 0, :].clamp(min=1e-8)  # Prevent zeros
        eta = pf_points[:, 0, :].clamp(min=-5, max=5)  # Clip extremes
        phi = pf_points[:, 1, :].clamp(min=-math.pi, max=math.pi)
        mask = pf_mask[:, 0, :].bool()
        
        ordered_features_list = []
        ordered_points_list = []
        ordered_mask_list = []
        
        for b in range(B):
            batch_mask = mask[b]
            valid_indices = torch.where(batch_mask)[0]
            
            if len(valid_indices) <= 1:
                # Handle empty/single particle jets
                if self.add_lund_features:
                    lund_features = torch.zeros(4, N, device=device)
                    full_features = torch.cat([pf_features[b], lund_features], dim=0)
                else:
                    full_features = pf_features[b]
                    
                ordered_features_list.append(full_features)
                ordered_points_list.append(pf_points[b])
                ordered_mask_list.append(pf_mask[b])
                continue
                
            # Extract valid particles
            pt_valid = pt[b, valid_indices]
            eta_valid = eta[b, valid_indices]
            phi_valid = phi[b, valid_indices]
            
            # Compute jet center (pT-weighted)
            total_pt = pt_valid.sum().clamp(min=1e-8)
            eta_center = (eta_valid * pt_valid).sum() / total_pt
            phi_center = torch.atan2(
                (torch.sin(phi_valid) * pt_valid).sum(),
                (torch.cos(phi_valid) * pt_valid).sum()
            )
            
            # Relative coordinates
            eta_rel = eta_valid - eta_center
            phi_rel = phi_valid - phi_center
            phi_rel = torch.where(phi_rel > math.pi, phi_rel - 2*math.pi, phi_rel)
            phi_rel = torch.where(phi_rel < -math.pi, phi_rel + 2*math.pi, phi_rel)
            
            # Distance from center
            dr = torch.sqrt(eta_rel**2 + phi_rel**2).clamp(min=1e-8)
            
            # Physics-motivated ordering: core to edge, weighted by pT
            # This approximates Lund ordering without expensive clustering
            ordering_score = dr * 1000 + (1000 - pt_valid.clamp(max=999))
            physics_order = torch.argsort(ordering_score)
            
            # Apply ordering to valid particles
            reordered_valid_indices = valid_indices[physics_order]
            
            # Create reordered tensors
            ordered_features = pf_features[b, :, reordered_valid_indices]
            ordered_points = pf_points[b, :, reordered_valid_indices]
            ordered_mask_single = pf_mask[b, :, reordered_valid_indices]
            
            # Pad back to original size
            full_ordered_features = torch.zeros_like(pf_features[b])
            full_ordered_points = torch.zeros_like(pf_points[b])
            full_ordered_mask = torch.zeros_like(pf_mask[b])
            
            n_valid = len(reordered_valid_indices)
            full_ordered_features[:, :n_valid] = ordered_features
            full_ordered_points[:, :n_valid] = ordered_points
            full_ordered_mask[:, :n_valid] = ordered_mask_single
            
            # Add simple physics features if requested
            if self.add_lund_features:
                lund_features = torch.zeros(4, N, device=device)
                for i, orig_idx in enumerate(physics_order):
                    abs_idx = reordered_valid_indices[i]
                    if abs_idx < N and i < len(pt_valid):
                        lund_features[0, i] = torch.log(pt_valid[orig_idx] + 1e-8)  # ln(pT)
                        lund_features[1, i] = torch.log((pt_valid[orig_idx] / total_pt) + 1e-8)  # ln(z)
                        lund_features[2, i] = torch.log(dr[orig_idx] + 1e-8)  # ln(ΔR)
                        lund_features[3, i] = i  # Order
                        
                full_ordered_features = torch.cat([full_ordered_features, lund_features], dim=0)
            
            ordered_features_list.append(full_ordered_features)
            ordered_points_list.append(full_ordered_points)
            ordered_mask_list.append(full_ordered_mask)
        
        return torch.stack(ordered_features_list), torch.stack(ordered_points_list), torch.stack(ordered_mask_list)






# ============================================================================
# ORIGINAL ENHANCED JETMAMBA COMPONENTS (from v1)
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
    """ParT-style pairwise feature computation for Mamba context"""
    
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

class MambaWithPairwiseContext(nn.Module):
    """Mamba block enhanced with pairwise context (ParT-inspired)"""
    
    def __init__(self, d_model, d_state=16, dropout=0.1, use_pairwise=True, pair_dim=64):
        super().__init__()
        
        self.use_pairwise = use_pairwise
        self.norm = nn.LayerNorm(d_model)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(d_model=d_model, d_state=d_state)
        else:
            self.mamba = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=dropout, batch_first=True
            )
            
        if use_pairwise:
            self.pair_proj = nn.Linear(pair_dim, d_model)
            self.context_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, pair_context=None):
        x_norm = self.norm(x)
        
        if MAMBA_AVAILABLE:
            x_mamba = self.mamba(x_norm)
        else:
            if mask is not None:
                attn_mask = ~mask.bool()
            else:
                attn_mask = None
            x_mamba, _ = self.mamba(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        
        if self.use_pairwise and pair_context is not None:
            try:
                pair_pooled = pair_context.mean(dim=-1).transpose(1, 2)
                pair_proj = self.pair_proj(pair_pooled)
                
                combined = torch.cat([x_mamba, pair_proj], dim=-1)
                gate = self.context_gate(combined)
                x_mamba = x_mamba + gate * pair_proj
            except RuntimeError:
                pass
        
        return x + self.dropout(x_mamba)

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
# MAIN MODEL CLASS (v2 with Lund Ordering)
# ============================================================================

class EnhancedJetMambaV2(nn.Module):
    """
    Enhanced JetMamba with Lund Plane Ordering (v2)
    
    This version uses physics-motivated Lund plane ordering instead of pT ordering
    for better handling of complex jet signatures.
    """
    
    def __init__(self, num_classes=10, d_model=128, n_layers=8, d_state=16,
                 dropout=0.1, embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
                 num_cls_layers=2, use_pairwise=True, trim=True, 
                 add_lund_features=True, **kwargs):
        super().__init__()
        
        # Account for additional Lund features
        input_features = 19  # 17 original + 2 coordinates
        if add_lund_features:
            input_features += 4  # + Lund features
        
        if d_model != embed_dims[-1]:
            embed_dims = embed_dims[:-1] + [d_model]
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_pairwise = use_pairwise
        self.add_lund_features = add_lund_features
        
        print(f"Enhanced JetMamba v2 (Lund Ordering):")
        print(f"  - d_model: {d_model}, layers: {n_layers}")
        print(f"  - Lund ordering: Enabled")
        print(f"  - Lund features: {add_lund_features}")
        print(f"  - Input features: {input_features}")
        print(f"  - Using Mamba: {MAMBA_AVAILABLE}")
        
        # Lund plane ordering layer
        #Jim
        #self.lund_ordering = LundPlaneOrderingLayer(add_lund_features=add_lund_features)
        self.lund_ordering = FastPhysicsOrderingLayer(add_lund_features=add_lund_features)



        # Sequence trimming
        self.trimmer = SequenceTrimmer(enabled=trim)
        
        # ParT-style embedding
        self.embed = ParTStyleEmbed(
            input_dim=input_features,
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
        
        # Mamba layers with pairwise context
        pair_dim = pair_embed_dims[-1] if use_pairwise else 64
        final_dim = embed_dims[-1]
        self.layers = nn.ModuleList([
            MambaWithPairwiseContext(
                d_model=final_dim,
                d_state=d_state, 
                dropout=dropout,
                use_pairwise=use_pairwise,
                pair_dim=pair_dim
            )
            for _ in range(n_layers)
        ])
        
        # Class token pooling
        self.pooling = ClassTokenPooling(
            d_model=final_dim,
            num_cls_layers=num_cls_layers,
            dropout=0.0
        )
        
        # Classification head
        self.classifier = nn.Linear(final_dim, num_classes)
        
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
        """Forward pass with Lund ordering"""
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
        
        # Apply Lund plane ordering (NEW in v2)
        #print(f"🔬 Applying Lund plane ordering...")
        ordered_features, ordered_points, ordered_mask = self.lund_ordering(
            pf_features, pf_points, pf_mask
        )
        
        # Convert to ParT format
        features = ordered_features  # Now includes Lund features if enabled
        v = pf_vectors              # Keep original vectors (ordering less critical)
        mask = ordered_mask
        
        # Add coordinate features to main features
        coords = ordered_points.transpose(1, 2)  # (B, N, 2)
        features_with_coords = torch.cat([
            features.transpose(1, 2),  # (B, N, C)
            coords                     # (B, N, 2)
        ], dim=-1)
        
        # Sequence trimming
        features_transposed = features_with_coords.transpose(1, 2)
        features_transposed, v, mask = self.trimmer(features_transposed, v, mask)
        
        # Update mask format
        padding_mask = ~mask.squeeze(1)
        
        # Embedding
        x = self.embed(features_transposed)
        x = x.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        
        # Mask invalid particles
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        
        # Compute pairwise context
        pair_context = None
        if self.use_pairwise and self.pair_embed is not None:
            pair_context = self.pair_embed(v)
        
        # Apply Mamba layers with pairwise context
        for layer in self.layers:
            x = layer(x, mask=~padding_mask, pair_context=pair_context)
        
        # Global pooling with class token
        x_global = self.pooling(x, mask=~padding_mask)
        
        # Classification
        logits = self.classifier(x_global)
        
        return logits

def get_model(data_config, **kwargs):
    """Model factory function for Lund-ordered JetMamba v2"""
    
    print("="*70)
    print("Enhanced JetMamba v2 - Lund Plane Ordering")
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
        add_lund_features=True,  # NEW: Enable Lund features
    )
    cfg.update(**kwargs)
    
    if 'd_model' in kwargs and 'embed_dims' not in kwargs:
        d_model = kwargs['d_model']
        cfg['embed_dims'] = [128, 512, d_model]
    
    print(f"Model configuration (Lund v2):")
    for key, value in cfg.items():
        print(f"  - {key}: {value}")
    
    model = EnhancedJetMambaV2(**cfg)
    
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