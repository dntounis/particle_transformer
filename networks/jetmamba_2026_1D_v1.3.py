import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


def particle_pt(pf_vectors, eps=1e-8):
    return torch.sqrt((pf_vectors[:, 0, :] ** 2 + pf_vectors[:, 1, :] ** 2).clamp(min=eps))


def delta_r(pf_points, eps=1e-8):
    return torch.sqrt((pf_points[:, 0, :] ** 2 + pf_points[:, 1, :] ** 2).clamp(min=eps))


def delta_phi(phi_a, phi_b):
    return torch.atan2(torch.sin(phi_a - phi_b), torch.cos(phi_a - phi_b))


def sort_and_trim_particles(pf_features, pf_vectors, pf_points, pf_mask, trim=True):
    bsz, feat_dim, max_particles = pf_features.shape
    device = pf_features.device

    packed_features = pf_features.new_zeros(bsz, feat_dim, max_particles)
    packed_vectors = pf_vectors.new_zeros(bsz, pf_vectors.size(1), max_particles)
    packed_points = pf_points.new_zeros(bsz, pf_points.size(1), max_particles)
    packed_mask = torch.zeros(bsz, 1, max_particles, dtype=torch.bool, device=device)

    pt = particle_pt(pf_vectors)
    abs_eta = pf_points[:, 0, :].abs()
    dr = delta_r(pf_points)

    valid_counts = []
    for batch_idx in range(bsz):
        valid = pf_mask[batch_idx, 0].bool()
        indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        valid_counts.append(int(indices.numel()))
        if indices.numel() == 0:
            continue

        indices = indices[torch.argsort(dr[batch_idx, indices], stable=True)]
        indices = indices[torch.argsort(abs_eta[batch_idx, indices], stable=True)]
        indices = indices[torch.argsort(pt[batch_idx, indices], descending=True, stable=True)]

        n_valid = indices.numel()
        packed_features[batch_idx, :, :n_valid] = pf_features[batch_idx, :, indices]
        packed_vectors[batch_idx, :, :n_valid] = pf_vectors[batch_idx, :, indices]
        packed_points[batch_idx, :, :n_valid] = pf_points[batch_idx, :, indices]
        packed_mask[batch_idx, 0, :n_valid] = True

    target_len = max(valid_counts) if valid_counts else max_particles
    target_len = max(target_len, 1)
    if not trim:
        target_len = max_particles

    return (
        packed_features[:, :, :target_len],
        packed_vectors[:, :, :target_len],
        packed_points[:, :, :target_len],
        packed_mask[:, :, :target_len],
    )


def reverse_valid_prefix(x, valid_mask):
    lengths = valid_mask.sum(dim=1)
    batch_size, seq_len, dim = x.shape
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    reverse_positions = (lengths.unsqueeze(1) - 1 - positions).clamp(min=0)
    gather_idx = torch.where(positions < lengths.unsqueeze(1), reverse_positions, positions)
    return torch.gather(x, 1, gather_idx.unsqueeze(-1).expand(-1, -1, dim))


class ParticleFeatureEmbed(nn.Module):
    def __init__(self, input_dim, embed_dims, activation="gelu"):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        modules = []
        current_dim = input_dim
        for dim in embed_dims:
            modules.extend(
                [
                    nn.LayerNorm(current_dim),
                    nn.Linear(current_dim, dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                ]
            )
            current_dim = dim
        self.embed = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_bn(x)
        x = x.permute(2, 0, 1).contiguous()
        return self.embed(x)


class ContinuousPositionEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = max(d_model // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, coords):
        scale = coords.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return self.net(coords / scale)


class AnchorPairwiseRelation(nn.Module):
    def __init__(self, d_model, num_anchors=4, pair_dim=64, dropout=0.1, eps=1e-8):
        super().__init__()
        self.num_anchors = num_anchors
        self.eps = eps

        hidden_dim = max(pair_dim, 32)
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pair_dim),
            nn.GELU(),
        )
        self.rel_proj = nn.Linear(pair_dim, d_model)
        self.gate_proj = nn.Linear(pair_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pf_vectors, pf_points, valid_mask):
        batch_size, seq_len, _ = x.shape
        if seq_len == 0:
            return x

        num_anchors = min(self.num_anchors, seq_len)
        anchor_vectors = pf_vectors[:, :, :num_anchors]
        anchor_points = pf_points[:, :, :num_anchors]
        anchor_mask = valid_mask[:, :num_anchors]

        pt = particle_pt(pf_vectors, eps=self.eps).unsqueeze(-1)
        anchor_pt = particle_pt(anchor_vectors, eps=self.eps).unsqueeze(1)

        delta_eta = pf_points[:, 0, :].unsqueeze(-1) - anchor_points[:, 0, :].unsqueeze(1)
        delta_phi_vals = delta_phi(
            pf_points[:, 1, :].unsqueeze(-1), anchor_points[:, 1, :].unsqueeze(1)
        )
        delta = torch.sqrt((delta_eta.square() + delta_phi_vals.square()).clamp(min=self.eps))

        pt_min = torch.minimum(pt, anchor_pt)
        ln_delta = torch.log(delta.clamp(min=self.eps))
        lnkt = torch.log((pt_min * delta).clamp(min=self.eps))
        lnz = torch.log((pt_min / (pt + anchor_pt).clamp(min=self.eps)).clamp(min=self.eps))

        pair_features = torch.stack(
            [ln_delta, lnkt, lnz, delta_eta, delta_phi_vals],
            dim=-1,
        )
        pair_embedding = self.pair_mlp(pair_features)

        pair_mask = valid_mask.unsqueeze(-1) & anchor_mask.unsqueeze(1)
        pair_embedding = pair_embedding * pair_mask.unsqueeze(-1).to(pair_embedding.dtype)
        denom = pair_mask.sum(dim=2, keepdim=True).clamp(min=1).to(pair_embedding.dtype)
        rel_summary = pair_embedding.sum(dim=2) / denom

        rel = self.rel_proj(rel_summary)
        gate = torch.sigmoid(self.gate_proj(rel_summary))
        rel = self.dropout(gate * rel)
        rel = rel.masked_fill(~valid_mask.unsqueeze(-1), 0)
        return x + rel


class BidirectionalResidualMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        fusion_mode="concat",
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        if fusion_mode == "concat":
            self.fuse = nn.Sequential(
                nn.Linear(2 * d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
            )
        elif fusion_mode == "sum":
            self.fuse = None
        else:
            raise ValueError("fusion_mode must be 'concat' or 'sum'")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask):
        x_norm = self.norm(x)

        x_fwd = self.mamba_fwd(x_norm)

        x_rev = reverse_valid_prefix(x_norm, valid_mask)
        x_bwd = self.mamba_bwd(x_rev)
        x_bwd = reverse_valid_prefix(x_bwd, valid_mask)

        if self.fusion_mode == "sum":
            fused = 0.5 * (x_fwd + x_bwd)
        else:
            fused = self.fuse(torch.cat([x_fwd, x_bwd], dim=-1))

        return x + self.dropout(fused)


class JetMamba2026_1D_v1_3(nn.Module):
    def __init__(
        self,
        num_classes=10,
        d_model=128,
        n_layers=8,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        embed_dims=(128, 512, 128),
        trim=True,
        fusion_mode="concat",
        num_anchors=4,
        pair_dim=64,
        **kwargs
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required for jetmamba_2026_1D_v1.3") from _MAMBA_IMPORT_ERROR

        embed_dims = list(embed_dims)
        if embed_dims[-1] != d_model:
            embed_dims[-1] = d_model

        self.num_classes = num_classes
        self.trim = trim
        self.input_dim = 20

        self.embed = ParticleFeatureEmbed(self.input_dim, embed_dims)
        self.pos_embed = ContinuousPositionEmbedding(d_model)
        self.anchor_relation = AnchorPairwiseRelation(
            d_model=d_model,
            num_anchors=num_anchors,
            pair_dim=pair_dim,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                BidirectionalResidualMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    fusion_mode=fusion_mode,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, *args, **kwargs):
        if args:
            if len(args) < 4:
                raise ValueError("Expected at least 4 positional arguments")
            pf_points, pf_features, pf_vectors, pf_mask = args[:4]
        else:
            pf_points = kwargs["pf_points"]
            pf_features = kwargs["pf_features"]
            pf_vectors = kwargs["pf_vectors"]
            pf_mask = kwargs["pf_mask"]

        pf_features, pf_vectors, pf_points, pf_mask = sort_and_trim_particles(
            pf_features, pf_vectors, pf_points, pf_mask, trim=self.trim
        )

        valid_mask = pf_mask[:, 0].bool()
        dr = delta_r(pf_points).unsqueeze(1)
        features_with_geometry = torch.cat([pf_features, pf_points, dr], dim=1)

        x = self.embed(features_with_geometry).permute(1, 0, 2).contiguous()

        coords = torch.cat([pf_points, dr], dim=1).transpose(1, 2).contiguous()
        x = x + self.pos_embed(coords)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
        x = self.anchor_relation(x, pf_vectors, pf_points, valid_mask)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)

        for layer in self.layers:
            x = layer(x, valid_mask)
            x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)

        pooled = x.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        logits = self.classifier(self.final_norm(pooled))
        return logits


def get_model(data_config, **kwargs):
    cfg = dict(
        num_classes=len(data_config.label_value),
        d_model=128,
        n_layers=8,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        embed_dims=[128, 512, 128],
        trim=True,
        fusion_mode="concat",
        num_anchors=4,
        pair_dim=64,
    )
    cfg.update(**kwargs)

    model = JetMamba2026_1D_v1_3(**cfg)
    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
            **{"softmax": {0: "N"}},
        },
    }
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
