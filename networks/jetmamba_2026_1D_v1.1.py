import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class JetMamba2026_1D_v1_1(nn.Module):
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
        **kwargs
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required for jetmamba_2026_1D_v1.1") from _MAMBA_IMPORT_ERROR

        embed_dims = list(embed_dims)
        if embed_dims[-1] != d_model:
            embed_dims[-1] = d_model

        self.num_classes = num_classes
        self.trim = trim
        self.input_dim = 20

        self.embed = ParticleFeatureEmbed(self.input_dim, embed_dims)
        self.pos_embed = ContinuousPositionEmbedding(d_model)
        self.layers = nn.ModuleList(
            [
                ResidualMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
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

        for layer in self.layers:
            x = layer(x)
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
    )
    cfg.update(**kwargs)

    model = JetMamba2026_1D_v1_1(**cfg)
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
