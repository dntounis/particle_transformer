import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


TWO_DMAMBA_ROOT = "/global/cfs/cdirs/m2616/dntounis/Mamba_SSMs/git_repos/2DMamba"
if TWO_DMAMBA_ROOT not in sys.path:
    sys.path.insert(0, TWO_DMAMBA_ROOT)

try:
    import v2dmamba_scan
except ImportError as exc:
    v2dmamba_scan = None
    _V2D_IMPORT_ERROR = exc
else:
    _V2D_IMPORT_ERROR = None


def particle_pt(pf_vectors, eps=1e-8):
    return torch.sqrt((pf_vectors[:, 0, :] ** 2 + pf_vectors[:, 1, :] ** 2).clamp(min=eps))


def delta_r(pf_points, eps=1e-8):
    return torch.sqrt((pf_points[:, 0, :] ** 2 + pf_points[:, 1, :] ** 2).clamp(min=eps))


class V2DSelectiveScanFn(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
        HH=None,
        WW=None,
    ):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(1)
            ctx.squeeze_C = True

        if HH is None or WW is None or HH * WW != u.shape[-1]:
            raise ValueError("HH and WW must satisfy HH * WW == sequence length")

        out, x, *rest = v2dmamba_scan.fwd(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus, HH, WW
        )
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        ctx.HH = HH
        ctx.WW = WW

        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, x[:, :, -1, 1::2])

        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
        out_z = rest[0]
        return out_z if not return_last_state else (out_z, x[:, :, -1, 1::2])

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = v2dmamba_scan.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            dout,
            x,
            out,
            None,
            ctx.delta_softplus,
            False,
            ctx.HH,
            ctx.WW,
        )

        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC

        return (
            du,
            ddelta,
            dA,
            dB,
            dC,
            dD if D is not None else None,
            dz,
            ddelta_bias if delta_bias is not None else None,
            None,
            None,
            None,
            None,
        )


def v2d_selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    HH=None,
    WW=None,
):
    return V2DSelectiveScanFn.apply(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, HH, WW
    )


def build_particle_lattice(
    pf_features,
    pf_vectors,
    pf_points,
    pf_mask,
    grid_h=8,
    grid_w=16,
    eta_span=0.8,
):
    bsz, feat_dim, max_particles = pf_features.shape
    device = pf_features.device
    token_dim = feat_dim + 3

    lattice = pf_features.new_zeros(bsz, grid_h, grid_w, token_dim)
    lattice_mask = torch.zeros(bsz, grid_h, grid_w, dtype=torch.bool, device=device)
    cell_coords = pf_features.new_zeros(bsz, grid_h, grid_w, 2)

    row_coords = torch.linspace(-1.0, 1.0, grid_h, device=device)
    col_coords = torch.linspace(-1.0, 1.0, grid_w, device=device)
    cell_coords[..., 0] = row_coords.view(1, grid_h, 1)
    cell_coords[..., 1] = col_coords.view(1, 1, grid_w)

    pt = particle_pt(pf_vectors)
    dr = delta_r(pf_points)

    for batch_idx in range(bsz):
        valid = torch.nonzero(pf_mask[batch_idx, 0].bool(), as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue

        order = valid[torch.argsort(pt[batch_idx, valid], descending=True, stable=True)]
        row_fill = torch.zeros(grid_h, dtype=torch.long, device=device)
        overflow = []

        for particle_idx in order.tolist():
            eta_value = float(torch.clamp(pf_points[batch_idx, 0, particle_idx], -eta_span, eta_span))
            row = int(math.floor((eta_value + eta_span) / (2 * eta_span) * grid_h))
            row = min(max(row, 0), grid_h - 1)
            col = int(row_fill[row].item())
            if col < grid_w:
                token = torch.cat(
                    [
                        pf_features[batch_idx, :, particle_idx],
                        pf_points[batch_idx, :, particle_idx],
                        dr[batch_idx, particle_idx].unsqueeze(0),
                    ],
                    dim=0,
                )
                lattice[batch_idx, row, col] = token
                lattice_mask[batch_idx, row, col] = True
                row_fill[row] += 1
            else:
                overflow.append(particle_idx)

        if overflow:
            empty_cells = torch.nonzero(~lattice_mask[batch_idx], as_tuple=False)
            for particle_idx, (row, col) in zip(overflow, empty_cells.tolist()):
                token = torch.cat(
                    [
                        pf_features[batch_idx, :, particle_idx],
                        pf_points[batch_idx, :, particle_idx],
                        dr[batch_idx, particle_idx].unsqueeze(0),
                    ],
                    dim=0,
                )
                lattice[batch_idx, row, col] = token
                lattice_mask[batch_idx, row, col] = True

    return lattice, lattice_mask, cell_coords


class LatticeEmbed(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.proj(x)


class CellPositionEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = max(d_model // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, coords):
        return self.net(coords)


class V2DMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.dwconv = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x, mask=None):
        if x.device.type != "cuda":
            raise RuntimeError("jetmamba_2026_2D_v1.1 requires CUDA for v2dmamba_scan")

        bsz, grid_h, grid_w, _ = x.shape
        residual = x
        x_norm = self.norm(x)

        xz = self.in_proj(x_norm)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_conv = self.dwconv(x_branch.permute(0, 3, 1, 2).contiguous())
        x_conv = F.silu(x_conv.permute(0, 2, 3, 1).contiguous())

        delta_bc = self.x_proj(x_conv)
        delta, B, C = torch.split(
            delta_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        u = x_conv.reshape(bsz, grid_h * grid_w, self.d_inner).transpose(1, 2).contiguous()
        z = z_branch.reshape(bsz, grid_h * grid_w, self.d_inner).transpose(1, 2).contiguous()
        delta = F.linear(delta, self.dt_proj.weight, None)
        delta = delta.reshape(bsz, grid_h * grid_w, self.d_inner).transpose(1, 2).contiguous()
        B = B.reshape(bsz, grid_h * grid_w, self.d_state).transpose(1, 2).contiguous()
        C = C.reshape(bsz, grid_h * grid_w, self.d_state).transpose(1, 2).contiguous()

        y = v2d_selective_scan_fn(
            u,
            delta,
            -torch.exp(self.A_log.float()),
            B,
            C,
            D=self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            HH=grid_h,
            WW=grid_w,
        )
        y = y.transpose(1, 2).reshape(bsz, grid_h, grid_w, self.d_inner)
        y = self.out_proj(y)
        if mask is not None:
            y = y * mask.unsqueeze(-1).to(y.dtype)
        return residual + self.dropout(y)


class JetMamba2026_2D_v1_1(nn.Module):
    def __init__(
        self,
        num_classes=10,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=3,
        expand=2,
        dropout=0.1,
        grid_h=8,
        grid_w=16,
        eta_span=0.8,
        **kwargs
    ):
        super().__init__()
        if v2dmamba_scan is None:
            raise ImportError("v2dmamba_scan is required for jetmamba_2026_2D_v1.1") from _V2D_IMPORT_ERROR

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.eta_span = eta_span
        self.token_dim = 20

        self.embed = LatticeEmbed(self.token_dim, d_model)
        self.cell_pos_embed = CellPositionEmbedding(d_model)
        self.pad_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.blocks = nn.ModuleList(
            [
                V2DMambaBlock(
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
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
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

        lattice, lattice_mask, cell_coords = build_particle_lattice(
            pf_features,
            pf_vectors,
            pf_points,
            pf_mask,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            eta_span=self.eta_span,
        )

        x = self.embed(lattice) + self.cell_pos_embed(cell_coords)
        x = torch.where(lattice_mask.unsqueeze(-1), x, self.pad_token.expand_as(x))

        for block in self.blocks:
            x = block(x, mask=lattice_mask)

        valid = lattice_mask.view(lattice_mask.size(0), -1)
        x_flat = x.view(x.size(0), -1, x.size(-1))
        x_flat = x_flat * valid.unsqueeze(-1).to(x.dtype)
        pooled = x_flat.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        logits = self.classifier(self.final_norm(pooled))
        return logits


def get_model(data_config, **kwargs):
    cfg = dict(
        num_classes=len(data_config.label_value),
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=3,
        expand=2,
        dropout=0.1,
        grid_h=8,
        grid_w=16,
        eta_span=0.8,
    )
    cfg.update(**kwargs)

    model = JetMamba2026_2D_v1_1(**cfg)
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
