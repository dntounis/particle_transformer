#!/usr/bin/env python
"""
benchmark_component_profiling.py

Profile each model's forward pass at the component level using CUDA events.
Breaks down inference latency into: preprocessing, embedding, positional encoding,
relational/pairwise, core blocks, cls blocks, pooling, and classifier.

This quantifies WHY Mamba doesn't follow O(N) and ParT doesn't follow O(N²) at
tested sequence lengths — dominated by fixed overhead vs parallel GPU saturation.

Outputs results to: analysis/results/component_profiling.json

Usage:
    python scripts/benchmark_component_profiling.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.cuda.amp as amp

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = "/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer"
WEAVER_CORE = "/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/weaver-core"
DATA_CONFIG_PATH = os.path.join(BASE_DIR, "data/JetClass/JetClass_full.yaml")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "component_profiling.json")

SEQUENCE_LENGTHS = [32, 64, 128, 256, 512, 1024]

BATCH_SIZES = {
    32: 512,
    64: 512,
    128: 512,
    256: 256,
    512: 128,
    1024: 64,
}

BATCH_SIZES_PART = {
    32: 512,
    64: 512,
    128: 512,
    256: 128,
    512: 64,
    1024: 32,
}

WARMUP_ITERS = 10
TIMED_ITERS = 50
DEVICE = "cuda:0"

MODELS = [
    {
        "name": "JetMamba-UniDir",
        "short_name": "jetmamba_unidir",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.1.py"),
        "is_part": False,
        "variant": "v1.1",
    },
    {
        "name": "JetMamba-BiDir",
        "short_name": "jetmamba_bidir",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.2.py"),
        "is_part": False,
        "variant": "v1.2",
    },
    {
        "name": "JetMamba-Rel",
        "short_name": "jetmamba_rel",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.3.py"),
        "is_part": False,
        "variant": "v1.3",
    },
    {
        "name": "JetMamba-Hybrid",
        "short_name": "jetmamba_hybrid",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v2.1.py"),
        "is_part": False,
        "variant": "v2.1",
    },
    {
        "name": "ParT",
        "short_name": "ParT",
        "network_config": os.path.join(BASE_DIR, "networks/example_ParticleTransformer.py"),
        "is_part": True,
        "variant": "ParT",
    },
]


def import_module(path, name="_network_module"):
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data_config(data_config_path):
    sys.path.insert(0, WEAVER_CORE)
    from weaver.utils.data.config import DataConfig
    data_config = DataConfig.load(data_config_path, load_observers=False, load_reweight_info=False)
    return data_config


def load_model_random_weights(network_config_path, data_config, device):
    network_module = import_module(network_config_path)
    model, model_info = network_module.get_model(data_config)
    model = model.to(device)
    model.eval()
    return model, model_info


def create_synthetic_inputs_at_length(model_info, batch_size, seq_len, device):
    inputs = []
    for name in model_info["input_names"]:
        shape = model_info["input_shapes"][name]
        n_features = shape[1]
        tensor_shape = (batch_size, n_features, seq_len)
        if 'mask' in name:
            x = torch.ones(tensor_shape, dtype=torch.float32, device=device)
        else:
            x = torch.randn(tensor_shape, dtype=torch.float32, device=device)
        inputs.append(x)
    return tuple(inputs)


# ============================================================================
# Component-level profiled forward passes
# ============================================================================

# Load shared utility functions from v1.1 (identical across all JetMamba variants).
# Can't use `from networks.X import Y` because filenames contain dots.
_jetmamba_utils_mod = None

def _get_jetmamba_utils():
    global _jetmamba_utils_mod
    if _jetmamba_utils_mod is None:
        _jetmamba_utils_mod = import_module(
            os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.1.py"),
            "_jetmamba_utils"
        )
    return _jetmamba_utils_mod


def cuda_event_pair():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def profiled_forward_jetmamba_v11(model, inputs):
    """Profile JetMamba v1.1 (unidirectional) component by component."""
    pf_points, pf_features, pf_vectors, pf_mask = inputs[:4]
    timings = {}

    _utils = _get_jetmamba_utils()
    sort_and_trim_particles = _utils.sort_and_trim_particles
    delta_r = _utils.delta_r

    # -- preprocessing (sort_and_trim + geometry computation)
    s, e = cuda_event_pair()
    s.record()
    pf_features, pf_vectors, pf_points, pf_mask = sort_and_trim_particles(
        pf_features, pf_vectors, pf_points, pf_mask, trim=model.trim
    )
    valid_mask = pf_mask[:, 0].bool()
    dr = delta_r(pf_points).unsqueeze(1)
    features_with_geometry = torch.cat([pf_features, pf_points, dr], dim=1)
    e.record()
    torch.cuda.synchronize()
    timings["preprocessing"] = s.elapsed_time(e)

    # -- embedding
    s, e = cuda_event_pair()
    s.record()
    x = model.embed(features_with_geometry).permute(1, 0, 2).contiguous()
    e.record()
    torch.cuda.synchronize()
    timings["embedding"] = s.elapsed_time(e)

    # -- positional encoding
    s, e = cuda_event_pair()
    s.record()
    coords = torch.cat([pf_points, dr], dim=1).transpose(1, 2).contiguous()
    x = x + model.pos_embed(coords)
    x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
    e.record()
    torch.cuda.synchronize()
    timings["positional_encoding"] = s.elapsed_time(e)

    # -- core blocks (all Mamba layers)
    s, e = cuda_event_pair()
    s.record()
    for layer in model.layers:
        x = layer(x)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
    e.record()
    torch.cuda.synchronize()
    timings["core_blocks"] = s.elapsed_time(e)

    # -- pooling + classifier
    s, e = cuda_event_pair()
    s.record()
    pooled = x.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
    logits = model.classifier(model.final_norm(pooled))
    e.record()
    torch.cuda.synchronize()
    timings["pooling_classifier"] = s.elapsed_time(e)

    return logits, timings


def profiled_forward_jetmamba_v12(model, inputs):
    """Profile JetMamba v1.2 (bidirectional) component by component."""
    pf_points, pf_features, pf_vectors, pf_mask = inputs[:4]
    timings = {}

    _utils = _get_jetmamba_utils()
    sort_and_trim_particles = _utils.sort_and_trim_particles
    delta_r = _utils.delta_r

    # -- preprocessing
    s, e = cuda_event_pair()
    s.record()
    pf_features, pf_vectors, pf_points, pf_mask = sort_and_trim_particles(
        pf_features, pf_vectors, pf_points, pf_mask, trim=model.trim
    )
    valid_mask = pf_mask[:, 0].bool()
    dr = delta_r(pf_points).unsqueeze(1)
    features_with_geometry = torch.cat([pf_features, pf_points, dr], dim=1)
    e.record()
    torch.cuda.synchronize()
    timings["preprocessing"] = s.elapsed_time(e)

    # -- embedding
    s, e = cuda_event_pair()
    s.record()
    x = model.embed(features_with_geometry).permute(1, 0, 2).contiguous()
    e.record()
    torch.cuda.synchronize()
    timings["embedding"] = s.elapsed_time(e)

    # -- positional encoding
    s, e = cuda_event_pair()
    s.record()
    coords = torch.cat([pf_points, dr], dim=1).transpose(1, 2).contiguous()
    x = x + model.pos_embed(coords)
    x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
    e.record()
    torch.cuda.synchronize()
    timings["positional_encoding"] = s.elapsed_time(e)

    # -- core blocks (bidirectional Mamba)
    s, e = cuda_event_pair()
    s.record()
    for layer in model.layers:
        x = layer(x, valid_mask)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
    e.record()
    torch.cuda.synchronize()
    timings["core_blocks"] = s.elapsed_time(e)

    # -- pooling + classifier
    s, e = cuda_event_pair()
    s.record()
    pooled = x.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
    logits = model.classifier(model.final_norm(pooled))
    e.record()
    torch.cuda.synchronize()
    timings["pooling_classifier"] = s.elapsed_time(e)

    return logits, timings


def profiled_forward_jetmamba_v13(model, inputs):
    """Profile JetMamba v1.3 (bidirectional + anchor relational) component by component."""
    pf_points, pf_features, pf_vectors, pf_mask = inputs[:4]
    timings = {}

    _utils = _get_jetmamba_utils()
    sort_and_trim_particles = _utils.sort_and_trim_particles
    delta_r = _utils.delta_r

    # -- preprocessing
    s, e = cuda_event_pair()
    s.record()
    pf_features, pf_vectors, pf_points, pf_mask = sort_and_trim_particles(
        pf_features, pf_vectors, pf_points, pf_mask, trim=model.trim
    )
    valid_mask = pf_mask[:, 0].bool()
    dr = delta_r(pf_points).unsqueeze(1)
    features_with_geometry = torch.cat([pf_features, pf_points, dr], dim=1)
    e.record()
    torch.cuda.synchronize()
    timings["preprocessing"] = s.elapsed_time(e)

    # -- embedding
    s, e = cuda_event_pair()
    s.record()
    x = model.embed(features_with_geometry).permute(1, 0, 2).contiguous()
    e.record()
    torch.cuda.synchronize()
    timings["embedding"] = s.elapsed_time(e)

    # -- positional encoding
    s, e = cuda_event_pair()
    s.record()
    coords = torch.cat([pf_points, dr], dim=1).transpose(1, 2).contiguous()
    x = x + model.pos_embed(coords)
    x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
    e.record()
    torch.cuda.synchronize()
    timings["positional_encoding"] = s.elapsed_time(e)

    # -- relational (anchor pairwise)
    s, e = cuda_event_pair()
    s.record()
    x = model.anchor_relation(x, pf_vectors, pf_points, valid_mask)
    x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
    e.record()
    torch.cuda.synchronize()
    timings["relational"] = s.elapsed_time(e)

    # -- core blocks (bidirectional Mamba)
    s, e = cuda_event_pair()
    s.record()
    for layer in model.layers:
        x = layer(x, valid_mask)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
    e.record()
    torch.cuda.synchronize()
    timings["core_blocks"] = s.elapsed_time(e)

    # -- pooling + classifier
    s, e = cuda_event_pair()
    s.record()
    pooled = x.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
    logits = model.classifier(model.final_norm(pooled))
    e.record()
    torch.cuda.synchronize()
    timings["pooling_classifier"] = s.elapsed_time(e)

    return logits, timings


def profiled_forward_jetmamba_v21(model, inputs):
    """Profile JetMamba v2.1 (bidirectional + sparse attention) component by component."""
    pf_points, pf_features, pf_vectors, pf_mask = inputs[:4]
    timings = {}

    _utils = _get_jetmamba_utils()
    sort_and_trim_particles = _utils.sort_and_trim_particles
    delta_r = _utils.delta_r

    # -- preprocessing
    s, e = cuda_event_pair()
    s.record()
    pf_features, pf_vectors, pf_points, pf_mask = sort_and_trim_particles(
        pf_features, pf_vectors, pf_points, pf_mask, trim=model.trim
    )
    valid_mask = pf_mask[:, 0].bool()
    dr = delta_r(pf_points).unsqueeze(1)
    features_with_geometry = torch.cat([pf_features, pf_points, dr], dim=1)
    e.record()
    torch.cuda.synchronize()
    timings["preprocessing"] = s.elapsed_time(e)

    # -- embedding
    s, e = cuda_event_pair()
    s.record()
    x = model.embed(features_with_geometry).permute(1, 0, 2).contiguous()
    e.record()
    torch.cuda.synchronize()
    timings["embedding"] = s.elapsed_time(e)

    # -- positional encoding
    s, e = cuda_event_pair()
    s.record()
    coords = torch.cat([pf_points, dr], dim=1).transpose(1, 2).contiguous()
    x = x + model.pos_embed(coords)
    x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
    e.record()
    torch.cuda.synchronize()
    timings["positional_encoding"] = s.elapsed_time(e)

    # -- core blocks (Mamba) separately from attention blocks
    mamba_time = 0.0
    attention_time = 0.0
    for layer_idx, layer in enumerate(model.layers):
        s, e = cuda_event_pair()
        s.record()
        x = layer(x, valid_mask)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
        e.record()
        torch.cuda.synchronize()
        mamba_time += s.elapsed_time(e)

        layer_key = str(layer_idx)
        if layer_key in model.attention_layers:
            s, e = cuda_event_pair()
            s.record()
            x = model.attention_layers[layer_key](x, valid_mask)
            e.record()
            torch.cuda.synchronize()
            attention_time += s.elapsed_time(e)

    timings["core_blocks_mamba"] = mamba_time
    timings["core_blocks_attention"] = attention_time
    timings["core_blocks"] = mamba_time + attention_time

    # -- pooling + classifier
    s, e = cuda_event_pair()
    s.record()
    pooled = x.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
    logits = model.classifier(model.final_norm(pooled))
    e.record()
    torch.cuda.synchronize()
    timings["pooling_classifier"] = s.elapsed_time(e)

    return logits, timings


def profiled_forward_part(model, inputs):
    """Profile ParT component by component via model.mod internals."""
    pf_points, pf_features, pf_vectors, pf_mask = inputs[:4]
    timings = {}

    part = model.mod  # Inner ParticleTransformer

    # -- preprocessing (trimmer + padding_mask)
    s, e = cuda_event_pair()
    s.record()
    with torch.no_grad():
        x, v, mask, uu = part.trimmer(pf_features, pf_vectors, pf_mask, None)
        padding_mask = ~mask.squeeze(1)
    e.record()
    torch.cuda.synchronize()
    timings["preprocessing"] = s.elapsed_time(e)

    # -- embedding
    s, e = cuda_event_pair()
    s.record()
    x = part.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)
    e.record()
    torch.cuda.synchronize()
    timings["embedding"] = s.elapsed_time(e)

    # -- pair embedding (O(N²) pairwise features)
    s, e = cuda_event_pair()
    s.record()
    attn_mask = None
    if v is not None and part.pair_embed is not None:
        attn_mask = part.pair_embed(v, None).view(-1, v.size(-1), v.size(-1))
    e.record()
    torch.cuda.synchronize()
    timings["pair_embed"] = s.elapsed_time(e)

    # -- core blocks (self-attention)
    s, e = cuda_event_pair()
    s.record()
    for block in part.blocks:
        x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
    e.record()
    torch.cuda.synchronize()
    timings["core_blocks"] = s.elapsed_time(e)

    # -- cls blocks (class-token attention)
    s, e = cuda_event_pair()
    s.record()
    cls_tokens = part.cls_token.expand(1, x.size(1), -1)
    for block in part.cls_blocks:
        cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
    e.record()
    torch.cuda.synchronize()
    timings["cls_blocks"] = s.elapsed_time(e)

    # -- norm + classifier
    s, e = cuda_event_pair()
    s.record()
    x_cls = part.norm(cls_tokens).squeeze(0)
    output = part.fc(x_cls) if part.fc is not None else x_cls
    e.record()
    torch.cuda.synchronize()
    timings["pooling_classifier"] = s.elapsed_time(e)

    return output, timings


# ============================================================================
# Dispatch
# ============================================================================

PROFILED_FORWARDS = {
    "v1.1": profiled_forward_jetmamba_v11,
    "v1.2": profiled_forward_jetmamba_v12,
    "v1.3": profiled_forward_jetmamba_v13,
    "v2.1": profiled_forward_jetmamba_v21,
    "ParT": profiled_forward_part,
}


def measure_total_forward(model, inputs):
    """Measure total forward pass time with a single CUDA event pair."""
    s, e = cuda_event_pair()
    s.record()
    _ = model(*inputs)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def profile_model_at_length(model, model_info, variant, batch_size, seq_len, device):
    """Run profiling for one (model, seq_len) combination."""
    inputs = create_synthetic_inputs_at_length(model_info, batch_size, seq_len, device)
    profiled_fwd = PROFILED_FORWARDS[variant]

    # Warmup
    with torch.no_grad(), amp.autocast():
        for _ in range(WARMUP_ITERS):
            _ = model(*inputs)
            torch.cuda.synchronize()

    # Timed runs: component-level
    all_timings = []
    with torch.no_grad(), amp.autocast():
        for _ in range(TIMED_ITERS):
            _, timings = profiled_fwd(model, inputs)
            all_timings.append(timings)

    # Timed runs: total forward
    total_times = []
    with torch.no_grad(), amp.autocast():
        for _ in range(TIMED_ITERS):
            t = measure_total_forward(model, inputs)
            total_times.append(t)

    # Aggregate
    components = list(all_timings[0].keys())
    result = {"batch_size": batch_size}

    for comp in components:
        vals = [t[comp] for t in all_timings]
        result[comp] = {
            "mean_ms": round(float(np.mean(vals)), 4),
            "std_ms": round(float(np.std(vals)), 4),
            "per_jet_us": round(float(np.mean(vals)) / batch_size * 1000.0, 3),
        }

    total_arr = np.array(total_times)
    result["total"] = {
        "mean_ms": round(float(np.mean(total_arr)), 4),
        "std_ms": round(float(np.std(total_arr)), 4),
        "per_jet_us": round(float(np.mean(total_arr)) / batch_size * 1000.0, 3),
    }

    # Coverage check
    sum_of_parts = sum(result[c]["mean_ms"] for c in components if not c.startswith("core_blocks_"))
    # For v2.1, core_blocks is already the sum of mamba+attention, so use it
    coverage_pct = (sum_of_parts / result["total"]["mean_ms"]) * 100 if result["total"]["mean_ms"] > 0 else 0
    result["coverage_pct"] = round(coverage_pct, 1)

    del inputs
    torch.cuda.empty_cache()
    return result


def main():
    print("=" * 70)
    print(" Component-Level GPU Profiling")
    print(f" Sequence lengths: {SEQUENCE_LENGTHS}")
    print(f" Warmup: {WARMUP_ITERS}, Timed: {TIMED_ITERS}")
    print(f" Device: {DEVICE}")
    print("=" * 70)
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print()

    # Change to BASE_DIR so relative imports work
    os.chdir(BASE_DIR)
    sys.path.insert(0, BASE_DIR)

    print("Loading data config...")
    data_config = load_data_config(DATA_CONFIG_PATH)
    print()

    results = {
        "metadata": {
            "sequence_lengths": SEQUENCE_LENGTHS,
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
            "device": DEVICE,
            "gpu_name": gpu_name,
            "gpu_memory_gb": round(gpu_mem_gb, 1),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "models": {},
    }

    for model_def in MODELS:
        name = model_def["name"]
        short_name = model_def["short_name"]
        variant = model_def["variant"]
        is_part = model_def["is_part"]

        print("=" * 70)
        print(f" Model: {name} (variant={variant})")
        print("=" * 70)

        if not os.path.isfile(model_def["network_config"]):
            print(f"  SKIP: network config not found at {model_def['network_config']}")
            continue

        try:
            model, model_info = load_model_random_weights(
                model_def["network_config"], data_config, DEVICE
            )
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            continue

        model_results = {"name": name, "variant": variant, "scaling": {}}
        batch_size_map = BATCH_SIZES_PART if is_part else BATCH_SIZES

        for seq_len in SEQUENCE_LENGTHS:
            batch_size = batch_size_map[seq_len]
            print(f"  N={seq_len:4d}, batch_size={batch_size:4d} ... ", end="", flush=True)

            try:
                result = profile_model_at_length(
                    model, model_info, variant, batch_size, seq_len, DEVICE
                )
                total_us = result["total"]["per_jet_us"]
                coverage = result["coverage_pct"]
                print(f"total={total_us:.1f} us/jet, coverage={coverage:.0f}%")
                model_results["scaling"][str(seq_len)] = result

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM! Skipping.")
                    torch.cuda.empty_cache()
                    model_results["scaling"][str(seq_len)] = {"error": "OOM"}
                else:
                    print(f"ERROR: {e}")
                    torch.cuda.empty_cache()
                    model_results["scaling"][str(seq_len)] = {"error": str(e)}

        results["models"][short_name] = model_results

        del model
        torch.cuda.empty_cache()
        print()

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 70)
    print(f" Results saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
