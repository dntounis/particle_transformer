#!/usr/bin/env python
"""
benchmark_scaling.py

Measure inference latency vs sequence length N for all 5 models.
Demonstrates O(N) scaling of Mamba variants vs O(N^2) of ParT.

Creates synthetic inputs at varying N (32, 64, 128, 256, 512, 1024),
with adaptive batch sizes to avoid OOM at large N.

Outputs results to: analysis/results/scaling_study.json

Usage:
    python scripts/benchmark_scaling.py
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "scaling_study.json")

SEQUENCE_LENGTHS = [32, 64, 128, 256, 512, 1024]

# Adaptive batch sizes: reduce at large N to avoid OOM (especially for ParT)
BATCH_SIZES = {
    32: 512,
    64: 512,
    128: 512,
    256: 256,
    512: 128,
    1024: 64,
}

# For ParT, further reduce batch size at large N due to O(N^2) memory
BATCH_SIZES_PART = {
    32: 512,
    64: 512,
    128: 512,
    256: 128,
    512: 64,
    1024: 32,
}

WARMUP_BATCHES = 5
TIMED_BATCHES = 50
DEVICE = "cuda:0"

# Model definitions
MODELS = [
    {
        "name": "JetMamba-UniDir",
        "short_name": "jetmamba_unidir",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.1.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/jetmamba_2026_1D_v1.1_best_epoch_state.pt"),
        "is_part": False,
    },
    {
        "name": "JetMamba-BiDir",
        "short_name": "jetmamba_bidir",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.2.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/jetmamba_2026_1D_v1.2_best_epoch_state.pt"),
        "is_part": False,
    },
    {
        "name": "JetMamba-Rel",
        "short_name": "jetmamba_rel",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.3.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/jetmamba_2026_1D_v1.3_best_epoch_state.pt"),
        "is_part": False,
    },
    {
        "name": "JetMamba-Hybrid",
        "short_name": "jetmamba_hybrid",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v2.1.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/jetmamba_2026_1D_v2.1_best_epoch_state.pt"),
        "is_part": False,
    },
    {
        "name": "ParT",
        "short_name": "ParT",
        "network_config": os.path.join(BASE_DIR, "networks/example_ParticleTransformer.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/ParT_full.pt"),
        "is_part": True,
    },
]


def import_module(path, name="_network_module"):
    """Import a Python module from file path."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data_config(data_config_path):
    """Load weaver DataConfig to get input shapes."""
    sys.path.insert(0, WEAVER_CORE)
    from weaver.utils.data.config import DataConfig
    data_config = DataConfig.load(data_config_path, load_observers=False, load_reweight_info=False)
    return data_config


def load_model(network_config_path, checkpoint_path, data_config, device):
    """Load model from network config and checkpoint."""
    network_module = import_module(network_config_path)
    model, model_info = network_module.get_model(data_config)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model, model_info


def create_synthetic_inputs_at_length(model_info, batch_size, seq_len, device):
    """
    Create synthetic input tensors with a specific sequence length.

    The model_info contains input_shapes like (1, n_features, 128).
    We replace the last dimension (sequence length) with our target seq_len.
    """
    inputs = []
    for name in model_info["input_names"]:
        shape = model_info["input_shapes"][name]
        # shape is (1, n_features, orig_seq_len) -- replace batch and seq_len
        n_features = shape[1]
        tensor_shape = (batch_size, n_features, seq_len)
        if 'mask' in name:
            x = torch.ones(tensor_shape, dtype=torch.float32, device=device)
        else:
            x = torch.randn(tensor_shape, dtype=torch.float32, device=device)
        inputs.append(x)
    return tuple(inputs)


def benchmark_at_length(model, inputs, warmup=WARMUP_BATCHES, timed=TIMED_BATCHES):
    """
    Run warmup + timed batches and measure GPU execution time.
    Returns array of per-batch times in milliseconds.
    """
    device = next(model.parameters()).device

    with torch.no_grad(), amp.autocast():
        for _ in range(warmup):
            _ = model(*inputs)
            torch.cuda.synchronize(device)

    times_ms = []
    with torch.no_grad(), amp.autocast():
        for _ in range(timed):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(*inputs)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    return np.array(times_ms)


def main():
    print("=" * 70)
    print(" Sequence Length Scaling Study")
    print(f" Sequence lengths: {SEQUENCE_LENGTHS}")
    print(f" Warmup batches: {WARMUP_BATCHES}")
    print(f" Timed batches: {TIMED_BATCHES}")
    print(f" Device: {DEVICE}")
    print("=" * 70)
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print()

    # Load data config (needed for model initialization)
    print("Loading data config...")
    data_config = load_data_config(DATA_CONFIG_PATH)
    print()

    results = {
        "metadata": {
            "sequence_lengths": SEQUENCE_LENGTHS,
            "warmup_batches": WARMUP_BATCHES,
            "timed_batches": TIMED_BATCHES,
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
        is_part = model_def["is_part"]

        print("=" * 70)
        print(f" Model: {name}")
        print("=" * 70)

        if not os.path.isfile(model_def["network_config"]):
            print(f"  SKIP: network config not found")
            continue
        if not os.path.isfile(model_def["checkpoint"]):
            print(f"  SKIP: checkpoint not found")
            continue

        try:
            model, model_info = load_model(
                model_def["network_config"],
                model_def["checkpoint"],
                data_config,
                DEVICE,
            )
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        model_results = {"name": name, "scaling": {}}
        batch_size_map = BATCH_SIZES_PART if is_part else BATCH_SIZES

        for seq_len in SEQUENCE_LENGTHS:
            batch_size = batch_size_map[seq_len]
            print(f"  N={seq_len:4d}, batch_size={batch_size:4d} ... ", end="", flush=True)

            try:
                inputs = create_synthetic_inputs_at_length(
                    model_info, batch_size, seq_len, DEVICE
                )
                times_ms = benchmark_at_length(model, inputs)

                avg_ms = float(np.mean(times_ms))
                std_ms = float(np.std(times_ms))
                # Normalize to per-jet latency for fair comparison across batch sizes
                per_jet_us = (avg_ms / batch_size) * 1000.0

                print(f"avg={avg_ms:.2f}ms/batch, per_jet={per_jet_us:.2f}us")

                model_results["scaling"][str(seq_len)] = {
                    "batch_size": batch_size,
                    "avg_ms_per_batch": round(avg_ms, 4),
                    "std_ms_per_batch": round(std_ms, 4),
                    "per_jet_latency_us": round(per_jet_us, 3),
                    "throughput_jets_per_sec": round(batch_size / (avg_ms / 1000.0), 1),
                }

                del inputs
                torch.cuda.empty_cache()

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
