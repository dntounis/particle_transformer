#!/usr/bin/env python
"""
benchmark_speed.py

Benchmark inference speed for all 5 trained models on synthetic data.
Measures avg/std latency per batch, throughput (jets/sec), and per-jet latency.

Outputs results to: analysis/results/inference_speed.json

Usage:
    python scripts/benchmark_speed.py
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "inference_speed.json")

BATCH_SIZE = 512
WARMUP_BATCHES = 10
TIMED_BATCHES = 100
DEVICE = "cuda:0"

# Model definitions
MODELS = [
    {
        "name": "JetMamba-UniDir",
        "short_name": "jetmamba_unidir",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.1.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/jetmamba_2026_1D_v1.1_best_epoch_state.pt"),
    },
    {
        "name": "JetMamba-BiDir",
        "short_name": "jetmamba_bidir",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.2.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/jetmamba_2026_1D_v1.2_best_epoch_state.pt"),
    },
    {
        "name": "JetMamba-Rel",
        "short_name": "jetmamba_rel",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v1.3.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/jetmamba_2026_1D_v1.3_best_epoch_state.pt"),
    },
    {
        "name": "JetMamba-Hybrid",
        "short_name": "jetmamba_hybrid",
        "network_config": os.path.join(BASE_DIR, "networks/jetmamba_2026_1D_v2.1.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/jetmamba_2026_1D_v2.1_best_epoch_state.pt"),
    },
    {
        "name": "ParT",
        "short_name": "ParT",
        "network_config": os.path.join(BASE_DIR, "networks/example_ParticleTransformer.py"),
        "checkpoint": os.path.join(BASE_DIR, "models/ParT_full.pt"),
    },
]


def import_module(path, name="_network_module"):
    """Import a Python module from file path (same as weaver's import_tools)."""
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

    # Load checkpoint weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model, model_info


def create_synthetic_inputs(model_info, batch_size, device):
    """Create synthetic input tensors matching the model's expected shapes."""
    inputs = []
    for name in model_info["input_names"]:
        shape = model_info["input_shapes"][name]
        # shape is (1, n_features, seq_len) -- replace batch dim with our batch_size
        tensor_shape = (batch_size,) + shape[1:]
        if 'mask' in name:
            x = torch.ones(tensor_shape, dtype=torch.float32, device=device)
        else:
            x = torch.randn(tensor_shape, dtype=torch.float32, device=device)
        inputs.append(x)
    return tuple(inputs)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_model(model, inputs, warmup=WARMUP_BATCHES, timed=TIMED_BATCHES):
    """
    Run warmup + timed batches and measure GPU execution time.
    Uses torch.cuda.synchronize() for accurate timing.
    """
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad(), amp.autocast():
        for _ in range(warmup):
            _ = model(*inputs)
            torch.cuda.synchronize(device)

    # Timed runs
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
    print(" Inference Speed Benchmark")
    print(f" Batch size: {BATCH_SIZE}")
    print(f" Warmup batches: {WARMUP_BATCHES}")
    print(f" Timed batches: {TIMED_BATCHES}")
    print(f" Device: {DEVICE}")
    print("=" * 70)
    print()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print()

    # Load data config
    print("Loading data config...")
    data_config = load_data_config(DATA_CONFIG_PATH)
    print(f"  Input names: {list(data_config.input_names)}")
    print(f"  Input shapes: {dict(data_config.input_shapes)}")
    print()

    # Results container
    results = {
        "metadata": {
            "batch_size": BATCH_SIZE,
            "warmup_batches": WARMUP_BATCHES,
            "timed_batches": TIMED_BATCHES,
            "device": DEVICE,
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "models": {},
    }

    for model_def in MODELS:
        name = model_def["name"]
        short_name = model_def["short_name"]
        print("-" * 70)
        print(f" Benchmarking: {name}")
        print(f"   Network: {model_def['network_config']}")
        print(f"   Checkpoint: {model_def['checkpoint']}")
        print("-" * 70)

        # Verify files
        if not os.path.isfile(model_def["network_config"]):
            print(f"  SKIP: network config not found")
            continue
        if not os.path.isfile(model_def["checkpoint"]):
            print(f"  SKIP: checkpoint not found")
            continue

        # Load model
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

        # Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Create synthetic inputs
        inputs = create_synthetic_inputs(model_info, BATCH_SIZE, DEVICE)
        input_shapes_str = {k: list(model_info["input_shapes"][k]) for k in model_info["input_names"]}
        print(f"  Input shapes (per-sample): {input_shapes_str}")
        print(f"  Batch input shapes: {[tuple(x.shape) for x in inputs]}")

        # Run benchmark
        print(f"  Running {WARMUP_BATCHES} warmup + {TIMED_BATCHES} timed batches...")
        try:
            times_ms = benchmark_model(model, inputs)
        except Exception as e:
            print(f"  ERROR during benchmark: {e}")
            # Clean up
            del model
            torch.cuda.empty_cache()
            continue

        # Compute statistics
        avg_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))
        throughput = BATCH_SIZE / (avg_ms / 1000.0)  # jets/sec
        latency_us = (avg_ms / BATCH_SIZE) * 1000.0  # microseconds per jet

        print(f"  Results:")
        print(f"    Avg latency:   {avg_ms:.3f} +/- {std_ms:.3f} ms/batch")
        print(f"    Throughput:    {throughput:,.0f} jets/sec")
        print(f"    Per-jet:       {latency_us:.2f} us/jet")
        print()

        results["models"][short_name] = {
            "name": name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "avg_ms_per_batch": round(avg_ms, 4),
            "std_ms_per_batch": round(std_ms, 4),
            "throughput_jets_per_sec": round(throughput, 1),
            "latency_us_per_jet": round(latency_us, 3),
            "all_times_ms": [round(t, 4) for t in times_ms.tolist()],
        }

        # Clean up to free GPU memory between models
        del model, inputs
        torch.cuda.empty_cache()

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 70)
    print(f" Results saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
