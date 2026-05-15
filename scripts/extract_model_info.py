#!/usr/bin/env python3
"""
extract_model_info.py

Parse the --print logs from weaver to extract parameter counts and FLOPs.
Also loads inference_speed.json to combine into a single summary table.

Outputs: analysis/results/model_summary.json

Usage:
    python scripts/extract_model_info.py
"""

import os
import re
import json

BASE_DIR = "/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer"
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "analysis/results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "model_summary.json")

MODELS = [
    {"name": "JetMamba-UniDir", "short_name": "jetmamba_unidir", "log": "print_jetmamba_v1.1.log"},
    {"name": "JetMamba-BiDir", "short_name": "jetmamba_bidir", "log": "print_jetmamba_v1.2.log"},
    {"name": "JetMamba-Rel", "short_name": "jetmamba_rel", "log": "print_jetmamba_v1.3.log"},
    {"name": "JetMamba-Hybrid", "short_name": "jetmamba_hybrid", "log": "print_jetmamba_v2.1.log"},
    {"name": "ParT", "short_name": "ParT", "log": "print_ParT.log"},
]


def parse_print_log(filepath):
    """Extract params and FLOPs from a weaver --print log."""
    if not os.path.isfile(filepath):
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    result = {}

    # Look for parameter count patterns
    # weaver prints: "Total params: 1,234,567" or similar
    params_match = re.search(r'Total params[:\s]*([\d,]+)', content, re.IGNORECASE)
    if params_match:
        result['total_params'] = int(params_match.group(1).replace(',', ''))

    trainable_match = re.search(r'Trainable params[:\s]*([\d,]+)', content, re.IGNORECASE)
    if trainable_match:
        result['trainable_params'] = int(trainable_match.group(1).replace(',', ''))

    # Look for FLOPs patterns
    # weaver/torchinfo prints: "Total mult-adds (M): 123.45" or "FLOPs: 123456789"
    flops_patterns = [
        r'Total mult-adds \(([GMK])\)[:\s]*([\d.]+)',
        r'FLOPs[:\s]*([\d,.]+)',
        r'MACs[:\s]*([\d,.]+)',
        r'total_flops[:\s]*([\d,.]+)',
    ]

    for pattern in flops_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                unit, value = groups
                value = float(value)
                multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9}
                result['flops'] = int(value * multipliers.get(unit, 1))
            else:
                result['flops'] = int(groups[0].replace(',', ''))
            break

    # Also look for model summary lines with "Forward/backward pass size"
    size_match = re.search(r'Forward/backward pass size \(MB\)[:\s]*([\d.]+)', content)
    if size_match:
        result['forward_pass_mb'] = float(size_match.group(1))

    return result if result else None


def load_speed_results():
    """Load inference_speed.json if available."""
    speed_file = os.path.join(RESULTS_DIR, "inference_speed.json")
    if os.path.isfile(speed_file):
        with open(speed_file, 'r') as f:
            return json.load(f)
    return None


def load_classification_results():
    """Load classification_results.json if available."""
    cls_file = os.path.join(RESULTS_DIR, "classification_results.json")
    if os.path.isfile(cls_file):
        with open(cls_file, 'r') as f:
            return json.load(f)
    return None


def main():
    print("Extracting model information from --print logs...")
    print(f"Log directory: {LOG_DIR}")
    print()

    summary = {}

    for model_def in MODELS:
        name = model_def["name"]
        short_name = model_def["short_name"]
        log_path = os.path.join(LOG_DIR, model_def["log"])

        print(f"  {name}: ", end="")
        info = parse_print_log(log_path)
        if info:
            print(f"params={info.get('total_params', '?'):,}, flops={info.get('flops', '?')}")
        else:
            print("log not found or no data")
            info = {}

        summary[short_name] = {"name": name, **info}

    # Merge speed results
    speed_data = load_speed_results()
    if speed_data:
        print("\n  Merging inference speed data...")
        for short_name, model_info in speed_data.get("models", {}).items():
            if short_name in summary:
                summary[short_name]["latency_us_per_jet"] = model_info.get("latency_us_per_jet")
                summary[short_name]["throughput_jets_per_sec"] = model_info.get("throughput_jets_per_sec")

    # Merge classification results
    cls_data = load_classification_results()
    if cls_data:
        print("  Merging classification results...")
        for model_name, metrics in cls_data.items():
            # Match model display name to short_name
            for short_name, info in summary.items():
                if info["name"] == model_name:
                    if isinstance(metrics, dict):
                        summary[short_name]["accuracy"] = metrics.get("overall_accuracy")
                        summary[short_name]["macro_auc"] = metrics.get("macro_auc")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {OUTPUT_FILE}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'Params':>12} {'FLOPs':>12} {'Latency':>12} {'Accuracy':>10}")
    print("-" * 80)
    for short_name, info in summary.items():
        params_str = f"{info.get('total_params', 0):,}" if 'total_params' in info else "---"
        flops_str = f"{info.get('flops', 0)/1e6:.1f}M" if 'flops' in info else "---"
        lat_str = f"{info.get('latency_us_per_jet', 0):.1f}us" if 'latency_us_per_jet' in info else "---"
        acc_str = f"{info.get('accuracy', 0)*100:.2f}%" if 'accuracy' in info else "---"
        print(f"{info['name']:<20} {params_str:>12} {flops_str:>12} {lat_str:>12} {acc_str:>10}")
    print("=" * 80)


if __name__ == "__main__":
    main()
