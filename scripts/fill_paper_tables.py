#!/usr/bin/env python3
"""
fill_paper_tables.py

Once all results are available (inference, speed, scaling), this script
prints the LaTeX table content that can be pasted into the paper.

Reads from:
  - analysis/results/classification_results.json
  - analysis/results/inference_speed.json
  - analysis/results/scaling_study.json
  - analysis/results/model_summary.json

Usage:
    python scripts/fill_paper_tables.py
"""

import os
import json
import sys

BASE_DIR = "/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer"
RESULTS_DIR = os.path.join(BASE_DIR, "analysis/results")

MODEL_ORDER = [
    ("JetMamba-UniDir", "jetmamba_unidir"),
    ("JetMamba-BiDir", "jetmamba_bidir"),
    ("JetMamba-Rel", "jetmamba_rel"),
    ("JetMamba-Hybrid", "jetmamba_hybrid"),
    ("ParT", "ParT"),  # Uses ParT_full.pt (official published checkpoint)
]

CLASS_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def format_params(n):
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(n)


def print_main_results_table(cls_data):
    """Print tab:main_results content."""
    print("\n" + "=" * 70)
    print("TABLE: tab:main_results (Overall Performance)")
    print("=" * 70)
    print("Model & Accuracy (\\%) & AUC & Complexity \\\\")
    print("\\midrule")

    complexities = {
        "JetMamba-UniDir": "$\\mathcal{O}(N)$",
        "JetMamba-BiDir": "$\\mathcal{O}(N)$",
        "JetMamba-Rel": "$\\mathcal{O}(NK)$",
        "JetMamba-Hybrid": "$\\mathcal{O}(NK)$",
        "ParT": "$\\mathcal{O}(N^2)$",
    }

    for display_name, short_name in MODEL_ORDER:
        if display_name in cls_data:
            metrics = cls_data[display_name]
            acc = metrics.get("overall_accuracy", 0) * 100
            auc_val = metrics.get("macro_auc", 0)
            complexity = complexities[display_name]
            print(f"{display_name:<20} & {acc:.2f} & {auc_val:.4f} & {complexity} \\\\")
        else:
            print(f"{display_name:<20} & --- & --- & {complexities[display_name]} \\\\")


def print_per_class_table(cls_data):
    """Print tab:per_class content."""
    print("\n" + "=" * 70)
    print("TABLE: tab:per_class (Per-Class Accuracy)")
    print("=" * 70)
    print("Class & UniDir & BiDir & Rel & Hybrid & ParT \\\\")
    print("\\midrule")

    for cls in CLASS_NAMES:
        row = f"{cls:<6}"
        for display_name, _ in MODEL_ORDER:
            if display_name in cls_data:
                per_class = cls_data[display_name].get("per_class_accuracy", {})
                val = per_class.get(cls, None)
                if val is not None:
                    row += f" & {val*100:.2f}"
                else:
                    row += " & ---"
            else:
                row += " & ---"
        row += " \\\\"
        print(row)


def print_rejection_table(cls_data):
    """Print tab:rejection content."""
    print("\n" + "=" * 70)
    print("TABLE: tab:rejection (Background Rejection)")
    print("=" * 70)

    tasks = [
        ("Hbb", "QCD", "$H{\\to}b\\bar{b}$ vs QCD"),
        ("Hcc", "QCD", "$H{\\to}c\\bar{c}$ vs QCD"),
        ("H4q", "QCD", "$H{\\to}4q$ vs QCD"),
        ("Tbqq", "QCD", "$t{\\to}bqq$ vs QCD"),
    ]

    print("Task & UniDir & BiDir & Rel & Hybrid & ParT \\\\")
    print("\\midrule")

    for sig, bkg, label in tasks:
        row = f"{label:<25}"
        for display_name, _ in MODEL_ORDER:
            if display_name in cls_data:
                rej_data = cls_data[display_name].get("rejections", {})
                key = f"{sig}_vs_{bkg}"
                if key in rej_data:
                    val = rej_data[key].get("50%", None)
                    if val and val != "inf":
                        row += f" & {int(float(val)):,}"
                    else:
                        row += " & $\\infty$"
                else:
                    row += " & ---"
            else:
                row += " & ---"
        row += " \\\\"
        print(row)


def print_efficiency_table(speed_data, summary_data):
    """Print tab:efficiency content."""
    print("\n" + "=" * 70)
    print("TABLE: tab:efficiency (Computational Efficiency)")
    print("=" * 70)
    print("Model & Parameters & FLOPs/jet & Latency ($\\mu$s/jet) \\\\")
    print("\\midrule")

    for display_name, short_name in MODEL_ORDER:
        params_str = "---"
        flops_str = "---"
        lat_str = "---"

        if summary_data and short_name in summary_data:
            info = summary_data[short_name]
            if "total_params" in info:
                params_str = format_params(info["total_params"])
            if "flops" in info:
                flops_str = f"{info['flops']/1e6:.1f}M"

        if speed_data and short_name in speed_data.get("models", {}):
            lat = speed_data["models"][short_name].get("latency_us_per_jet", None)
            if lat:
                lat_str = f"{lat:.1f}"

        print(f"{display_name:<20} & {params_str:>8} & {flops_str:>8} & {lat_str:>8} \\\\")


def print_scaling_summary(scaling_data):
    """Print scaling study key numbers for the paper text."""
    print("\n" + "=" * 70)
    print("SCALING STUDY: Key numbers for paper text")
    print("=" * 70)

    if not scaling_data:
        print("  scaling_study.json not found")
        return

    # Compare ParT vs JetMamba at different N
    for N in [128, 512, 1024]:
        n_str = str(N)
        print(f"\n  At N={N}:")
        for display_name, short_name in MODEL_ORDER:
            if short_name in scaling_data.get("models", {}):
                scaling = scaling_data["models"][short_name].get("scaling", {})
                if n_str in scaling and "per_jet_latency_us" in scaling[n_str]:
                    lat = scaling[n_str]["per_jet_latency_us"]
                    print(f"    {display_name:<20}: {lat:.1f} us/jet")

        # Compute speedup ratios
        part_data = scaling_data.get("models", {}).get("ParT", {}).get("scaling", {})
        if n_str in part_data and "per_jet_latency_us" in part_data[n_str]:
            part_lat = part_data[n_str]["per_jet_latency_us"]
            print(f"\n    Speedup vs ParT at N={N}:")
            for display_name, short_name in MODEL_ORDER[:-1]:
                model_scaling = scaling_data.get("models", {}).get(short_name, {}).get("scaling", {})
                if n_str in model_scaling and "per_jet_latency_us" in model_scaling[n_str]:
                    model_lat = model_scaling[n_str]["per_jet_latency_us"]
                    speedup = part_lat / model_lat
                    print(f"      {display_name:<20}: {speedup:.1f}x faster")


def main():
    print("JetMamba Paper Table Generator")
    print("Reading results from:", RESULTS_DIR)

    cls_data = load_json("classification_results.json")
    speed_data = load_json("inference_speed.json")
    scaling_data = load_json("scaling_study.json")
    summary_data = load_json("model_summary.json")

    if cls_data:
        print_main_results_table(cls_data)
        print_per_class_table(cls_data)
        print_rejection_table(cls_data)
    else:
        print("\n  classification_results.json not yet available")

    print_efficiency_table(speed_data, summary_data)
    print_scaling_summary(scaling_data)

    print("\n" + "=" * 70)
    print("Done. Copy the table content above into the .tex file.")
    print("=" * 70)


if __name__ == "__main__":
    main()
