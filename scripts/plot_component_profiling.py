#!/usr/bin/env python
"""
plot_component_profiling.py

Generate stacked bar chart showing per-component latency breakdown at N=128 (left)
and N=1024 (right) for all 5 models.

Output: analysis/figures/component_profiling.pdf + .png

Usage:
    PATH="/sdf/data/atlas/u/dntounis/miniconda3/envs/tex-tools/bin:$PATH" \
      /sdf/data/atlas/u/dntounis/miniconda3/bin/python3.12 scripts/plot_component_profiling.py
"""

import json
import sys
import os

sys.path.insert(0, "/sdf/home/d/dntounis/.claude/skills/scientific-plotting/scripts")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import make_figure, setup_global_rcparams, load_palette

# ---------------------------------------------------------------------------
# Style setup (generic, serif, LaTeX)
# ---------------------------------------------------------------------------
setup_global_rcparams()

from generic_helpers import setup_generic_style
setup_generic_style(use_latex=False)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = "/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer"
INPUT_FILE = os.path.join(BASE_DIR, "analysis/results/component_profiling.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/figures")
OUTPUT_STEM = os.path.join(OUTPUT_DIR, "component_profiling")

MODEL_ORDER = ["jetmamba_unidir", "jetmamba_bidir", "jetmamba_rel", "jetmamba_hybrid", "ParT"]
MODEL_LABELS = {
    "jetmamba_unidir": "JetMamba\nUniDir",
    "jetmamba_bidir": "JetMamba\nBiDir",
    "jetmamba_rel": "JetMamba\nRel",
    "jetmamba_hybrid": "JetMamba\nHybrid",
    "ParT": "ParT",
}

COMPONENT_ORDER = [
    "preprocessing",
    "embedding",
    "positional_encoding",
    "relational",
    "pair_embed",
    "core_blocks",
    "cls_blocks",
    "pooling_classifier",
]

COMPONENT_LABELS = {
    "preprocessing": "Preprocessing",
    "embedding": "Embedding",
    "positional_encoding": "Position Encoding",
    "relational": "Anchor Relational",
    "pair_embed": r"Pair Embed ($\mathcal{O}(N^2)$)",
    "core_blocks": "Core Blocks (SSM/Attn)",
    "cls_blocks": "CLS Blocks",
    "pooling_classifier": "Pooling + Classifier",
}

COMPONENT_COLORS = {
    "preprocessing": "#4477AA",
    "embedding": "#66CCEE",
    "positional_encoding": "#228833",
    "relational": "#CCBB44",
    "pair_embed": "#EE6677",
    "core_blocks": "#AA3377",
    "cls_blocks": "#BBBBBB",
    "pooling_classifier": "#555555",
}


def load_results():
    with open(INPUT_FILE) as f:
        return json.load(f)


def get_component_latencies(model_data, seq_len_str):
    """Extract per-jet latencies (µs) per component for a given sequence length."""
    entry = model_data["scaling"].get(seq_len_str, {})
    if "error" in entry:
        return {}
    latencies = {}
    for comp in COMPONENT_ORDER:
        if comp in entry and isinstance(entry[comp], dict):
            latencies[comp] = entry[comp]["per_jet_us"]
    return latencies


def plot_stacked_bars(results, seq_lengths=("128", "1024")):
    """Create side-by-side stacked bar plots for two sequence lengths."""
    n_panels = len(seq_lengths)
    fig_width = 7.0  # double_col
    fig_height = 3.5

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for panel_idx, seq_len in enumerate(seq_lengths):
        ax = axes[panel_idx]
        x_pos = np.arange(len(MODEL_ORDER))
        bar_width = 0.6

        bottom = np.zeros(len(MODEL_ORDER))
        legend_handles = []
        legend_labels = []

        for comp in COMPONENT_ORDER:
            heights = []
            for model_key in MODEL_ORDER:
                model_data = results["models"].get(model_key, {})
                latencies = get_component_latencies(model_data, seq_len)
                heights.append(latencies.get(comp, 0.0))

            heights = np.array(heights)
            if heights.sum() == 0:
                continue

            bars = ax.bar(x_pos, heights, bar_width, bottom=bottom,
                         color=COMPONENT_COLORS[comp], edgecolor="white",
                         linewidth=0.3, label=COMPONENT_LABELS[comp])
            bottom += heights
            legend_handles.append(bars)
            legend_labels.append(COMPONENT_LABELS[comp])

            # Add percentage labels inside segments > 15% of total
            for i, (h, b) in enumerate(zip(heights, bottom - heights)):
                total = bottom[i]  # total so far won't be final until loop ends
                # We'll add labels after the loop

        # Add percentage labels after all bars are stacked
        for i, model_key in enumerate(MODEL_ORDER):
            model_data = results["models"].get(model_key, {})
            latencies = get_component_latencies(model_data, seq_len)
            total = sum(latencies.values())
            if total == 0:
                continue

            cum = 0.0
            for comp in COMPONENT_ORDER:
                val = latencies.get(comp, 0.0)
                if val == 0:
                    continue
                frac = val / total
                if frac > 0.15:
                    mid_y = cum + val / 2
                    ax.text(x_pos[i], mid_y, f"{frac*100:.0f}%",
                            ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")
                cum += val

        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
        ax.set_ylabel(r"Per-jet latency ($\mu$s)")
        ax.set_title(f"$N = {seq_len}$")

        # Set y-limit with some headroom
        ymax = bottom.max() * 1.15
        ax.set_ylim(0, ymax)

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), fontsize=9,
               frameon=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save PDF (primary)
    pdf_path = OUTPUT_STEM + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {pdf_path}")

    # Save PNG
    png_path = OUTPUT_STEM + ".png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {png_path}")

    plt.close(fig)
    return pdf_path, png_path


def plot_scaling_lines(results):
    """Secondary figure: component latency vs N for each model (line plot)."""
    fig_width = 7.0
    fig_height = 4.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height),
                             constrained_layout=True)

    seq_lengths = results["metadata"]["sequence_lengths"]
    x = np.array(seq_lengths)

    # Left panel: JetMamba models (all variants show core_blocks)
    ax = axes[0]
    mamba_models = ["jetmamba_unidir", "jetmamba_bidir", "jetmamba_rel", "jetmamba_hybrid"]
    okabe_ito = load_palette("okabe_ito")
    linestyles = ["-", "--", "-.", ":"]

    for i, model_key in enumerate(mamba_models):
        model_data = results["models"].get(model_key, {})
        core_us = []
        for n in seq_lengths:
            lat = get_component_latencies(model_data, str(n))
            core_us.append(lat.get("core_blocks", np.nan))
        label = MODEL_LABELS[model_key].replace("\n", " ")
        ax.plot(x, core_us, marker="o", markersize=4, color=okabe_ito[i+1],
                linestyle=linestyles[i], linewidth=1.5, label=label)

    ax.set_xlabel(r"Sequence length $N$")
    ax.set_ylabel(r"Core blocks latency ($\mu$s/jet)")
    ax.set_title("JetMamba: SSM blocks")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8, frameon=False)

    # Right panel: ParT component breakdown vs N
    ax = axes[1]
    part_data = results["models"].get("ParT", {})
    part_components = ["preprocessing", "embedding", "pair_embed", "core_blocks",
                       "cls_blocks", "pooling_classifier"]
    part_colors = ["#4477AA", "#66CCEE", "#EE6677", "#AA3377", "#BBBBBB", "#555555"]

    for comp, color in zip(part_components, part_colors):
        vals = []
        for n in seq_lengths:
            lat = get_component_latencies(part_data, str(n))
            vals.append(lat.get(comp, np.nan))
        ax.plot(x, vals, marker="s", markersize=4, color=color,
                linewidth=1.5, label=COMPONENT_LABELS[comp])

    ax.set_xlabel(r"Sequence length $N$")
    ax.set_ylabel(r"Per-component latency ($\mu$s/jet)")
    ax.set_title("ParT: component scaling")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    scaling_pdf = OUTPUT_STEM + "_scaling.pdf"
    fig.savefig(scaling_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {scaling_pdf}")

    scaling_png = OUTPUT_STEM + "_scaling.png"
    fig.savefig(scaling_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {scaling_png}")

    plt.close(fig)


def main():
    if not os.path.isfile(INPUT_FILE):
        print(f"ERROR: Results file not found: {INPUT_FILE}")
        print("Run benchmark_component_profiling.py first.")
        sys.exit(1)

    results = load_results()
    print(f"Loaded results from: {INPUT_FILE}")
    print(f"Models: {list(results['models'].keys())}")
    print(f"GPU: {results['metadata']['gpu_name']}")
    print()

    plot_stacked_bars(results, seq_lengths=("128", "1024"))
    print()
    plot_scaling_lines(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
