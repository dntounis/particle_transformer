#!/usr/bin/env python3
"""
plot_training_curves.py

Parse training logs for all 5 models and produce:
1. Validation accuracy vs epoch (all models on one plot)
2. Training loss vs epoch (all models on one plot)

Outputs to: analysis/results/figures/training_curves_*.pdf

Usage:
    python scripts/plot_training_curves.py
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = "/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer"
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/results/figures")

MODELS = [
    {"name": "JetMamba-UniDir", "log": "jetmamba_2026_1D_v1.1_23Mar2026.log", "color": "#332288", "ls": "-"},
    {"name": "JetMamba-BiDir", "log": "jetmamba_2026_1D_v1.2_23Mar2026.log", "color": "#88CCEE", "ls": "--"},
    {"name": "JetMamba-Rel", "log": "jetmamba_2026_1D_v1.3_26Mar2026.log", "color": "#44AA99", "ls": "-."},
    {"name": "JetMamba-Hybrid", "log": "jetmamba_2026_1D_v2.1_26Mar2026.log", "color": "#DDCC77", "ls": ":"},
    {"name": "ParT", "log": "ParT_3Aug2025.log", "color": "#CC6677", "ls": "-"},
]


def parse_log(filepath):
    """Extract epoch-level validation accuracy and training loss."""
    val_data = []
    train_data = []

    val_pattern = re.compile(
        r'Epoch #(\d+): Current validation metric: ([\d.]+) \(best: ([\d.]+)\)'
    )
    train_pattern = re.compile(
        r'Train AvgLoss: ([\d.]+), AvgAcc: ([\d.]+)'
    )
    epoch_pattern = re.compile(r'Epoch #(\d+) training')

    current_epoch = 0

    with open(filepath, 'r') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            val_match = val_pattern.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                val_acc = float(val_match.group(2))
                best_acc = float(val_match.group(3))
                val_data.append((epoch, val_acc, best_acc))

            train_match = train_pattern.search(line)
            if train_match:
                loss = float(train_match.group(1))
                acc = float(train_match.group(2))
                train_data.append((current_epoch, loss, acc))

    return val_data, train_data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })

    # Parse all logs
    all_val = {}
    all_train = {}
    for model_def in MODELS:
        log_path = os.path.join(LOG_DIR, model_def["log"])
        if not os.path.isfile(log_path):
            print(f"  SKIP {model_def['name']}: log not found")
            continue
        val_data, train_data = parse_log(log_path)
        if val_data:
            all_val[model_def["name"]] = val_data
            print(f"  {model_def['name']}: {len(val_data)} validation epochs, best={max(v[1] for v in val_data):.5f}")
        if train_data:
            all_train[model_def["name"]] = train_data

    if not all_val:
        print("No validation data found. Exiting.")
        return

    # Plot 1: Validation accuracy vs epoch
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_def in MODELS:
        name = model_def["name"]
        if name not in all_val:
            continue
        epochs = [v[0] for v in all_val[name]]
        accs = [v[1] * 100 for v in all_val[name]]
        ax.plot(epochs, accs, color=model_def["color"], linestyle=model_def["ls"],
                linewidth=2, label=name, alpha=0.9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Training Convergence: Validation Accuracy")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "training_curves_val_accuracy.pdf")
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {outpath}")

    # Plot 2: Training loss vs epoch
    if all_train:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for model_def in MODELS:
            name = model_def["name"]
            if name not in all_train:
                continue
            epochs = [t[0] for t in all_train[name]]
            losses = [t[1] for t in all_train[name]]
            ax.plot(epochs, losses, color=model_def["color"], linestyle=model_def["ls"],
                    linewidth=2, label=name, alpha=0.9)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Convergence: Loss")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        fig.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, "training_curves_loss.pdf")
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {outpath}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"{'Model':<20} {'Best Val Acc':<14} {'Final Epoch':<12}")
    print("-" * 60)
    for model_def in MODELS:
        name = model_def["name"]
        if name in all_val:
            best = max(v[1] for v in all_val[name])
            last_epoch = all_val[name][-1][0]
            print(f"{name:<20} {best*100:.2f}%        epoch {last_epoch}")
    print("=" * 60)


if __name__ == "__main__":
    main()
