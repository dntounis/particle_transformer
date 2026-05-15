#!/usr/bin/env python3
"""
Comprehensive analysis script for JetClass jet classification inference results.

Reads inference ROOT files produced by weaver-core's --predict mode, computes
classification metrics, and produces publication-quality comparison plots.

Requirements: uproot, numpy, scikit-learn, matplotlib

Usage:
    python produce_comparison_plots.py
"""

import os
import sys
import json
import warnings
from collections import OrderedDict

import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             accuracy_score, roc_auc_score)

# ==============================================================================
# Configuration
# ==============================================================================

# Base path (parent of analysis/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "Jim_results")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis", "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# JetClass 10 classes in canonical order
CLASS_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

# Model definitions: display_name -> filename
MODELS = OrderedDict([
    ("JetMamba-UniDir", "2026_jetmamba_v1.1_predict_ALL.root"),
    ("JetMamba-BiDir",  "2026_jetmamba_v1.2_predict_ALL.root"),
    ("JetMamba-Rel",    "2026_jetmamba_v1.3_predict_ALL.root"),
    ("JetMamba-Hybrid", "2026_jetmamba_v2.1_predict_ALL.root"),
    ("ParT",            "2026_ParT_predict_ALL.root"),
])

# Signal efficiency working points for rejection computation
SIGNAL_EFFICIENCIES = [0.30, 0.50, 0.70, 0.90]

# Signal vs background pairs for rejection curves
REJECTION_PAIRS = [
    ("Hbb", "QCD"),
    ("Hcc", "QCD"),
    ("H4q", "QCD"),
    ("Tbqq", "QCD"),
]

# Colorblind-friendly palette (from Paul Tol's scheme)
MODEL_COLORS = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#AA4499",  # purple
    "#882255",  # wine
]

MODEL_LINESTYLES = ['-', '--', '-.', ':', '-']
MODEL_MARKERS = ['o', 's', '^', 'D', 'v']


# ==============================================================================
# Plotting style
# ==============================================================================

def set_publication_style():
    """Set matplotlib parameters for publication-quality plots."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.7,
        'ytick.minor.width': 0.7,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.grid': False,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })


# ==============================================================================
# Data loading
# ==============================================================================

def load_model_data(filepath):
    """
    Load scores and labels from a ROOT file.

    Auto-discovers score and label branches by pattern matching.
    Returns: (scores_dict, labels_dict, class_order)
        - scores_dict: {class_name: np.array of scores}
        - labels_dict: {class_name: np.array of truth labels (0/1)}
        - class_order: list of class names in branch discovery order
    """
    with uproot.open(filepath) as f:
        # Find the Events tree (handle versioned keys like 'Events;5')
        tree_key = None
        for key in f.keys():
            if key.startswith('Events'):
                tree_key = key
                break
        if tree_key is None:
            raise ValueError(f"No 'Events' tree found in {filepath}")

        tree = f[tree_key]
        branches = tree.keys()

        # Auto-discover score branches: 'score_label_XXX' or 'score_XXX'
        score_branches = sorted([b for b in branches if b.startswith('score_')])
        # Auto-discover label branches: 'label_XXX' (but not '_label_')
        label_branches = sorted([b for b in branches
                                  if b.startswith('label_') and not b.startswith('_')])

        if not score_branches:
            raise ValueError(f"No score branches found in {filepath}")
        if not label_branches:
            raise ValueError(f"No label branches found in {filepath}")

        # Extract class names from branch names
        # Handle both 'score_label_XXX' and 'score_XXX' patterns
        def extract_class_from_score(branch_name):
            if branch_name.startswith('score_label_'):
                return branch_name[len('score_label_'):]
            elif branch_name.startswith('score_'):
                return branch_name[len('score_'):]
            return branch_name

        def extract_class_from_label(branch_name):
            if branch_name.startswith('label_'):
                return branch_name[len('label_'):]
            return branch_name

        score_classes = [extract_class_from_score(b) for b in score_branches]
        label_classes = [extract_class_from_label(b) for b in label_branches]

        # Read all data at once for efficiency
        all_branches_to_read = score_branches + label_branches
        data = tree.arrays(all_branches_to_read, library="np")

        scores_dict = {}
        for branch, cls in zip(score_branches, score_classes):
            scores_dict[cls] = data[branch]

        labels_dict = {}
        for branch, cls in zip(label_branches, label_classes):
            labels_dict[cls] = data[branch]

        # Use canonical class order if all classes present, otherwise branch order
        class_order = []
        for cls in CLASS_NAMES:
            if cls in scores_dict and cls in labels_dict:
                class_order.append(cls)
        # Add any extra classes not in canonical list
        for cls in score_classes:
            if cls not in class_order and cls in labels_dict:
                class_order.append(cls)

        return scores_dict, labels_dict, class_order


def load_all_models():
    """Load data from all configured models. Skip missing files with warning."""
    model_data = OrderedDict()

    for model_name, filename in MODELS.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.isfile(filepath):
            print(f"  WARNING: Skipping {model_name} - file not found: {filepath}")
            continue

        print(f"  Loading {model_name} from {filename}...")
        try:
            scores, labels, class_order = load_model_data(filepath)
            n_events = len(next(iter(labels.values())))
            print(f"    -> {n_events:,} events, {len(class_order)} classes: {class_order}")
            model_data[model_name] = {
                'scores': scores,
                'labels': labels,
                'class_order': class_order,
                'n_events': n_events,
            }
        except Exception as e:
            print(f"  WARNING: Error loading {model_name}: {e}")
            continue

    return model_data


# ==============================================================================
# Metric computation
# ==============================================================================

def compute_predictions(scores_dict, class_order):
    """Convert score dictionaries to a score matrix and predicted labels."""
    # Build score matrix: (n_events, n_classes)
    score_matrix = np.column_stack([scores_dict[cls] for cls in class_order])
    predicted = np.argmax(score_matrix, axis=1)
    return score_matrix, predicted


def compute_true_labels(labels_dict, class_order):
    """Convert one-hot label dictionaries to integer true labels."""
    label_matrix = np.column_stack([labels_dict[cls] for cls in class_order])
    true_labels = np.argmax(label_matrix, axis=1)
    return label_matrix, true_labels


def compute_overall_metrics(scores_dict, labels_dict, class_order):
    """Compute overall accuracy and macro-averaged AUC."""
    score_matrix, predicted = compute_predictions(scores_dict, class_order)
    label_matrix, true_labels = compute_true_labels(labels_dict, class_order)

    accuracy = accuracy_score(true_labels, predicted) * 100.0

    # Macro-averaged AUC (one-vs-rest), computed manually to handle missing classes
    per_class_aucs = []
    for i in range(len(class_order)):
        binary_true = (true_labels == i).astype(int)
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            continue  # Skip classes with no positive or no negative examples
        try:
            auc_val = roc_auc_score(binary_true, score_matrix[:, i])
            per_class_aucs.append(auc_val)
        except ValueError:
            continue

    macro_auc = float(np.mean(per_class_aucs)) if per_class_aucs else float('nan')

    return accuracy, macro_auc


def compute_per_class_metrics(scores_dict, labels_dict, class_order):
    """Compute per-class accuracy and per-class AUC (one-vs-rest)."""
    score_matrix, predicted = compute_predictions(scores_dict, class_order)
    label_matrix, true_labels = compute_true_labels(labels_dict, class_order)

    per_class_accuracy = {}
    per_class_auc = {}

    for i, cls in enumerate(class_order):
        # Per-class accuracy: fraction correctly classified among true class members
        mask = (true_labels == i)
        if mask.sum() > 0:
            per_class_accuracy[cls] = (predicted[mask] == i).mean() * 100.0
        else:
            per_class_accuracy[cls] = float('nan')

        # Per-class AUC (one-vs-rest)
        try:
            binary_true = (true_labels == i).astype(int)
            per_class_auc[cls] = roc_auc_score(binary_true, score_matrix[:, i])
        except ValueError:
            per_class_auc[cls] = float('nan')

    return per_class_accuracy, per_class_auc


def compute_rejection(scores_dict, labels_dict, signal_class, background_class,
                      signal_efficiencies):
    """
    Compute background rejection (1/FPR) at fixed signal efficiencies.

    Uses the score for the signal class as the discriminant.
    """
    if signal_class not in scores_dict or background_class not in labels_dict:
        return {f"{int(eff*100)}%": float('nan') for eff in signal_efficiencies}

    # Select events that are either signal or background (binary problem)
    sig_mask = labels_dict[signal_class].astype(bool)
    bkg_mask = labels_dict[background_class].astype(bool)
    combined_mask = sig_mask | bkg_mask

    if combined_mask.sum() == 0:
        return {f"{int(eff*100)}%": float('nan') for eff in signal_efficiencies}

    y_true = sig_mask[combined_mask].astype(int)
    y_score = scores_dict[signal_class][combined_mask]

    fpr, tpr, _ = roc_curve(y_true, y_score)

    rejections = {}
    for eff in signal_efficiencies:
        # Find the FPR at the working point closest to the desired TPR
        idx = np.searchsorted(tpr, eff)
        if idx >= len(fpr) or fpr[idx] == 0:
            rejections[f"{int(eff*100)}%"] = float('inf')
        else:
            rejections[f"{int(eff*100)}%"] = float(1.0 / fpr[idx])

    return rejections


def compute_roc_data(scores_dict, labels_dict, signal_class, background_class):
    """Compute ROC curve data (TPR, 1/FPR) for signal vs background."""
    sig_mask = labels_dict[signal_class].astype(bool)
    bkg_mask = labels_dict[background_class].astype(bool)
    combined_mask = sig_mask | bkg_mask

    if combined_mask.sum() == 0:
        return None, None, None

    y_true = sig_mask[combined_mask].astype(int)
    y_score = scores_dict[signal_class][combined_mask]

    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Compute AUC
    roc_auc = auc(fpr, tpr)

    # Compute 1/FPR (rejection), handle division by zero
    with np.errstate(divide='ignore'):
        rejection = np.where(fpr > 0, 1.0 / fpr, np.nan)

    return tpr, rejection, roc_auc


# ==============================================================================
# Plotting functions
# ==============================================================================

def plot_roc_curves(model_data, signal_class, background_class, output_path):
    """Plot background rejection vs signal efficiency for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (model_name, data) in enumerate(model_data.items()):
        tpr, rejection, roc_auc = compute_roc_data(
            data['scores'], data['labels'], signal_class, background_class
        )
        if tpr is None:
            continue

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ls = MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)]

        label = f"{model_name} (AUC = {roc_auc:.4f})"
        ax.plot(tpr, rejection, color=color, linestyle=ls, linewidth=1.8,
                label=label)

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel(f"Background rejection (1/$\\epsilon_{{bkg}}$)")
    ax.set_title(f"{signal_class} vs {background_class}")
    ax.set_yscale('log')
    ax.set_xlim(0.3, 1.0)

    # Set reasonable y-axis limits
    ax.set_ylim(bottom=1)

    ax.legend(loc='upper right', framealpha=0.9)
    ax.tick_params(which='both', direction='in', top=True, right=True)

    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_confusion_matrix(scores_dict, labels_dict, class_order, model_name,
                          output_path):
    """Plot normalized confusion matrix."""
    score_matrix, predicted = compute_predictions(scores_dict, class_order)
    _, true_labels = compute_true_labels(labels_dict, class_order)

    cm = confusion_matrix(true_labels, predicted, labels=range(len(class_order)))
    # Normalize by row (true class)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues',
                   vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Classification rate (%)")

    # Add text annotations
    thresh = 50.0
    for i in range(len(class_order)):
        for j in range(len(class_order)):
            val = cm_normalized[i, j]
            color = "white" if val > thresh else "black"
            text = f"{val:.1f}" if val >= 0.5 else ""
            ax.text(j, i, text, ha="center", va="center", color=color,
                    fontsize=9)

    ax.set_xticks(range(len(class_order)))
    ax.set_yticks(range(len(class_order)))
    ax.set_xticklabels(class_order, rotation=45, ha='right')
    ax.set_yticklabels(class_order)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion Matrix - {model_name}")

    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_overall_accuracy_comparison(model_data, all_results, output_path):
    """Bar chart comparing overall accuracy across models."""
    model_names = []
    accuracies = []

    for model_name in model_data.keys():
        if model_name in all_results:
            model_names.append(model_name)
            accuracies.append(all_results[model_name]['overall_accuracy'])

    if not model_names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(model_names))
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(model_names))]

    bars = ax.bar(x, accuracies, color=colors, edgecolor='black', linewidth=0.5,
                  width=0.6)

    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha='right')
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Model Comparison: Overall Classification Accuracy")
    ax.set_ylim(bottom=max(0, min(accuracies) - 5),
                top=min(100, max(accuracies) + 3))

    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_per_class_accuracy(model_data, all_results, output_path):
    """Grouped bar chart comparing per-class accuracy across models."""
    model_names = [m for m in model_data.keys() if m in all_results]
    if not model_names:
        return

    # Use the class order from the first model
    classes = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(14, 6))

    n_models = len(model_names)
    n_classes = len(classes)
    bar_width = 0.8 / n_models
    x = np.arange(n_classes)

    for i, model_name in enumerate(model_names):
        per_class = all_results[model_name].get('per_class_accuracy', {})
        accs = [per_class.get(cls, 0) for cls in classes]
        offset = (i - n_models / 2 + 0.5) * bar_width
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.bar(x + offset, accs, bar_width, label=model_name,
               color=color, edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_ylabel("Per-class Accuracy (%)")
    ax.set_title("Per-class Classification Accuracy")
    ax.legend(loc='lower right', ncol=2)

    # Set y-axis to start near lowest accuracy
    all_accs = []
    for model_name in model_names:
        per_class = all_results[model_name].get('per_class_accuracy', {})
        all_accs.extend(per_class.values())
    if all_accs:
        ymin = max(0, min(all_accs) - 10)
        ax.set_ylim(bottom=ymin, top=min(100, max(all_accs) + 5))

    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_macro_auc_comparison(model_data, all_results, output_path):
    """Bar chart comparing macro-averaged AUC across models."""
    model_names = []
    aucs = []

    for model_name in model_data.keys():
        if model_name in all_results:
            model_names.append(model_name)
            aucs.append(all_results[model_name]['macro_auc'])

    if not model_names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(model_names))
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(model_names))]

    bars = ax.bar(x, aucs, color=colors, edgecolor='black', linewidth=0.5,
                  width=0.6)

    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.0002,
                f'{a:.4f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha='right')
    ax.set_ylabel("Macro-averaged AUC")
    ax.set_title("Model Comparison: Macro-averaged AUC (One-vs-Rest)")
    ax.set_ylim(bottom=max(0.9, min(aucs) - 0.01),
                top=min(1.0, max(aucs) + 0.005))

    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"    Saved: {output_path}")


# ==============================================================================
# Summary printing
# ==============================================================================

def print_summary_table(all_results):
    """Print a formatted summary table to stdout."""
    if not all_results:
        print("No results to display.")
        return

    print("\n" + "=" * 100)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 100)

    # Overall metrics
    print(f"\n{'Model':<20s} {'Accuracy (%)':<14s} {'Macro AUC':<12s} "
          f"{'N events':<12s}")
    print("-" * 60)
    for model_name, results in all_results.items():
        acc = results['overall_accuracy']
        mauc = results['macro_auc']
        n = results.get('n_events', 'N/A')
        print(f"{model_name:<20s} {acc:<14.2f} {mauc:<12.4f} {str(n):<12s}")

    # Rejection at 50% efficiency for key channels
    print(f"\n{'Model':<20s} ", end="")
    for sig, bkg in REJECTION_PAIRS:
        print(f"{sig} vs {bkg} (50%){'':<4s}", end="")
    print()
    print("-" * (20 + len(REJECTION_PAIRS) * 22))
    for model_name, results in all_results.items():
        print(f"{model_name:<20s} ", end="")
        for sig, bkg in REJECTION_PAIRS:
            key = f"rejection_{sig}_vs_{bkg}"
            if key in results:
                rej = results[key].get('50%', float('nan'))
                if np.isinf(rej):
                    print(f"{'inf':<22s}", end="")
                else:
                    print(f"{rej:<22.0f}", end="")
            else:
                print(f"{'N/A':<22s}", end="")
        print()

    # Per-class accuracy
    print(f"\n{'Model':<20s}", end="")
    for cls in CLASS_NAMES:
        print(f" {cls:<8s}", end="")
    print()
    print("-" * (20 + len(CLASS_NAMES) * 9))
    for model_name, results in all_results.items():
        per_class = results.get('per_class_accuracy', {})
        print(f"{model_name:<20s}", end="")
        for cls in CLASS_NAMES:
            val = per_class.get(cls, float('nan'))
            print(f" {val:<8.1f}", end="")
        print()

    print("\n" + "=" * 100)


# ==============================================================================
# Scaling study plot
# ==============================================================================

def plot_scaling_study(output_path):
    """
    Plot per-jet latency vs sequence length from scaling_study.json.
    Produces a log-log plot showing O(N) vs O(N^2) scaling.
    """
    scaling_file = os.path.join(OUTPUT_DIR, "scaling_study.json")
    if not os.path.isfile(scaling_file):
        print(f"  SKIP: scaling_study.json not found (run benchmark_scaling.py first)")
        return

    with open(scaling_file, 'r') as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))

    model_order = ["jetmamba_unidir", "jetmamba_bidir", "jetmamba_rel", "jetmamba_hybrid", "ParT"]
    display_names = ["JetMamba-UniDir", "JetMamba-BiDir", "JetMamba-Rel", "JetMamba-Hybrid", "ParT"]

    for idx, (short_name, display_name) in enumerate(zip(model_order, display_names)):
        if short_name not in data["models"]:
            continue

        scaling = data["models"][short_name]["scaling"]
        Ns = []
        latencies = []
        for n_str, result in sorted(scaling.items(), key=lambda x: int(x[0])):
            if isinstance(result, dict) and "per_jet_latency_us" in result:
                Ns.append(int(n_str))
                latencies.append(result["per_jet_latency_us"])

        if len(Ns) < 2:
            continue

        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        linestyle = MODEL_LINESTYLES[idx % len(MODEL_LINESTYLES)]
        marker = MODEL_MARKERS[idx % len(MODEL_MARKERS)]

        ax.plot(Ns, latencies, color=color, linestyle=linestyle, marker=marker,
                markersize=6, linewidth=2, label=display_name)

    # Add reference slopes
    ns_ref = np.array([32, 1024])
    # O(N) reference
    y0_linear = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1.0
    # Plot after data so we can scale appropriately
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # Reference lines (manual placement after auto-scaling)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ref_ns = np.array([64, 512])
    ref_linear = ylims[0] * 2.0 * (ref_ns / ref_ns[0])
    ref_quad = ylims[0] * 2.0 * (ref_ns / ref_ns[0])**2
    ax.plot(ref_ns, ref_linear, 'k--', alpha=0.3, linewidth=1.5, label=r'$\mathcal{O}(N)$')
    ax.plot(ref_ns, ref_quad, 'k:', alpha=0.3, linewidth=1.5, label=r'$\mathcal{O}(N^2)$')

    ax.set_xlabel("Sequence length $N$ (particles per jet)")
    ax.set_ylabel("Per-jet latency ($\\mu$s)")
    ax.set_title("Inference Latency Scaling")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks([32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(['32', '64', '128', '256', '512', '1024'])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ==============================================================================
# Efficiency bubble plot (FLOPs vs accuracy, with throughput as marker size)
# ==============================================================================

def plot_efficiency_bubble(all_results, output_path):
    """
    Plot FLOPs vs accuracy scatter, with inference throughput as marker size.
    Reads model_summary.json for FLOPs, inference_speed.json for throughput.
    """
    summary_file = os.path.join(OUTPUT_DIR, "model_summary.json")
    speed_file = os.path.join(OUTPUT_DIR, "inference_speed.json")

    if not os.path.isfile(summary_file):
        print(f"  SKIP: model_summary.json not found for bubble plot")
        return

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    speed_data = None
    if os.path.isfile(speed_file):
        with open(speed_file, 'r') as f:
            speed_data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))

    model_keys = [
        ("JetMamba-UniDir", "jetmamba_unidir"),
        ("JetMamba-BiDir", "jetmamba_bidir"),
        ("JetMamba-Rel", "jetmamba_rel"),
        ("JetMamba-Hybrid", "jetmamba_hybrid"),
        ("ParT", "ParT"),
    ]

    for idx, (display_name, short_key) in enumerate(model_keys):
        if display_name not in all_results:
            continue
        if short_key not in summary:
            continue

        acc = all_results[display_name]["overall_accuracy"]
        flops_val = float(summary[short_key].get("flops", 0))

        flops_M = flops_val / 1e6

        # Marker size: proportional to throughput (or fixed if no speed data)
        size = 200
        if speed_data and short_key in speed_data.get("models", {}):
            throughput = speed_data["models"][short_key].get("throughput_jets_per_sec", None)
            if throughput:
                size = max(50, min(800, throughput / 500))

        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        marker = MODEL_MARKERS[idx % len(MODEL_MARKERS)]

        ax.scatter(flops_M, acc, s=size, c=color, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=5, label=display_name)

    ax.set_xlabel("FLOPs (millions)")
    ax.set_xscale('log')
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Accuracy vs Computational Cost")
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {output_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("JetClass Classification Analysis")
    print("=" * 70)

    # Create output directories
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")

    # Set plotting style
    set_publication_style()

    # ---- Load data ----
    print("\n[1/5] Loading model predictions...")
    model_data = load_all_models()

    if not model_data:
        print("\nERROR: No models could be loaded. Exiting.")
        sys.exit(1)

    print(f"\n  Successfully loaded {len(model_data)} model(s): "
          f"{list(model_data.keys())}")

    # ---- Compute metrics ----
    print("\n[2/5] Computing classification metrics...")
    all_results = OrderedDict()

    for model_name, data in model_data.items():
        print(f"  {model_name}...")
        scores = data['scores']
        labels = data['labels']
        class_order = data['class_order']

        # Overall metrics
        accuracy, macro_auc = compute_overall_metrics(scores, labels, class_order)

        # Per-class metrics
        per_class_accuracy, per_class_auc = compute_per_class_metrics(
            scores, labels, class_order
        )

        # Rejection for each signal/background pair
        rejection_results = {}
        for sig, bkg in REJECTION_PAIRS:
            if sig in scores and bkg in labels:
                rej = compute_rejection(scores, labels, sig, bkg,
                                        SIGNAL_EFFICIENCIES)
                rejection_results[f"rejection_{sig}_vs_{bkg}"] = rej

        all_results[model_name] = {
            'overall_accuracy': round(accuracy, 2),
            'macro_auc': round(macro_auc, 6),
            'per_class_accuracy': {k: round(v, 2) for k, v in per_class_accuracy.items()},
            'per_class_auc': {k: round(v, 6) for k, v in per_class_auc.items()},
            'n_events': data['n_events'],
        }
        all_results[model_name].update(rejection_results)

        print(f"    Accuracy: {accuracy:.2f}%, Macro AUC: {macro_auc:.4f}")

    # ---- Generate plots ----
    print("\n[3/5] Generating ROC curves...")
    for sig, bkg in REJECTION_PAIRS:
        output_path = os.path.join(FIGURES_DIR, f"roc_{sig}_vs_{bkg}.pdf")
        plot_roc_curves(model_data, sig, bkg, output_path)

    print("\n[4/5] Generating confusion matrices...")
    # Plot confusion matrix for best JetMamba and ParT
    jetmamba_models = [m for m in model_data.keys() if 'JetMamba' in m]
    if jetmamba_models:
        # Find best JetMamba by accuracy
        best_jm = max(jetmamba_models,
                      key=lambda m: all_results[m]['overall_accuracy'])
        data = model_data[best_jm]
        output_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{best_jm.replace(' ', '_')}.pdf")
        plot_confusion_matrix(data['scores'], data['labels'],
                              data['class_order'], best_jm, output_path)

    if "ParT" in model_data:
        data = model_data["ParT"]
        output_path = os.path.join(FIGURES_DIR, "confusion_matrix_ParT.pdf")
        plot_confusion_matrix(data['scores'], data['labels'],
                              data['class_order'], "ParT", output_path)

    print("\n[5/5] Generating comparison plots...")
    # Overall accuracy bar chart
    plot_overall_accuracy_comparison(
        model_data, all_results,
        os.path.join(FIGURES_DIR, "overall_accuracy_comparison.pdf")
    )

    # Per-class accuracy grouped bar chart
    plot_per_class_accuracy(
        model_data, all_results,
        os.path.join(FIGURES_DIR, "per_class_accuracy_comparison.pdf")
    )

    # Macro AUC comparison
    plot_macro_auc_comparison(
        model_data, all_results,
        os.path.join(FIGURES_DIR, "macro_auc_comparison.pdf")
    )

    # Scaling study (if available)
    plot_scaling_study(os.path.join(FIGURES_DIR, "scaling_latency.pdf"))

    # Efficiency bubble plot (FLOPs vs accuracy)
    plot_efficiency_bubble(
        all_results,
        os.path.join(FIGURES_DIR, "efficiency_bubble.pdf")
    )

    # ---- Save results to JSON ----
    json_path = os.path.join(OUTPUT_DIR, "classification_results.json")

    # Convert any inf/nan to string for JSON serialization
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, float):
            if np.isinf(obj):
                return "inf"
            elif np.isnan(obj):
                return "nan"
            return obj
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    json_results = sanitize_for_json(dict(all_results))
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {json_path}")

    # ---- Print summary ----
    print_summary_table(all_results)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
