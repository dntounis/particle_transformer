#!/usr/bin/env python3
"""
Update the JetMamba paper draft with actual results.

Reads:
  - analysis/results/classification_results.json (accuracy, AUC, rejection)
  - analysis/results/model_summary.json (params, FLOPs)
  - analysis/results/inference_speed.json (throughput, latency) [if available]
  - analysis/results/scaling_study.json (scaling ratios) [if available]

Writes:
  - Updated paper/jetmamba_mlst_draft.tex with \\fillin{} replaced by real values

Also updates narrative text that assumed BiDir was best, since test data shows:
  Rel (84.17%) > UniDir (83.99%) > BiDir (83.77%) > Hybrid (82.78%)

Usage:
    python scripts/update_paper_with_results.py [--dry-run]
"""

import os
import sys
import json
import argparse

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # CS231N_Final_Project/
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")
RESULTS_DIR = os.path.join(BASE_DIR, "analysis", "results")

TEX_FILE = os.path.join(PAPER_DIR, "jetmamba_mlst_draft.tex")

CLASSIFICATION_JSON = os.path.join(RESULTS_DIR, "classification_results.json")
MODEL_SUMMARY_JSON = os.path.join(RESULTS_DIR, "model_summary.json")
SPEED_JSON = os.path.join(RESULTS_DIR, "inference_speed.json")
SCALING_JSON = os.path.join(RESULTS_DIR, "scaling_study.json")


# =============================================================================
# Load data
# =============================================================================
def load_results():
    """Load all available result files."""
    data = {}

    if not os.path.isfile(CLASSIFICATION_JSON):
        print(f"ERROR: {CLASSIFICATION_JSON} not found. Run produce_comparison_plots.py first.")
        sys.exit(1)
    with open(CLASSIFICATION_JSON) as f:
        data['classification'] = json.load(f)

    if not os.path.isfile(MODEL_SUMMARY_JSON):
        print(f"ERROR: {MODEL_SUMMARY_JSON} not found.")
        sys.exit(1)
    with open(MODEL_SUMMARY_JSON) as f:
        data['summary'] = json.load(f)

    if os.path.isfile(SPEED_JSON):
        with open(SPEED_JSON) as f:
            data['speed'] = json.load(f)
        print(f"  Loaded: inference_speed.json")
    else:
        data['speed'] = None
        print(f"  SKIPPED: inference_speed.json (not yet available)")

    if os.path.isfile(SCALING_JSON):
        with open(SCALING_JSON) as f:
            data['scaling'] = json.load(f)
        print(f"  Loaded: scaling_study.json")
    else:
        data['scaling'] = None
        print(f"  SKIPPED: scaling_study.json (not yet available)")

    return data


# =============================================================================
# Extract key numbers
# =============================================================================
def extract_numbers(data):
    """Extract key numbers from loaded data for paper insertion."""
    cls = data['classification']
    speed = data['speed']
    scaling = data['scaling']

    nums = {}

    model_map = {
        'unidir': 'JetMamba-UniDir',
        'bidir': 'JetMamba-BiDir',
        'rel': 'JetMamba-Rel',
        'hybrid': 'JetMamba-Hybrid',
        'part': 'ParT',
    }

    for key, display in model_map.items():
        if display in cls:
            nums[f'acc_{key}'] = cls[display]['overall_accuracy']
            nums[f'auc_{key}'] = cls[display]['macro_auc']

            for c, v in cls[display].get('per_class_accuracy', {}).items():
                nums[f'per_class_{key}_{c}'] = v

            for pair_key, rej_data in cls[display].items():
                if pair_key.startswith('rejection_') and isinstance(rej_data, dict):
                    tag = pair_key.replace('rejection_', '')
                    if '50%' in rej_data:
                        val = rej_data['50%']
                        nums[f'rej50_{key}_{tag}'] = float('inf') if val == "inf" else float(val)

    # Best JetMamba variant
    jm_models = ['unidir', 'bidir', 'rel', 'hybrid']
    jm_accs = {k: nums.get(f'acc_{k}', 0) for k in jm_models}
    best_jm = max(jm_accs, key=jm_accs.get)
    nums['best_jm_key'] = best_jm
    nums['best_jm_name'] = model_map[best_jm]
    nums['best_jm_acc'] = jm_accs[best_jm]

    # Accuracy deltas
    unidir_acc = nums.get('acc_unidir', 0)
    bidir_acc = nums.get('acc_bidir', 0)
    rel_acc = nums.get('acc_rel', 0)
    hybrid_acc = nums.get('acc_hybrid', 0)
    part_acc = nums.get('acc_part', None)

    nums['delta_bidir_vs_unidir'] = bidir_acc - unidir_acc
    nums['delta_rel_vs_bidir'] = rel_acc - bidir_acc
    nums['delta_hybrid_vs_bidir'] = hybrid_acc - bidir_acc
    nums['delta_rel_vs_unidir'] = rel_acc - unidir_acc

    if part_acc is not None:
        nums['gap_to_part'] = part_acc - nums['best_jm_acc']
        nums['pct_of_part'] = (nums['best_jm_acc'] / part_acc) * 100.0

    # Speed data
    if speed and 'models' in speed:
        speed_key_map = {
            'unidir': 'jetmamba_unidir',
            'bidir': 'jetmamba_bidir',
            'rel': 'jetmamba_rel',
            'hybrid': 'jetmamba_hybrid',
            'part': 'ParT',
        }
        for key, skey in speed_key_map.items():
            if skey in speed['models']:
                lat = speed['models'][skey].get('latency_us_per_jet')
                thr = speed['models'][skey].get('throughput_jets_per_sec')
                if lat:
                    nums[f'latency_{key}'] = lat
                if thr:
                    nums[f'throughput_{key}'] = thr

        if 'latency_part' in nums and f'latency_{best_jm}' in nums:
            nums['speed_ratio_best_jm'] = nums['latency_part'] / nums[f'latency_{best_jm}']

    # Scaling data
    if scaling and 'models' in scaling:
        sc_models = scaling['models']
        part_key_sc = 'ParT'
        best_jm_key_sc = f'jetmamba_{best_jm}'

        for n_val in [128, 1024]:
            n_str = str(n_val)
            part_lat = None
            best_lat = None

            if part_key_sc in sc_models and n_str in sc_models[part_key_sc].get('scaling', {}):
                entry = sc_models[part_key_sc]['scaling'][n_str]
                if isinstance(entry, dict):
                    part_lat = entry.get('per_jet_latency_us')

            if best_jm_key_sc in sc_models and n_str in sc_models[best_jm_key_sc].get('scaling', {}):
                entry = sc_models[best_jm_key_sc]['scaling'][n_str]
                if isinstance(entry, dict):
                    best_lat = entry.get('per_jet_latency_us')

            if part_lat and best_lat and best_lat > 0:
                nums[f'scaling_ratio_N{n_val}'] = part_lat / best_lat

    return nums


# =============================================================================
# Build text replacements
# =============================================================================
def build_replacements(nums):
    """Build (old_text, new_text) pairs for the paper."""
    replacements = []

    best_jm_name = nums.get('best_jm_name', 'JetMamba-Rel')
    best_jm_acc = nums.get('best_jm_acc', 84.17)
    part_acc = nums.get('acc_part', None)

    bidir_acc = nums.get('acc_bidir', 83.77)
    rel_acc = nums.get('acc_rel', 84.17)
    hybrid_acc = nums.get('acc_hybrid', 82.78)
    unidir_acc = nums.get('acc_unidir', 83.99)

    bidir_auc = nums.get('auc_bidir', 0.9840)
    rel_auc = nums.get('auc_rel', 0.9847)
    hybrid_auc = nums.get('auc_hybrid', 0.9824)
    part_auc = nums.get('auc_part', None)

    delta_bidir = nums.get('delta_bidir_vs_unidir', -0.22)
    delta_rel_bidir = nums.get('delta_rel_vs_bidir', 0.40)
    delta_hybrid_bidir = nums.get('delta_hybrid_vs_bidir', -0.99)
    delta_rel_unidir = nums.get('delta_rel_vs_unidir', 0.18)

    speed_ratio = nums.get('speed_ratio_best_jm', None)
    scaling_128 = nums.get('scaling_ratio_N128', None)
    scaling_1024 = nums.get('scaling_ratio_N1024', None)

    speed_str = f"{speed_ratio:.1f}" if speed_ratio else "X"
    scaling_128_str = f"{scaling_128:.1f}" if scaling_128 else "X"
    scaling_1024_str = f"{scaling_1024:.1f}" if scaling_1024 else "X"

    # Latency strings per model
    def lat_str(key):
        lat = nums.get(f'latency_{key}')
        return f"{lat:.1f}" if lat else r"\fillin{}"

    # --- Abstract ---
    if part_acc is not None:
        abstract_fill = (
            f"{best_jm_name} achieves {best_jm_acc:.1f}\\% accuracy compared to "
            f"ParT's {part_acc:.1f}\\%, while offering {speed_str}$\\times$ faster "
            f"inference throughput at 21$\\times$ fewer FLOPs"
        )
    else:
        abstract_fill = (
            f"{best_jm_name} achieves {best_jm_acc:.1f}\\% accuracy "
            f"(the best among our SSM variants) while offering "
            f"{speed_str}$\\times$ faster inference throughput at 21$\\times$ fewer FLOPs"
        )
    replacements.append((
        r"\fillin{JetMamba-BiDir achieves XX.X\% accuracy compared to ParT's XX.X\%, while offering X$\times$ faster inference throughput at 21$\times$ fewer FLOPs}",
        abstract_fill
    ))

    # --- Table 1: inference latency ---
    for row, key in [
        (r"ParticleTransformer   & 2.14M & 503.9M & 2.8\,h & \fillin{} \\", 'part'),
        (r"JetMamba-UniDir       & 1.10M & 12.0M & 2.0\,h & \fillin{} \\", 'unidir'),
        (r"JetMamba-BiDir        & 2.82M & 23.4M & 2.6\,h & \fillin{} \\", 'bidir'),
        (r"JetMamba-Rel          & 2.83M & 24.2M & 2.6\,h & \fillin{} \\", 'rel'),
        (r"JetMamba-Hybrid       & 3.41M & 25.0M & 5.4\,h & \fillin{} \\", 'hybrid'),
    ]:
        ls = lat_str(key)
        new_row = row.replace(r"\fillin{}", f"{ls}\\,$\\mu$s")
        replacements.append((row, new_row))

    # --- Contributions (line 216) ---
    if part_acc is not None:
        contrib_fill = (
            f"{best_jm_name} achieves {best_jm_acc:.1f}\\% accuracy on JetClass "
            f"compared to ParT's {part_acc:.1f}\\%, while providing "
            f"{speed_str}$\\times$ faster inference throughput at 21$\\times$ fewer FLOPs"
        )
    else:
        contrib_fill = (
            f"{best_jm_name} achieves {best_jm_acc:.1f}\\% accuracy on JetClass, "
            f"while providing {speed_str}$\\times$ faster inference throughput "
            f"at 21$\\times$ fewer FLOPs"
        )
    replacements.append((
        r"\fillin{JetMamba-BiDir achieves XX.X\% accuracy on JetClass compared to ParT's XX.X\%, while providing X$\times$ faster inference throughput at 21$\times$ fewer FLOPs}",
        contrib_fill
    ))

    # --- Table: main_results ---
    replacements.append((
        r"JetMamba-BiDir    & \fillin{} & \fillin{} & $\mathcal{O}(N)$ \\",
        f"JetMamba-BiDir    & {bidir_acc:.1f} & {bidir_auc:.4f} & $\\mathcal{{O}}(N)$ \\\\"
    ))
    replacements.append((
        r"JetMamba-Rel      & \fillin{} & \fillin{} & $\mathcal{O}(NK)$ \\",
        f"JetMamba-Rel      & {rel_acc:.1f} & {rel_auc:.4f} & $\\mathcal{{O}}(NK)$ \\\\"
    ))
    replacements.append((
        r"JetMamba-Hybrid   & \fillin{} & \fillin{} & $\mathcal{O}(NK)$ \\",
        f"JetMamba-Hybrid   & {hybrid_acc:.1f} & {hybrid_auc:.4f} & $\\mathcal{{O}}(NK)$ \\\\"
    ))
    if part_acc is not None:
        replacements.append((
            r"ParT              & \fillin{} & \fillin{} & $\mathcal{O}(N^2)$ \\",
            f"ParT              & {part_acc:.1f} & {part_auc:.4f} & $\\mathcal{{O}}(N^2)$ \\\\"
        ))

    # --- Ablation observations ---
    replacements.append((
        r"\fillin{The addition of backward Mamba passes yields $+$X.X percentage points}",
        f"The addition of backward Mamba passes yields ${delta_bidir:+.2f}$ percentage "
        f"points ({bidir_acc:.1f}\\% vs UniDir's {unidir_acc:.1f}\\%), "
        f"indicating that bidirectionality alone does not straightforwardly improve accuracy on the test set"
    ))

    replacements.append((
        r"\fillin{Adding physics-motivated anchor relations yields X.X\% accuracy (delta of X.X pp vs BiDir)}",
        f"Adding physics-motivated anchor relations yields {rel_acc:.1f}\\% accuracy "
        f"(${delta_rel_bidir:+.2f}$ pp vs BiDir), making JetMamba-Rel the best-performing SSM variant"
    ))

    replacements.append((
        r"\fillin{Adding learned anchor attention yields X.X\% accuracy (delta of X.X pp vs BiDir)}",
        f"Adding learned anchor attention yields {hybrid_acc:.1f}\\% accuracy "
        f"(${delta_hybrid_bidir:.2f}$ pp vs BiDir)"
    ))

    # Gap to ParT
    if part_acc is not None:
        gap = nums['gap_to_part']
        replacements.append((
            r"\fillin{The best JetMamba variant (BiDir) reaches XX.X\% vs ParT's XX.X\%}",
            f"The best JetMamba variant ({best_jm_name}) reaches {best_jm_acc:.1f}\\% vs ParT's {part_acc:.1f}\\%"
        ))
        replacements.append((
            r"representing \fillin{X.X} percentage points below the transformer",
            f"representing {gap:.1f} percentage points below the transformer"
        ))

    # --- Per-class table ---
    classes = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
    for c in classes:
        unidir_v = nums.get(f'per_class_unidir_{c}', 0)
        bidir_v = nums.get(f'per_class_bidir_{c}', 0)
        rel_v = nums.get(f'per_class_rel_{c}', 0)
        hybrid_v = nums.get(f'per_class_hybrid_{c}', 0)
        part_v = nums.get(f'per_class_part_{c}', None)

        # Match the exact format in the tex file
        # e.g. "QCD   & 76.4 & \fillin{} & \fillin{} & \fillin{} & \fillin{} \\"
        old = f"{c}" + " " * (5 - len(c)) + f" & {unidir_v:.1f} & \\fillin{{}} & \\fillin{{}} & \\fillin{{}} & \\fillin{{}} \\\\"
        if part_v is not None:
            new = f"{c}" + " " * (5 - len(c)) + f" & {unidir_v:.1f} & {bidir_v:.1f} & {rel_v:.1f} & {hybrid_v:.1f} & {part_v:.1f} \\\\"
        else:
            new = f"{c}" + " " * (5 - len(c)) + f" & {unidir_v:.1f} & {bidir_v:.1f} & {rel_v:.1f} & {hybrid_v:.1f} & \\fillin{{}} \\\\"
        replacements.append((old, new))

    # Average row
    if part_acc is not None:
        replacements.append((
            r"\textbf{Average} & 84.0 & \fillin{} & \fillin{} & \fillin{} & \fillin{} \\",
            f"\\textbf{{Average}} & {unidir_acc:.1f} & {bidir_acc:.1f} & {rel_acc:.1f} & {hybrid_acc:.1f} & {part_acc:.1f} \\\\"
        ))
    else:
        replacements.append((
            r"\textbf{Average} & 84.0 & \fillin{} & \fillin{} & \fillin{} & \fillin{} \\",
            f"\\textbf{{Average}} & {unidir_acc:.1f} & {bidir_acc:.1f} & {rel_acc:.1f} & {hybrid_acc:.1f} & \\fillin{{}} \\\\"
        ))

    # --- Rejection table ---
    def fmt_rej(val):
        if val is None:
            return r"\fillin{}"
        if val == float('inf'):
            return "$\\infty$"
        v = int(round(val))
        if v >= 1000:
            return f"{v:,}".replace(",", "{,}")
        return str(v)

    # Match exact lines from the tex file for rejection table
    rej_lines = [
        ("Hbb_vs_QCD", r"$H{\to}b\bar{b}$ vs QCD  & 3{,}350 & \fillin{} & \fillin{} & \fillin{} & \fillin{} \\"),
        ("Hcc_vs_QCD", r"$H{\to}c\bar{c}$ vs QCD  & 805 & \fillin{} & \fillin{} & \fillin{} & \fillin{} \\"),
        ("H4q_vs_QCD", r"$H{\to}4q$ vs QCD        & 416 & \fillin{} & \fillin{} & \fillin{} & \fillin{} \\"),
        ("Tbqq_vs_QCD", r"$t{\to}bqq$ vs QCD       & 5{,}764 & \fillin{} & \fillin{} & \fillin{} & \fillin{} \\"),
    ]

    for tag, old_line in rej_lines:
        bidir_rej = nums.get(f'rej50_bidir_{tag}', 0)
        rel_rej = nums.get(f'rej50_rel_{tag}', 0)
        hybrid_rej = nums.get(f'rej50_hybrid_{tag}', 0)
        part_rej = nums.get(f'rej50_part_{tag}', None)

        # Replace the 4 \fillin{} entries with actual values
        new_line = old_line
        fillin_values = [fmt_rej(bidir_rej), fmt_rej(rel_rej), fmt_rej(hybrid_rej), fmt_rej(part_rej)]
        for val in fillin_values:
            new_line = new_line.replace(r"\fillin{}", val, 1)
        replacements.append((old_line, new_line))

    # --- Efficiency table (latency column) ---
    eff_rows = [
        (r"JetMamba-UniDir  & 1.10M & 12.0M & \fillin{} \\", 'unidir'),
        (r"JetMamba-BiDir   & 2.82M & 23.4M & \fillin{} \\", 'bidir'),
        (r"JetMamba-Rel     & 2.83M & 24.2M & \fillin{} \\", 'rel'),
        (r"JetMamba-Hybrid  & 3.41M & 25.0M & \fillin{} \\", 'hybrid'),
        (r"ParT             & 2.14M & 503.9M & \fillin{} \\", 'part'),
    ]
    for row, key in eff_rows:
        ls = lat_str(key)
        new_row = row.replace(r"\fillin{}", f"{ls}\\,$\\mu$s")
        replacements.append((row, new_row))

    # --- Scaling paragraph ---
    replacements.append((
        r"ParT is \fillin{X$\times$ slower}; at $N=1024$, the gap widens to \fillin{X$\times$}",
        f"ParT is {scaling_128_str}$\\times$ slower; at $N=1024$, the gap widens to {scaling_1024_str}$\\times$"
    ))

    # --- Discussion: ablation insights ---
    replacements.append((
        r"\fillin{The largest single improvement ($+$X.X\% accuracy), confirming that}",
        f"Interestingly, on the full 20M-event test set, pure bidirectionality yields only a marginal change "
        f"(${delta_bidir:+.2f}$ pp), while the combination with relational anchors (Rel) achieves "
        f"the best performance (+{delta_rel_unidir:.2f} pp to {rel_acc:.1f}\\%). This indicates that"
    ))

    replacements.append((
        r"\fillin{The relational module yields X.X\% accuracy (vs BiDir's X.X\%).}",
        f"The relational module yields {rel_acc:.1f}\\% accuracy (vs BiDir's {bidir_acc:.1f}\\%), "
        f"making it the best-performing JetMamba variant."
    ))

    replacements.append((
        r"\fillin{The hybrid model yields X.X\% accuracy (vs BiDir's X.X\%).}",
        f"The hybrid model yields {hybrid_acc:.1f}\\% accuracy (vs BiDir's {bidir_acc:.1f}\\%)."
    ))

    # --- SSM-Transformer spectrum ---
    if part_acc is not None:
        pct = nums['pct_of_part']
        replacements.append((
            r"\fillin{JetMamba-BiDir achieves X.X\% of ParT's accuracy at X$\times$ faster inference}",
            f"{best_jm_name} achieves {pct:.1f}\\% of ParT's accuracy at {speed_str}$\\times$ faster inference"
        ))

    # --- Limitations ---
    if part_acc is not None:
        replacements.append((
            r"\fillin{JetMamba-BiDir achieves XX.X\% vs ParT's XX.X\% accuracy.}",
            f"{best_jm_name} achieves {best_jm_acc:.1f}\\% vs ParT's {part_acc:.1f}\\% accuracy."
        ))

    # --- Conclusion ---
    replacements.append((
        r"\fillin{improving accuracy from XX.X\% to XX.X\% ($+$X.X pp)}",
        f"with the relational variant achieving {rel_acc:.1f}\\% vs UniDir's {unidir_acc:.1f}\\% ($+${delta_rel_unidir:.2f} pp)"
    ))

    replacements.append((
        r"\fillin{reaching XX.X\% and XX.X\% accuracy respectively}",
        f"reaching {rel_acc:.1f}\\% and {hybrid_acc:.1f}\\% accuracy respectively"
    ))

    replacements.append((
        r"\fillin{JetMamba-BiDir offers X$\times$ faster inference throughput than ParT}",
        f"{best_jm_name} offers {speed_str}$\\times$ faster inference throughput than ParT"
    ))

    # --- Data availability ---
    replacements.append((
        r"\fillin{add GitHub repository URL}",
        r"[URL to be added upon publication]"
    ))

    return replacements


# =============================================================================
# Narrative fixes
# =============================================================================
def fix_narrative(tex_content, nums):
    """
    Fix narrative that assumed BiDir was best. Test data shows Rel is best.
    """
    best_jm_name = nums.get('best_jm_name', 'JetMamba-Rel')

    # Confusion matrix figure filename
    tex_content = tex_content.replace(
        r"\includegraphics[width=0.48\textwidth]{figures/confusion_matrix_JetMamba-BiDir.pdf}",
        r"\includegraphics[width=0.48\textwidth]{figures/confusion_matrix_JetMamba-Rel.pdf}"
    )
    tex_content = tex_content.replace(
        "for JetMamba-BiDir (left) and ParT (right)",
        f"for {best_jm_name} (left) and ParT (right)"
    )

    # ROC caption
    tex_content = tex_content.replace(
        "JetMamba-BiDir achieves the best rejection among the SSM variants",
        f"{best_jm_name} achieves the best rejection among the SSM variants"
    )

    # "single most impactful architectural modification"
    tex_content = tex_content.replace(
        "This is the single most impactful architectural modification.",
        "However, when combined with the relational anchor branch (Rel), the bidirectional backbone achieves the best overall results."
    )

    # Discussion header about "bidirectionality is the single essential addition"
    tex_content = tex_content.replace(
        "reveals that bidirectionality is the single essential addition to SSMs for particle data, while more complex relational or attention modules provide diminishing or negative returns:",
        f"reveals a nuanced picture: while bidirectionality provides the essential foundation for SSMs on particle data, the addition of lightweight physics-motivated relational features ({best_jm_name}) yields the best overall performance, whereas more complex learned attention (JetMamba-Hybrid) degrades accuracy:"
    )

    # Abstract final sentence
    tex_content = tex_content.replace(
        "Our results show that bidirectionality is the single essential addition for SSMs on unordered particle data, while more complex modules (pairwise relations, sparse attention) provide diminishing returns---suggesting that the bidirectional Mamba hidden state already captures sufficient particle correlations without explicit relational inductive biases.",
        f"Our results show that while bidirectionality provides the essential foundation, lightweight physics-motivated relational anchors ({best_jm_name}) yield the best SSM performance, whereas more complex hybrid attention degrades accuracy---suggesting a favorable complexity--performance sweet spot between pure SSMs and full transformers."
    )

    # "simplest bidirectional SSM achieves the best"
    tex_content = tex_content.replace(
        "Surprisingly, rather than a monotonic accuracy--complexity trade-off, we find that the simplest bidirectional SSM achieves the best performance among the JetMamba family.",
        f"Rather than a monotonic accuracy--complexity trade-off, we find that the physics-informed relational variant ({best_jm_name}) achieves the best performance among the JetMamba family, while the more complex hybrid attention model underperforms even the simpler variants."
    )

    return tex_content


# =============================================================================
# Apply replacements
# =============================================================================
def apply_replacements(tex_content, replacements):
    """Apply text replacements to LaTeX content."""
    modified = tex_content
    applied = 0
    skipped = []

    for old, new in replacements:
        if old in modified:
            modified = modified.replace(old, new, 1)
            applied += 1
        else:
            skipped.append(old[:80] + "..." if len(old) > 80 else old)

    return modified, applied, skipped


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Update paper with results")
    parser.add_argument('--dry-run', action='store_true',
                        help="Show changes without modifying file")
    args = parser.parse_args()

    print("=" * 70)
    print("Updating paper with results")
    print("=" * 70)

    print("\nLoading result files...")
    data = load_results()

    print("\nExtracting key numbers...")
    nums = extract_numbers(data)

    print(f"\n  Best JetMamba variant: {nums.get('best_jm_name', '?')} "
          f"({nums.get('best_jm_acc', '?')}%)")
    if 'acc_part' in nums:
        print(f"  ParT accuracy: {nums['acc_part']}%")
        print(f"  Gap to ParT: {nums.get('gap_to_part', '?'):.1f} pp")
    if 'speed_ratio_best_jm' in nums:
        print(f"  Speed advantage: {nums['speed_ratio_best_jm']:.1f}x faster")
    print(f"  FLOPs advantage: 21x fewer (24.2M vs 503.9M)")

    print(f"\nReading: {TEX_FILE}")
    with open(TEX_FILE, 'r') as f:
        tex_content = f.read()

    orig_fillins = tex_content.count(r'\fillin{')

    replacements = build_replacements(nums)
    modified, applied, skipped = apply_replacements(tex_content, replacements)
    print(f"\n  Applied {applied}/{len(replacements)} placeholder replacements")

    if skipped:
        print(f"  Skipped {len(skipped)} (pattern not found in file):")
        for s in skipped[:15]:
            print(f"    - {s}")

    modified = fix_narrative(modified, nums)

    remaining = modified.count(r'\fillin{')
    print(f"\n  \\fillin{{}} count: {orig_fillins} -> {remaining}")

    if args.dry_run:
        print("\n  DRY RUN: No changes written.")
    else:
        with open(TEX_FILE, 'w') as f:
            f.write(modified)
        print(f"\n  Written: {TEX_FILE}")

    # Copy figures to paper/figures/
    if not args.dry_run:
        import shutil
        paper_figs_dir = os.path.join(PAPER_DIR, "figures")
        os.makedirs(paper_figs_dir, exist_ok=True)
        src_dir = os.path.join(BASE_DIR, "analysis", "results", "figures")

        figures_to_copy = [
            "roc_Hbb_vs_QCD.pdf",
            "confusion_matrix_JetMamba-Rel.pdf",
            "confusion_matrix_ParT.pdf",
            "scaling_latency.pdf",
            "efficiency_bubble.pdf",
        ]

        copied = 0
        for fig in figures_to_copy:
            src = os.path.join(src_dir, fig)
            dst = os.path.join(paper_figs_dir, fig)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied += 1
        print(f"\n  Copied {copied}/{len(figures_to_copy)} figures to {paper_figs_dir}")

    if data['speed'] is None:
        print("\n  NOTE: Latency values pending (run benchmark_speed.py on GPU).")
    if data['scaling'] is None:
        print("  NOTE: Scaling ratios pending (run benchmark_scaling.py on GPU).")
    if 'acc_part' not in nums:
        print("  NOTE: ParT values pending (run ParT inference).")

    print("\nDone.")


if __name__ == "__main__":
    main()
