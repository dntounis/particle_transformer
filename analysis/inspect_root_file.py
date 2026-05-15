#!/usr/bin/env python3
"""
Inspect a ROOT file produced by weaver-core's --predict mode.
Prints tree names, branch names, number of entries, and a few sample values.

Usage:
    python inspect_root_file.py <path_to_root_file>
    python inspect_root_file.py  # uses default path
"""

import sys
import os
import uproot
import numpy as np


def inspect_file(filepath):
    """Open a ROOT file and print detailed information about its contents."""

    if not os.path.isfile(filepath):
        print(f"ERROR: File not found: {filepath}")
        return

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File: {filepath}")
    print(f"Size: {file_size_mb:.1f} MB")
    print("=" * 80)

    with uproot.open(filepath) as f:
        keys = f.keys()
        print(f"\nTop-level keys ({len(keys)}):")
        for key in keys:
            print(f"  {key}")

        # Find tree objects
        trees = [k for k in keys if hasattr(f[k], 'keys')]
        if not trees:
            print("\nNo TTree objects found.")
            return

        for tree_name in trees:
            tree = f[tree_name]
            print(f"\n{'=' * 80}")
            print(f"Tree: {tree_name}")
            print(f"Entries: {tree.num_entries:,}")
            print(f"{'=' * 80}")

            branches = tree.keys()
            print(f"\nBranches ({len(branches)}):")

            # Categorize branches
            score_branches = [b for b in branches if b.startswith('score_')]
            label_branches = [b for b in branches if b.startswith('label_')]
            observer_branches = [b for b in branches
                                 if not b.startswith('score_') and not b.startswith('label_')]

            if score_branches:
                print(f"\n  Score branches ({len(score_branches)}):")
                for b in sorted(score_branches):
                    print(f"    {b}")

            if label_branches:
                print(f"\n  Label/truth branches ({len(label_branches)}):")
                for b in sorted(label_branches):
                    print(f"    {b}")

            if observer_branches:
                print(f"\n  Observer/other branches ({len(observer_branches)}):")
                for b in sorted(observer_branches):
                    print(f"    {b}")

            # Print sample values for a few branches
            print(f"\n  Sample values (first 5 entries):")
            n_sample = min(5, tree.num_entries)
            sample_branches = (score_branches[:3] + label_branches[:3] +
                               observer_branches[:3])
            for b in sample_branches:
                try:
                    vals = tree[b].array(entry_stop=n_sample, library="np")
                    print(f"    {b}: {vals}")
                except Exception as e:
                    print(f"    {b}: ERROR reading - {e}")

            # Check label encoding
            if label_branches:
                print(f"\n  Label value distribution (full dataset):")
                for b in sorted(label_branches):
                    if b.startswith('_'):
                        continue
                    try:
                        vals = tree[b].array(library="np")
                        unique, counts = np.unique(vals, return_counts=True)
                        total = len(vals)
                        dist_str = ", ".join(
                            [f"{v}: {c:,} ({100*c/total:.1f}%)"
                             for v, c in zip(unique, counts)]
                        )
                        print(f"    {b}: {dist_str}")
                    except Exception as e:
                        print(f"    {b}: ERROR - {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default: use the first available ALL file
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_files = [
            os.path.join(base, "Jim_results", "2026_1D_v1.1_predict_ALL.root"),
            os.path.join(base, "Jim_results", "22June2025_JetMamba_sequence_predict_ALL.root"),
            os.path.join(base, "Jim_results", "27June2025_JetMamba_sequence_v3_predict_ALL.root"),
            os.path.join(base, "Jim_results", "9Nov2025_JetMamba_v7_predict_ALL.root"),
        ]
        filepath = None
        for f in default_files:
            if os.path.isfile(f):
                filepath = f
                break
        if filepath is None:
            print("No default ROOT file found. Please provide a path as argument.")
            print(f"Usage: python {sys.argv[0]} <path_to_root_file>")
            sys.exit(1)

    inspect_file(filepath)
