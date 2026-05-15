#!/bin/bash
###############################################################################
# run_ParT_inference_split.sh
#
# Run ParT inference in two halves to avoid OOM, then merge ROOT files.
# The full 20M test set (200 files) is split into two 100-file batches.
#
# Usage:
#   bash scripts/run_ParT_inference_split.sh
###############################################################################

set -euo pipefail

BASE_DIR="/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer"
WEAVER_TRAIN="/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/weaver-core/weaver/train.py"
DATA_CONFIG="${BASE_DIR}/data/JetClass/JetClass_full.yaml"
TEST_DIR="${BASE_DIR}/datasets/JetClass/Pythia/test_20M"
OUTPUT_DIR="${BASE_DIR}/Jim_results"
LOG_DIR="${BASE_DIR}/logs"

NET_CFG="${BASE_DIR}/networks/example_ParticleTransformer.py"
CKPT="${BASE_DIR}/models/ParT_full.pt"

GPUS="0,1,2,3"
BATCH_SIZE=2048
NUM_WORKERS=2

CONDA_BASE="/sdf/data/atlas/u/dntounis/miniconda3"
CONDA_ENV="2dmamba_v2"

# Activate conda
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

echo "============================================================"
echo " ParT Inference (Split into 2 halves)"
echo " Started: $(date)"
echo "============================================================"

# Get all test files sorted, split into two halves
mapfile -t ALL_FILES < <(ls "${TEST_DIR}"/*.root | sort)
TOTAL=${#ALL_FILES[@]}
HALF=$(( TOTAL / 2 ))

echo "Total test files: ${TOTAL}"
echo "Half 1: files 0-$((HALF-1)) (${HALF} files)"
echo "Half 2: files ${HALF}-$((TOTAL-1)) ($((TOTAL-HALF)) files)"
echo ""

# Create temp directories with symlinks so weaver can glob *.root
TMPDIR_HALF1=$(mktemp -d)
TMPDIR_HALF2=$(mktemp -d)
trap 'rm -rf "${TMPDIR_HALF1}" "${TMPDIR_HALF2}"' EXIT

for f in "${ALL_FILES[@]:0:${HALF}}"; do
    ln -s "$f" "${TMPDIR_HALF1}/"
done
for f in "${ALL_FILES[@]:${HALF}}"; do
    ln -s "$f" "${TMPDIR_HALF2}/"
done

echo "Temp dir half1: ${TMPDIR_HALF1} ($(ls "${TMPDIR_HALF1}" | wc -l) symlinks)"
echo "Temp dir half2: ${TMPDIR_HALF2} ($(ls "${TMPDIR_HALF2}" | wc -l) symlinks)"
echo ""

OUT_HALF1="${OUTPUT_DIR}/2026_ParT_predict_half1.root"
OUT_HALF2="${OUTPUT_DIR}/2026_ParT_predict_half2.root"
OUT_FINAL="${OUTPUT_DIR}/2026_ParT_predict_ALL.root"

# --- Half 1 ---
if [ -f "${OUT_HALF1}" ] && [ -s "${OUT_HALF1}" ]; then
    echo "SKIPPING Half 1: output already exists ($(du -h "${OUT_HALF1}" | cut -f1))"
else
    echo "Running Half 1..."
    echo "  Start: $(date)"
    python "${WEAVER_TRAIN}" \
        --predict \
        --data-config "${DATA_CONFIG}" \
        --network-config "${NET_CFG}" \
        -m "${CKPT}" \
        --gpus "${GPUS}" \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --use-amp \
        --predict-output "${OUT_HALF1}" \
        -t "${TMPDIR_HALF1}/*.root" \
        2>&1 | tee "${LOG_DIR}/predict_ParT_half1.log"
    echo "  End: $(date)"
fi
echo ""

# --- Half 2 ---
if [ -f "${OUT_HALF2}" ] && [ -s "${OUT_HALF2}" ]; then
    echo "SKIPPING Half 2: output already exists ($(du -h "${OUT_HALF2}" | cut -f1))"
else
    echo "Running Half 2..."
    echo "  Start: $(date)"
    python "${WEAVER_TRAIN}" \
        --predict \
        --data-config "${DATA_CONFIG}" \
        --network-config "${NET_CFG}" \
        -m "${CKPT}" \
        --gpus "${GPUS}" \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --use-amp \
        --predict-output "${OUT_HALF2}" \
        -t "${TMPDIR_HALF2}/*.root" \
        2>&1 | tee "${LOG_DIR}/predict_ParT_half2.log"
    echo "  End: $(date)"
fi
echo ""

# --- Merge ---
echo "Merging halves into final output..."
python -c "
import uproot
import numpy as np
import os

out1 = '${OUT_HALF1}'
out2 = '${OUT_HALF2}'
out_final = '${OUT_FINAL}'

print(f'  Reading {out1}...')
f1 = uproot.open(out1)
tree1 = f1[f1.keys()[0]]
data1 = tree1.arrays(library='np')

print(f'  Reading {out2}...')
f2 = uproot.open(out2)
tree2 = f2[f2.keys()[0]]
data2 = tree2.arrays(library='np')

print(f'  Half 1: {len(list(data1.values())[0])} events')
print(f'  Half 2: {len(list(data2.values())[0])} events')

# Concatenate
merged = {}
for key in data1.keys():
    merged[key] = np.concatenate([data1[key], data2[key]], axis=0)

total = len(list(merged.values())[0])
print(f'  Merged: {total} events')

# Write
print(f'  Writing {out_final}...')
with uproot.recreate(out_final) as fout:
    fout['Events'] = merged

print('  Done!')
"
echo ""

echo "============================================================"
echo " ParT Inference Complete!"
echo " Finished: $(date)"
echo " Output: ${OUT_FINAL}"
echo "============================================================"
