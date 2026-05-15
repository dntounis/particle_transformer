#!/bin/bash
###############################################################################
# run_all_inference.sh
#
# Run inference (--predict) and model profiling (--print) for all 5 trained
# models on the full JetClass test set.
#
# Usage: Run inside an interactive GPU allocation on the S3DF ampere partition:
#   salloc -p ampere -n 1 --gpus 4 -t 08:00:00
#   bash scripts/run_all_inference.sh
#
# All paths are absolute so the script can be invoked from any directory.
###############################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR="/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer"
WEAVER_TRAIN="/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/weaver-core/weaver/train.py"
DATA_CONFIG="${BASE_DIR}/data/JetClass/JetClass_full.yaml"
TEST_DATA="${BASE_DIR}/datasets/JetClass/Pythia/test_20M/*.root"
OUTPUT_DIR="${BASE_DIR}/Jim_results"
LOG_DIR="${BASE_DIR}/logs"

GPUS="0,1,2,3"
BATCH_SIZE=2048
NUM_WORKERS=2

CONDA_BASE="/sdf/data/atlas/u/dntounis/miniconda3"
CONDA_ENV="2dmamba_v2"

# ============================================================================
# Model definitions: NAME | NETWORK_CONFIG | CHECKPOINT
# ============================================================================
declare -a MODEL_NAMES=(
    "jetmamba_v1.1"
    "jetmamba_v1.2"
    "jetmamba_v1.3"
    "jetmamba_v2.1"
    "ParT"
)

declare -a NETWORK_CONFIGS=(
    "${BASE_DIR}/networks/jetmamba_2026_1D_v1.1.py"
    "${BASE_DIR}/networks/jetmamba_2026_1D_v1.2.py"
    "${BASE_DIR}/networks/jetmamba_2026_1D_v1.3.py"
    "${BASE_DIR}/networks/jetmamba_2026_1D_v2.1.py"
    "${BASE_DIR}/networks/example_ParticleTransformer.py"
)

declare -a CHECKPOINTS=(
    "${BASE_DIR}/models/jetmamba_2026_1D_v1.1_best_epoch_state.pt"
    "${BASE_DIR}/models/jetmamba_2026_1D_v1.2_best_epoch_state.pt"
    "${BASE_DIR}/models/jetmamba_2026_1D_v1.3_best_epoch_state.pt"
    "${BASE_DIR}/models/jetmamba_2026_1D_v2.1_best_epoch_state.pt"
    "${BASE_DIR}/models/ParT_full.pt"
)

NUM_MODELS=${#MODEL_NAMES[@]}

# ============================================================================
# Environment setup
# ============================================================================
echo "============================================================"
echo " JetClass Inference & Benchmarking Script"
echo " Started: $(date)"
echo "============================================================"

# Activate conda (disable -u temporarily; conda activation scripts reference unset vars)
set +u
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    export PATH="${CONDA_BASE}/bin:${PATH}"
fi
conda activate "${CONDA_ENV}"
set -u
echo "Conda env: ${CONDA_ENV}"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs visible: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# ============================================================================
# Phase 1: Run --predict for all models on the full test set
# ============================================================================
echo "============================================================"
echo " Phase 1: Inference (--predict) on full test set"
echo "============================================================"
echo ""

for ((i=0; i<NUM_MODELS; i++)); do
    NAME="${MODEL_NAMES[$i]}"
    NET_CFG="${NETWORK_CONFIGS[$i]}"
    CKPT="${CHECKPOINTS[$i]}"
    PREDICT_OUT="${OUTPUT_DIR}/2026_${NAME}_predict_ALL.root"

    echo "------------------------------------------------------------"
    echo " Model: ${NAME}"
    echo " Network: ${NET_CFG}"
    echo " Checkpoint: ${CKPT}"
    echo " Output: ${PREDICT_OUT}"
    echo "------------------------------------------------------------"

    # Verify files exist
    if [ ! -f "${NET_CFG}" ]; then
        echo "ERROR: Network config not found: ${NET_CFG}"
        continue
    fi
    if [ ! -f "${CKPT}" ]; then
        echo "ERROR: Checkpoint not found: ${CKPT}"
        continue
    fi

    # Skip if output already exists and is non-empty
    if [ -f "${PREDICT_OUT}" ] && [ -s "${PREDICT_OUT}" ]; then
        echo " SKIPPING: Output already exists ($(du -h "${PREDICT_OUT}" | cut -f1))"
        echo ""
        continue
    fi

    START_TIME=$(date +%s.%N)
    echo " Start time: $(date)"

    python "${WEAVER_TRAIN}" \
        --predict \
        --data-config "${DATA_CONFIG}" \
        --network-config "${NET_CFG}" \
        -m "${CKPT}" \
        --gpus "${GPUS}" \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --use-amp \
        --predict-output "${PREDICT_OUT}" \
        -t "${TEST_DATA}" \
        2>&1 | tee "${LOG_DIR}/predict_${NAME}.log"

    END_TIME=$(date +%s.%N)
    ELAPSED=$(python -c "print(f'{${END_TIME} - ${START_TIME}:.2f}')")
    echo " End time: $(date)"
    echo " Elapsed: ${ELAPSED} seconds"
    echo ""
done

# ============================================================================
# Phase 2: Run --print for all models to get params/FLOPs
# ============================================================================
echo "============================================================"
echo " Phase 2: Model profiling (--print) for params/FLOPs"
echo "============================================================"
echo ""

for ((i=0; i<NUM_MODELS; i++)); do
    NAME="${MODEL_NAMES[$i]}"
    NET_CFG="${NETWORK_CONFIGS[$i]}"
    CKPT="${CHECKPOINTS[$i]}"
    PRINT_LOG="${LOG_DIR}/print_${NAME}.log"

    echo "------------------------------------------------------------"
    echo " Model: ${NAME}"
    echo "------------------------------------------------------------"

    # Skip if print log already exists and contains FLOPs info
    if [ -f "${PRINT_LOG}" ] && grep -q "FLOPs" "${PRINT_LOG}" 2>/dev/null; then
        echo " SKIPPING: Print log already exists"
        echo ""
        continue
    fi

    python "${WEAVER_TRAIN}" \
        --print \
        --data-config "${DATA_CONFIG}" \
        --network-config "${NET_CFG}" \
        -m "${CKPT}" \
        --gpus "" \
        --batch-size ${BATCH_SIZE} \
        2>&1 | tee "${PRINT_LOG}"

    echo ""
done

# ============================================================================
# Phase 3: Run speed benchmark
# ============================================================================
echo "============================================================"
echo " Phase 3: Speed benchmark (scripts/benchmark_speed.py)"
echo "============================================================"
echo ""

python "${BASE_DIR}/scripts/benchmark_speed.py"

# ============================================================================
# Phase 4: Sequence length scaling study
# ============================================================================
echo "============================================================"
echo " Phase 4: Scaling study (scripts/benchmark_scaling.py)"
echo "============================================================"
echo ""

python "${BASE_DIR}/scripts/benchmark_scaling.py"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo " All done!"
echo " Finished: $(date)"
echo "============================================================"
echo ""
echo " Prediction outputs:  ${OUTPUT_DIR}/2026_*_predict_ALL.root"
echo " Print logs:          ${LOG_DIR}/print_*.log"
echo " Speed benchmark:     ${BASE_DIR}/analysis/results/inference_speed.json"
echo " Scaling study:       ${BASE_DIR}/analysis/results/scaling_study.json"
echo ""
