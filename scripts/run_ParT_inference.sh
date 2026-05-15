#!/bin/bash
###############################################################################
# run_ParT_inference.sh
#
# Run ParT inference with the properly trained Aug 2025 checkpoint.
# Use this if the main run_all_inference.sh used the wrong ParT checkpoint.
#
# Usage: Run inside an interactive GPU allocation:
#   salloc -p ampere -n 1 --gpus 4 -t 02:00:00
#   bash scripts/run_ParT_inference.sh
###############################################################################

set -euo pipefail

BASE_DIR="/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer"
WEAVER_TRAIN="/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/weaver-core/weaver/train.py"
DATA_CONFIG="${BASE_DIR}/data/JetClass/JetClass_full.yaml"
TEST_DATA="${BASE_DIR}/datasets/JetClass/Pythia/test_20M/*.root"
OUTPUT_DIR="${BASE_DIR}/Jim_results"
LOG_DIR="${BASE_DIR}/logs"

CONDA_BASE="/sdf/data/atlas/u/dntounis/miniconda3"
CONDA_ENV="2dmamba_v2"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

echo "Running ParT inference with ParT_full.pt (official published model)"
echo "Started: $(date)"

python "${WEAVER_TRAIN}" \
    --predict \
    --data-config "${DATA_CONFIG}" \
    --network-config "${BASE_DIR}/networks/example_ParticleTransformer.py" \
    -m "${BASE_DIR}/models/ParT_full.pt" \
    --gpus 0,1,2,3 \
    --batch-size 2048 \
    --num-workers 4 \
    --use-amp \
    --predict-output "${OUTPUT_DIR}/2026_ParT_predict_ALL.root" \
    -t ${TEST_DATA} \
    2>&1 | tee "${LOG_DIR}/predict_ParT_full.log"

echo ""
echo "ParT inference complete: $(date)"
echo "Output: ${OUTPUT_DIR}/2026_ParT_predict_ALL.root"
