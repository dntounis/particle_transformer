#!/bin/bash
###############################################################################
# finalize_paper.sh
#
# Run the full analysis pipeline and update the paper with results.
# Execute after all inference outputs are available in Jim_results/.
#
# Usage:
#   bash scripts/finalize_paper.sh
###############################################################################

set -euo pipefail

BASE_DIR="/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer"
PAPER_DIR="/fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/paper"
CONDA_BASE="/sdf/data/atlas/u/dntounis/miniconda3"
CONDA_ENV="2dmamba_v2"

# Activate conda
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

echo "============================================================"
echo " Paper Finalization Pipeline"
echo " Started: $(date)"
echo "============================================================"

# Check which inference outputs are available
echo ""
echo "[0/4] Checking available inference outputs..."
for f in 2026_jetmamba_v1.1_predict_ALL.root \
         2026_jetmamba_v1.2_predict_ALL.root \
         2026_jetmamba_v1.3_predict_ALL.root \
         2026_jetmamba_v2.1_predict_ALL.root \
         2026_ParT_predict_ALL.root; do
    if [ -f "${BASE_DIR}/Jim_results/$f" ]; then
        size=$(du -h "${BASE_DIR}/Jim_results/$f" | cut -f1)
        echo "  OK: $f ($size)"
    else
        echo "  MISSING: $f"
    fi
done

# Step 1: Run comparison plots and metrics
echo ""
echo "[1/4] Running classification analysis..."
cd "${BASE_DIR}"
python analysis/produce_comparison_plots.py

# Step 2: Run speed benchmark (if GPU available)
echo ""
echo "[2/4] Running speed benchmark..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    python scripts/benchmark_speed.py
    echo "  Speed benchmark complete."
else
    echo "  SKIP: No GPU available for speed benchmark."
fi

# Step 3: Run scaling study (if GPU available)
echo ""
echo "[3/4] Running scaling study..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    python scripts/benchmark_scaling.py
    echo "  Scaling study complete."
else
    echo "  SKIP: No GPU available for scaling study."
fi

# Step 4: Update paper with results
echo ""
echo "[4/4] Updating paper .tex with results..."
python scripts/update_paper_with_results.py

# Copy figures to paper directory
echo ""
echo "Copying figures to paper directory..."
mkdir -p "${PAPER_DIR}/figures"
cp -v "${BASE_DIR}/analysis/results/figures/"*.pdf "${PAPER_DIR}/figures/" 2>/dev/null || true

echo ""
echo "============================================================"
echo " Done! $(date)"
echo "============================================================"
echo ""
echo " Paper:   ${PAPER_DIR}/jetmamba_mlst_draft.tex"
echo " Figures: ${PAPER_DIR}/figures/"
echo " Results: ${BASE_DIR}/analysis/results/"
echo ""
echo " Next steps:"
echo "   1. Review remaining \\fillin{} placeholders"
echo "   2. Compile: cd ${PAPER_DIR} && pdflatex jetmamba_mlst_draft"
echo "   3. Check figures render correctly"
echo ""
