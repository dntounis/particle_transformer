#!/bin/bash
#SBATCH --job-name=jetmamba_component_profiling
#SBATCH --partition=ampere
#SBATCH --account=atlas:default
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/component_profiling_%j.out
#SBATCH --error=logs/component_profiling_%j.err

export CONDA_PREFIX=/sdf/data/atlas/u/dntounis/miniconda3
export PATH=${CONDA_PREFIX}/bin/:$PATH
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda activate 2dmamba_v2

cd /fs/ddn/sdf/group/atlas/d/dntounis/CS231N_Final_Project/git_repos/particle_transformer

python scripts/benchmark_component_profiling.py

echo "Done. Results at: analysis/results/component_profiling.json"
