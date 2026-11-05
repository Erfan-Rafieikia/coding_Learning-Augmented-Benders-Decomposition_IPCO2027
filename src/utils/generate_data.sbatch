#!/bin/bash
#SBATCH --job-name=scflp_gen
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=scflp_gen_%j.out
#SBATCH --error=scflp_gen_%j.err

set -euo pipefail

echo "=============================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "=============================="

# ---- micromamba activation (batch-safe) ----
export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"

echo "micromamba exe: $(command -v micromamba)"
micromamba --version

eval "$(micromamba shell hook -s bash)"
micromamba activate bd_ml

echo "Python being used:"
which python
python --version
# ------------------------------------------

# ========== DYNAMIC PROJECT ROOT DETECTION ==========
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "PROJECT_ROOT detected as: $PROJECT_ROOT"

# Go to project directory
cd "$PROJECT_ROOT"

# Run data generation
python -u src/utils/SCFLP_data_generation_version4.py

echo "=============================="
echo "Job finished at $(date)"
echo "=============================="
