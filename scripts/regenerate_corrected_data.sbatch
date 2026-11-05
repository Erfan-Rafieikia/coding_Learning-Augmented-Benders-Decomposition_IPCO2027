#!/bin/bash
#SBATCH --account=def-brostami-ab_cpu
#SBATCH --job-name=labd_regen_data
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-6
#SBATCH --output=labd_regen_data-%A_%a.out
# Array tasks 0..5: SCFLP p1..p71 at S = 100,300,500,700,900,1100 (one size per task, 16 files in
# parallel, resumable: resubmitting skips files that are already complete).
# Array task 6: SCMND S=1100 with the unchanged batch generator.
# Nothing in DATA/ is replaced until you run:  python scripts/regenerate_corrected_data.py --data-root $DATA --install
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="${LABD_PYTHON:-$HOME/micromamba/envs/bd_ml/bin/python}"
DATA="${LABD_DATA:-$HOME/Erfan_projects/Benders_Integration_with_ML/DATA}"
SIZES=(100 300 500 700 900 1100)
if (( SLURM_ARRAY_TASK_ID < 6 )); then
  "$PY" scripts/regenerate_corrected_data.py --data-root "$DATA" \
        --generate-scflp "${SIZES[$SLURM_ARRAY_TASK_ID]}" --jobs "$SLURM_CPUS_PER_TASK"
else
  "$PY" scripts/regenerate_corrected_data.py --data-root "$DATA" --generate-scmnd 1100
fi
