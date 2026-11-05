#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --time=16:00:00

set -euo pipefail

# These MUST be provided by submit.sh via --export
: "${PROJECT_ROOT:?PROJECT_ROOT not set}"
: "${CONFIG_ROOT:?CONFIG_ROOT not set}"
: "${RESULTS_ROOT_OVERRIDE:?RESULTS_ROOT_OVERRIDE not set}"
: "${UNIQUE_ID:?UNIQUE_ID not set}"
: "${TIMESTAMP:?TIMESTAMP not set}"
: "${PROBLEM_CLASSES:?PROBLEM_CLASSES not set}"

LIST_FILE="$CONFIG_ROOT/task_list.txt"
RUN_IDENTIFIER="${UNIQUE_ID}_${TIMESTAMP}"

echo "=== Job start ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}, ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"
echo "Project Run Identifier: $RUN_IDENTIFIER"
echo "CONFIG_ROOT=$CONFIG_ROOT"
echo "LIST_FILE=$LIST_FILE"
echo "================="

# Pick correct override file and problem class for this array task
TASK_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")
OVERRIDE_CONFIG=$(echo "$TASK_LINE" | cut -d'|' -f1)
PROBLEM_CLASS=$(echo "$TASK_LINE" | cut -d'|' -f2)

echo "Task Config: OVERRIDE_CONFIG=$OVERRIDE_CONFIG, PROBLEM_CLASS=$PROBLEM_CLASS"

# ========== EXPLICIT GUROBI LICENSE SELECTION ==========
# sbatch --export=ALL can inherit a Gurobi module's token-server settings
# from the submitting terminal.  This branch uses the approved personal
# academic WLS license, not Nibi's access-restricted shared token server.
if type module >/dev/null 2>&1; then
  module unload gurobi 2>/dev/null || true
fi
export GRB_LICENSE_FILE="$HOME/gurobi.lic"
EXPECTED_GUROBI_LICENSE_ID="${EXPECTED_GUROBI_LICENSE_ID:-2623115}"
if [[ ! -r "$GRB_LICENSE_FILE" ]]; then
  echo "ERROR: WLS license file is missing or unreadable: $GRB_LICENSE_FILE" >&2
  exit 1
fi
if ! grep -Fxq "LICENSEID=$EXPECTED_GUROBI_LICENSE_ID" "$GRB_LICENSE_FILE" || \
   ! grep -q '^WLSACCESSID=' "$GRB_LICENSE_FILE"; then
  echo "ERROR: Expected academic WLS license $EXPECTED_GUROBI_LICENSE_ID in $GRB_LICENSE_FILE" >&2
  exit 1
fi
echo "Selected personal WLS license ID: $EXPECTED_GUROBI_LICENSE_ID"

# ========== ENVIRONMENT ACTIVATION (ROBUST) ==========
# Export MAMBA_ROOT_PREFIX to prevent "unbound variable" error
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PATH="$HOME/.local/bin:$PATH"

# Initialize micromamba shell
eval "$($HOME/.local/bin/micromamba shell hook -s bash)" || {
  echo "⚠️  Warning: micromamba hook failed, attempting manual activation..."
}

# Activate the bd_ml environment with error handling
if micromamba activate bd_ml 2>/dev/null; then
  echo "✅ Successfully activated bd_ml environment via micromamba"
elif conda activate bd_ml 2>/dev/null; then
  echo "✅ Successfully activated bd_ml environment via conda (fallback)"
else
  echo "❌ ERROR: Failed to activate bd_ml environment"
  echo "ERROR: Refusing to run experiments outside bd_ml." >&2
  exit 1
fi

python --version
echo "Python executable: $(which python)"
echo ""

# Thread limits
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ========== CREATE PROBLEM-CLASS-SPECIFIC TASK DIRECTORY ==========
# Each task creates a folder in its problem class's results directory
# Folder naming: task_{ARRAY_TASK_ID}_{SLURM_JOB_ID}
# Use RESULTS_ROOT_OVERRIDE from submit.sh for remote execution
SNAP_ROOT="$RESULTS_ROOT_OVERRIDE/${PROBLEM_CLASS}_class/RESULTS/$RUN_IDENTIFIER"
TASK_DIR="$SNAP_ROOT/task_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
mkdir -p "$TASK_DIR"

# Pass run metadata to Python
export RUN_IDENTIFIER="$RUN_IDENTIFIER"
export RUN_PROBLEM_CLASS="$PROBLEM_CLASS"
export RUN_DIRECTORY="$TASK_DIR"
export RESULTS_ROOT_OVERRIDE="$RESULTS_ROOT_OVERRIDE"  # Pass results root override to Python
export SLURM_JOB_ID="$SLURM_JOB_ID"                    # Pass Job ID to Python
export OVERRIDE_CONFIG_NAME="$OVERRIDE_CONFIG"         # Pass config filename
export ARRAY_TASK_ID="$SLURM_ARRAY_TASK_ID"           # Pass array task ID

# ========== ISOLATED TASK WORKING DIRECTORY ==========
TASK_WORKDIR="$SLURM_TMPDIR/bd_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TASK_WORKDIR"

# Copy config + override into the isolated task working directory
/bin/cp -f "$CONFIG_ROOT/config.py" "$TASK_WORKDIR/config.py"
/bin/cp -f "$CONFIG_ROOT/$OVERRIDE_CONFIG" "$TASK_WORKDIR/config_override.py"

# Ensure imports prefer the task workdir first (then src)
export PYTHONPATH="$TASK_WORKDIR:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Run from isolated directory so "import config" and "import config_override" are per-task safe
cd "$TASK_WORKDIR"
# Gurobi reads gurobi.env when its environment is created. This applies to
# master and scenario models, independently of BLAS/OMP thread settings.
printf 'Threads %s\n' "$SLURM_CPUS_PER_TASK" > gurobi.env
if [[ "${LABD_FROZEN_PILOT:-0}" == "1" ]]; then
  cp "$CONFIG_ROOT/source_commit.txt" "$TASK_DIR/source_commit.txt"
  cp "$TASK_WORKDIR/config_override.py" "$TASK_DIR/config_override.py"
  cp "$TASK_WORKDIR/gurobi.env" "$TASK_DIR/gurobi.env"
  scontrol show job "$SLURM_JOB_ID" > "$TASK_DIR/slurm_job.txt"
  python - <<'ENV' > "$TASK_DIR/environment.txt"
import importlib.metadata as m
import os, platform
print('python', platform.python_version())
print('platform', platform.platform())
print('cpu_allocated', os.environ['SLURM_CPUS_PER_TASK'])
for name in ('gurobipy', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'gensim'):
    print(name, m.version(name))
ENV
  lscpu >> "$TASK_DIR/environment.txt"
fi

echo "Running from TASK_WORKDIR=$TASK_WORKDIR"
echo "Output directory: $TASK_DIR"
python -V
which python

# ========== EXECUTE MAIN WITH LOGGING ==========
# Run main.py and capture output to both SLURM logs and per-task file
# Capture the Python exit status even when it is nonzero, so a failure can
# be logged before the Slurm task exits as FAILED.
set +e
srun python -c '
import config, os, runpy
from pathlib import Path
assert Path(config.__file__).resolve() == (Path.cwd()/"config.py").resolve(), config.__file__
runpy.run_path(str(Path(os.environ["PROJECT_ROOT"])/"src/main.py"), run_name="__main__")
'  2>&1 | tee "$TASK_DIR/output_task_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ "$EXIT_CODE" == "0" && "${LABD_FROZEN_PILOT:-0}" == "1" ]]; then
  # main.py records caught errors as workbook rows; they are not Slurm successes.
  python - "$TASK_DIR/solution_log.xlsx" <<'CHECK' || EXIT_CODE=$?
import sys
from openpyxl import load_workbook
from utils.exclusive_profiler import TIMING_SCHEMA_VERSION, SOLUTION_TIME_METRIC_VERSION
from utils.convergence_diagnostics import TRACE_SCHEMA_VERSION
with_path = sys.argv[1]
wb = load_workbook(with_path, read_only=True, data_only=True)
try:
    rows = wb.active.iter_rows(values_only=True)
    header = next(rows)
    records = [dict(zip(header, row)) for row in rows if any(v is not None for v in row)]
    if len(records) != 1:
        raise RuntimeError('Expected one pilot result')
    status = str(records[0].get('stopping_reason', ''))
    if status.startswith('CODE_ERROR') or status == 'SOLUTION_NULL':
        raise RuntimeError('Pilot recorded '+status)
    for field, expected in (
        ('timing_schema_version', TIMING_SCHEMA_VERSION),
        ('solution_time_metric_version', SOLUTION_TIME_METRIC_VERSION),
        ('convergence_trace_schema_version', TRACE_SCHEMA_VERSION),
    ):
        if records[0].get(field) != expected:
            raise RuntimeError(f'Pilot did not record current {field}: expected {expected}')
finally:
    wb.close()
CHECK
fi

echo "=== Job end ==="
echo "Date: $(date)"
echo "Exit Code: $EXIT_CODE"
exit "$EXIT_CODE"
