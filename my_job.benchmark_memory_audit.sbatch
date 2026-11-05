#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --time=16:00:00
# The sbatch command-line explicitly overrides this default to 2G or 8G.
# No optimization code or underlying job launcher is changed.
set -euo pipefail
: "${PROJECT_ROOT:?}"
echo "=== Memory audit: requested resources and actual allocation ==="
echo "Node: $(hostname)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-unset}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"
scontrol show job "$SLURM_JOB_ID" -o || true
echo "=== CPU model ==="
lscpu | grep -E 'Architecture:|Model name:|CPU\(s\):|Thread\(s\) per core:|Socket\(s\):' || true
echo "=== Python and numerical library versions ==="
envpython="$HOME/micromamba/envs/bd_ml/bin/python"
if [[ -x "$envpython" ]]; then
  "$envpython" - <<'PY'
import sys
print("Python:", sys.version.replace("\n"," "))
for name in ("gurobipy", "numpy", "scipy", "pandas"):
    try:
        module = __import__(name)
        version = module.gurobi.version() if name == "gurobipy" else getattr(module,"__version__","unknown")
        print(name, version)
    except Exception as exc:
        print(name, type(exc).__name__, str(exc))
PY
fi
echo "=== Execute unchanged regular Nibi job launcher ==="
exec /bin/bash "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch"
