#!/usr/bin/env bash
set -euo pipefail

# Parametric objective-dominance pilots matched to completed P-median runs.
# Run from a SEPARATE Git worktree on branch agent/parametric-objective-dominance.
# This script never changes files under the original experiment master.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The original experiment input data remain on disk in the existing checkout.
# Read them in place; never copy or change datasets inside the new worktree.
INPUT_DATA_ROOT="${LABD_DATA_ROOT:-$HOME/Erfan_projects/Benders_Integration_with_ML/DATA}"
if [[ ! -d "$INPUT_DATA_ROOT" ]]; then
    echo "ERROR: original dataset directory missing: $INPUT_DATA_ROOT" >&2
    exit 1
fi
BASE_MASTER="${BASE_MASTER:-$HOME/scratch/Benders_Integration_with_ML/RESULTS/MASTER_RUNS/pmedian_nearest_20261016_160602}"
RESULTS_ROOT_OVERRIDE="${RESULTS_ROOT_OVERRIDE:-$HOME/scratch/Benders_Integration_with_ML/RESULTS}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
BRANCH="agent/parametric-objective-dominance"
PILOT_IDS=(46 66 76 199 232)

if [[ "$(git -C "$PROJECT_ROOT" branch --show-current)" != "$BRANCH" ]]; then
    echo "ERROR: expected separate worktree on $BRANCH" >&2
    exit 1
fi
if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || (( MAX_CONCURRENT > 4 )); then
    echo "ERROR: choose MAX_CONCURRENT between 1 and 4 (shared 20-session WLS license)." >&2
    exit 1
fi
if [[ ! -s "$BASE_MASTER/task_list.txt" || ! -f "$BASE_MASTER/code_commit.txt" ]]; then
    echo "ERROR: original P-median manifest missing in $BASE_MASTER" >&2
    exit 1
fi
if [[ "$(wc -l < "$BASE_MASTER/task_list.txt")" -ne 306 ]]; then
    echo "ERROR: expected original 306-task manifest" >&2
    exit 1
fi

# Git worktrees exclude untracked datasets. Verify every pilot's input file
# before allocating Slurm resources; the data must match the original runs.
for id in "${PILOT_IDS[@]}"; do
    task_line="$(sed -n "${id}p" "$BASE_MASTER/task_list.txt")"
    config_rel="${task_line%%|*}"
    config_file="$BASE_MASTER/$config_rel"
    if [[ ! -r "$config_file" ]]; then
        echo "ERROR: original task $id config not readable: $config_file" >&2
        exit 1
    fi
    instance="$(sed -nE 's/^[[:space:]]*"(SCFLP|SCMND)": \["([^"]+)"\].*$/\2/p' "$config_file" | head -n 1)"
    if [[ -z "$instance" ]]; then
        echo "ERROR: cannot parse instance path from $config_file" >&2
        exit 1
    fi
    if [[ ! -f "$INPUT_DATA_ROOT/$instance" ]]; then
        echo "ERROR: task $id input data is not available: $INPUT_DATA_ROOT/$instance" >&2
        echo "All input files must be present under the original DATA directory." >&2
        exit 1
    fi
done
POINTER="$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/latest_parametric_guard_pilot.txt"
if [[ -f "$POINTER" ]]; then
    echo "ERROR: a guard pilot batch already exists: $(cat "$POINTER")" >&2
    echo "Check its status instead of submitting duplicates." >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_IDENTIFIER="parametric_guard_pilot_${TIMESTAMP}"
PILOT_ROOT="$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/${RUN_IDENTIFIER}"
mkdir -p "$PILOT_ROOT/configs" "$PILOT_ROOT/SLURM_LOGS"
cp "$BASE_MASTER/config.py" "$PILOT_ROOT/config.py"
cp "$BASE_MASTER/task_list.txt" "$PILOT_ROOT/task_list.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$PILOT_ROOT/code_commit.txt"
printf '%s\n' "$BASE_MASTER" > "$PILOT_ROOT/baseline_master.txt"
printf '%s\n' "${PILOT_IDS[*]}" > "$PILOT_ROOT/pilot_task_ids.txt"

for id in "${PILOT_IDS[@]}"; do
    task_line="$(sed -n "${id}p" "$BASE_MASTER/task_list.txt")"
    config_rel="${task_line%%|*}"
    problem="${task_line#*|}"
    config_path="$BASE_MASTER/$config_rel"
    if [[ ! -r "$config_path" ]]; then
        echo "ERROR: missing original config $config_path" >&2
        exit 1
    fi
    if ! grep -Fq 'predictor_ML_Augmented_BD_grid = ["PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"]' "$config_path"; then
        echo "ERROR: task $id is not the expected parametric nearest-P-median method" >&2
        exit 1
    fi
    cp "$config_path" "$PILOT_ROOT/$config_rel"
    cat >> "$PILOT_ROOT/$config_rel" <<'PYGUARD'

# Objective-dominance pilot: retain all original settings and only enable guard.
parametric_objective_guard = True
import sys as _guard_sys
_guard_main = _guard_sys.modules.get("__main__")
if _guard_main is not None:
    setattr(_guard_main, "parametric_objective_guard", True)
PYGUARD
    echo "Matched pilot task $id ($problem): $config_rel"
done

ARRAY_SPEC="$(IFS=,; echo "${PILOT_IDS[*]}")%${MAX_CONCURRENT}"
# The same existing submission job script sets 32 CPUs, 8GB/CPU and 16h wall time.
SUBMISSION="$(sbatch \
    --account=def-brostami-ab_cpu \
    --cpus-per-task=32 --mem-per-cpu=8G --time=16:00:00 \
    --job-name=labd_objguard \
    --array="$ARRAY_SPEC" \
    --output="$PILOT_ROOT/SLURM_LOGS/slurm-%x-%A_%a.out" \
    --error="$PILOT_ROOT/SLURM_LOGS/slurm-%x-%A_%a.err" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONFIG_ROOT="$PILOT_ROOT",RESULTS_ROOT_OVERRIDE="$RESULTS_ROOT_OVERRIDE",LABD_DATA_ROOT="$INPUT_DATA_ROOT",UNIQUE_ID="parametric_guard_pilot",TIMESTAMP="$TIMESTAMP",PROBLEM_CLASSES="MIXED" \
    "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch")"
JOB_ID="${SUBMISSION##* }"
if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: unexpected Slurm submission response: $SUBMISSION" >&2
    exit 1
fi
echo "$JOB_ID" > "$PILOT_ROOT/job_id.txt"
printf '%s\n' "$PILOT_ROOT" > "$POINTER"
echo "$SUBMISSION"
echo "Pilot Job ID: $JOB_ID"
echo "Pilot results root: $PILOT_ROOT"
echo "Total pilots: ${#PILOT_IDS[@]}, max concurrent: $MAX_CONCURRENT"
echo "Input datasets (read-only): $INPUT_DATA_ROOT"
