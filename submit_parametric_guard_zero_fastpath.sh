#!/usr/bin/env bash
set -euo pipefail

# Four-arm same-code comparison: nonparametric, unguarded parametric,
# standard objective guard, and zero-gamma fastpath guard.
# Submits no jobs before all original configuration/data checks pass.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="agent/parametric-guard-zero-fastpath"
BASE_MASTER="${BASE_MASTER:-$HOME/scratch/Benders_Integration_with_ML/RESULTS/MASTER_RUNS/pmedian_nearest_20261016_160602}"
RESULTS_ROOT_OVERRIDE="${RESULTS_ROOT_OVERRIDE:-$HOME/scratch/Benders_Integration_with_ML/RESULTS}"
INPUT_DATA_ROOT="${LABD_DATA_ROOT:-$HOME/Erfan_projects/Benders_Integration_with_ML/DATA}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
PILOT_TASKS="${PILOT_TASKS:-46 199}"

[[ "$(git -C "$PROJECT_ROOT" branch --show-current)" == "$BRANCH" ]] || {
    echo "ERROR: use worktree on $BRANCH" >&2; exit 1;
}
[[ "$MAX_CONCURRENT" =~ ^[1-4]$ ]] || {
    echo "ERROR: MAX_CONCURRENT must be 1..4" >&2; exit 1;
}
[[ -s "$BASE_MASTER/task_list.txt" && -f "$BASE_MASTER/config.py" ]] || {
    echo "ERROR: missing original master task list or config" >&2; exit 1;
}
[[ -d "$INPUT_DATA_ROOT" ]] || {
    echo "ERROR: original input data directory not found" >&2; exit 1;
}
read -r -a TARGETS <<< "$PILOT_TASKS"
(( ${#TARGETS[@]} > 0 )) || { echo "ERROR: empty PILOT_TASKS" >&2; exit 1; }
declare -A seen=()
for task in "${TARGETS[@]}"; do
    case "$task" in
      46|66|76|199|232) ;;
      *) echo "ERROR: only original matched pilots 46 66 76 199 232 supported" >&2; exit 1 ;;
    esac
    [[ ! -v seen[$task] ]] || { echo "ERROR: duplicate pilot $task" >&2; exit 1; }
    seen[$task]=1
    source="$BASE_MASTER/configs/$(printf 'task_%04d.py' "$task")"
    [[ -r "$source" ]] || { echo "ERROR: missing original config: $source" >&2; exit 1; }
    grep -Fq 'predictor_ML_Augmented_BD_grid = ["PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"]' "$source" || {
      echo "ERROR: task $task is not the matched nearest P-median ridge." >&2; exit 1;
    }
    instance="$(sed -nE 's/^[[:space:]]*"(SCFLP|SCMND)": \["([^"]+)"\].*$/\2/p' "$source" | head -n 1)"
    [[ -n "$instance" && -r "$INPUT_DATA_ROOT/$instance" ]] || {
      echo "ERROR: original DATA missing $instance" >&2; exit 1;
    }
    if ((task < 199)); then baseline=$((task-45)); else baseline=$((task-108)); fi
    [[ -r "$BASE_MASTER/configs/$(printf 'task_%04d.py' "$baseline")" ]] || {
      echo "ERROR: missing matching nonparametric task $baseline" >&2; exit 1;
    }
done

timestamp="$(date +%Y%m%d_%H%M%S)"
run="parametric_guard_zero_fastpath_$timestamp"
root="$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/$run"
[[ ! -e "$root" ]] || { echo "ERROR: target already exists: $root" >&2; exit 1; }
mkdir -p "$root/configs" "$root/SLURM_LOGS"
cp "$BASE_MASTER/config.py" "$root/config.py"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$root/code_commit.txt"
printf '%s\n' "$BASE_MASTER" > "$root/original_master.txt"
printf '%s\n' "$INPUT_DATA_ROOT" > "$root/input_data_root.txt"
printf 'new_task_id\tmethod\toriginal_task_id\tpair_parametric_task_id\tproblem_class\n' > "$root/task_pairs.tsv"
: > "$root/task_list.txt"

idx=0
for target in "${TARGETS[@]}"; do
    if ((target < 199)); then problem=SCFLP; baseline=$((target-45))
    else problem=SCMND; baseline=$((target-108)); fi
    for arm in NONPARAM OFF GUARD FAST; do
        idx=$((idx+1))
        original="$target"
        guard=False
        fast=False
        [[ "$arm" == NONPARAM ]] && original="$baseline"
        [[ "$arm" == GUARD || "$arm" == FAST ]] && guard=True
        [[ "$arm" == FAST ]] && fast=True
        source="$BASE_MASTER/configs/$(printf 'task_%04d.py' "$original")"
        dest="configs/$(printf 'task_%04d.py' "$idx")"
        cp "$source" "$root/$dest"
        cat >> "$root/$dest" <<PYOVERRIDE

# Identical instrumentation code and original settings across all arms.
# Disable diagnostic shadow probes: this compares true production overhead.
parametric_objective_guard = $guard
parametric_guard_gamma_zero_fastpath = $fast
parametric_diagnostic_probe = False
import sys as _fast_sys
_fast_main = _fast_sys.modules.get("__main__")
if _fast_main is not None:
    setattr(_fast_main, "parametric_objective_guard", parametric_objective_guard)
    setattr(_fast_main, "parametric_guard_gamma_zero_fastpath", parametric_guard_gamma_zero_fastpath)
    setattr(_fast_main, "parametric_diagnostic_probe", parametric_diagnostic_probe)
PYOVERRIDE
        printf '%s|%s\n' "$dest" "$problem" >> "$root/task_list.txt"
        printf '%s\t%s\t%s\t%s\t%s\n' "$idx" "$arm" "$original" "$target" "$problem" >> "$root/task_pairs.tsv"
        echo "Prepared $idx: original $target / $problem / $arm"
    done
done

submission="$(sbatch \
  --account=def-brostami-ab_cpu --cpus-per-task=32 --mem-per-cpu=8G --time=16:00:00 \
  --job-name=labd_zero_fast \
  --array="1-${idx}%${MAX_CONCURRENT}" \
  --output="$root/SLURM_LOGS/slurm-%x-%A_%a.out" \
  --error="$root/SLURM_LOGS/slurm-%x-%A_%a.err" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONFIG_ROOT="$root",RESULTS_ROOT_OVERRIDE="$RESULTS_ROOT_OVERRIDE",LABD_DATA_ROOT="$INPUT_DATA_ROOT",UNIQUE_ID="parametric_guard_zero_fastpath",TIMESTAMP="$timestamp",PROBLEM_CLASSES="MIXED" \
  "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch")"
job="${submission##* }"
[[ "$job" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch response $submission" >&2; exit 1; }
printf '%s\n' "$job" > "$root/job_id.txt"
printf '%s\n' "$root" > "$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/latest_parametric_guard_zero_fastpath.txt"
echo "$submission"
echo "Job ID: $job"
echo "Run root: $root"
echo "Tasks: $idx / max concurrent: $MAX_CONCURRENT"
echo "Original datasets read-only: $INPUT_DATA_ROOT"
