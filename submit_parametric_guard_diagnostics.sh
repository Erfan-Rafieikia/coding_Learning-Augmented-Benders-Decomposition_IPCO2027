#!/usr/bin/env bash
set -euo pipefail

# Three matched methods on each selected original input instance.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="agent/parametric-guard-instrumentation"
BASE_MASTER="${BASE_MASTER:-$HOME/scratch/Benders_Integration_with_ML/RESULTS/MASTER_RUNS/pmedian_nearest_20261016_160602}"
RESULTS_ROOT_OVERRIDE="${RESULTS_ROOT_OVERRIDE:-$HOME/scratch/Benders_Integration_with_ML/RESULTS}"
INPUT_DATA_ROOT="${LABD_DATA_ROOT:-$HOME/Erfan_projects/Benders_Integration_with_ML/DATA}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DIAGNOSTIC_TASKS="${DIAGNOSTIC_TASKS:-46 199}"

[[ "$(git -C "$PROJECT_ROOT" branch --show-current)" == "$BRANCH" ]] || {
  echo "ERROR: expected isolated branch $BRANCH" >&2; exit 1;
}
[[ "$MAX_CONCURRENT" =~ ^[1-4]$ ]] || {
  echo "ERROR: MAX_CONCURRENT must be 1..4" >&2; exit 1;
}
[[ -f "$BASE_MASTER/config.py" && -s "$BASE_MASTER/task_list.txt" ]] || {
  echo "ERROR: missing original experiment master." >&2; exit 1;
}
[[ -d "$INPUT_DATA_ROOT" ]] || {
  echo "ERROR: original DATA directory unavailable." >&2; exit 1;
}
read -r -a TARGETS <<< "$DIAGNOSTIC_TASKS"
(( ${#TARGETS[@]} > 0 )) || { echo "ERROR: no targets specified" >&2; exit 1; }
declare -A seen=()
for task in "${TARGETS[@]}"; do
  case "$task" in
    46|66|76|199|232) ;;
    *) echo "ERROR: allowed matched tasks are 46 66 76 199 232; got $task" >&2; exit 1 ;;
  esac
  [[ ! -v seen[$task] ]] || { echo "ERROR: duplicate task $task" >&2; exit 1; }
  seen[$task]=1
  src="$BASE_MASTER/configs/$(printf 'task_%04d.py' "$task")"
  [[ -r "$src" ]] || { echo "ERROR: missing $src" >&2; exit 1; }
  grep -Fq 'predictor_ML_Augmented_BD_grid = ["PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"]' "$src" || {
    echo "ERROR: task $task is not nearest P-median ridge." >&2; exit 1;
  }
  instance="$(sed -nE 's/^[[:space:]]*"(SCFLP|SCMND)": \["([^"]+)"\].*$/\2/p' "$src" | head -n 1)"
  [[ -n "$instance" && -f "$INPUT_DATA_ROOT/$instance" ]] || {
    echo "ERROR: original DATA missing $instance" >&2; exit 1;
  }
  if ((task < 199)); then nonparam=$((task-45)); else nonparam=$((task-108)); fi
  [[ -r "$BASE_MASTER/configs/$(printf 'task_%04d.py' "$nonparam")" ]] || {
    echo "ERROR: matching nonparametric task $nonparam missing." >&2; exit 1;
  }
done

timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="parametric_guard_diag_$timestamp"
root="$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/$run_id"
[[ ! -e "$root" ]] || { echo "ERROR: diagnostic run already exists" >&2; exit 1; }
mkdir -p "$root/configs" "$root/SLURM_LOGS"
cp "$BASE_MASTER/config.py" "$root/config.py"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$root/code_commit.txt"
printf '%s\n' "$BASE_MASTER" > "$root/original_master.txt"
printf '%s\n' "$INPUT_DATA_ROOT" > "$root/input_data_root.txt"
printf 'new_task_id\tmethod\toriginal_task_id\tpair_parametric_task_id\tproblem_class\n' > "$root/task_pairs.tsv"
: > "$root/task_list.txt"

idx=0
for target in "${TARGETS[@]}"; do
  if ((target < 199)); then problem=SCFLP; nonparam=$((target-45))
  else problem=SCMND; nonparam=$((target-108)); fi
  for method in NONPARAM OFF ON; do
    idx=$((idx+1))
    oldtask="$target"; guard=False
    [[ "$method" == NONPARAM ]] && oldtask="$nonparam"
    [[ "$method" == ON ]] && guard=True
    src="$BASE_MASTER/configs/$(printf 'task_%04d.py' "$oldtask")"
    rel="configs/$(printf 'task_%04d.py' "$idx")"
    cp "$src" "$root/$rel"
    cat >> "$root/$rel" <<PYDIAG

# Controlled instrumentation on all paired parametric variants.
parametric_objective_guard = $guard
parametric_diagnostic_probe = True
import sys as _diag_sys
_diag_main = _diag_sys.modules.get("__main__")
if _diag_main is not None:
    setattr(_diag_main, "parametric_objective_guard", parametric_objective_guard)
    setattr(_diag_main, "parametric_diagnostic_probe", parametric_diagnostic_probe)
PYDIAG
    printf '%s|%s\n' "$rel" "$problem" >> "$root/task_list.txt"
    printf '%s\t%s\t%s\t%s\t%s\n' "$idx" "$method" "$oldtask" "$target" "$problem" >> "$root/task_pairs.tsv"
    echo "Prepared $idx : $problem task $target ($method)"
  done
done

result="$(sbatch \
  --account=def-brostami-ab_cpu \
  --cpus-per-task=32 --mem-per-cpu=8G --time=16:00:00 \
  --job-name=labd_guard_diag \
  --array="1-${idx}%${MAX_CONCURRENT}" \
  --output="$root/SLURM_LOGS/slurm-%x-%A_%a.out" \
  --error="$root/SLURM_LOGS/slurm-%x-%A_%a.err" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONFIG_ROOT="$root",RESULTS_ROOT_OVERRIDE="$RESULTS_ROOT_OVERRIDE",LABD_DATA_ROOT="$INPUT_DATA_ROOT",UNIQUE_ID="parametric_guard_diag",TIMESTAMP="$timestamp",PROBLEM_CLASSES="MIXED" \
  "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch")"
job="${result##* }"
[[ "$job" =~ ^[0-9]+$ ]] || { echo "ERROR: bad sbatch response $result" >&2; exit 1; }
printf '%s\n' "$job" > "$root/job_id.txt"
printf '%s\n' "$root" > "$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/latest_parametric_guard_diag.txt"
echo "$result"
echo "Diagnostic job ID: $job"
echo "Diagnostic run root: $root"
echo "Matched tasks: $idx, concurrency: $MAX_CONCURRENT"
echo "Data read from original (unchanged): $INPUT_DATA_ROOT"
