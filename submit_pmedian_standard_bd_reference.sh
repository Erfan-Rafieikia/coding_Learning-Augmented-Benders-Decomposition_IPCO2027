#!/usr/bin/env bash
set -euo pipefail
# Exact/Standard-BD reference for the two repair pilots. No source experiments
# or source datasets are changed. Each run has a unique master/results root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="agent/pmedian-standard-bd-ratio-audit"
BASE_MASTER="${BASE_MASTER:-$HOME/scratch/Benders_Integration_with_ML/RESULTS/MASTER_RUNS/pmedian_nearest_20261016_160602}"
RESULTS="${RESULTS_ROOT_OVERRIDE:-$HOME/scratch/Benders_Integration_with_ML/RESULTS}"
DATA="${LABD_DATA_ROOT:-$HOME/Erfan_projects/Benders_Integration_with_ML/DATA}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
PILOT_TASKS="${PILOT_TASKS:-46 199}"
[[ "$(git -C "$PROJECT_ROOT" branch --show-current)" == "$BRANCH" ]] || {
  echo "ERROR: run in branch $BRANCH" >&2; exit 1;
}
[[ "$MAX_CONCURRENT" =~ ^[1-4]$ ]] || {
  echo "ERROR: MAX_CONCURRENT must be 1..4" >&2; exit 1;
}
[[ -s "$BASE_MASTER/task_list.txt" && -f "$BASE_MASTER/config.py" && -d "$DATA" ]] || {
  echo "ERROR: original experiment master or input DATA absent" >&2; exit 1;
}
read -r -a targets <<< "$PILOT_TASKS"
(( ${#targets[@]} > 0 )) || { echo "ERROR: no targets" >&2; exit 1; }
declare -A seen=()
for target in "${targets[@]}"; do
  case "$target" in 46|199) ;; *) echo "ERROR: only 46 and 199 supported in initial paired audit" >&2; exit 1;; esac
  [[ ! -v seen[$target] ]] || { echo "ERROR: duplicate target" >&2; exit 1; }
  seen[$target]=1
  source="$BASE_MASTER/configs/$(printf 'task_%04d.py' "$target")"
  [[ -r "$source" ]] || { echo "ERROR: missing $source" >&2; exit 1; }
  grep -Fq 'predictor_ML_Augmented_BD_grid = ["PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"]' "$source" || {
    echo "ERROR: source task $target not the nearest-anchor parametric experiment" >&2; exit 1;
  }
  instance="$(sed -nE 's/^[[:space:]]*"(SCFLP|SCMND)": \["([^"]+)"\].*$/\2/p' "$source" | head -n 1)"
  [[ -n "$instance" && -r "$DATA/$instance" ]] || {
    echo "ERROR: source input absent or unparseable for task $target" >&2; exit 1;
  }
done
timestamp="$(date +%Y%m%d_%H%M%S)"
run="pmedian_standard_bd_reference_$timestamp"
root="$RESULTS/MASTER_RUNS/$run"
[[ ! -e "$root" ]] || { echo "ERROR: output already exists: $root" >&2; exit 1; }
mkdir -p "$root/configs" "$root/SLURM_LOGS"
cp "$BASE_MASTER/config.py" "$root/config.py"
# _apply_runtime_config_override in src/main.py reads method_categories_grid
# from per-task overrides; the exact callback is imported from config.py.
printf '\n# Reference baseline: same enhanced exact integer+LP Benders callbacks.\nexact_callback_types_grid = ["int_and_LP"]\n' >> "$root/config.py"
printf '%s\n' "$BASE_MASTER" > "$root/original_master.txt"
printf '%s\n' "$DATA" > "$root/input_data_root.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$root/code_commit.txt"
printf 'new_task_id\tpair_parametric_task_id\tproblem_class\n' > "$root/task_pairs.tsv"
: > "$root/task_list.txt"
idx=0
for target in "${targets[@]}"; do
  idx=$((idx+1))
  if (( target < 199 )); then problem=SCFLP; else problem=SCMND; fi
  src="$BASE_MASTER/configs/$(printf 'task_%04d.py' "$target")"
  dest="configs/$(printf 'task_%04d.py' "$idx")"
  cp "$src" "$root/$dest"
  cat >> "$root/$dest" <<'PYEXACT'
# Change only the solution method; retain exact same data, instance,
# stochastic capacity choice, SCMND formulation and stopping criteria.
method_categories_grid = ["Exact"]
PYEXACT
  printf '%s|%s\n' "$dest" "$problem" >> "$root/task_list.txt"
  printf '%s\t%s\t%s\n' "$idx" "$target" "$problem" >> "$root/task_pairs.tsv"
  echo "Prepared Standard BD $idx: original target $target / $problem"
done
submission="$(sbatch \
  --account=def-brostami-ab_cpu --cpus-per-task=32 --mem-per-cpu=8G --time=16:00:00 \
  --job-name=labd_bd_reference --array="1-$idx%$MAX_CONCURRENT" \
  --output="$root/SLURM_LOGS/slurm-%x-%A_%a.out" \
  --error="$root/SLURM_LOGS/slurm-%x-%A_%a.err" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONFIG_ROOT="$root",RESULTS_ROOT_OVERRIDE="$RESULTS",LABD_DATA_ROOT="$DATA",UNIQUE_ID=pmedian_standard_bd_reference,TIMESTAMP="$timestamp",PROBLEM_CLASSES=MIXED \
  "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch")"
job="${submission##* }"
[[ "$job" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch output: $submission" >&2; exit 1; }
printf '%s\n' "$job" > "$root/job_id.txt"
printf '%s\n' "$root" > "$RESULTS/MASTER_RUNS/latest_pmedian_standard_bd_reference.txt"
echo "$submission"
echo "Job ID: $job"
echo "Run root: $root"
echo "Standard BD tasks: $idx / max concurrent: $MAX_CONCURRENT"
echo "Original input DATA and saved experiments untouched."
