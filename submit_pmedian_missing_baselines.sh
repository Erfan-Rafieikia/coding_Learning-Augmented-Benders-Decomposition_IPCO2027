#!/usr/bin/env bash
set -euo pipefail
# Six new Random Sampling, TREE and CLUSTERING runs on the two pilot instances.
# Same source configs and job infrastructure as nearest-anchor P-median pilots.
# All prior configurations, results and source datasets remain untouched.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="agent/pmedian-standard-bd-ratio-audit"
BASE_MASTER="${BASE_MASTER:-$HOME/scratch/Benders_Integration_with_ML/RESULTS/MASTER_RUNS/pmedian_nearest_20261016_160602}"
RESULTS="${RESULTS_ROOT_OVERRIDE:-$HOME/scratch/Benders_Integration_with_ML/RESULTS}"
DATA="${LABD_DATA_ROOT:-$HOME/Erfan_projects/Benders_Integration_with_ML/DATA}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
[[ "$(git -C "$PROJECT_ROOT" branch --show-current)" == "$BRANCH" ]] || {
  echo "ERROR: expected isolated worktree on $BRANCH" >&2; exit 1;
}
[[ "$MAX_CONCURRENT" =~ ^[1-4]$ ]] || {
  echo "ERROR: MAX_CONCURRENT must be 1..4" >&2; exit 1;
}
[[ -s "$BASE_MASTER/task_list.txt" && -f "$BASE_MASTER/config.py" && -d "$DATA" ]] || {
  echo "ERROR: original master or DATA is absent" >&2; exit 1;
}
for target in 46 199; do
  src="$BASE_MASTER/configs/$(printf 'task_%04d.py' "$target")"
  [[ -r "$src" ]] || { echo "ERROR: missing $src" >&2; exit 1; }
  grep -Fq 'predictor_ML_Augmented_BD_grid = ["PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"]' "$src" || {
    echo "ERROR: unexpected source P-median pilot task $target" >&2; exit 1;
  }
  instance="$(sed -nE 's/^[[:space:]]*"(SCFLP|SCMND)": \["([^"]+)"\].*$/\2/p' "$src" | head -n 1)"
  [[ -n "$instance" && -r "$DATA/$instance" ]] || {
    echo "ERROR: missing or unparseable instance for task $target" >&2; exit 1;
  }
  grep -Fq 'selected_subproblems_ratio_grid = [0.05]' "$src" || { echo "ERROR: unexpected selection ratio" >&2; exit 1; }
  grep -Fq 'ml_callback_types_grid = ["ML_at_both"]' "$src" || { echo "ERROR: unexpected callback" >&2; exit 1; }
done
timestamp="$(date +%Y%m%d_%H%M%S)"
run="pmedian_missing_baselines_$timestamp"
root="$RESULTS/MASTER_RUNS/$run"
[[ ! -e "$root" ]] || { echo "ERROR: result dir exists: $root" >&2; exit 1; }
mkdir -p "$root/configs" "$root/SLURM_LOGS"
cp "$BASE_MASTER/config.py" "$root/config.py"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$root/code_commit.txt"
printf '%s\n' "$BASE_MASTER" > "$root/original_master.txt"
printf '%s\n' "$DATA" > "$root/input_data_root.txt"
printf 'new_task_id\tpair_parametric_task_id\tproblem_class\tmethod\n' > "$root/task_pairs.tsv"
: > "$root/task_list.txt"
idx=0
for target in 46 199; do
  if (( target == 46 )); then problem=SCFLP; else problem=SCMND; fi
  for arm in RANDOM TREE CLUSTERING; do
    idx=$((idx+1))
    src="$BASE_MASTER/configs/$(printf 'task_%04d.py' "$target")"
    dest="configs/$(printf 'task_%04d.py' "$idx")"
    cp "$src" "$root/$dest"
    case "$arm" in
      RANDOM) ml_type="Random_Sampling"; predictor="TREE";;
      TREE) ml_type="ML_Augmented_BD"; predictor="TREE";;
      CLUSTERING) ml_type="ML_Augmented_BD"; predictor="CLUSTERING";;
    esac
    # The source per-task configs already mirror all pilot data, warm-up,
    # feature and callback settings; override ONLY predictor/family.
    cat >> "$root/$dest" <<PYOVERRIDE
# Same exact source instance, formulation, and experimental settings as
# P-median pilot target $target; alternative baseline method only.
method_categories_grid = ["ML"]
ml_method_types_grid = ["$ml_type"]
predictor_ML_Augmented_BD_grid = ["$predictor"]
PYOVERRIDE
    printf '%s|%s\n' "$dest" "$problem" >> "$root/task_list.txt"
    printf '%s\t%s\t%s\t%s\n' "$idx" "$target" "$problem" "$arm" >> "$root/task_pairs.tsv"
    echo "Prepared $idx: $problem / $arm"
  done
done
submission="$(sbatch \
  --account=def-brostami-ab_cpu --cpus-per-task=32 --mem-per-cpu=8G --time=16:00:00 \
  --job-name=labd_baseline6 --array="1-6%$MAX_CONCURRENT" \
  --output="$root/SLURM_LOGS/slurm-%x-%A_%a.out" \
  --error="$root/SLURM_LOGS/slurm-%x-%A_%a.err" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONFIG_ROOT="$root",RESULTS_ROOT_OVERRIDE="$RESULTS",LABD_DATA_ROOT="$DATA",UNIQUE_ID=pmedian_missing_baselines,TIMESTAMP="$timestamp",PROBLEM_CLASSES=MIXED \
  "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch")"
job="${submission##* }"
[[ "$job" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected Slurm response: $submission" >&2; exit 1; }
echo "$job" > "$root/job_id.txt"
echo "$root" > "$RESULTS/MASTER_RUNS/latest_pmedian_missing_baselines.txt"
echo "$submission"
echo "Job ID: $job"
echo "Run root: $root"
echo "Tasks: 6 / max concurrent: $MAX_CONCURRENT"
echo "Original experiments and datasets were not modified."
