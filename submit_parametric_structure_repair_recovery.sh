#!/usr/bin/env bash
set -euo pipefail

# Retry ONLY original failed STRUCT arms (SCFLP task 3 and SCMND task 6)
# in a fresh immutable result root. Keep completed original comparator runs.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="agent/parametric-structure-aware-repair"
RESULTS_ROOT_OVERRIDE="${RESULTS_ROOT_OVERRIDE:-$HOME/scratch/Benders_Integration_with_ML/RESULTS}"
ORIGINAL_ROOT="${ORIGINAL_ROOT:-$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/parametric_structure_repair_20261018_200802}"
INPUT_DATA_ROOT="${LABD_DATA_ROOT:-$HOME/Erfan_projects/Benders_Integration_with_ML/DATA}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"

[[ "$(git -C "$PROJECT_ROOT" branch --show-current)" == "$BRANCH" ]] || {
    echo "ERROR: not on $BRANCH" >&2; exit 1;
}
[[ "$MAX_CONCURRENT" =~ ^[1-2]$ ]] || {
    echo "ERROR: MAX_CONCURRENT must be 1 or 2" >&2; exit 1;
}
[[ -r "$ORIGINAL_ROOT/code_commit.txt" && -r "$ORIGINAL_ROOT/task_pairs.tsv" && -r "$ORIGINAL_ROOT/config.py" ]] || {
    echo "ERROR: original run metadata not found" >&2; exit 1;
}
[[ -d "$INPUT_DATA_ROOT" ]] || { echo "ERROR: original DATA directory unavailable" >&2; exit 1; }
SOURCE_SHA="$(cat "$ORIGINAL_ROOT/code_commit.txt")"
CURRENT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid source commit SHA" >&2; exit 1; }
git -C "$PROJECT_ROOT" merge-base --is-ancestor "$SOURCE_SHA" "$CURRENT_SHA" || {
    echo "ERROR: current branch does not descend from original pilot revision" >&2; exit 1;
}
# Only validator, diagnostics/testing/launch scripts may have changed.
# A change to the predictor, solver, first-stage logic or data invalidates
# reuse of comparator runs; in that case rerun all three arms instead.
while IFS= read -r changed; do
    [[ -z "$changed" ]] && continue
    case "$changed" in
        src/utils/config_utils.py|tests/*|scripts/compare_parametric_structure_repair.py|submit_parametric_structure_repair_recovery.sh)
            ;;
        *)
            echo "ERROR: numerical or unrelated code changed: $changed" >&2
            echo "Rerun a full matched pilot instead of mixing comparisons." >&2
            exit 1
            ;;
    esac
done < <(git -C "$PROJECT_ROOT" diff --name-only "$SOURCE_SHA" "$CURRENT_SHA")

for old in 3 6; do
    oldconfig="$ORIGINAL_ROOT/configs/$(printf 'task_%04d.py' "$old")"
    [[ -r "$oldconfig" ]] || { echo "ERROR: missing source override $oldconfig" >&2; exit 1; }
    grep -Fq 'parametric_repair_mode_grid = ["STRUCTURE_AWARE"]' "$oldconfig" || {
        echo "ERROR: original task $old was not structure-aware" >&2; exit 1;
    }
    [[ "$(awk -F '\t' -v n="$old" 'NR>1 && $1==n {print $2}' "$ORIGINAL_ROOT/task_pairs.tsv")" == STRUCT ]] || {
        echo "ERROR: source task $old does not map to STRUCT arm" >&2; exit 1;
    }
done

timestamp="$(date +%Y%m%d_%H%M%S)"
run="parametric_structure_repair_recovery_$timestamp"
root="$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/$run"
[[ ! -e "$root" ]] || { echo "ERROR: retry run already exists" >&2; exit 1; }
mkdir -p "$root/configs" "$root/SLURM_LOGS"
cp "$ORIGINAL_ROOT/config.py" "$root/config.py"
printf '%s\n' "$CURRENT_SHA" > "$root/code_commit.txt"
printf '%s\n' "$ORIGINAL_ROOT" > "$root/comparison_original_root.txt"
printf '%s\n' "$SOURCE_SHA" > "$root/comparison_original_commit.txt"
printf '%s\n' "$INPUT_DATA_ROOT" > "$root/input_data_root.txt"
printf 'new_task_id\tmethod\toriginal_task_id\tpair_parametric_task_id\tproblem_class\n' > "$root/task_pairs.tsv"
: > "$root/task_list.txt"

for new in 1 2; do
    if (( new == 1 )); then old=3; problem=SCFLP; target=46
    else old=6; problem=SCMND; target=199; fi
    dest="configs/$(printf 'task_%04d.py' "$new")"
    cp "$ORIGINAL_ROOT/configs/$(printf 'task_%04d.py' "$old")" "$root/$dest"
    printf '%s|%s\n' "$dest" "$problem" >> "$root/task_list.txt"
    printf '%s\tSTRUCT\t%s\t%s\t%s\n' "$new" "$target" "$target" "$problem" >> "$root/task_pairs.tsv"
    echo "Retry task $new: original failed arm $old / $problem / STRUCT"
done

submission="$(sbatch \
  --account=def-brostami-ab_cpu --cpus-per-task=32 --mem-per-cpu=8G --time=16:00:00 \
  --job-name=labd_struct_retry \
  --array="1-2%${MAX_CONCURRENT}" \
  --output="$root/SLURM_LOGS/slurm-%x-%A_%a.out" \
  --error="$root/SLURM_LOGS/slurm-%x-%A_%a.err" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONFIG_ROOT="$root",RESULTS_ROOT_OVERRIDE="$RESULTS_ROOT_OVERRIDE",LABD_DATA_ROOT="$INPUT_DATA_ROOT",UNIQUE_ID="parametric_structure_repair_recovery",TIMESTAMP="$timestamp",PROBLEM_CLASSES="MIXED" \
  "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch")"
job="${submission##* }"
[[ "$job" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch response $submission" >&2; exit 1; }
printf '%s\n' "$job" > "$root/job_id.txt"
printf '%s\n' "$root" > "$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/latest_parametric_structure_repair_recovery.txt"
echo "$submission"
echo "Job ID: $job"
echo "Recovery run root: $root"
echo "Original comparator root: $ORIGINAL_ROOT"
echo "Tasks: 2 / max concurrent: $MAX_CONCURRENT"
echo "After completion, compare with:"
echo "  python scripts/compare_parametric_structure_repair.py '$ORIGINAL_ROOT' '$root'"
