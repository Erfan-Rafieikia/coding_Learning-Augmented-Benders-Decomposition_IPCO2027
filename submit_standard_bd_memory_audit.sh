#!/usr/bin/env bash
set -euo pipefail
# Controlled memory audit: same Standard BD, 32 CPUs, 2GB vs 8GB per CPU.
# Defaults: 3 repetitions x 2 problems x 2 memory allocations = 12 tasks.
# Results and originals are read-only; results saved under a new master root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_ROOT="${REF_ROOT:-$HOME/scratch/Benders_Integration_with_ML/RESULTS/MASTER_RUNS/pmedian_standard_bd_reference_20261019_125011}"
RESULTS="${RESULTS_ROOT_OVERRIDE:-$HOME/scratch/Benders_Integration_with_ML/RESULTS}"
DATA="${LABD_DATA_ROOT:-$HOME/Erfan_projects/Benders_Integration_with_ML/DATA}"
REPEATS="${REPEATS:-3}"
[[ "$(git -C "$PROJECT_ROOT" branch --show-current)" == agent/pmedian-standard-bd-ratio-audit ]] || {
  echo "ERROR: must run in branch agent/pmedian-standard-bd-ratio-audit" >&2; exit 1;
}
[[ "$REPEATS" =~ ^[1-9]$ ]] || { echo "ERROR: REPEATS must be 1..9" >&2; exit 1; }
[[ -d "$DATA" && -r "$REF_ROOT/config.py" && -s "$REF_ROOT/task_list.txt" ]] || {
  echo "ERROR: source exact BD run or read-only DATA missing" >&2; exit 1;
}
grep -Fq 'exact_callback_types_grid = ["int_and_LP"]' "$REF_ROOT/config.py" || {
  echo "ERROR: expected exact callback int_and_LP" >&2; exit 1;
}
for idx in 1 2; do
  source="$REF_ROOT/configs/$(printf 'task_%04d.py' "$idx")"
  [[ -r "$source" ]] || { echo "ERROR: missing $source" >&2; exit 1; }
  grep -Fq 'method_categories_grid = ["Exact"]' "$source" || {
    echo "ERROR: source task is not exact BD" >&2; exit 1;
  }
  instance="$(sed -nE 's/^[[:space:]]*"(SCFLP|SCMND)": \["([^"]+)"\].*$/\2/p' "$source" | head -n 1)"
  [[ -n "$instance" && -r "$DATA/$instance" ]] || {
    echo "ERROR: source data absent for $source" >&2; exit 1;
  }
done
stamp="$(date +%Y%m%d_%H%M%S)"
root="$RESULTS/MASTER_RUNS/standard_bd_memory_audit_$stamp"
[[ ! -e "$root" ]] || { echo "ERROR: output already exists" >&2; exit 1; }
mkdir -p "$root"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$root/code_commit.txt"
sha256sum "$PROJECT_ROOT/src/Problem/master_problem.py" \
          "$PROJECT_ROOT/src/Problem/callbacks.py" \
          "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch" > "$root/source_sha256.txt"
printf '%s\n' "$REF_ROOT" > "$root/source_reference.txt"
printf 'memory_per_cpu\treplication\tpilot\tproblem\ttask\trun_id\n' > "$root/manifest.tsv"
printf 'memory_per_cpu\tjob_id\trun_id\n' > "$root/jobs.tsv"
for memory in 2G 8G; do
  batch="$root/$memory"
  mkdir -p "$batch/configs" "$batch/SLURM_LOGS"
  cp "$REF_ROOT/config.py" "$batch/config.py"
  : > "$batch/task_list.txt"
  task=0
  for ((repeat=1; repeat<=REPEATS; repeat++)); do
    for source_task in 1 2; do
      task=$((task+1))
      if ((source_task==1)); then problem=SCFLP; pilot=46; else problem=SCMND; pilot=199; fi
      file="configs/$(printf 'task_%04d.py' "$task")"
      cp "$REF_ROOT/configs/$(printf 'task_%04d.py' "$source_task")" "$batch/$file"
      run_id="bd_memory_${memory}_$stamp"
      printf '%s|%s\n' "$file" "$problem" >> "$batch/task_list.txt"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$memory" "$repeat" "$pilot" "$problem" "$task" "$run_id" >> "$root/manifest.tsv"
    done
  done
  echo "Prepared $task runs at $memory per CPU"
done
for memory in 2G 8G; do
  batch="$root/$memory"
  run_id="bd_memory_${memory}_$stamp"
  submission="$(sbatch \
    --account=def-brostami-ab_cpu \
    --cpus-per-task=32 --mem-per-cpu="$memory" --time=16:00:00 \
    --job-name="bd_memory_$memory" \
    --array="1-$((2*REPEATS))%1" \
    --output="$batch/SLURM_LOGS/slurm-%x-%A_%a.out" \
    --error="$batch/SLURM_LOGS/slurm-%x-%A_%a.err" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONFIG_ROOT="$batch",RESULTS_ROOT_OVERRIDE="$RESULTS",LABD_DATA_ROOT="$DATA",UNIQUE_ID="bd_memory_$memory",TIMESTAMP="$stamp",PROBLEM_CLASSES=MIXED \
    "$PROJECT_ROOT/my_job.benchmark_memory_audit.sbatch")"
  job_id="${submission##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: submission failed: $submission" >&2; exit 1;
  }
  printf '%s\t%s\t%s\n' "$memory" "$job_id" "$run_id" >> "$root/jobs.tsv"
  echo "Requested $memory: $submission"
done
printf '%s\n' "$root" > "$RESULTS/MASTER_RUNS/latest_standard_bd_memory_audit.txt"
echo "Master run: $root"
echo "Two Slurm arrays, each max concurrent 1. Original experiments untouched."
