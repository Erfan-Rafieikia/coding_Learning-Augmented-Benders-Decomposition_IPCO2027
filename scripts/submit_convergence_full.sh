#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?Usage: bash scripts/submit_convergence_full.sh FULL_ROOT [--dry-run|--submit]}"
MODE="${2:---dry-run}"
[[ "$MODE" == "--dry-run" || "$MODE" == "--submit" ]] || { echo "Invalid mode" >&2; exit 1; }
ROOT="$(realpath "$ROOT")"
[[ "$ROOT" == *"/MASTER_RUNS/labd_convergence_full_"* ]] || { echo "Unexpected full root" >&2; exit 1; }
PROJECT_ROOT="$ROOT/source"
PY="${LABD_PYTHON:-$HOME/micromamba/envs/bd_ml/bin/python}"
"$PY" "$PROJECT_ROOT/scripts/verify_convergence_full_snapshot.py" "$ROOT"
[[ -r "$HOME/gurobi.lic" ]] || { echo "Missing personal Gurobi license" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/scripts/run_convergence_full_array.sbatch" ]] || { echo "Missing full array wrapper" >&2; exit 1; }
[[ ! -e "$ROOT/job_id.txt" && ! -e "$ROOT/submission.claim" ]] || { echo "Already submitted or claimed; stop." >&2; exit 1; }
MAX_CONCURRENT="${MAX_CONCURRENT:-18}"
[[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] && (( MAX_CONCURRENT >= 1 && MAX_CONCURRENT <= 18 )) || { echo "MAX_CONCURRENT must be 1..18" >&2; exit 1; }
GAP="$(cat "$ROOT/target_gap.txt")"
[[ "$GAP" == "0.001" || "$GAP" == "0.01" ]] || { echo "Unsupported Gurobi target" >&2; exit 1; }
DATA="$(cat "$ROOT/source_data_root.txt")"
RESULTS_ROOT="$(dirname "$(dirname "$ROOT")")"
TIMESTAMP="${ROOT##*labd_convergence_full_}"
[[ "$TIMESTAMP" =~ ^(0p1pct|1pct)_[0-9]{8}_[0-9]{6}$ ]] || { echo "Bad run identifier" >&2; exit 1; }
if [[ "$GAP" == "0.001" ]]; then [[ "$TIMESTAMP" == 0p1pct_* ]] || exit 1; else [[ "$TIMESTAMP" == 1pct_* ]] || exit 1; fi
echo "FROZEN FULL RUN: $ROOT"
echo "Gap: $GAP; Tasks: 1-1221; max running: $MAX_CONCURRENT"
echo "16 CPUs/task, 128 GiB/task, 24h Slurm, 3h Gurobi"
echo "ALL original 1221 methods/repair variants preserved (including 297 existing STRUCTURE_AWARE runs)."
echo "SCMND configuration copies use hard MANUSCRIPT_ENVELOPE."
if [[ "$MODE" == "--dry-run" ]]; then
  echo "DRY RUN VERIFIED; NO JOBS SUBMITTED."
  exit 0
fi
mkdir "$ROOT/submission.claim" || { echo "Submission already claimed" >&2; exit 1; }
cp "$ROOT/source_commit.txt" "$ROOT/code_commit.txt"
out="$(sbatch \
  --account=def-brostami-ab_cpu \
  --cpus-per-task=16 --mem-per-cpu=8G --time=24:00:00 \
  --job-name=labd_conv_full \
  --array="1-1221%$MAX_CONCURRENT" \
  --output="$ROOT/SLURM_LOGS/slurm-%x-%A_%a.out" \
  --error="$ROOT/SLURM_LOGS/slurm-%x-%A_%a.err" \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG_ROOT=$ROOT,RESULTS_ROOT_OVERRIDE=$RESULTS_ROOT,LABD_DATA_ROOT=$DATA,UNIQUE_ID=labd_convergence_full,TIMESTAMP=$TIMESTAMP,PROBLEM_CLASSES=MIXED,LABD_MIP_GAP_TARGET=$GAP" \
  "$PROJECT_ROOT/scripts/run_convergence_full_array.sbatch")"
JOB_ID="${out##* }"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Ambiguous sbatch response: $out" >&2; exit 1; }
printf "%s\n" "$JOB_ID" > "$ROOT/job_id.txt"
echo "$out"
echo "Full sweep SUBMITTED: $JOB_ID"
echo "Monitor: squeue -j $JOB_ID"
echo "Root: $ROOT"
