#!/usr/bin/env bash
# Submit an IPCO root made by prepare_ipco_root.py, with the SAME Slurm resources,
# Gurobi license and per-task job script as the journal's campaign 1
# (16 CPUs, 8G per CPU, account def-brostami-ab_cpu). Wall time: TIME_LIMIT env var,
# default 24:00:00 as in campaign 1, e.g. TIME_LIMIT=3-00:00:00.
# Usage: bash scripts/ipco/submit_ipco_root.sh IPCO_ROOT [--dry-run|--submit]
set -euo pipefail
ROOT="$(realpath "${1:?Usage: submit_ipco_root.sh IPCO_ROOT [--dry-run|--submit]}")"
MODE="${2:---dry-run}"
[[ "$MODE" == "--dry-run" || "$MODE" == "--submit" ]] || { echo "Invalid mode" >&2; exit 1; }
[[ "$ROOT" == *"/MASTER_RUNS/ipco_"* ]] || { echo "Not an IPCO root" >&2; exit 1; }
PROJECT_ROOT="$ROOT/source"
[[ -f "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch" ]] || { echo "Missing job script in source" >&2; exit 1; }
[[ -r "$HOME/gurobi.lic" ]] || { echo "Missing personal Gurobi license" >&2; exit 1; }
[[ ! -e "$ROOT/job_id.txt" ]] || { echo "Already submitted: $(cat "$ROOT/job_id.txt")" >&2; exit 1; }
N="$(wc -l < "$ROOT/task_list.txt")"
GAP="$(cat "$ROOT/target_gap.txt")"
DATA="$(cat "$ROOT/source_data_root.txt")"
RESULTS_ROOT="$(dirname "$(dirname "$ROOT")")"
TIMESTAMP="${ROOT##*/MASTER_RUNS/}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
# Slurm wall time per task. Campaign 1 used 24:00:00 (Gurobi itself stops at 3 h).
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
echo "IPCO ROOT: $ROOT"
echo "Slurm wall time per task: $TIME_LIMIT"
echo "Tasks: 1-$N | gap $GAP | data $DATA | code $(cat "$ROOT/source_commit.txt")"
echo "Results: $RESULTS_ROOT/SCFLP_class/RESULTS/labd_ipco_${TIMESTAMP}/"
if [[ "$MODE" == "--dry-run" ]]; then echo "DRY RUN; NO JOBS SUBMITTED."; exit 0; fi
out="$(sbatch \
  --account=def-brostami-ab_cpu \
  --cpus-per-task=16 --mem-per-cpu=8G --time="$TIME_LIMIT" \
  --job-name=labd_ipco \
  --array="1-$N%$MAX_CONCURRENT" \
  --output="$ROOT/SLURM_LOGS/slurm-%x-%A_%a.out" \
  --error="$ROOT/SLURM_LOGS/slurm-%x-%A_%a.err" \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG_ROOT=$ROOT,RESULTS_ROOT_OVERRIDE=$RESULTS_ROOT,LABD_DATA_ROOT=$DATA,UNIQUE_ID=labd_ipco,TIMESTAMP=$TIMESTAMP,PROBLEM_CLASSES=SCFLP,LABD_MIP_GAP_TARGET=$GAP,LABD_FROZEN_PILOT=1" \
  "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch")"
JOB_ID="${out##* }"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Ambiguous sbatch response: $out" >&2; exit 1; }
echo "$JOB_ID" > "$ROOT/job_id.txt"
echo "SUBMITTED job $JOB_ID ($N tasks). Monitor: squeue -j $JOB_ID"
