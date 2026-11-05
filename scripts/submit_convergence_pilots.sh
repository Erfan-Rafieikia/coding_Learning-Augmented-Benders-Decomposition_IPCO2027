#!/usr/bin/env bash
set -euo pipefail

# Prepare with scripts/prepare_convergence_pilots.py first.
# Intentionally submits NOTHING unless explicitly invoked with --submit.
# One 20-task Slurm array ensures the 18-task global throttle applies
# across BOTH gap targets, rather than independently to two arrays.
ROOT="${1:?Usage: bash scripts/submit_convergence_pilots.sh PILOT_ROOT [--dry-run|--submit]}"
MODE="${2:---dry-run}"
SUBMITTER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(realpath "$ROOT")"
PROJECT_ROOT="$ROOT/source"
PY="${LABD_PYTHON:-$HOME/micromamba/envs/bd_ml/bin/python}"
"$PY" "$SUBMITTER_ROOT/scripts/verify_convergence_snapshot.py" "$ROOT"
MAX_CONCURRENT="${MAX_CONCURRENT:-18}"
[[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] && (( MAX_CONCURRENT >= 1 && MAX_CONCURRENT <= 18 )) || {
  echo "ERROR: MAX_CONCURRENT must be 1..18 (20 Gurobi WLS sessions total; reserve 2)." >&2
  exit 1
}
[[ -f "$ROOT/task_list.txt" && -f "$ROOT/pilot_manifest.tsv" && -f "$ROOT/config.py" ]] || {
  echo "ERROR: invalid prepared pilot root: $ROOT" >&2
  exit 1
}
[[ "$(wc -l < "$ROOT/task_list.txt")" -eq 20 ]] || {
  echo "ERROR: expected precisely 20 prepared pilot tasks." >&2
  exit 1
}
[[ "$ROOT" == *"/MASTER_RUNS/labd_convergence_pilot_"* ]] || {
  echo "ERROR: unexpected pilot root name." >&2
  exit 1
}
DATA="$(cat "$ROOT/source_data_root.txt")"
[[ -d "$DATA" ]] || { echo "ERROR: missing original data root." >&2; exit 1; }
RESULTS_ROOT="$(dirname "$(dirname "$ROOT")")"
TIMESTAMP="${ROOT##*labd_convergence_pilot_}"
[[ "$TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]] || {
  echo "ERROR: timestamp not parseable from pilot root." >&2
  exit 1
}
[[ "$MODE" == "--dry-run" || "$MODE" == "--submit" ]] || {
  echo "ERROR: use --dry-run or --submit." >&2
  exit 1
}
[[ -f "$PROJECT_ROOT/scripts/run_convergence_pilot_array.sbatch" ]] || {
  echo "ERROR: missing per-task gap wrapper." >&2
  exit 1
}
echo "Project: $PROJECT_ROOT"
echo "Pilot:   $ROOT"
echo "Input:   $DATA"
echo "Gap 1:   tasks 1-10 / 1% (0.01)"
echo "Gap 2:   tasks 11-20 / 0.1% (0.001)"
echo "Resources: 16 CPU / 128 GiB per task / 24h Slurm / 3h Gurobi"
echo "ONE Slurm array: --array=1-20%$MAX_CONCURRENT (GLOBAL max $MAX_CONCURRENT running)"
if [[ "$MODE" == "--dry-run" ]]; then
  echo "DRY RUN ONLY; no Slurm jobs submitted."
  exit 0
fi
[[ ! -e "$ROOT/job_id.txt" && ! -e "$ROOT/job_id_gap_1pct.txt" && ! -e "$ROOT/job_id_gap_0p1pct.txt" ]] || {
  echo "ERROR: pilot jobs already recorded; refusing duplicate submission." >&2
  exit 1
}
[[ -r "$HOME/gurobi.lic" ]] || {
  echo "ERROR: missing Gurobi WLS license file." >&2
  exit 1
}
# Atomic claim prevents simultaneous duplicate submissions. On an ambiguous
# scheduler response retain the claim and inspect Slurm before retrying.
mkdir "$ROOT/submission.claim" || { echo "ERROR: this pilot was already claimed for submission." >&2; exit 1; }
cp "$ROOT/source_commit.txt" "$ROOT/code_commit.txt"

submission="$(sbatch \
  --account=def-brostami-ab_cpu \
  --cpus-per-task=16 --mem-per-cpu=8G --time=24:00:00 \
  --job-name=labd_conv_pilot \
  --array="1-20%$MAX_CONCURRENT" \
  --output="$ROOT/SLURM_LOGS/slurm-%x-%A_%a.out" \
  --error="$ROOT/SLURM_LOGS/slurm-%x-%A_%a.err" \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG_ROOT=$ROOT,RESULTS_ROOT_OVERRIDE=$RESULTS_ROOT,LABD_DATA_ROOT=$DATA,UNIQUE_ID=labd_convergence_pilot,TIMESTAMP=$TIMESTAMP,PROBLEM_CLASSES=MIXED" \
  "$PROJECT_ROOT/scripts/run_convergence_pilot_array.sbatch")"
job_id="${submission##* }"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "ERROR: unexpected Slurm response: $submission" >&2
  exit 1
}
printf '%s\n' "$job_id" > "$ROOT/job_id.txt"
printf '%s\n' "$ROOT" > "$RESULTS_ROOT/MASTER_RUNS/latest_convergence_pilot.txt"
echo "$submission"
echo "One globally throttled pilot array submitted. Job ID: $job_id"
echo "Monitor: squeue -j $job_id"
echo "Pilot root: $ROOT"
