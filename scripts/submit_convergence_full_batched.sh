#!/usr/bin/env bash
# Resume a frozen full sweep in disjoint quota-safe Slurm arrays.
# One live LABD full array at a time => <=18 simultaneous optimization tasks.
# Existing rejected monolithic 'submission.claim' is preserved as evidence,
# but is not treated as an active batch. NEVER delete batch_inflight.txt blindly.
set -euo pipefail
ROOT="${1:?Usage: bash scripts/submit_convergence_full_batched.sh FULL_ROOT [--dry-run|--submit]}"
MODE="${2:---dry-run}"
[[ "$MODE" == "--dry-run" || "$MODE" == "--submit" ]] || { echo "Use --dry-run or --submit" >&2; exit 1; }
ROOT="$(realpath "$ROOT")"
[[ "$ROOT" == *"/MASTER_RUNS/labd_convergence_full_"* ]] || { echo "Not a frozen full root" >&2; exit 1; }
PROJECT_ROOT="$ROOT/source"
PY="${LABD_PYTHON:-$HOME/micromamba/envs/bd_ml/bin/python}"
[[ -f "$PROJECT_ROOT/scripts/verify_convergence_full_snapshot.py" ]] || { echo "Frozen verifier missing" >&2; exit 1; }
"$PY" "$PROJECT_ROOT/scripts/verify_convergence_full_snapshot.py" "$ROOT"
[[ -r "$HOME/gurobi.lic" ]] || { echo "Gurobi WLS license missing" >&2; exit 1; }
[[ ! -e "$ROOT/job_id.txt" ]] || { echo "Legacy monolithic job ID exists; inspect Slurm before any batched submission" >&2; exit 1; }
[[ -x "$(command -v sbatch)" ]] || { echo "sbatch unavailable: run on Nibi" >&2; exit 1; }
[[ -x "$(command -v squeue)" && -x "$(command -v sacct)" ]] || { echo "Slurm query tools unavailable" >&2; exit 1; }
MAX_CONCURRENT="${MAX_CONCURRENT:-18}"
[[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] && (( MAX_CONCURRENT >= 1 && MAX_CONCURRENT <= 18 )) || { echo "MAX_CONCURRENT must be 1..18" >&2; exit 1; }
BATCH_SIZE="${LABD_FULL_BATCH_SIZE:-256}"
[[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] && (( BATCH_SIZE >= 1 && BATCH_SIZE <= 256 )) || { echo "LABD_FULL_BATCH_SIZE must be 1..256" >&2; exit 1; }
GAP="$(cat "$ROOT/target_gap.txt")"
[[ "$GAP" == "0.001" || "$GAP" == "0.01" ]] || { echo "Unexpected MIPGap" >&2; exit 1; }
DATA="$(cat "$ROOT/source_data_root.txt")"
RESULTS_ROOT="$(dirname "$(dirname "$ROOT")")"
TIMESTAMP="${ROOT##*labd_convergence_full_}"
[[ "$TIMESTAMP" =~ ^(0p1pct|1pct)_[0-9]{8}_[0-9]{6}$ ]] || { echo "Bad campaign name" >&2; exit 1; }
if [[ "$GAP" == "0.001" ]]; then [[ "$TIMESTAMP" == 0p1pct_* ]] || exit 1; else [[ "$TIMESTAMP" == 1pct_* ]] || exit 1; fi
LEDGER="$ROOT/batch_submissions.tsv"
INFLIGHT="$ROOT/batch_inflight.txt"
LOCK="$ROOT/batch_submission.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Another batch operation or stale batch lock exists: $LOCK" >&2
  exit 1
fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT
[[ ! -e "$INFLIGHT" ]] || { echo "Unresolved previous batch attempt; inspect $INFLIGHT and Slurm before retrying" >&2; exit 1; }
LAST_END=0
LAST_JOB=""
if [[ -f "$LEDGER" ]]; then
  while IFS=$'\t' read -r START END JOB SUBMITTED; do
    [[ -n "$START" ]] || continue
    [[ "$START" =~ ^[0-9]+$ && "$END" =~ ^[0-9]+$ && "$JOB" =~ ^[0-9]+$ ]] || { echo "Corrupt batch ledger" >&2; exit 1; }
    (( START == LAST_END + 1 && END >= START && END <= 1221 )) || { echo "Noncontiguous batch ledger" >&2; exit 1; }
    LAST_END="$END"
    LAST_JOB="$JOB"
  done < "$LEDGER"
fi
if (( LAST_END == 1221 )); then
  echo "All 1221 task IDs have previously been submitted in nonoverlapping batches."
  echo "This does NOT certify success; audit the result workbooks."
  exit 0
fi
if [[ -n "$LAST_JOB" ]]; then
  if ! PREVIOUS_QUEUE="$(squeue -h -j "$LAST_JOB" -o "%i" 2>&1)"; then
    echo "Cannot verify previous batch status: $PREVIOUS_QUEUE" >&2
    exit 1
  fi
  if [[ -n "$PREVIOUS_QUEUE" ]]; then
    echo "Batch job $LAST_JOB is still queued/running; not submitting another overlapping batch."
    echo "$PREVIOUS_QUEUE" | head -n 4
    exit 0
  fi
  if ! HISTORY="$(sacct -n -X -P -j "$LAST_JOB" --format=JobIDRaw,State 2>&1)"; then
    echo "Unable to confirm previous batch accounting: $HISTORY" >&2
    exit 1
  fi
  if ! printf "%s\n" "$HISTORY" | grep -Eq "^${LAST_JOB}([_|.]|[|])"; then
    echo "Previous job not found in Slurm accounting; refusing possible duplicate." >&2
    exit 1
  fi
fi
# Account-wide running-limit guard: do not overlap another LABD full campaign.
if ! ACTIVE="$(squeue -h -u "$USER" -n labd_conv_full -o "%i" 2>&1)"; then
  echo "Cannot inspect active full arrays: $ACTIVE" >&2
  exit 1
fi
if [[ -n "$ACTIVE" ]]; then
  echo "Another full LABD array is in Slurm; refusing overlapping launch:"
  echo "$ACTIVE" | head -n 5
  exit 0
fi
NEXT=$((LAST_END + 1))
LIMIT=$(( NEXT + BATCH_SIZE - 1 ))
(( LIMIT <= 1221 )) || LIMIT=1221
echo "Campaign: $ROOT"
echo "Gap $GAP; previously submitted IDs 1..$LAST_END; next IDs $NEXT..$LIMIT"
echo "One array at a time; at most $MAX_CONCURRENT running (16 CPU, 128 GiB per task)"
if [[ "$MODE" == "--dry-run" ]]; then
  echo "DRY RUN ONLY; no Slurm submission."
  exit 0
fi
# Marker is written BEFORE sbatch. If response is ambiguous, leave it in place.
SIZE=$BATCH_SIZE
while (( SIZE >= 1 )); do
  END=$((NEXT + SIZE - 1))
  (( END <= 1221 )) || END=1221
  printf "start=%s end=%s prepared=%s\n" "$NEXT" "$END" "$(date -u +%FT%TZ)" > "$INFLIGHT"
  echo "Requesting Slurm array ${NEXT}-${END}%${MAX_CONCURRENT}..."
  if RESPONSE="$(sbatch \
    --account=def-brostami-ab_cpu \
    --cpus-per-task=16 --mem-per-cpu=8G --time=24:00:00 \
    --job-name=labd_conv_full \
    --array="${NEXT}-${END}%${MAX_CONCURRENT}" \
    --output="$ROOT/SLURM_LOGS/slurm-%x-%A_%a.out" \
    --error="$ROOT/SLURM_LOGS/slurm-%x-%A_%a.err" \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG_ROOT=$ROOT,RESULTS_ROOT_OVERRIDE=$RESULTS_ROOT,LABD_DATA_ROOT=$DATA,UNIQUE_ID=labd_convergence_full,TIMESTAMP=$TIMESTAMP,PROBLEM_CLASSES=MIXED,LABD_MIP_GAP_TARGET=$GAP" \
    "$PROJECT_ROOT/scripts/run_convergence_full_array.sbatch" 2>&1)"; then
    echo "$RESPONSE"
    JOB_ID="$(printf "%s\n" "$RESPONSE" | sed -n "s/^Submitted batch job \([0-9][0-9]*\)$/\1/p")"
    if [[ -z "$JOB_ID" ]]; then
      echo "Ambiguous successful sbatch response; KEEP $INFLIGHT and inspect Slurm manually." >&2
      exit 1
    fi
    printf "%s\t%s\t%s\t%s\n" "$NEXT" "$END" "$JOB_ID" "$(date -u +%FT%TZ)" >> "$LEDGER"
    rm "$INFLIGHT"
    echo "SUBMITTED: Job $JOB_ID, IDs $NEXT..$END / 1221 (not duplicated)"
    echo "Track: squeue -j $JOB_ID"
    echo "Ledger: $LEDGER"
    exit 0
  else
    echo "$RESPONSE" >&2
    if grep -Eq "AssocMaxSubmitJobLimit|job submit limit|Job violates accounting/QOS policy" <<< "$RESPONSE"; then
      if (( SIZE == 1 )); then
        rm "$INFLIGHT"
        echo "Even a single task hit Slurm submission quota; retry after quota frees." >&2
        exit 1
      fi
      SIZE=$(( (SIZE + 1) / 2 ))
      echo "Definite quota rejection; trying a smaller batch of $SIZE tasks."
      rm "$INFLIGHT"
      continue
    fi
    echo "Unclassified submission failure: keeping $INFLIGHT to avoid duplicates." >&2
    exit 1
  fi
done
echo "No safe batch size available" >&2
exit 1
