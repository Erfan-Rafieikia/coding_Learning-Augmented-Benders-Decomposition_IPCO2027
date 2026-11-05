#!/usr/bin/env bash
# Frozen 1,221-run full sweep; submitter provides resources and fixed gap.
set -euo pipefail
: "${PROJECT_ROOT:?}"
: "${CONFIG_ROOT:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
: "${LABD_MIP_GAP_TARGET:?}"
PY="${LABD_PYTHON:-$HOME/micromamba/envs/bd_ml/bin/python}"
task="$SLURM_ARRAY_TASK_ID"
if [[ ! "$task" =~ ^[0-9]+$ ]] || (( task < 1 || task > 1221 )); then
  echo "Invalid full campaign task ID: $task" >&2
  exit 1
fi
"$PY" "$PROJECT_ROOT/scripts/verify_convergence_full_snapshot.py" \
  "$CONFIG_ROOT" --task-id "$task"
expected_gap="$(cat "$CONFIG_ROOT/target_gap.txt")"
if [[ "$LABD_MIP_GAP_TARGET" != "$expected_gap" ]]; then
  echo "Full campaign gap mismatch" >&2
  exit 1
fi
export LABD_FROZEN_PILOT=1
echo "FULL LABD 1221-run task $task / gap $LABD_MIP_GAP_TARGET"
exec /bin/bash "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch"
