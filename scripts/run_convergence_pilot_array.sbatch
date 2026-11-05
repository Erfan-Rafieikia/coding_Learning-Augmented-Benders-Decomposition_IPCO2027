#!/usr/bin/env bash
# One Slurm array provides a GLOBAL 18-task concurrency ceiling across both
# 1% and 0.1% targets. Resource requests are set by the submitting sbatch call.
set -euo pipefail

: "${PROJECT_ROOT:?Missing PROJECT_ROOT}"
: "${CONFIG_ROOT:?Missing CONFIG_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?This script must be run as a Slurm array task}"

PY="${LABD_PYTHON:-$HOME/micromamba/envs/bd_ml/bin/python}"
"$PY" "$PROJECT_ROOT/scripts/verify_convergence_snapshot.py" "$CONFIG_ROOT"
export LABD_FROZEN_PILOT=1

task="${SLURM_ARRAY_TASK_ID}"
if [[ ! "${task}" =~ ^[0-9]+$ ]] || (( task < 1 || task > 20 )); then
  echo "ERROR: unexpected pilot task ID: ${task}" >&2
  exit 1
fi

if (( task <= 10 )); then
  export LABD_MIP_GAP_TARGET="0.01"
  gap_label="1%"
else
  export LABD_MIP_GAP_TARGET="0.001"
  gap_label="0.1%"
fi

# Independently confirm the selected gap agrees with the prepared manifest.
manifest_row="$(awk -F '\t' -v id="$task" 'NR>1 && $1==id {print; count++} END {if(count!=1)exit 1}' "$CONFIG_ROOT/pilot_manifest.tsv")" || {
  echo "ERROR: missing or duplicated pilot manifest row for task ${task}" >&2
  exit 1
}
manifest_gap="$(printf '%s\n' "$manifest_row" | awk -F '\t' '{print $2}')"
if [[ "$manifest_gap" != "$LABD_MIP_GAP_TARGET" ]]; then
  echo "ERROR: gap discrepancy for task $task: manifest=$manifest_gap, planned=$LABD_MIP_GAP_TARGET" >&2
  exit 1
fi

echo "LABD convergence pilot task $task uses target gap $gap_label ($LABD_MIP_GAP_TARGET)."
echo "All tasks share the same globally throttled Slurm array: at most 18 running."
exec /bin/bash "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch"
