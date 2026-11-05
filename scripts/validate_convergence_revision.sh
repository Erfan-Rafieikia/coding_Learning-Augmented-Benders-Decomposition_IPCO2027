#!/usr/bin/env bash
# Focused checks only. Does not prepare, submit or modify experiments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${LABD_PYTHON:-$HOME/micromamba/envs/bd_ml/bin/python}"
"$PY" -m compileall -q src scripts tests
bash -n scripts/submit_convergence_pilots.sh scripts/run_convergence_pilot_array.sbatch my_job.pmedian_nearest.alliance.sbatch
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
"$PY" -m pytest -q \
  tests/test_convergence_diagnostics.py tests/test_exclusive_timing_schema.py \
  tests/test_timing_attribution.py tests/test_solver_event_trace.py tests/test_cut_construction_timing.py tests/test_convergence_snapshot.py tests/test_final_reporting.py \
  tests/test_parametric_predictors.py tests/test_parametric_objective_guard.py \
  tests/test_parametric_guard_zero_fastpath.py tests/test_parametric_structure_aware_repair.py \
  tests/test_pmedian_nearest_transfer.py tests/test_scmnd_formulation_switch.py \
  tests/test_feature_generation_manuscript.py
# These 60 tiny solves cover the 52-case method matrix, two hard-capacity checks,
# and six workbook/CSV exports.
# They exercise actual timers but do not replace the matched Nibi benchmark pilot.
LABD_RUN_SOLVER_SMOKE=1 "$PY" -m pytest -q tests/test_timing_solver_smoke.py
printf 'Validation passed. Source HEAD: '
git rev-parse HEAD

if ! git diff --quiet HEAD; then
  echo "Tracked worktree changes were tested; commit them before pilot preparation."
fi
