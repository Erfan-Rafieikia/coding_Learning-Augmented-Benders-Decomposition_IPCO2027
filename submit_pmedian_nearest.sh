#!/bin/bash
set -euo pipefail

# =============================================================================
# ADDITIVE SCENARIO-SPECIFIC P-MEDIAN EXPERIMENTS
# Two new predictors only; old P-median methods and experiments are untouched.
# Run modes: prepare -> pilot -> verify -> main -> verify -> appendix.
# Three-hour Gurobi time limit remains unchanged; 16h Slurm postprocessing room.
# =============================================================================

ACCOUNT="def-brostami-ab_cpu"
CPUS_PER_TASK=32
MEM_PER_CPU="8G"
WALLTIME="16:00:00"
# Academic WLS license: 20 approved concurrent sessions; default to 12.
MAX_CONCURRENT="${MAX_CONCURRENT:-12}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SRC_DIR="$PROJECT_ROOT/src"
RESULTS_ROOT_OVERRIDE="${RESULTS_ROOT_OVERRIDE:-/home/rafieiki/scratch/Benders_Integration_with_ML/RESULTS}"
mkdir -p "$RESULTS_ROOT_OVERRIDE"

METHODS=(
  "PMEDIAN_NEAREST_TRANSFER"
  "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"
)

RATIOS=("0.05" "0.15" "0.25")

SCFLP_INSTANCES=(
  "SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p4_stochastic.txt"
  "SCFLP/scenarios_s300_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p4_stochastic.txt"
  "SCFLP/scenarios_s500_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p4_stochastic.txt"
  "SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p5_stochastic.txt"
  "SCFLP/scenarios_s300_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p5_stochastic.txt"
  "SCFLP/scenarios_s500_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p5_stochastic.txt"
  "SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p10_stochastic.txt"
  "SCFLP/scenarios_s300_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p10_stochastic.txt"
  "SCFLP/scenarios_s500_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p10_stochastic.txt"
  "SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p20_stochastic.txt"
  "SCFLP/scenarios_s300_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p20_stochastic.txt"
  "SCFLP/scenarios_s500_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p20_stochastic.txt"
  "SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p50_stochastic.txt"
  "SCFLP/scenarios_s300_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p50_stochastic.txt"
  "SCFLP/scenarios_s500_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p50_stochastic.txt"
)

SCMND_INSTANCES=(
  "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r01.1.stoch.json"
  "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r01.1.stoch.json"
  "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r01.1.stoch.json"
  "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r01.2.stoch.json"
  "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r01.2.stoch.json"
  "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r01.2.stoch.json"
  "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r02.1.stoch.json"
  "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r02.1.stoch.json"
  "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r02.1.stoch.json"
  "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r02.2.stoch.json"
  "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r02.2.stoch.json"
  "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r02.2.stoch.json"
  "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r04.1.stoch.json"
  "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r04.1.stoch.json"
  "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r04.1.stoch.json"
  "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r04.2.stoch.json"
  "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r04.2.stoch.json"
  "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r04.2.stoch.json"
)

SCMND_FORMULATIONS=(
  "MANUSCRIPT_ENVELOPE"
  "CURRENT_SOFT_CAPACITY"
)


# This file deliberately does not call the legacy submit.sh.
#
# Actual task ordering: SCFLP is generated for BOTH methods first, followed
# by SCMND for each method/formulation. Explicitly verify all 306 configs.
validate_manifest() {
  local root="$1" from="$2" to="$3" problem="$4" method="$5" formulation="${6:-}"
  local idx cfg
  for ((idx=from; idx<=to; idx++)); do
    cfg="$(printf '%s/configs/task_%04d.py' "$root" "$idx")"
    if [[ ! -f "$cfg" ]]; then
      echo "ERROR: missing task configuration $cfg" >&2
      return 1
    fi
    if ! grep -Fqx "predictor_ML_Augmented_BD_grid = [\"$method\"]" "$cfg"; then
      echo "ERROR: task $idx has incorrect method; expected $method" >&2
      return 1
    fi
    if ! grep -Fq "    \"$problem\": [" "$cfg"; then
      echo "ERROR: task $idx has incorrect problem class; expected $problem" >&2
      return 1
    fi
    if [[ -n "$formulation" ]]; then
      if ! grep -Fqx "scmnd_formulation_grid = [\"$formulation\"]" "$cfg"; then
        echo "ERROR: task $idx has incorrect SCMND formulation; expected $formulation" >&2
        return 1
      fi
    fi
  done
}
validate_all_manifest_groups() {
  local root="$1"
  validate_manifest "$root" 1 45 SCFLP PMEDIAN_NEAREST_TRANSFER
  validate_manifest "$root" 46 90 SCFLP PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR
  validate_manifest "$root" 91 144 SCMND PMEDIAN_NEAREST_TRANSFER MANUSCRIPT_ENVELOPE
  validate_manifest "$root" 145 198 SCMND PMEDIAN_NEAREST_TRANSFER CURRENT_SOFT_CAPACITY
  validate_manifest "$root" 199 252 SCMND PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR MANUSCRIPT_ENVELOPE
  validate_manifest "$root" 253 306 SCMND PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR CURRENT_SOFT_CAPACITY
}
EXPECTED_BRANCH="agent/pmedian-scenario-specific-transfer"
CURRENT_BRANCH="$(git -C "$PROJECT_ROOT" branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: switch to $EXPECTED_BRANCH first (current: $CURRENT_BRANCH)" >&2
  exit 1
fi
MODE="${1:-prepare}"
LATEST_MASTER_FILE="$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/latest_pmedian_nearest_master.txt"

if [[ "$MODE" != "prepare" ]]; then
  MASTER_ROOT="${2:-}"
  if [[ -z "$MASTER_ROOT" ]]; then
    if [[ ! -r "$LATEST_MASTER_FILE" ]]; then
      echo "ERROR: no prepared master. First run: bash submit_pmedian_nearest.sh prepare" >&2
      exit 1
    fi
    MASTER_ROOT="$(cat "$LATEST_MASTER_FILE")"
  fi
  if [[ ! -r "$MASTER_ROOT/task_list.txt" || ! -r "$MASTER_ROOT/code_commit.txt" ]]; then
    echo "ERROR: missing task list or code_commit snapshot in $MASTER_ROOT" >&2
    exit 1
  fi
  TASK_COUNT="$(wc -l < "$MASTER_ROOT/task_list.txt")"
  if [[ "$TASK_COUNT" -ne 306 ]]; then
    echo "ERROR: expected 306 new-method tasks; found $TASK_COUNT" >&2
    exit 1
  fi
  if [[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" != "$(cat "$MASTER_ROOT/code_commit.txt")" ]]; then
    echo "ERROR: local source commit differs from prepared experiment snapshot." >&2
    exit 1
  fi
  # Validate the WHOLE task manifest, not two assumed array indexes.
  # This guard prevents mistakenly sending appendix formulations in the main array.
  validate_all_manifest_groups "$MASTER_ROOT"
  case "$MODE" in
    pilot)
      ARRAY_SPEC="1,46,91,199%4"
      ;;
    main)
      if [[ ! -f "$MASTER_ROOT/pilot_verified.txt" ]]; then
        echo "ERROR: verify the four main-paper pilots before submitting the 194 remaining main-paper runs." >&2
        echo "Run: python scripts/verify_pmedian_nearest_pilots.py '$MASTER_ROOT'" >&2
        exit 1
      fi
      ARRAY_SPEC="2-45,47-90,92-144,200-252%$MAX_CONCURRENT"
      ;;
    appendix)
      if [[ ! -f "$MASTER_ROOT/main_verified.txt" ]]; then
        echo "ERROR: main-paper results must be verified before starting appendix runs." >&2
        echo "Run: python scripts/verify_pmedian_nearest_main.py '$MASTER_ROOT'" >&2
        exit 1
      fi
      ARRAY_SPEC="145-198,253-306%$MAX_CONCURRENT"
      ;;
    *)
      echo "Usage: bash submit_pmedian_nearest.sh {prepare|pilot|main|appendix} [master_root]" >&2
      exit 2
      ;;
  esac
  if [[ -f "$MASTER_ROOT/job_id_$MODE.txt" ]]; then
    echo "ERROR: $MODE has already been submitted as job $(cat "$MASTER_ROOT/job_id_$MODE.txt")." >&2
    echo "Refusing to submit duplicate work; inspect status and results first." >&2
    exit 1
  fi
  RUN_TIMESTAMP="${MASTER_ROOT##*/pmedian_nearest_}"
  if [[ ! "$RUN_TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
    echo "ERROR: unexpected prepared run directory name: $MASTER_ROOT" >&2
    exit 1
  fi
  SLURM_LOG_DIR="$MASTER_ROOT/SLURM_LOGS"
  submission="$(sbatch \
    --account="$ACCOUNT" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem-per-cpu="$MEM_PER_CPU" \
    --time="$WALLTIME" \
    --job-name="labd_pmednear" \
    --array="$ARRAY_SPEC" \
    --output="$SLURM_LOG_DIR/slurm-%x-%A_%a.out" \
    --error="$SLURM_LOG_DIR/slurm-%x-%A_%a.err" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONFIG_ROOT="$MASTER_ROOT",RESULTS_ROOT_OVERRIDE="$RESULTS_ROOT_OVERRIDE",UNIQUE_ID="pmedian_nearest_transfer",TIMESTAMP="$RUN_TIMESTAMP",PROBLEM_CLASSES="MIXED" \
    "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch")"
  JOB_ID="$(echo "$submission" | awk '{print $4}')"
  echo "$JOB_ID" > "$MASTER_ROOT/job_id_$MODE.txt"
  echo "Submitted $MODE array: JobID=$JOB_ID, tasks=$ARRAY_SPEC, master=$MASTER_ROOT"
  exit 0
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
UNIQUE_ID="pmedian_nearest_transfer"
MASTER_ROOT="$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/pmedian_nearest_${TIMESTAMP}"
CONFIG_DIR="$MASTER_ROOT/configs"
SLURM_LOG_DIR="$MASTER_ROOT/SLURM_LOGS"
TASK_LIST="$MASTER_ROOT/task_list.txt"

mkdir -p "$CONFIG_DIR" "$SLURM_LOG_DIR"
: > "$TASK_LIST"

# my_job.sbatch expects a base config in CONFIG_ROOT.
cp "$SRC_DIR/config.py" "$MASTER_ROOT/config.py"
cp "$SRC_DIR/config.py" "$MASTER_ROOT/snapshot_config.py"
if [[ -f "$SRC_DIR/config_base.py" ]]; then
  cp "$SRC_DIR/config_base.py" "$MASTER_ROOT/snapshot_config_base.py"
fi
cp "$PROJECT_ROOT/submit_pmedian_nearest.sh" "$MASTER_ROOT/snapshot_submit.sh"
cp "$PROJECT_ROOT/my_job.pmedian_nearest.alliance.sbatch" "$MASTER_ROOT/snapshot_my_job.sbatch"

append_remote_override_application() {
  local override_file="$1"
  cat <<'PYEOF' >> "$override_file"

# ========== REMOTE OVERRIDE APPLICATION ==========
# main.py imports config values before config_override.py.  Copy each task's
# settings into the running __main__ module before create_parameter_grid().
import sys as _sys
_running_main = _sys.modules.get("__main__")
if _running_main is not None:
    _override_names = (
        "problem_classes_to_instances",
        "method_categories_grid",
        "exact_callback_types_grid",
        "ml_method_types_grid",
        "ml_callback_types_grid",
        "predictor_ML_Augmented_BD_grid",
        "selected_subproblems_ratio_grid",
        "use_covariates_grid",
        "n_trials_grid",
        "m_trials_grid",
        "feature_dim_grid",
        "use_stochastic_capacity_grid",
        "switching_strategy",
        "scmnd_formulation_grid",
        "n_walks",
        "walk_length",
        "window",
        "min_count",
        "sg",
        "seed",
        "feature_transition_temperature",
        "feature_stability_epsilon",
        "feature_transition_rule",
        "tree_min_impurity_decrease_grid",
        "tree_quantile_grid_size_grid",
        "clustering_random_state_grid",
        "clustering_init_method_grid",
        "clustering_max_iter_grid",
        "clustering_tol_grid",
        "parametric_L_d_grid",
        "parametric_L_v_grid",
        "parametric_regularization_grid",
        "parametric_kappa_d_grid",
        "parametric_kappa_v_grid",
        "parametric_ridge_beta_grid",
        "parametric_rhs_dim_grid",
        "parametric_rhs_ridge_beta_features_grid",
        "parametric_rhs_ridge_beta_rhs_grid",
        "parametric_repair_mode_grid",
        "parametric_feasibility_tolerance",
        "parametric_repair_safety",
        "parametric_pmedian_solver",
        "parametric_pmedian_time_limit",
        "parametric_training_time_limit",
        "parametric_solver_output",
    )
    for _name in _override_names:
        if _name in globals():
            setattr(_running_main, _name, globals()[_name])
PYEOF
}

write_task() {
  local problem_class="$1"
  local instance="$2"
  local method="$3"
  local ratio="$4"
  local formulation="${5:-}"

  local task_id
  task_id=$(($(wc -l < "$TASK_LIST") + 1))
  local config_name
  config_name=$(printf "configs/task_%04d.py" "$task_id")
  local config_file="$MASTER_ROOT/$config_name"

  cat <<EOF2 > "$config_file"
problem_classes_to_instances = {
    "$problem_class": ["$instance"]
}

method_categories_grid = ["ML"]
exact_callback_types_grid = ["int_and_LP"]
ml_method_types_grid = ["ML_Augmented_BD"]
ml_callback_types_grid = ["ML_at_both"]
predictor_ML_Augmented_BD_grid = ["$method"]

selected_subproblems_ratio_grid = [$ratio]
use_covariates_grid = ["features_only"]
n_trials_grid = [5]
m_trials_grid = [0]
feature_dim_grid = [3]
use_stochastic_capacity_grid = [False]
switching_strategy = ["no_switch"]

# Same representation construction used in the paper-facing experiments.
n_walks = 2
walk_length = 5
window = 5
min_count = 1
sg = 1
seed = 42
feature_transition_temperature = 1.0
feature_stability_epsilon = 1e-12
feature_transition_rule = "MANUSCRIPT"

# Same TREE and CLUSTERING settings as the existing nonparametric methods.
tree_min_impurity_decrease_grid = [0.0]
tree_quantile_grid_size_grid = [10]
clustering_random_state_grid = [42]
clustering_init_method_grid = ["k-means++"]
clustering_max_iter_grid = [300]
clustering_tol_grid = [1e-4]

# Same RHS-Ridge specification as the retained computational implementation.
parametric_L_d_grid = [1e5]
parametric_L_v_grid = [1e5]
parametric_regularization_grid = [1e-6]
parametric_kappa_d_grid = [1.0]
parametric_kappa_v_grid = [1.0]
parametric_ridge_beta_grid = [1e-3]
parametric_rhs_dim_grid = [3]
parametric_rhs_ridge_beta_features_grid = [1e-3]
parametric_rhs_ridge_beta_rhs_grid = [1e-3]
parametric_repair_mode_grid = ["LOCAL"]
parametric_feasibility_tolerance = 1e-7
parametric_repair_safety = 1e-10

# The final RHS-Ridge runs use the deterministic build-and-swap p-median
# heuristic. This setting is active for PMEDIAN_WEIGHTED_TRANSFER and is
# harmless/unused for the tree- and cluster-anchor variants.
parametric_pmedian_solver = "GREEDY"
parametric_pmedian_time_limit = 300.0
parametric_training_time_limit = 300.0
parametric_solver_output = 0
EOF2

  if [[ "$problem_class" == "SCMND" ]]; then
    echo "scmnd_formulation_grid = [\"$formulation\"]" >> "$config_file"
  fi

  append_remote_override_application "$config_file"
  echo "$config_name|$problem_class" >> "$TASK_LIST"
}

# SCFLP: 2 methods x 15 instance/scenario files x 3 exact shares = 90 tasks.
for method in "${METHODS[@]}"; do
  for instance in "${SCFLP_INSTANCES[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      write_task "SCFLP" "$instance" "$method" "$ratio"
    done
  done
done

# SCMND: 2 methods x 18 files x 3 exact shares x 2 formulations = 216 tasks.
for method in "${METHODS[@]}"; do
  for formulation in "${SCMND_FORMULATIONS[@]}"; do
    for instance in "${SCMND_INSTANCES[@]}"; do
      for ratio in "${RATIOS[@]}"; do
        write_task "SCMND" "$instance" "$method" "$ratio" "$formulation"
      done
    done
  done
done

TASK_COUNT=$(wc -l < "$TASK_LIST")
EXPECTED_TASKS=306
if [[ "$TASK_COUNT" -ne "$EXPECTED_TASKS" ]]; then
  echo "ERROR: expected $EXPECTED_TASKS tasks but generated $TASK_COUNT" >&2
  exit 1
fi

# Save an explicit, human-readable design manifest alongside the task list.
cat > "$MASTER_ROOT/experiment_design.txt" <<EOF2
Scenario-specific p-median experiment design
==========================================
Branch: agent/pmedian-scenario-specific-transfer
Methods:
  - PMEDIAN_NEAREST_TRANSFER
  - PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR

Primary settings:
  seed = 42
  time limit = 10800 sec (hardcoded in master_problem.py)
  target MIP gap = 0.01 (hardcoded in master_problem.py)
  warm-up iterations = 5
  exact shares = 0.05, 0.15, 0.25
  feature specification = features_only
  callback policy = ML_at_both
  feature_dim = 3
  n_walks = 2
  walk_length = 5
  p-median solver = GREEDY
  RHS Ridge repair = LOCAL

SCFLP tasks: 90
SCMND tasks: 216 (both MANUSCRIPT_ENVELOPE and CURRENT_SOFT_CAPACITY)
Total tasks: $TASK_COUNT
EOF2


# Validate every case in the newly prepared array before marking it ready.
validate_all_manifest_groups "$MASTER_ROOT"
echo "Validated all 306 task configs: methods, classes, and SCMND formulations."
echo "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" > "$MASTER_ROOT/code_commit.txt"
echo "$MASTER_ROOT" > "$LATEST_MASTER_FILE"
echo "Prepared $TASK_COUNT NEW P-median configurations (90 SCFLP, 216 SCMND)."
echo "Master root: $MASTER_ROOT"
echo "Preview: sed -n '1p;46p;91p;145p;199p;253p' '$TASK_LIST'"
echo "Main-paper first: run four pilots, then 194 other main-paper cases."
echo "Appendix later: 108 CURRENT_SOFT_CAPACITY cases (separate submission)."
echo "See README_PMEDIAN_NEAREST.md for commands and verification gates."
