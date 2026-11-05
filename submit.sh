#!/bin/bash
set -euo pipefail

# =============================================================================
# CROSS-WEIGHT PRODUCTION EXPERIMENTS
# =============================================================================
# New experiments requested after separating:
#   (i) the transfer/representative weighting mechanism, and
#   (ii) the parametric versus nonparametric prediction mechanism.
#
# Existing results are NOT rerun.  This script submits only the three missing
# combinations:
#   1) PMEDIAN_WEIGHTED_TRANSFER
#      - nonparametric use of the same p-median/count-weighted common anchor
#        used by the existing RHS-Ridge implementation;
#   2) PARAMETRIC_RHS_RIDGE_TREE_ANCHOR
#      - RHS Ridge with the same tree transfer weights as LABD--Tree;
#   3) PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR
#      - RHS Ridge with the same clustering transfer weights as LABD--Cluster.
#
# Universal settings intentionally match the paper-facing experiments:
#   seed=42, |K_0|=5, m_trials=0, feature_dim=3, n_walks=2,
#   walk_length=5, ML_at_both, features_only, exact shares 5/15/25%.
# The master problem itself retains MIPGap=0.01 and TimeLimit=10800 sec (3 h)
# in src/Problem/master_problem.py.
#
# Each Slurm array task contains exactly ONE method x instance x ratio x
# formulation configuration.  Therefore a 3-hour solver limit fits inside the
# 4-hour allocation without allowing several 3-hour cases to accumulate in a
# single array task.
# =============================================================================

ACCOUNT="def-brostami-ab_cpu"
CPUS_PER_TASK=32
MEM_PER_CPU="8G"
WALLTIME="04:00:00"
# Academic WLS license 2623115 now has 20 approved sessions for this project.
# Use 12 by default to leave room for retained WLS tokens and other jobs.
# Override with MAX_CONCURRENT only after verifying actual session usage.
MAX_CONCURRENT="${MAX_CONCURRENT:-12}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SRC_DIR="$PROJECT_ROOT/src"
RESULTS_ROOT_OVERRIDE="${RESULTS_ROOT_OVERRIDE:-/home/rafieiki/scratch/Benders_Integration_with_ML/RESULTS}"
mkdir -p "$RESULTS_ROOT_OVERRIDE"

METHODS=(
  "PMEDIAN_WEIGHTED_TRANSFER"
  "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR"
  "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR"
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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
UNIQUE_ID="weighting_cross"
MASTER_ROOT="$RESULTS_ROOT_OVERRIDE/MASTER_RUNS/weighting_cross_${TIMESTAMP}"
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
cp "$PROJECT_ROOT/submit.sh" "$MASTER_ROOT/snapshot_submit.sh"
cp "$PROJECT_ROOT/my_job.sbatch" "$MASTER_ROOT/snapshot_my_job.sbatch"

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

# SCFLP: 3 methods x 15 instance/scenario files x 3 exact shares = 135 tasks.
for method in "${METHODS[@]}"; do
  for instance in "${SCFLP_INSTANCES[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      write_task "SCFLP" "$instance" "$method" "$ratio"
    done
  done
done

# SCMND: 3 methods x 18 files x 3 exact shares x 2 formulations = 324 tasks.
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
EXPECTED_TASKS=459
if [[ "$TASK_COUNT" -ne "$EXPECTED_TASKS" ]]; then
  echo "ERROR: expected $EXPECTED_TASKS tasks but generated $TASK_COUNT" >&2
  exit 1
fi

# Save an explicit, human-readable design manifest alongside the task list.
cat > "$MASTER_ROOT/experiment_design.txt" <<EOF2
Cross-weight experiment design
==============================
Branch: agent/weighting-cross-combinations
Methods:
  - PMEDIAN_WEIGHTED_TRANSFER
  - PARAMETRIC_RHS_RIDGE_TREE_ANCHOR
  - PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR

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

SCFLP tasks: 135
SCMND tasks: 324 (both MANUSCRIPT_ENVELOPE and CURRENT_SOFT_CAPACITY)
Total tasks: $TASK_COUNT
EOF2

if [[ "${SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  echo "========================================================="
  echo "Dry run complete. No jobs submitted."
  echo "Generated $TASK_COUNT task configurations."
  echo "Master root: $MASTER_ROOT"
  echo "Task list:   $TASK_LIST"
  echo "========================================================="
  exit 0
fi

submission=$(sbatch \
  --account="$ACCOUNT" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem-per-cpu="$MEM_PER_CPU" \
  --time="$WALLTIME" \
  --job-name="labd_crossw" \
  --array="1-${TASK_COUNT}%${MAX_CONCURRENT}" \
  --output="$SLURM_LOG_DIR/slurm-%x-%A_%a.out" \
  --error="$SLURM_LOG_DIR/slurm-%x-%A_%a.err" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONFIG_ROOT="$MASTER_ROOT",RESULTS_ROOT_OVERRIDE="$RESULTS_ROOT_OVERRIDE",UNIQUE_ID="$UNIQUE_ID",TIMESTAMP="$TIMESTAMP",PROBLEM_CLASSES="MIXED" \
  "$PROJECT_ROOT/my_job.sbatch")

job_id=$(echo "$submission" | awk '{print $4}')
echo "$job_id" > "$MASTER_ROOT/job_id.txt"

echo "========================================================="
echo "Submitted cross-weight production experiment array."
echo "Job ID:       $job_id"
echo "Array tasks:  $TASK_COUNT (max concurrent: $MAX_CONCURRENT)"
echo "Master root:  $MASTER_ROOT"
echo "Task list:    $TASK_LIST"
echo "Check status: squeue -u $USER"
echo "========================================================="
