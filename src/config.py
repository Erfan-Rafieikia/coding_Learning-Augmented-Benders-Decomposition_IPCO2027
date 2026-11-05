"""Local smoke-test configuration for the cross-weight LABD experiments.

The complete experiment defaults remain in config_base.py. This wrapper keeps
local runs small while exercising the three NEW combinations added on branch
"agent/weighting-cross-combinations".

Remote Digital Research Alliance execution is controlled by submit.sh, which
creates one isolated override per method x instance x exact-share x formulation.
"""

from config_base import *  # noqa: F401,F403

# =============================================================================
# LOCAL CROSS-WEIGHT SMOKE TEST
# =============================================================================
method_categories_grid = ["ML"]
ml_method_types_grid = ["ML_Augmented_BD"]
ml_callback_types_grid = ["ML_at_both"]
predictor_ML_Augmented_BD_grid = [
    "PMEDIAN_WEIGHTED_TRANSFER",
    "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR",
    "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR",
]

selected_subproblems_ratio_grid = [0.05]
use_covariates_grid = ["features_only"]
n_trials_grid = [5]
m_trials_grid = [0]
feature_dim_grid = [3]
use_stochastic_capacity_grid = [False]
switching_strategy = ["no_switch"]

# One small SCFLP instance for local verification.
problem_classes_to_instances = {
    "SCFLP": [
        "SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p10_stochastic.txt",
    ]
}

# Representation settings matched to the production experiment.
n_walks = 2
walk_length = 5
window = 5
min_count = 1
sg = 1
seed = 42
feature_transition_temperature = 1.0
feature_stability_epsilon = 1e-12
feature_transition_rule = "MANUSCRIPT"

# Same nonparametric selector hyperparameters as the existing TREE/CLUSTERING runs.
tree_min_impurity_decrease_grid = [0.0]
tree_quantile_grid_size_grid = [10]
clustering_random_state_grid = [42]
clustering_init_method_grid = ["k-means++"]
clustering_max_iter_grid = [300]
clustering_tol_grid = [1e-4]

# Same RHS-Ridge specification as the existing paper-facing implementation.
parametric_rhs_dim_grid = [3]
parametric_rhs_ridge_beta_features_grid = [1e-3]
parametric_rhs_ridge_beta_rhs_grid = [1e-3]
parametric_repair_mode_grid = ["LOCAL"]
parametric_feasibility_tolerance = 1e-7
parametric_repair_safety = 1e-10
parametric_pmedian_solver = "GREEDY"
parametric_pmedian_time_limit = 300.0
parametric_training_time_limit = 300.0
parametric_solver_output = 0
