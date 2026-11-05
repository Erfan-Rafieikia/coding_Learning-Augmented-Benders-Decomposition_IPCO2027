"""
Minimal test to verify diagnostic tracking is working.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Problem.data import read_scflp_data
from Problem.master_problem import solve_master_problem
from log_config import setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Read data
data_path = "DATA/SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p5_stochastic.txt"
data = read_scflp_data(data_path)
covar_vectors = data.covariate_vectors if hasattr(data, 'covariate_vectors') else {}

logger.info("="*80)
logger.info("TESTING DIAGNOSTIC TRACKING")
logger.info("="*80)

# Create a simple method_config
method_config = {
    "method_category": "ML",
    "ml_method_type": "ML_Augmented_BD_with_duals",
    "predictor_type": "CLUSTERING",
    "ml_callback_type": "ML_at_integer_only",
    "n_trials": 5,
    "m_trials": 0,
    "selected_subproblems_ratio": 0.15,
    "n_neighbors": 3,
    "knn_find_optimal_k": False,
    "tree_min_impurity_decrease": 0.0,
    "tree_quantile_grid_size": 10,
    "clustering_random_state": 42,
    "clustering_init_method": "k-means++",
    "clustering_max_iter": 300,
    "clustering_tol": 1e-4,
    "switching_strategy": "no_switch",
    "use_covariates": False,
    "use_stochastic_capacity": False
}

# Solve
logger.info("Starting solve...")
solution = solve_master_problem(
    problem_type="SCFLP",
    data=data,
    selected_subproblems=None,
    covariate_vectors=covar_vectors,
    method_config=method_config
)

logger.info("="*80)
logger.info("DIAGNOSTIC RESULTS:")
logger.info("="*80)
logger.info(f"total_unique_points: {solution.total_unique_points}")
logger.info(f"total_repeated_hits: {solution.total_repeated_hits}")
logger.info(f"repeated_hits_with_cuts: {solution.repeated_hits_with_cuts}")
logger.info(f"callback_counts: {solution.callback_counts}")
logger.info(f"mipsol_root: {solution.callback_counts['mipsol_root']}")
logger.info(f"mipsol_nonroot: {solution.callback_counts['mipsol_nonroot']}")
logger.info(f"mipnode_root: {solution.callback_counts['mipnode_root']}")
logger.info(f"mipnode_nonroot: {solution.callback_counts['mipnode_nonroot']}")
logger.info(f"avg_mipsol_per_nonroot_node: {solution.avg_mipsol_per_nonroot_node}")
logger.info(f"avg_mipnode_per_nonroot_node: {solution.avg_mipnode_per_nonroot_node}")
logger.info(f"num_solutions_reevaluated: {solution.objective_reevaluation_results.get('num_snapshots', 0) if solution.objective_reevaluation_results else 0}")
logger.info("="*80)

if solution.total_unique_points == 0:
    logger.error("❌ DIAGNOSTIC TRACKING IS NOT WORKING!")
else:
    logger.info("✅ DIAGNOSTIC TRACKING IS WORKING!")
