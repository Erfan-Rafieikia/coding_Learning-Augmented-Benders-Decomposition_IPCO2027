"""
main.py

This module serves as the main execution engine and experimental orchestrator for the
SCFLP-focused ML-augmented Benders decomposition framework. It coordinates experimental
evaluation of multiple ML-augmented Benders decomposition methods on SCFLP problem instances.

Current Focus: SCFLP + All Clustering Methods
--------------------------------------------
**As of October 2025, this framework supports SCFLP problems with k-NN, TREE, DUAL_CLUSTERING, 
DUAL_CLUSTERING_WITH_DUAL, UNCERTAINTY_CLUSTERING, and BERTSIMAS_RANDOM_SAMPLING prediction** methods to provide 
comprehensive ML-augmented Benders decomposition evaluation.

Overview:
---------
The main module implements a sophisticated experimental framework that:
1. Generates comprehensive parameter grids respecting logical dependencies
2. Orchestrates systematic execution across problem instances
3. Collects detailed performance metrics and solution statistics
4. Manages data persistence and result organization
5. Provides logging and progress tracking for long-running experiments

Key Components:
---------------

1. **Parameter Grid Generation**:
   - Intelligent combination of experimental parameters
   - Dependency-aware grid construction
   - Optimization to avoid infeasible parameter combinations
   - Support for both traditional and ML-augmented Benders variants

2. **Experimental Orchestration**:
   - Sequential execution across problem classes and instances
   - Systematic parameter space exploration
   - Progress tracking and status reporting
   - Error handling and recovery mechanisms

3. **Data Management**:
   - Structured logging of solution statistics including true objective value comparison
   - Feature vector collection and persistence with datetime-based CSV naming
   - Training diagnostics capture and storage
   - Excel-based result organization with ordered columns
   - Incremental saving of experimental results for real-time progress tracking
   - Automatic datetime timestamp integration for traceability
   - True vs ML-augmented objective value tracking (original_Prob_Obj_Value column)

4. **Performance Analysis**:
   - Comprehensive timing breakdowns
   - Cut generation statistics by source and type
   - ML prediction performance metrics
   - Switching strategy behavior analysis

Architecture:
-------------

**Experimental Flow**:
1. **Initialization**: Load configuration and set up logging
2. **Grid Generation**: Create parameter combinations with dependency logic
3. **Instance Processing**: Iterate through problem classes and instances
4. **Problem Solving**: Execute ML-augmented Benders with specified parameters
5. **Data Collection**: Gather solution statistics and performance metrics
6. **Result Persistence**: Save logs, features, and diagnostics to Excel files

**Parameter Grid System**:
The framework generates parameter combinations considering logical dependencies:
- Traditional Benders (use_prediction=False): Simplified parameter sets
- ML-augmented Benders (use_prediction=True): Full parameter exploration
- Method-specific constraints (KNN vs REG parameter relevance)
- Fallback strategy dependencies

Grid Generation Logic:
----------------------

**Traditional Benders Configuration**:
- Simplified parameter set for baseline comparison
- Fixed prediction method and reduced parameter exploration
- Focus on switching strategy variations

**ML-Augmented Benders Configuration**:
- Full parameter space exploration
- Method-specific parameter selection (KNN vs REG)
- Comprehensive feature and selection parameter combinations
- Training budget and objective weight variations

**Dependency Management**:
- use_prediction=False → prediction_method='REG', minimal other parameters
- prediction_method='KNN' → regression parameters (L_d, L_v, p_d, p_v) use defaults
- prediction_method='TREE' → regression parameters (L_d, L_v, p_d, p_v) use defaults
- prediction_method='SIMPLE_AVG' → regression parameters (L_d, L_v, p_d, p_v) use defaults
- fallback_method='solve' → n_neighbors only relevant for primary prediction
- switching_strategy constraints on trial numbers

Data Collection Framework:
--------------------------

**Solution Statistics**:
- Objective values and first-stage solutions
- Comprehensive timing breakdowns (solving, feature generation, training)
- Cut generation statistics by source (ML, exact, fallback) and type
- Branch-and-bound node exploration metrics
- Switching strategy event tracking

**Feature Vector Management**:
- Collection of scenario embeddings across all instances
- Feature parameter documentation for reproducibility
- Integration with graph-based feature generation system

**Training Diagnostics** (REG method):
- Per-iteration training performance metrics
- Budget utilization analysis (L_d, L_v)
- Feasibility constraint satisfaction tracking
- Model convergence and optimization status

Output Organization:
--------------------

**Solution Log (solution_log.xlsx)**:
Comprehensive results table with columns:
- Problem identification (class, instance, parameters)
- Algorithm configuration (prediction method, switching strategy)
- Performance metrics (timing, cut counts, node exploration)
- ML-specific statistics (predictions, fallbacks, training time)
- Switching behavior (events, iterations, permanent switches)

**Feature Database (all_scenario_features.xlsx)**:
- Scenario feature vectors across all instances
- Feature generation parameters for each vector
- Problem context and instance information

**Training Diagnostics (training_diagnostics.xlsx)**:
- Per-iteration training performance (REG method only)
- Budget constraint analysis and slack variables
- Model optimization status and convergence metrics

Column Ordering System:
-----------------------

The DESIRED_SOLVING_LOG_COLUMNS list ensures consistent Excel output structure:
1. **Problem Identification**: class, instance, dimensions
2. **Algorithm Configuration**: prediction settings, switching strategy
3. **Core Parameters**: trials, subproblem selection, neighbors
4. **Feature Engineering**: individual parameters (n_walks, walk_length, etc.)
5. **Performance Metrics**: timing, objective values, solution quality
   - objective_value: Final optimal objective value from master problem
   - original_Prob_Obj_Value: True objective value calculated by solving all subproblems exactly
     (only computed when use_prediction=True to compare with ML-augmented solution)
6. **Cut Generation**: detailed breakdown by source and type
7. **ML Performance**: predictions, fallbacks, training statistics
8. **Switching Behavior**: events, iterations, permanent switches

Subproblem Selection Methods:
-----------------------------

The framework supports four distinct subproblem selection strategies:

**Trial-Cut-Frequency Selection (Default)**:
- Leverages historical cut generation patterns from trial phase
- Prioritizes subproblems that contributed most cuts during classic Benders iterations
- Integrates with callbacks.py for comprehensive trial cut count aggregation
- Provides data-driven selection based on empirical subproblem importance

**P-Median Optimization**:
- Formulates subproblem selection as facility location problem
- Minimizes total distance from all subproblems to selected representatives
- Uses Gurobi optimization for exact solution of p-median model
- Ensures mathematically optimal representative subset selection

**K-Means Clustering**:
- Applies unsupervised machine learning for feature-based grouping
- Clusters subproblems in feature space and selects cluster centroids
- Provides balanced representation across feature distribution
- Requires scikit-learn for clustering implementation

**Effective Subproblem Selection**:
- Statistical analysis of subproblem impact and effectiveness
- Legacy method maintained for compatibility with existing experiments
- Focuses on subproblems with highest individual contribution metrics

Error Handling and Recovery:
----------------------------

**File Access Management**:
- Automatic timestamp-based file naming for locked Excel files
- Graceful handling of permission errors
- Alternative save locations for concurrent execution

**Execution Resilience**:
- Instance-level error isolation (failures don't stop entire experiment)
- Comprehensive error logging with context
- Progress preservation for partial completion analysis

**Data Validation**:
- Parameter combination feasibility checking
- Solution validity verification
- Missing data handling and default value assignment

Integration Points:
-------------------

**Configuration System** (config.py):
- Parameter import and grid value access
- Problem instance mapping and file path resolution
- Feature engineering parameter specification

**Problem Solving** (master_problem.py):
- Master problem instantiation with parameter passing
- Solution object extraction and statistics gathering
- Integration with callback-based cut generation

**Data Loading** (Problem.data):
- Problem instance reading and data structure creation
- Multi-format support (text, Excel, pickle)
- Data validation and preprocessing

**ML Components**:
- Feature generation coordination
- Training diagnostic collection
- Prediction performance tracking

Performance Considerations:
---------------------------

**Scalability**:
- Parameter grid size management (can generate 100K+ combinations)
- Memory-efficient instance processing
- Incremental result saving for large experiments

**Resource Management**:
- Instance-level memory cleanup
- Temporary file management
- Progress reporting for long-running experiments

**Output Optimization**:
- Structured Excel output with proper formatting
- Efficient DataFrame operations for large result sets
- Compressed storage for feature vectors

Experimental Design Support:
----------------------------

**Parameter Space Exploration**:
- Systematic coverage of algorithm variants
- Statistical significance through multiple instances
- Baseline comparison with traditional Benders

**Reproducibility**:
- Comprehensive parameter logging
- Random seed management
- Version control integration

**Analysis Preparation**:
- Structured output for statistical analysis
- Visualization-ready data formats
- Cross-instance comparison facilitation

Usage Patterns:
---------------

**Full Experimental Evaluation**:
```python
# Generates comprehensive parameter grid and executes full evaluation
python main.py
```

**Subset Evaluation**:
```python
# Modify problem_classes_to_instances in config.py for focused testing
# Adjust grid parameters for reduced scope
```

**Development Testing**:
```python
# Use minimal parameter grids for algorithm development and debugging
# Enable detailed logging for diagnostic purposes
```

Extensions and Customization:
-----------------------------

**New Problem Types**:
1. Add problem class to config.py problem_classes_to_instances
2. Implement data reader in Problem.data
3. Add master formulation to master_problem.py
4. Update main.py for any problem-specific handling

**Additional Metrics**:
1. Extend Solution dataclass in master_problem.py
2. Update column list in DESIRED_SOLVING_LOG_COLUMNS
3. Modify result collection logic in main execution loop

**Custom Experimental Designs**:
1. Modify create_parameter_grid() for specific parameter combinations
2. Add filtering logic for targeted experiments
3. Implement custom result aggregation and analysis

Dependencies:
-------------
- Problem.data: Instance loading and data structures
- Problem.master_problem: Optimization and solution extraction
- log_config: Logging configuration and setup
- config: Parameter definitions and experimental settings
- Dual_Prediction.training: Training diagnostic collection

"""

import os
import sys
import time
import json
import datetime
import numpy as np
import pandas as pd
from itertools import product
import logging

# Add the src directory to the Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========== Local Imports ==========
from Problem.data import read_scflp_data, read_nesp_data, read_sumnd_data, read_scmnd_data
from Problem.master_problem import solve_master_problem, Solution
from log_config import setup_logging
from utils.run_registry import (
    get_unique_identifier,
    get_timestamp_str,
    get_execution_mode,
    get_problem_class_from_config,
    create_run_directory,
    save_config_snapshot,
    get_run_identifier_display
)
from utils.config_utils import (
    get_method_category, 
    should_execute_step,
    get_ml_method_type,
    should_register_lp_callbacks,
    should_use_ml_for_unselected
)
from config import (
    DATA_ROOT, RESULTS_ROOT, get_problem_class_roots,
    problem_classes_to_instances,
    n_walks, walk_length, feature_dim, window, min_count, sg, seed,
    feature_transition_temperature, feature_stability_epsilon,
    feature_transition_rule,
    selected_subproblems_ratio,
    switching_strategy, use_stochastic_capacity,
    # New grid parameters
    method_categories_grid,
    exact_callback_types_grid,
    ml_method_types_grid,
    ml_callback_types_grid,
    predictor_ML_Augmented_BD_grid,
    predictor_ML_Augmented_BD_with_duals_grid,
    predictor_Covariate_Similarity_grid,
    predictor_Uncertainty_Similarity_grid,
    # Phase parameters
    n_trials_grid, m_trials_grid,
    selected_subproblems_ratio_grid, feature_dim_grid,
    use_covariates_grid, use_stochastic_capacity_grid, scmnd_formulation_grid,
    # Predictor-specific grid parameters
    n_neighbors_grid, knn_find_optimal_k_grid,
    tree_min_impurity_decrease_grid, tree_quantile_grid_size_grid,
    clustering_random_state_grid, clustering_init_method_grid,
    clustering_max_iter_grid, clustering_tol_grid,
    # Parametric predictor parameters
    parametric_L_d_grid, parametric_L_v_grid,
    parametric_regularization_grid,
    parametric_kappa_d_grid, parametric_kappa_v_grid,
    parametric_ridge_beta_grid,
    parametric_rhs_dim_grid,
    parametric_rhs_ridge_beta_features_grid,
    parametric_rhs_ridge_beta_rhs_grid,
    parametric_repair_mode_grid,
    parametric_feasibility_tolerance, parametric_repair_safety,
    parametric_objective_guard, parametric_diagnostic_probe,
    parametric_guard_gamma_zero_fastpath,
    parametric_pmedian_solver, parametric_pmedian_time_limit,
    parametric_training_time_limit, parametric_solver_output,
)
from Dual_Prediction.prediction import TRAINING_CALL_LOGS
from Dual_Prediction.parametric import PARAMETRIC_PREDICTORS

# ========== Logging Setup ==========
logger = logging.getLogger(__name__)
setup_logging()

# ========== Run Registry / Execution Environment Initialization ==========
# Initialize run tracking (works for both local and remote execution)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\src", "")
if not os.path.exists(os.path.join(PROJECT_ROOT, "RESULTS")):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get or create run directory based on execution environment
if 'RUN_DIRECTORY' in os.environ:
    # Remote execution (SLURM) - directory already created by submit.sh
    RUN_DIRECTORY = os.environ['RUN_DIRECTORY']
    RUN_IDENTIFIER = os.environ.get('RUN_IDENTIFIER', f"{get_unique_identifier()}_{get_timestamp_str()}")
    PROBLEM_CLASS = os.environ.get('RUN_PROBLEM_CLASS', get_problem_class_from_config())
    logger.info(f"Using SLURM-created run directory: {RUN_DIRECTORY}")
else:
    # Local execution - create run directory
    # First, load config to detect problem class
    try:
        import config as cfg
        PROBLEM_CLASS = get_problem_class_from_config(cfg)
    except:
        PROBLEM_CLASS = "SCFLP"
    
    RUN_DIRECTORY = create_run_directory(PROBLEM_CLASS, PROJECT_ROOT)
    RUN_IDENTIFIER = f"{get_unique_identifier()}_{get_timestamp_str()}"
    save_config_snapshot(RUN_DIRECTORY, PROJECT_ROOT, {'problem_class': PROBLEM_CLASS})
    logger.info(f"Created local run directory: {RUN_DIRECTORY}")

# Display run identifier prominently
print("\n" + "="*60)
print(get_run_identifier_display())
print("="*60 + "\n")
logger.info(f"Run Identifier: {RUN_IDENTIFIER}")
logger.info(f"Problem Class: {PROBLEM_CLASS}")
logger.info(f"Execution Mode: {get_execution_mode()}")
logger.info(f"Run Directory: {RUN_DIRECTORY}")



def _apply_runtime_config_override():
    """Apply per-task overrides when config_override.py is present.
    
    Overrides the following from config_override.py if present:
    - problem_classes_to_instances: Problem instances for this task
    - method_categories_grid: Method categories (["Exact"] or ["ML"])
    - ml_method_types_grid: ML method types for this task
    - predictor_ML_Augmented_BD_grid: One or more LABD predictors for this task
    - scmnd_formulation_grid: One or more executable SCMND formulations
    """
    global problem_classes_to_instances, method_categories_grid, ml_method_types_grid
    global predictor_ML_Augmented_BD_grid, scmnd_formulation_grid

    try:
        import config_override as override_config
    except ImportError:
        logger.info("No config_override.py detected; using base config.py settings")
        return

    # Override problem_classes_to_instances
    if hasattr(override_config, "problem_classes_to_instances"):
        problem_classes_to_instances = override_config.problem_classes_to_instances
        logger.info(
            "Applied config_override.py with problem classes: %s",
            list(problem_classes_to_instances.keys()),
        )
    else:
        logger.warning(
            "config_override.py found but missing problem_classes_to_instances; using base config.py settings"
        )
    
    # Override method_categories_grid
    if hasattr(override_config, "method_categories_grid"):
        method_categories_grid = override_config.method_categories_grid
        logger.info(
            "Applied method_categories_grid from config_override.py: %s",
            method_categories_grid,
        )
    
    # Override ml_method_types_grid
    if hasattr(override_config, "ml_method_types_grid"):
        ml_method_types_grid = override_config.ml_method_types_grid
        logger.info(
            "Applied ml_method_types_grid from config_override.py: %s",
            ml_method_types_grid,
        )

    # A submission task can isolate one ML_Augmented_BD predictor.  This keeps
    # failures and runtimes independent across the six requested predictors.
    if hasattr(override_config, "predictor_ML_Augmented_BD_grid"):
        predictor_ML_Augmented_BD_grid = (
            override_config.predictor_ML_Augmented_BD_grid
        )
        logger.info(
            "Applied predictor_ML_Augmented_BD_grid from config_override.py: %s",
            predictor_ML_Augmented_BD_grid,
        )

    # Original and manuscript SCMND runs are emitted as separate SLURM tasks.
    if hasattr(override_config, "scmnd_formulation_grid"):
        scmnd_formulation_grid = override_config.scmnd_formulation_grid
        logger.info(
            "Applied scmnd_formulation_grid from config_override.py: %s",
            scmnd_formulation_grid,
        )


_apply_runtime_config_override()

from utils.gurobi_cut_diagnostics import CUT_COUNT_FIELDS
from utils.convergence_diagnostics import (build_solver_bound_trace, write_bound_trace,
    verified_relative_gap, _finite, TRACE_SCHEMA_VERSION)
from utils.exclusive_profiler import (TIMING_CATEGORIES, TIMING_PHASE_TAGS, reconcile_end_to_end,
    solution_time_excluding_reporting, TIMING_SCHEMA_VERSION, SOLUTION_TIME_METRIC_VERSION)
from utils.parametric_iteration_sidecar import externalize_parametric_iteration_records

# Desired column order for the solution log Excel output
DESIRED_SOLVING_LOG_COLUMNS = [
    "problem_class", "instance_filename", "num_subproblems", "num_first_stage_variables",
    # Method architecture columns (NEW - from restructured config.py)
    "method_category", "ml_method_type", "ml_callback_type", "exact_callback_type", "predictor_type",
    "use_prediction", "prediction_method", "dual_difference_pred_loss", "obj_difference_pred_loss",
    "fallback_method", "switching_strategy", "KNN_k_neighbors", "knn_find_optimal_k", 
    # tree_max_depth: removed for adaptive TREE method
    "tree_min_impurity_decrease", "tree_quantile_grid_size",
    # Clustering parameters (for CLUSTERING predictor)
    "clustering_random_state", "clustering_init_method", "clustering_max_iter", "clustering_tol",
    "selected_subproblems_ratio", "n_trials", "m_trials",
    "subproblem_selection_method", "scenario_generation_method", "scenario_info_dict", 
    "n_walks", "walk_length",
    "feature_dim", "window", "min_count", "sg", "seed",
    "objective_value", "original_Prob_Obj_Value", "first_stage_optimnal_values",
    "solution_time_sec", "feature_gen_time_sec", "unified_selection_training_time_sec", "phase0_execution_time_sec", "phase0_solution_generation_time_sec", 
    "phase1_execution_time_sec", "phase2_execution_time_sec",
    "num_fallback_used", "num_predictions", "bnb_nodes", "nodes_explored",
    "cuts_mip_selected", "cuts_rel_selected", "cuts_mip_ml", "cuts_rel_ml",
    "cuts_mip_fallback", "cuts_rel_fallback", "cuts_mip_trial", "cuts_rel_trial",
    "cuts_mip_unselected_total", "cuts_rel_unselected_total", "cuts_mip_exact_switch", "cuts_rel_exact_switch", "phase0_cuts_added",
    "num_switch_events", "switch_iterations", "permanently_switched", "time_limit_sec", "stopping_reason", "iteration",
    "use_covariates", "use_stochastic_capacity", "scmnd_formulation",
    "feature_transition_rule", "feature_transition_temperature", "feature_stability_epsilon",
    "parametric_config", "parametric_anchor_method",
    "parametric_objective_guard_enabled", "parametric_objective_guard_checks",
    "parametric_objective_guard_accepted", "parametric_objective_guard_rejected",
    "parametric_objective_guard_rejected_negative",
    "parametric_objective_guard_total_improvement",
    "parametric_objective_guard_gamma_zero_fastpath",
    "parametric_objective_guard_skipped_zero",
    "parametric_objective_guard_skipped_equal_anchor", "parametric_objective_guard_evaluated",
    "parametric_objective_guard_time_sec",
    "parametric_repair_mode_actual",
    "parametric_structure_repair_attempts",
    "parametric_structure_repair_feasible_candidates",
    "parametric_structure_repair_anchor_fallbacks",
    "parametric_structure_repair_nonanchor_candidates",
    "parametric_structure_repair_positive_objectives",
    "parametric_structure_repair_positive_gain_total",
    "parametric_structure_repair_selected_kinds",
    "parametric_diagnostic_probe_enabled",
    "parametric_diag_gamma_zero",
    "parametric_diag_gamma_positive_count",
    "parametric_diag_gamma_positive_mean",
    "parametric_diag_gamma_max",
    "parametric_diag_gamma_gt_1e12",
    "parametric_diag_gamma_gt_1e8",
    "parametric_diag_gamma_gt_1e6",
    "parametric_diag_gamma_gt_1e3",
    "parametric_diag_gamma_gt_01",
    "parametric_diag_probe_checks",
    "parametric_diag_probe_improved",
    "parametric_diag_probe_negative",
    "parametric_diag_probe_tied",
    "parametric_diag_probe_positive_total",
    "parametric_diag_probe_negative_total",
    "parametric_diag_probe_max_positive",
    "parametric_diag_probe_anchor_check_time_sec",
    "parametric_diag_probe_objective_time_sec",
    "parametric_diag_probe_total_time_sec",
    "parametric_training_calls", "parametric_training_failures",
    "parametric_training_time_sec", "parametric_prediction_time_sec",
    "parametric_feasibility_time_sec", "parametric_repair_time_sec",
    "parametric_prediction_opportunities", "parametric_raw_predictions", "parametric_raw_infeasible_predictions",
    "parametric_raw_infeasible_rate", "parametric_repaired_predictions",
    "parametric_exact_fallback_subproblems", "parametric_exact_fallback_rate",
    "parametric_selected_exact_subproblems", "parametric_total_exact_subproblems",
    "parametric_algorithm_exact_subproblem_solves",
    "parametric_algorithm_exact_subproblem_time_sec",
    "parametric_reevaluation_exact_subproblem_solves",
    "parametric_reevaluation_exact_subproblem_time_sec",
    "parametric_cuts_added", "parametric_cut_addition_rate",
    "parametric_mean_cut_violation", "parametric_max_cut_violation",
    "parametric_mean_repair_gamma", "parametric_min_repair_gamma",
    "parametric_max_condition_number", "parametric_pmedian_time_sec",
    "parametric_pmedian_objective", "parametric_pmedian_status",
    "parametric_iteration_records",
    # Category classification columns (added for better experiment organization and analysis)
    "pred_or_exact", "approach_category", "feature_generation_method",
    # Gurobi bound information
    "ObjBound_LB",
    # Callback diagnostic columns (Task 1, 2, 3 - Feb 2026)
    # Task 1: Point signature tracking
    "total_unique_points", "total_repeated_hits", "repeated_hits_with_cuts",
    # Task 2: Root vs non-root callback tracking
    "mipsol_root", "mipsol_nonroot", "mipnode_root", "mipnode_nonroot",
    "avg_mipsol_per_nonroot_node", "avg_mipnode_per_nonroot_node",
    # Task 3: Objective re-evaluation
    "num_solutions_reevaluated", "max_objective_drift", "mean_objective_drift", "std_objective_drift",
    # Gurobi parameters logging (added for MIPGap and TimeLimit tracking)
    "MIPGap_target", "MIPGap_Stop",
    "convergence_trace_path", "verified_final_relative_gap",
    "experiment_solve_wall_time_sec", "master_solver_internal_time_sec",
    "callback_exact_subproblem_time_sec", "callback_other_time_sec",
    "experiment_outside_solver_time_sec",
    "solver_component_breakdown_json", "solver_component_sum_sec",
    "gurobi_solver_runtime_sec", "data_loading_time_sec",
    "final_objective_validation_time_sec", "candidate_reevaluation_time_sec",
    "model_build_and_other_time_sec",
    "end_to_end_component_breakdown_json", "end_to_end_component_sum_sec",
    "end_to_end_component_reconciliation_error_sec",
    "end_to_end_component_counts_json", "timing_phase_tags_json",
    "timing_schema_version", "master_optimize_wall_time_sec",
    "solution_time_metric_version", "end_to_end_wall_time_sec", "solution_component_sum_sec",
    "exact_cut_construction_time_sec", "ml_cut_construction_time_sec",
    "exact_cut_construction_count", "ml_cut_construction_count", "cut_construction_timing_version",
    "final_exact_evaluation_time_sec", "final_exact_evaluation_count",
    "final_exact_subproblem_solves", "final_exact_evaluation_status", "convergence_trace_schema_version",
    *CUT_COUNT_FIELDS
]

def create_parameter_grid():
    """
    Generate parameter combinations using new method_config architecture.
    
    This function creates configurations based on the restructured 6-step pattern with clear 
    separation between method categories, ML method types, and predictor types.
    
    Architecture:
    -------------
    1. **Method Category Selection**: Exact vs ML
    2. **For Exact Methods**: Generate configs with exact_callback_types
    3. **For ML Methods**: Generate configs for each (ml_method_type, predictor_type, ml_callback_type)
       combination
    
    Returns:
    --------
    list of dict: Each dict contains a complete method_config plus problem instance info
    """
    param_combinations = []
    
    for problem_class, instance_list in problem_classes_to_instances.items():
        for method_category in method_categories_grid:
            
            if method_category == 'Exact':
                # ========== EXACT METHODS ==========
                # Generate configurations for standard Benders decomposition
                for exact_callback_type in exact_callback_types_grid:
                    for n_trial in n_trials_grid:
                        for use_stoch_cap in use_stochastic_capacity_grid:
                            # Exact methods don't need m_trials (no Phase 0)
                            m_trial = 0
                            
                            method_config = {
                                'method_category': 'Exact',
                                'exact_callback_type': exact_callback_type,
                                'n_trials': n_trial,  
                                'm_trials': m_trial,
                                'switching_strategy': switching_strategy[0] if switching_strategy else 'no_switch',
                                'use_stochastic_capacity': use_stoch_cap,
                                # Features not needed for Exact but included for consistency
                                'feature_params': {
                                    'n_walks': n_walks,
                                    'walk_length': walk_length,
                                    'feature_dim': feature_dim_grid[0] if feature_dim_grid else feature_dim,
                                    'window': window,
                                    'min_count': min_count,
                                    'sg': sg,
                                    'seed': seed,
                                    'temperature': feature_transition_temperature,
                                    'stability_epsilon': feature_stability_epsilon,
                                    'transition_rule': feature_transition_rule,
                                }
                            }
                            
                            param_combinations.append({
                                'problem_class': problem_class,
                                'instance_list': instance_list,
                                'method_config': method_config
                            })
            
            else:  # ML category
                # ========== ML METHODS ==========
                # Generate configurations for 5 ML method types
                for ml_method_type in ml_method_types_grid:
                    for ml_callback_type in ml_callback_types_grid:
                        
                        # Determine predictor types for this ML method
                        if ml_method_type == 'ML_Augmented_BD':
                            predictor_types = predictor_ML_Augmented_BD_grid
                        elif ml_method_type == 'ML_Augmented_BD_with_duals':
                            predictor_types = predictor_ML_Augmented_BD_with_duals_grid
                        elif ml_method_type == 'Covariate_Similarity':
                            predictor_types = predictor_Covariate_Similarity_grid
                        elif ml_method_type == 'Uncertainty_Similarity':
                            predictor_types = predictor_Uncertainty_Similarity_grid
                        elif ml_method_type == 'Random_Sampling':
                            predictor_types = [None]  # No predictor needed
                        
                        for predictor_type in predictor_types:
                            for n_trial in n_trials_grid:
                                for m_trial in m_trials_grid:
                                    for ratio in selected_subproblems_ratio_grid:
                                        for feat_dim in feature_dim_grid:
                                            for switch_strat in switching_strategy:
                                                for use_stoch_cap in use_stochastic_capacity_grid:
                                                    # use_covariates parameter is meaningful only for ML_Augmented_BD.
                                                    # For other ML method types, fix to features_only to avoid
                                                    # generating duplicate configurations that behave identically.
                                                    use_covariates_options = (
                                                        use_covariates_grid
                                                        if ml_method_type == 'ML_Augmented_BD'
                                                        else ['features_only']
                                                    )
                                                    
                                                    # Base method config
                                                    method_config = {
                                                        'method_category': 'ML',
                                                        'ml_method_type': ml_method_type,
                                                        'ml_callback_type': ml_callback_type,
                                                        'n_trials': n_trial,
                                                        'm_trials': m_trial,
                                                        'selected_subproblems_ratio': ratio,
                                                        'switching_strategy': switch_strat,
                                                        'use_stochastic_capacity': use_stoch_cap,
                                                        'feature_params': {
                                                            'n_walks': n_walks,
                                                            'walk_length': walk_length,
                                                            'feature_dim': feat_dim,
                                                            'window': window,
                                                            'min_count': min_count,
                                                            'sg': sg,
                                                            'seed': seed,
                                                            'temperature': feature_transition_temperature,
                                                            'stability_epsilon': feature_stability_epsilon,
                                                            'transition_rule': feature_transition_rule,
                                                        }
                                                    }
                                                    
                                                    # Add predictor-specific parameters
                                                    if predictor_type == 'KNN':
                                                        for n_neigh in n_neighbors_grid:
                                                            for find_k in knn_find_optimal_k_grid:
                                                                for use_cov in use_covariates_options:
                                                                    config = method_config.copy()
                                                                    config['predictor_type'] = 'KNN'
                                                                    config['n_neighbors'] = n_neigh
                                                                    config['knn_find_optimal_k'] = find_k
                                                                    config['use_covariates'] = use_cov
                                                                    
                                                                    param_combinations.append({
                                                                        'problem_class': problem_class,
                                                                        'instance_list': instance_list,
                                                                        'method_config': config
                                                                    })
                                                    
                                                    elif predictor_type == 'TREE':
                                                        for min_impurity in tree_min_impurity_decrease_grid:
                                                            for quantile_size in tree_quantile_grid_size_grid:
                                                                for use_cov in use_covariates_options:
                                                                    config = method_config.copy()
                                                                    config['predictor_type'] = 'TREE'
                                                                    config['tree_min_impurity_decrease'] = min_impurity
                                                                    config['tree_quantile_grid_size'] = quantile_size
                                                                    config['use_covariates'] = use_cov
                                                                    
                                                                    param_combinations.append({
                                                                        'problem_class': problem_class,
                                                                        'instance_list': instance_list,
                                                                        'method_config': config
                                                                    })
                                                    
                                                    elif predictor_type == 'CLUSTERING':
                                                        for random_state in clustering_random_state_grid:
                                                            for init_method in clustering_init_method_grid:
                                                                for max_iter in clustering_max_iter_grid:
                                                                    for tol in clustering_tol_grid:
                                                                        for use_cov in use_covariates_options:
                                                                            config = method_config.copy()
                                                                            config['predictor_type'] = 'CLUSTERING'
                                                                            config['clustering_random_state'] = random_state
                                                                            config['clustering_init_method'] = init_method
                                                                            config['clustering_max_iter'] = max_iter
                                                                            config['clustering_tol'] = tol
                                                                            config['use_covariates'] = use_cov
                                                                            
                                                                            param_combinations.append({
                                                                                'problem_class': problem_class,
                                                                                'instance_list': instance_list,
                                                                                'method_config': config
                                                                            })
                                                    elif predictor_type in {'PMEDIAN_WEIGHTED_TRANSFER', 'PMEDIAN_NEAREST_TRANSFER'}:
                                                        for use_cov in use_covariates_options:
                                                            config = method_config.copy()
                                                            config['predictor_type'] = predictor_type
                                                            config['use_covariates'] = use_cov
                                                            # Match the representative-selection settings used
                                                            # by the existing RHS-Ridge p-median anchor.
                                                            config.update({
                                                                'parametric_pmedian_solver': parametric_pmedian_solver,
                                                                'parametric_pmedian_time_limit': parametric_pmedian_time_limit,
                                                                'parametric_solver_output': parametric_solver_output,
                                                            })
                                                            param_combinations.append({
                                                                'problem_class': problem_class,
                                                                'instance_list': instance_list,
                                                                'method_config': config
                                                            })

                                                    elif predictor_type in PARAMETRIC_PREDICTORS:
                                                        common = {
                                                            'parametric_feasibility_tolerance': parametric_feasibility_tolerance,
                                                            'parametric_repair_safety': parametric_repair_safety,
                                                            'parametric_objective_guard': parametric_objective_guard,
                                                            'parametric_guard_gamma_zero_fastpath': parametric_guard_gamma_zero_fastpath,
                                                            'parametric_diagnostic_probe': parametric_diagnostic_probe,
                                                            'parametric_pmedian_solver': parametric_pmedian_solver,
                                                            'parametric_pmedian_time_limit': parametric_pmedian_time_limit,
                                                            'parametric_training_time_limit': parametric_training_time_limit,
                                                            'parametric_solver_output': parametric_solver_output,
                                                        }
                                                        options = []
                                                        if predictor_type == 'PARAMETRIC_MANUSCRIPT':
                                                            for L_d, L_v, reg in product(
                                                                parametric_L_d_grid,
                                                                parametric_L_v_grid,
                                                                parametric_regularization_grid,
                                                            ):
                                                                options.append({
                                                                    'parametric_L_d': L_d,
                                                                    'parametric_L_v': L_v,
                                                                    'parametric_regularization': reg,
                                                                })
                                                        elif predictor_type == 'PARAMETRIC_PENALIZED_ERRORS':
                                                            for kappa_d, kappa_v, reg in product(
                                                                parametric_kappa_d_grid,
                                                                parametric_kappa_v_grid,
                                                                parametric_regularization_grid,
                                                            ):
                                                                options.append({
                                                                    'parametric_kappa_d': kappa_d,
                                                                    'parametric_kappa_v': kappa_v,
                                                                    'parametric_regularization': reg,
                                                                })
                                                        elif predictor_type == 'PARAMETRIC_RIDGE_REPAIR':
                                                            for beta, repair_mode in product(
                                                                parametric_ridge_beta_grid,
                                                                parametric_repair_mode_grid,
                                                            ):
                                                                options.append({
                                                                    'parametric_ridge_beta': beta,
                                                                    'parametric_repair_mode': repair_mode,
                                                                })
                                                        else:
                                                            for rhs_dim, beta_f, beta_g, repair_mode in product(
                                                                parametric_rhs_dim_grid,
                                                                parametric_rhs_ridge_beta_features_grid,
                                                                parametric_rhs_ridge_beta_rhs_grid,
                                                                parametric_repair_mode_grid,
                                                            ):
                                                                options.append({
                                                                    'parametric_rhs_dim': rhs_dim,
                                                                    'parametric_rhs_ridge_beta_features': beta_f,
                                                                    'parametric_rhs_ridge_beta_rhs': beta_g,
                                                                    'parametric_repair_mode': repair_mode,
                                                                })

                                                        # The two cross-weight RHS-Ridge variants
                                                        # reuse exactly the same TREE/CLUSTERING
                                                        # selector hyperparameters as their
                                                        # nonparametric counterparts.
                                                        selection_options = [{}]
                                                        if predictor_type == 'PARAMETRIC_RHS_RIDGE_TREE_ANCHOR':
                                                            selection_options = [
                                                                {
                                                                    'tree_min_impurity_decrease': min_impurity,
                                                                    'tree_quantile_grid_size': quantile_size,
                                                                }
                                                                for min_impurity, quantile_size in product(
                                                                    tree_min_impurity_decrease_grid,
                                                                    tree_quantile_grid_size_grid,
                                                                )
                                                            ]
                                                        elif predictor_type == 'PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR':
                                                            selection_options = [
                                                                {
                                                                    'clustering_random_state': random_state,
                                                                    'clustering_init_method': init_method,
                                                                    'clustering_max_iter': max_iter,
                                                                    'clustering_tol': tol,
                                                                }
                                                                for random_state, init_method, max_iter, tol in product(
                                                                    clustering_random_state_grid,
                                                                    clustering_init_method_grid,
                                                                    clustering_max_iter_grid,
                                                                    clustering_tol_grid,
                                                                )
                                                            ]

                                                        for extra in options:
                                                            for selection_extra in selection_options:
                                                                for use_cov in use_covariates_options:
                                                                    config = method_config.copy()
                                                                    config['predictor_type'] = predictor_type
                                                                    config['use_covariates'] = use_cov
                                                                    config.update(common)
                                                                    config.update(extra)
                                                                    config.update(selection_extra)
                                                                    param_combinations.append({
                                                                        'problem_class': problem_class,
                                                                        'instance_list': instance_list,
                                                                        'method_config': config,
                                                                    })
                                                    
                                                    elif predictor_type is None:  # Random_Sampling
                                                        config = method_config.copy()
                                                        config['use_covariates'] = 'features_only'  # Not applicable
                                                        # No predictor parameters needed
                                                        
                                                        param_combinations.append({
                                                            'problem_class': problem_class,
                                                            'instance_list': instance_list,
                                                            'method_config': config
                                                        })
    
    # Expand only SCMND runs across the requested executable formulations.
    # Other problem classes receive the preserved default solely for uniform
    # logging/configuration plumbing.
    expanded_combinations = []
    for entry in param_combinations:
        formulations = (
            scmnd_formulation_grid
            if str(entry['problem_class']).upper() == 'SCMND'
            else ['MANUSCRIPT_ENVELOPE']
        )
        for formulation in formulations:
            if str(entry['problem_class']).upper() == 'SCMND' and formulation != 'MANUSCRIPT_ENVELOPE':
                raise ValueError("This campaign requires hard-capacity SCMND: MANUSCRIPT_ENVELOPE")
            expanded = entry.copy()
            expanded['method_config'] = entry['method_config'].copy()
            expanded['method_config']['scmnd_formulation'] = formulation
            expanded_combinations.append(expanded)

    logger.info(f"Generated {len(expanded_combinations)} parameter combinations")
    return expanded_combinations


# ========== STATUS TRACKING FUNCTIONS (Q5 - ENHANCED) ==========
def get_status_file_path():
    """
    Get the central status file path.
    For remote execution: CONFIG_ROOT/task_status.txt
    For local execution: RUN_DIRECTORY/task_status.txt
    """
    if 'CONFIG_ROOT' in os.environ:
        # Remote execution (SLURM)
        config_root = os.environ['CONFIG_ROOT']
        return os.path.join(config_root, "task_status.txt")
    else:
        # Local execution
        return os.path.join(RUN_DIRECTORY, "task_status.txt")


def initialize_status_tracking():
    """
    Initialize task status tracking file with header showing Job ID and Task List.
    Called once per job (by first task).
    
    Creates a status file showing:
    - Job ID and Run Identifier
    - All tasks with their ID, problem class, config name, and status
    - Task ID format: task_{ARRAY_TASK_ID}_{SLURM_JOB_ID}
    """
    status_file = get_status_file_path()
    status_dir = os.path.dirname(status_file)
    
    # Create parent directory if needed
    os.makedirs(status_dir, exist_ok=True)
    
    # Get Job ID and Task List path
    job_id = os.environ.get('SLURM_JOB_ID', 'LOCAL')
    task_list_path = os.environ.get('CONFIG_ROOT', RUN_DIRECTORY)
    task_list_file = os.path.join(task_list_path, "task_list.txt")
    
    # Only initialize if it doesn't exist
    if not os.path.exists(status_file):
        try:
            # Read all task configurations from task_list.txt
            task_configs = {}  # task_idx -> (config_name, problem_class)
            if os.path.exists(task_list_file):
                with open(task_list_file, 'r') as f:
                    for task_idx, line in enumerate(f, 1):
                        task_line = line.strip()
                        if '|' in task_line:
                            config_name, problem_class = task_line.split('|', 1)
                            task_configs[task_idx] = (config_name.strip(), problem_class.strip())
                        else:
                            # Fallback if format is incorrect
                            task_configs[task_idx] = (task_line, "UNKNOWN")
            
            # Write initialized status file
            with open(status_file, 'w') as f:
                f.write("=" * 90 + "\n")
                f.write("JOB STATUS\n")
                f.write("=" * 90 + "\n")
                f.write(f"Job ID:           {job_id}\n")
                f.write(f"Run ID:           {RUN_IDENTIFIER}\n")
                f.write(f"Started:          {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Tasks:      {len(task_configs)}\n\n")
                
                f.write("Task Status:\n")
                f.write("=" * 90 + "\n")
                
                # Add initial status for all tasks
                for task_idx in sorted(task_configs.keys()):
                    config_name, problem_class = task_configs[task_idx]
                    task_id = f"task_{task_idx}_{job_id}"
                    f.write(f"Task {task_idx:2d} [{problem_class:6s}] [{task_id}]: PENDING ({config_name})\n")
                
                f.write("\n" + "=" * 90 + "\n")
                f.write(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Initialized task status file: {status_file}")
        except Exception as e:
            logger.warning(f"Could not initialize status file: {e}")


def update_task_status(status, task_id=None):
    """
    Update the central task status file with all current task statuses.
    This reads the full task list and updates ALL task statuses in one operation (overwrite pattern).
    
    Parameters:
    -----------
    status : str
        Current status of this specific task: "RUNNING", "COMPLETED", "FAILED"
    task_id : str, optional
        Task identifier (array task index, from environment or derived)
    """
    if task_id is None:
        task_id = os.environ.get('ARRAY_TASK_ID', os.environ.get('SLURM_ARRAY_TASK_ID', None))
    
    job_id = os.environ.get('SLURM_JOB_ID', 'LOCAL')
    status_file = get_status_file_path()
    task_list_path = os.environ.get('CONFIG_ROOT', RUN_DIRECTORY)
    task_list_file = os.path.join(task_list_path, "task_list.txt")
    
    try:
        # Read all task configs from task_list.txt (format: "config_name|PROBLEM_CLASS")
        task_configs = {}  # task_idx -> (config_name, problem_class)
        if os.path.exists(task_list_file):
            with open(task_list_file, 'r') as f:
                for task_idx, line in enumerate(f, 1):
                    task_line = line.strip()
                    if '|' in task_line:
                        config_name, problem_class = task_line.split('|', 1)
                        task_configs[task_idx] = (config_name.strip(), problem_class.strip())
                    else:
                        # Fallback if format is incorrect
                        task_configs[task_idx] = (task_line, "UNKNOWN")
        
        # Read existing statuses to preserve them (if file exists)
        task_statuses = {}  # task_idx -> status
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Parse lines like: "Task  1 [SCFLP  ] [task_1_12408533]: PENDING (config_override_...)"
                        if line.startswith("Task ") and "]:" in line:
                            # Extract task number
                            parts = line.split("[")
                            if len(parts) >= 2:
                                task_num_str = parts[0].replace("Task", "").strip()
                                try:
                                    task_num = int(task_num_str)
                                    # Extract status (after "]:")
                                    status_part = line.split("]:")[-1].strip()
                                    status_value = status_part.split()[0]  # First word is status
                                    task_statuses[task_num] = status_value
                                except ValueError:
                                    pass
            except Exception as e:
                logger.debug(f"Could not read existing statuses: {e}")
        
        # Update current task status (convert task_id to int if it's a string)
        try:
            task_idx = int(task_id) if task_id else None
            if task_idx is not None:
                task_statuses[task_idx] = status
        except (ValueError, TypeError):
            logger.warning(f"Could not convert task_id to int: {task_id}")
        
        # Write updated status file
        with open(status_file, 'w') as f:
            # Header
            f.write("=" * 90 + "\n")
            f.write("JOB STATUS\n")
            f.write("=" * 90 + "\n")
            f.write(f"Job ID:           {job_id}\n")
            f.write(f"Run ID:           {RUN_IDENTIFIER}\n")
            f.write(f"Started:          {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tasks:      {len(task_configs)}\n\n")
            
            f.write("Task Status:\n")
            f.write("=" * 90 + "\n")
            
            # Task statuses
            completed_count = 0
            for task_idx in sorted(task_configs.keys()):
                config_name, problem_class = task_configs[task_idx]
                task_id_full = f"task_{task_idx}_{job_id}"
                current_status = task_statuses.get(task_idx, "PENDING")
                if current_status == "COMPLETED":
                    completed_count += 1
                f.write(f"Task {task_idx:2d} [{problem_class:6s}] [{task_id_full}]: {current_status:9s} ({config_name})\n")
            
            # Footer with progress
            f.write("\n" + "=" * 90 + "\n")
            progress_pct = (completed_count / len(task_configs) * 100) if len(task_configs) > 0 else 0
            f.write(f"Overall Progress: {completed_count}/{len(task_configs)} ({progress_pct:.0f}%)\n")
            f.write(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.debug(f"Updated task status: Task {task_id} = {status} (Job {job_id})")
    except Exception as e:
        logger.warning(f"Could not update status file: {e}")


def save_experiment_result_incrementally(log_entry, solving_log_path):
    """
    Save a single experiment result to Excel file incrementally.
    
    This function appends a single experimental result to the existing Excel file,
    enabling real-time result collection and progress monitoring during long-running
    experimental evaluations.
    
    Parameters:
    -----------
    log_entry : dict
        Single experiment result dictionary with all required columns
        Format: Matches DESIRED_SOLVING_LOG_COLUMNS structure
        Content: Performance metrics, parameters, and algorithm statistics
    
    solving_log_path : str
        Absolute path to the Excel file for incremental result storage
        File must exist and have proper column headers initialized
    
    Implementation:
    ---------------
    Uses pandas to read existing Excel file, append new result, and save back.
    Handles file locking and permission errors with timestamped backup files.
    Maintains column order consistency with DESIRED_SOLVING_LOG_COLUMNS.
    
    Error Handling:
    ---------------
    - Permission errors: Creates timestamped backup file
    - Missing columns: Adds missing columns with None values
    - File corruption: Logs errors but continues execution
    """
    try:
        # Read existing results
        existing_df = pd.read_excel(solving_log_path)
        
        # Create DataFrame for new result
        new_result_df = pd.DataFrame([
            externalize_parametric_iteration_records(
                log_entry, solving_log_path, row_number=len(existing_df) + 1
            )
        ])
        
        # Ensure all required columns exist in both DataFrames
        for col in DESIRED_SOLVING_LOG_COLUMNS:
            if col not in existing_df.columns:
                existing_df[col] = None
            if col not in new_result_df.columns:
                new_result_df[col] = None
        
        # Reorder columns to match desired order
        existing_df = existing_df[DESIRED_SOLVING_LOG_COLUMNS]
        new_result_df = new_result_df[DESIRED_SOLVING_LOG_COLUMNS]
        
        # Append new result
        updated_df = pd.concat([existing_df, new_result_df], ignore_index=True)
        
        # Save back to Excel
        updated_df.to_excel(solving_log_path, index=False)
        
    except PermissionError:
        # File might be open in another application, try alternative name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = solving_log_path.replace('.xlsx', f'_backup_{timestamp}.xlsx')
        
        # Save to backup location
        try:
            existing_df = pd.read_excel(solving_log_path)
            new_result_df = pd.DataFrame([
                externalize_parametric_iteration_records(
                    log_entry, solving_log_path, row_number=len(existing_df) + 1
                )
            ])
            
            for col in DESIRED_SOLVING_LOG_COLUMNS:
                if col not in existing_df.columns:
                    existing_df[col] = None
                if col not in new_result_df.columns:
                    new_result_df[col] = None
            
            existing_df = existing_df[DESIRED_SOLVING_LOG_COLUMNS]
            new_result_df = new_result_df[DESIRED_SOLVING_LOG_COLUMNS]
            updated_df = pd.concat([existing_df, new_result_df], ignore_index=True)
            updated_df.to_excel(backup_path, index=False)
            
            logger.warning(f"Main Excel file locked, saved result to backup: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to save experiment result: {e}")
    
    except Exception as e:
        logger.error(f"Error saving experiment result incrementally: {e}")


def save_features_to_csv(solution, params, problem_class, instance_filename, num_first_stage_variables, feature_csv_dir, solution_time_sec=None):
    """
    Save feature vectors to CSV file with experiment characteristics.
    
    This function creates individual CSV files for feature vectors from ML experiments,
    including comprehensive experiment metadata for traceability and analysis.
    
    Parameters:
    -----------
    solution : Solution
        Solution object containing covariate_vectors and other results
        Must have covariate_vectors attribute with scenario embeddings
    
    params : dict
        Parameter combination dictionary for this experiment
        Contains all algorithm configuration parameters
    
    problem_class : str
        Problem type identifier ('UFL', 'UHL', 'SCFLP', 'SSLP')
    
    instance_filename : str
        Problem instance filename for identification
    
    num_first_stage_variables : int
        Number of first-stage decision variables in the problem
    
    feature_csv_dir : str
        Directory path for saving feature CSV files
    
    Output File Format:
    -------------------
    CSV files named: features_{problem_class}_{instance_filename}_{timestamp}.csv
    Where timestamp format: YYYYMMDD_HHMMSS_mmm (includes milliseconds for uniqueness)
    
    Example: features_UFL_instance1_pkl_20250821_143025_123.csv
    
    Columns include:
    - scenario_idx: Scenario identifier
    - feat_0, feat_1, ..., feat_n: Feature vector components
    - Experiment characteristics: problem details, parameters, timing
    
    Benefits:
    ---------
    - Immediate feature availability for analysis
    - Experiment traceability through metadata inclusion
    - Separate files enable parallel processing and focused analysis
    - CSV format compatible with various analysis tools
    - Datetime-based naming prevents file conflicts across multiple runs
    """
    # Check if ML features are available (only for ML-enabled configurations)
    ml_enabled = params['use_prediction'] not in ["none"]
    if not (hasattr(solution, "covariate_vectors") and isinstance(solution.covariate_vectors, dict) and 
            len(solution.covariate_vectors) > 0 and ml_enabled):
        return  # No features to save
    
    try:
        # Create feature DataFrame
        feature_df = pd.DataFrame.from_dict(solution.covariate_vectors, orient='index')
        feature_df.index.name = "scenario_idx"
        feature_df.columns = [f"feat_{i}" for i in range(feature_df.shape[1])]
        feature_df.reset_index(inplace=True)
        
        # Add experiment characteristics
        feature_df["problem_class"] = problem_class
        feature_df["instance_filename"] = instance_filename
        feature_df["num_first_stage_variables"] = num_first_stage_variables
        feature_df["use_prediction"] = params['use_prediction']
        feature_df["prediction_method"] = params['prediction_method'] if ml_enabled else ""
        feature_df["switching_strategy"] = params['switching_strategy'] if ml_enabled else ""
        feature_df["selected_subproblems_ratio"] = params['selected_subproblems_ratio'] if ml_enabled else ""
        feature_df["n_trials"] = params['n_trials']
        feature_df["m_trials"] = params['m_trials']
        feature_df["subproblem_selection_method"] = params.get('subproblem_selection_method', 'integrated') if ml_enabled else ""
        # Handle KNN-specific parameters
        if ml_enabled and params['prediction_method'].upper() == 'KNN':
            knn_find_optimal_k_val = params.get('knn_find_optimal_k', False)
            if knn_find_optimal_k_val:
                feature_df["KNN_k_neighbors"] = getattr(solution, "knn_optimal_k", None) or params['n_neighbors']
            else:
                feature_df["KNN_k_neighbors"] = params['n_neighbors']
            feature_df["knn_find_optimal_k"] = knn_find_optimal_k_val
        else:
            feature_df["KNN_k_neighbors"] = ""
            feature_df["knn_find_optimal_k"] = ""
            
        feature_df["fallback_method"] = params['fallback_method'] if ml_enabled else ""
        
        # Add feature generation parameters (only when prediction is used)
        feature_df["n_walks"] = n_walks if ml_enabled else ""
        feature_df["walk_length"] = walk_length if ml_enabled else ""
        feature_df["feature_dim"] = params['feature_dim'] if ml_enabled else ""
        feature_df["window"] = window if ml_enabled else ""
        feature_df["min_count"] = min_count if ml_enabled else ""
        feature_df["sg"] = sg if ml_enabled else ""
        feature_df["seed"] = seed if ml_enabled else ""
        
        # Add solution statistics if available
        objective_value = getattr(solution, "objective_value", None)
        feature_df["objective_value"] = objective_value
        # Set original_Prob_Obj_Value based on use_prediction
        if ml_enabled:
            feature_df["original_Prob_Obj_Value"] = getattr(solution, "original_Prob_Obj_Value", None)
        else:
            feature_df["original_Prob_Obj_Value"] = objective_value
        feature_df["solution_time_sec"] = solution_time_sec
        feature_df["solution_time_metric_version"] = SOLUTION_TIME_METRIC_VERSION
        feature_df["gurobi_solver_runtime_sec"] = getattr(solution, "solution_time", None)
        feature_df["final_exact_evaluation_time_sec"] = solution.final_objective_validation_time_sec
        feature_df["feature_gen_time_sec"] = getattr(solution, "feature_gen_time_sec", None) if ml_enabled else ""
        feature_df["phase0_solution_generation_time_sec"] = getattr(solution, "phase0_solution_generation_time_sec", None)
        feature_df["phase0_cuts_added"] = getattr(solution, "phase0_cuts_added", None)
        feature_df["num_predictions"] = getattr(solution, "num_predictions", None) if ml_enabled else ""
        feature_df["num_fallback_used"] = getattr(solution, "num_fallback_used", None) if ml_enabled else ""
        
        # Create filename with timestamp for uniqueness across multiple runs
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        clean_instance = instance_filename.replace('/', '_').replace('\\', '_').replace('.', '_')
        feature_filename = f"features_{problem_class}_{clean_instance}_{timestamp}.csv"
        feature_path = os.path.join(feature_csv_dir, feature_filename)
        
        # Save to CSV
        feature_df.to_csv(feature_path, index=False)
        logger.info(f"   Features saved to: {feature_path}")
        
    except Exception as e:
        logger.error(f"Error saving features to CSV: {e}")


def main():
    """
    Execute comprehensive experimental evaluation of ML-augmented Benders decomposition.
    
    This is the main orchestrator function that conducts systematic experimental evaluation
    of the ML-augmented Benders decomposition algorithm across the complete parameter space.
    It coordinates parameter grid generation, problem instance loading, algorithm execution,
    result collection, and data export for comprehensive research analysis.
    
    The function implements a robust experimental framework that:
    1. Generates 106,926 parameter combinations using dependency-aware grid construction
    2. Executes ML-augmented Benders decomposition for each configuration
    3. Collects comprehensive performance metrics and diagnostic information
    4. Exports results to structured Excel files for analysis
    5. Maintains detailed logging for debugging and research reproducibility
    
    Experimental Workflow:
    ----------------------
    
    **Phase 1: Initialization**
    - Configure logging system with comprehensive diagnostic output
    - Initialize result collection data structures
    - Generate complete parameter grid using create_parameter_grid()
    - Log experimental scope and parameter space size
    
    **Phase 2: Parameter Grid Execution**
    For each parameter combination (106,926 total):
    - Load problem instance data from DATA/ directory
    - Configure feature parameters and ML settings
    - Execute solve_master_problem() with current parameter configuration
    - Collect comprehensive Solution object with performance metrics
    - Extract and store detailed results for analysis
    
    **Phase 3: Data Collection and Export**
    - Aggregate solving logs across all experimental runs
    - Collect feature generation data and training diagnostics
    - Export structured Excel files with multiple sheets:
      * Solving logs with performance metrics
      * Feature vectors for analysis
      * Training diagnostics (when available)
    - Generate summary statistics and experimental metadata
    
    **Phase 4: Logging and Cleanup**
    - Log experimental completion statistics
    - Provide result file locations and summary information
    - Clean up temporary data and release resources
    
    Data Flow Architecture:
    -----------------------
    
    **Input Sources**:
    ```
    config.py → Parameter definitions and experimental settings
    DATA/ → Problem instance files (UFL, UHL, SCFLP, SSLP)
    create_parameter_grid() → Systematic parameter combinations
    ```
    
    **Core Processing**:
    ```
    Parameter Grid → solve_master_problem() → Solution Objects
                  ↓
    Result Aggregation → Excel Export Generation
    ```
    
    **Output Destinations**:
    ```
    RESULTS/solving_logs.xlsx → Performance metrics and statistics
    RESULTS/covariate_vectors.xlsx → Scenario embeddings and feature data
    RESULTS/training_diagnostics.xlsx → ML model training logs
    logs/app.log → Comprehensive execution logging
    ```
    
    Result Collection Structure:
    ----------------------------
    
    **Solving Logs** (primary results sheet):
    Each experimental run contributes one row with:
    - **Problem Context**: problem_class, instance_filename, num_subproblems
    - **Algorithm Configuration**: All parameter grid values
    - **Performance Metrics**: objective_value, solution_time, various timing breakdowns
    - **Cut Statistics**: Detailed breakdown by cut type and iteration
    - **Algorithm State**: fallback usage, prediction counts, switching events
    - **Branch-and-Bound**: node counts and exploration statistics
    
    **Feature Vectors** (feature analysis sheet):
    Each scenario contributes one row with:
    - **Context**: problem_class, instance_filename, scenario identifier
    - **Feature Values**: Complete feature_dim-dimensional embedding vector
    - **Metadata**: Feature generation parameters and timing information
    
    **Training Diagnostics** (ML analysis sheet):
    Each ML training event contributes one row with:
    - **Training Context**: iteration, problem details, training parameters
    - **Training Results**: slack variables, optimization status, convergence metrics
    - **Budget Analysis**: L_d and L_v budget utilization and effectiveness
    
    Problem Instance Management:
    ----------------------------
    
    **Data Loading Pipeline**:
    ```python
    for problem_class in ['UFL', 'UHL', 'SCFLP', 'SSLP']:
        for instance_file in problem_classes_to_instances[problem_class]:
            data = read_problem_data(problem_class, instance_path)
            # Execute experimental runs for this instance
    ```
    
    **Instance Metadata Extraction**:
    - Automatic detection of problem dimensions (customers, facilities, scenarios)
    - First-stage variable counting for normalization and analysis
    - Subproblem counting for algorithmic parameter scaling
    
    **Error Handling**:
    - Graceful handling of missing instance files
    - Robust data loading with format validation
    - Comprehensive error logging for debugging
    
    Performance Monitoring:
    -----------------------
    
    **Progress Tracking**:
    - Real-time logging of experimental progress (X/106,926 completed)
    - Parameter configuration display for current experiment
    - Estimated completion time and performance statistics
    
    **Resource Management**:
    - Memory usage monitoring for large parameter grids
    - Efficient data structure management for result collection
    - Periodic garbage collection for long-running experiments
    
    **Quality Assurance**:
    - Result validation and consistency checking
    - Detection of algorithm failures and recovery mechanisms
    - Comprehensive diagnostic information for problem investigation
    
    Export System:
    --------------
    
    **Excel File Generation**:
    Uses pandas.ExcelWriter with comprehensive formatting:
    - Multiple sheets for different data types
    - Structured column ordering for analysis consistency
    - Metadata sheets with experimental parameters and settings
    
    **File Organization**:
    ```
    RESULTS/
    ├── solving_logs.xlsx          # Primary experimental results
    ├── all_scenario_features.xlsx # Feature engineering analysis
    └── training_diagnostics.xlsx  # ML training performance analysis
    ```
    
    **Column Ordering Strategy**:
    Systematic organization for analysis workflow:
    1. Problem identification (class, instance, dimensions)
    2. Algorithm configuration (parameters)
    3. Performance metrics (timing, objective values)
    4. Detailed statistics (cuts, iterations, nodes)
    5. Algorithm state (fallbacks, predictions, switching)
    
    Research Integration:
    ---------------------
    
    **Experimental Design Support**:
    - Comprehensive parameter space coverage for sensitivity analysis
    - Systematic comparison between ML and exact solving approaches
    - Detailed algorithm performance characterization
    
    **Statistical Analysis Preparation**:
    - Structured data format optimized for statistical software
    - Consistent variable naming and data types
    - Complete experimental metadata for reproducibility
    
    **Algorithm Development Support**:
    - Detailed diagnostic information for debugging
    - Performance profiling data for optimization
    - Comparative analysis across algorithm variants
    
    Scalability Considerations:
    ---------------------------
    
    **Large-Scale Execution**:
    - Memory-efficient result collection for 106,926 experiments
    - Incremental Excel writing for memory management
    - Checkpoint capability for long-running experiments
    
    **Distributed Computing Readiness**:
    - Parameter grid can be partitioned for parallel execution
    - Independent experimental runs with minimal shared state
    - Result aggregation designed for distributed collection
    
    **Performance Optimization**:
    - Efficient data structure reuse across experiments
    - Optimized parameter grid generation and iteration
    - Minimal overhead between experimental runs
    
    Error Recovery and Robustness:
    ------------------------------
    
    **Experiment-Level Error Handling**:
    - Individual experiment failures don't terminate entire run
    - Comprehensive error logging with parameter context
    - Graceful degradation with partial result collection
    
    **Data Integrity**:
    - Result validation before export
    - Backup data collection for critical experimental runs
    - Consistency checking across related data structures
    
    **System Robustness**:
    - Resource cleanup after failed experiments
    - Recovery from memory constraints or system issues
    - Detailed diagnostic information for system debugging
    
    Example Execution Flow:
    -----------------------
    ```python
    if __name__ == "__main__":
        main()
        # Produces:
        # - 106,926 experimental runs across parameter space
        # - Comprehensive Excel exports with performance data
        # - Detailed logs for debugging and analysis
        # - Structured results for statistical evaluation
    ```
    
    **Typical Execution Output**:
    ```
    INFO: Generated 106926 parameter combinations to test
    INFO: [1/106926] Solving [UFL] instance_1.pkl
    INFO:   Parameters: use_pred=True, method=REG, switch=no_switch...
    INFO: [2/106926] Solving [UFL] instance_2.pkl
    ...
    INFO: Experimental evaluation complete: 106926 runs finished
    INFO: Results exported to RESULTS/solving_logs.xlsx
    INFO: Feature data exported to RESULTS/all_scenario_features.xlsx
    ```
    
    This comprehensive experimental framework provides the foundation for systematic
    evaluation of ML-augmented optimization algorithms, enabling robust research
    conclusions and thorough algorithm analysis across diverse problem instances
    and parameter configurations.
    """
    # ========== STEP 0: Initialize logs and prepare incremental saving ==========
    # Use the run directory created during initialization
    # This ensures consistent storage across local and remote execution
    run_results_root = RUN_DIRECTORY
    
    # Initialize status tracking (Q5)
    initialize_status_tracking()
    update_task_status("RUNNING")
    
    # Use the run identifier (which includes unique ID and timestamp) for consistent file naming
    solving_log_path = os.path.join(run_results_root, "solution_log.xlsx")
    feature_csv_dir = os.path.join(run_results_root, "features_csv")
    
    # Create directories if they don't exist
    os.makedirs(run_results_root, exist_ok=True)
    os.makedirs(feature_csv_dir, exist_ok=True)
    
    # Initialize Excel file with headers for this run
    # Create empty DataFrame with desired columns and save to initialize file
    empty_df = pd.DataFrame(columns=DESIRED_SOLVING_LOG_COLUMNS)
    empty_df.to_excel(solving_log_path, index=False)
    logger.info(f"Initialized solution log file for this run: {solving_log_path}")

    # Create sophisticated parameter grid respecting dependencies
    param_combinations = create_parameter_grid()
    
    # Note: create_parameter_grid() already logs the count, so we don't repeat it here
    logger.info(f"Results will be saved incrementally to: {solving_log_path}")
    logger.info(f"Feature vectors will be saved incrementally to: {feature_csv_dir}")
    
    # Save config snapshots in the run directory for reproducibility
    if get_execution_mode() == 'local':
        save_config_snapshot(
            run_results_root,
            PROJECT_ROOT,
            {'problem_class': PROBLEM_CLASS, 'run_identifier': RUN_IDENTIFIER}
        )
        logger.info(f"Config snapshots saved to run directory: {run_results_root}")


    # ========== STEP 1+2: Solve with embedded learning in Benders ==========
    # Keep writing all result rows, but make the Slurm task fail if any
    # experiment had a Python/solver error or returned no solution.
    failed_experiments = 0
    for i, params in enumerate(param_combinations):
        problem_class = params['problem_class']
        instance_list = params['instance_list']
        method_config = params['method_config']
        
        # Extract configuration for easier access throughout the code
        method_category = get_method_category(method_config)
        ml_enabled = (method_category == 'ML')
        
        # Create flattened parameter dict for backward compatibility with existing code
        # This allows the rest of the code to access params['key'] instead of params['method__config']['key']
        flat_params = {
            'problem_class': problem_class,
            'instance_list': instance_list,
            'method_config': method_config,
            'method_category': method_category,
            'n_trials': method_config['n_trials'],
            'm_trials': method_config['m_trials'],
            'switching_strategy': method_config['switching_strategy'],
            'use_stochastic_capacity': method_config['use_stochastic_capacity'],
            'scmnd_formulation': method_config.get('scmnd_formulation', 'MANUSCRIPT_ENVELOPE'),
            'feature_dim': method_config['feature_params']['feature_dim'],
            'use_prediction': ml_enabled,  # For backward compatibility
            'fallback_method': 'solve',  # Default fallback method
        }
        for key, value in method_config.items():
            if str(key).startswith('parametric_'):
                flat_params[key] = value
        
        # Add method-specific parameters
        if ml_enabled:
            flat_params.update({
                'ml_method_type': method_config['ml_method_type'],
                'ml_callback_type': method_config['ml_callback_type'],
                'selected_subproblems_ratio': method_config['selected_subproblems_ratio'],
                'predictor_type': method_config.get('predictor_type'),
                'use_covariates': method_config.get('use_covariates', False),
                # Deprecated but kept for logging backward compatibility
                'prediction_method': method_config.get('predictor_type', method_config['ml_method_type']),
            })
            
            # Add predictor-specific parameters
            if method_config.get('predictor_type') == 'KNN':
                flat_params.update({
                    'n_neighbors': method_config.get('n_neighbors', 3),
                    'knn_find_optimal_k': method_config.get('knn_find_optimal_k', False),
                })
            elif method_config.get('predictor_type') in {
                'TREE', 'PARAMETRIC_RHS_RIDGE_TREE_ANCHOR'
            }:
                flat_params.update({
                    'tree_min_impurity_decrease': method_config.get('tree_min_impurity_decrease', 0.0),
                    'tree_quantile_grid_size': method_config.get('tree_quantile_grid_size', 10),
                })
            elif method_config.get('predictor_type') in {
                'CLUSTERING', 'PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR'
            }:
                flat_params.update({
                    'clustering_random_state': method_config.get('clustering_random_state', 42),
                    'clustering_init_method': method_config.get('clustering_init_method', 'k-means++'),
                    'clustering_max_iter': method_config.get('clustering_max_iter', 300),
                    'clustering_tol': method_config.get('clustering_tol', 1e-4),
                })
        else:  # Exact method
            flat_params.update({
                'exact_callback_type': method_config['exact_callback_type'],
                'selected_subproblems_ratio': 0.0,  # Not applicable
                'predictor_type': None,
                'use_covariates': False,
                'prediction_method': 'None',  # For backward compatibility
            })
        
        # Replace params with flat_params for the rest of the loop
        params = flat_params
        
        for instance_filename in instance_list:
            # Inclusive experiment timer: data loading through exact post-solve verification.
            # Result-file export happens afterward and is excluded by convention.
            t_experiment_start = time.perf_counter()
            class_data_root, _, _ = get_problem_class_roots(problem_class)
            # Handle different problem classes with correct data readers
            if problem_class.upper() == "SCFLP":
                instance_path = os.path.join(class_data_root, instance_filename)
                t_data_start = time.perf_counter()
                data = read_scflp_data(instance_path)
                t_data = time.perf_counter() - t_data_start
                logger.info(f" Reading SCFLP data for {instance_filename}")
                num_subproblems = len(getattr(data, "S", []))
            elif problem_class.upper() in ["NESP", "NE-SP"]:
                # instance_filename should be relative to class-specific DATA_ROOT
                instance_path = os.path.join(class_data_root, instance_filename)
                t_data_start = time.perf_counter()
                data = read_nesp_data(instance_path, weight_mode="uniform")  # or "demand"
                t_data = time.perf_counter() - t_data_start
                logger.info(f" Reading NE-SP data for {instance_filename}")
                num_subproblems = data.n_od_pairs
            elif problem_class.upper() == "SUMND":
                # instance_filename should be relative to class-specific DATA_ROOT
                instance_path = os.path.join(class_data_root, instance_filename)
                t_data_start = time.perf_counter()
                data = read_sumnd_data(instance_path)
                t_data = time.perf_counter() - t_data_start
                logger.info(f" Reading SUMND data for {instance_filename}")
                num_subproblems = len(data.S)
            elif problem_class.upper() == "SCMND":
                # SCMND uses the same JSON schema as SUMND but preserves real capacities.
                instance_path = os.path.join(class_data_root, instance_filename)
                t_data_start = time.perf_counter()
                data = read_scmnd_data(
                    instance_path,
                    scmnd_formulation=params.get(
                        'scmnd_formulation', 'MANUSCRIPT_ENVELOPE'
                    ),
                )
                t_data = time.perf_counter() - t_data_start
                logger.info(f" Reading SCMND data for {instance_filename}")
                num_subproblems = len(data.S)
            else:
                raise ValueError(f"Unsupported problem class: {problem_class}")

            # Calculate actual number of selected subproblems from ratio
            n_selected_subproblems = max(1, int(params['selected_subproblems_ratio'] * num_subproblems))
            
            logger.info(f"[{i+1}/{len(param_combinations)}] Solving [{problem_class}] {instance_filename}")
            logger.info(f"  Parameters: method_category={params['method_category']}, use_pred={params['use_prediction']}, "
                       f"method={params.get('ml_method_type', params.get('exact_callback_type'))}, "
                       f"predictor={params.get('predictor_type', 'N/A')}, "
                       f"switch={params['switching_strategy']}, n_trials={params['n_trials']}, "
                       f"ratio={params['selected_subproblems_ratio']}, n_selected={n_selected_subproblems}, feat_dim={params['feature_dim']}")

            # Focus: SCFLP, SUMND, NESP - other problem types exist as stubs
            if problem_class == "SCFLP":
                num_first_stage_variables = getattr(data, 'J', np.array([])).shape[0]  # number of facilities
            elif problem_class.upper() in ["SUMND", "SCMND"]:
                num_first_stage_variables = len(data.A)  # number of arcs
            elif problem_class.upper() in ["NESP", "NE-SP"]:
                num_first_stage_variables = len(data.A)  # number of arcs
            else:
                # Other problem types (UFL, UHL, SSLP, CMND, MCFL) - stubs only
                num_first_stage_variables = 0
                logger.warning(f"Problem type {problem_class} not fully implemented. SCFLP/SUMND/NESP focus only.")

            # Solve with error handling to prevent single failures from stopping entire experiment
            solution =None
            error_type = None
            error_message = None
            
            try:
                # Build method_config dictionary directly from experiment parameters
                method_config = {
                    'method_category': params['method_category'],
                    'use_prediction': params['use_prediction'],
                    'switching_strategy': params['switching_strategy'],
                    'use_covariates': params['use_covariates'], 
                    'use_stochastic_capacity': params['use_stochastic_capacity'],
                    'scmnd_formulation': params.get(
                        'scmnd_formulation', 'MANUSCRIPT_ENVELOPE'
                    ),
                    'n_trials': params['n_trials'],
                    'm_trials': params['m_trials'],
                    'n_selected_subproblems': n_selected_subproblems,
                    'feature_params': {
                        'n_walks': n_walks,
                        'walk_length': walk_length,
                        'feature_dim': params['feature_dim'],
                        'window': window,
                        'min_count': min_count,
                        'sg': sg,
                        'seed': seed,
                        'temperature': feature_transition_temperature,
                        'stability_epsilon': feature_stability_epsilon,
                        'transition_rule': feature_transition_rule,
                    }
                }
                
                # Add method-specific configuration
                if params['method_category'] == 'ML':
                    method_config['ml_method_type'] = params['ml_method_type']
                    method_config['predictor_type'] = params.get('predictor_type')
                    method_config['ml_callback_type'] = params['ml_callback_type']
                    
                    # Add predictor-specific parameters. The cross-anchor
                    # RHS-Ridge variants need BOTH the original Tree/Cluster
                    # selector settings and the RHS-Ridge parametric settings.
                    predictor_type = params.get('predictor_type')
                    if predictor_type == 'KNN':
                        method_config['n_neighbors'] = params.get('n_neighbors', 3)
                        method_config['knn_find_optimal_k'] = params.get('knn_find_optimal_k', False)
                    elif predictor_type in {'TREE', 'PARAMETRIC_RHS_RIDGE_TREE_ANCHOR'}:
                        method_config['tree_min_impurity_decrease'] = params.get('tree_min_impurity_decrease', 0.0)
                        method_config['tree_quantile_grid_size'] = params.get('tree_quantile_grid_size', 10)
                    elif predictor_type in {'CLUSTERING', 'PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR'}:
                        method_config['clustering_random_state'] = params.get('clustering_random_state', 42)
                        method_config['clustering_init_method'] = params.get('clustering_init_method', 'k-means++')
                        method_config['clustering_max_iter'] = params.get('clustering_max_iter', 300)
                        method_config['clustering_tol'] = params.get('clustering_tol', 1e-4)

                    if predictor_type in PARAMETRIC_PREDICTORS or predictor_type in {'PMEDIAN_WEIGHTED_TRANSFER', 'PMEDIAN_NEAREST_TRANSFER'}:
                        for key, value in params.items():
                            if str(key).startswith('parametric_'):
                                method_config[key] = value
                    
                    # Add Random_Sampling specific parameters
                    if params['ml_method_type'] == 'Random_Sampling':
                        method_config['bertsimas_random_sampling_random_state'] = params.get('bertsimas_random_sampling_random_state', 42)
                else:  # Exact
                    method_config['exact_callback_type'] = params['exact_callback_type']
                
                solution = solve_master_problem(
                    problem_type=problem_class,
                    data=data,
                    selected_subproblems=[],
                    covariate_vectors={},
                    method_config=method_config
                )
                t_sol = time.perf_counter() - t_experiment_start
                logger.info(f" Instance  [{problem_class}] {instance_filename}: {t_sol:.2f} seconds including final reporting.")
                
            except Exception as e:
                t_sol = time.perf_counter() - t_experiment_start
                error_type = type(e).__name__
                error_message = str(e)
                logger.error(f" Instance  [{problem_class}] {instance_filename} FAILED after {t_sol:.2f} seconds.")
                logger.error(f" Error type: {error_type}")
                logger.error(f" Error message: {error_message}")
                logger.info(f" Continuing with next parameter combination...")

            # Logging - Handle both successful solutions and failures
            if solution is not None:
                # Record only Gurobi's own master UB/LB observations.
                bound_points = build_solver_bound_trace(getattr(solution, "bound_trace", None))
                trace_path = None
                if bound_points:
                    import re
                    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_filename))[:72]
                    safe_method = re.sub(
                        r"[^A-Za-z0-9_.-]+", "_",
                        str(params.get("predictor_type") or params.get("exact_callback_type") or "method")
                    )[:72]
                    trace_name = f"{i:04d}_{safe_instance}_{safe_method}_gap{getattr(solution, 'mip_gap_target', 'na')}.csv"
                    trace_path = write_bound_trace(
                        bound_points, os.path.join(run_results_root, "convergence_traces", trace_name)
                    )
                final_original = _finite(getattr(solution, "original_Prob_Obj_Value", None))
                final_lb = _finite(getattr(solution, "ObjBound_LB", None))
                last_certified_gap = (verified_relative_gap(final_lb, final_original)
                                      if final_original is not None and final_lb is not None else None)
                buckets = getattr(solution, "solver_timing_buckets", None) or {}
                component_breakdown = getattr(solution, "solver_component_breakdown", None) or {}
                accounted_solver = sum(float(v) for v in buckets.values())
                outside_solver = max(0.0, float(t_sol) - accounted_solver)

                # Additive, EXCLUSIVE end-to-end decomposition, including data
                # loading and the separate final reporting evaluation. Its sum
                # reconciles to end_to_end_wall_time_sec, NOT solution_time_sec.
                # Phase times and overlapping legacy measurements are diagnostics,
                # NOT additive categories.
                data_loading = float(t_data)
                final_validation = float(
                    getattr(solution, "final_objective_validation_time_sec", 0.0) or 0.0
                )
                candidate_recheck = 0.0  # Retired: never reevaluate historical candidates.
                algorithm_solution_time = solution_time_excluding_reporting(t_sol, final_validation)
                gurobi_solver_runtime = float(
                    getattr(solution, "solution_time", 0.0) or 0.0
                )
                logger.info(f"Solution time: {algorithm_solution_time:.6f}s; final reporting: {final_validation:.6f}s")
                # Stable additive 33-key schema with zero entries for
                # inapplicable operations. This avoids confusing blank and 0.
                timed_components = {
                    category: float(component_breakdown.get(category, 0.0))
                    for category in TIMING_CATEGORIES
                }
                if not component_breakdown:
                    timed_components["master_solver_internal"] = gurobi_solver_runtime
                timed_components.update({
                    "input_data_load_validation": data_loading,
                    "master_model_construction": float(getattr(
                        solution, "master_model_construction_time_sec", 0.0
                    ) or 0.0),
                    "terminal_exact_original_objective_validation": final_validation,
                    "all_candidate_expost_reevaluation": candidate_recheck,
                })
                component_counts = {
                    category: int((getattr(solution, "timing_component_counts", None)
                                  or {}).get(category, 0))
                    for category in TIMING_CATEGORIES
                }
                component_counts["input_data_load_validation"] = 1
                component_counts["master_model_construction"] = 1
                component_counts["terminal_exact_original_objective_validation"] = (
                    int(getattr(solution, "final_exact_evaluation_count", 0))
                )
                component_counts["all_candidate_expost_reevaluation"] = 0
                component_counts["master_solver_internal"] = 1
                timed_components, component_counts = reconcile_end_to_end(
                    t_sol, timed_components, component_counts
                )
                remainder = timed_components["remaining_unclassified_overhead"]
                end_to_end_sum = sum(timed_components.values())
                # Extract stopping reason from solution
                stopping_reason = getattr(solution, 'stopping_reason', 'UNKNOWN')
                
                # Build ordered log entry ensuring column order requirements
                reg_method = str(params['prediction_method']).upper() == "REG"
                
                # Determine ML-specific values based on use_prediction
                if params['use_prediction']:
                    # ML prediction is enabled - populate ML-specific fields
                    prediction_method_value = params['prediction_method']
                    knn_k_neighbors_value = ""
                    knn_find_optimal_k_value = ""
                    subproblem_selection_method_value = ""
                    
                    if params['prediction_method'].upper() == 'KNN':
                        knn_find_optimal_k_value = params.get('knn_find_optimal_k', False)
                        subproblem_selection_method_value = "p-median"
                        
                        if knn_find_optimal_k_value:
                            # Use optimal k found during training
                            knn_k_neighbors_value = getattr(solution, "knn_optimal_k", None) or params['n_neighbors']
                        else:
                            # Use fixed k from parameters
                            knn_k_neighbors_value = params['n_neighbors']
                    elif params.get('predictor_type') in {
                        'TREE', 'PARAMETRIC_RHS_RIDGE_TREE_ANCHOR'
                    }:
                        subproblem_selection_method_value = "tree-medoid"
                    elif params.get('predictor_type') in {
                        'CLUSTERING', 'PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR'
                    }:
                        subproblem_selection_method_value = "clustering"
                    elif params.get('predictor_type') == 'PMEDIAN_WEIGHTED_TRANSFER':
                        subproblem_selection_method_value = "p-median-count-weighted"
                    elif params.get('predictor_type') == 'PMEDIAN_NEAREST_TRANSFER':
                        subproblem_selection_method_value = "p-median-nearest-representative"
                    elif params.get('predictor_type') == 'PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR':
                        subproblem_selection_method_value = "p-median-nearest-representative-rhs-ridge"
                    elif params.get('predictor_type') == 'PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN':
                        subproblem_selection_method_value = "p-median-direct-ridge-mean-anchor"
                    elif params.get('predictor_type') == 'PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_EXACTFB':
                        subproblem_selection_method_value = "p-median-direct-ridge-exact-fallback"
                    elif params.get('predictor_type') in PARAMETRIC_PREDICTORS:
                        subproblem_selection_method_value = "p-median"
                else:
                    # Traditional Benders - ML-specific fields should be blank
                    prediction_method_value = ""
                    knn_k_neighbors_value = ""
                    knn_find_optimal_k_value = ""
                    subproblem_selection_method_value = ""
                
                # Create scenario info dictionary with ALL parameters (null for irrelevant methods)
                distribution = getattr(data, "distribution", "")
                scenario_info_dict = {
                    "distribution": distribution,
                    "empirical_cv_mean": getattr(data, "empirical_cv_mean", 0.0),
                    "r_overall": getattr(data, "r_overall", 0.0),
                }
                
                # Add ALL parameters but with null values for irrelevant methods
                if distribution.lower() == "merve_normal":
                    scenario_info_dict["target_cv"] = None  # Not used for Merve_Normal
                    scenario_info_dict["sigma_lower"] = getattr(data, "sigma_lower", 0.0)
                    scenario_info_dict["sigma_upper"] = getattr(data, "sigma_upper", 0.0)
                else:
                    # CV-based methods (normal, uniform, gamma, lognormal)
                    scenario_info_dict["target_cv"] = getattr(data, "target_cv", 0.0)
                    scenario_info_dict["sigma_lower"] = None  # Not used for CV methods
                    scenario_info_dict["sigma_upper"] = None  # Not used for CV methods
                
                # Calculate unified selection and training time
                unified_time = (getattr(solution, "subproblem_selection_time_sec", None) or 0.0) + (getattr(solution, "predictor_training_time", None) or 0.0)
                parametric_diagnostics = getattr(solution, "parametric_diagnostics", None) or {}
                parametric_config = {
                    key: value
                    for key, value in params.items()
                    if str(key).startswith('parametric_')
                }
                
                # Keep full precision in the original workbook, for both methods.
                objective_value = getattr(solution, "objective_value", None)
                original_prob_obj_value = getattr(solution, "original_Prob_Obj_Value", None)

                log_entry = {
                    "problem_class": problem_class,
                    "instance_filename": instance_filename,
                    "num_subproblems": len(getattr(data, "S", [])),
                    "num_first_stage_variables": num_first_stage_variables,
                    # Method architecture parameters (from restructured config.py)
                    "method_category": params['method_category'],
                    "ml_method_type": params.get('ml_method_type', "") if params['use_prediction'] else "",
                    "ml_callback_type": params.get('ml_callback_type', "") if params['use_prediction'] else "",
                    "exact_callback_type": params.get('exact_callback_type', "") if not params['use_prediction'] else "",
                    "predictor_type": params.get('predictor_type', "") if params['use_prediction'] else "",
                    "use_prediction": params['use_prediction'],
                    "prediction_method": prediction_method_value,
                    "dual_difference_pred_loss": params.get(
                        'parametric_L_d', params.get('L_d')
                    ) if params['use_prediction'] else "",
                    "obj_difference_pred_loss": params.get(
                        'parametric_L_v', params.get('L_v')
                    ) if params['use_prediction'] else "",
                    "fallback_method": params['fallback_method'] if params['use_prediction'] else "",
                    "switching_strategy": params['switching_strategy'] if params['use_prediction'] else "",
                    "KNN_k_neighbors": knn_k_neighbors_value,
                    "knn_find_optimal_k": knn_find_optimal_k_value,
                    # tree_max_depth: removed for adaptive TREE method
                    # tree_min_leaf_size: removed for Tree-First approach (first occurrence)
                    "tree_min_impurity_decrease": params.get('tree_min_impurity_decrease', "") if params['use_prediction'] and params['prediction_method'].upper() == 'TREE' else "",
                    "tree_quantile_grid_size": params.get('tree_quantile_grid_size', "") if params['use_prediction'] and params['prediction_method'].upper() == 'TREE' else "",
                    # Clustering parameters (applicable when predictor_type is CLUSTERING)
                    "clustering_random_state": params.get('clustering_random_state', "") if params['use_prediction'] and params.get('predictor_type') == 'CLUSTERING' else "",
                    "clustering_init_method": params.get('clustering_init_method', "") if params['use_prediction'] and params.get('predictor_type') == 'CLUSTERING' else "",
                    "clustering_max_iter": params.get('clustering_max_iter', "") if params['use_prediction'] and params.get('predictor_type') == 'CLUSTERING' else "",
                    "clustering_tol": params.get('clustering_tol', "") if params['use_prediction'] and params.get('predictor_type') == 'CLUSTERING' else "",
                    "selected_subproblems_ratio": params['selected_subproblems_ratio'] if params['use_prediction'] else "",
                    "n_trials": params['n_trials'] if params['use_prediction'] else "",
                    "m_trials": params['m_trials'] if params['use_prediction'] else "",
                    "subproblem_selection_method": subproblem_selection_method_value,
                    "scenario_generation_method": getattr(data, "scenario_generation_method", None),
                    "scenario_info_dict": str(scenario_info_dict),
                    "n_walks": n_walks if params['use_prediction'] else "",
                    "walk_length": walk_length if params['use_prediction'] else "",
                    "feature_dim": params['feature_dim'] if params['use_prediction'] else "",
                    "window": window if params['use_prediction'] else "",
                    "min_count": min_count if params['use_prediction'] else "",
                    "sg": sg if params['use_prediction'] else "",
                    "seed": seed if params['use_prediction'] else "",
                    "objective_value": objective_value,
                    "original_Prob_Obj_Value": original_prob_obj_value,
                    "first_stage_optimnal_values": getattr(solution, "first_stage_optimnal_values", None),
                    "solution_time_sec": algorithm_solution_time,  # Excludes final reporting evaluation
                    "feature_gen_time_sec": getattr(solution, "feature_gen_time_sec", None) if params['use_prediction'] else "",
                    "unified_selection_training_time_sec": (unified_time if unified_time > 0 else None) if params['use_prediction'] else "",
                    "phase0_execution_time_sec": getattr(solution, "phase0_execution_time_sec", None) if params['use_prediction'] else "",
                    "phase0_solution_generation_time_sec": getattr(solution, "phase0_solution_generation_time_sec", None) if params['use_prediction'] else "",
                    "phase1_execution_time_sec": getattr(solution, "phase1_execution_time_sec", None) if params['use_prediction'] else "",
                    "phase2_execution_time_sec": (
                        getattr(solution, "phase2_execution_time_sec", None) 
                        if params['use_prediction'] and getattr(solution, "phase2_execution_time_sec", None) is not None
                        else (
                            # Calculate Phase 2 time as: solution_time - phase0_time - phase1_time
                            (getattr(solution, "solution_time", 0) or 0) - 
                            (getattr(solution, "phase0_execution_time_sec", 0) or 0) - 
                            (getattr(solution, "phase1_execution_time_sec", 0) or 0)
                            if params['use_prediction'] and 
                               getattr(solution, "solution_time", None) is not None and 
                               getattr(solution, "phase1_execution_time_sec", None) is not None
                            else ""
                        )
                    ) if params['use_prediction'] else "",
                    "num_fallback_used": getattr(solution, "num_fallback_used", None) if params['use_prediction'] else "",
                    "num_predictions": getattr(solution, "num_predictions", None) if params['use_prediction'] else "",
                    "bnb_nodes": getattr(solution, "num_bnb_nodes", None),
                    "nodes_explored": getattr(solution, "total_nodes_explored", None),
                    "cuts_mip_selected": (sum(getattr(solution, "num_cuts_mip_selected", {}).values()) if hasattr(solution, "num_cuts_mip_selected") else None) if params['use_prediction'] else "",
                    "cuts_rel_selected": (sum(getattr(solution, "num_cuts_rel_selected", {}).values()) if hasattr(solution, "num_cuts_rel_selected") else None) if params['use_prediction'] else "",
                    "cuts_mip_ml": (sum(getattr(solution, "num_cuts_mip_ml", {}).values()) if hasattr(solution, "num_cuts_mip_ml") else None) if params['use_prediction'] else "",
                    "cuts_rel_ml": (sum(getattr(solution, "num_cuts_rel_ml", {}).values()) if hasattr(solution, "num_cuts_rel_ml") else None) if params['use_prediction'] else "",
                    "cuts_mip_fallback": (sum(getattr(solution, "num_cuts_mip_fallback", {}).values()) if hasattr(solution, "num_cuts_mip_fallback") else None) if params['use_prediction'] else "",
                    "cuts_rel_fallback": (sum(getattr(solution, "num_cuts_rel_fallback", {}).values()) if hasattr(solution, "num_cuts_rel_fallback") else None) if params['use_prediction'] else "",
                    "cuts_mip_trial": sum(getattr(solution, "num_cuts_mip_trial", {}).values()) if hasattr(solution, "num_cuts_mip_trial") else None,
                    "cuts_rel_trial": sum(getattr(solution, "num_cuts_rel_trial", {}).values()) if hasattr(solution, "num_cuts_rel_trial") else None,
                    "cuts_mip_unselected_total": ((
                        sum(getattr(solution, "num_cuts_mip_ml", {}).values()) +
                        sum(getattr(solution, "num_cuts_mip_fallback", {}).values()) +
                        sum(getattr(solution, "num_cuts_mip_exact_switch", {}).values())
                    ) if hasattr(solution, "num_cuts_mip_ml") and hasattr(solution, "num_cuts_mip_fallback") else None) if params['use_prediction'] else "",
                    "cuts_rel_unselected_total": ((
                        sum(getattr(solution, "num_cuts_rel_ml", {}).values()) +
                        sum(getattr(solution, "num_cuts_rel_fallback", {}).values()) +
                        sum(getattr(solution, "num_cuts_rel_exact_switch", {}).values())
                    ) if hasattr(solution, "num_cuts_rel_ml") and hasattr(solution, "num_cuts_rel_fallback") else None) if params['use_prediction'] else "",
                    "cuts_mip_exact_switch": (sum(getattr(solution, "num_cuts_mip_exact_switch", {}).values()) if hasattr(solution, "num_cuts_mip_exact_switch") else None) if params['use_prediction'] else "",
                    "cuts_rel_exact_switch": (sum(getattr(solution, "num_cuts_rel_exact_switch", {}).values()) if hasattr(solution, "num_cuts_rel_exact_switch") else None) if params['use_prediction'] else "",
                    "phase0_cuts_added": getattr(solution, "phase0_cuts_added", None),
                    "num_switch_events": getattr(solution, "num_switch_events", None) if params['use_prediction'] else "",
                    "switch_iterations": str(getattr(solution, "switch_iterations", [])) if params['use_prediction'] else "",  # Convert list to string for Excel
                    "permanently_switched": getattr(solution, "permanently_switched", None) if params['use_prediction'] else "",
                    "time_limit_sec": getattr(solution, "time_limit_sec", None),  # Time limit set in Gurobi parameters
                    "stopping_reason": stopping_reason,
                    "iteration": getattr(solution, "iteration", None),
                    "use_covariates": (
                        params['use_covariates']
                        if params['use_prediction'] and params.get('ml_method_type') == 'ML_Augmented_BD'
                        else None
                    ),
                    "use_stochastic_capacity": params['use_stochastic_capacity'],
                    "scmnd_formulation": params.get('scmnd_formulation', 'MANUSCRIPT_ENVELOPE'),
                    "feature_transition_rule": feature_transition_rule if params['use_prediction'] else "",
                    "feature_transition_temperature": feature_transition_temperature if params['use_prediction'] else "",
                    "feature_stability_epsilon": feature_stability_epsilon if params['use_prediction'] else "",
                    "parametric_config": json.dumps(parametric_config, default=str, sort_keys=True) if parametric_config else "",
                    "parametric_anchor_method": parametric_diagnostics.get('anchor_method'),
                    "parametric_direct_clip_negative": parametric_diagnostics.get('direct_clip_negative'),
                    "parametric_direct_anchor_shrink": parametric_diagnostics.get('direct_anchor_shrink'),
                    "parametric_direct_negative_clipped": parametric_diagnostics.get('direct_negative_clipped'),
                    "parametric_direct_gamma_evaluations": parametric_diagnostics.get('direct_gamma_evaluations'),
                    "parametric_direct_gamma_positive": parametric_diagnostics.get('direct_gamma_positive'),
                    "parametric_direct_exact_fallback_requested": parametric_diagnostics.get('direct_exact_fallback_requested'),
                    "parametric_objective_guard_enabled": parametric_diagnostics.get('objective_guard_enabled'),
                    "parametric_objective_guard_checks": parametric_diagnostics.get('objective_guard_checks'),
                    "parametric_objective_guard_accepted": parametric_diagnostics.get('objective_guard_accepted'),
                    "parametric_objective_guard_rejected": parametric_diagnostics.get('objective_guard_rejected'),
                    "parametric_objective_guard_rejected_negative": parametric_diagnostics.get('objective_guard_rejected_negative'),
                    "parametric_objective_guard_total_improvement": parametric_diagnostics.get('objective_guard_total_improvement'),
                    "parametric_objective_guard_gamma_zero_fastpath": parametric_diagnostics.get('objective_guard_gamma_zero_fastpath'),
                    "parametric_objective_guard_skipped_zero": parametric_diagnostics.get('objective_guard_skipped_zero'),
                    "parametric_objective_guard_skipped_equal_anchor": parametric_diagnostics.get('objective_guard_skipped_equal_anchor'),
                    "parametric_objective_guard_evaluated": parametric_diagnostics.get('objective_guard_evaluated'),
                    "parametric_objective_guard_time_sec": parametric_diagnostics.get('objective_guard_time_sec'),
                    "parametric_repair_mode_actual": parametric_diagnostics.get('repair_mode'),
                    "parametric_structure_repair_attempts": parametric_diagnostics.get('structure_repair_attempts'),
                    "parametric_structure_repair_feasible_candidates": parametric_diagnostics.get('structure_repair_feasible_candidates'),
                    "parametric_structure_repair_anchor_fallbacks": parametric_diagnostics.get('structure_repair_anchor_fallbacks'),
                    "parametric_structure_repair_nonanchor_candidates": parametric_diagnostics.get('structure_repair_nonanchor_candidates'),
                    "parametric_structure_repair_positive_objectives": parametric_diagnostics.get('structure_repair_positive_objectives'),
                    "parametric_structure_repair_positive_gain_total": parametric_diagnostics.get('structure_repair_positive_gain_total'),
                    "parametric_structure_repair_selected_kinds": json.dumps(parametric_diagnostics.get('structure_repair_selected_kinds', {}), sort_keys=True),
                    "parametric_diagnostic_probe_enabled": parametric_diagnostics.get('diagnostic_probe_enabled'),
                    "parametric_diag_gamma_zero": parametric_diagnostics.get('diag_gamma_zero'),
                    "parametric_diag_gamma_positive_count": parametric_diagnostics.get('diag_gamma_positive_count'),
                    "parametric_diag_gamma_positive_mean": parametric_diagnostics.get('diag_gamma_positive_mean'),
                    "parametric_diag_gamma_max": parametric_diagnostics.get('diag_gamma_max'),
                    "parametric_diag_gamma_gt_1e12": parametric_diagnostics.get('diag_gamma_gt_1e12'),
                    "parametric_diag_gamma_gt_1e8": parametric_diagnostics.get('diag_gamma_gt_1e8'),
                    "parametric_diag_gamma_gt_1e6": parametric_diagnostics.get('diag_gamma_gt_1e6'),
                    "parametric_diag_gamma_gt_1e3": parametric_diagnostics.get('diag_gamma_gt_1e3'),
                    "parametric_diag_gamma_gt_01": parametric_diagnostics.get('diag_gamma_gt_01'),
                    "parametric_diag_probe_checks": parametric_diagnostics.get('diag_probe_checks'),
                    "parametric_diag_probe_improved": parametric_diagnostics.get('diag_probe_improved'),
                    "parametric_diag_probe_negative": parametric_diagnostics.get('diag_probe_negative'),
                    "parametric_diag_probe_tied": parametric_diagnostics.get('diag_probe_tied'),
                    "parametric_diag_probe_positive_total": parametric_diagnostics.get('diag_probe_positive_total'),
                    "parametric_diag_probe_negative_total": parametric_diagnostics.get('diag_probe_negative_total'),
                    "parametric_diag_probe_max_positive": parametric_diagnostics.get('diag_probe_max_positive'),
                    "parametric_diag_probe_anchor_check_time_sec": parametric_diagnostics.get('diag_probe_anchor_check_time_sec'),
                    "parametric_diag_probe_objective_time_sec": parametric_diagnostics.get('diag_probe_objective_time_sec'),
                    "parametric_diag_probe_total_time_sec": parametric_diagnostics.get('diag_probe_total_time_sec'),
                    "parametric_training_calls": parametric_diagnostics.get('training_calls'),
                    "parametric_training_failures": parametric_diagnostics.get('training_failures'),
                    "parametric_training_time_sec": parametric_diagnostics.get('training_time_total_sec'),
                    "parametric_prediction_time_sec": parametric_diagnostics.get('prediction_time_total_sec'),
                    "parametric_feasibility_time_sec": parametric_diagnostics.get('feasibility_time_total_sec'),
                    "parametric_repair_time_sec": parametric_diagnostics.get('repair_time_total_sec'),
                    "parametric_prediction_opportunities": parametric_diagnostics.get('prediction_opportunities'),
                    "parametric_raw_predictions": parametric_diagnostics.get('raw_predictions'),
                    "parametric_raw_infeasible_predictions": parametric_diagnostics.get('raw_infeasible_predictions'),
                    "parametric_raw_infeasible_rate": parametric_diagnostics.get('raw_infeasible_rate'),
                    "parametric_repaired_predictions": parametric_diagnostics.get('repaired_predictions'),
                    "parametric_exact_fallback_subproblems": parametric_diagnostics.get('exact_fallback_subproblems'),
                    "parametric_exact_fallback_rate": parametric_diagnostics.get('exact_fallback_rate'),
                    "parametric_selected_exact_subproblems": parametric_diagnostics.get('selected_exact_subproblems'),
                    "parametric_total_exact_subproblems": parametric_diagnostics.get('total_exact_subproblems'),
                    "parametric_algorithm_exact_subproblem_solves": parametric_diagnostics.get('algorithm_exact_subproblem_solves'),
                    "parametric_algorithm_exact_subproblem_time_sec": parametric_diagnostics.get('algorithm_exact_subproblem_time_sec'),
                    "parametric_reevaluation_exact_subproblem_solves": parametric_diagnostics.get('reevaluation_exact_subproblem_solves'),
                    "parametric_reevaluation_exact_subproblem_time_sec": parametric_diagnostics.get('reevaluation_exact_subproblem_time_sec'),
                    "parametric_cuts_added": parametric_diagnostics.get('parametric_cuts_added'),
                    "parametric_cut_addition_rate": parametric_diagnostics.get('parametric_cut_addition_rate'),
                    "parametric_mean_cut_violation": parametric_diagnostics.get('mean_cut_violation'),
                    "parametric_max_cut_violation": parametric_diagnostics.get('max_cut_violation'),
                    "parametric_mean_repair_gamma": parametric_diagnostics.get('mean_repair_gamma'),
                    "parametric_min_repair_gamma": parametric_diagnostics.get('min_repair_gamma'),
                    "parametric_max_condition_number": parametric_diagnostics.get('max_condition_number'),
                    "parametric_pmedian_time_sec": parametric_diagnostics.get('pmedian', {}).get('runtime_sec'),
                    "parametric_pmedian_objective": parametric_diagnostics.get('pmedian', {}).get('objective'),
                    "parametric_pmedian_status": parametric_diagnostics.get('pmedian', {}).get('status'),
                    "parametric_iteration_records": json.dumps(
                        parametric_diagnostics.get('iteration_records', []),
                        default=str,
                    ) if parametric_diagnostics else "",
                    "ObjBound_LB": getattr(solution, "ObjBound_LB", None),
                    # Task 1: Point signature tracking
                    "total_unique_points": getattr(solution, "total_unique_points", 0),
                    "total_repeated_hits": getattr(solution, "total_repeated_hits", 0),
                    "repeated_hits_with_cuts": getattr(solution, "repeated_hits_with_cuts", 0),
                    # Task 2: Root vs non-root callback tracking
                    "mipsol_root": getattr(solution, "callback_counts", {}).get('mipsol_root', 0) if hasattr(solution, "callback_counts") and solution.callback_counts else 0,
                    "mipsol_nonroot": getattr(solution, "callback_counts", {}).get('mipsol_nonroot', 0) if hasattr(solution, "callback_counts") and solution.callback_counts else 0,
                    "mipnode_root": getattr(solution, "callback_counts", {}).get('mipnode_root', 0) if hasattr(solution, "callback_counts") and solution.callback_counts else 0,
                    "mipnode_nonroot": getattr(solution, "callback_counts", {}).get('mipnode_nonroot', 0) if hasattr(solution, "callback_counts") and solution.callback_counts else 0,
                    "avg_mipsol_per_nonroot_node": getattr(solution, "avg_mipsol_per_nonroot_node", 0.0),
                    "avg_mipnode_per_nonroot_node": getattr(solution, "avg_mipnode_per_nonroot_node", 0.0),
                    # Retired historical-candidate diagnostics; no values fabricated.
                    "num_solutions_reevaluated": 0,
                    "max_objective_drift": None,
                    "mean_objective_drift": None,
                    "std_objective_drift": None,
                    # Gurobi parameters logging
                    "MIPGap_target": getattr(solution, "mip_gap_target", None),  # Target MIPGap value set as Gurobi parameter
                    "MIPGap_Stop": getattr(solution, "mip_gap_stop", None),  # MIPGap value when solver stopped
                    "convergence_trace_path": trace_path,
                    "verified_final_relative_gap": last_certified_gap,
                    "experiment_solve_wall_time_sec": t_sol,
                    "master_solver_internal_time_sec": buckets.get("master_solver_internal_sec", 0.0),
                    "callback_exact_subproblem_time_sec": buckets.get("callback_exact_subproblem_sec", 0.0),
                    "callback_other_time_sec": buckets.get("callback_other_sec", 0.0),
                    "experiment_outside_solver_time_sec": outside_solver,
                    "solver_component_breakdown_json": json.dumps(component_breakdown, sort_keys=True),
                    "solver_component_sum_sec": sum(float(v) for v in component_breakdown.values()),
                    "gurobi_solver_runtime_sec": gurobi_solver_runtime,
                    "master_optimize_wall_time_sec": getattr(solution, "master_optimize_wall_time_sec", None),
                    "data_loading_time_sec": data_loading,
                    "final_objective_validation_time_sec": final_validation,
                    "candidate_reevaluation_time_sec": candidate_recheck,
                    "model_build_and_other_time_sec": remainder,
                    "end_to_end_component_breakdown_json": json.dumps(timed_components, sort_keys=True),
                    "end_to_end_component_sum_sec": end_to_end_sum,
                    "end_to_end_component_reconciliation_error_sec": end_to_end_sum - t_sol,
                    "end_to_end_component_counts_json": json.dumps(component_counts, sort_keys=True),
                    "timing_phase_tags_json": json.dumps(TIMING_PHASE_TAGS, sort_keys=True),
                    "timing_schema_version": TIMING_SCHEMA_VERSION,
                    "solution_time_metric_version": SOLUTION_TIME_METRIC_VERSION,
                    "end_to_end_wall_time_sec": t_sol,
                    "solution_component_sum_sec": end_to_end_sum - final_validation,
                    "final_exact_evaluation_time_sec": final_validation,
                    "final_exact_evaluation_count": solution.final_exact_evaluation_count,
                    "final_exact_subproblem_solves": solution.final_exact_subproblem_solves,
                    "final_exact_evaluation_status": solution.final_exact_evaluation_status,
                    "convergence_trace_schema_version": TRACE_SCHEMA_VERSION,
                    **(solution.cut_construction_breakdown or {}),
                    **(solution.gurobi_cut_statistics or {}),
                }
                
                # Add category classification columns for better experiment organization and analysis
                from utils.config_utils import get_pred_or_exact, get_approach_category, get_feature_generation_method
                pred_or_exact = get_pred_or_exact(params)
                approach_category = get_approach_category(params)
                feature_gen_method = get_feature_generation_method(params)
                
                log_entry["pred_or_exact"] = pred_or_exact
                log_entry["approach_category"] = approach_category
                log_entry["feature_generation_method"] = feature_gen_method
                
                # Save experiment result incrementally to Excel
                save_experiment_result_incrementally(log_entry, solving_log_path)

                # Save feature vectors if they exist to CSV
                if hasattr(solution, "feature_vectors") and isinstance(solution.feature_vectors, dict) and len(solution.feature_vectors) > 0 and params['use_prediction']:
                    save_features_to_csv(
                        solution, params, problem_class, instance_filename, 
                        num_first_stage_variables, feature_csv_dir, solution_time_sec=algorithm_solution_time
                    )
            else:
                failed_experiments += 1
                # Determine ML-specific values for failure case based on use_prediction
                if params['use_prediction']:
                    # ML prediction is enabled - populate ML-specific fields
                    prediction_method_value = params['prediction_method']
                    knn_k_neighbors_value = ""
                    knn_find_optimal_k_value = ""
                    subproblem_selection_method_value = ""
                    
                    if params['prediction_method'].upper() == 'KNN':
                        knn_find_optimal_k_value = params.get('knn_find_optimal_k', False)
                        subproblem_selection_method_value = "p-median"
                        knn_k_neighbors_value = params['n_neighbors']  # Use parameter since no solution
                    elif params['prediction_method'].upper() == 'TREE':
                        subproblem_selection_method_value = "tree-medoid"
                        # KNN fields remain empty for TREE method
                else:
                    # Traditional Benders - ML-specific fields should be blank
                    prediction_method_value = ""
                    knn_k_neighbors_value = ""
                    knn_find_optimal_k_value = ""
                    subproblem_selection_method_value = ""
                
                # Create scenario info dictionary for failure case with ALL parameters (null for irrelevant methods)
                distribution = getattr(data, "distribution", "")
                scenario_info_dict = {
                    "distribution": distribution,
                    "empirical_cv_mean": getattr(data, "empirical_cv_mean", 0.0),
                    "r_overall": getattr(data, "r_overall", 0.0),
                }
                
                # Add ALL parameters but with null values for irrelevant methods
                if distribution.lower() == "merve_normal":
                    scenario_info_dict["target_cv"] = None  # Not used for Merve_Normal
                    scenario_info_dict["sigma_lower"] = getattr(data, "sigma_lower", 0.0)
                    scenario_info_dict["sigma_upper"] = getattr(data, "sigma_upper", 0.0)
                else:
                    # CV-based methods (normal, uniform, gamma, lognormal)
                    scenario_info_dict["target_cv"] = getattr(data, "target_cv", 0.0)
                    scenario_info_dict["sigma_lower"] = None  # Not used for CV methods
                    scenario_info_dict["sigma_upper"] = None  # Not used for CV methods
                
                log_entry = {
                    "problem_class": problem_class,
                    "instance_filename": instance_filename,
                    "num_subproblems": len(getattr(data, "S", [])),
                    "num_first_stage_variables": num_first_stage_variables,
                    # Method architecture parameters (from restructured config.py)
                    "method_category": params['method_category'],
                    "ml_method_type": params.get('ml_method_type', "") if params['use_prediction'] else "",
                    "ml_callback_type": params.get('ml_callback_type', "") if params['use_prediction'] else "",
                    "exact_callback_type": params.get('exact_callback_type', "") if not params['use_prediction'] else "",
                    "predictor_type": params.get('predictor_type', "") if params['use_prediction'] else "",
                    "use_prediction": params['use_prediction'],
                    "prediction_method": prediction_method_value,
                    "dual_difference_pred_loss": params.get('L_d', None) if params['use_prediction'] else "",
                    "obj_difference_pred_loss": params.get('L_v', None) if params['use_prediction'] else "",
                    "fallback_method": params['fallback_method'] if params['use_prediction'] else "",
                    "switching_strategy": params['switching_strategy'] if params['use_prediction'] else "",
                    "KNN_k_neighbors": knn_k_neighbors_value,
                    "knn_find_optimal_k": knn_find_optimal_k_value,
                    # tree_max_depth: removed for adaptive TREE method
                    # tree_min_leaf_size: removed for Tree-First approach (second occurrence)
                    "tree_min_impurity_decrease": params.get('tree_min_impurity_decrease', "") if params['use_prediction'] and params['prediction_method'].upper() == 'TREE' else "",
                    "tree_quantile_grid_size": params.get('tree_quantile_grid_size', "") if params['use_prediction'] and params['prediction_method'].upper() == 'TREE' else "",
                    # Clustering parameters (applicable when predictor_type is CLUSTERING)
                    "clustering_random_state": params.get('clustering_random_state', "") if params['use_prediction'] and params.get('predictor_type') == 'CLUSTERING' else "",
                    "clustering_init_method": params.get('clustering_init_method', "") if params['use_prediction'] and params.get('predictor_type') == 'CLUSTERING' else "",
                    "clustering_max_iter": params.get('clustering_max_iter', "") if params['use_prediction'] and params.get('predictor_type') == 'CLUSTERING' else "",
                    "clustering_tol": params.get('clustering_tol', "") if params['use_prediction'] and params.get('predictor_type') == 'CLUSTERING' else "",
                    "selected_subproblems_ratio": params['selected_subproblems_ratio'] if params['use_prediction'] else "",
                    "n_trials": params['n_trials'] if params['use_prediction'] else "",
                    "m_trials": params['m_trials'] if params['use_prediction'] else "",
                    "subproblem_selection_method": subproblem_selection_method_value,
                    "scenario_generation_method": getattr(data, "scenario_generation_method", None),
                    "scenario_info_dict": str(scenario_info_dict),
                    "n_walks": n_walks if params['use_prediction'] else "",
                    "walk_length": walk_length if params['use_prediction'] else "",
                    "feature_dim": params['feature_dim'] if params['use_prediction'] else "",
                    "window": window if params['use_prediction'] else "",
                    "min_count": min_count if params['use_prediction'] else "",
                    "sg": sg if params['use_prediction'] else "",
                    "seed": seed if params['use_prediction'] else "",
                    "objective_value": None,
                    "original_Prob_Obj_Value": None,  # Both None for failure case
                    "first_stage_optimnal_values": None,
                    "solution_time_sec": None,
                    "feature_gen_time_sec": None if params['use_prediction'] else "",
                    "unified_selection_training_time_sec": None if params['use_prediction'] else "",
                    "phase0_execution_time_sec": None if params['use_prediction'] else "",
                    "phase0_solution_generation_time_sec": None if params['use_prediction'] else "",
                    "phase1_execution_time_sec": None if params['use_prediction'] else "",
                    "phase2_execution_time_sec": None if params['use_prediction'] else "",
                    "num_fallback_used": None if params['use_prediction'] else "",
                    "num_predictions": None if params['use_prediction'] else "",
                    "bnb_nodes": None,
                    "nodes_explored": None,
                    "cuts_mip_selected": None if params['use_prediction'] else "",
                    "cuts_rel_selected": None if params['use_prediction'] else "",
                    "cuts_mip_ml": None if params['use_prediction'] else "",
                    "cuts_rel_ml": None if params['use_prediction'] else "",
                    "cuts_mip_fallback": None if params['use_prediction'] else "",
                    "cuts_rel_fallback": None if params['use_prediction'] else "",
                    "cuts_mip_trial": None,
                    "cuts_rel_trial": None,
                    "cuts_mip_unselected_total": None if params['use_prediction'] else "",
                    "cuts_rel_unselected_total": None if params['use_prediction'] else "",
                    "cuts_mip_exact_switch": None if params['use_prediction'] else "",
                    "cuts_rel_exact_switch": None if params['use_prediction'] else "",
                    "phase0_cuts_added": None,
                    "num_switch_events": None if params['use_prediction'] else "",
                    "switch_iterations": None if params['use_prediction'] else "",
                    "permanently_switched": None if params['use_prediction'] else "",
                    "time_limit_sec": None,  # Time limit - None for failure case
                    "stopping_reason": f"CODE_ERROR_{error_type}" if error_type else "SOLUTION_NULL",
                    "iteration": None,
                    "use_covariates": (
                        params['use_covariates']
                        if params['use_prediction'] and params.get('ml_method_type') == 'ML_Augmented_BD'
                        else None
                    ),
                    "use_stochastic_capacity": params['use_stochastic_capacity'],
                    "ObjBound_LB": None,
                    # Task 1: Point signature tracking
                    "total_unique_points": 0,
                    "total_repeated_hits": 0,
                    "repeated_hits_with_cuts": 0,
                    # Task 2: Root vs non-root callback tracking
                    "mipsol_root": 0,
                    "mipsol_nonroot": 0,
                    "mipnode_root": 0,
                    "mipnode_nonroot": 0,
                    "avg_mipsol_per_nonroot_node": 0.0,
                    "avg_mipnode_per_nonroot_node": 0.0,
                    # Retired historical-candidate diagnostics; no values fabricated.
                    "num_solutions_reevaluated": 0,
                    "max_objective_drift": None,
                    "mean_objective_drift": None,
                    "std_objective_drift": None,
                    # Gurobi parameters logging
                    "MIPGap_target": None,  # MIPGap target - None for failure case
                    "MIPGap_Stop": None,  # MIPGap Stop - None for failure case
                }
                
                # Add category classification columns for better experiment organization (failure case)
                from utils.config_utils import get_pred_or_exact, get_approach_category, get_feature_generation_method
                pred_or_exact = get_pred_or_exact(params)
                approach_category = get_approach_category(params)
                feature_gen_method = get_feature_generation_method(params)
                
                log_entry["pred_or_exact"] = pred_or_exact
                log_entry["approach_category"] = approach_category
                log_entry["feature_generation_method"] = feature_gen_method
                
                # Save failed experiment result incrementally
                save_experiment_result_incrementally(log_entry, solving_log_path)

    # Save per-iteration training diagnostics (REG only) if any data exists
    if TRAINING_CALL_LOGS:
        training_log_path = os.path.join(run_results_root, "training_diagnostics.xlsx")
        try:
            pd.DataFrame(TRAINING_CALL_LOGS).to_excel(training_log_path, index=False)
            logger.info(f"\n Training diagnostics saved to: {training_log_path}")
        except PermissionError:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            alt_tlog_path = os.path.join(run_results_root, f"training_diagnostics_{timestamp}.xlsx")
            pd.DataFrame(TRAINING_CALL_LOGS).to_excel(alt_tlog_path, index=False)
            logger.info(f"\n Training diagnostics saved to: {alt_tlog_path} (original file was locked)")
    
    logger.info(f"\nExperimental execution completed. All results saved incrementally to {solving_log_path}")
    logger.info(f"Feature vectors saved incrementally to: {feature_csv_dir}")

    # Preserve all saved results, while reporting code failures accurately to
    # Slurm (which cannot infer failures from Excel stopping_reason).
    if failed_experiments:
        logger.error(f"{failed_experiments} experiment(s) failed; see solution_log.xlsx for CODE_ERROR_/SOLUTION_NULL entries.")
        update_task_status("FAILED")
        raise SystemExit(1)

    update_task_status("COMPLETED")


if __name__ == "__main__":
    main()
