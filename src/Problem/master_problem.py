"""
master_problem.py

This module implements the master problem formulations for various optimization problems
in the ML-augmented Benders decomposition framework. It serves as the central orchestrator
for the decomposition algorithm, managing first-stage decisions and coordinating with
the callback mechanism for cut generation.

Overview:
---------
The master problem represents the first-stage decisions and recourse estimates (theta variables)
in the Benders decomposition. This module supports multiple problem types including:
- Uncapacitated Facility Location (UFL)
- Uncapacitated Hub Location (UHL) 
- Stochastic Capacitated Facility Location Problem (SCFLP)
- Stochastic Service Location Problem (SSLP)
- Capacitated Multicommodity Network Design (CMND) [planned]
- Maximum Coverage Facility Location (MCFL) [planned]

Key Components:
---------------

1. **Solution Dataclass**: 
   - Comprehensive data structure capturing all solution metrics
   - Timing statistics for each phase of the algorithm
   - Cut generation statistics broken down by source and type
   - Feature vectors and prediction performance metrics
   - Switching strategy statistics for enhanced algorithms

2. **Master Problem Formulations**:
   - Problem-specific mathematical formulations
   - First-stage decision variables (facility locations, hub assignments, etc.)
   - Second-stage recourse variables (theta) representing expected costs
   - Constraint definitions specific to each problem type

3. **Callback Integration**:
   - Instantiation and configuration of ML-augmented callbacks supporting k-NN, TREE, and DUAL_CLUSTERING methods
   - Parameter passing for prediction methods and learning configurations
   - Cut generation coordination between master and subproblems

Architecture:
-------------
The module follows a factory pattern where `solve_master_problem()` dispatches to
problem-specific formulations based on the problem type. Each formulation:

1. **Model Setup**: Creates Gurobi model with appropriate variables and constraints
2. **Callback Configuration**: Instantiates Callback class with problem-specific parameters
3. **Optimization**: Solves with lazy constraint generation via callbacks
4. **Result Processing**: Extracts solution statistics and performance metrics

Problem-Specific Formulations:
------------------------------

**UFL (Uncapacitated Facility Location)**:
- Variables: y_j (facility opening), theta_i (recourse for customer i)
- Constraints: At least one facility must be opened
- Objective: Minimize fixed facility costs + expected service costs

**SCFLP (Stochastic Capacitated Facility Location)**:
- Variables: y_j (facility opening), theta_s (recourse for scenario s)
- Constraints: Facility capacity and demand satisfaction
- Objective: Minimize facility costs + expected transportation/penalty costs

**SSLP (Stochastic Service Location Problem)**:
- Variables: y_j (facility opening), theta_s (recourse for scenario s)
- Constraints: Service level requirements and capacity constraints
- Objective: Minimize opening costs + expected service/penalty costs

**UHL (Uncapacitated Hub Location)**:
- Variables: x_ij (hub assignments), theta_s (recourse for OD pairs)
- Constraints: Hub opening, flow conservation, capacity
- Objective: Minimize hub costs + expected transportation costs

Integration with ML Framework:
------------------------------
The master problem integrates with the ML-augmented Benders framework through:

1. **Feature Engineering**: Passes feature generation parameters to callbacks
2. **Prediction Models**: Configures k-NN, TREE, and DUAL_CLUSTERING models for dual prediction
3. **Subproblem Selection**: Manages integrated selection strategies for each prediction method
4. **Switching Strategies**: Implements adaptive exact/ML switching logic
5. **Performance Tracking**: Collects comprehensive timing and cut statistics

Callback Coordination:
----------------------
The Callback class receives:
- Problem data and first-stage variables
- ML configuration (prediction method, neighbors, trials)
- Learning parameters (feature dimensions, selection methods)
- Switching strategy configuration
- Budget parameters for training (L_d, L_v, p_d, p_v)

Solution Statistics:
--------------------
The Solution dataclass captures:
- **Timing**: Feature generation, subproblem selection, training times
- **Cut Analysis**: Detailed breakdown by source (ML, exact, fallback) and type (MIP/relaxation)
- **ML Performance**: Prediction accuracy, fallback usage, training diagnostics
- **Switching Behavior**: Event tracking, iteration logs, permanent switches
- **B&B Statistics**: Node counts, exploration metrics

Use Cases:
----------
This module is the primary entry point for solving large-scale stochastic optimization
problems where traditional Benders decomposition becomes computationally prohibitive.
The ML augmentation enables:
- Significant reduction in subproblem solving time
- Scalability to larger instance sizes
- Adaptive algorithm behavior based on problem characteristics
- Comprehensive performance analysis and debugging capabilities

Performance Considerations:
---------------------------
- Lazy constraint generation minimizes master problem size
- Feature caching reduces computational overhead
- Adaptive switching strategies balance accuracy and efficiency
- Detailed logging enables performance optimization

Dependencies:
-------------
- Problem.data: Data structures for all problem types
- Problem.callbacks: ML-augmented callback implementation
- gurobipy: Mathematical optimization solver
- Dual_Prediction.training: Regression model training
- feature_engineering: Graph-based feature generation
- subproblem_selection: Intelligent scenario selection

Error Handling:
---------------
Comprehensive error handling for:
- Infeasible/unbounded models
- Solver failures and timeouts
- Invalid problem configurations
- Missing data or parameter validation

Extensions:
-----------
The framework is designed for easy extension to new problem types:
1. Add data class to Problem.data
2. Implement master formulation following existing patterns
3. Add subproblem dual formulation to subproblem_dual.py
4. Update dispatcher in solve_master_problem()

"""

from dataclasses import dataclass
import sys
import os

# Add the src directory to the Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Problem.callbacks import Callback
from Problem.data import (
    SCFLPData,
    NESPData,
    SUMNDData,
    SCMNDData,
    SCMND_FORMULATION_CURRENT,
    SCMND_FORMULATION_MANUSCRIPT,
    scmnd_envelope_demands,
)
from gurobipy import GRB, Model, quicksum
import numpy as np
from typing import Optional
import logging
import time
from utils.config_utils import normalize_use_prediction

# Set up logging
logger = logging.getLogger(__name__)

WRITE_MP_LP = False  # Set to True if you want to write the master problem LP file


def _log_final_master_solution(problem_type: str, objective_value, first_stage_values) -> None:
    """Log solved master objective and first-stage variables with visual separators."""
    separator = "=" * 80
    logger.info(separator)
    logger.info("MASTER PROBLEM SOLVED | problem_type=%s", problem_type)
    logger.info("Objective value: %s", objective_value)
    if isinstance(first_stage_values, dict):
        sorted_values = sorted(first_stage_values.items(), key=lambda item: str(item[0]))
        logger.info("First-stage variables (%d): %s", len(sorted_values), sorted_values)
    else:
        logger.info("First-stage variables: %s", first_stage_values)
    logger.info(separator)

@dataclass
class Solution:
    """
    Comprehensive solution dataclass for ML-augmented Benders decomposition results.
    
    This dataclass encapsulates all solution statistics, performance metrics, and diagnostic
    information collected during the execution of the ML-augmented Benders decomposition
    algorithm. It serves as the primary data structure for result analysis, experimental
    evaluation, and performance monitoring.
    
    Attributes:
    -----------
    **Solution Statistics**:
        objective_value (float): Final optimal objective value of the master problem
        original_Prob_Obj_Value (Optional[float]): True objective value of the original problem
            - When use_prediction="none": None (objective_value is already the true value)
            - When use_prediction != "none": Calculated by solving all subproblems exactly with the first-stage solution
            - Represents the actual first-stage cost + true second-stage cost without ML approximations
        first_stage_optimnal_values (object): Optimal first-stage decision variable values
            - UFL: Facility opening decisions {facility_j: binary_value}
            - UHL: Hub opening decisions {(node_i, node_j): binary_value}
            - SCFLP/SSLP: Facility/service opening decisions {facility_j: binary_value}
    
    **Performance Timing**:
        solution_time (Optional[float]): Gurobi Runtime (legacy internal field); main.py exports the separate algorithm solution_time_sec
        feature_gen_time_sec (Optional[float]): Time spent on scenario feature generation (seconds)
        subproblem_selection_time_sec (Optional[float]): Time for subproblem selection strategy (seconds)
        predictor_training_time (Optional[float]): Time for ML model training (seconds) - unified for all methods (REG, KNN, TREE, SIMPLE_AVG)
        KNN_closest_neighbor_time (Optional[float]): Deprecated - now unified with predictor_training_time (seconds)
    
    **Cut Generation Statistics**:
        All cut statistics are dictionaries with iteration numbers as keys and counts as values.
        
        *Selected Subproblem Cuts*:
            num_cuts_mip_selected (dict): Cuts from exactly solved selected subproblems at MIP nodes
            num_cuts_rel_selected (dict): Cuts from exactly solved selected subproblems at LP relaxation nodes
        
        *Unified ML Cuts*:
            num_cuts_mip_ml (dict): Cuts from ML predictions at MIP nodes (all methods: REG, KNN, TREE, SIMPLE_AVG)
            num_cuts_rel_ml (dict): Cuts from ML predictions at LP relaxation nodes (all methods: REG, KNN, TREE, SIMPLE_AVG)
        
        *Unselected Subproblem Cuts*:
            num_cuts_mip_unselected (dict): Cuts from unselected subproblems (via ML) at MIP nodes
            num_cuts_rel_unselected (dict): Cuts from unselected subproblems (via ML) at LP relaxation nodes
        
        *Fallback Cuts*:
            num_cuts_mip_fallback (dict): Cuts from fallback exact solving at MIP nodes
            num_cuts_rel_fallback (dict): Cuts from fallback exact solving at LP relaxation nodes
        
        *Trial/Feature Generation Cuts*:
            num_cuts_mip_trial (dict): Cuts from trial solves for feature generation at MIP nodes
            num_cuts_rel_trial (dict): Cuts from trial solves for feature generation at LP relaxation nodes
        
        *Switching Strategy Cuts*:
            num_cuts_mip_exact_switch (dict): Cuts after switching to exact mode at MIP nodes
            num_cuts_rel_exact_switch (dict): Cuts after switching to exact mode at LP relaxation nodes
    
    **Feature Engineering**:
        covariate_vectors (dict): Computed scenario covariate embeddings
            Format: {scenario_id: numpy.array([feature_1, feature_2, ..., feature_dim])}
            Generated by: feature_engineering.feature_generation.compute_scenario_features_from_duals()
            Used by: ML prediction models for dual variable estimation
    
    **Algorithm State Tracking**:
        iteration (int): Current Benders iteration number
        num_fallback_used (int): Total number of fallback exact solves triggered
        num_predictions (int): Total number of ML predictions made during execution
    
    **Switching Strategy Tracking**:
        num_switch_events (int): Number of times algorithm switched from ML to exact mode
        switch_iterations (list): List of iteration numbers where switching events occurred
        permanently_switched (bool): Flag indicating if algorithm has permanently switched to exact mode
    
    **Branch-and-Bound Statistics**:
        num_bnb_nodes (int): Total number of branch-and-bound nodes processed in master problem
        total_nodes_explored (int): Cumulative count of all nodes explored across all iterations
    
    Integration Points:
    -------------------
    **Input Sources**:
        - master_problem.solve_master_problem(): Primary solution statistics and timing
        - callbacks.py: Cut statistics and algorithm state tracking
        - feature_generation.py: Feature vector computation and timing
        - subproblem_selection/: Subproblem selection timing and statistics
    
    **Output Destinations**:
        - main.py: Result collection and Excel export generation
        - Research analysis: Performance evaluation and algorithm comparison
        - Logging system: Diagnostic information and debugging
    
    **Usage in Experimental Framework**:
        The Solution object is returned by solve_master_problem() and collected in main.py
        for batch experimental evaluation across multiple problem instances and parameter
        configurations.
    
    Example Usage:
    --------------
    ```python
    solution = solve_master_problem(
        'UFL', data, selected_subproblems, feature_vectors, 
        n_neighbors=5, prediction_method='REG'
    )
    
    print(f"Optimal value: {solution.objective_value}")
    print(f"Solution time: {solution.solution_time} seconds")
    print(f"ML predictions made: {solution.num_predictions}")
    print(f"Fallback exact solves: {solution.num_fallback_used}")
    ```
    """
    # Solution statistics
    objective_value: float
    first_stage_optimnal_values: object
    # Timing statistics
    solution_time: Optional[float]
    feature_gen_time_sec: Optional[float]
    subproblem_selection_time_sec: Optional[float]
    predictor_training_time: Optional[float]  # Unified training time for all ML methods (REG, KNN, TREE, SIMPLE_AVG)
    KNN_closest_neighbor_time: Optional[float]  # Deprecated: now unified with predictor_training_time
    # Fine-grained cut statistics
    num_cuts_mip_selected: dict
    num_cuts_rel_selected: dict
    num_cuts_mip_ml: dict  # Unified ML cuts across all methods (REG, KNN, TREE, SIMPLE_AVG)
    num_cuts_rel_ml: dict  # Unified ML cuts across all methods (REG, KNN, TREE, SIMPLE_AVG)
    num_cuts_mip_unselected: dict
    num_cuts_rel_unselected: dict
    num_cuts_mip_fallback: dict
    num_cuts_rel_fallback: dict
    num_cuts_mip_trial: dict
    num_cuts_rel_trial: dict
    # Switching strategy cut statistics
    num_cuts_mip_exact_switch: dict
    num_cuts_rel_exact_switch: dict
    # Features
    covariate_vectors: dict
    # Prediction and fallback statistics
    iteration: int
    # Fields with default values must come last
    original_Prob_Obj_Value: Optional[float] = None
    num_fallback_used: int = 0
    num_predictions: int = 0
    # KNN-specific parameters
    knn_optimal_k: Optional[int] = None  # Optimal k value found during KNN training (when find_optimal_k=True)
    # Switching strategy statistics
    num_switch_events: int = 0
    switch_iterations: Optional[list] = None
    permanently_switched: bool = False
    # Nodes and exploration statistics
    num_bnb_nodes: int = 0
    total_nodes_explored: int = 0
    # Phase 0 statistics
    phase0_cuts_added: int = 0
    phase0_execution_time_sec: Optional[float] = None  # Phase 0 warm-up execution time
    phase0_solution_generation_time_sec: Optional[float] = None  # Phase 0 solution generation time
    # Phase 1 & 2 statistics
    phase1_execution_time_sec: Optional[float] = None  # Phase 1 traditional Benders execution time
    phase2_execution_time_sec: Optional[float] = None  # Phase 2 ML-augmented Benders execution time
    # Algorithm termination
    stopping_reason: Optional[str] = None  # Reason for algorithm termination
    # Stopping reason tracking
    stopping_reason: str = "OPTIMAL"
    # Gurobi bound information
    ObjBound_LB: Optional[float] = None  # Gurobi's lower bound (ObjBound) from branch-and-bound tree
    
    # ===== NEW DIAGNOSTIC FIELDS (Feb 2026) =====
    # Task 1: Point signature tracking
    total_unique_points: int = 0
    total_repeated_hits: int = 0
    repeated_hits_with_cuts: int = 0
    
    # Task 2: Root vs non-root callback tracking  
    callback_counts: Optional[dict] = None  # {'mipsol_root': int, 'mipsol_nonroot': int, 'mipnode_root': int, 'mipnode_nonroot': int}
    # Task 2 averages (non-root divided by num_bnb_nodes for fair comparison)
    avg_mipsol_per_nonroot_node: float = 0.0
    avg_mipnode_per_nonroot_node: float = 0.0
    
    # Task 3: Objective re-evaluation results
    objective_reevaluation_results: Optional[dict] = None  # Retired; historical reevaluation is disabled
    bound_trace: Optional[list] = None  # Observed Gurobi master LB/UB samples
    gurobi_cut_statistics: Optional[dict] = None  # Solver aggregate counts, not exact/ML acceptance
    solver_timing_buckets: Optional[dict] = None  # Three mutually exclusive top-level categories
    solver_component_breakdown: Optional[dict] = None  # Additive fine-grained timing with residual
    final_objective_validation_time_sec: float = 0.0  # Reporting only; excluded from solution_time_sec
    final_exact_evaluation_count: int = 0
    final_exact_subproblem_solves: int = 0
    final_exact_evaluation_status: str = "no_incumbent"
    candidate_reevaluation_time_sec: float = 0.0  # Retired compatibility field; always zero
    master_model_construction_time_sec: float = 0.0
    timing_component_counts: Optional[dict] = None
    cut_construction_breakdown: Optional[dict] = None  # Subsets of cut evaluation/generation time
    # Detailed costs, feasibility/repair rates, conditioning, and fallback counts
    # for PARAMETRIC_* predictors. Empty for all other methods.
    parametric_diagnostics: Optional[dict] = None
    
    # Gurobi parameters logging
    mip_gap_stop: Optional[float] = None  # MIPGap value when solver stopped (final optimality gap)
    mip_gap_target: Optional[float] = None  # Target MIPGap value set as Gurobi parameter
    master_optimize_wall_time_sec: Optional[float] = None
    time_limit_sec: Optional[float] = None  # Time limit in seconds set as Gurobi parameter


def _set_params(mod: Model):
    """
    Configure Gurobi model parameters for Benders decomposition callback integration.
    
    This utility function sets essential Gurobi parameters required for proper integration
    with the ML-augmented Benders decomposition callback system. The primary configuration
    enables lazy constraint generation, which allows the callback to add Benders cuts
    dynamically during the branch-and-bound process.
    
    Parameters:
    -----------
    mod : gurobipy.Model
        Gurobi optimization model instance that will be solved with callback integration.
        This model represents the master problem formulation for any supported problem type
        (UFL, UHL, SCFLP, SSLP).
    
    Configuration Details:
    ----------------------
    **LazyConstraints = 1**: 
        Enables lazy constraint callback functionality in Gurobi. This setting allows:
        - Dynamic cut generation during branch-and-bound
        - Callback invocation at integer feasible solutions
        - Separation of Benders cuts on-demand rather than a priori
        - Integration with ML prediction system for cut generation
    
    **TimeLimit = 10800**:
        Sets maximum optimization time to 10800 seconds (3 hours). This prevents
        individual problem instances from running indefinitely and ensures
        experimental progress. When time limit is reached, Gurobi returns the
        best solution found so far.
    
    Integration Points:
    -------------------
    **Called by**: solve_master_problem() before model optimization
    **Enables**: callbacks.py callback system integration
    **Required for**: Proper Benders cut generation and ML prediction workflow
    
    Algorithm Workflow Context:
    ---------------------------
    1. Master problem model construction (UFL/UHL/SCFLP/SSLP formulations)
    2. **_set_params(model)** ← Configuration for callback integration
    3. Callback attachment with ML-augmented cut generation
    4. Model optimization with dynamic cut addition
    
    Note:
    -----
    This function must be called before attaching any lazy constraint callbacks to ensure
    proper Gurobi configuration. Without this setting, the callback system will not
    function correctly and cuts will not be generated.
    
    Example Usage:
    --------------
    ```python
    model = Model("UFL_Master")
    # ... model construction ...
    _set_params(model)  # Enable lazy constraints and set time limit
    model.optimize(callback_function)  # Now callbacks work properly
    ```
    """
    # Enable lazy constraints
    mod.Params.LazyConstraints = 1
    
    # Stop when relative optimality gap is at most 1%
    requested_gap = float(os.environ.get("LABD_MIP_GAP_TARGET", "0.01"))
    if not (0.0 < requested_gap < 1.0):
        raise ValueError("LABD_MIP_GAP_TARGET must be in (0, 1); use 0.001 for 0.1%.")
    mod.Params.MIPGap = requested_gap

    # Set time limit to 10800 seconds (3 hours)
    mod.Params.TimeLimit = 10800


def _determine_stopping_reason(model_status, runtime=None, nodecount=None):
    """
    Determine the stopping reason for Gurobi optimization based on solver status and parameters.
    
    This function analyzes the Gurobi model status and optional runtime/node information
    to classify the type of stopping condition that terminated the optimization process.
    This enables systematic tracking of solver behavior across different problem instances
    and parameter configurations.
    
    Parameters:
    -----------
    model_status : int
        Gurobi model status code (e.g., GRB.OPTIMAL, GRB.TIME_LIMIT, etc.)
    runtime : float, optional
        Actual runtime in seconds (used for time limit detection)
    nodecount : int, optional
        Number of nodes explored (used for node limit detection)
        
    Returns:
    --------
    str
        Stopping reason classification:
        - "OPTIMAL": Problem solved to optimality
        - "TIME_LIMIT": Stopped due to time limit (TimeLimit parameter)
        - "NODE_LIMIT": Stopped due to node limit (NodeLimit parameter)
        - "ITERATION_LIMIT": Stopped due to iteration limits (IterationLimit, BarIterLimit)
        - "GAP_TOLERANCE": Stopped due to gap tolerance (MIPGap, MIPGapAbs)
        - "MEMORY_LIMIT": Stopped due to memory limit (MemLimit parameter)
        - "SOLUTION_LIMIT": Stopped due to solution limit (SolutionLimit parameter)
        - "CUTOFF": Stopped due to cutoff value (Cutoff parameter)
        - "USER_INTERRUPT": User interrupted the solver
        - "INFEASIBLE": Problem is infeasible
        - "UNBOUNDED": Problem is unbounded
        - "INFEASIBLE_OR_UNBOUNDED": Problem is infeasible or unbounded
        - "NUMERIC_ERROR": Numerical difficulties encountered
        - "LICENSE_ERROR": License-related issues
        - "SOLVER_ERROR": Other solver errors
        - "UNKNOWN": Unknown status or error
    """
    # Map Gurobi status codes to stopping reasons
    status_mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE", 
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT", 
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "USER_INTERRUPT",
        GRB.NUMERIC: "NUMERIC_ERROR",
        GRB.SUBOPTIMAL: "GAP_TOLERANCE",  # Usually means gap tolerance was reached
        GRB.INPROGRESS: "USER_INTERRUPT",  # Optimization was interrupted
        GRB.USER_OBJ_LIMIT: "CUTOFF",  # User objective limit (cutoff) reached
    }
    
    # Check for specific stopping conditions first
    if model_status in status_mapping:
        return status_mapping[model_status]
    
    # Handle unknown or uncommon status codes
    try:
        # Try to get status name for logging
        status_name = f"GUROBI_STATUS_{model_status}"
        logger.warning(f"Unknown Gurobi status code: {model_status}")
        return f"UNKNOWN_{status_name}"
    except:
        return "UNKNOWN"


def _is_exact_method(use_prediction):
    return normalize_use_prediction(use_prediction) in {
        "none", "int_exact", "int_and_LP_ML_for_none"
    }


def _final_objective_report(problem_type, data, first_stage_values, objective_value,
                            use_prediction, use_stochastic_capacity=True):
    """Report the returned decision only, with a separate documentation cost.

    Exact Benders reuses ObjVal with no postsolve scenario solves. Every inexact
    method evaluates one returned decision across all scenarios, irrespective of
    the stopping reason. Never substitute another historical decision.
    """
    report = dict(original_Prob_Obj_Value=None,
                  final_objective_validation_time_sec=0.0,
                  final_exact_evaluation_count=0, final_exact_subproblem_solves=0,
                  final_exact_evaluation_status="no_incumbent")
    if first_stage_values is None:
        return report
    if _is_exact_method(use_prediction):
        report.update(original_Prob_Obj_Value=objective_value,
                      final_exact_evaluation_status="reused_gurobi_objective")
        return report
    stats = {"scenario_solves": 0}
    started = time.perf_counter()
    value = _calculate_true_objective_value(
        problem_type, data, first_stage_values, use_prediction,
        use_stochastic_capacity, evaluation_stats=stats,
    )
    elapsed = time.perf_counter() - started
    valid = value is not None and np.isfinite(value)
    report.update(original_Prob_Obj_Value=float(value) if valid else None,
                  final_objective_validation_time_sec=elapsed,
                  final_exact_evaluation_count=1,
                  final_exact_subproblem_solves=stats["scenario_solves"],
                  final_exact_evaluation_status="evaluated" if valid else "failed")
    return report


def _calculate_true_objective_value(problem_type, data, first_stage_values, use_prediction,
                                    use_stochastic_capacity=True, evaluation_stats=None):
    """
    Calculate the true objective value of the original problem by solving all subproblems exactly.
    
    Args:
        problem_type (str): Problem type ('UFL', 'UHL', 'SCFLP', 'SSLP')
        data: Problem data object
        first_stage_values: Solution values for first-stage variables
        use_prediction (str or bool): ML control mode - if "none", return None (use existing objective_value)
        use_stochastic_capacity (bool): Whether to use scenario-specific capacities (True) or original capacities (False)
        
    Returns:
        float or None: True objective value if ML is enabled, None otherwise
    """
    # Normalize use_prediction and check if ML is enabled
    if _is_exact_method(use_prediction):
        # When not using prediction, the objective_value is already the true value
        return None
    
    if first_stage_values is None:
        return None
        
    try:
        from Problem.subproblem_dual import solve_dual_subproblem as solve_exact

        def solve_dual_subproblem(*args, **kwargs):
            if evaluation_stats is not None:
                evaluation_stats["scenario_solves"] += 1
            result = solve_exact(*args, **kwargs)
            if not np.isfinite(result[0]):
                raise ValueError("Final scenario objective is not finite")
            return result
        
        pt = problem_type.upper()
        
        # Calculate first-stage cost
        first_stage_cost = 0.0
        
        if pt == "UFL":
            # First-stage cost: Σ_j f_j * y_j
            for j in data.F:
                if j in first_stage_values:
                    first_stage_cost += data.f[j] * first_stage_values[j]
            
            # Second-stage cost: solve all customer subproblems
            second_stage_cost = 0.0
            for i in data.I:
                try:
                    obj_val, _, _ = solve_dual_subproblem(pt, data, i, first_stage_values, use_stochastic_capacity)
                    second_stage_cost += obj_val
                except Exception as e:
                    logger.warning(f"Failed to solve subproblem {i}: {e}")
                    return None
                    
        elif pt == "UHL":
            # First-stage cost: assignment costs + hub opening costs
            # Assignment costs: Σ_i Σ_k beta * dist[i,k] * x[i,k]  
            # Hub opening costs: Σ_k f[k] * h[k]
            for i in data.N:
                for k in data.N:
                    assign_key = (i, k)
                    if assign_key in first_stage_values:
                        first_stage_cost += data.beta * data.dist[i, k] * first_stage_values[assign_key]
            
            # Hub opening costs
            for k in data.N:
                hub_key = k  # or however hub opening variables are keyed
                if hub_key in first_stage_values:
                    first_stage_cost += data.f[k] * first_stage_values[hub_key]
            
            # Second-stage cost: solve all OD pair subproblems
            second_stage_cost = 0.0
            for s_idx in range(len(data.S)):
                try:
                    obj_val, _, _ = solve_dual_subproblem(pt, data, s_idx, first_stage_values, use_stochastic_capacity)
                    second_stage_cost += obj_val
                except Exception as e:
                    logger.warning(f"Failed to solve subproblem {s_idx}: {e}")
                    return None
                    
        elif pt == "SCFLP":
            # First-stage cost: Σ_j fixed_costs_j * y_j
            for j in data.J:
                if j in first_stage_values:
                    first_stage_cost += data.fixed_costs[j] * first_stage_values[j]
            
            # Second-stage cost: solve all scenario subproblems
            second_stage_cost = 0.0
            scenario_prob = 1.0 / len(data.S)  # Uniform scenario probabilities
            for s_idx in range(len(data.S)):
                try:
                    obj_val, _, _ = solve_dual_subproblem(pt, data, s_idx, first_stage_values, use_stochastic_capacity)
                    second_stage_cost += scenario_prob * obj_val
                except Exception as e:
                    logger.warning(f"Failed to solve subproblem {s_idx}: {e}")
                    return None
                    
        elif pt == "SSLP":
            # First-stage cost: Σ_j fixed_cost_j * y_j
            for j in data.F:
                if j in first_stage_values:
                    first_stage_cost += data.fixed_cost[j] * first_stage_values[j]
            
            # Second-stage cost: solve all scenario subproblems
            second_stage_cost = 0.0
            for s_idx in range(len(data.S)):
                try:
                    obj_val, _, _ = solve_dual_subproblem(pt, data, s_idx, first_stage_values, use_stochastic_capacity)
                    second_stage_cost += data.p[s_idx] * obj_val
                except Exception as e:
                    logger.warning(f"Failed to solve subproblem {s_idx}: {e}")
                    return None
        elif pt.upper() in ["NESP", "NE-SP"]:
            # First-stage cost: Σ_{(i,j)∈A} F_ij * x_ij (arc installation costs)
            for idx in range(len(data.A)):
                if idx in first_stage_values:
                    first_stage_cost += data.F[idx] * first_stage_values[idx]
            
            # Second-stage cost: solve all OD pair subproblems weighted by their weights
            second_stage_cost = 0.0
            for m_idx in range(data.n_od_pairs):
                try:
                    obj_val, _, _ = solve_dual_subproblem(pt, data, m_idx, first_stage_values, use_stochastic_capacity)
                    second_stage_cost += data.weights[m_idx] * obj_val
                except Exception as e:
                    logger.warning(f"Failed to solve OD pair subproblem {m_idx}: {e}")
                    return None
        elif pt == "SUMND":
            # First-stage cost: Σ_{(i,j)∈A} f_ij * y_ij (arc activation costs)
            for idx in range(len(data.A)):
                if idx in first_stage_values:
                    first_stage_cost += data.F[idx] * first_stage_values[idx]
            
            # Second-stage cost: solve all scenario subproblems weighted by probabilities
            second_stage_cost = 0.0
            for s_idx in range(len(data.S)):
                try:
                    obj_val, _, _ = solve_dual_subproblem(pt, data, s_idx, first_stage_values, use_stochastic_capacity)
                    second_stage_cost += data.probabilities[s_idx] * obj_val
                except Exception as e:
                    logger.warning(f"Failed to solve scenario subproblem {s_idx}: {e}")
                    return None
        elif pt == "SCMND":
            # First-stage cost: Σ_{(i,j)∈A} f_ij * y_ij (arc activation costs)
            for idx in range(len(data.A)):
                if idx in first_stage_values:
                    first_stage_cost += data.F[idx] * first_stage_values[idx]

            # Second-stage cost: solve all scenario subproblems weighted by probabilities
            second_stage_cost = 0.0
            for s_idx in range(len(data.S)):
                try:
                    obj_val, _, _ = solve_dual_subproblem(pt, data, s_idx, first_stage_values, use_stochastic_capacity)
                    second_stage_cost += data.probabilities[s_idx] * obj_val
                except Exception as e:
                    logger.warning(f"Failed to solve scenario subproblem {s_idx}: {e}")
                    return None
        else:
            logger.warning(f"True objective calculation not implemented for problem type: {pt}")
            return None
            
        return first_stage_cost + second_stage_cost
        
    except Exception as e:
        logger.warning(f"Error calculating true objective value: {e}")
        return None


def solve_master_problem(problem_type, data, selected_subproblems, covariate_vectors,
                         method_config=None):
    """
    Solve the master problem using three-phase ML-augmented Benders decomposition with switching strategies.
    
    This is the central orchestrator function that coordinates the entire ML-augmented Benders
    decomposition algorithm with optional Phase 0 warm-up. It constructs the appropriate master 
    problem formulation for the specified problem type, integrates the ML-based callback system 
    for dynamic cut generation, and manages the switching between ML prediction and exact solving modes.
    
    Three-Phase Algorithm Flow:
    - Phase 0 (Optional): Generate m_trials random feasible solutions, solve all subproblems exactly, 
      collect cuts to strengthen master problem before classic Benders begins
    - Phase 1: Traditional Benders for n_trials iterations, collecting high-quality dual training data
    - Phase 2: ML-augmented Benders with subproblem selection and dual prediction
    
    The function supports four major problem types (UFL, UHL, SCFLP, SSLP) with comprehensive
    ML integration including feature-based prediction, k-nearest neighbor fallback, and
    adaptive switching strategies for robust performance across diverse problem instances.
    
    Parameters:
    -----------
    problem_type : str
        Problem formulation type. Supported values:
        - 'UFL': Uncapacitated Facility Location
        - 'UHL': Uncapacitated Hub Location  
        - 'SCFLP': Stochastic Capacitated Facility Location
        - 'SSLP': Stochastic Service Location Problem
    
    data : SCFLPData
        Problem-specific data object containing:
        - **SCFLP**: customers, facilities, scenarios, demands, capacities, costs
        
    selected_subproblems : list
        Indices of subproblems selected for exact solving in each iteration.
        Format: Scenario indices [0, 1, 2, ...]
        Generated by: subproblem_selection strategies (p-median, effective_sub)
    
    covariate_vectors : dict
        Pre-computed scenario covariate embeddings for ML prediction.
        Format: {scenario_id: numpy.array([covariate_1, ..., covariate_dim])}
        Generated by: feature_engineering.feature_generation.compute_scenario_features_from_duals()
        Used by: ML models for dual variable prediction
    
    n_neighbors : int
        Number of nearest neighbors for k-NN prediction fallback.
        Typical values: 3-10, affects prediction quality vs computational cost
        Used when: ML regression fails or prediction_method='KNN'
    
    prediction_method : str, default='KNN'
        Primary ML prediction approach:
        - 'KNN': k-nearest neighbor based on feature similarity
        - 'REG': Constrained linear regression with feasibility guarantees
        - 'TREE': Decision tree with optimized scenario groupings
        - 'SIMPLE_AVG': Simple averaging of all selected scenarios
        Integration: Determines callbacks.py prediction pathway
    
    use_prediction : str or bool, default="both"
        Fine-grained ML control ("none", "integer", "continuous", "both")
        Backward compatibility: True -> "both", False -> "none"
        Enable/disable ML prediction system entirely.
        - True: Use ML for unselected subproblems
        - False: Solve all subproblems exactly (pure Benders)
        Debug option: Useful for comparing ML vs exact performance
    
    n_trials : int, default=5
        Number of trial solves for dual data collection in feature generation.
        Higher values: More robust features, increased computational cost
        Used by: Feature generation pipeline for training data
    
    m_trials : int, default=0
        Number of Phase 0 random feasible solutions to generate for warm-up.
        - 0: Skip Phase 0, use current two-phase LABD (Phase 1 → Phase 2)
        - >0: Generate m_trials random solutions, solve all subproblems exactly, collect cuts
        Phase 0 strengthens master problem with additional cuts before classic Benders begins
        Used by: phase0_generator.py for random solution generation
    
    feature_params : dict, optional
        Feature generation hyperparameters. If None, uses config defaults:
        - n_walks, walk_length, seed: Random walk generation
        - feature_dim, window, min_count, sg: Word2Vec embedding learning
        Integration: Passed to feature_engineering.feature_generation functions
    
    n_selected_subproblems : int, optional
        Override for number of subproblems to select exactly.
        If None: Uses subproblem_selection strategy defaults
        Research parameter: For sensitivity analysis of selection size
    
    switching_strategy : str, default="no_switch"
        ML-to-exact switching strategy for robust performance:
        - "no_switch": Always use ML when available
        - "alternate": Switch back and forth between ML and exact modes
        - "permanent_switch": Switch to exact mode permanently after trigger
        Integration: Controls callbacks.py switching logic
    
    **KNN Method Parameters**:
    n_neighbors : int, default=3
        Number of nearest neighbors for k-NN prediction.
        Controls: Trade-off between prediction accuracy and computational cost
        Used by: DualPredictor with method='KNN'
    
    knn_find_optimal_k : bool, default=False
        Whether to enumerate k values to find optimal (T,k) pair for KNN method.
        True: Search k ∈ {1,2,...,p} to minimize paper's k-NN objective
        False: Use fixed k = n_neighbors value
        Integration: Used by training_and_selection.solve_knn_integrated_selection()
    
    **TREE Method Parameters** (Adaptive Constraint-Free Approach):
    
    tree_min_leaf_size : REMOVED in Tree-First approach
        No longer needed - Tree-First approach uses proportional selection
        which guarantees exact scenario counts regardless of tree structure.
    
    tree_min_impurity_decrease : float, default=0.0
        Minimum impurity decrease threshold for tree splits.
        Controls: When to stop splitting nodes (early stopping)
        Higher values: Simpler trees, fewer splits, larger leaves
        Used by: TREE method greedy top-down construction
    
    tree_quantile_grid_size : int, default=10
        Number of quantile-based split threshold candidates per feature.
        Controls: Split quality vs computational cost trade-off
        Higher values: Better splits, increased training time
        Used by: TREE method split candidate generation
    
    **DUAL_CLUSTERING Method Parameters**:
    dual_clustering_random_state : int, default=42
        Random seed for K-means clustering initialization.
        Controls: Reproducibility of cluster assignments
        Used by: DualPredictor with method='DUAL_CLUSTERING'
    
    dual_clustering_init_method : str, default="k-means++"
        K-means initialization method for cluster center selection.
        Options: "k-means++", "random"
        "k-means++": Smart initialization for better convergence
        Used by: training_and_selection.solve_dual_clustering_selection()
    
    dual_clustering_max_iter : int, default=300
        Maximum number of iterations for K-means algorithm.
        Controls: Convergence timeout vs computational cost
        Higher values: Better convergence, increased training time
        Used by: DUAL_CLUSTERING method cluster optimization
    
    dual_clustering_tol : float, default=1e-4
        Convergence tolerance for K-means algorithm.
        Controls: When to stop K-means iterations
        Lower values: Tighter convergence, longer training time
        Used by: DUAL_CLUSTERING method convergence criteria
    
    **ML Training Parameters**:
    L_d : float, default=1e2
        L1 dual parameter error budget for constrained regression training.
        Controls: Trade-off between parameter accuracy and feasibility
        Used by: Dual_Prediction.training.train_feasible_predictor()
    
    L_v : float, default=1e2  
        Value discrepancy budget for dual objective prediction accuracy.
        Controls: Trade-off between objective accuracy and feasibility
        Used by: Dual_Prediction.training.train_feasible_predictor()
    
    p_d : float, default=1.0
        Objective weight for dual parameter slack in training optimization.
        Higher values: Prioritize dual accuracy over objective accuracy
        Used by: Dual_Prediction.training.train_feasible_predictor()
    
    p_v : float, default=1.0
        Objective weight for value slack in training optimization.
        Higher values: Prioritize objective accuracy over dual accuracy  
        Used by: Dual_Prediction.training.train_feasible_predictor()
    
    Returns:
    --------
    Solution
        Comprehensive solution object containing:
        - **Optimization Results**: 
          * objective_value: Final optimal objective value of the master problem
          * original_Prob_Obj_Value: True objective value calculated by solving all subproblems exactly
            (only computed when ML is enabled for ML vs exact comparison)
          * first_stage_optimnal_values: Optimal first-stage decision variable values
        - **Performance Metrics**: solution_time, feature_gen_time_sec, etc.
        - **Cut Statistics**: Detailed breakdown by iteration and cut type
        - **Algorithm State**: Switching events, fallback usage, prediction counts
        - **Diagnostic Information**: Branch-and-bound nodes, exploration statistics
    
    Algorithm Workflow:
    -------------------
    1. **Problem Type Detection**: Route to appropriate master formulation (UFL/UHL/SCFLP/SSLP)
    2. **Master Problem Construction**: 
       - Create Gurobi model with problem-specific variables and constraints
       - Add auxiliary variable θ for Benders cuts
       - Configure model parameters for callback integration
    3. **Callback Integration**:
       - Attach ML-augmented Benders callback with all parameters
       - Enable dynamic cut generation with switching strategy
       - Configure prediction method and training parameters
    4. **Optimization Execution**: 
       - Solve master problem with callback-driven cut generation
       - Collect comprehensive statistics and timing information
       - Handle optimization status and error conditions
    5. **Solution Assembly**: 
       - Extract optimal first-stage decisions
       - Calculate true objective value by solving all subproblems exactly (when ML is enabled)
       - Compile performance metrics and diagnostic data
       - Return comprehensive Solution object
    
    Problem-Specific Formulations:
    ------------------------------
    
    **UFL Master Problem**:
    ```
    min  Σ_j f_j * x_j + θ
    s.t. x_j ∈ {0,1}     ∀j ∈ F
         θ ≥ 0
         [Benders cuts added dynamically]
    ```
    Where: x_j = facility opening decisions, θ = second-stage cost approximation
    
    **UHL Master Problem**: 
    ```
    min  Σ_{k,l} c_{kl} * x_{kl} + θ
    s.t. Σ_l x_{kl} ≤ 1           ∀k ∈ N  
         Σ_k x_{kl} ≤ 1           ∀l ∈ N
         x_{kl} ∈ {0,1}           ∀k,l ∈ N
         θ ≥ 0
         [Benders cuts added dynamically]
    ```
    Where: x_{kl} = hub connection decisions, θ = routing cost approximation
    
    **SCFLP Master Problem**:
    ```
    min  Σ_j f_j * y_j + θ  
    s.t. y_j ∈ {0,1}     ∀j ∈ J
         θ ≥ 0
         [Benders cuts added dynamically]
    ```
    Where: y_j = facility opening decisions, θ = expected second-stage cost
    
    **SSLP Master Problem**:
    ```
    min  Σ_j f_j * y_j + θ
    s.t. y_j ∈ {0,1}     ∀j ∈ J  
         θ ≥ 0
         [Benders cuts added dynamically]
    ```
    Where: y_j = service facility decisions, θ = expected service cost
    
    Integration Points:
    -------------------
    **Input Dependencies**:
    - Problem.data: Problem instance data structures
    - feature_engineering.feature_generation: Pre-computed feature vectors
    - subproblem_selection: Selected subproblem indices
    - config.py: Default parameter values and experimental settings
    
    **Core Integrations**:
    - Problem.callbacks: ML-augmented Benders callback system
    - Dual_Prediction.training: Constrained ML model training
    - Problem.subproblem_dual: Exact dual subproblem solving
    
    **Output Destinations**: 
    - main.py: Solution collection for experimental evaluation
    - Excel export: Performance metrics and cut statistics
    - Logging system: Algorithm diagnostics and debugging information
    
    Performance Characteristics:
    ----------------------------
    **Time Complexity**: 
    - Master problem: O(iterations × (selected_subproblems + ML_predictions))
    - Feature generation: O(scenarios² × trials) for similarity computation
    - ML training: O(training_scenarios × feature_dim) per iteration
    
    **Memory Usage**:
    - Master problem: O(first_stage_variables + accumulated_cuts)
    - Feature vectors: O(scenarios × feature_dim)
    - Callback state: O(iterations × subproblems) for statistics
    
    **Convergence Properties**:
    - Finite convergence guaranteed (standard Benders theory)
    - ML acceleration typically 2-5x speedup over pure Benders
    - Switching strategies provide robustness against ML prediction failures
    
    Error Handling:
    ---------------
    **Optimization Failures**:
    - Non-optimal status detection and reporting
    - Graceful degradation with diagnostic information
    - Status code preservation for debugging
    
    **ML Integration Failures**:
    - Automatic fallback to exact solving
    - Prediction quality monitoring and switching triggers
    - Comprehensive error logging and recovery
    
    **Data Validation**:
    - Problem type compatibility checking
    - Parameter range validation
    - Input format verification
    
    Example Usage:
    --------------
    ```python
    # Standard ML-augmented solving
    solution = solve_master_problem(
        'UFL', ufl_data, [0, 1, 2], feature_vectors, 
        n_neighbors=5, prediction_method='REG'
    )
    
    # Pure exact solving for comparison
    solution_exact = solve_master_problem(
        'UFL', ufl_data, [0, 1, 2], feature_vectors,
        n_neighbors=5, use_prediction=False
    )
    
    # Robust solving with permanent switching
    solution_robust = solve_master_problem(
        'SCFLP', scflp_data, selected_scenarios, features,
        n_neighbors=3, switching_strategy="permanent_switch",
        L_d=50, L_v=100
    )
    ```
    
    Research Applications:
    ----------------------
    This function serves as the core experimental unit for research in:
    - ML-augmented optimization algorithm development
    - Switching strategy effectiveness analysis  
    - Feature engineering impact on prediction quality
    - Algorithm performance comparison across problem types
    - Scalability analysis for large-scale stochastic optimization
    """
    _master_construction_started = time.perf_counter()
    pt = problem_type.upper()

    # Extract use_prediction and use_stochastic_capacity from method_config for helper function calls
    if method_config is not None:
        method_category = method_config.get('method_category', 'ML')
        use_stochastic_capacity = method_config.get('use_stochastic_capacity', True)
        
        # Derive use_prediction string for compatibility with helper functions
        if method_category == 'Exact':
            callback_type = method_config.get('exact_callback_type', 'int_only')
            if callback_type == 'int_only':
                use_prediction = 'int_exact'
            else:  # int_and_LP
                use_prediction = 'int_and_LP_ML_for_none'
        else:  # ML
            callback_type = method_config.get('ml_callback_type', 'ML_at_both')
            if callback_type == 'ML_at_both':
                use_prediction = 'int_and_LP_ML_for_both'
            elif callback_type == 'ML_at_integer_exact_at_LP':
                use_prediction = 'int_and_LP_ML_for_int'
            elif callback_type == 'ML_at_LP_exact_at_integer':
                use_prediction = 'int_and_LP_ML_for_LP'
            elif callback_type == 'ML_at_integer_only':
                use_prediction = 'int_ML'
            else:
                raise ValueError(f"Unknown ml_callback_type: {callback_type}")
    else:
        # Defaults if method_config is somehow None
        use_prediction = 'int_and_LP_ML_for_both'
        use_stochastic_capacity = True

    # ---------------------------
    # UFL Master Problem
    # ---------------------------
    if pt == "UFL":
        with Model("UFL_Master") as mod:
            _set_params(mod)

            # First-stage: open facility y_j
            first_stage_values = mod.addVars(data.F, vtype=GRB.BINARY, name="y")
            # Second-stage recourse estimates theta_i
            theta_vars = mod.addVars(data.S, lb=0.0, name="theta")

            # At least one facility open
            mod.addConstr(
                quicksum(first_stage_values[j] for j in data.F) >= 1,
                name="AtLeastOneFacility"
            )

            # Objective: fixed costs + recourse estimates
            total_cost = quicksum(data.f[j] * first_stage_values[j] for j in data.F) + \
                         quicksum(theta_vars[s] for s in data.S)
            mod.setObjective(total_cost, GRB.MINIMIZE)

            # Callback
            callback = Callback(
                problem_type=problem_type,
                data=data,
                first_stage_values=first_stage_values,
                theta_vars=theta_vars,
                selected_subproblems=selected_subproblems,
                covariate_vectors=covariate_vectors,
                method_config=method_config
            )
            if WRITE_MP_LP:
                mod.write(f"{mod.ModelName}.lp")

            master_model_construction_time_sec = (
                time.perf_counter() - _master_construction_started
            )
            _optimize_started = time.perf_counter()
            mod.optimize(callback)
            master_optimize_wall_time_sec = time.perf_counter() - _optimize_started
            callback.raise_if_failed()
            callback.append_terminal_bounds(mod)

            # Determine stopping reason and extract solution information
            stopping_reason = _determine_stopping_reason(mod.Status, mod.Runtime, mod.NodeCount)
            
            # Initialize default values for incomplete solutions
            obj = None
            y_vals = None
            sol_time = float(mod.Runtime) if hasattr(mod, 'Runtime') else None
            
            # Extract solution information if available
            if mod.Status == GRB.OPTIMAL:
                obj = mod.ObjVal
                y_vals = mod.getAttr("x", first_stage_values)
            elif mod.Status in [GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, 
                               GRB.SOLUTION_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT]:
                # Solver stopped early but may have found a solution
                try:
                    if mod.SolCount > 0:  # At least one solution found
                        obj = mod.ObjVal
                        y_vals = mod.getAttr("x", first_stage_values)
                    else:
                        obj = float('inf')  # No feasible solution found
                except Exception as e:
                    logger.warning(f"Error extracting solution after early termination: {e}")
                    obj = float('inf')
            elif mod.Status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                # Problem is infeasible or unbounded
                obj = float('inf') if mod.Status == GRB.INFEASIBLE else float('-inf')
            else:
                # Other error conditions
                logger.warning(f"UFL optimization ended with status {mod.Status}: {stopping_reason}")
                obj = float('inf')

            # Calculate true objective value if using prediction
            final_report = _final_objective_report(
                problem_type, data, y_vals, obj, use_prediction, use_stochastic_capacity
            )

            _log_final_master_solution(problem_type, obj, y_vals)

            return Solution(
                objective_value=obj if obj is not None else float('inf'),
                **final_report,
                first_stage_optimnal_values=y_vals,
                solution_time=sol_time,
                feature_gen_time_sec=callback.feature_gen_time_sec,
                subproblem_selection_time_sec=callback.subproblem_selection_time_sec,
                predictor_training_time=callback.predictor_training_time,
                KNN_closest_neighbor_time=callback.KNN_closest_neighbor_time,
                phase0_execution_time_sec=callback.phase0_execution_time_sec,
                phase0_solution_generation_time_sec=callback.phase0_solution_generation_time_sec,
                phase1_execution_time_sec=callback.phase1_execution_time_sec,
                phase2_execution_time_sec=callback.phase2_execution_time_sec,
                num_cuts_mip_selected=callback.num_cuts_mip_selected,
                num_cuts_rel_selected=callback.num_cuts_rel_selected,
                num_cuts_mip_ml=callback.num_cuts_mip_ml,
                num_cuts_rel_ml=callback.num_cuts_rel_ml,
                num_cuts_mip_unselected=callback.num_cuts_mip_unselected,
                num_cuts_rel_unselected=callback.num_cuts_rel_unselected,
                num_cuts_mip_trial=callback.num_cuts_mip_trial,
                num_cuts_rel_trial=callback.num_cuts_rel_trial,
                num_cuts_mip_fallback=callback.num_cuts_mip_fallback,
                num_cuts_rel_fallback=callback.num_cuts_rel_fallback,
                num_cuts_mip_exact_switch=callback.num_cuts_mip_exact_switch,
                num_cuts_rel_exact_switch=callback.num_cuts_rel_exact_switch,
                iteration=callback.iteration,
                num_fallback_used=callback.num_fallback_used,
                num_predictions=callback.num_predictions,
                num_switch_events=callback.num_switch_events,
                switch_iterations=callback.switch_iterations.copy() if callback.switch_iterations else [],
                permanently_switched=callback.permanently_switched,
                num_bnb_nodes=int(mod.NodeCount) if hasattr(mod, 'NodeCount') else 0,
                total_nodes_explored=callback.total_nodes_explored,
                covariate_vectors=callback.feature_vectors,
                knn_optimal_k=getattr(callback.dual_predictor, 'optimal_k', None) if hasattr(callback, 'dual_predictor') and callback.dual_predictor else None,
                stopping_reason=stopping_reason
            )

    # ---------------------------
    # UHL Master Problem
    # ---------------------------
    elif pt == "UHL":
        with Model("UHL_Master") as mod:
            _set_params(mod)

            # First-stage: hub opening and assignment
            open_vars   = mod.addVars(data.N, vtype=GRB.BINARY, name="h")
            assign_vars = mod.addVars(data.N, data.N, vtype=GRB.BINARY, name="x")

            # Recourse estimates theta_{i,j} for each OD-pair
            # Convert data.S (array of [i,j]) into list of tuple keys
            od_pairs    = [tuple(pair) for pair in data.S]
            theta_vars  = mod.addVars(od_pairs, lb=0.0, name="theta")

            # Constraints:
            # Each node assigned to exactly one hub
            mod.addConstrs(
                (quicksum(assign_vars[i, k] for k in data.N) == 1 for i in data.N),
                name="AssignOneHub"
            )
            # Assignment only if hub is open
            mod.addConstrs(
                (assign_vars[i, k] <= open_vars[k] for i in data.N for k in data.N),
                name="AssignIfOpen"
            )
            # Exactly p hubs open (data.p must be provided)
            mod.addConstr(
                quicksum(open_vars[k] for k in data.N) == data.p,
                name="OpenPHubs"
            )

            # Objective:
            #   Sum assignment costs + hub fixed costs + recourse thetas
            assign_cost = data.beta * data.dist
            total_cost = quicksum(
                assign_cost[i, k] * assign_vars[i, k]
                for i in data.N for k in data.N
            ) + quicksum(
                data.f[k] * open_vars[k] for k in data.N
            ) + quicksum(
                theta_vars[i, j] for (i, j) in od_pairs
            )
            mod.setObjective(total_cost, GRB.MINIMIZE)

            # Callback to generate Benders cuts
            callback = Callback(
                problem_type=problem_type,
                data=data,
                first_stage_values=assign_vars,
                theta_vars=theta_vars,
                selected_subproblems=selected_subproblems,
                covariate_vectors=covariate_vectors,
                method_config=method_config
            )
            if WRITE_MP_LP:
                mod.write(f"{mod.ModelName}.lp")

            master_model_construction_time_sec = (
                time.perf_counter() - _master_construction_started
            )
            _optimize_started = time.perf_counter()
            mod.optimize(callback)
            master_optimize_wall_time_sec = time.perf_counter() - _optimize_started
            callback.raise_if_failed()
            callback.append_terminal_bounds(mod)

            # Finalize phase timing after optimization completes
            callback.finalize_phase_timing()

            # Determine stopping reason and extract solution information
            stopping_reason = _determine_stopping_reason(mod.Status, mod.Runtime, mod.NodeCount)
            
            # Initialize default values for incomplete solutions
            obj = None
            x_vals = None
            sol_time = float(mod.Runtime) if hasattr(mod, 'Runtime') else None
            
            # Extract solution information if available
            if mod.Status == GRB.OPTIMAL:
                obj = mod.ObjVal
                x_vals = mod.getAttr("x", assign_vars)
            elif mod.Status in [GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, 
                               GRB.SOLUTION_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT]:
                # Solver stopped early but may have found a solution
                try:
                    if mod.SolCount > 0:  # At least one solution found
                        obj = mod.ObjVal
                        x_vals = mod.getAttr("x", assign_vars)
                    else:
                        obj = float('inf')  # No feasible solution found
                except Exception as e:
                    logger.warning(f"Error extracting solution after early termination: {e}")
                    obj = float('inf')
            elif mod.Status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                # Problem is infeasible or unbounded
                obj = float('inf') if mod.Status == GRB.INFEASIBLE else float('-inf')
            else:
                # Other error conditions
                logger.warning(f"UHL optimization ended with status {mod.Status}: {stopping_reason}")
                obj = float('inf')

            # Calculate true objective value if using prediction
            final_report = _final_objective_report(
                problem_type, data, x_vals, obj, use_prediction, use_stochastic_capacity
            )

            _log_final_master_solution(problem_type, obj, x_vals)

            return Solution(
                objective_value=obj if obj is not None else float('inf'),
                **final_report,
                first_stage_optimnal_values=x_vals,
                solution_time=sol_time,
                feature_gen_time_sec=callback.feature_gen_time_sec,
                subproblem_selection_time_sec=callback.subproblem_selection_time_sec,
                predictor_training_time=callback.predictor_training_time,
                KNN_closest_neighbor_time=callback.KNN_closest_neighbor_time,
                phase0_execution_time_sec=callback.phase0_execution_time_sec,
                phase0_solution_generation_time_sec=callback.phase0_solution_generation_time_sec,
                phase1_execution_time_sec=callback.phase1_execution_time_sec,
                phase2_execution_time_sec=callback.phase2_execution_time_sec,
                num_cuts_mip_selected=callback.num_cuts_mip_selected,
                num_cuts_rel_selected=callback.num_cuts_rel_selected,
                num_cuts_mip_ml=callback.num_cuts_mip_ml,
                num_cuts_rel_ml=callback.num_cuts_rel_ml,
                num_cuts_mip_unselected=callback.num_cuts_mip_unselected,
                num_cuts_rel_unselected=callback.num_cuts_rel_unselected,
                num_cuts_mip_trial=callback.num_cuts_mip_trial,
                num_cuts_rel_trial=callback.num_cuts_rel_trial,
                num_cuts_mip_fallback=callback.num_cuts_mip_fallback,
                num_cuts_rel_fallback=callback.num_cuts_rel_fallback,
                num_cuts_mip_exact_switch=callback.num_cuts_mip_exact_switch,
                num_cuts_rel_exact_switch=callback.num_cuts_rel_exact_switch,
                iteration=callback.iteration,
                num_fallback_used=callback.num_fallback_used,
                num_predictions=callback.num_predictions,
                num_switch_events=callback.num_switch_events,
                switch_iterations=callback.switch_iterations.copy() if callback.switch_iterations else [],
                permanently_switched=callback.permanently_switched,
                num_bnb_nodes=int(mod.NodeCount) if hasattr(mod, 'NodeCount') else 0,
                total_nodes_explored=callback.total_nodes_explored,
                covariate_vectors=callback.feature_vectors,
                knn_optimal_k=getattr(callback.dual_predictor, 'optimal_k', None) if hasattr(callback, 'dual_predictor') and callback.dual_predictor else None,
                stopping_reason=stopping_reason
            )

    elif pt == "SCFLP":
        with Model("SCFLP_Master") as mod:
            _set_params(mod)

            # First-stage: open facility y_j (indexed by J)
            first_stage_values = mod.addVars(data.J, vtype=GRB.BINARY, name="y")
            # Second-stage recourse estimates theta_s (indexed by S)
            theta_vars = mod.addVars(data.S, lb=0.0, name="theta")

            # At least one facility open constraint
            mod.addConstr(
                quicksum(first_stage_values[j] for j in data.J) >= 1,
                name="AtLeastOneFacility"
            )
            
            # Capacity feasibility constraints
            if use_stochastic_capacity and hasattr(data, 'capacity_scenarios') and data.capacity_scenarios is not None:
                # When using stochastic capacities, add one constraint per scenario
                # ensuring that activated capacity for each scenario is sufficient for its demand
                for s in data.S:
                    scenario_demand_sum = sum(data.scenarios[s])  # Total demand in scenario s
                    scenario_capacities = data.capacity_scenarios[s]  # Capacities for scenario s
                    mod.addConstr(
                        quicksum(scenario_capacities[j] * first_stage_values[j] for j in data.J) >= scenario_demand_sum,
                        name=f"CapacityFeasibility_s{s}"
                    )
            else:
                # Original constraint: ensure enough capacity for worst-case demand
                mod.addConstr(
                    quicksum(data.capacities[j] * first_stage_values[j] for j in data.J) >= data.max_demand_sum_over_scenario,
                    name="CapacityFeasibility"
                )

            # Objective: fixed costs + expected recourse costs
            obj_expr = quicksum(data.fixed_costs[j] * first_stage_values[j] for j in data.J) + \
                       (1.0 / len(data.S)) * quicksum(theta_vars[s] for s in data.S)
            mod.setObjective(obj_expr, GRB.MINIMIZE)

            # Callback
            callback = Callback(
                problem_type=problem_type,
                data=data,
                first_stage_values=first_stage_values,
                theta_vars=theta_vars,
                selected_subproblems=selected_subproblems,
                covariate_vectors=covariate_vectors,
                method_config=method_config
            )
            if WRITE_MP_LP:
                mod.write(f"{mod.ModelName}.lp")

            master_model_construction_time_sec = (
                time.perf_counter() - _master_construction_started
            )
            _optimize_started = time.perf_counter()
            mod.optimize(callback)
            master_optimize_wall_time_sec = time.perf_counter() - _optimize_started
            callback.raise_if_failed()
            callback.append_terminal_bounds(mod)

            # Finalize phase timing after optimization completes
            callback.finalize_phase_timing()

            # Extract final solution and callback state information
            stopping_reason = _determine_stopping_reason(mod.Status, mod.Runtime, mod.NodeCount)
            x_vals = None
            sol_time = float(mod.Runtime) if hasattr(mod, 'Runtime') else None
            
            # Extract solution information if available
            if mod.Status == GRB.OPTIMAL:
                obj = mod.ObjVal
                obj_bound = mod.ObjBound  # Capture lower bound from branch-and-bound
                x_vals = mod.getAttr("x", first_stage_values)
            elif mod.Status in [GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, 
                               GRB.SOLUTION_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT]:
                # Solver stopped early but may have found a solution
                try:
                    if mod.SolCount > 0:  # At least one solution found
                        obj = mod.ObjVal
                        obj_bound = mod.ObjBound  # Capture lower bound
                        x_vals = mod.getAttr("x", first_stage_values)
                    else:
                        obj = float('inf')  # No feasible solution found
                        obj_bound = None
                except Exception as e:
                    logger.warning(f"Error extracting solution after early termination: {e}")
                    obj = float('inf')
                    obj_bound = None
            elif mod.Status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                # Problem is infeasible or unbounded
                obj = float('inf') if mod.Status == GRB.INFEASIBLE else float('-inf')
                obj_bound = None
            else:
                # Other error conditions
                logger.warning(f"SCFLP optimization ended with status {mod.Status}: {stopping_reason}")
                obj = float('inf')
                obj_bound = None

            # Calculate true objective value if using prediction
            final_report = _final_objective_report(
                problem_type, data, x_vals, obj, use_prediction, use_stochastic_capacity
            )

            _log_final_master_solution(problem_type, obj, x_vals)
            
            
            # Get comprehensive callback diagnostics
            callback_diagnostics = callback.get_callback_diagnostics()
            
            # DEBUG: Log callback diagnostic state before creating Solution object
            logger.info(f"DEBUG - Callback state before Solution creation:")
            logger.info(f"  total_unique_points: {callback.total_unique_points}")
            logger.info(f"  total_repeated_hits: {callback.total_repeated_hits}")
            logger.info(f"  repeated_hits_with_cuts: {callback.repeated_hits_with_cuts}")
            logger.info(f"  callback_counts: {callback.callback_counts}")

            # Extract Gurobi parameters for logging
            mip_gap_stop = mod.MIPGap if hasattr(mod, 'MIPGap') else None
            time_limit_value = mod.Params.TimeLimit if hasattr(mod, 'Params') and hasattr(mod.Params, 'TimeLimit') else None

            return Solution(
                objective_value=obj if obj is not None else float('inf'),
                **final_report,
                first_stage_optimnal_values=x_vals,
                solution_time=sol_time,
                feature_gen_time_sec=callback.feature_gen_time_sec,
                subproblem_selection_time_sec=callback.subproblem_selection_time_sec,
                predictor_training_time=callback.predictor_training_time,
                KNN_closest_neighbor_time=callback.KNN_closest_neighbor_time,
                phase0_execution_time_sec=callback.phase0_execution_time_sec,
                phase0_solution_generation_time_sec=callback.phase0_solution_generation_time_sec,
                phase1_execution_time_sec=callback.phase1_execution_time_sec,
                phase2_execution_time_sec=callback.phase2_execution_time_sec,
                num_cuts_mip_selected=callback.num_cuts_mip_selected,
                num_cuts_rel_selected=callback.num_cuts_rel_selected,
                num_cuts_mip_ml=callback.num_cuts_mip_ml,
                num_cuts_rel_ml=callback.num_cuts_rel_ml,
                num_cuts_mip_unselected=callback.num_cuts_mip_unselected,
                num_cuts_rel_unselected=callback.num_cuts_rel_unselected,
                num_cuts_mip_trial=callback.num_cuts_mip_trial,
                num_cuts_rel_trial=callback.num_cuts_rel_trial,
                num_cuts_mip_fallback=callback.num_cuts_mip_fallback,
                num_cuts_rel_fallback=callback.num_cuts_rel_fallback,
                num_cuts_mip_exact_switch=callback.num_cuts_mip_exact_switch,
                num_cuts_rel_exact_switch=callback.num_cuts_rel_exact_switch,
                iteration=callback.iteration,
                num_fallback_used=callback.num_fallback_used,
                num_predictions=callback.num_predictions,
                num_switch_events=callback.num_switch_events,
                switch_iterations=callback.switch_iterations.copy() if callback.switch_iterations else [],
                permanently_switched=callback.permanently_switched,
                num_bnb_nodes=int(mod.NodeCount) if hasattr(mod, 'NodeCount') else 0,
                total_nodes_explored=callback.total_nodes_explored,
                phase0_cuts_added=callback.phase0_cuts_added,
                covariate_vectors=callback.feature_vectors,
                knn_optimal_k=getattr(callback.dual_predictor, 'optimal_k', None) if hasattr(callback, 'dual_predictor') and callback.dual_predictor else None,
                stopping_reason=stopping_reason,
                ObjBound_LB=obj_bound,
                # NEW DIAGNOSTIC FIELDS (Feb 2026)
                # Task 1: Point signature tracking
                total_unique_points=callback.total_unique_points,
                total_repeated_hits=callback.total_repeated_hits,
                repeated_hits_with_cuts=callback.repeated_hits_with_cuts,
                # Task 2: Root vs non-root callback tracking  
                callback_counts=callback.callback_counts.copy(),
                # Task 2 averages: non-root callbacks divided by number of non-root nodes
                avg_mipsol_per_nonroot_node=(callback.callback_counts['mipsol_nonroot'] / max(mod.NodeCount - 1, 1)) if hasattr(mod, 'NodeCount') and mod.NodeCount > 1 else 0.0,
                avg_mipnode_per_nonroot_node=(callback.callback_counts['mipnode_nonroot'] / max(mod.NodeCount - 1, 1)) if hasattr(mod, 'NodeCount') and mod.NodeCount > 1 else 0.0,
                bound_trace=callback.bound_trace.copy(),
                gurobi_cut_statistics=callback.gurobi_cut_statistics.export(),
                master_model_construction_time_sec=master_model_construction_time_sec,
                master_optimize_wall_time_sec=master_optimize_wall_time_sec,
                timing_component_counts=callback.timing_component_counts(),
                cut_construction_breakdown=callback.exclusive_profiler.cut_construction_breakdown(),
                solver_timing_buckets=callback.solver_timing_buckets(master_optimize_wall_time_sec),
                solver_component_breakdown=callback.solver_component_breakdown(master_optimize_wall_time_sec),
                parametric_diagnostics=(
                    callback.get_parametric_diagnostics()
                ),
                # Gurobi parameters logging
                mip_gap_stop=mip_gap_stop,
                mip_gap_target=mod.Params.MIPGap if hasattr(mod, 'Params') and hasattr(mod.Params, 'MIPGap') else None,
                time_limit_sec=time_limit_value
            )
            
            # Extract data attributes
            I = data.I  # customers
            J = data.J  # facilities
            S = data.S  # scenarios
            fixed_costs = data.fixed_costs
            capacities = data.capacities
            
            # First-stage: facility opening decisions
            y = mod.addVars(J, vtype=GRB.BINARY, name="y")
            # Second-stage: recourse estimates for each scenario
            theta = mod.addVars(S, vtype=GRB.CONTINUOUS, lb=0.0, name="theta")

            # Objective: fixed cost + expected recourse cost
            mod.setObjective(
                quicksum(fixed_costs[j] * y[j] for j in J) +
                (1 / len(S)) * quicksum(theta[s] for s in S),
                GRB.MINIMIZE
            )

            # Relative complete recourse condition: total opened capacity ≥ worst-case total demand
            worst_case_demand = data.max_demand_sum_over_scenario
            mod.addConstr(quicksum(capacities[j] * y[j] for j in J) >= worst_case_demand,
                          name="recourse_feasibility")

            # Callback to generate Benders cuts
            callback = Callback(
                problem_type=problem_type,
                data=data,
                first_stage_values=y,
                theta_vars=theta,
                selected_subproblems=selected_subproblems,
                covariate_vectors=feature_vectors,
                prediction_method=prediction_method,
                n_neighbors=n_neighbors,
                use_prediction=use_prediction,
                n_trials=n_trials,
                m_trials=m_trials,
                feature_params=feature_params,
                n_selected_subproblems=n_selected_subproblems,
                switching_strategy=switching_strategy,
                knn_find_optimal_k=knn_find_optimal_k,
                # TREE parameters
                # tree_max_depth: removed for adaptive TREE method
                # tree_min_leaf_size: removed for Tree-First approach
                tree_min_impurity_decrease=tree_min_impurity_decrease,
                tree_quantile_grid_size=tree_quantile_grid_size,
                # DUAL_CLUSTERING parameters
                dual_clustering_random_state=dual_clustering_random_state,
                dual_clustering_init_method=dual_clustering_init_method,
                dual_clustering_max_iter=dual_clustering_max_iter,
                dual_clustering_tol=dual_clustering_tol,
                # UNCERTAINTY_CLUSTERING parameters
                uncertainty_clustering_random_state=uncertainty_clustering_random_state,
                uncertainty_clustering_init_method=uncertainty_clustering_init_method,
                uncertainty_clustering_max_iter=uncertainty_clustering_max_iter,
                uncertainty_clustering_tol=uncertainty_clustering_tol,
                # BERTSIMAS_RANDOM_SAMPLING parameters
                bertsimas_random_sampling_random_state=bertsimas_random_sampling_random_state,
                # Feature enhancement parameters
                use_covariates=use_covariates
            )
            
            if WRITE_MP_LP:
                mod.write(f"{mod.ModelName}.lp")

            master_model_construction_time_sec = (
                time.perf_counter() - _master_construction_started
            )
            _optimize_started = time.perf_counter()
            mod.optimize(callback)
            master_optimize_wall_time_sec = time.perf_counter() - _optimize_started
            callback.raise_if_failed()
            callback.append_terminal_bounds(mod)

            # Finalize phase timing after optimization completes
            callback.finalize_phase_timing()

            # Determine stopping reason and extract solution information
            stopping_reason = _determine_stopping_reason(mod.Status, mod.Runtime, mod.NodeCount)
            
            # Initialize default values for incomplete solutions
            obj = None
            y_vals = None
            sol_time = float(mod.Runtime) if hasattr(mod, 'Runtime') else None
            
            # Extract solution information if available
            if mod.Status == GRB.OPTIMAL:
                obj = mod.ObjVal
                y_vals = mod.getAttr("x", y)
            elif mod.Status in [GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, 
                               GRB.SOLUTION_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT]:
                # Solver stopped early but may have found a solution
                try:
                    if mod.SolCount > 0:  # At least one solution found
                        obj = mod.ObjVal
                        y_vals = mod.getAttr("x", y)
                    else:
                        obj = float('inf')  # No feasible solution found
                except Exception as e:
                    logger.warning(f"Error extracting solution after early termination: {e}")
                    obj = float('inf')
            elif mod.Status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                # Problem is infeasible or unbounded
                obj = float('inf') if mod.Status == GRB.INFEASIBLE else float('-inf')
            else:
                # Other error conditions
                logger.warning(f"SCFLP optimization ended with status {mod.Status}: {stopping_reason}")
                obj = float('inf')

            # Calculate true objective value if using prediction
            final_report = _final_objective_report(
                problem_type, data, y_vals, obj, use_prediction, use_stochastic_capacity
            )
            
            
            # Get comprehensive callback diagnostics
            callback_diagnostics = callback.get_callback_diagnostics()
            
            # DEBUG: Log callback diagnostic state before creating Solution object
            logger.info(f"DEBUG - Callback state before Solution creation:")
            logger.info(f"  total_unique_points: {callback.total_unique_points}")
            logger.info(f"  total_repeated_hits: {callback.total_repeated_hits}")
            logger.info(f"  repeated_hits_with_cuts: {callback.repeated_hits_with_cuts}")
            logger.info(f"  callback_counts: {callback.callback_counts}")

            return Solution(
                objective_value=obj if obj is not None else float('inf'),
                **final_report,
                first_stage_optimnal_values=y_vals,
                solution_time=sol_time,
                feature_gen_time_sec=callback.feature_gen_time_sec,
                subproblem_selection_time_sec=callback.subproblem_selection_time_sec,
                predictor_training_time=callback.predictor_training_time,
                KNN_closest_neighbor_time=callback.KNN_closest_neighbor_time,
                num_cuts_mip_selected=callback.num_cuts_mip_selected,
                num_cuts_rel_selected=callback.num_cuts_rel_selected,
                num_cuts_mip_ml=callback.num_cuts_mip_ml,
                num_cuts_rel_ml=callback.num_cuts_rel_ml,
                num_cuts_mip_unselected=callback.num_cuts_mip_unselected,
                num_cuts_rel_unselected=callback.num_cuts_rel_unselected,
                num_cuts_mip_trial=callback.num_cuts_mip_trial,
                num_cuts_rel_trial=callback.num_cuts_rel_trial,
                num_cuts_mip_fallback=callback.num_cuts_mip_fallback,
                num_cuts_rel_fallback=callback.num_cuts_rel_fallback,
                num_cuts_mip_exact_switch=callback.num_cuts_mip_exact_switch,
                num_cuts_rel_exact_switch=callback.num_cuts_rel_exact_switch,
                iteration=callback.iteration,
                num_fallback_used=callback.num_fallback_used,
                num_predictions=callback.num_predictions,
                num_switch_events=callback.num_switch_events,
                switch_iterations=callback.switch_iterations.copy() if callback.switch_iterations else [],
                permanently_switched=callback.permanently_switched,
                num_bnb_nodes=int(mod.NodeCount) if hasattr(mod, 'NodeCount') else 0,
                total_nodes_explored=callback.total_nodes_explored,
                covariate_vectors=callback.feature_vectors,
                phase0_cuts_added=callback.phase0_cuts_added,
                phase0_execution_time_sec=callback.phase0_execution_time_sec,
                phase0_solution_generation_time_sec=callback.phase0_solution_generation_time_sec,
                phase1_execution_time_sec=callback.phase1_execution_time_sec,
                phase2_execution_time_sec=callback.phase2_execution_time_sec,
                knn_optimal_k=getattr(callback.dual_predictor, 'optimal_k', None) if hasattr(callback, 'dual_predictor') and callback.dual_predictor else None,
                stopping_reason=stopping_reason,
                ObjBound_LB=mod.ObjBound if hasattr(mod, 'ObjBound') else None,
                # New diagnostic fields
                total_unique_points=callback.total_unique_points,
                total_repeated_hits=callback.total_repeated_hits,
                repeated_hits_with_cuts=callback.repeated_hits_with_cuts,
                callback_counts=callback.callback_counts.copy(),
                # Task 2 averages: non-root callbacks divided by number of non-root nodes
                avg_mipsol_per_nonroot_node=(callback.callback_counts['mipsol_nonroot'] / max(mod.NodeCount - 1, 1)) if hasattr(mod, 'NodeCount') and mod.NodeCount > 1 else 0.0,
                avg_mipnode_per_nonroot_node=(callback.callback_counts['mipnode_nonroot'] / max(mod.NodeCount - 1, 1)) if hasattr(mod, 'NodeCount') and mod.NodeCount > 1 else 0.0,
                bound_trace=callback.bound_trace.copy(),
                gurobi_cut_statistics=callback.gurobi_cut_statistics.export(),
                master_model_construction_time_sec=master_model_construction_time_sec,
                master_optimize_wall_time_sec=master_optimize_wall_time_sec,
                timing_component_counts=callback.timing_component_counts(),
                cut_construction_breakdown=callback.exclusive_profiler.cut_construction_breakdown(),
                solver_timing_buckets=callback.solver_timing_buckets(master_optimize_wall_time_sec),
                solver_component_breakdown=callback.solver_component_breakdown(master_optimize_wall_time_sec),
            )

    elif pt == "SSLP":
        with Model("SSLP_Master") as mod:
            _set_params(mod)

            # ── 1st-stage variables: open a server j ∈ F
            y = mod.addVars(data.F, vtype=GRB.BINARY, name="y")

            # ── 2nd-stage surrogate: one θ_s per scenario
            theta = mod.addVars(data.S, lb=0.0, name="theta")

            # (optional) force at least one server
            mod.addConstr(quicksum(y[j] for j in data.F) >= 1,
                          name="AtLeastOneServer")

            # ── objective: fixed cost  +  expected recourse
            total_cost = (
                  quicksum(data.fixed_cost[j] * y[j] for j in data.F)
                + quicksum(data.p[s] * theta[s] for s in data.S)
            )
            mod.setObjective(total_cost, GRB.MINIMIZE)

            # ── callback: Benders cuts & ML-augmented prediction
            callback = Callback(
                problem_type        = problem_type,
                data                = data,
                first_stage_values  = y,
                theta_vars          = theta,
                selected_subproblems= selected_subproblems,
                covariate_vectors   = covariate_vectors,
                prediction_method   = prediction_method,
                n_neighbors         = n_neighbors,
                use_prediction      = use_prediction,
                n_trials            = n_trials,
                m_trials            = m_trials,
                feature_params      = feature_params,
                n_selected_subproblems = n_selected_subproblems,
                switching_strategy  = switching_strategy,
                knn_find_optimal_k  = knn_find_optimal_k,
                # TREE parameters
                # tree_max_depth: removed for adaptive TREE method
                # tree_min_leaf_size: removed for Tree-First approach
                tree_min_impurity_decrease=tree_min_impurity_decrease,
                tree_quantile_grid_size=tree_quantile_grid_size,
                # DUAL_CLUSTERING parameters
                dual_clustering_random_state=dual_clustering_random_state,
                dual_clustering_init_method=dual_clustering_init_method,
                dual_clustering_max_iter=dual_clustering_max_iter,
                dual_clustering_tol=dual_clustering_tol,
                # UNCERTAINTY_CLUSTERING parameters
                uncertainty_clustering_random_state=uncertainty_clustering_random_state,
                uncertainty_clustering_init_method=uncertainty_clustering_init_method,
                uncertainty_clustering_max_iter=uncertainty_clustering_max_iter,
                uncertainty_clustering_tol=uncertainty_clustering_tol,
                # BERTSIMAS_RANDOM_SAMPLING parameters
                bertsimas_random_sampling_random_state=bertsimas_random_sampling_random_state,
                # Feature enhancement parameters
                use_covariates=use_covariates
            )
            if WRITE_MP_LP:
                mod.write(f"{mod.ModelName}.lp")

            master_model_construction_time_sec = (
                time.perf_counter() - _master_construction_started
            )
            _optimize_started = time.perf_counter()
            mod.optimize(callback)
            master_optimize_wall_time_sec = time.perf_counter() - _optimize_started
            callback.raise_if_failed()
            callback.append_terminal_bounds(mod)

            # Determine stopping reason and extract solution information
            stopping_reason = _determine_stopping_reason(mod.Status, mod.Runtime, mod.NodeCount)
            
            # Initialize default values for incomplete solutions
            obj = None
            y_vals = None
            sol_time = float(mod.Runtime) if hasattr(mod, 'Runtime') else None
            
            # Extract solution information if available
            if mod.Status == GRB.OPTIMAL:
                obj = mod.ObjVal
                y_vals = mod.getAttr("x", y)
            elif mod.Status in [GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, 
                               GRB.SOLUTION_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT]:
                # Solver stopped early but may have found a solution
                try:
                    if mod.SolCount > 0:  # At least one solution found
                        obj = mod.ObjVal
                        y_vals = mod.getAttr("x", y)
                    else:
                        obj = float('inf')  # No feasible solution found
                except Exception as e:
                    logger.warning(f"Error extracting solution after early termination: {e}")
                    obj = float('inf')
            elif mod.Status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                # Problem is infeasible or unbounded
                obj = float('inf') if mod.Status == GRB.INFEASIBLE else float('-inf')
            else:
                # Other error conditions
                logger.warning(f"SSLP optimization ended with status {mod.Status}: {stopping_reason}")
                obj = float('inf')

            # Calculate true objective value if using prediction
            final_report = _final_objective_report(
                problem_type, data, y_vals, obj, use_prediction, use_stochastic_capacity
            )

            _log_final_master_solution(problem_type, obj, y_vals)

            # Return solution with comprehensive error handling
            sol = Solution(
                objective_value              = obj if obj is not None else float('inf'),
                **final_report,
                first_stage_optimnal_values  = y_vals,
                solution_time                = sol_time,
                feature_gen_time_sec         = callback.feature_gen_time_sec,
                subproblem_selection_time_sec= callback.subproblem_selection_time_sec,
                predictor_training_time      = callback.predictor_training_time,
                KNN_closest_neighbor_time    = callback.KNN_closest_neighbor_time,
                phase0_execution_time_sec    = callback.phase0_execution_time_sec,
                phase0_solution_generation_time_sec = callback.phase0_solution_generation_time_sec,
                phase1_execution_time_sec    = callback.phase1_execution_time_sec,
                phase2_execution_time_sec    = callback.phase2_execution_time_sec,
                num_cuts_mip_selected        = callback.num_cuts_mip_selected,
                num_cuts_rel_selected        = callback.num_cuts_rel_selected,
                num_cuts_mip_ml              = callback.num_cuts_mip_ml,
                num_cuts_rel_ml              = callback.num_cuts_rel_ml,
                num_cuts_mip_unselected      = callback.num_cuts_mip_unselected,
                num_cuts_rel_unselected      = callback.num_cuts_rel_unselected,
                num_cuts_mip_trial           = callback.num_cuts_mip_trial,
                num_cuts_rel_trial           = callback.num_cuts_rel_trial,
                num_cuts_mip_fallback        = callback.num_cuts_mip_fallback,
                num_cuts_rel_fallback        = callback.num_cuts_rel_fallback,
                num_cuts_mip_exact_switch    = callback.num_cuts_mip_exact_switch,
                num_cuts_rel_exact_switch    = callback.num_cuts_rel_exact_switch,
                iteration                    = callback.iteration,
                num_fallback_used            = callback.num_fallback_used,
                num_predictions              = callback.num_predictions,
                num_switch_events            = callback.num_switch_events,
                switch_iterations            = callback.switch_iterations.copy() if callback.switch_iterations else [],
                permanently_switched         = callback.permanently_switched,
                num_bnb_nodes                = int(mod.NodeCount) if hasattr(mod, 'NodeCount') else 0,
                total_nodes_explored         = callback.total_nodes_explored,
                covariate_vectors            = callback.feature_vectors,
                stopping_reason              = stopping_reason
            )
            return sol

    # ---------------------------
    # NE-SP Master Problem
    # ---------------------------
    elif pt == "NESP" or pt == "NE-SP":
        with Model("NESP_Master") as mod:
            _set_params(mod)

            # First-stage: arc activation decisions x_ij for real arcs
            # Create variables indexed by arc position in A
            first_stage_values = mod.addVars(len(data.A), vtype=GRB.BINARY, name="x")
            
            # Second-stage recourse estimates theta_m for each OD pair
            # Multi-cut formulation: one theta per OD pair
            theta_vars = mod.addVars(range(data.n_od_pairs), lb=0.0, name="theta")

            # Objective: construction cost + weighted recourse costs
            # F_ij * x_ij + sum_m omega_m * theta_m
            construction_cost = quicksum(data.F[idx] * first_stage_values[idx] for idx in range(len(data.A)))
            recourse_cost = quicksum(data.weights[m] * theta_vars[m] for m in range(data.n_od_pairs))
            
            mod.setObjective(construction_cost + recourse_cost, GRB.MINIMIZE)
            
            # No explicit first-stage constraints needed for basic NE-SP
            # (Can add budget or other constraints if needed)

            # Callback for Benders cut generation
            callback = Callback(
                problem_type=problem_type,
                data=data,
                first_stage_values=first_stage_values,
                theta_vars=theta_vars,
                selected_subproblems=selected_subproblems,
                covariate_vectors=covariate_vectors,
                method_config=method_config
            )
            
            if WRITE_MP_LP:
                mod.write(f"{mod.ModelName}.lp")

            master_model_construction_time_sec = (
                time.perf_counter() - _master_construction_started
            )
            _optimize_started = time.perf_counter()
            mod.optimize(callback)
            master_optimize_wall_time_sec = time.perf_counter() - _optimize_started
            callback.raise_if_failed()
            callback.append_terminal_bounds(mod)

            # Finalize phase timing after optimization completes
            callback.finalize_phase_timing()

            # Determine stopping reason and extract solution information
            stopping_reason = _determine_stopping_reason(mod.Status, mod.Runtime, mod.NodeCount)
            
            # Initialize default values for incomplete solutions
            obj = None
            x_vals = None
            sol_time = float(mod.Runtime) if hasattr(mod, 'Runtime') else None
            
            # Extract solution information if available
            if mod.Status == GRB.OPTIMAL:
                obj = mod.ObjVal
                obj_bound = mod.ObjBound
                x_vals = mod.getAttr("x", first_stage_values)
            elif mod.Status in [GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, 
                               GRB.SOLUTION_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT]:
                # Solver stopped early but may have found a solution
                try:
                    if mod.SolCount > 0:  # At least one solution found
                        obj = mod.ObjVal
                        obj_bound = mod.ObjBound
                        x_vals = mod.getAttr("x", first_stage_values)
                    else:
                        obj = float('inf')  # No feasible solution found
                        obj_bound = None
                except Exception as e:
                    logger.warning(f"Error extracting solution after early termination: {e}")
                    obj = float('inf')
                    obj_bound = None
            elif mod.Status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                # Problem is infeasible or unbounded
                obj = float('inf') if mod.Status == GRB.INFEASIBLE else float('-inf')
                obj_bound = None
            else:
                # Other error conditions
                logger.warning(f"NE-SP optimization ended with status {mod.Status}: {stopping_reason}")
                obj = float('inf')
                obj_bound = None

            # Calculate true objective value if using prediction
            # For NE-SP, we always use deterministic capacity (no stochastic capacity option)
            final_report = _final_objective_report(
                problem_type, data, x_vals, obj, use_prediction, use_stochastic_capacity=False
            )

            _log_final_master_solution(problem_type, obj, x_vals)
            
            
            # Get comprehensive callback diagnostics
            callback_diagnostics = callback.get_callback_diagnostics()
            
            # DEBUG: Log callback diagnostic state
            # logger.info(f"DEBUG - NE-SP Callback state before Solution creation:")
            # logger.info(f"  total_unique_points: {callback.total_unique_points}")
            # logger.info(f"  total_repeated_hits: {callback.total_repeated_hits}")
            # logger.info(f"  repeated_hits_with_cuts: {callback.repeated_hits_with_cuts}")
            # logger.info(f"  callback_counts: {callback.callback_counts}")

            return Solution(
                objective_value=obj if obj is not None else float('inf'),
                **final_report,
                first_stage_optimnal_values=x_vals,
                solution_time=sol_time,
                feature_gen_time_sec=callback.feature_gen_time_sec,
                subproblem_selection_time_sec=callback.subproblem_selection_time_sec,
                predictor_training_time=callback.predictor_training_time,
                KNN_closest_neighbor_time=callback.KNN_closest_neighbor_time,
                phase0_execution_time_sec=callback.phase0_execution_time_sec,
                phase0_solution_generation_time_sec=callback.phase0_solution_generation_time_sec,
                phase1_execution_time_sec=callback.phase1_execution_time_sec,
                phase2_execution_time_sec=callback.phase2_execution_time_sec,
                num_cuts_mip_selected=callback.num_cuts_mip_selected,
                num_cuts_rel_selected=callback.num_cuts_rel_selected,
                num_cuts_mip_ml=callback.num_cuts_mip_ml,
                num_cuts_rel_ml=callback.num_cuts_rel_ml,
                num_cuts_mip_unselected=callback.num_cuts_mip_unselected,
                num_cuts_rel_unselected=callback.num_cuts_rel_unselected,
                num_cuts_mip_trial=callback.num_cuts_mip_trial,
                num_cuts_rel_trial=callback.num_cuts_rel_trial,
                num_cuts_mip_fallback=callback.num_cuts_mip_fallback,
                num_cuts_rel_fallback=callback.num_cuts_rel_fallback,
                num_cuts_mip_exact_switch=callback.num_cuts_mip_exact_switch,
                num_cuts_rel_exact_switch=callback.num_cuts_rel_exact_switch,
                iteration=callback.iteration,
                num_fallback_used=callback.num_fallback_used,
                num_predictions=callback.num_predictions,
                num_switch_events=callback.num_switch_events,
                switch_iterations=callback.switch_iterations.copy() if callback.switch_iterations else [],
                permanently_switched=callback.permanently_switched,
                num_bnb_nodes=int(mod.NodeCount) if hasattr(mod, 'NodeCount') else 0,
                total_nodes_explored=callback.total_nodes_explored,
                phase0_cuts_added=callback.phase0_cuts_added,
                covariate_vectors=callback.feature_vectors,
                knn_optimal_k=getattr(callback.dual_predictor, 'optimal_k', None) if hasattr(callback, 'dual_predictor') and callback.dual_predictor else None,
                stopping_reason=stopping_reason,
                ObjBound_LB=obj_bound,
                # NEW DIAGNOSTIC FIELDS (Feb 2026)
                total_unique_points=callback.total_unique_points,
                total_repeated_hits=callback.total_repeated_hits,
                repeated_hits_with_cuts=callback.repeated_hits_with_cuts,
                callback_counts=callback.callback_counts.copy(),
                avg_mipsol_per_nonroot_node=(callback.callback_counts['mipsol_nonroot'] / max(mod.NodeCount - 1, 1)) if hasattr(mod, 'NodeCount') and mod.NodeCount > 1 else 0.0,
                avg_mipnode_per_nonroot_node=(callback.callback_counts['mipnode_nonroot'] / max(mod.NodeCount - 1, 1)) if hasattr(mod, 'NodeCount') and mod.NodeCount > 1 else 0.0,
                bound_trace=callback.bound_trace.copy(),
                gurobi_cut_statistics=callback.gurobi_cut_statistics.export(),
                master_model_construction_time_sec=master_model_construction_time_sec,
                master_optimize_wall_time_sec=master_optimize_wall_time_sec,
                timing_component_counts=callback.timing_component_counts(),
                cut_construction_breakdown=callback.exclusive_profiler.cut_construction_breakdown(),
                solver_timing_buckets=callback.solver_timing_buckets(master_optimize_wall_time_sec),
                solver_component_breakdown=callback.solver_component_breakdown(master_optimize_wall_time_sec),
            )

    # ---------------------------
    # SUMND Master Problem
    # ---------------------------
    elif pt in ["SUMND", "SCMND"]:
        weights = np.asarray(data.probabilities, dtype=float)
        if weights.shape != (len(data.S),) or not np.allclose(
            weights, 1.0 / len(data.S), rtol=0.0, atol=1e-12
        ):
            raise ValueError("This campaign's master uses uniform scenario weights; "
                             "nonuniform input would invalidate verified-gap comparisons")
        with Model(f"{pt}_Master") as mod:
            _set_params(mod)

            # SUMND/SCMND master build logs are intentionally disabled to keep runtime logs concise.

            # First-stage: arc activation decisions y_ij for arcs in A
            # Create binary variables indexed by arc position in A
            first_stage_values = mod.addVars(len(data.A), vtype=GRB.BINARY, name="y")
            
            # Second-stage recourse estimates theta_s for each scenario
            # Multi-cut formulation: one theta per scenario
            theta_vars = mod.addVars(data.S, lb=0.0, name="theta")

            scmnd_formulation = str(
                getattr(data, "scmnd_formulation", SCMND_FORMULATION_CURRENT)
            ).upper()
            use_manuscript_scmnd = (
                pt == "SCMND" and scmnd_formulation == SCMND_FORMULATION_MANUSCRIPT
            )

            # Auxiliary master flows.  The preserved current formulation routes
            # one normalized unit per commodity and links each flow separately
            # to y.  The selectable manuscript formulation routes each
            # commodity's maximum sampled demand and enforces the shared fixed
            # capacity envelope sum_k z[a,k] <= u[a] y[a].
            commodity_ids = [int(k) for k in data.K]
            node_ids = [int(i) for i in data.N]
            z = mod.addVars(
                [(arc_idx, k) for arc_idx in range(len(data.A)) for k in commodity_ids],
                lb=0.0,
                ub=GRB.INFINITY if use_manuscript_scmnd else 1.0,
                name="z"
            )

            outgoing_arcs_by_node = {i: [] for i in node_ids}
            incoming_arcs_by_node = {i: [] for i in node_ids}
            for arc_idx, (tail, head) in enumerate(data.A):
                tail_i = int(tail)
                head_i = int(head)
                if tail_i in outgoing_arcs_by_node:
                    outgoing_arcs_by_node[tail_i].append(arc_idx)
                if head_i in incoming_arcs_by_node:
                    incoming_arcs_by_node[head_i].append(arc_idx)

            envelope_demand = (
                scmnd_envelope_demands(data)
                if use_manuscript_scmnd
                else np.ones(len(commodity_ids), dtype=float)
            )

            # Per-commodity flow conservation.  The RHS is one in the current
            # formulation and the maximum sampled demand in the manuscript one.
            for k in commodity_ids:
                o_k = int(data.origins[k])
                t_k = int(data.destinations[k])
                for i in node_ids:
                    amount = float(envelope_demand[k])
                    rhs = amount if i == o_k else (-amount if i == t_k else 0.0)
                    mod.addConstr(
                        quicksum(z[(arc_idx, k)] for arc_idx in outgoing_arcs_by_node[i]) -
                        quicksum(z[(arc_idx, k)] for arc_idx in incoming_arcs_by_node[i]) == rhs,
                        name=f"{pt}_MasterFlowBalance_k{k}_i{i}"
                    )

            if use_manuscript_scmnd:
                mod.addConstrs(
                    (
                        quicksum(z[(arc_idx, k)] for k in commodity_ids)
                        <= float(data.u_base[arc_idx]) * first_stage_values[arc_idx]
                        for arc_idx in range(len(data.A))
                    ),
                    name="SCMND_EnvelopeCapacity",
                )
            else:
                # Preserve the current normalized per-commodity linking model.
                mod.addConstrs(
                    (
                        z[(arc_idx, k)] <= first_stage_values[arc_idx]
                        for arc_idx in range(len(data.A))
                        for k in commodity_ids
                    ),
                    name=f"{pt}_MasterFlowLink"
                )

            # Objective: fixed activation cost + average recourse estimate
            # sum_{(i,j) in A} f_ij * y_ij + (1/|S|) * sum_s theta_s
            activation_cost = quicksum(data.F[idx] * first_stage_values[idx] for idx in range(len(data.A)))
            recourse_cost = (1.0 / len(data.S)) * quicksum(theta_vars[s] for s in data.S)
            
            mod.setObjective(activation_cost + recourse_cost, GRB.MINIMIZE)

            # Callback for Benders cut generation
            callback = Callback(
                problem_type=problem_type,
                data=data,
                first_stage_values=first_stage_values,
                theta_vars=theta_vars,
                selected_subproblems=selected_subproblems,
                covariate_vectors=covariate_vectors,
                method_config=method_config
            )
            
            if WRITE_MP_LP:
                mod.write(f"{mod.ModelName}.lp")

            master_model_construction_time_sec = (
                time.perf_counter() - _master_construction_started
            )
            _optimize_started = time.perf_counter()
            mod.optimize(callback)
            master_optimize_wall_time_sec = time.perf_counter() - _optimize_started
            callback.raise_if_failed()
            callback.append_terminal_bounds(mod)

            # Finalize phase timing after optimization completes
            callback.finalize_phase_timing()

            # Determine stopping reason and extract solution information
            stopping_reason = _determine_stopping_reason(mod.Status, mod.Runtime, mod.NodeCount)
            
            # Initialize default values for incomplete solutions
            obj = None
            x_vals = None
            sol_time = float(mod.Runtime) if hasattr(mod, 'Runtime') else None
            
            # Extract solution information if available
            if mod.Status == GRB.OPTIMAL:
                obj = mod.ObjVal
                obj_bound = mod.ObjBound
                x_vals = mod.getAttr("x", first_stage_values)
            elif mod.Status in [GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, 
                               GRB.SOLUTION_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT]:
                # Solver stopped early but may have found a solution
                try:
                    if mod.SolCount > 0:  # At least one solution found
                        obj = mod.ObjVal
                        obj_bound = mod.ObjBound
                        x_vals = mod.getAttr("x", first_stage_values)
                    else:
                        obj = float('inf')  # No feasible solution found
                        obj_bound = None
                except Exception as e:
                    logger.warning(f"Error extracting solution after early termination: {e}")
                    obj = float('inf')
                    obj_bound = None
            elif mod.Status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                # Problem is infeasible or unbounded
                obj = float('inf') if mod.Status == GRB.INFEASIBLE else float('-inf')
                obj_bound = None
            else:
                # Other error conditions
                logger.warning(f"{pt} optimization ended with status {mod.Status}: {stopping_reason}")
                obj = float('inf')
                obj_bound = None

            # Calculate true objective value if using prediction.
            # For SUMND/SCMND, capacities are scenario-dependent.
            final_report = _final_objective_report(
                problem_type, data, x_vals, obj, use_prediction, use_stochastic_capacity=True
            )

            
            # Get comprehensive callback diagnostics
            callback_diagnostics = callback.get_callback_diagnostics()
            
            # Extract Gurobi parameters for logging
            mip_gap_stop = mod.MIPGap if hasattr(mod, 'MIPGap') else None
            time_limit_value = mod.Params.TimeLimit if hasattr(mod, 'Params') and hasattr(mod.Params, 'TimeLimit') else None
            
            return Solution(
                objective_value=obj if obj is not None else float('inf'),
                **final_report,
                first_stage_optimnal_values=x_vals,
                solution_time=sol_time,
                feature_gen_time_sec=callback.feature_gen_time_sec,
                subproblem_selection_time_sec=callback.subproblem_selection_time_sec,
                predictor_training_time=callback.predictor_training_time,
                KNN_closest_neighbor_time=callback.KNN_closest_neighbor_time,
                phase0_execution_time_sec=callback.phase0_execution_time_sec,
                phase0_solution_generation_time_sec=callback.phase0_solution_generation_time_sec,
                phase1_execution_time_sec=callback.phase1_execution_time_sec,
                phase2_execution_time_sec=callback.phase2_execution_time_sec,
                num_cuts_mip_selected=callback.num_cuts_mip_selected,
                num_cuts_rel_selected=callback.num_cuts_rel_selected,
                num_cuts_mip_ml=callback.num_cuts_mip_ml,
                num_cuts_rel_ml=callback.num_cuts_rel_ml,
                num_cuts_mip_unselected=callback.num_cuts_mip_unselected,
                num_cuts_rel_unselected=callback.num_cuts_rel_unselected,
                num_cuts_mip_trial=callback.num_cuts_mip_trial,
                num_cuts_rel_trial=callback.num_cuts_rel_trial,
                num_cuts_mip_fallback=callback.num_cuts_mip_fallback,
                num_cuts_rel_fallback=callback.num_cuts_rel_fallback,
                num_cuts_mip_exact_switch=callback.num_cuts_mip_exact_switch,
                num_cuts_rel_exact_switch=callback.num_cuts_rel_exact_switch,
                iteration=callback.iteration,
                num_fallback_used=callback.num_fallback_used,
                num_predictions=callback.num_predictions,
                num_switch_events=callback.num_switch_events,
                switch_iterations=callback.switch_iterations.copy() if callback.switch_iterations else [],
                permanently_switched=callback.permanently_switched,
                num_bnb_nodes=int(mod.NodeCount) if hasattr(mod, 'NodeCount') else 0,
                total_nodes_explored=callback.total_nodes_explored,
                phase0_cuts_added=callback.phase0_cuts_added,
                covariate_vectors=callback.feature_vectors,
                knn_optimal_k=getattr(callback.dual_predictor, 'optimal_k', None) if hasattr(callback, 'dual_predictor') and callback.dual_predictor else None,
                stopping_reason=stopping_reason,
                ObjBound_LB=obj_bound,
                # NEW DIAGNOSTIC FIELDS (Feb 2026)
                total_unique_points=callback.total_unique_points,
                total_repeated_hits=callback.total_repeated_hits,
                repeated_hits_with_cuts=callback.repeated_hits_with_cuts,
                callback_counts=callback.callback_counts.copy(),
                avg_mipsol_per_nonroot_node=(callback.callback_counts['mipsol_nonroot'] / max(mod.NodeCount - 1, 1)) if hasattr(mod, 'NodeCount') and mod.NodeCount > 1 else 0.0,
                avg_mipnode_per_nonroot_node=(callback.callback_counts['mipnode_nonroot'] / max(mod.NodeCount - 1, 1)) if hasattr(mod, 'NodeCount') and mod.NodeCount > 1 else 0.0,
                bound_trace=callback.bound_trace.copy(),
                gurobi_cut_statistics=callback.gurobi_cut_statistics.export(),
                master_model_construction_time_sec=master_model_construction_time_sec,
                master_optimize_wall_time_sec=master_optimize_wall_time_sec,
                timing_component_counts=callback.timing_component_counts(),
                cut_construction_breakdown=callback.exclusive_profiler.cut_construction_breakdown(),
                solver_timing_buckets=callback.solver_timing_buckets(master_optimize_wall_time_sec),
                solver_component_breakdown=callback.solver_component_breakdown(master_optimize_wall_time_sec),
                parametric_diagnostics=(
                    callback.get_parametric_diagnostics()
                ),
                # Gurobi parameters logging
                mip_gap_stop=mip_gap_stop,
                mip_gap_target=mod.Params.MIPGap if hasattr(mod, 'Params') and hasattr(mod.Params, 'MIPGap') else None,
                time_limit_sec=time_limit_value
            )

    elif pt in ["UFL", "UHL", "SSLP", "CMND", "MCFL"]:
        raise NotImplementedError(f"{pt} master implementation removed - focusing on SCFLP only")
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
