"""
callbacks.py

This module implements a customized callback for Benders decomposition within Gurobi's
branch-and-bound framework, enhanced with machine learning–based dual prediction using
the unified DualPredictor architecture supporting k-NN, TREE, DUAL_CLUSTERING, 
DUAL_CLUSTERING_WITH_DUAL, UNCERTAINTY_CLUSTERING, and BERTSIMAS_RANDOM_SAMPLING methods.

Overview:
---------
The core class `Callback` supports a three-phase decomposition strategy for large-scale 
two-stage stochastic problems, utilizing the unified `DualPredictor` class from 
`Dual_Prediction.prediction` for seamless integration of k-NN, TREE, DUAL_CLUSTERING, 
DUAL_CLUSTERING_WITH_DUAL, UNCERTAINTY_CLUSTERING, and BERTSIMAS_RANDOM_SAMPLING prediction methods.

PHASE 0: Warm-up Phase (Optional - if m_trials > 0)
    - Generate `m_trials` random feasible first-stage solutions using phase0_generator
    - For each solution, solve all dual subproblems exactly to collect optimal duals
    - Collect all cuts and add them to the master problem at the end of Phase 0
    - Provides warm-start cuts to strengthen master problem before Phase 1 begins

PHASE 1: Traditional Benders Decomposition (Learning Phase)
    - For the first `n_trials` iterations, the callback solves all dual subproblems
      exactly using the current first-stage solution
    - Exact cuts are generated at ALL registered callback points (integer and/or LP)
    - The duals are stored for future training of the unified predictor
    - Combined with Phase 0 duals for feature generation at end of Phase 1

PHASE 2: Learning-Augmented Benders Decomposition (Context-Aware Prediction)
    - **ONE-TIME SETUP**: At transition from Phase 1 to Phase 2:
        1. Generate feature vectors from combined Phase 0 + Phase 1 duals
        2. Call `_train_unified()` which handles both subproblem selection and training
        3. Create method-specific mappings (k-NN neighbors, TREE leaf assignments, or clustering mappings)
    - **ITERATIVE PREDICTION**: For each subsequent iteration:
        1. Check use_prediction method to determine if ML or exact for current context
        2. If ML: Solve selected subproblems exactly, predict duals for unselected ones
        3. If exact: Solve all subproblems exactly (traditional Benders)
        4. Automatic fallback handling when predictions fail
    
    Supported Methods:
        - **KNN**: k-Nearest neighbors with integrated selection and mapping
        - **TREE**: Tree-based medoid selection with dual reconstruction
        - **DUAL_CLUSTERING**: K-means clustering in feature space with cluster representatives
        - **DUAL_CLUSTERING_WITH_DUAL**: K-medoids clustering in dual space with medoid representatives
        - **UNCERTAINTY_CLUSTERING**: K-means clustering on uncertainty vectors with Phase 0/1 skipping
        - **BERTSIMAS_RANDOM_SAMPLING**: Random scenario sampling with simple averaging, skips Phase 0/1
    
    Switching strategies for handling prediction failures:
        - **no_switch**: Always use ML prediction with fallback only on failure
        - **alternate**: Solve exactly once when no cuts generated, then resume ML
        - **permanent_switch**: Switch permanently to exact solving when ML fails

Approach Categories (Category System for Cleaner Code):
------------------------------------------------------
The callback uses a category-based classification system for cleaner control flow logic:

**Category 1: Exact** (Pure Benders Decomposition - Baseline)
  - Methods: int_exact, int_and_LP_ML_for_none
  - Behavior: Standard Benders with NO phases - solves ALL subproblems exactly at EVERY iteration
  - No Phase 0, No Phase 1, No Phase 2
  - No ML predictions, no feature generation, no training, no partitioning
  - No iteration cap (runs until Gurobi finds optimal solution)
  - int_exact: Cuts only at integer points
  - int_and_LP_ML_for_none: Cuts at both integer and LP relaxation points

**Category 2: ML_Augmented_BD** (Proposed Method - Main Contribution)
  - Methods: KNN, TREE, DUAL_CLUSTERING
  - Phase 0: Execute if m_trials > 0
  - Phase 1: Execute for n_trials iterations  
  - Features: Generated from dual vectors via graph-based methods
  - Training: 3 variants (KNN, TREE, Clustering) for partitioning and prediction

**Category 3: ML_Augmented_BD_with_duals** (Benchmark)
  - Methods: DUAL_CLUSTERING_WITH_DUAL
  - Phase 0: Execute if m_trials > 0
  - Phase 1: Execute for n_trials iterations
  - Features: Use dual vectors directly (no transformation)
  - Training: Clustering in dual space

**Category 4: Covariate_Similarity** (Benchmark)
  - Methods: COVARIATE_CLUSTERING
  - Phase 0/1: Skipped
  - Features: Use input covariate vectors
  - Training: Clustering on covariates

**Category 5: Uncertainty_Similarity** (Benchmark)
  - Methods: UNCERTAINTY_CLUSTERING
  - Phase 0/1: Skipped
  - Features: Use uncertainty vectors (demand patterns)
  - Training: Clustering on uncertainty

**Category 6: Random_Sampling** (Benchmark)
  - Methods: BERTSIMAS_RANDOM_SAMPLING
  - Phase 0/1: Skipped
  - Features: None (random sampling)
  - Training: Random selection + simple averaging

The category system (defined in utils/config_utils.py) provides helper functions like
should_execute_phase0(), should_generate_features_from_duals(), etc., enabling clean,
maintainable control flow without scattered method name checks throughout the code.

Integrated Prediction Methods:
------------------------------
All prediction methods include their own optimal subproblem selection strategy:

- **KNN Method**: k-Nearest neighbors with integrated optimal selection (fully implemented)
  - Uses paper's mathematical objective: min_{T⊆S,|T|=p} Σ_{s∈S\\T} (1/k) Σ_{t∈T_k(s)} d_F(f^s, f^t)
  - Predicts using uniform averaging over k nearest selected scenarios
  - Guarantees dual feasibility for all predicted scenarios
  - Best accuracy but higher computational cost during training

Callback Trigger Points and Registration:
-----------------------------------------
The callback registration depends on the `use_prediction` parameter:

**Methods with Both Integer and LP Callbacks** (LP callbacks registered):
- `int_and_LP_ML_for_int`: Phase1: exact at both | Phase2: ML at integer, exact at LP
- `int_and_LP_ML_for_LP`: Phase1: exact at both | Phase2: exact at integer, ML at LP  
- `int_and_LP_ML_for_none`: Phase1: exact at both | Phase2: exact at both (standard Benders with LP cuts)
- `int_and_LP_ML_for_both`: Phase1: exact at both | Phase2: ML at both

**Methods with Integer Callbacks Only** (LP callbacks NOT registered):
- `int_ML`: Phase1: exact at integer | Phase2: ML at integer
- `int_exact`: Phase1: exact at integer | Phase2: exact at integer (standard Benders)

**Gurobi Callback Types**:
- `MIPSOL`: Always active for integer-feasible solutions. Cuts added via `cbLazy`.
- `MIPNODE`: Conditionally active for LP relaxations based on method. Cuts added via `cbCut`.

**Backward Compatibility**:
- `True` → maps to `int_and_LP_ML_for_both`
 - `False` → maps to `int_exact`

Cut Tracking:
-------------
The callback maintains counters for each scenario to track the number and source of cuts:
- Phase 1: `*_trial`
- Phase 2:
    - Selected subproblems: `*_selected`
    - KNN predictor cuts: `*_ml_knn`
    - Fallback methods: `*_fallback`
All cut types are tracked separately for relaxed and integer solutions.

Integration with Unified Architecture:
--------------------------------------
The callback uses the integrated prediction and selection approach:

1. **Initialization**: Creates `DualPredictor` instance with chosen prediction method
2. **Integrated Training**: Calls `predictor._train_unified()` which handles both selection and training  
3. **Prediction Phase**: Uses method-specific mappings for dual prediction
4. **No Separate Selection**: Subproblem selection integrated within prediction method
5. **Fallback Handling**: Automatic fallback when predictions fail or generate no cuts

Example usage in callback initialization:
```python
# Unified predictor with integrated selection
self.dual_predictor = DualPredictor(
    method=prediction_method,  # KNN (SCFLP focus)
    problem_type=problem_type,
    data=data,
    n_neighbors=n_neighbors    # KNN-specific parameters
)
```

Removed Parameters:
-------------------
The following parameters are no longer needed due to integrated architecture:
- `subproblem_selection_method`: Selection now integrated within prediction method
- Separate selection method configurations: "trial-cut-freq", "p-median", "kmeans", "effective_sub"

Use Cases:
----------
This implementation is suitable for large-scale Benders problems with many scenarios,
where solving every subproblem is computationally expensive, and the unified predictive
architecture can provide good dual estimates with consistent interfaces across all methods.

"""


from gurobipy import GRB, quicksum
from .subproblem_dual import solve_dual_subproblem
from .data import scmnd_capacity_vector
from Dual_Prediction.prediction import DualPredictor
from Dual_Prediction.parametric import is_parametric_predictor
from utils.exclusive_profiler import ExclusiveProfiler, profiled, profile_call, TIMING_CATEGORIES
from utils.convergence_diagnostics import _finite
from utils.gurobi_cut_diagnostics import GurobiCutStatistics
import numpy as np
from feature_engineering.feature_generation import compute_scenario_features_from_duals
import time
import logging
import warnings
import sys
import os

# Import Phase 0 generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0_generator import generate_phase0_solutions

# Import config utilities for new method configuration system
from utils.config_utils import (
    # Core method configuration
    get_method_category,
    get_ml_method_type,
    get_predictor_type,
    should_execute_step,
    should_register_lp_callbacks,
    should_use_ml_for_unselected,
    get_input_data_for_step5,
    get_uncertainty_vectors_for_problem,
    validate_method_config,
    get_method_description,
    get_steps_description,
    # Backward compatibility functions (for legacy code during transition)
    normalize_use_prediction,
    ml_enabled_for_context,
    uses_ml_in_phase2,
    should_execute_phase0,
    should_execute_phase1,
    should_generate_features_from_duals,
    uses_direct_dual_vectors,
    uses_input_covariates,
    uses_uncertainty_vectors,
    uses_random_sampling,
    get_approach_category,
    get_pred_or_exact
)

from utils.exclusive_profiler import get_profiled_logger

logger = get_profiled_logger(__name__)

class Callback:
    """
    Unified Benders decomposition callback supporting three-phase ML-augmented optimization.
    """
    
    # Class-level constant: Numerical tolerance for cut violation checks
    # Prevents missing cuts due to floating-point arithmetic errors
    EPS = 1e-6
    
    def __init__(
        self,
        problem_type,
        data,
        first_stage_values,
        theta_vars,
        selected_subproblems,
        covariate_vectors,
        method_config
    ):
        """
        Initialize the callback for learning-augmented Benders decomposition with 6-step pattern.

        Args:
            problem_type (str): Type of problem (e.g., SCFLP).
            data (Data): Problem data object.
            first_stage_values (dict): First-stage Gurobi decision variables.
            theta_vars (dict): Second-stage cost variables (theta).
            selected_subproblems (list): List of selected subproblem indices.
            covariate_vectors (dict): Covariate vectors for scenarios (if applicable).
            method_config (dict): Required method configuration with keys:
                - 'method_category': 'Exact' or 'ML'
                - 'exact_callback_type': 'int_only' or 'int_and_LP' (if Exact)
                - 'ml_method_type': ML method subtype (if ML)
                - 'ml_callback_type': ML callback strategy (if ML)
                - 'predictor_type': 'KNN', 'TREE', or 'CLUSTERING' (if ML, except Random_Sampling)
                - 'n_trials': Phase 1 iterations
                - 'm_trials': Phase 0 iterations
                - 'selected_subproblems_ratio': Selection ratio
                - Predictor-specific parameters
                - 'switching_strategy': 'no_switch', 'alternate', or 'permanent_switch'
                - 'use_covariates': 'features_only', 'features_plus_covariates', or 'covariates_only'
                - 'use_covariates': Whether to use covariates with features (for ML_Augmented_BD)
                - 'use_stochastic_capacity': Whether to use scenario-specific capacities
        """

        # Problem and master references
        self.problem_type = str(problem_type).upper()
        self.data = data
        self.first_stage_values = first_stage_values
        self.theta = theta_vars
        
        # Store selected subproblems and covariate vectors
        self.selected_subproblems = selected_subproblems if selected_subproblems is not None else []
        self.covariate_vectors = covariate_vectors if covariate_vectors is not None else {}

        # Validate method_config is provided
        if method_config is None:
            raise ValueError("method_config parameter is required. Legacy parameter passing is no longer supported.")
        
        # New unified configuration system
        self.method_config = method_config.copy()
        
        # Validate configuration
        is_valid, error_msg = validate_method_config(self.method_config)
        if not is_valid:
            raise ValueError(f"Invalid method_config: {error_msg}")
        
        # Extract core method information
        self.method_category = get_method_category(self.method_config)
        self.ml_method_type = get_ml_method_type(self.method_config)
        self.predictor_type = get_predictor_type(self.method_config)
        
        # Extract iteration parameters
        self.n_trials = int(self.method_config.get('n_trials', 5))
        self.m_trials = int(self.method_config.get('m_trials', 0))
        
        # Extract predictor-specific parameters
        self.n_neighbors = int(self.method_config.get('n_neighbors', 5))
        self.knn_find_optimal_k = bool(self.method_config.get('knn_find_optimal_k', False))
        self.tree_min_leaf_size = int(self.method_config.get('tree_min_leaf_size', 5))
        self.tree_min_impurity_decrease = float(self.method_config.get('tree_min_impurity_decrease', 0.0))
        self.tree_quantile_grid_size = int(self.method_config.get('tree_quantile_grid_size', 10))
        
        # Unified clustering parameters
        self.clustering_random_state = int(self.method_config.get('clustering_random_state', 42))
        self.clustering_init_method = str(self.method_config.get('clustering_init_method', 'k-means++'))
        self.clustering_max_iter = int(self.method_config.get('clustering_max_iter', 300))
        self.clustering_tol = float(self.method_config.get('clustering_tol', 1e-4))
        
        # Other configuration
        self.switching_strategy = str(self.method_config.get('switching_strategy', 'no_switch'))
        raw_use_covariates = self.method_config.get('use_covariates')
        if raw_use_covariates is None:
            # Fallback to default
            self.use_covariates = 'features_only'
        else:
            self.use_covariates = str(raw_use_covariates)
        
        # Validate use_covariates (tri-state: features_only, features_plus_covariates, covariates_only)
        valid_use_covariates_modes = {'features_only', 'features_plus_covariates', 'covariates_only'}
        if self.use_covariates not in valid_use_covariates_modes:
            logger.warning(
                f"Invalid use_covariates='{self.use_covariates}'. "
                "Falling back to 'features_only'."
            )
            self.use_covariates = 'features_only'
        self.use_stochastic_capacity = bool(self.method_config.get('use_stochastic_capacity', False))
        self.feature_params = self.method_config.get('feature_params', {})
        self.n_selected_subproblems = self.method_config.get('n_selected_subproblems')
        self.selected_subproblems_ratio = float(self.method_config.get('selected_subproblems_ratio', 0.2))
        
        # Derive legacy compatibility attributes from method_config
        # These maintain compatibility with existing helper methods and Phase 2 logic
        # without extensive refactoring
        
        # 1. self.use_prediction - for callback type determination
        if self.method_category == 'Exact':
            callback_type = self.method_config.get('exact_callback_type', 'int_only')
            if callback_type == 'int_only':
                self.use_prediction = 'int_exact'
            else:  # int_and_LP
                self.use_prediction = 'int_and_LP_ML_for_none'
        else:  # ML
            callback_type = self.method_config.get('ml_callback_type', 'ML_at_both')
            if callback_type == 'ML_at_both':
                self.use_prediction = 'int_and_LP_ML_for_both'
            elif callback_type == 'ML_at_integer_exact_at_LP':
                self.use_prediction = 'int_and_LP_ML_for_int'
            elif callback_type == 'ML_at_LP_exact_at_integer':
                self.use_prediction = 'int_and_LP_ML_for_LP'
            elif callback_type == 'ML_at_integer_only':
                self.use_prediction = 'int_ML'
            else:
                raise ValueError(f"Unknown ml_callback_type: {callback_type}")
        
        # 2. self.approach_category - for feature generation logic
        self.approach_category = self.ml_method_type if self.method_category == 'ML' else 'Exact'
        
        # 3. self.prediction_method - for predictor-specific logic
        if self.method_category == 'ML':
            ml_to_legacy = {
                'ML_Augmented_BD': {
                    'KNN': 'KNN',
                    'TREE': 'TREE',
                    'CLUSTERING': 'DUAL_CLUSTERING',
                    'PARAMETRIC_MANUSCRIPT': 'PARAMETRIC_MANUSCRIPT',
                    'PARAMETRIC_PENALIZED_ERRORS': 'PARAMETRIC_PENALIZED_ERRORS',
                    'PARAMETRIC_RIDGE_REPAIR': 'PARAMETRIC_RIDGE_REPAIR',
                    'PARAMETRIC_RHS_RIDGE_REPAIR': 'PARAMETRIC_RHS_RIDGE_REPAIR',
                    'PARAMETRIC_RHS_RIDGE_TREE_ANCHOR': 'PARAMETRIC_RHS_RIDGE_TREE_ANCHOR',
                    'PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR': 'PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR',
                    'PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR': 'PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR',
                    'PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN': 'PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN',
                    'PMEDIAN_WEIGHTED_TRANSFER': 'PMEDIAN_WEIGHTED_TRANSFER',
                    'PMEDIAN_NEAREST_TRANSFER': 'PMEDIAN_NEAREST_TRANSFER',
                },
                'ML_Augmented_BD_with_duals': {
                    'KNN': 'DUAL_CLUSTERING_WITH_DUAL',
                    'TREE': 'DUAL_CLUSTERING_WITH_DUAL',
                    'CLUSTERING': 'DUAL_CLUSTERING_WITH_DUAL'
                },
                'Uncertainty_Similarity': {
                    'KNN': 'UNCERTAINTY_CLUSTERING',
                    'TREE': 'UNCERTAINTY_CLUSTERING',
                    'CLUSTERING': 'UNCERTAINTY_CLUSTERING'
                },
                'Covariate_Similarity': {
                    'KNN': 'COVARIATE_CLUSTERING',
                    'TREE': 'COVARIATE_CLUSTERING',
                    'CLUSTERING': 'COVARIATE_CLUSTERING'
                },
                'Random_Sampling': {
                    None: 'BERTSIMAS_RANDOM_SAMPLING'
                }
            }
            method_map = ml_to_legacy.get(self.ml_method_type, {})
            self.prediction_method = method_map.get(self.predictor_type, 'KNN')
        else:
            self.prediction_method = 'SIMPLE_AVG'  # Placeholder for Exact methods
        
        # 4. Legacy pred_or_exact field (for older validation logic)
        self.pred_or_exact = get_pred_or_exact(self.use_prediction)
        
        # 5. fallback_method - for handling prediction failures
        self.fallback_method = 'SOLVE'  # Default: solve exactly when prediction fails
        
        # Determine execution plan for 6 steps
        self.execute_step = {i: should_execute_step(self.method_config, i) for i in range(1, 7)}
        
        # Log method configuration
        logger.info(f"Method configured: {get_method_description(self.method_config)}")
        logger.info(f"Execution plan: {get_steps_description(self.method_config)}")
        
        # Switching strategy validation
        valid_strategies = {"no_switch", "alternate", "permanent_switch"}
        if self.switching_strategy not in valid_strategies:
            raise ValueError(f"switching_strategy must be one of {valid_strategies}, got: {self.switching_strategy}")
        self.permanently_switched = False  # Flag for permanent_switch mode

        # Initialize unified dual predictor with new architecture (only for ML methods)
        # ml_method_type and predictor_type are set above in both new and legacy paths
        if self.method_category == 'ML' and self.ml_method_type is not None:
            self.dual_predictor = DualPredictor(
                ml_method_type=self.ml_method_type,
                predictor_type=self.predictor_type,
                problem_type=self.problem_type,
                data=self.data,
                n_neighbors=self.n_neighbors,
                min_samples_leaf=self.tree_min_leaf_size,
                # Unified clustering parameters for all clustering methods
                clustering_random_state=self.clustering_random_state,
                clustering_init_method=self.clustering_init_method,
                clustering_max_iter=self.clustering_max_iter,
                clustering_tol=self.clustering_tol,
                # Random sampling parameters
                random_sampling_random_state=self.clustering_random_state,
                use_stochastic_capacity=self.use_stochastic_capacity,
                parametric_config=self.method_config,
            )
        else:
            # For Exact methods, no predictor is needed
            self.dual_predictor = None

        # Internal states
        self.iteration = 0
        self.x_trials = []
        self.first_stage_by_iteration = {}
        self.dual_vectors = {}
        self.features_computed = False
        self.trained_models = {}  # Legacy - replaced by dual_predictor
        self.feature_vectors = covariate_vectors if covariate_vectors is not None else {}
        self.selected_subproblems = set(selected_subproblems or []) if selected_subproblems is not None else set()
        self.knn_neighbors = {}  # Legacy - replaced by dual_predictor

        # Phase initialization: Determine starting phase based on method configuration
        if self.execute_step[3]:  # Phase 1 (Standard Benders iterations)
            self.phase = 1
            logger.info(f"Starting in Phase 1 (Standard Benders) for {self.n_trials} iterations")
        elif self.execute_step[6]:  # Phase 2 (Skip directly to ML phase)
            self.phase = 2
            logger.info("Skipping Phase 1, starting directly in Phase 2")
        else:
            self.phase = 0
            logger.info("No phases configured (Exact method)")

        # Phase 0 state variables
        self.phase0_executed = False
        self.phase0_solutions = []
        self.phase0_dual_vectors = {}  # Storage for Phase 0 duals
        self.phase0_cuts_added = 0
        
        # Logging control flags
        self.phase1_skip_logged = False  # Flag to log "Skipping Phase 1" only once

        # Cut counters indexed by sid
        #
        # Mathematical Relationships and Cut Classification Framework:
        # ===========================================================
        #
        # The learning-augmented Benders decomposition algorithm generates optimality cuts
        # from different sources and at different solution types. These counters track cuts
        # by their generation method and solution context for performance analysis.
        #
        # ALGORITHM PHASES:
        # ----------------
        # Phase 1 (iterations 1 ≤ t ≤ n_trials): Traditional Benders - solve all scenarios exactly
        # Phase 2 (iterations t > n_trials): ML-augmented Benders - selected + predicted scenarios
        #
        # SOLUTION TYPES:
        # --------------
        # • MIP: Integer-feasible solutions (MIPSOL callback) → adds lazy constraints
        # • REL: LP relaxation solutions (MIPNODE callback) → adds cutting planes
        #
        # CUT SOURCE CLASSIFICATION:
        # -------------------------
        # Let S = {0, 1, ..., |scenarios|-1} be all scenario indices
        # Let S_sel ⊆ S be selected subproblems for ML phase
        # Let S_unsel = S \ S_sel be unselected subproblems
        #
        # Mathematical Cut Relationships:
        # ==============================
        #
        # 1. PHASE PARTITIONING:
        #    Total_cuts(sid) = Trial_cuts(sid) + Phase2_cuts(sid)
        #    where:
        #    • Trial_cuts(sid) = num_cuts_mip_trial[sid] + num_cuts_rel_trial[sid]
        #    • Phase2_cuts(sid) = Selected_cuts(sid) + Unselected_cuts(sid)
        #
        # 2. SELECTED SUBPROBLEM CUTS (Phase 2, sid ∈ S_sel):
        #    Selected_cuts(sid) = num_cuts_mip_selected[sid] + num_cuts_rel_selected[sid]
        #    Source: Exact dual subproblem solving for selected scenarios
        #
        # 3. UNSELECTED SUBPROBLEM CUTS (Phase 2, sid ∈ S_unsel):
        #    Unselected_cuts(sid) = ML_cuts(sid) + Fallback_cuts(sid) + Switch_cuts(sid)
        #    where:
        #    • ML_cuts(sid) = num_cuts_mip_ml[sid] + num_cuts_rel_ml[sid] (unified across all ML methods)
        #    • Fallback_cuts(sid) = num_cuts_mip_fallback[sid] + num_cuts_rel_fallback[sid]
        #    • Switch_cuts(sid) = num_cuts_mip_exact_switch[sid] + num_cuts_rel_exact_switch[sid]
        #
        # 4. MUTUAL EXCLUSIVITY CONSTRAINTS:
        #    For any scenario sid and solution type:
        #    • EITHER in selected OR unselected (never both simultaneously)
        #    • ML prediction method (KNN) tracked uniformly via ML_cuts
        #    • Fallback only when primary prediction fails
        #    • Switch cuts only when switching strategies are active
        #
        # 5. CONSERVATION EQUATIONS:
        #    ∀ sid ∈ S:
        #    Total_MIP_cuts(sid) = num_cuts_mip_trial[sid] + num_cuts_mip_selected[sid] + 
        #                          num_cuts_mip_ml[sid] + num_cuts_mip_fallback[sid] + 
        #                          num_cuts_mip_exact_switch[sid]
        #    
        #    Total_REL_cuts(sid) = num_cuts_rel_trial[sid] + num_cuts_rel_selected[sid] + 
        #                          num_cuts_rel_ml[sid] + num_cuts_rel_fallback[sid] + 
        #                          num_cuts_rel_exact_switch[sid]
        #
        # 6. CONDITIONAL RELATIONSHIPS:
        #    • If use_prediction = False: Only trial + fallback cuts generated
        #    • If use_prediction = True: ML cuts generated regardless of prediction_method
        #    • If switching_strategy ≠ 'no_switch': Additional exact_switch cuts possible
        #
        # 7. ALGORITHMIC IMPLICATIONS:
        #    • High trial cuts → Scenarios frequently violated in Phase 1
        #    • High selected cuts → Selected scenarios remain important in Phase 2
        #    • High ML cuts → Successful prediction-based cut generation
        #    • High fallback cuts → Frequent ML prediction failures
        #    • High switch cuts → Switching strategies actively triggered
        #
        # Cut Quality Hierarchy (typically):
        # Trial_cuts ≈ Selected_cuts > Switch_cuts > Fallback_cuts > ML_cuts
        # (exact > predicted, but ML cuts provide computational efficiency)
        #
        n_scen = self.n_subproblems
        self.num_cuts_mip_selected = {sid: 0 for sid in range(n_scen)}      # Phase 2: Exact cuts from selected scenarios at integer solutions
        self.num_cuts_rel_selected = {sid: 0 for sid in range(n_scen)}      # Phase 2: Exact cuts from selected scenarios at LP relaxations
        self.num_cuts_mip_ml = {sid: 0 for sid in range(n_scen)}            # Phase 2: ML-predicted cuts at integer solutions (all methods)
        self.num_cuts_rel_ml = {sid: 0 for sid in range(n_scen)}            # Phase 2: ML-predicted cuts at LP relaxations (all methods)
        self.num_cuts_mip_fallback = {sid: 0 for sid in range(n_scen)}      # Phase 2: Fallback cuts when ML fails at integer solutions
        self.num_cuts_rel_fallback = {sid: 0 for sid in range(n_scen)}      # Phase 2: Fallback cuts when ML fails at LP relaxations
        self.num_cuts_mip_trial = {sid: 0 for sid in range(n_scen)}         # Phase 1: Traditional Benders cuts at integer solutions
        self.num_cuts_rel_trial = {sid: 0 for sid in range(n_scen)}         # Phase 1: Traditional Benders cuts at LP relaxations
        
        # Additional cut counters for switching strategies
        self.num_cuts_mip_exact_switch = {sid: 0 for sid in range(n_scen)}
        self.num_cuts_rel_exact_switch = {sid: 0 for sid in range(n_scen)}

        # Tracking total nodes explored
        self.total_nodes_explored = 0
        
        # ===== TASK 1: Point signature tracking =====
        # Detect repeated first-stage solution points and track cut generation
        self.first_stage_signatures = {}  # {signature_hash: {'count': int, 'iterations': [], 'cuts_added': []}}
        self.total_unique_points = 0
        self.total_repeated_hits = 0
        self.repeated_hits_with_cuts = 0
        
        # ===== TASK 2: Root vs non-root callback tracking =====
        # Track callback invocations by type and location
        self.callback_counts = {
            'mipsol_root': 0,
            'mipsol_nonroot': 0,
            'mipnode_root': 0,
            'mipnode_nonroot': 0
        }
        
        # Solver-observed bound history; no historical decision snapshots.
        self.bound_trace = []  # Gurobi master bounds, observed online.
        self.gurobi_cut_statistics = GurobiCutStatistics()
        self._submitted_cut_counts = dict.fromkeys(("exact_lazy", "exact_user", "ml_lazy", "ml_user"), 0)
        self._last_bound_pair = None
        self.exclusive_profiler = ExclusiveProfiler()
        self._phase0_running = False
        self._callback_exception = None
        self.callback_wall_time_sec = 0.0

        # Timing information - 6-step pattern
        self.step_timings = {
            'step1_phase0_generation': None,      # Step 1: Phase 0 random solution generation
            'step2_phase0_cuts': None,            # Step 2: Phase 0 cut generation  
            'step3_phase1_iterations': None,      # Step 3: Phase 1 standard Benders
            'step4_feature_generation': None,     # Step 4: Feature generation from duals
            'step5_partitioning_training': None,  # Step 5: Partitioning and predictor training
            'step6_phase2_iterations': None       # Step 6: Phase 2 ML-augmented iterations
        }
        
        # Detailed timing (legacy, for backward compatibility)
        self.feature_gen_time_sec = None
        self.subproblem_selection_time_sec = None
        self.predictor_training_time = None
        self.KNN_closest_neighbor_time = None

        # Tracking predictions and fallback usage
        self.num_fallback_used = 0
        self.num_predictions = 0

        # Exact-subproblem accounting. Objective re-evaluation is separated so
        # it can be reported without being mistaken for Benders work.
        self.exact_subproblem_solves = 0
        self.exact_subproblem_time_sec = 0.0
        self.reevaluation_exact_subproblem_solves = 0
        self.reevaluation_exact_subproblem_time_sec = 0.0

        # Phase timing attributes  
        self.phase0_execution_time_sec = None
        self.phase0_solution_generation_time_sec = None
        
        # Phase 1 & 2 timing attributes
        self.phase1_start_time = None
        self.phase1_execution_time_sec = None
        self.phase2_start_time = None
        self.phase2_execution_time_sec = None

        # Switching strategy tracking
        self.num_switch_events = 0  # Total number of times switching logic was triggered
        self.switch_iterations = []  # List of iteration numbers when switching occurred
        self.cuts_from_unselected_this_iter = 0  # Track cuts from unselected in current iteration

        # Training budgets and weights (REG parameters removed - KNN only)
        # self.L_d, self.L_v, self.p_d, self.p_v no longer needed for KNN method

        # Dual storage for KNN training
        self.all_trial_duals = {}  # Storage for {(scenario_id, iteration): dual_dict} from Phase 1

    def _exact_profile_label(self, context):
        if context == "reevaluation":
            return "all_candidate_expost_reevaluation"
        if self.approach_category in {"Exact", "Random_Sampling"}:
            return "standard_or_random_exact_subproblem_solving"
        if self._phase0_running:
            return "phase0_exact_subproblem_solving"
        if self.iteration <= self.n_trials:
            return "phase1_exact_subproblem_solving"
        if context in {"fallback", "switch", "phase2_exact"}:
            return "phase2_fallback_exact_subproblem_solving"
        return "phase2_selected_exact_subproblem_solving"

    def _solve_dual_exact(self, sid, first_stage_values, context="algorithm"):
        """Solve one dual subproblem and account for its complete wall time."""
        start = time.perf_counter()
        try:
            args = (self.problem_type, self.data, sid, first_stage_values)
            if context == "reevaluation":
                # The enclosing postsolve timer owns this entire call. Do not
                # reactivate callback profiling and leak nested logging into
                # solver categories after optimize() has returned.
                result = solve_dual_subproblem(
                    *args, use_stochastic_capacity=self.use_stochastic_capacity
                )
            else:
                result = self.exclusive_profiler.call(
                    self._exact_profile_label(context), solve_dual_subproblem,
                    *args, use_stochastic_capacity=self.use_stochastic_capacity,
                )
        finally:
            elapsed = time.perf_counter() - start
            if context == "reevaluation":
                self.reevaluation_exact_subproblem_time_sec += elapsed
            else:
                self.exact_subproblem_time_sec += elapsed

        if context == "reevaluation":
            self.reevaluation_exact_subproblem_solves += 1
        else:
            self.exact_subproblem_solves += 1
        return result

    def get_parametric_diagnostics(self):
        """Return predictor diagnostics plus complete exact-solve accounting."""
        if self.dual_predictor is None:
            return {}
        diagnostics = self.dual_predictor.get_parametric_diagnostics()
        if not diagnostics:
            return {}
        diagnostics.update(
            {
                "algorithm_exact_subproblem_solves": self.exact_subproblem_solves,
                "algorithm_exact_subproblem_time_sec": self.exact_subproblem_time_sec,
                "reevaluation_exact_subproblem_solves": (
                    self.reevaluation_exact_subproblem_solves
                ),
                "reevaluation_exact_subproblem_time_sec": (
                    self.reevaluation_exact_subproblem_time_sec
                ),
            }
        )
        return diagnostics

    @property
    def n_subproblems(self):
        """
        Get the number of subproblems based on problem type.
        
        For SCFLP: Returns number of scenarios
        For SUMND: Returns number of scenarios
        For NE-SP: Returns number of OD pairs
        
        Returns:
            int: Number of subproblems
        """
        if self.problem_type in ["NESP", "NE-SP"]:
            return self.data.n_od_pairs
        else:  # SCFLP, SUMND, and others (scenario-based problems)
            return len(self.data.S)

    def execute_phase0(self, mod):
        """
        Execute Phase 0: Generate random feasible first-stage solutions and solve all subproblems.
        
        This method implements Phase 0 of the extended Learning-Augmented Benders Decomposition:
        1. Generate m_trials random feasible first-stage solutions using phase0_generator
        2. For each solution, solve all subproblems exactly to collect optimal duals
        3. Collect all optimality cuts from these duals
        4. Add all collected cuts to the master problem at once (before Phase 1 starts)
        5. Store Phase 0 duals for later combination with Phase 1 duals in feature generation
        
        Phase 0 strengthens the master problem before Phase 1 begins, providing:
        - Better LP relaxation bounds through additional cuts
        - More diverse training data for ML prediction in Phase 2
        - Improved exploration of the first-stage solution space
        
        Note: Cuts are collected during Phase 0 execution but only added to the master problem
        at the end, ensuring Phase 1 starts with a strengthened master problem.
        
        Args:
            mod (gurobi.Model): The master problem model to add cuts to
        """
        if self.m_trials <= 0 or self.phase0_executed:
            logger.info(f"Skipping Phase 0: m_trials={self.m_trials}, already_executed={self.phase0_executed}")
            return
            
        logger.info(f"Starting Phase 0: Generating {self.m_trials} random feasible solutions and solving all subproblems")
        
        start_time = time.time()
        
        try:
            # Generate Phase 0 random feasible solutions
            solution_generation_start = time.time()
            self.phase0_solutions = self.exclusive_profiler.call(
                "phase0_solution_generation", generate_phase0_solutions,

                problem_data=self.data,
                m_trials=self.m_trials,
                problem_class=self.problem_type
            )
            self.phase0_solution_generation_time_sec = time.time() - solution_generation_start
            
            logger.info(f"Generated {len(self.phase0_solutions)} Phase 0 solutions in {self.phase0_solution_generation_time_sec:.3f}s")
            
            # Collect all Phase 0 cuts before adding them to master problem
            phase0_cuts_to_add = []  # List of (duals, sid) tuples
            
            # Process each Phase 0 solution
            for phase0_iter, x_phase0 in enumerate(self.phase0_solutions):
                # logger.debug(f"Processing Phase 0 solution {phase0_iter + 1}/{len(self.phase0_solutions)}")
                
                # Convert numpy array to first-stage solution dictionary
                first_stage_sol = self._array_to_first_stage_dict(x_phase0)
                
                # Solve all subproblems for this Phase 0 solution
                phase0_duals_this_iter = {}
                cuts_added_this_iter = 0
                
                for sid in range(self.n_subproblems):
                    try:
                        # Solve dual subproblem exactly
                        result = self._solve_dual_exact(sid, first_stage_sol)
                        
                        if result is not None:
                            # Structure the dual solution
                            structured_duals = self.structure_duals(result)
                            
                            # Convert structured duals to numpy array (same as Phase 1)
                            if self.problem_type.upper() == "UFL":
                                vec = np.concatenate([
                                    [structured_duals['lambda']],
                                    [structured_duals['mu'][j] for j in self.data.D]
                                ])
                            elif self.problem_type.upper() == "UHL":
                                vec = np.concatenate([
                                    [structured_duals['v'][k] for k in self.data.K],
                                    [structured_duals['u'][l] for l in self.data.N]
                                ])
                            elif self.problem_type == "SCFLP":
                                vec = np.concatenate([
                                    [structured_duals['mu'][i] for i in self.data.I],  # Mu values for all customers
                                    [structured_duals['nu'][j] for j in self.data.J]   # Nu values for all facilities
                                ])
                            elif self.problem_type.upper() in ["SUMND", "SCMND"]:
                                # For SUMND: concatenate node-commodity duals and arc capacity duals
                                pi_vals = structured_duals['pi']
                                alpha_vals = structured_duals['alpha']
                                # Extract pi values in consistent order: all nodes for commodity 0, then all nodes for commodity 1, etc.
                                vec = np.concatenate([
                                    [pi_vals[(i, k)] for k in self.data.K for i in self.data.N],
                                    [alpha_vals[idx] for idx in range(len(self.data.A))]
                                ])
                            elif self.problem_type == "SSLP":
                                # keep the vector compact: only π-values
                                vec = np.array([structured_duals["pi"][i] for i in self.data.C])
                            else:
                                raise ValueError(f"Unsupported problem type for dual conversion: {self.problem_type}")
                            
                            # Store Phase 0 duals for later feature generation (trial_duals format)
                            # Use negative iteration numbers to distinguish Phase 0 from Phase 1
                            # Format: (trial_id, scenario_id) to match Phase 1 format
                            dual_key = (-(phase0_iter + 1), sid)
                            phase0_duals_this_iter[dual_key] = vec
                            
                            # Also store in all_trial_duals for training (using negative iteration numbers to distinguish from Phase 1)
                            if self.prediction_method in ['TREE', 'DUAL_CLUSTERING_WITH_DUAL']:
                                self.all_trial_duals[(-(phase0_iter + 1), sid)] = vec.copy()
                            
                            # Check if cut should be added (compare with current theta = 0 initially)
                            dual_obj_val = self.compute_dual_obj(structured_duals, sid, first_stage_sol)
                            
                            # Collect cut for later addition to master problem
                            # (theta starts at 0, so any positive dual objective adds value)
                            if dual_obj_val > 1e-6:  # Small tolerance for numerical precision
                                phase0_cuts_to_add.append((structured_duals, sid))
                                cuts_added_this_iter += 1
                                
                    except Exception as e:
                        logger.warning(f"Failed to solve Phase 0 subproblem {sid} for solution {phase0_iter}: {e}")
                        continue
                
                # Store duals from this Phase 0 iteration
                self.phase0_dual_vectors.update(phase0_duals_this_iter)
                
                # logger.debug(f"Phase 0 iteration {phase0_iter + 1}: Collected {cuts_added_this_iter} cuts")
            
            # Add all collected Phase 0 cuts to the master problem
            logger.info(f"Adding {len(phase0_cuts_to_add)} Phase 0 cuts to master problem")
            total_cuts_added = 0
            
            for duals, sid in phase0_cuts_to_add:
                try:
                    self.add_optimality_cut(mod, duals, sid, integer_solution=True)
                    total_cuts_added += 1
                except Exception as e:
                    logger.warning(f"Failed to add Phase 0 cut for scenario {sid}: {e}")
                    
            self.phase0_cuts_added = total_cuts_added  # Update with actual cuts added
            logger.info(f"Successfully added {total_cuts_added} Phase 0 cuts to master problem")
            
            # Mark Phase 0 as completed
            self.phase0_executed = True
            
            execution_time = time.time() - start_time
            self.phase0_execution_time_sec = execution_time  # Store for logging/results
            
            logger.info(f"Phase 0 completed: {len(self.phase0_solutions)} solutions, "
                       f"{self.phase0_cuts_added} cuts generated and added to master problem, "
                       f"{len(self.phase0_dual_vectors)} dual vectors stored, "
                       f"execution time: {execution_time:.2f}s")
                       
        except Exception as e:
            logger.error(f"Phase 0 execution failed: {e}")
            # Continue without Phase 0 if it fails
            self.phase0_executed = True  # Prevent retry
            self.phase0_solutions = []
            self.phase0_dual_vectors = {}

    def _array_to_first_stage_dict(self, x_array):
        """
        Convert numpy array first-stage solution to dictionary format expected by subproblem solver.
        
        Args:
            x_array (np.ndarray): Binary array indicating which facilities/nodes are open
            
        Returns:
            dict: First-stage solution dictionary compatible with solve_dual_subproblem
        """
        first_stage_dict = {}
        
        if self.problem_type in ["UFL", "SCFLP"]:
            # Facility location problems: map to facility indices
            for j, value in enumerate(x_array):
                first_stage_dict[j] = float(value)
                
        elif self.problem_type == "UHL":
            # Hub location problems: map to node indices  
            for k, value in enumerate(x_array):
                first_stage_dict[k] = float(value)
                
        else:
            # Generic mapping for other problem types
            for i, value in enumerate(x_array):
                first_stage_dict[i] = float(value)
                
        return first_stage_dict

    def __call__(self, mod, where):
        """
        Main Gurobi callback entry point for learning-augmented Benders decomposition.
        
        This method makes the Callback class callable, allowing it to be used as a Gurobi callback
        function. It intercepts the optimization process at specific points during branch-and-bound
        to add Benders optimality cuts. The method determines the type of solution (integer vs. 
        fractional) and delegates the actual cut generation logic to handle_callback().
        
        Gurobi automatically calls this method during optimization when:
        1. An integer-feasible solution is found (MIPSOL)
        2. An LP relaxation is solved optimally (MIPNODE) - if enabled
        
        Args:
            mod (gurobi.Model): The Gurobi model object being optimized. Provides access to:
                - Solution values via mod.cbGetSolution() or mod.cbGetNodeRel()
                - Cut addition via mod.cbLazy() or mod.cbCut()
                - Callback status information via mod.cbGet()
            where (int): Gurobi callback location constant indicating when callback was triggered:
                - GRB.Callback.MIPSOL: At integer-feasible solutions
                - GRB.Callback.MIPNODE: At LP relaxation nodes (if use_prediction='continuous' or 'both')
                
        Callback behavior:
            - **MIPSOL (Always active)**: 
                * Triggered when Gurobi finds integer solutions
                * Calls handle_callback(mod, integer_solution=True)
                * Adds cuts via mod.cbLazy() (lazy constraints)
                * Used for strengthening integer solutions
                
            - **MIPNODE (Optional)**: 
                * Only active if use_prediction='continuous' or 'both'
                * Triggered at LP relaxation solutions with optimal status
                * Calls handle_callback(mod, integer_solution=False) 
                * Adds cuts via mod.cbCut() (cutting planes)
                * Used for strengthening LP relaxations early
                * Tracks branch-and-bound node exploration statistics
                
        Relationship to other methods:
            - **handle_callback()**: Receives the actual cut generation logic delegation
            - **_handle_fallback()**: Called by handle_callback() when ML prediction fails
            - **add_optimality_cut()**: Called by handle_callback() to add cuts to the model
            
        Flow control:
            1. __call__() determines solution type and context
            2. handle_callback() implements the two-phase Benders logic
            3. _handle_fallback() provides backup when ML fails
            4. All methods ultimately call add_optimality_cut() to modify the model
            
        Performance considerations:
            - MIPSOL cuts are essential for correctness
            - MIPNODE cuts can accelerate convergence but add computational overhead
            - Node exploration tracking helps evaluate algorithm efficiency
            
        Note: This method implements the Gurobi callback interface, making the class
              compatible with mod.optimize(callback_instance).
        """
        cb_begin = time.perf_counter()
        try:
            with self.exclusive_profiler.measure("callback_other_logic"):
                self._dispatch_callback(mod, where)
        except Exception as exc:
            # Gurobi may report and swallow Python callback exceptions. Retain
            # the failure, terminate, and re-raise after optimize() returns.
            if self._callback_exception is None:
                self._callback_exception = exc
            mod.terminate()
        finally:
            self.callback_wall_time_sec += time.perf_counter() - cb_begin

    def raise_if_failed(self):
        if self._callback_exception is not None:
            raise RuntimeError("Benders callback failed; this run is invalid") from self._callback_exception

    def _dispatch_callback(self, mod, where):
        if where == GRB.Callback.MESSAGE:
            profile_call("diagnostics_logging_file_io", self.gurobi_cut_statistics.feed_message,
                         mod.cbGet(GRB.Callback.MSG_STRING))
        elif where == GRB.Callback.MIP:
            self._record_solver_bounds(mod, where)
        elif where == GRB.Callback.MIPSOL:
            # TASK 2: Track root vs non-root MIPSOL callbacks
            current_node = mod.cbGet(GRB.Callback.MIPSOL_NODCNT)
            is_root = (current_node == 0)
            if is_root:
                self.callback_counts['mipsol_root'] += 1
            else:
                self.callback_counts['mipsol_nonroot'] += 1
            
            profile_call("diagnostics_logging_file_io", print, f"Callback triggered at integer solution (MIPSOL), iteration {self.iteration + 1}, node={current_node}")
            self.iteration += 1
            self._record_solver_bounds(mod, where, event="solver_iteration_start")
            self.handle_callback(mod, integer_solution=True) #! integer_solution=True will be used to count cuts for integer solutions inside true callback
            self._record_solver_bounds(mod, where, event="solver_iteration_end")
        elif where == GRB.Callback.MIPNODE:
            status = mod.cbGet(GRB.Callback.MIPNODE_STATUS)
            if self._should_handle_relaxation_callbacks() and status == GRB.OPTIMAL:
                current = mod.cbGet(GRB.Callback.MIPNODE_NODCNT)
                
                # TASK 2: Track root vs non-root MIPNODE callbacks
                is_root = (current == 0)
                if is_root:
                    self.callback_counts['mipnode_root'] += 1
                else:
                    self.callback_counts['mipnode_nonroot'] += 1
                
                self.total_nodes_explored = max(self.total_nodes_explored, current) #? What is this used for?
                profile_call("diagnostics_logging_file_io", print, f"Callback triggered at LP relaxation (MIPNODE), iteration {self.iteration + 1}, node={current}")
                self.iteration += 1
                self._record_solver_bounds(mod, where, event="solver_iteration_start")
                self.handle_callback(mod, integer_solution=False) #! integer_solution=False will be used to count cuts for relaxed solutions inside false callback
                self._record_solver_bounds(mod, where, event="solver_iteration_end")
            else:
                self._record_solver_bounds(mod, where)

    @profiled("diagnostics_logging_file_io")
    def _record_solver_bounds(self, mod, where=GRB.Callback.MIP, *, event=None):
        """Record changes without throttling, plus every handled iteration boundary.

        Query codes only in their valid callback context. MIPSOL_OBJ is a
        candidate, not an accepted incumbent, and is deliberately never used.
        End records describe the bounds visible before Gurobi processes cuts.
        """
        context, lb_code, ub_code, node_code = {
            GRB.Callback.MIP: ("MIP", GRB.Callback.MIP_OBJBND, GRB.Callback.MIP_OBJBST, GRB.Callback.MIP_NODCNT),
            GRB.Callback.MIPSOL: ("MIPSOL", GRB.Callback.MIPSOL_OBJBND, GRB.Callback.MIPSOL_OBJBST, GRB.Callback.MIPSOL_NODCNT),
            GRB.Callback.MIPNODE: ("MIPNODE", GRB.Callback.MIPNODE_OBJBND, GRB.Callback.MIPNODE_OBJBST, GRB.Callback.MIPNODE_NODCNT),
        }[where]
        elapsed = float(mod.cbGet(GRB.Callback.RUNTIME))
        pair = (_finite(mod.cbGet(lb_code)), _finite(mod.cbGet(ub_code)))
        bound_changed = pair != self._last_bound_pair
        cut_count, count_changed = None, False
        if where == GRB.Callback.MIP:
            cut_count = mod.cbGet(GRB.Callback.MIP_CUTCNT)
            count_changed = self.gurobi_cut_statistics.observe(cut_count, elapsed)
            cut_count = int(cut_count)
        if event is None and not (bound_changed or count_changed):
            return
        reason = event or ("solver_initial" if self._last_bound_pair is None else
                           "solver_bound_change" if bound_changed else "solver_cut_change")
        self.bound_trace.append({
            'elapsed_seconds': elapsed,
            'event': reason,
            'iteration': self.iteration,
            'node_count': float(mod.cbGet(node_code)),
            'lower_bound': pair[0], 'master_upper_bound': pair[1],
            'callback_context': context, 'bound_changed': int(bound_changed),
            'gurobi_mip_cut_count': cut_count,
            'gurobi_cut_count_changed': int(count_changed),
            **self.cut_submission_totals(),
        })
        self._last_bound_pair = pair

    def cut_submission_totals(self):
        counts = self._submitted_cut_counts
        return {
            'submitted_exact_cuts': counts['exact_lazy'] + counts['exact_user'],
            'submitted_ml_cuts': counts['ml_lazy'] + counts['ml_user'],
            'submitted_lazy_cuts': counts['exact_lazy'] + counts['ml_lazy'],
            'submitted_user_cuts': counts['exact_user'] + counts['ml_user'],
        }

    def append_terminal_bounds(self, model):
        """Capture the final global lower bound after optimize() returns."""
        self.gurobi_cut_statistics.finalize()
        try:
            pair = (_finite(model.ObjBound), _finite(model.ObjVal) if model.SolCount > 0 else None)
            self.bound_trace.append({
                'elapsed_seconds': float(model.Runtime),
                'event': 'solver_final',
                'iteration': self.iteration,
                'node_count': float(model.NodeCount),
                'lower_bound': pair[0], 'master_upper_bound': pair[1],
                'callback_context': 'TERMINAL',
                'bound_changed': int(pair != self._last_bound_pair),
                # MIP_CUTCNT is unavailable here; never backfill a stale value.
                'gurobi_mip_cut_count': None, 'gurobi_cut_count_changed': 0,
                **self.cut_submission_totals(),
            })
            self._last_bound_pair = pair
        except (AttributeError, ValueError):
            pass

    def solver_timing_buckets(self, solver_runtime):
        """Disjoint solver-time buckets; phase times elsewhere may overlap."""
        whole = float(solver_runtime or 0)
        callback = self.callback_wall_time_sec
        exact = self.exact_subproblem_time_sec
        if not all(np.isfinite(v) and v >= 0 for v in (whole, callback, exact)):
            raise RuntimeError("Invalid solver/callback wall time")
        if callback > whole + 1e-6 or exact > callback + 1e-6:
            raise RuntimeError("Solver/callback/exact intervals are not nested")
        return {
            'master_solver_internal_sec': max(0.0, whole - callback),
            'callback_exact_subproblem_sec': exact,
            'callback_other_sec': max(0.0, callback - exact),
        }

    def solver_component_breakdown(self, solver_runtime):
        """Additive measured components + callback residual + Gurobi-internal time."""
        buckets = self.solver_timing_buckets(solver_runtime)
        total_callback = (
            buckets["callback_exact_subproblem_sec"] +
            buckets["callback_other_sec"]
        )
        profiled = {
            key: value for key, value in self.exclusive_profiler.snapshot().items()
            if key != "all_candidate_expost_reevaluation"
        }
        used = sum(profiled.values())
        # Never silently rescale measured category times. Unexpected overlap
        # invalidates a run's breakdown and must be fixed before production.
        overrun = used - total_callback
        tolerance = max(0.05, 1e-4 * max(1.0, float(solver_runtime or 0.0)))
        if overrun > tolerance:
            raise RuntimeError(
                f"Exclusive timer overlap {overrun:.6f}s exceeds {tolerance:.6f}s"
            )
        # Sub-millisecond clock differences come from timing two different
        # interfaces. Assign only the small difference to callback-other.
        profiled["callback_other_logic"] = (
            profiled.get("callback_other_logic", 0.0)
            + (total_callback - used)
        )
        if profiled["callback_other_logic"] < -1e-9:
            raise RuntimeError(
                "Exclusive callback category became negative after reconciliation"
            )
        profiled["callback_other_logic"] = max(
            0.0, profiled["callback_other_logic"]
        )
        profiled["master_solver_internal"] = buckets["master_solver_internal_sec"]
        # Guaranteed 33-key schema, with zero for non-applicable operations.
        unknown = sorted(set(profiled).difference(TIMING_CATEGORIES))
        if unknown:
            raise RuntimeError("Unclassified timing labels: " + ", ".join(unknown))
        return {key: float(profiled.get(key, 0.0)) for key in TIMING_CATEGORIES}

    def timing_component_counts(self):
        # Candidate re-evaluation is a postsolve category, not inside Gurobi.
        counts = self.exclusive_profiler.count_schema()
        counts["master_solver_internal"] = 1  # One optimize interval residual.
        return counts

    def _should_handle_relaxation_callbacks(self):
        """
        Determine if LP relaxation callbacks should be handled.
        
        LP relaxation callbacks are registered for methods that generate cuts
        at both integer and LP relaxation points:
        - int_and_LP_ML_for_int
        - int_and_LP_ML_for_LP
        - int_and_LP_ML_for_none
        - int_and_LP_ML_for_both
        
        Methods that only use integer callbacks:
        - int_ML
        - int_exact
        
        Returns:
            bool: True if LP relaxation callbacks should be registered
        """
        return should_register_lp_callbacks(self.use_prediction)
    
    def _uses_ml_in_phase2(self):
        """
        Check if the current method uses ML prediction at all in Phase 2.
        Used to determine if Phase 0 and feature generation should be executed.
        
        Returns:
            bool: True if method uses ML in Phase 2, False if always exact
        """
        return uses_ml_in_phase2(self.use_prediction)
    
    def _should_use_ml_in_phase2(self, integer_solution):
        """
        Determine if ML should be used for the current context in Phase 2.
        
        Phase 1 always uses exact solving regardless of this function.
        
        Args:
            integer_solution (bool): True for integer solutions, False for LP relaxations
            
        Returns:
            bool: True if ML should be used in Phase 2 for this context
        """
        normalized = normalize_use_prediction(self.use_prediction)
        
        # Check method-specific Phase 2 behavior
        if normalized == "int_and_LP_ML_for_int":
            return integer_solution  # ML at integer, exact at LP
        elif normalized == "int_and_LP_ML_for_LP":
            return not integer_solution  # exact at integer, ML at LP
        elif normalized == "int_and_LP_ML_for_relax":
            return not integer_solution  # Legacy name - same as int_and_LP_ML_for_LP
        elif normalized == "int_and_LP_ML_for_none":
            return False  # Always exact (standard Benders with LP cuts)
        elif normalized == "int_and_LP_ML_for_both":
            return True  # Always ML
        elif normalized == "int_ML":
            return integer_solution  # ML at integer only (LP not registered anyway)
        elif normalized == "int_exact":
            return False  # Always exact (standard Benders)
        # Legacy formats
        elif normalized == "int_and_LP_ML_for_selected":
            return integer_solution  # Legacy: ML at integer for selected scenarios
        elif normalized == "int_only_ML_for_integer":
            return integer_solution  # ML at integer only
        elif "ML_for" in normalized:
            return True  # Default to using ML for any unrecognized ML_for format
        else:
            raise ValueError(f"Unknown use_prediction method: {normalized}")

    def handle_callback(self, mod, integer_solution=True):
        """
        Handles the callback logic for both integer and relaxed solutions.
        integer_solution: bool
            If True, processes integer solutions (MIPSOL) and adds cuts.
            If False, processes relaxed solutions (MIPNODE) and adds cuts.
        """
        # Use appropriate solution getters
        get_val = mod.cbGetSolution if integer_solution else mod.cbGetNodeRel
        first_stage_sol = get_val(self.first_stage_values) # !Get current first-stage solution values passed to callback
        theta_val = get_val(self.theta) # !Get current theta values passed to callback
        n_scen = self.n_subproblems # !Number of subproblems (scenarios for SCFLP, OD pairs for NE-SP)
        
        # Log current master problem solution when integer solution is found in Benders iterations.
        # Disabled for SUMND to keep runtime logs focused on data-loading only.
        if integer_solution and self.problem_type.upper() not in ["SUMND", "SCMND"]:
            try:
                # Compute current objective partial sum from theta values
                current_theta_sum = sum(theta_val.values()) if isinstance(theta_val, dict) else sum(theta_val)
            except:
                current_theta_sum = None
            
            # Format first-stage solution for logging (show nonzero variables)
            if isinstance(first_stage_sol, dict):
                nonzero_vars = {k: v for k, v in first_stage_sol.items() if v > 1e-6}
            else:
                try:
                    nonzero_vars = {i: v for i, v in enumerate(first_stage_sol) if v > 1e-6}
                except:
                    nonzero_vars = first_stage_sol
            
            # logger.info("=" * 80)
            # logger.info(f"BENDERS ITERATION {self.iteration} - INTEGER SOLUTION FOUND")
            # logger.info(f"Current recourse estimate sum: {current_theta_sum}")
            # logger.info(f"First-stage variables (nonzero, count={len(nonzero_vars)}): {nonzero_vars}")
            # logger.info("=" * 80)
        
        # Track repeated callback point signatures for cut diagnostics only.
        # BUG FIX (Feb 23, 2026): Calculate cuts_before for both integer and relaxation solutions
        if integer_solution:
            cuts_before = sum(self.num_cuts_mip_selected.values()) + sum(self.num_cuts_mip_trial.values())
        else:
            cuts_before = sum(self.num_cuts_rel_selected.values()) + sum(self.num_cuts_rel_trial.values())
        
        
        # Create a mapping of scenario indices to theta keys for easy access
        # This must be done BEFORE Phase 0 execution since Phase 0 adds cuts that need theta_keys
        self.theta_keys = {}
        for sid in range(n_scen):
            self.theta_keys[sid] = self.scenario_key(sid) # !Map scenario index to theta key. Used because in some problem classes like UHL, scenario is represented with a pair (i,j)instead of just a single index. So, we need to map it to the key used in theta_vars to deal with single index
        
        # DIAGNOSTIC: Verify theta_key consistency
        key_mismatches = []
        for sid in range(n_scen):
            key1 = self.scenario_key(sid)
            key2 = self.theta_keys[sid]
            if key1 != key2:
                key_mismatches.append((sid, key1, key2))
        if key_mismatches:
            logger.error(f"THETA_KEY MISMATCH DETECTED: {len(key_mismatches)} mismatches")
            for sid, k1, k2 in key_mismatches[:5]:
                logger.error(f"  scenario {sid}: scenario_key()={k1}, theta_keys={k2}")
        else:
            pass
            # logger.info(f"Theta key consistency verified for {n_scen} scenarios")

        # PHASE 0: Execute warm-up random solutions before any Benders iterations
        # Use step-based system to determine if Phase 0 should execute
        # should_execute_step(method_config, 1) checks both method type AND m_trials > 0 internally
        if not self.phase0_executed and self.m_trials > 0 and self.execute_step[1]:
            logger.info(f"Executing Phase 0 for method '{self.ml_method_type}' before starting Benders iterations")
            self._phase0_running = True
            try:
                self.execute_phase0(mod)
            finally:
                self._phase0_running = False
        
        # ===== EXACT METHODS: Standard Benders Decomposition =====
        # For "Exact" category (int_exact, int_and_LP_ML_for_none):
        # - No Phase 0, No Phase 2, No ML, No feature generation
        # - Solve ALL subproblems exactly at EVERY iteration (standard Benders)
        # - No iteration cap (runs until Gurobi finds optimal solution)
        # - This is the baseline against which ML methods are compared
        if self.approach_category == "Exact":
            context = "integer" if integer_solution else "LP relaxation"
            profile_call("diagnostics_logging_file_io", print, f"Standard Benders: Exact solving at {context}, iteration {self.iteration}")
            
            # DIAGNOSTIC: Track cut statistics for comparison
            cuts_added_this_iter = 0
            violations_detected = 0
            max_violation = 0.0
            
            for sid in range(n_scen):
                res = self._solve_dual_exact(sid, first_stage_sol)
                duals = self.structure_duals(res)
                obj = self.compute_dual_obj(duals, sid, first_stage_sol)
                
                # CONSISTENCY FIX: Add epsilon tolerance (was missing, causing inconsistency with ML methods)
                theta_key = self.scenario_key(sid)
                violation = obj - theta_val[theta_key]
                
                # BUG FIX (Feb 22, 2026): Use self.EPS tolerance to avoid counting numerical noise as violations
                if violation > self.EPS:
                    violations_detected += 1
                    max_violation = max(max_violation, violation)
                
                if obj > theta_val[theta_key] + self.EPS:
                    self.add_optimality_cut(mod, duals, sid, integer_solution)
                    cuts_added_this_iter += 1
                    if integer_solution:
                        self.num_cuts_mip_trial[sid] += 1
                    else:
                        self.num_cuts_rel_trial[sid] += 1
            
            # DIAGNOSTIC: Log cut generation statistics (commented - too verbose per iteration)
            # logger.info(f"Exact method iter {self.iteration}: {cuts_added_this_iter} cuts added, "
            #            f"{violations_detected} violations detected, max violation: {max_violation:.6e}")
            
            # TASK 1: Track point signature for Exact methods
            self._track_point_signature(first_stage_sol, cuts_added_this_iter, self.iteration)
            
            # Note: iteration is already incremented in __call__ before handle_callback
            return  # Done with this callback - continue Benders until convergence
        
        # ===== ML METHODS: Skip Phase 1 for categories that don't need training data =====
        # Categories like Uncertainty_Similarity, Covariate_Similarity, Random_Sampling
        # skip directly to Phase 2 as they don't need dual-based training
        # BUG FIX (Feb 22, 2026): Do NOT artificially bump iteration counter when Phase 1 is skipped
        # The iteration counter should start at 1 (after first callback) and proceed naturally
        # The in_phase1 check below will automatically evaluate to False when Phase 1 is skipped
        # BUG FIX (Feb 22, 2026): Add guard flag to log "Skipping Phase 1" message only once
        if not should_execute_phase1(self.approach_category) and not self.phase1_skip_logged:
            logger.info(f"{self.approach_category} category: Skipping Phase 1, going directly to Phase 2")
            self.phase1_skip_logged = True
        
        # Check if this method uses ML in Phase 2
        uses_ml = self._uses_ml_in_phase2()
        
        # ===== PHASE 1: Collect training data for ML methods =====
        # For ML methods: Phase 1 lasts for n_trials iterations, then transitions to Phase 2
        # Use category system to determine Phase 1 execution
        in_phase1 = (uses_ml and self.iteration <= self.n_trials)
        
        if in_phase1 and should_execute_phase1(self.approach_category):
            # Start timing Phase 1 if this is the first iteration
            if self.iteration == 1 and self.phase1_start_time is None:
                self.phase1_start_time = time.time()
            
            context = "integer" if integer_solution else "LP relaxation"
            phase_label = f"Phase 1 (iter {self.iteration})" if uses_ml else "Standard Benders"
            profile_call("diagnostics_logging_file_io", print, f"{phase_label}: Exact solving at {context}, iteration {self.iteration}")
            if uses_ml:
                self.first_stage_by_iteration[self.iteration] = {
                    key: float(value) for key, value in first_stage_sol.items()
                }
            
            for sid in range(n_scen):
                res = self._solve_dual_exact(sid, first_stage_sol)
                
                duals = self.structure_duals(res)
                obj = self.compute_dual_obj(duals, sid, first_stage_sol)
                if obj > theta_val[self.scenario_key(sid)] + self.EPS:
                    self.add_optimality_cut(mod, duals, sid, integer_solution)
                    if integer_solution:
                        self.num_cuts_mip_trial[sid] += 1
                    else:
                        self.num_cuts_rel_trial[sid] += 1
                
                # Store dual vectors ONLY for ML methods (needed for training)
                # Skip storage for non-ML methods to save memory and time
                if duals is not None and uses_ml:
                    # Store ALL duals during Phase 1 training for complete training dataset
                    self.all_trial_duals[(sid, self.iteration)] = duals.copy()
                    
                    if self.problem_type == 'UFL':
                        vec = np.concatenate([
                            [duals['lambda']],
                            [duals['mu'][j] for j in self.data.F]
                        ])
                    elif self.problem_type == "UHL":
                        vec = np.concatenate([
                            [duals['v'][k] for k in self.data.N],
                            [duals['u'][l] for l in self.data.N]
                        ])
                    elif self.problem_type == "SCFLP":
                        vec = np.concatenate([
                            [duals['mu'][i] for i in self.data.I],
                            [duals['nu'][j] for j in self.data.J]
                        ])
                    elif self.problem_type == "SSLP":
                        vec = np.array([duals["pi"][i] for i in self.data.C])
                    elif self.problem_type.upper() in ["NESP", "NE-SP"]:
                        # For NE-SP: concatenate node potentials and arc multipliers
                        pi_vals = duals['pi']
                        alpha_vals = duals['alpha']
                        # pi_vals and alpha_vals are dictionaries, so we need to extract values
                        vec = np.concatenate([
                            [pi_vals[node] for node in sorted(pi_vals.keys())],
                            [alpha_vals[idx] for idx in sorted(alpha_vals.keys())]
                        ])
                    elif self.problem_type.upper() in ["SUMND", "SCMND"]:
                        # For SUMND: concatenate node-commodity duals and arc capacity duals
                        pi_vals = duals['pi']
                        alpha_vals = duals['alpha']
                        # Extract pi values in consistent order: all nodes for commodity 0, then all nodes for commodity 1, etc.
                        vec = np.concatenate([
                            [pi_vals[(i, k)] for k in self.data.K for i in self.data.N],
                            [alpha_vals[idx] for idx in range(len(self.data.A))]
                        ])

                    self.dual_vectors[(self.iteration-1, sid)] = vec
            
            # Store first-stage solutions ONLY for ML methods (needed for feature generation)
            if uses_ml:
                if self.problem_type=='UFL':
                    xv = [first_stage_sol[int(j)] for j in self.data.F]
                    self.x_trials.append(np.array(xv))
                elif self.problem_type=='UHL':
                    n = len(self.data.N)
                    mat = np.array([ first_stage_sol[i,k] for i in self.data.N for k in self.data.N ])
                    self.x_trials.append(mat)
                elif self.problem_type == "SCFLP":
                    xv = [first_stage_sol[int(j)] for j in self.data.J]
                    self.x_trials.append(np.array(xv))
                elif self.problem_type.upper() in ["SUMND", "SCMND"]:
                    # For SUMND: first-stage variables are arc activation decisions
                    xv = [first_stage_sol[idx] for idx in range(len(self.data.A))]
                    self.x_trials.append(np.array(xv))
                elif self.problem_type == "SSLP":
                    xv = [first_stage_sol[int(j)] for j in self.data.F]
                    self.x_trials.append(np.array(xv))
            
            # For non-ML methods, continue with Phase 1 indefinitely (no transition)
            if not uses_ml:
                # TASK 1: Track point signature before returning
                cuts_after = sum(self.num_cuts_mip_selected.values()) + sum(self.num_cuts_mip_trial.values()) if integer_solution else sum(self.num_cuts_rel_selected.values()) + sum(self.num_cuts_rel_trial.values())
                cuts_added = cuts_after - cuts_before
                self._track_point_signature(first_stage_sol, cuts_added, self.iteration)
                return
            
            # For ML methods, return after Phase 1 iterations
            # TASK 1: Track point signature before returning
            cuts_after = sum(self.num_cuts_mip_selected.values()) + sum(self.num_cuts_mip_trial.values()) if integer_solution else sum(self.num_cuts_rel_selected.values()) + sum(self.num_cuts_rel_trial.values())
            cuts_added = cuts_after - cuts_before
            self._track_point_signature(first_stage_sol, cuts_added, self.iteration)
            return

        # PHASE 2: ML-aided Benders (iteration > n_trials)
        # Only reached by ML methods; non-ML methods continue in Phase 1 indefinitely
        # Features and training done ONCE at the transition from Phase 1 to Phase 2
        if not self.features_computed:
            # End timing Phase 1 and start timing Phase 2
            if self.phase1_start_time is not None and self.phase1_execution_time_sec is None:
                self.phase1_execution_time_sec = time.time() - self.phase1_start_time
                
            # Start timing Phase 2
            if  self.phase2_start_time is None:
                self.phase2_start_time = time.time()
            
            # BUG FIX (Feb 22, 2026): Clarify messaging for methods that skip Phase 1
            if should_execute_phase1(self.approach_category):
                profile_call("diagnostics_logging_file_io", print, f"Transition to Phase 2: Computing features and training predictor for category '{self.approach_category}', iteration {self.iteration}, integer_solution={integer_solution}")
            else:
                profile_call("diagnostics_logging_file_io", print, f"Phase 2: Computing features and training predictor for category '{self.approach_category}' (Phase 1 skipped), iteration {self.iteration}, integer_solution={integer_solution}")
            ts = time.time()
            
            # Handle feature generation based on approach category (cleaner than method-specific checks)
            if uses_uncertainty_vectors(self.approach_category):
                # For uncertainty-based methods, use demand vectors as features
                profile_call("diagnostics_logging_file_io", print, f"{self.approach_category}: Using demand vectors as uncertainty vectors")
                if self.problem_type == "SCFLP":
                    # Create uncertainty vectors from scenario demand data
                    self.feature_vectors = {}
                    for sid in range(self.n_subproblems):
                        # Use scenario demand vectors as uncertainty vectors
                        self.feature_vectors[sid] = self.data.scenarios[sid].copy()
                    profile_call("diagnostics_logging_file_io", print, f"Created uncertainty vectors for {len(self.feature_vectors)} scenarios")
                elif self.problem_type in ["NESP", "NE-SP"]:
                    # For NE-SP, uncertainty vectors could be OD-specific features
                    # (e.g., demand values, origin/destination characteristics)
                    self.feature_vectors = {}
                    for m_idx in range(self.n_subproblems):
                        # Use OD pair demand as uncertainty vector
                        self.feature_vectors[m_idx] = np.array([self.data.demands[m_idx]])
                    profile_call("diagnostics_logging_file_io", print, f"Created uncertainty vectors for {len(self.feature_vectors)} OD pairs")
                elif self.problem_type.upper() in ["SUMND", "SCMND"]:
                    # For SUMND/SCMND, use demand scenarios as uncertainty vectors
                    self.feature_vectors = {}
                    for sid in range(self.n_subproblems):
                        # Use demand scenario vector as uncertainty vector
                        if hasattr(self.data, 'demand_scenarios') and self.data.demand_scenarios is not None:
                            self.feature_vectors[sid] = np.array([self.data.demand_scenarios[sid][k] for k in self.data.K])
                        else:
                            # Fallback: create simple vector from number of commodities
                            self.feature_vectors[sid] = np.ones(len(self.data.K))
                    profile_call("diagnostics_logging_file_io", print, f"Created uncertainty vectors for {len(self.feature_vectors)} SUMND/SCMND scenarios")
                else:
                    raise NotImplementedError(f"{self.approach_category} not implemented for {self.problem_type}")
                self.feature_gen_time_sec = time.time() - ts
            
            elif uses_input_covariates(self.approach_category):
                # For covariate-based methods, use covariate vectors from input data
                profile_call("diagnostics_logging_file_io", print, f"{self.approach_category}: Using covariate vectors for feature generation")
                if self.problem_type == "SCFLP":
                    if self.data.covariate_vectors is not None:
                        # Create feature vectors from covariate data
                        self.feature_vectors = {}
                        for sid in range(len(self.data.covariate_vectors)):
                            # Use covariate vectors as features
                            self.feature_vectors[sid] = self.data.covariate_vectors[sid].copy()
                        profile_call("diagnostics_logging_file_io", print, f"Created covariate vectors for {len(self.feature_vectors)} SCFLP scenarios")
                    else:
                        raise ValueError(f"{self.approach_category} requires SCFLP data with covariate_vectors")
                elif self.problem_type.upper() in ["SUMND", "SCMND"]:
                    if hasattr(self.data, 'covariate_vectors') and self.data.covariate_vectors is not None:
                        # Create feature vectors from covariate data
                        self.feature_vectors = {}
                        for sid in range(len(self.data.covariate_vectors)):
                            # Use covariate vectors as features
                            self.feature_vectors[sid] = self.data.covariate_vectors[sid].copy()
                        profile_call("diagnostics_logging_file_io", print, f"Created covariate vectors for {len(self.feature_vectors)} SUMND/SCMND scenarios")
                    else:
                        # Fallback: use demand scenario data as covariates
                        self.feature_vectors = {}
                        for sid in range(self.n_subproblems):
                            if hasattr(self.data, 'demand_scenarios') and self.data.demand_scenarios is not None:
                                self.feature_vectors[sid] = np.array([self.data.demand_scenarios[sid][k] for k in self.data.K])
                            else:
                                self.feature_vectors[sid] = np.ones(len(self.data.K))
                        profile_call("diagnostics_logging_file_io", print, f"Created fallback covariate vectors (from demand) for {len(self.feature_vectors)} SUMND/SCMND scenarios")
                else:
                    raise NotImplementedError(f"{self.approach_category} not implemented for {self.problem_type}")
                self.feature_gen_time_sec = time.time() - ts
            
            elif uses_random_sampling(self.approach_category):
                # For random sampling methods, skip feature generation entirely
                profile_call("diagnostics_logging_file_io", print, f"{self.approach_category}: Skipping feature generation - no features needed")
                self.feature_vectors = {}  # Empty feature vectors since we don't use them
                # Create dummy feature vectors for compatibility with unified training
                for sid in range(self.n_subproblems):
                    self.feature_vectors[sid] = np.array([0.0])  # Dummy feature vector
                profile_call("diagnostics_logging_file_io", print, f"Created dummy feature vectors for {len(self.feature_vectors)} subproblems")
                self.feature_gen_time_sec = time.time() - ts
            
            elif uses_direct_dual_vectors(self.approach_category):
                # For methods using dual vectors directly without feature transformation
                profile_call("diagnostics_logging_file_io", print, f"{self.approach_category}: Using dual vectors directly for clustering")
                
                # Don't generate features - use raw dual vectors
                # Dual vectors are already stored in self.all_trial_duals from Phase 0/1
                # Just prepare metadata for compatibility with unified training interface
                
                # Create empty feature_vectors to indicate we're using duals directly
                self.feature_vectors = {}  # Empty - not using feature-based approach
                
                # Store dual vector count for logging
                total_duals = len(self.phase0_dual_vectors) + len(self.dual_vectors)
                profile_call("diagnostics_logging_file_io", print, f"Prepared {total_duals} dual vectors for direct clustering (no feature transformation)")
                profile_call("diagnostics_logging_file_io", print, f"  Phase 0 duals: {len(self.phase0_dual_vectors)}")
                profile_call("diagnostics_logging_file_io", print, f"  Phase 1 duals: {len(self.dual_vectors)}")
                
                # all_trial_duals already contains the structured dual vectors from Phase 0/1
                # These will be passed directly to training_and_selection.py
                
                self.feature_gen_time_sec = time.time() - ts
            
            elif should_generate_features_from_duals(self.approach_category):
                # Standard feature generation from dual vectors (ML_Augmented_BD category: KNN, TREE, DUAL_CLUSTERING)
                # DEBUG: Track dual vector state
                profile_call("diagnostics_logging_file_io", print, f"DEBUG: Phase 2 feature generation at iteration {self.iteration}, integer_solution={integer_solution}")
                profile_call("diagnostics_logging_file_io", print, f"DEBUG: self.phase0_dual_vectors has {len(self.phase0_dual_vectors)} entries")
                profile_call("diagnostics_logging_file_io", print, f"DEBUG: self.dual_vectors has {len(self.dual_vectors)} entries")
                if len(self.dual_vectors) > 0:
                    profile_call("diagnostics_logging_file_io", print, f"DEBUG: dual_vectors keys (first 5): {list(self.dual_vectors.keys())[:5]}")
                
                # Combine Phase 0 and Phase 1 dual vectors for feature generation
                combined_duals = {}
                combined_duals.update(self.phase0_dual_vectors)  # Add Phase 0 duals
                combined_duals.update(self.dual_vectors)  # Add Phase 1 duals
                
                profile_call("diagnostics_logging_file_io", print, f"Computing features from {len(self.phase0_dual_vectors)} Phase 0 + {len(self.dual_vectors)} Phase 1 = {len(combined_duals)} total dual vectors")

                requested_use_covariates = self.use_covariates

                # Build covariate feature map when available. Supports SCFLP/SUMND/SCMND data readers.
                covariate_feature_vectors = None
                if (
                    self.approach_category == "ML_Augmented_BD"
                    and self.problem_type in {"SCFLP", "SUMND", "SCMND"}
                    and hasattr(self.data, 'covariate_vectors')
                    and self.data.covariate_vectors is not None
                ):
                    raw_covariates = self.data.covariate_vectors
                    covariate_feature_vectors = {}
                    if isinstance(raw_covariates, dict):
                        for sid in range(n_scen):
                            if sid in raw_covariates:
                                covariate_feature_vectors[sid] = np.asarray(raw_covariates[sid]).copy()
                    else:
                        max_sid = min(n_scen, len(raw_covariates))
                        for sid in range(max_sid):
                            covariate_feature_vectors[sid] = np.asarray(raw_covariates[sid]).copy()

                    if not covariate_feature_vectors:
                        covariate_feature_vectors = None

                # covariates_only mode: skip Step 4 feature generation entirely when covariates are usable.
                if requested_use_covariates == 'covariates_only' and covariate_feature_vectors is not None:
                    self.feature_vectors = covariate_feature_vectors
                    profile_call("diagnostics_logging_file_io", print,
                        f"use_covariates=covariates_only: Skipping Step 4 feature generation and "
                        f"using covariates directly for {len(self.feature_vectors)} scenarios"
                    )
                else:
                    # Generate base features from dual vectors when they are needed.
                    base_feature_vectors = self.exclusive_profiler.call(
                        "callback_other_logic", compute_scenario_features_from_duals,

                        combined_duals, self.feature_params
                    )  # Create feature vectors from dual vectors across all trials (Phase 0 + Phase 1)

                    # Handle case where no features could be generated
                    if not base_feature_vectors:
                        profile_call("diagnostics_logging_file_io", print, "[Warning] No feature vectors could be generated from dual vectors.")
                        profile_call("diagnostics_logging_file_io", print, "This usually indicates insufficient dual vector collection in Phase 0/1.")
                        profile_call("diagnostics_logging_file_io", print, "Falling back to solve all scenarios exactly (no ML prediction).")
                        # Set feature vectors to empty and continue with exact solving
                        self.feature_vectors = {}
                        self.feature_gen_time_sec = time.time()-ts
                        # Select all scenarios since we can't use ML prediction
                        self.selected_subproblems = set(self.scenario_key(s) for s in range(n_scen))
                        self.features_computed = True  # Fix: Set flag even on failure to prevent repeated attempts

                        # TASK 1: Track point signature before returning (feature generation failed case)
                        if integer_solution:
                            cuts_after = sum(self.num_cuts_mip_selected.values()) + sum(self.num_cuts_mip_trial.values())
                        else:
                            cuts_after = sum(self.num_cuts_rel_selected.values()) + sum(self.num_cuts_rel_trial.values())
                        cuts_added = cuts_after - cuts_before
                        self._track_point_signature(first_stage_sol, cuts_added, self.iteration)
                        return

                    if requested_use_covariates == 'features_plus_covariates' and covariate_feature_vectors is not None:
                        profile_call("diagnostics_logging_file_io", print, f"use_covariates=features_plus_covariates: Aggregating generated features with covariates for {self.approach_category}")

                        # Create aggregated feature vectors (generated features + covariates)
                        self.feature_vectors = {}
                        for scenario_id in base_feature_vectors.keys():
                            if scenario_id in covariate_feature_vectors:
                                generated_features = base_feature_vectors[scenario_id]
                                covariate_features = covariate_feature_vectors[scenario_id]

                                aggregated_features = np.concatenate([generated_features, covariate_features])
                                self.feature_vectors[scenario_id] = aggregated_features
                            else:
                                profile_call("diagnostics_logging_file_io", print, f"Warning: Scenario {scenario_id} missing covariate vector, using only generated features")
                                self.feature_vectors[scenario_id] = base_feature_vectors[scenario_id]

                        profile_call("diagnostics_logging_file_io", print, f"Created aggregated feature vectors for {len(self.feature_vectors)} scenarios")

                    else:
                        if requested_use_covariates == 'covariates_only' and covariate_feature_vectors is None:
                            profile_call("diagnostics_logging_file_io", print,
                                "use_covariates=covariates_only requested but covariates are unavailable; "
                                "falling back to features_only"
                            )
                        elif requested_use_covariates == 'features_plus_covariates' and covariate_feature_vectors is None:
                            profile_call("diagnostics_logging_file_io", print,
                                "use_covariates=features_plus_covariates requested but covariates are unavailable; "
                                "using generated features only"
                            )

                        self.feature_vectors = base_feature_vectors
                
                self.feature_gen_time_sec = time.time()-ts
                if self.feature_vectors is not None:
                    profile_call("diagnostics_logging_file_io", print, "Feature vectors created successfully.")
                else:
                    profile_call("diagnostics_logging_file_io", print, "[Warning] Feature vectors are None.")
                    
            # select subproblems
            # Ensure p is a valid integer; default to selecting all scenarios when None or invalid
            p = self.n_selected_subproblems
            if p is None or not isinstance(p, int):
                if self.feature_vectors is not None:
                    p = len(self.feature_vectors)
                else:
                    p = self.n_subproblems  # Fallback to number of subproblems

            ts = time.time()
            
            # All prediction methods now use unified approach - selection done in training
            logger.info(f"{self.prediction_method} method selected - will use unified selection+training approach")
            self.selected_subproblems = set()  # Will be set by unified training
            self.method_integrated_mapping = {}   # Will be set by unified training
            
            self.subproblem_selection_time_sec = time.time()-ts

            # Train unified dual predictor ONLY once (at end of Phase 1, beginning of Phase 2)
            # Do NOT retrain in subsequent Phase 2 iterations - clustering would change and break mappings
            # logger.debug(f"Before training check: dual_predictor.is_trained = {self.dual_predictor.is_trained}")
            # logger.debug(f"Feature vectors available: {self.feature_vectors is not None}")
            # logger.debug(f"Phase 1 dual vectors count: {len(self.all_trial_duals)}")
            # logger.debug(f"Phase 0 dual vectors count: {len(self.phase0_dual_vectors)}")
            
            # CRITICAL FIX (April 18, 2026): Only train if not already trained
            # Training should happen once at end of Phase 1 and then used for ALL Phase 2 iterations
            # Retraining in Phase 2 with new features would create a different clustering and break all predictions
            should_train = not self.dual_predictor.is_trained
            if self.dual_predictor.is_trained:
                # Predictor already trained - use it for this Phase 2 iteration without retraining
                # This ensures consistent scenario-to-representative mappings throughout Phase 2
                logger.debug(f"Predictor already trained - using existing mappings for Phase 2 iteration {self.iteration}")
                should_train = False
            
            if should_train:
                logger.info(f"Training unified dual predictor with method={self.prediction_method}")
                ts = time.time()
                try:
                    # Use unified prediction method approach (selection + training integrated)
                    method_params = {}
                    if self.prediction_method == 'KNN':
                        method_params = {
                            'k': self.n_neighbors,
                            'find_optimal_k': getattr(self, 'knn_find_optimal_k', False)
                        }
                    elif self.prediction_method == 'TREE':
                        # Use all_trial_duals which already contains both Phase 0 and Phase 1 duals
                        # in the correct format: {(scenario_id, iteration): dual_dict}
                        method_params = {
                            # D_max: removed for adaptive TREE method
                            'n_min': self.tree_min_leaf_size,
                            'min_impurity_decrease': self.tree_min_impurity_decrease,
                            'quantile_grid_size': self.tree_quantile_grid_size,
                            'all_trial_duals': self.all_trial_duals  # Use unified dual storage (consistent with DUAL_CLUSTERING_WITH_DUAL)
                        }
                    elif self.prediction_method == 'DUAL_CLUSTERING':
                        method_params = {
                            'random_state': self.clustering_random_state,
                            'init_method': self.clustering_init_method,
                            'max_iter': self.clustering_max_iter,
                            'tol': self.clustering_tol,
                            'all_trial_duals': self.all_trial_duals  # Required for ML_Augmented_BD_with_duals
                        }
                    elif self.prediction_method == 'DUAL_CLUSTERING_WITH_DUAL':
                        # Use all_trial_duals which already contains both Phase 0 and Phase 1 duals
                        # in the correct format: {(scenario_id, iteration): dual_dict}
                        method_params = {
                            'random_state': getattr(self, 'dual_clustering_with_dual_random_state', 42),
                            'max_iter': getattr(self, 'dual_clustering_with_dual_max_iter', 300),
                            'tol': getattr(self, 'dual_clustering_with_dual_tol', 1e-6),
                            'all_trial_duals': self.all_trial_duals  # Use unified dual storage
                        }
                    elif self.prediction_method == 'UNCERTAINTY_CLUSTERING':
                        # UNCERTAINTY_CLUSTERING uses demand vectors directly
                        method_params = {
                            'random_state': self.clustering_random_state,
                            'init_method': self.clustering_init_method,
                            'max_iter': self.clustering_max_iter,
                            'tol': self.clustering_tol,
                            'uncertainty_vectors': self.feature_vectors  # Pass uncertainty vectors (demand vectors)
                        }
                    elif self.prediction_method == 'COVARIATE_CLUSTERING':
                        # COVARIATE_CLUSTERING uses covariate vectors directly
                        method_params = {
                            'random_state': self.clustering_random_state,
                            'init_method': self.clustering_init_method,
                            'max_iter': self.clustering_max_iter,
                            'tol': self.clustering_tol,
                            'covariate_vectors': self.feature_vectors  # Pass covariate vectors
                        }
                    elif self.prediction_method == 'BERTSIMAS_RANDOM_SAMPLING':
                        # BERTSIMAS_RANDOM_SAMPLING uses random sampling parameters
                        method_params = {
                            'random_state': getattr(self, 'bertsimas_random_sampling_random_state', 42)
                        }
                    elif is_parametric_predictor(self.predictor_type):
                        method_params = {
                            'all_trial_duals': self.all_trial_duals,
                            'first_stage_by_iteration': self.first_stage_by_iteration,
                        }
                    elif self.prediction_method == 'REG':
                        # Future REG parameters will go here
                        method_params = {}
                    elif self.prediction_method == 'SIMPLE_AVG':
                        # Future SIMPLE_AVG parameters will go here
                        method_params = {}
                    
                    # Use unified prediction method that handles both selection and mapping
                    unified_result = self.exclusive_profiler.call(
                        "callback_other_logic", self.dual_predictor._train_unified,

                        feature_vectors=self.feature_vectors,
                        p=p,  # Number of scenarios to select
                        method_params=method_params
                    )
                    
                    # Extract selected scenarios and mapping from unified result
                    if "__selected_scenarios__" in unified_result:
                        self.selected_subproblems = set(unified_result["__selected_scenarios__"])
                        logger.info(f"{self.prediction_method} unified training selected {len(self.selected_subproblems)} scenarios")
                        
                        # CRITICAL FIX (April 18, 2026): Ensure authoritative sync between callback and predictor
                        # After training, both self.selected_subproblems and self.dual_predictor.selected_subproblems
                        # should contain exactly the same scenarios. Use predictor's version as authoritative since it
                        # comes directly from the selection algorithm (clustering, KNN, tree, etc.)
                        if self.predictor_type != 'RANDOM_SAMPLING':  # Random sampling doesn't have fixed selection
                            predictor_selected = self.dual_predictor.selected_subproblems if hasattr(self.dual_predictor, 'selected_subproblems') else set()
                            
                            if predictor_selected and self.selected_subproblems != predictor_selected:
                                logger.warning(
                                    f"CRITICAL: Mismatch between callback and predictor selected scenarios:\n"
                                    f"  From unified_result: {sorted(self.selected_subproblems)}\n"
                                    f"  From dual_predictor: {sorted(predictor_selected)}"
                                )
                                # Sync: use predictor as authoritative source
                                self.selected_subproblems = predictor_selected.copy()
                                logger.warning(f"Synchronized: Now using {len(self.selected_subproblems)} selected scenarios from predictor")
                    
                    if "__method_mapping__" in unified_result:
                        # Store method-specific mapping with generic name
                        self.method_integrated_mapping = unified_result["__method_mapping__"]
                        # BUG FIX (Feb 22, 2026): Count only actual scenario mappings (exclude metadata keys starting with '__')
                        num_mapped = sum(1 for key in self.method_integrated_mapping.keys() if not str(key).startswith('__'))
                        logger.info(f"{self.prediction_method} unified training created mappings for {num_mapped} unselected scenarios")
                    
                    # Get unified training time from DualPredictor
                    self.predictor_training_time = self.dual_predictor.training_time or 0.0
                    self.KNN_closest_neighbor_time = self.dual_predictor.training_time or 0.0  # For backward compatibility, set to same value
                    
                    logger.info(f"Unified predictor training completed in {self.predictor_training_time:.2f} seconds")
                    
                    # Verify training was successful
                    if not self.dual_predictor.is_trained:
                        logger.error("Training completed but is_trained is still False. This is a bug.")
                        raise RuntimeError("Training completed but predictor not marked as trained")
                    
                    # Verify selected_subproblems (except for BERTSIMAS_RANDOM_SAMPLING which uses dynamic selection)
                    if self.prediction_method != 'BERTSIMAS_RANDOM_SAMPLING':
                        if (not hasattr(self.dual_predictor, 'selected_subproblems') or 
                            self.dual_predictor.selected_subproblems is None or
                            len(self.dual_predictor.selected_subproblems) == 0):
                            logger.error("Training completed but selected_subproblems is empty. This indicates training failure.")
                            raise RuntimeError("Training completed but no scenarios were selected")
                        
                        logger.info(f"Training verification passed: predictor ready with {len(self.dual_predictor.selected_subproblems)} selected scenarios")
                    else:
                        # BERTSIMAS_RANDOM_SAMPLING uses dynamic random selection at each iteration
                        logger.info(f"Training verification passed: BERTSIMAS_RANDOM_SAMPLING ready for dynamic scenario selection")
                    
                    # Store training info in legacy attributes for compatibility
                    training_info = self.dual_predictor.get_training_info()
                    # Store method-specific mappings in predictor for later use
                    if self.prediction_method == 'KNN':
                        # KNN neighbors already stored in predictor during unified training
                        logger.info(f"KNN training completed with {len(getattr(self.dual_predictor, 'knn_neighbors', {}))} neighbor mappings")
                    elif self.prediction_method == 'TREE':
                        # TREE mappings stored in predictor
                        logger.info(f"TREE training completed")
                    elif self.prediction_method == 'REG':
                        # REG mappings stored in predictor  
                        logger.info(f"REG training completed")
                        
                except Exception as e:
                    logger.error(f"Unified predictor training failed: {e}")
                    logger.error(f"Training context: method={self.prediction_method}, feature_vectors={self.feature_vectors is not None}, p={p}")
                    logger.error(f"Dual data: all_trial_duals={len(self.all_trial_duals)}, phase0_duals={len(self.phase0_dual_vectors)}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    self.dual_predictor.is_trained = False
                    raise  # Re-raise to make the error visible
                    
            else:
                # Skipping training - use predictor's existing trained state
                # This ensures Phase 2 uses the same scenario selection and mappings as Phase 1
                logger.debug(f"Phase 2 iteration {self.iteration}: Using predictor's trained state without retraining")
                
                # Ensure selected_subproblems is set from the predictor's Phase 1 training
                if hasattr(self.dual_predictor, 'selected_subproblems') and self.dual_predictor.selected_subproblems:
                    self.selected_subproblems = self.dual_predictor.selected_subproblems.copy()
                    logger.debug(f"Selected {len(self.selected_subproblems)} scenarios from Phase 1 training")
                else:
                    logger.error("Predictor trained but selected_subproblems not available. This is a bug.")
                    raise RuntimeError("Predictor trained but selected_subproblems missing")
                    
            self.features_computed = True #! Set flag to True to avoid recomputing features in every callback iteration

        # PHASE 2: Determine if ML should be used for this context
        # Check method-specific behavior for current callback type (integer vs LP)
        use_ml_this_iteration = self._should_use_ml_in_phase2(integer_solution)
        
        if not use_ml_this_iteration:
            # PHASE 2 EXACT MODE: Solve all subproblems exactly without ML
            context = "integer" if integer_solution else "LP relaxation"
            method = normalize_use_prediction(self.use_prediction)
            profile_call("diagnostics_logging_file_io", print, f"Phase 2: Exact solving at {context} (method={method}), iteration {self.iteration}")
            
            for sid in range(n_scen):
                res = self._solve_dual_exact(sid, first_stage_sol, context="phase2_exact")
                
                duals = self.structure_duals(res)
                obj = self.compute_dual_obj(duals, sid, first_stage_sol)
                if obj > theta_val[self.theta_keys[sid]] + self.EPS:
                    self.add_optimality_cut(mod, duals, sid, integer_solution)
                    # Count as phase 2 trial cuts for consistency
                    if integer_solution:
                        self.num_cuts_mip_trial[sid] += 1
                    else:
                        self.num_cuts_rel_trial[sid] += 1
            
            # TASK 1: Track point signature before returning
            if integer_solution:
                cuts_after = sum(self.num_cuts_mip_selected.values()) + sum(self.num_cuts_mip_trial.values())
            else:
                cuts_after = sum(self.num_cuts_rel_selected.values()) + sum(self.num_cuts_rel_trial.values())
            cuts_added = cuts_after - cuts_before
            self._track_point_signature(first_stage_sol, cuts_added, self.iteration)
            
            return  # Exit after exact solving
        
        # PHASE 2 ML MODE: Use ML prediction for unselected subproblems
        context = "integer" if integer_solution else "LP relaxation"
        method = normalize_use_prediction(self.use_prediction)
        profile_call("diagnostics_logging_file_io", print, f"Phase 2: ML-augmented solving at {context} (method={method}), iteration {self.iteration}")
        
        # PHASE 2: Solve selected subproblems exactly and predict duals for unselected ones
        # exact solves for selected
        dual_cache = {}
        parametric_mode = is_parametric_predictor(self.predictor_type)
        
        # Handle dynamic sampling for BERTSIMAS_RANDOM_SAMPLING method
        if self.prediction_method == "BERTSIMAS_RANDOM_SAMPLING":
            # Randomly sample scenarios at each iteration
            import random
            random_state = getattr(self, 'bertsimas_random_sampling_random_state', 42)
            random.seed(random_state + self.iteration)  # Different seed each iteration
            
            all_scenarios = list(range(n_scen))
            # For BERTSIMAS_RANDOM_SAMPLING, selected_subproblems is empty, so use configured value
            p = self.n_selected_subproblems if self.n_selected_subproblems else max(1, int(0.2 * n_scen))
            
            # Random sampling without replacement
            current_selected = set(random.sample(all_scenarios, min(p, len(all_scenarios))))
            profile_call("diagnostics_logging_file_io", print, f"BERTSIMAS_RANDOM_SAMPLING iteration {self.iteration}: randomly selected {len(current_selected)} scenarios: {sorted(current_selected)}")
        else:
            # Use fixed selected subproblems for other methods
            current_selected = self.selected_subproblems
            
            logger.warning(f"Phase 2 iter {self.iteration}: selected_subproblems={sorted(self.selected_subproblems)}")
            
            # CRITICAL FIX (April 18, 2026): Ensure all clustering-selected scenarios are in current_selected
            # For clustering methods, verify that selected scenarios from the predictor are all in current_selected
            # If clustering selected a scenario but it's not in current_selected, it won't be solved exactly
            # and won't be available in dual_cache for cluster members to reference.
            
            # Check if this is a clustering method by checking for clustering_mappings
            if (hasattr(self.dual_predictor, 'predictor_mappings') and 
                self.dual_predictor.predictor_mappings and
                hasattr(self.dual_predictor, 'selected_subproblems') and
                self.dual_predictor.selected_subproblems):
                
                predictor_selected = self.dual_predictor.selected_subproblems
                logger.warning(f"  predictor_selected={sorted(predictor_selected)}")
                logger.warning(f"  current_selected={sorted(current_selected)}")
                
                # Check for any mismatch
                if current_selected != predictor_selected:
                    logger.warning(
                        f"CLUSTERING: Syncing selected scenarios\n"
                        f"  callback has: {sorted(current_selected)}\n"
                        f"  predictor selected: {sorted(predictor_selected)}"
                    )
                    # Ensure all clustering-selected scenarios are included
                    missing_scenarios = predictor_selected - current_selected
                    extra_scenarios = current_selected - predictor_selected
                    
                    if missing_scenarios:
                        logger.warning(f"  CRITICAL: Adding missing representative scenarios: {sorted(missing_scenarios)}")
                        current_selected = current_selected.union(missing_scenarios)
                    
                    if extra_scenarios:
                        logger.warning(f"  Removing extra scenarios: {sorted(extra_scenarios)}")
                        current_selected = current_selected - extra_scenarios
                    
                    logger.warning(f"  After sync: current_selected={sorted(current_selected)}")
                else:
                    logger.warning(f"  No mismatch - predictor_selected == current_selected")
        
        # DIAGNOSTIC: Track selected subproblem cut statistics
        selected_cuts_added = 0
        selected_violations = 0
        selected_max_violation = 0.0
        
        #! Solve selected subproblems exactly and add cuts
        if self.problem_type.upper() in ("UFL", "UHL", "SSLP", "SCFLP", "NESP", "NE-SP", "SUMND", "SCMND"):
            for sid in current_selected:
                res = self._solve_dual_exact(
                    sid, first_stage_sol,
                    context="algorithm" if self.dual_predictor.is_trained else "fallback",
                )
                
                duals = self.structure_duals(res)
                dual_cache[sid] = duals
                if parametric_mode:
                    self.dual_predictor.record_parametric_selected_exact_solves(1)
                obj = self.compute_dual_obj(duals, sid, first_stage_sol)
                
                # DIAGNOSTIC: Track violations
                violation = obj - theta_val[self.theta_keys[sid]]
                # BUG FIX (Feb 22, 2026): Use self.EPS tolerance to avoid counting numerical noise as violations
                if violation > self.EPS:
                    selected_violations += 1
                    selected_max_violation = max(selected_max_violation, violation)
                
                if obj > theta_val[self.theta_keys[sid]] + self.EPS:
                    self.add_optimality_cut(mod, duals, sid, integer_solution)
                    selected_cuts_added += 1
                    # Exact duals from selected subproblems
                    if integer_solution:
                        self.num_cuts_mip_selected[sid]+=1
                    else:
                        self.num_cuts_rel_selected[sid]+=1
        
        # DIAGNOSTIC: Log selected subproblem statistics (commented - too verbose per iteration)
        # logger.info(f"Selected subproblems iter {self.iteration}: {selected_cuts_added} cuts added from {len(current_selected)} selected, "
        #            f"{selected_violations} violations detected, max violation: {selected_max_violation:.6e}")

        # prepare unselected set of ids
        # CRITICAL FIX (April 18, 2026): Use current_selected instead of self.selected_subproblems
        # because current_selected may have been updated to include missing clustering representatives
        all_idxs = set(range(n_scen))
        if self.prediction_method == "BERTSIMAS_RANDOM_SAMPLING":
            unselected = all_idxs - current_selected
        else:
            unselected = all_idxs - current_selected  # Use current_selected which may include added scenarios

        # Handle permanent switch: solve all subproblems exactly if permanently switched
        if self.permanently_switched:
            profile_call("diagnostics_logging_file_io", print, f"Permanent switch active: solving all unselected subproblems exactly")
            for sid in unselected:
                res = self._solve_dual_exact(sid, first_stage_sol, context="switch")
                
                duals = self.structure_duals(res)
                obj = self.compute_dual_obj(duals, sid, first_stage_sol)
                if obj > theta_val[self.theta_keys[sid]] + self.EPS:
                    self.add_optimality_cut(mod, duals, sid, integer_solution)
                    # These are exact switch cuts after permanent switch
                    if integer_solution:
                        self.num_cuts_mip_exact_switch[sid] += 1
                    else:
                        self.num_cuts_rel_exact_switch[sid] += 1
            
            # TASK 1: Track point signature before returning (permanent switch case)
            if integer_solution:
                cuts_after = sum(self.num_cuts_mip_selected.values()) + sum(self.num_cuts_mip_trial.values()) + sum(self.num_cuts_mip_ml.values()) + sum(self.num_cuts_mip_fallback.values()) + sum(self.num_cuts_mip_exact_switch.values())
            else:
                cuts_after = sum(self.num_cuts_rel_selected.values()) + sum(self.num_cuts_rel_trial.values()) + sum(self.num_cuts_rel_ml.values()) + sum(self.num_cuts_rel_fallback.values()) + sum(self.num_cuts_rel_exact_switch.values())
            cuts_added = cuts_after - cuts_before
            self._track_point_signature(first_stage_sol, cuts_added, self.iteration)
            
            # Skip ML prediction when permanently switched
            return

        if parametric_mode:
            prepared = self.dual_predictor.prepare_parametric_iteration(
                solved_duals=dual_cache,
                first_stage_values=first_stage_sol,
                predicted_ids=unselected,
                iteration=self.iteration,
            )
            if not prepared:
                # This is the exact fallback prescribed by Algorithm
                # parametric-labd when the manuscript training model is
                # infeasible; it also safely handles unexpected solver failure.
                self.num_fallback_used += 1
                self.dual_predictor.record_parametric_exact_fallback(len(unselected))
                self._handle_fallback(
                    mod,
                    unselected,
                    first_stage_sol,
                    theta_val,
                    dual_cache,
                    integer_solution,
                )
                cuts_after = (
                    sum(self.num_cuts_mip_selected.values())
                    + sum(self.num_cuts_mip_ml.values())
                    + sum(self.num_cuts_mip_fallback.values())
                    + sum(self.num_cuts_mip_exact_switch.values())
                    if integer_solution
                    else sum(self.num_cuts_rel_selected.values())
                    + sum(self.num_cuts_rel_ml.values())
                    + sum(self.num_cuts_rel_fallback.values())
                    + sum(self.num_cuts_rel_exact_switch.values())
                )
                self._track_point_signature(
                    first_stage_sol, cuts_after - cuts_before, self.iteration
                )
                return

        # Reset cuts counter for unselected subproblems at start of prediction phase
        self.cuts_from_unselected_this_iter = 0

        # Unified prediction for unselected subproblems using DualPredictor
        if self.dual_predictor.is_trained:
            ts = time.time()
            handled_prediction_ids = set()
            try:
                # Use KNN unified predictor directly
                preds = self.exclusive_profiler.call(
                    "callback_other_logic" if is_parametric_predictor(self.predictor_type)
                    else "nonparametric_dual_transfer",
                    self.dual_predictor.predict, unselected, dual_cache
                )
                method_label = self.prediction_method
                
                # Note: We don't overwrite predictor_training_time here since it should only
                # capture the initial training time, not per-iteration prediction/retraining time
                prediction_time = time.time() - ts
                self.num_predictions += 1
                
                # DIAGNOSTIC: Track ML prediction cut statistics
                ml_cuts_added = 0
                ml_violations = 0
                ml_max_violation = 0.0
                parametric_validation_fallback = False
                
                # Process predictions and add cuts
                for sid in unselected:
                    duals = preds[sid]
                    if (
                        parametric_mode
                        and not self.dual_predictor.validate_parametric_dual(duals)
                    ):
                        # Never construct a cut from an uncertified parametric
                        # vector.  Solve only the affected subproblem exactly.
                        self.dual_predictor.record_parametric_exact_fallback(1)
                        self._handle_fallback(
                            mod,
                            [sid],
                            first_stage_sol,
                            theta_val,
                            dual_cache,
                            integer_solution,
                        )
                        parametric_validation_fallback = True
                        handled_prediction_ids.add(sid)
                        continue
                    obj = self.compute_dual_obj(duals, sid, first_stage_sol)
                    
                    # DIAGNOSTIC: Track violations
                    violation = obj - theta_val[self.theta_keys[sid]]
                    # BUG FIX (Feb 22, 2026): Use self.EPS tolerance to avoid counting numerical noise as violations
                    if violation > self.EPS:
                        ml_violations += 1
                        ml_max_violation = max(ml_max_violation, violation)
                    
                    cut_is_violated = obj > theta_val[self.theta_keys[sid]] + self.EPS
                    if cut_is_violated:
                        self.add_optimality_cut(mod, duals, sid, integer_solution, cut_origin="ml")
                        self.cuts_from_unselected_this_iter += 1  # Track cuts from unselected
                        ml_cuts_added += 1
                        
                        # Update unified ML counters
                        if integer_solution:
                            self.num_cuts_mip_ml[sid] += 1
                        else:
                            self.num_cuts_rel_ml[sid] += 1
                    if parametric_mode:
                        self.dual_predictor.record_parametric_cut_evaluation(
                            violation, cut_is_violated
                        )
                    handled_prediction_ids.add(sid)

                if parametric_validation_fallback:
                    self.num_fallback_used += 1
                
                # DIAGNOSTIC: Log ML prediction statistics (commented - too verbose per iteration)
                # logger.info(f"ML predictions iter {self.iteration}: {ml_cuts_added} ML cuts added, "
                #            f"{ml_violations} violations from predictions, max violation: {ml_max_violation:.6e}")
                
                # logger.debug(f"Unified {method_label} prediction completed for {len(unselected)} scenarios")
                
            except Exception as e:
                logger.warning(f"Unified prediction failed: {e}. Using fallback method.")
                remaining = set(unselected) - handled_prediction_ids
                if remaining:
                    self.num_fallback_used += 1
                if parametric_mode:
                    self.dual_predictor.record_parametric_exact_fallback(len(remaining))
                if remaining:
                    self._handle_fallback(
                        mod,
                        remaining,
                        first_stage_sol,
                        theta_val,
                        dual_cache,
                        integer_solution
                    )
        else:
            logger.warning("Dual predictor not trained. Using fallback method.")
            self.num_fallback_used += 1
            self._handle_fallback(
                mod,
                unselected,
                first_stage_sol,
                theta_val,
                dual_cache,
                integer_solution
            )

        # SWITCHING STRATEGY LOGIC
        # Only apply switching strategies when use_prediction=True and not in no_switch mode
        if self.use_prediction and self.switching_strategy != "no_switch":
            # Check if any cuts were added from unselected subproblems in this iteration
            if self.cuts_from_unselected_this_iter == 0:
                self._apply_switching_strategy(mod, unselected, first_stage_sol, theta_val, integer_solution)
        
        # TASK 1: Track point signature at end of Phase 2 iteration
        cuts_after = sum(self.num_cuts_mip_selected.values()) + sum(self.num_cuts_mip_ml.values()) + sum(self.num_cuts_mip_fallback.values()) + sum(self.num_cuts_mip_exact_switch.values()) if integer_solution else sum(self.num_cuts_rel_selected.values()) + sum(self.num_cuts_rel_ml.values()) + sum(self.num_cuts_rel_fallback.values()) + sum(self.num_cuts_rel_exact_switch.values())
        cuts_added = cuts_after - cuts_before
        self._track_point_signature(first_stage_sol, cuts_added, self.iteration)
        
        # End of Phase 2 iteration - callback returns control to Gurobi

    def _apply_switching_strategy(self, mod, unselected, first_stage_sol, theta_val, integer_solution):
        """
        Apply switching strategy logic when no cuts are added from unselected subproblems.
        
        This method implements the enhanced Benders decomposition switching strategies:
        - "alternate": Periodically solve unselected subproblems exactly for verification
        - "permanent_switch": Permanently switch to exact solving when ML becomes insufficient
        
        Args:
            mod: Gurobi model for adding cuts
            unselected: Set of unselected subproblem indices  
            first_stage_sol: Current first-stage solution
            theta_val: Current theta variable values
            integer_solution: Boolean flag for solution type
        """
        # No cuts from unselected subproblems - trigger switching logic
        profile_call("diagnostics_logging_file_io", print, f"No cuts from unselected subproblems in iteration {self.iteration}. Applying {self.switching_strategy} strategy.")
        
        self.num_switch_events += 1
        self.switch_iterations.append(self.iteration)
        
        # Solve unselected subproblems exactly
        cuts_added_from_exact = self._solve_unselected_exactly(
            mod, unselected, first_stage_sol, theta_val, integer_solution
        )
        
        if cuts_added_from_exact == 0:
            # No cuts from exact solving either - algorithm should terminate
            profile_call("diagnostics_logging_file_io", print, f"No cuts found from exact solving of unselected subproblems. Optimality reached.")
            # Note: Gurobi will naturally terminate when no cuts are added
            return
        else:
            profile_call("diagnostics_logging_file_io", print, f"Added {cuts_added_from_exact} cuts from exact solving of unselected subproblems.")
            
            # Handle permanent switch if strategy requires it
            if self.switching_strategy == "permanent_switch":
                self.permanently_switched = True
                profile_call("diagnostics_logging_file_io", print, "Permanently switched to exact solving mode for remaining iterations.")

    def _solve_unselected_exactly(self, mod, unselected, first_stage_sol, theta_val, integer_solution):
        """
        Solve unselected subproblems exactly and add cuts if violations found.
        
        Returns:
            int: Number of cuts added from exact solving
        """
        cuts_added = 0
        
        for sid in unselected:
            res = self._solve_dual_exact(sid, first_stage_sol, context="switch")
            duals = self.structure_duals(res)
            obj = self.compute_dual_obj(duals, sid, first_stage_sol)
            
            if obj > theta_val[self.theta_keys[sid]] + self.EPS:
                self.add_optimality_cut(mod, duals, sid, integer_solution)
                cuts_added += 1
                
                # Track cuts from exact switch solving
                if integer_solution:
                    self.num_cuts_mip_exact_switch[sid] += 1
                else:
                    self.num_cuts_rel_exact_switch[sid] += 1
        
        return cuts_added

    def get_switching_statistics(self):
        """
        Return statistics about switching strategy usage.
        
        Returns:
            dict: Dictionary containing switching strategy statistics
        """
        return {
            "switching_strategy": self.switching_strategy,
            "num_switch_events": self.num_switch_events,
            "switch_iterations": self.switch_iterations.copy(),
            "permanently_switched": self.permanently_switched,
            "num_cuts_exact_switch_mip": sum(self.num_cuts_mip_exact_switch.values()),
            "num_cuts_exact_switch_rel": sum(self.num_cuts_rel_exact_switch.values()),
            "cuts_exact_switch_by_scenario_mip": dict(self.num_cuts_mip_exact_switch),
            "cuts_exact_switch_by_scenario_rel": dict(self.num_cuts_rel_exact_switch)
        }

    def finalize_phase_timing(self):
        """
        Finalize phase timing when optimization completes.
        Call this after optimization to ensure all phase timings are properly calculated.
        """
        current_time = time.time()
        
        # If Phase 1 was started but not ended (algorithm ended during Phase 1)
        if self.phase1_start_time is not None and self.phase1_execution_time_sec is None:
            self.phase1_execution_time_sec = current_time - self.phase1_start_time
            
        # If Phase 2 was started but not ended (algorithm ended during Phase 2)
        if self.phase2_start_time is not None and self.phase2_execution_time_sec is None:
            self.phase2_execution_time_sec = current_time - self.phase2_start_time

    def _handle_fallback(self, mod, subs, xbar, theta_val, dual_cache, integer_solution=True):
        """
        Execute fallback strategy for unselected subproblems when primary ML prediction fails or is disabled.
        
        This method implements a backup mechanism to obtain dual variables for unselected subproblems
        when the primary prediction method (typically regression) fails to train or produce valid
        predictions. It ensures that Benders decomposition can continue even when ML components fail,
        maintaining algorithmic correctness while potentially sacrificing computational efficiency.
        
        Fallback Strategies:
        ------------------
        **SOLVE**: Exact dual subproblem solving
        - Calls solve_dual_subproblem() for each unselected subproblem
        - Guarantees optimal dual variables and cuts
        - Computationally expensive but maintains solution quality
        - Used when accuracy is prioritized over speed
        
        **KNN**: K-nearest neighbor dual approximation
        - Uses knn_predict() to estimate duals from similar selected subproblems
        - Leverages pre-computed neighbor mappings from compute_knn_neighbors()
        - Fast but approximate, quality depends on feature similarity
        - Used when speed is prioritized over perfect accuracy
        
        When This Method Is Called:
        -------------------------
        1. **Primary Prediction Failure (prediction_method='REG')**:
           - train_feasible_predictor() returns empty/invalid trained_models
           - Regression training fails due to insufficient data or numerical issues
           - Called from handle_callback() when self.trained_models is empty
           
        2. **Disabled Prediction (use_prediction=False)**:
           - ML is completely disabled, traditional Benders used in Phase 2
           - All unselected subproblems handled via fallback
           - Called directly from handle_callback() without attempting prediction
           
        3. **KNN Fallback During Mixed Strategy**:
           - Primary method works for most subproblems
           - Individual subproblems may still require fallback handling
           
        Mathematical Context:
        -------------------
        For each unselected subproblem sid:
        1. **Dual Acquisition**: π* = fallback_method(sid, current_solution)
        2. **Violation Check**: dual_obj(π*, sid, x̄) > θ[sid] ?
        3. **Cut Generation**: If violated, add θ[sid] ≥ dual_obj(π*, sid, x)
        
        Where:
        - π* = optimal (SOLVE) or approximate (KNN) dual variables
        - x̄ = current first-stage solution from Gurobi callback
        - θ[sid] = current second-stage cost estimate for scenario sid
        
        Args:
        ----
        mod (gurobi.Model): Gurobi model for adding optimality cuts via cbLazy/cbCut
        subs (set/list): Collection of unselected subproblem indices requiring fallback
        xbar (dict): Current first-stage solution values from Gurobi callback
        theta_val (dict): Current theta variable values for all scenarios
        dual_cache (dict): Cache of dual solutions from selected subproblems
            Format: {selected_sid: structured_duals_dict}
            Used by KNN fallback to compute neighbor averages
        integer_solution (bool): Solution type flag for cut tracking and addition method
            - True: Integer solution (MIPSOL) → use cbLazy, track *_mip_* counters
            - False: LP relaxation (MIPNODE) → use cbCut, track *_rel_* counters
        
        Cut Tracking Logic:
        -----------------
        The method maintains separate counters based on fallback method and solution type:
        
        **SOLVE Fallback**:
        - Integer solutions: num_cuts_mip_fallback[sid] += 1
        - LP relaxations: num_cuts_rel_fallback[sid] += 1
        
        **KNN Fallback**:
        - Integer solutions: num_cuts_mip_ml[sid] += 1  
        - LP relaxations: num_cuts_rel_ml[sid] += 1
        
        This tracking enables detailed performance analysis of different cut sources.
        
        Error Handling:
        --------------
        - Raises ValueError for unsupported fallback_method values
        - Assumes dual_cache is properly populated for KNN fallback
        - Relies on solve_dual_subproblem() error handling for SOLVE fallback
        
        Performance Considerations:
        -------------------------
        - **SOLVE**: O(|subs| × subproblem_complexity) - can be expensive for large |subs|
        - **KNN**: O(|subs| × k × dual_dimension) - much faster, linear in problem size
        - Trade-off between solution quality (SOLVE) and computational speed (KNN)
        - Fallback frequency impacts overall algorithm performance
        
        Integration with Main Algorithm:
        ------------------------------
        1. **Trigger**: Called when handle_callback() detects prediction failure
        2. **Input**: Unselected subproblems that need dual variable estimation  
        3. **Process**: Apply chosen fallback strategy to obtain duals
        4. **Output**: Add optimality cuts to strengthen master problem
        5. **Tracking**: Update cut counters for performance monitoring
        
        Example Usage Flow:
        -----------------
        ```
        # In handle_callback(), after prediction failure:
        if not self.trained_models:  # Regression failed
            self.num_fallback_used += 1
            self._handle_fallback(mod, unselected, first_stage_sol, 
                                theta_val, dual_cache, integer_solution)
        ```
        
        Design Rationale:
        ---------------
        This method ensures robustness of the learning-augmented Benders algorithm by:
        1. **Maintaining correctness**: Always provides valid dual estimates
        2. **Graceful degradation**: Falls back to traditional methods when ML fails
        3. **Flexibility**: Supports multiple fallback strategies (exact/approximate)
        4. **Transparency**: Tracks fallback usage for algorithm analysis
        """
        for sid in subs: #unselected subproblem indicies 
            if self.fallback_method=='SOLVE':
                res = self._solve_dual_exact(sid, xbar, context="fallback")
                duals = self.structure_duals(res) #get the output of Gorubi and save duals in proper format 
            elif self.fallback_method == "KNN":
                # Use unified predictor for KNN fallback if available, otherwise use legacy method
                if hasattr(self, 'dual_predictor') and self.dual_predictor.predictor_type == 'KNN' and self.dual_predictor.is_trained:
                    knn_pred = self.exclusive_profiler.call(
                        "nonparametric_dual_transfer", self.dual_predictor.predict, [sid], dual_cache
                    )
                    duals = knn_pred[sid]
                elif hasattr(self, 'knn_neighbors') and self.knn_neighbors:
                    duals = self.knn_predict([sid], dual_cache)[sid]
                else:
                    # Compute KNN on the fly if no trained predictor available
                    if not hasattr(self, 'knn_neighbors') or not self.knn_neighbors:
                        self.knn_neighbors = self.compute_knn_neighbors()
                    duals = self.knn_predict([sid], dual_cache)[sid]
            else:
                raise ValueError(f"Unsupported fallback_method: {self.fallback_method}")   
            
            obj = self.compute_dual_obj(duals, sid, xbar)
            if obj > theta_val[self.theta_keys[sid]] + self.EPS:
                self.add_optimality_cut(
                    mod, duals, sid, integer_solution,
                    cut_origin="exact" if self.fallback_method == "SOLVE" else "ml",
                )
                if self.fallback_method=='SOLVE':
                    if integer_solution: self.num_cuts_mip_fallback[sid]+=1
                    else:               self.num_cuts_rel_fallback[sid]+=1
                else: # KNN fallback
                    if integer_solution: self.num_cuts_mip_ml[sid]+=1
                    else:               self.num_cuts_rel_ml[sid]+=1

    def predict_duals(self, subs):
        """
        [LEGACY METHOD - Use DualPredictor.predict() for new code]
        
        Predict dual variables for unselected subproblems using trained parametric regression models.
        
        This method implements the core machine learning prediction component of the learning-augmented
        Benders decomposition algorithm. It uses linear regression models trained on selected subproblem
        dual solutions to predict dual variables for unselected subproblems, enabling approximate
        optimality cut generation without solving expensive dual subproblems.
        
        **Note**: This method is maintained for backward compatibility. New code should use
        the unified DualPredictor class which handles REG, KNN, and TREE methods consistently.
        
        Mathematical Foundation:
        ----------------------
        For each dual variable type and each unselected subproblem s:
        
        **Linear Prediction Model**: π̂ₛ = wᵀ · φₛ
        
        Where:
        - π̂ₛ = predicted dual variable for subproblem s
        - w = trained weight vector (from train_feasible_predictor)
        - φₛ = feature vector for subproblem s (from feature engineering)
        - ᵀ = transpose operation
        
        **Problem-Specific Predictions**:
        
        *UFL (Uncapacitated Facility Location)*:
        - λ̂ₛ = w_λᵀ · φₛ (customer demand dual)
        - μ̂ⱼₛ = w_μⱼᵀ · φₛ ∀j ∈ F (facility capacity duals)
        
        *UHL (Uncapacitated Hub Location)*:
        - v̂ₖₛ = w_vₖᵀ · φₛ ∀k ∈ N (origin node duals)
        - û_lₛ = w_uₗᵀ · φₛ ∀l ∈ N (destination node duals)
        
        *SSLP (Server Location Problem)*:
        - π̂ᵢₛ = w_πᵢᵀ · φₛ ∀i ∈ C (customer demand duals)
        - γ̂ᵢⱼₛ = w_γᵢⱼᵀ · φₛ ∀(i,j) ∈ C×F (service assignment duals)
        
        *SCFLP (Single-Source Capacitated Facility Location)*:
        - μ̂ᵢₛ = w_μᵢᵀ · φₛ ∀i ∈ I (customer demand duals)
        - ν̂ⱼₛ = w_νⱼᵀ · φₛ ∀j ∈ J (facility capacity duals)
        
        Training Context:
        ---------------
        The weight vectors w are obtained from train_feasible_predictor(), which:
        1. Uses dual solutions from selected subproblems as training targets
        2. Employs regularized regression with L1/L2 penalties for generalization
        3. Ensures feasibility constraints are respected in the dual space
        4. Optimizes prediction accuracy while maintaining cut validity
        
        Algorithm Flow:
        -------------
        1. **Input Validation**: Verify trained_models exist (from successful training)
        2. **Feature Extraction**: Retrieve φₛ for each unselected subproblem s
        3. **Vectorized Prediction**: Compute π̂ₛ = wᵀ · φₛ for each dual variable type
        4. **Structure Assembly**: Organize predictions into standardized dual dictionaries
        5. **Type Conversion**: Ensure numerical stability with explicit float conversion
        
        Integration with Benders Decomposition:
        -------------------------------------
        Predicted duals are used to:
        1. **Estimate second-stage costs**: via compute_dual_obj(π̂ₛ, s, x̄)
        2. **Generate optimality cuts**: θₛ ≥ dual_obj(π̂ₛ, s, x) if violation detected
        3. **Accelerate convergence**: by avoiding expensive exact subproblem solving
        4. **Maintain solution quality**: through careful feature engineering and training
        
        Args:
        ----
        subs (list/set): Collection of unselected subproblem indices requiring dual prediction
            - UFL/SCFLP/SSLP: Integer scenario indices [0, 1, 2, ...]
            - UHL: Origin-destination pair indices referencing data.S tuples
            
        Returns:
        -------
        dict: Nested dictionary mapping subproblem indices to predicted dual structures
            Format: {subproblem_id: structured_dual_dict}
            
            **UFL Structure**:
            {sid: {'lambda': float, 'mu': {j: float ∀j ∈ F}}}
            
            **UHL Structure**:
            {ij: {'v': {k: float ∀k ∈ N}, 'u': {l: float ∀l ∈ N}}}
            
            **SSLP Structure**:
            {sid: {'pi': {i: float ∀i ∈ C}, 'gamma': {(i,j): float ∀(i,j) ∈ C×F}}}
            
            **SCFLP Structure**:
            {sid: {'mu': {i: float ∀i ∈ I}, 'nu': {j: float ∀j ∈ J}}}
            
        Raises:
        ------
        ValueError: If self.trained_models is empty (training failed or not performed)
        ValueError: If unsupported problem_type encountered
        
        Quality Assurance:
        ----------------
        **Numerical Stability**:
        - Explicit numpy array conversion for reliable dot products
        - Float casting to prevent numerical precision issues
        - Consistent indexing to avoid key type mismatches
        
        **Model Validation** (handled externally):
        - Feature vector consistency checking
        - Training data quality validation
        - Cross-validation during model selection
        
        **Feasibility Considerations**:
        - Trained models incorporate dual feasibility constraints
        - Regularization helps prevent overfitting to training scenarios
        - Feature engineering captures problem structure for better generalization
        
        Performance Characteristics:
        --------------------------
        **Time Complexity**: O(|subs| × d × k) where:
        - |subs| = number of unselected subproblems
        - d = feature vector dimension
        - k = number of dual variables per subproblem
        
        **Space Complexity**: O(|subs| × k) for storing prediction results
        
        **Computational Benefits**:
        - **vs. Exact Solving**: 100-1000x faster than solve_dual_subproblem()
        - **vs. KNN**: More accurate with proper training, similar speed
        - **Scalability**: Linear in problem size, independent of subproblem complexity
        
        Usage Context:
        -------------
        Called from handle_callback() during Phase 2 when:
        1. **Successful Training**: self.trained_models populated by train_feasible_predictor()
        2. **Regression Method**: prediction_method='REG' (primary prediction mode)
        3. **Unselected Subproblems**: Applied to all scenarios not in selected_subproblems
        4. **Cut Generation**: Results feed into dual objective computation and cut addition
        
        Error Recovery:
        --------------
        If prediction fails or produces invalid results:
        1. **Immediate**: Method raises ValueError for caller handling
        2. **Callback Level**: handle_callback() catches and triggers _handle_fallback()
        3. **Fallback Options**: Exact solving (SOLVE) or k-nearest neighbors (KNN)
        4. **Tracking**: num_fallback_used counter incremented for performance analysis
        
        Example Prediction Flow:
        ----------------------
        ```
        # For UFL with 3 unselected scenarios and 2 facilities:
        subs = [2, 4, 7]
        predictions = self.predict_duals(subs)
        
        # Result structure:
        {
            2: {'lambda': 15.3, 'mu': {0: 2.1, 1: 3.7}},
            4: {'lambda': 12.8, 'mu': {0: 1.9, 1: 4.2}},
            7: {'lambda': 18.1, 'mu': {0: 2.8, 1: 3.1}}
        }
        ```
        
        Design Philosophy:
        ----------------
        This method embodies the core innovation of learning-augmented Benders decomposition:
        1. **Efficiency**: Dramatically reduces computational cost of dual estimation
        2. **Accuracy**: Maintains solution quality through careful ML model design  
        3. **Robustness**: Integrates seamlessly with traditional Benders fallbacks
        4. **Generality**: Supports multiple problem types with unified interface
        """
        warnings.warn(
            "predict_duals is deprecated. Use DualPredictor.predict() for new code. "
            "This method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not self.trained_models:
            raise ValueError("No trained models for prediction.")

        preds = {}

        if self.problem_type == "UFL":
            for sid in subs:
                phi = self.feature_vectors[sid]
                w_lambda = np.asarray(self.trained_models['lambda'])
                phi_array = np.asarray(phi)
                lam = float(np.dot(w_lambda, phi_array))
                mu_pred = {j: float(np.dot(np.asarray(self.trained_models['mu'][j]), phi_array)) for j in self.data.F}
                preds[sid] = {"lambda": lam, "mu": mu_pred}

        elif self.problem_type == "UHL":
            for ij in subs:
                phi = self.feature_vectors[ij]
                phi_array = np.asarray(phi)
                v_pred = {
                    int(k) if isinstance(k, np.integer) else k: 
                    float(np.dot(np.asarray(self.trained_models['v'][int(k) if isinstance(k, np.integer) else k]), phi_array)) 
                    for k in self.data.N
                }
                u_pred = {
                    int(l) if isinstance(l, np.integer) else l: 
                    float(np.dot(np.asarray(self.trained_models['u'][int(l) if isinstance(l, np.integer) else l]), phi_array)) 
                    for l in self.data.N
                }
                preds[ij] = {"v": v_pred, "u": u_pred}

        elif self.problem_type == "SCFLP":
            for sid in subs:
                phi = self.feature_vectors[sid]
                phi_array = np.asarray(phi)
                
                # Predict mu values for all customers
                mu_pred = {i: float(np.dot(np.asarray(self.trained_models['mu'][i]), phi_array)) 
                          for i in self.data.I}
                
                # Predict nu values for all facilities
                nu_pred = {j: float(np.dot(np.asarray(self.trained_models['nu'][j]), phi_array)) 
                          for j in self.data.J}
                
                preds[sid] = {"mu": mu_pred, "nu": nu_pred}

        elif self.problem_type == "SSLP":
            for sid in subs:
                # feature vector φs  (already built when subproblem was first solved)
                phi_array = np.asarray(self.feature_vectors[sid])

                # ---- π̂_i  =  w_i · φs
                pi_pred = {
                    i: float(np.dot(np.asarray(self.trained_models['pi'][i]), phi_array))
                    for i in self.data.C
                }

                # ---- γ̂_{ij}  =  w_{ij} · φs   for every (i,j)
                gamma_pred = {
                    (i, j): float(np.dot(
                        np.asarray(self.trained_models['gamma'][(i, j)]),
                        phi_array
                    ))
                    for i in self.data.C
                    for j in self.data.F
                }

                preds[sid] = {"pi": pi_pred, "gamma": gamma_pred}

        elif self.problem_type.upper() in ["SUMND", "SCMND"]:
            for sid in subs:
                phi = self.feature_vectors[sid]
                phi_array = np.asarray(phi)
                
                # Predict pi values for all node-commodity pairs
                pi_pred = {
                    (i, k): float(np.dot(np.asarray(self.trained_models['pi'][(i, k)]), phi_array))
                    for i in self.data.N
                    for k in self.data.K
                }
                
                # Predict alpha values for all arcs
                alpha_pred = {
                    idx: float(np.dot(np.asarray(self.trained_models['alpha'][idx]), phi_array))
                    for idx in range(len(self.data.A))
                }
                
                preds[sid] = {"pi": pi_pred, "alpha": alpha_pred}

        elif self.problem_type=='CMND':
            pass
        elif self.problem_type=='MCFL':
            pass
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")
        return preds


    @profiled("cuts_evaluation_generation")
    def compute_dual_obj(self, duals, sid, first_stage_sol):
        """
        Compute the dual objective value for a specific scenario given dual variables and first-stage solution.
        
        This method is essential for Benders decomposition as it calculates the dual subproblem objective
        value, which is compared against the current theta estimate to determine if an optimality cut
        should be added. The dual objective represents the "true" second-stage cost for the scenario
        given the current first-stage solution.
        
        Mathematical formulation varies by problem type:
        - UFL: λ - Σⱼ μⱼ * xⱼ (where λ = demand dual, μⱼ = facility capacity duals)
        - UHL: Σₖ vₖ * x_{i,k} + Σₗ uₗ * x_{j,l} (for OD pair (i,j))
        - SSLP: Σᵢ πᵢ * dᵢ - Σᵢⱼ γᵢⱼ * dᵢ * yⱼ (demand satisfaction duals)
        - SCFLP: Σᵢ λᵢ * dᵢ - Σᵢⱼ μᵢⱼ * dᵢ * yⱼ (capacity constraint duals)
        
        Args:
            duals (dict): Standardized dual variable dictionary from structure_duals():
                - UFL: {'lambda': float, 'mu': {j: float}}
                - UHL: {'v': {k: float}, 'u': {l: float}}
                - SSLP: {'pi': {i: float}, 'gamma': {(i,j): float}}
                - SCFLP: {'lambda': {i: float}, 'mu': {(i,j): float}}
            sid (int): Scenario index
            first_stage_sol (dict): Current first-stage variable values from Gurobi callback
                
        Returns:
            float: Dual objective value for the scenario. If this value > theta[scenario_key],
                   an optimality cut should be added to strengthen the master problem.
                   
        Purpose in Benders decomposition:
            1. **Cut generation criterion**: obj > theta ⟹ add optimality cut
            2. **Lower bound**: Provides true second-stage cost estimate
            3. **Convergence check**: When obj ≤ theta for all scenarios, optimality reached
            4. **ML validation**: Used to verify quality of predicted duals
            
        Used by:
            - Phase 1: Exact Benders cut generation decisions
            - Phase 2: ML prediction validation and cut decisions
            - Fallback methods: Exact dual verification
            - Cut violation detection across all solution methods
        """
        if self.problem_type=='UFL':
            lam = duals['lambda']; mu = duals['mu']
            return lam - sum(mu[j]*first_stage_sol[int(j)] for j in self.data.F)
        elif self.problem_type.upper() == "UHL":
            # Handle numpy arrays safely by converting to Python ints
            if isinstance(sid, np.integer):
                sid = int(sid)
                
            # Extract i,j from data.S safely
            if isinstance(self.data.S[sid], tuple):
                i, j = self.data.S[sid]
            elif isinstance(self.data.S[sid], np.ndarray):
                i, j = int(self.data.S[sid][0]), int(self.data.S[sid][1])
            else:
                i, j = self.data.S[sid], 0  # Fallback for simple case
                
            # Convert i, j to Python integers if they are NumPy types
            i = int(i) if isinstance(i, np.integer) else i
            j = int(j) if isinstance(j, np.integer) else j
            
            return (
                sum(duals['v'][int(k) if isinstance(k, np.integer) else k] * 
                    first_stage_sol[i, int(k) if isinstance(k, np.integer) else k] 
                    for k in self.data.N)
              + sum(duals['u'][int(l) if isinstance(l, np.integer) else l] * 
                    first_stage_sol[j, int(l) if isinstance(l, np.integer) else l] 
                    for l in self.data.N)
            )

        elif self.problem_type == "SSLP":
            s_name  = self.data.S[sid]
            demand  = self.data.demand[sid]      # Get row sid of demand matrix
            pi      = duals["pi"]
            gamma   = duals["gamma"]

            # πᵀd  −  Σ₍i,j₎ γ_{ij} d_i y_j
            obj = sum(
                pi[cust] * demand[idx]
                for idx, cust in enumerate(self.data.C)
            )

            obj -= sum(
                gamma[(cust, srv)] * demand[idx] * first_stage_sol[int(srv)]
                for idx, cust in enumerate(self.data.C)
                for srv in self.data.F
            )

            return obj
        elif self.problem_type.upper() == "SCFLP":
            # For SCFLP: duals = {"mu": mu_vals, "nu": nu_vals}
            # where mu_vals = {i: value} and nu_vals = {j: value}
            mu_vals = duals["mu"]    # Dual variables for demand constraints
            nu_vals = duals["nu"]    # Dual variables for capacity constraints
            
            # Validate that first_stage_sol contains all required facilities
            missing_facilities = [j for j in self.data.J if j not in first_stage_sol]
            if missing_facilities:
                raise KeyError(
                    f"first_stage_sol missing facilities: {missing_facilities[:10]}. "
                    f"Expected keys: {sorted(self.data.J)}, Got keys: {sorted(first_stage_sol.keys())}"
                )
            
            # Get demand for current scenario
            demand = self.data.scenarios[sid]
            
            # Determine which capacities to use based on use_stochastic_capacity parameter
            # CRITICAL FIX: Must match capacity selection logic in subproblem_dual.py
            if self.use_stochastic_capacity and hasattr(self.data, 'capacity_scenarios') and self.data.capacity_scenarios is not None:
                # Use scenario-specific capacities if available and enabled
                if sid < len(self.data.capacity_scenarios):
                    scenario_capacities = self.data.capacity_scenarios[sid]
                else:
                    # Fallback to original capacities if scenario index out of range
                    scenario_capacities = self.data.capacity
            else:
                # Use original capacities (consistent with use_stochastic_capacity=False)
                scenario_capacities = self.data.capacity
            
            # Compute dual objective: ∑_i μ_i * d_i - ∑_j ν_j * u_j * y_j
            # CRITICAL: Use the SAME index j for nu_vals[j], capacity[j], and first_stage_sol[j]
            first_term = sum(mu_vals[i] * demand[i] for i in self.data.I)
            
            # Second term: Use scenario-specific capacities (matching subproblem solver)
            second_term = sum(
                nu_vals[j] * scenario_capacities[j] * first_stage_sol[j]
                for j in self.data.J
            )
            
            return first_term - second_term
        elif self.problem_type.upper() in ["NESP", "NE-SP"]:
            # For NE-SP: duals = {"pi": pi_vals, "alpha": alpha_vals}
            # where pi_vals = {node: value} and alpha_vals = {arc_idx: value}
            pi_vals = duals["pi"]      # Node potentials
            alpha_vals = duals["alpha"]  # Arc multipliers
            
            # Get origin and destination for this OD pair
            o, t = self.data.M[sid]
            
            # Compute dual objective: π_o - π_t + Σ_{(i,j)∈A} x_ij * α_ij
            first_term = pi_vals[o] - pi_vals[t]
            
            # Second term: sum over activated arcs
            second_term = sum(
                first_stage_sol[idx] * alpha_vals[idx]
                for idx in range(len(self.data.A))
            )
            
            return first_term + second_term
        elif self.problem_type.upper() in ["SUMND", "SCMND"]:
            # For SUMND: duals = {"pi": {(i,k): value}, "alpha": {arc_idx: value}}
            pi_vals = duals["pi"]      # Node-commodity dual variables
            alpha_vals = duals["alpha"]  # Arc capacity dual variables
            
            # Get demand scenario
            demand_scenario = self.data.demand_scenarios[sid]
            
            # The manuscript SCMND formulation uses fixed capacities in both
            # its dual objective and cuts.  The preserved current formulation
            # uses scenario-specific capacities.
            if self.problem_type.upper() == "SCMND":
                capacity_scenario = scmnd_capacity_vector(self.data, sid)
            else:
                capacity_scenario = self.data.capacity_scenarios[sid]
            
            # Compute net supply for each node-commodity pair
            # d_i^{ks} = +d_k^s if i is origin of k, -d_k^s if i is destination of k, 0 otherwise
            first_term = 0.0
            for i in self.data.N:
                for k in self.data.K:
                    origin = self.data.origins[k]
                    dest = self.data.destinations[k]
                    if i == origin:
                        d_node_commodity = demand_scenario[k]
                    elif i == dest:
                        d_node_commodity = -demand_scenario[k]
                    else:
                        d_node_commodity = 0.0
                    first_term += pi_vals[(i, k)] * d_node_commodity
            
            # Compute dual objective: Σ_k Σ_i d_i^{ks} * π_i^k - Σ_{(i,j)∈A} u_ij^s * y_ij * α_ij^s
            second_term = sum(
                capacity_scenario[idx] * first_stage_sol[idx] * alpha_vals[idx]
                for idx in range(len(self.data.A))
            )
            
            return first_term - second_term
        elif self.problem_type.upper() == "CMND":
            pass
        elif self.problem_type.upper() == "MCFL":
            pass
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    def add_optimality_cut(self, mod, duals, sid, integer_solution=True, *, cut_origin="exact"):
        """Time construction by dual origin, excluding the nested insertion.

        Exact includes Phase 0/1, selected, fallback solves and exact switches.
        ML includes predictions and transfers, including Random and KNN fallback.
        The violation test stays at the caller; this does not change any cut.
        """
        if cut_origin not in ("exact", "ml"):
            raise ValueError("Unknown cut origin: " + str(cut_origin))
        with self.exclusive_profiler.measure("cuts_evaluation_generation", detail=cut_origin):
            result = self._add_optimality_cut_impl(mod, duals, sid, integer_solution)
        self._submitted_cut_counts[cut_origin + ("_lazy" if integer_solution else "_user")] += 1
        return result

    def _add_optimality_cut_impl(self, mod, duals, sid, integer_solution=True):
        if self.problem_type=='UFL':
            lam = duals['lambda']; mu = duals['mu']
            rhs = lam - quicksum(mu[j]*self.first_stage_values[j]
                                 for j in self.data.F)
        elif self.problem_type.upper() == "UHL":
            # Handle numpy arrays safely by converting to Python ints
            if isinstance(sid, np.integer):
                sid = int(sid)
                
            # Extract i,j from data.S safely
            if isinstance(self.data.S[sid], tuple):
                i, j = self.data.S[sid]
            elif isinstance(self.data.S[sid], np.ndarray):
                i, j = int(self.data.S[sid][0]), int(self.data.S[sid][1])
            else:
                i, j = self.data.S[sid], 0  # Fallback for simple case
                
            # Convert i, j to Python integers if they are NumPy types
            i = int(i) if isinstance(i, np.integer) else i
            j = int(j) if isinstance(j, np.integer) else j
            
            rhs = (
                quicksum(duals['v'][int(k) if isinstance(k, np.integer) else k]*
                         self.first_stage_values[i, int(k) if isinstance(k, np.integer) else k]
                         for k in self.data.N)
              + quicksum(duals['u'][int(l) if isinstance(l, np.integer) else l]*
                         self.first_stage_values[j, int(l) if isinstance(l, np.integer) else l]
                         for l in self.data.N)
            )

        elif self.problem_type == "SSLP":
            s_name  = self.data.S[sid]
            demand  = self.data.demand[sid]
            pi      = duals["pi"]
            gamma   = duals["gamma"]

            rhs = (
                quicksum(pi[i] * demand[idx] for idx, i in enumerate(self.data.C))
                - quicksum(
                gamma[(i, j)] * demand[idx] * self.first_stage_values[j]
                for idx, i in enumerate(self.data.C)
                for j   in self.data.F
                )
            )

        elif self.problem_type.upper() == "SCFLP":
            mu_vals = duals["mu"]    # Dual variables for demand constraints
            nu_vals = duals["nu"]    # Dual variables for capacity constraints
            
            # Get demand for current scenario
            demand = self.data.scenarios[sid]
            
            # Determine which capacities to use based on use_stochastic_capacity parameter
            if self.use_stochastic_capacity and hasattr(self.data, 'capacity_scenarios') and self.data.capacity_scenarios is not None:
                # Use scenario-specific capacities if available and enabled
                if sid < len(self.data.capacity_scenarios):
                    scenario_capacities = self.data.capacity_scenarios[sid]
                else:
                    # Fallback to original capacities if scenario index out of range
                    scenario_capacities = self.data.capacity
            else:
                # Use original capacities (static behavior)
                scenario_capacities = self.data.capacity
            
            # Build the cut: θ ≥ ∑_i μ_i * d_i - ∑_j ν_j * u_j * y_j
            rhs = (
                quicksum(mu_vals[i] * demand[i] for i in self.data.I) -
                quicksum(nu_vals[j] * scenario_capacities[j] * self.first_stage_values[j] 
                        for j in self.data.J)
            )
        elif self.problem_type.upper() in ["NESP", "NE-SP"]:
            pi_vals = duals["pi"]      # Node potentials
            alpha_vals = duals["alpha"]  # Arc multipliers
            
            # Get origin and destination for this OD pair
            o, t = self.data.M[sid]
            
            # Build the cut: θ_m ≥ π_o - π_t + Σ_{(i,j)∈A} x_ij * α_ij
            rhs = (
                pi_vals[o] - pi_vals[t] +
                quicksum(alpha_vals[idx] * self.first_stage_values[idx]
                        for idx in range(len(self.data.A)))
            )
        elif self.problem_type.upper() in ["SUMND", "SCMND"]:
            pi_vals = duals["pi"]      # Node-commodity dual variables π_i^k
            alpha_vals = duals["alpha"]  # Arc capacity dual variables α_ij^s
            
            # Get demand scenario for this scenario
            demand_scenario = self.data.demand_scenarios[sid]
            
            if self.problem_type.upper() == "SCMND":
                capacity_scenario = scmnd_capacity_vector(self.data, sid)
            else:
                capacity_scenario = self.data.capacity_scenarios[sid]
            
            # Compute net supply for each node-commodity pair
            # d_i^{ks} = +d_k^s if i is origin of k, -d_k^s if i is destination of k, 0 otherwise
            d_node_commodity = {}
            for i in self.data.N:
                for k in self.data.K:
                    origin = self.data.origins[k]
                    dest = self.data.destinations[k]
                    if i == origin:
                        d_node_commodity[(i, k)] = demand_scenario[k]
                    elif i == dest:
                        d_node_commodity[(i, k)] = -demand_scenario[k]
                    else:
                        d_node_commodity[(i, k)] = 0.0
            
            # Build the cut: θ_s ≥ Σ_k Σ_i d_i^{ks} * π_i^k - Σ_{(i,j)∈A} u_ij^s * α_ij^s * y_ij
            rhs = (
                quicksum(d_node_commodity[(i, k)] * pi_vals[(i, k)]
                        for i in self.data.N for k in self.data.K) -
                quicksum(capacity_scenario[idx] * alpha_vals[idx] * self.first_stage_values[idx]
                        for idx in range(len(self.data.A)))
            )
        elif self.problem_type.upper() == "CMND":
            pass
        elif self.problem_type.upper() == "MCFL":
            pass
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")
        if integer_solution:
            self.exclusive_profiler.call("cuts_insertion", mod.cbLazy, self.theta[self.theta_keys[sid]] >= rhs)
        else:
            self.exclusive_profiler.call("cuts_insertion", mod.cbCut, self.theta[self.theta_keys[sid]] >= rhs)

    def structure_duals(self, result):
        """
        Transform raw dual solver results into standardized dual variable dictionaries.
        
        This method is necessary because different problem types return dual solutions in
        different formats from solve_dual_subproblem(). We need a unified interface to
        access dual variables consistently across all problem types for:
        - Feature vector generation (dual_vectors storage)
        - Dual objective computation (compute_dual_obj)
        - Optimality cut generation (add_optimality_cut)
        - ML prediction training and inference
        
        Args:
            result (tuple): Raw result from solve_dual_subproblem(), format varies by problem:
                - UFL: (obj_val, {'lambda': float}, {j: mu_val})
                - UHL: (obj_val, {k: v_val}, {l: u_val})
                - SSLP: (obj_val, {i: pi_val}, {(i,j): gamma_val})
                - SCFLP: (obj_val, {i: mu_val}, {j: nu_val})
                
        Returns:
            dict: Standardized dual variable dictionary with problem-specific structure:
                - UFL: {'lambda': float, 'mu': {j: float}}
                - UHL: {'v': {k: float}, 'u': {l: float}}
                - SSLP: {'pi': {i: float}, 'gamma': {(i,j): float}}
                - SCFLP: {'mu': {i: float}, 'nu': {j: float}}
                
        Why this method exists:
            Different optimization solvers and problem formulations return dual solutions
            in varying formats. This standardization ensures that downstream methods
            (dual prediction, cut generation, feature engineering) can work uniformly
            across all problem types without knowing the specific solver output format.
            
        Used by:
            - Phase 1: Store duals for training (self.dual_vectors)
            - Phase 2: Cache selected subproblem duals (dual_cache)
            - ML training: Structure training data consistently
            - Cut generation: Provide standardized dual access
        """
        if self.problem_type.upper() == 'UFL':
            # result is (obj_val, {'lambda': float}, {j: float})
            _, primary, mu = result
            lam_val = primary['lambda']
            return {'lambda': lam_val, 'mu': mu}
        elif self.problem_type.upper() == "UHL":
            _, v_vals, u_vals = result
            return {'v': v_vals, 'u': u_vals}
        elif self.problem_type == "SSLP":
            _, pi_vals, gamma_vals = result   # result came from solve_dual_subproblem()
            return {"pi": pi_vals, "gamma": gamma_vals}
        elif self.problem_type.upper() == "SCFLP":
            _, mu_vals, nu_vals = result
            return {"mu": mu_vals, "nu": nu_vals}
        elif self.problem_type.upper() in ["NESP", "NE-SP"]:
            # result is (obj_val, {node: pi_val}, {arc_idx: alpha_val})
            _, pi_vals, alpha_vals = result
            return {"pi": pi_vals, "alpha": alpha_vals}
        elif self.problem_type.upper() in ["SUMND", "SCMND"]:
            # result is (obj_val, {(i,k): pi_val}, {arc_idx: alpha_val})
            _, pi_vals, alpha_vals = result
            return {"pi": pi_vals, "alpha": alpha_vals}
        elif self.problem_type.upper() == "CMND":
            pass   
        elif self.problem_type.upper() == "MCFL":
            pass
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    def compute_knn_neighbors(self):
        """
        [LEGACY METHOD - Use DualPredictor with method='KNN' for new code]
        
        Compute k-nearest neighbors for unselected subproblems based on feature vector similarity.
        
        **Note**: This method is maintained for backward compatibility. New code should use
        the unified DualPredictor class which handles KNN neighbor computation during training.
        
        This method implements a k-nearest neighbor (KNN) mapping between unselected and selected
        subproblems using Euclidean distance in the feature space. For each unselected subproblem,
        it identifies the k closest selected subproblems based on their feature vectors. This 
        mapping is essential for dual variable prediction when using KNN-based methods.
        
        Algorithm:
        ----------
        1. Identify unselected subproblems (all scenarios minus selected ones)
        2. For each unselected subproblem u:
           a. Compute Euclidean distances to all selected subproblems: ||φ_u - φ_s||₂
           b. Sort selected subproblems by distance (closest first)
           c. Store the k nearest neighbors (indices of selected subproblems)
        
        Mathematical Foundation:
        ----------------------
        For unselected subproblem u and selected subproblem s:
        - Feature vectors: φ_u, φ_s ∈ ℝᵈ (d-dimensional feature space)
        - Distance metric: d(u,s) = ||φ_u - φ_s||₂ = √(Σᵢ(φ_u,i - φ_s,i)²)
        - Neighborhood: N_k(u) = {s₁, s₂, ..., sₖ} where d(u,s₁) ≤ d(u,s₂) ≤ ... ≤ d(u,sₖ)
        
        Usage Context:
        -------------
        This method is called during Phase 2 (ML-aided Benders) initialization, specifically:
        1. **Condition**: Only when prediction_method='KNN' OR fallback_method='KNN'
        2. **Timing**: After subproblem selection but before dual prediction
        3. **Frequency**: Once per callback session (features_computed=True prevents recomputation)
        
        The computed neighbors are used in two scenarios:
        
        **Primary Prediction (prediction_method='KNN')**:
        - Used by knn_predict() to predict dual variables for unselected subproblems
        - Averages dual values from k nearest selected neighbors
        - Applied to all unselected subproblems when KNN is the primary prediction method
        
        **Fallback Prediction (fallback_method='KNN')**:
        - Used by _handle_fallback() when regression prediction fails
        - Provides backup dual estimation without solving expensive subproblems
        - Applied only when primary prediction (REG) fails and fallback='KNN'
        
        Returns:
        -------
        dict: Mapping from unselected subproblem indices to lists of k nearest neighbor indices
            Format: {unselected_sid: [neighbor1_sid, neighbor2_sid, ..., neighborK_sid]}
            Where neighbor indices are sorted by increasing distance
            
        Example:
        -------
        If scenarios 0,1,2 are selected and scenarios 3,4,5 are unselected with k=2:
        {3: [0, 1],  # scenario 3's closest neighbors are scenarios 0 and 1
         4: [1, 2],  # scenario 4's closest neighbors are scenarios 1 and 2  
         5: [0, 2]}  # scenario 5's closest neighbors are scenarios 0 and 2
         
        Performance Considerations:
        -------------------------
        - Time complexity: O(|unselected| × |selected| × d) where d is feature dimension
        - Space complexity: O(|unselected| × k) for storing neighbor mappings
        - Distance computation is the bottleneck for high-dimensional features
        - Pre-computed once and reused across all callback iterations in Phase 2
        
        Integration with ML Pipeline:
        ---------------------------
        1. **Feature Engineering**: Requires self.feature_vectors from compute_scenario_features_from_duals()
        2. **Subproblem Selection**: Requires self.selected_subproblems from select_subproblems_based_on_features()
        3. **Dual Prediction**: Feeds into knn_predict() for dual variable estimation
        4. **Cut Generation**: Enables approximate optimality cuts without exact subproblem solving
        """
        n_scen = self.n_subproblems
        all_idxs = set(range(n_scen))
        unselected = all_idxs - self.selected_subproblems
        sel = list(self.selected_subproblems)
        feats = self.feature_vectors
        knn = {}
        for u in unselected:
            dists = [(s, np.linalg.norm(feats[u]-feats[s])) for s in sel]
            dists.sort(key=lambda x: x[1])
            knn[u] = [s for s,_ in dists[:self.n_neighbors]]
        return knn

    def knn_predict(self, subs, dual_cache):
        """
        [LEGACY METHOD - Use DualPredictor.predict() with method='KNN' for new code]
        
        Predict dual variables for unselected subproblems using k-nearest neighbor averaging.
        
        **Note**: This method is maintained for backward compatibility. New code should use
        the unified DualPredictor class which handles KNN prediction consistently.
        
        This method implements a non-parametric machine learning approach for dual variable prediction
        in the learning-augmented Benders decomposition algorithm. It leverages the similarity of
        subproblems in feature space to predict dual variables by averaging the exact dual solutions
        from the k most similar selected subproblems, providing a fast and intuitive alternative to
        parametric regression models.
        
        Mathematical Foundation:
        ----------------------
        **K-Nearest Neighbor Prediction**: π̂ₛ = (1/k) · Σᵢ₌₁ᵏ πₛᵢ
        
        Where:
        - π̂ₛ = predicted dual variable for unselected subproblem s
        - k = number of nearest neighbors (self.n_neighbors)
        - sᵢ = i-th nearest neighbor from selected subproblems
        - πₛᵢ = exact dual variable from neighbor sᵢ (from dual_cache)
        - Averaging is performed component-wise for each dual variable type
        
        **Neighborhood Selection**: Based on pre-computed self.knn_neighbors mapping:
        - Feature space similarity: ||φₛ - φₛᵢ||₂ minimized
        - Neighbors sorted by increasing Euclidean distance
        - Computed once by compute_knn_neighbors() for efficiency
        
        **Problem-Specific Averaging**:
        
        *UFL (Uncapacitated Facility Location)*:
        - λ̂ₛ = (1/k) · Σᵢ λₛᵢ (customer demand dual)
        - μ̂ⱼₛ = (1/k) · Σᵢ μⱼₛᵢ ∀j ∈ F (facility capacity duals)
        
        *UHL (Uncapacitated Hub Location)*:
        - v̂ₖₛ = (1/k) · Σᵢ vₖₛᵢ ∀k ∈ N (origin node duals)
        - ûₗₛ = (1/k) · Σᵢ uₗₛᵢ ∀l ∈ N (destination node duals)
        
        *SSLP (Server Location Problem)*:
        - π̂ᵢₛ = (1/k) · Σₙ πᵢₛₙ ∀i ∈ C (customer demand duals)
        - γ̂ᵢⱼₛ = (1/k) · Σₙ γᵢⱼₛₙ ∀(i,j) ∈ C×F (service assignment duals)
        
        *SCFLP (Single-Source Capacitated Facility Location)*:
        - μ̂ᵢₛ = (1/k) · Σₙ μᵢₛₙ ∀i ∈ I (customer demand duals)
        - ν̂ⱼₛ = (1/k) · Σₙ νⱼₛₙ ∀j ∈ J (facility capacity duals)
        
        Algorithm Flow:
        -------------
        1. **Neighbor Retrieval**: Extract k nearest neighbors for each unselected subproblem
        2. **Dual Collection**: Gather exact dual values from neighbor cache entries
        3. **Component-wise Averaging**: Compute arithmetic mean for each dual variable
        4. **Structure Assembly**: Organize averaged predictions into standardized format
        5. **Type Conversion**: Ensure numerical stability with explicit float conversion
        
        Advantages of KNN Approach:
        -------------------------
        **Simplicity**:
        - No training phase required (non-parametric)
        - Intuitive averaging of similar solutions
        - Robust to outliers when k > 1
        
        **Speed**:
        - O(k) dual lookups per subproblem (very fast)
        - Pre-computed neighbor mappings eliminate distance calculations
        - Linear scaling with number of unselected subproblems
        
        **Reliability**:
        - Always produces valid predictions (no training failures)
        - Quality gracefully degrades with poor neighbor similarity
        - Naturally handles different problem structures
        
        **Local Adaptivity**:
        - Adapts predictions to local feature space regions
        - Captures non-linear relationships through locality
        - Respects problem-specific dual correlations
        
        Integration with Learning Framework:
        ----------------------------------
        **Primary Prediction Mode (prediction_method='KNN')**:
        - Direct alternative to regression-based predict_duals()
        - Applied to all unselected subproblems in Phase 2
        - Uses self.knn_neighbors computed during initialization
        
        **Fallback Prediction Mode (fallback_method='KNN')**:
        - Backup when regression training fails in _handle_fallback()
        - Provides robust dual estimation without expensive exact solving
        - Enables graceful degradation when parametric methods fail
        
        **Mixed Strategy Support**:
        - Can complement regression for specific subproblem subsets
        - Useful for scenarios with insufficient training data
        - Provides diversity in prediction ensemble approaches
        
        Args:
        ----
        subs (list/set): Collection of unselected subproblem indices requiring prediction
            - UFL/SCFLP/SSLP: Integer scenario indices [0, 1, 2, ...]
            - UHL: Origin-destination pair indices referencing data.S tuples
            
        dual_cache (dict): Cache of exact dual solutions from selected subproblems
            Format: {selected_sid: structured_dual_dict}
            - Populated during exact solving of selected subproblems
            - Contains standardized dual dictionaries from structure_duals()
            - Must include all neighbors referenced in self.knn_neighbors
            
        Returns:
        -------
        dict: Nested dictionary mapping subproblem indices to predicted dual structures
            Format: {subproblem_id: averaged_dual_dict}
            
            **UFL Structure**:
            {sid: {'lambda': float, 'mu': {j: float ∀j ∈ F}}}
            
            **UHL Structure**:
            {ij: {'v': {k: float ∀k ∈ N}, 'u': {l: float ∀l ∈ N}}}
            
            **SSLP Structure**:
            {sid: {'pi': {i: float ∀i ∈ C}, 'gamma': {(i,j): float ∀(i,j) ∈ C×F}}}
            
            **SCFLP Structure**:
            {sid: {'mu': {i: float ∀i ∈ I}, 'nu': {j: float ∀j ∈ J}}}
        
        Quality Considerations:
        ---------------------
        **Prediction Accuracy**:
        - Dependent on feature space quality and neighbor similarity
        - Better performance with higher k (smoother averaging)
        - Sensitive to distribution of selected subproblems
        
        **Dual Feasibility**:
        - Averaging preserves dual feasibility if neighbors are feasible
        - Convex combination of feasible solutions remains feasible
        - Natural regularization through neighbor diversity
        
        **Cut Quality**:
        - Produces valid optimality cuts for Benders decomposition
        - Cut strength may be weaker than exact duals
        - Still provides valuable problem strengthening
        
        Performance Characteristics:
        --------------------------
        **Time Complexity**: O(|subs| × k × d) where:
        - |subs| = number of unselected subproblems
        - k = number of neighbors (typically small: 3-10)
        - d = average number of dual variables per subproblem
        
        **Space Complexity**: O(|subs| × d) for storing predictions
        
        **Computational Benefits**:
        - **vs. Exact Solving**: 100-1000x faster than solve_dual_subproblem()
        - **vs. Regression**: Similar speed, no training overhead
        - **Memory Efficient**: Reuses cached dual solutions
        
        Usage Context:
        -------------
        **Called from handle_callback() when**:
        1. **Primary KNN Mode**: prediction_method='KNN'
        2. **Fallback from _handle_fallback()**: when fallback_method='KNN'
        3. **Mixed Strategies**: Individual subproblem prediction needs
        
        **Prerequisites**:
        1. **Neighbor Mapping**: self.knn_neighbors populated by compute_knn_neighbors()
        2. **Dual Cache**: dual_cache contains solutions for all required neighbors
        3. **Feature Vectors**: Valid feature space for similarity computation
        
        Error Handling:
        --------------
        **Robust Design**:
        - No explicit error checking (assumes valid inputs)
        - Relies on dual_cache completeness for neighbor access
        - Numpy mean() handles numerical stability automatically
        
        **Graceful Degradation**:
        - Poor neighbors still produce valid (if suboptimal) predictions
        - Missing cache entries would raise KeyError for debugging
        - Float conversion ensures numerical consistency
        
        Example Prediction Process:
        -------------------------
        ```python
        # For UFL with scenario 5 having neighbors [1, 3, 7]:
        subs = [5]
        neigh = self.knn_neighbors[5]  # [1, 3, 7]
        
        # Collect lambda values from neighbors
        lam_list = [dual_cache[1]['lambda'], dual_cache[3]['lambda'], dual_cache[7]['lambda']]
        # lam_list = [12.5, 15.2, 11.8]
        
        # Average: lam = (12.5 + 15.2 + 11.8) / 3 = 13.17
        lam = float(np.mean(lam_list))
        
        # Similar averaging for mu values per facility
        ```
        
        Design Philosophy:
        ----------------
        **Complementary to Regression**:
        - Provides non-parametric alternative to linear models
        - Captures local patterns that global models might miss
        - Offers robust fallback when training data is insufficient
        
        **Simplicity and Reliability**:
        - Easy to understand and debug prediction logic
        - No hyperparameter tuning beyond neighbor count
        - Consistent behavior across different problem instances
        
        **Integration with Benders**:
        - Maintains compatibility with exact Benders framework
        - Preserves dual interpretation and cut generation
        - Enables hybrid exact/approximate decomposition strategies
        """
        warnings.warn(
            "knn_predict is deprecated. Use DualPredictor with method='KNN' for new code. "
            "This method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        preds = {}
        if self.problem_type=='UFL':
            for sid in subs:
                neigh = self.knn_neighbors[sid]
                lam_list = [dual_cache[n]['lambda'] for n in neigh]
                mu_list_list = [dual_cache[n]['mu'] for n in neigh]
                lam = float(np.mean(lam_list))
                mu_pred = {j: float(np.mean([m[j] for m in mu_list_list]))
                           for j in self.data.F}
                preds[sid] = {'lambda': lam, 'mu': mu_pred}
        elif self.problem_type.upper() == "UHL":
            for sid in subs:
                neigh = self.knn_neighbors[sid]
                v_preds = {k: float(np.mean([dual_cache[n]['v'][k] for n in neigh]))
                           for k in self.data.N}
                u_preds = {l: float(np.mean([dual_cache[n]['u'][l] for n in neigh]))
                           for l in self.data.N}
                preds[sid] = {'v': v_preds, 'u': u_preds}
        
        elif self.problem_type.upper() == "SSLP":
            for sid in subs:
                neigh = self.knn_neighbors[sid]

                # ── π̂  : average π_i over the k nearest neighbours
                pi_pred = {
                    i: float(np.mean([dual_cache[n]['pi'][i] for n in neigh]))
                    for i in self.data.C
                }

                # ── γ̂ : average γ_{ij} over neighbours, for every (i,j) pair
                gamma_pred = {
                    (i, j): float(np.mean([dual_cache[n]['gamma'][(i, j)] for n in neigh]))
                    for i in self.data.C
                    for j in self.data.F
                }

                preds[sid] = {'pi': pi_pred, 'gamma': gamma_pred}

        elif self.problem_type.upper() == "SCFLP":
            for sid in subs:
                neigh = self.knn_neighbors[sid]
                
                # Average mu values over neighbors for all customers
                mu_pred = {i: float(np.mean([dual_cache[n]['mu'][i] for n in neigh]))
                          for i in self.data.I}
                
                # Average nu values over neighbors for all facilities
                nu_pred = {j: float(np.mean([dual_cache[n]['nu'][j] for n in neigh]))
                          for j in self.data.J}
                
                preds[sid] = {'mu': mu_pred, 'nu': nu_pred}

        elif self.problem_type.upper() in ["SUMND", "SCMND"]:
            for sid in subs:
                neigh = self.knn_neighbors[sid]
                
                # Average pi values over neighbors for all node-commodity pairs
                pi_pred = {
                    (i, k): float(np.mean([dual_cache[n]['pi'][(i, k)] for n in neigh]))
                    for i in self.data.N
                    for k in self.data.K
                }
                
                # Average alpha values over neighbors for all arcs
                alpha_pred = {
                    idx: float(np.mean([dual_cache[n]['alpha'][idx] for n in neigh]))
                    for idx in range(len(self.data.A))
                }
                
                preds[sid] = {'pi': pi_pred, 'alpha': alpha_pred}

        elif self.problem_type.upper() == "CMND":
            pass
        elif self.problem_type.upper() == "MCFL":
            pass
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")
        return preds

    def tree_predict(self, subs, dual_cache):
        """
        [LEGACY METHOD - Use DualPredictor.predict() with method='TREE' for new code]
        
        Predict dual variables for unselected subproblems using tree-based fixed mappings with uniform averaging.
        
        **Note**: This method is maintained for backward compatibility. New code should use
        the unified DualPredictor class which handles TREE prediction consistently.
        
        This method implements the tree-based prediction approach using fixed mappings from unselected to 
        selected scenarios, similar to KNN but with tree-optimized groupings and uniform weights.
        
        Mathematical Foundation:
        ----------------------
        **Tree-Based Fixed Mapping Prediction**: 
        For unselected scenario s with mapping to selected scenarios J_s:
        π̂_s = (1/|J_s|) * Σ_{j∈J_s} π_j
        
        Where:
        - π̂_s = predicted dual variable for unselected scenario s
        - J_s = fixed set of selected scenarios mapped to s (determined during training)
        - π_j = current dual solution for selected scenario j (from dual_cache)
        - Weights are uniform: w_j = 1/|J_s| for all j ∈ J_s
        
        **Fixed Mapping Algorithm**:
        1. **Retrieve Mappings**: Use pre-computed mappings from training phase
        2. **Uniform Averaging**: For each unselected scenario, average over its mapped selected scenarios
        3. **Component-wise Prediction**: Apply same mapping to all dual variable components  
        4. **Structure Assembly**: Organize predictions into standardized dual dictionaries
        
        **Key Properties**:
        - **Consistent with KNN**: Fixed neighbor assignments, uniform weights
        - **Tree-optimized groupings**: Mappings determined by decision tree partitioning
        - **Efficient**: No tree traversal needed in Phase 2, just lookup and averaging
        - **Variable group sizes**: Different unselected scenarios may have different numbers of neighbors
        
        Args:
        ----
        subs (list/set): Collection of unselected subproblem indices requiring prediction
        dual_cache (dict): Cache of exact dual solutions from selected subproblems
                          Format: {selected_sid: structured_dual_dict}
                          
        Returns:
        -------
        dict: Nested dictionary mapping subproblem indices to predicted dual structures
              Format identical to other prediction methods (KNN, REG) for seamless integration
              
        Example:
        -------
        ```
        # For unselected scenario 12 mapped to selected scenarios [1, 2, 4]:
        preds[12]['lambda'] = (dual_cache[1]['lambda'] + dual_cache[2]['lambda'] + dual_cache[4]['lambda']) / 3
        preds[12]['mu'][j] = (dual_cache[1]['mu'][j] + dual_cache[2]['mu'][j] + dual_cache[4]['mu'][j]) / 3
        ```
              
        Raises:
        ------
        ValueError: If tree mappings are not available or incompatible format
        """
        warnings.warn(
            "tree_predict is deprecated. Use DualPredictor with method='TREE' for new code. "
            "This method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Delegate to unified predictor if available and trained
        if hasattr(self, 'dual_predictor') and self.dual_predictor.is_trained and self.dual_predictor.predictor_type == 'TREE':
            return self.exclusive_profiler.call(
            "callback_other_logic" if is_parametric_predictor(self.predictor_type)
            else "nonparametric_dual_transfer",
            self.dual_predictor.predict, subs, dual_cache
        )
        else:
            raise ValueError("TREE predictor not available or not trained. Use unified training approach.")


    def scenario_key(self, sid):
        """
        Generate the appropriate key for accessing theta variables based on problem type.
        
        This method converts scenario indices (sid) into the proper keys used to index
        the theta dictionary (second-stage cost variables) in the master problem.
        The key format varies by problem type due to different problem structures.
        
        Args:
            sid (int): Scenario index (0 to n_scenarios-1)
            
        Returns:
            Key for accessing theta[key] in the master problem:
            - UFL: Returns sid directly (int)
            - UHL: Returns tuple(data.S[sid]) for origin-destination pairs
            - SSLP/SCFLP: Returns sid directly (int)
            
        Used in:
            - self.theta_keys mapping creation in handle_callback()
            - Accessing theta values: theta_val[self.scenario_key(sid)]
            - Adding optimality cuts: self.theta[self.theta_keys[sid]]
            - Dual objective violation checks
            
        Problem-specific behavior:
            - UFL: sid maps directly to customer scenarios
            - UHL: sid indexes into data.S to get (i,j) OD pair tuples
            - SSLP: sid maps to scenario names/demand vectors
            - SCFLP: sid maps to demand scenario indices
        """
        if self.problem_type == 'UFL':
            return sid
        elif self.problem_type == 'UHL':
            # For UHL, data.S[sid] is already a tuple (i,j), so we return tuple(data.S[sid])
            return tuple(self.data.S[sid])
        
        elif self.problem_type == "SSLP": 
            return sid
        else:
            return sid
    
    # ===== TASK 1: Point signature tracking methods =====
    
    def _compute_point_signature(self, first_stage_sol):
        """
        Compute a numerically-robust hash signature for a first-stage solution point.
        
        Uses rounding to 6 decimal places to avoid floating-point noise while maintaining
        sufficient precision for practical distinction between solutions.
        
        Args:
            first_stage_sol (dict): First-stage solution values {variable_id: value}
            
        Returns:
            str: Hash signature of the rounded solution point
        """
        # Sort by key for deterministic ordering
        sorted_items = sorted(first_stage_sol.items(), key=lambda x: str(x[0]))
        # Round to 6 decimal places to handle numerical noise
        rounded_vals = tuple(round(float(v), 6) for k, v in sorted_items)
        return str(hash(rounded_vals))
    
    def _track_point_signature(self, first_stage_sol, cuts_added, iteration):
        """
        Track first-stage solution point signatures to detect repeated visits.
        
        Updates internal tracking structures to count:
        - Total unique points seen
        - Repeated callback hits for same point
        - Repeated hits that generated new cuts
        
        Args:
            first_stage_sol (dict): First-stage solution values
            cuts_added (int): Number of cuts added in this callback
            iteration (int): Current iteration number
        """
        signature = self._compute_point_signature(first_stage_sol)
        
        # DEBUG: Log tracking activity
        logger.debug(f"_track_point_signature called: iter={iteration}, cuts={cuts_added}, sig={signature[:20]}...")
        
        if signature not in self.first_stage_signatures:
            # New unique point
            self.first_stage_signatures[signature] = {
                'count': 1,
                'iterations': [iteration],
                'cuts_added': [cuts_added]
            }
            self.total_unique_points += 1
            # logger.debug(f"New unique point detected. Total unique: {self.total_unique_points}")
        else:
            # Repeated point
            self.first_stage_signatures[signature]['count'] += 1
            self.first_stage_signatures[signature]['iterations'].append(iteration)
            self.first_stage_signatures[signature]['cuts_added'].append(cuts_added)
            self.total_repeated_hits += 1
            # logger.debug(f"Repeated point detected. Total repeated: {self.total_repeated_hits}")
            
            if cuts_added > 0:
                self.repeated_hits_with_cuts += 1
                # logger.debug(f"Repeated hit WITH cuts. Total: {self.repeated_hits_with_cuts}")
    
    def get_callback_diagnostics(self):
        """
        Return comprehensive diagnostic statistics for callback behavior analysis.
        
        Returns:
            dict: Diagnostic statistics including:
                - Task 1: Point signature and repeated callback statistics
                - Task 2: Root/non-root callback counts
        """
        # Analyze repeated hits with cuts
        repeated_points_with_cuts = []
        for sig, info in self.first_stage_signatures.items():
            if info['count'] > 1:
                # Check if any repeated visit added cuts
                cuts_on_repeats = [c for c in info['cuts_added'][1:] if c > 0]
                if cuts_on_repeats:
                    repeated_points_with_cuts.append({
                        'signature': sig,
                        'total_visits': info['count'],
                        'iterations': info['iterations'],
                        'cuts_added': info['cuts_added'],
                        'cuts_on_repeats': cuts_on_repeats
                    })
        
        return {
            # Task 1: Point signature tracking
            'point_signature_stats': {
                'total_unique_points': self.total_unique_points,
                'total_repeated_hits': self.total_repeated_hits,
                'repeated_hits_with_cuts': self.repeated_hits_with_cuts,
                'repeated_points_examples': repeated_points_with_cuts[:10]  # First 10 examples
            },
            # Task 2: Root vs non-root callback tracking
            'callback_location_stats': {
                'mipsol_root': self.callback_counts['mipsol_root'],
                'mipsol_nonroot': self.callback_counts['mipsol_nonroot'],
                'mipnode_root': self.callback_counts['mipnode_root'],
                'mipnode_nonroot': self.callback_counts['mipnode_nonroot'],
                'total_mipsol': self.callback_counts['mipsol_root'] + self.callback_counts['mipsol_nonroot'],
                'total_mipnode': self.callback_counts['mipnode_root'] + self.callback_counts['mipnode_nonroot']
            },
            # Branch-and-bound context
            'bnb_stats': {
                'total_nodes_explored': self.total_nodes_explored
            }
        }


    def _get_trial_cut_counts(self):
        """
        Calculate total cut counts from trial iterations for trial-cut-freq selection method.
        
        This method aggregates the number of cuts added for each subproblem during the
        trial phase (classic Benders iterations) to identify the most active subproblems.
        
        Returns:
            dict: Dictionary {subproblem_id: total_cut_count} for all scenarios
        """
        n_scen = self.n_subproblems
        trial_cut_counts = {}
        
        for sid in range(n_scen):
            # Sum cuts from both MIP and relaxation solutions during trial phase
            total_cuts = (self.num_cuts_mip_trial[sid] + 
                         self.num_cuts_rel_trial[sid])
            trial_cut_counts[sid] = total_cuts
        
        return trial_cut_counts

    @property
    def num_cuts_mip_unselected(self):
        return {sid: (
            self.num_cuts_mip_ml[sid] +
            self.num_cuts_mip_fallback[sid] +
            self.num_cuts_mip_exact_switch[sid]
        ) for sid in range(self.n_subproblems)}

    @property
    def num_cuts_rel_unselected(self):
        return {sid: (
            self.num_cuts_rel_ml[sid] +
            self.num_cuts_rel_fallback[sid] +
            self.num_cuts_rel_exact_switch[sid]
        ) for sid in range(self.n_subproblems)}
