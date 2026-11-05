"""
config.py

This module serves as the central configuration hub for the ML-augmented Benders decomposition
framework focused exclusively on SCFLP (Stochastic Capacitated Facility Location) problems.
It defines all parameters, settings, and experimental configurations used throughout the project.

Overview:
---------
The configuration system supports both single-value parameters for individual runs and
grid-based parameters for comprehensive experimental evaluation. This dual approach allows
for backward compatibility while enabling large-scale parameter sweeps for research purposes.

Current Focus: SCFLP with Restructured Method Architecture
----------------------------------------------------------
**As of February 2026, this implementation restructures the ML-augmented Benders framework** with clear separation between:
- **Method Categories**: Exact vs ML
- **ML Method Types**: ML_Augmented_BD, ML_Augmented_BD_with_duals, Covariate_Similarity, Uncertainty_Similarity, Random_Sampling
- **Predictor Types**: KNN, TREE, CLUSTERING (applicable across ML method types)
This ensures scalability, modularity, and easy extension to new problem classes beyond SCFLP.

Key Configuration Categories:
-----------------------------

1. **Directory Structure**:
   - Root paths for data, features, and results storage.
   - Organized hierarchy for SCFLP problem instances
   - Standardized output locations for logs and analysis

2. **SCFLP Problem Instance Management**:
   - Mapping of SCFLP scenarios and problem configurations
   - Instance selection for focused SCFLP testing
   - Other problem types exist as stubs for future extension

3. **Feature Engineering Parameters**:
   - Graph-based feature generation for SCFLP scenarios
   - Word2Vec embedding configuration (dimensions, window, walks)
   - Random walk parameters for scenario similarity computation
   - Reproducibility controls via seeding

4. **Method Architecture Configuration**:
   - Method categories: Exact (standard Benders) vs ML (5 distinct types)
   - Exact callback strategies: integer-only vs integer+LP
   - ML callback strategies: 4 variants for when to use ML vs exact cuts
   - Clear 6-step execution pattern with conditional step execution per method

5. **ML Method Types and Predictor Selection**:
   - **ML_Augmented_BD**: Full pipeline (Phase0→Phase1→Features→Training→Phase2)
   - **ML_Augmented_BD_with_duals**: Skip feature generation, use duals directly
   - **Covariate_Similarity**: Skip Phases 0/1, use input covariates for training
   - **Uncertainty_Similarity**: Skip Phases 0/1, use uncertainty vectors for training
   - **Random_Sampling**: Skip all training, dynamic random sampling in Phase2
   - Each method (except Random_Sampling) supports: KNN, TREE, or CLUSTERING predictor

6. **Switching Strategy Parameters**:
   - Behavioral modes for prediction failure handling
   - Adaptive switching between ML and exact solving
   - Permanent switching thresholds and conditions

7. **Experimental Grid Parameters**:
   - Method category selection (Exact, ML)
   - ML method type selection (5 types)
   - Predictor type selection per method (KNN, TREE, CLUSTERING)
   - Callback strategy selection (when to add cuts)
   - Predictor-specific hyperparameters

8. **SCFLP-Specific Settings**:
   - Focus on SCFLP problem instances and configurations
   - Other problem types exist as stubs for future extension
   - Streamlined configuration for current SCFLP focus

Configuration Architecture:
---------------------------

**Dual Parameter System**: Individual Variables + Grid Variables

The configuration system maintains both individual parameter variables and their corresponding 
grid versions to serve different architectural needs within the framework.

**Individual Variables** - Function Default Arguments:
```python
selected_subproblems_ratio = 0.2
# prediction_method = ["KNN"]  # KNN method includes its own selection strategy  # COMMENTED OUT - using DUAL_CLUSTERING_WITH_DUAL below
feature_dim = 3
```

Primary Uses:
- **Function signature defaults**: Enable functions to be called without explicit parameter specification
- **Quick testing and debugging**: Provide sensible defaults for rapid development
- **Backward compatibility**: Support existing code that expects single-value imports
- **Documentation references**: Serve as canonical "recommended" values for parameters

Example Integration:
```python
def solve_master_problem(problem_type, data, selected_subproblems, 
                        switching_strategy=switching_strategy[0],  # Uses first element as default
                        selected_subproblems_ratio=selected_subproblems_ratio):  # Default value
    # Function implementation uses these defaults when not explicitly provided
```

**Grid Variables** - Experimental Parameter Sweeps:
```python
selected_subproblems_ratio_grid = [0.1, 0.2, 0.3]  # Ratios of total scenarios to select
feature_dim_grid = [2, 3, 5]
n_neighbors_grid = [3, 5, 7]           # KNN-specific parameter grid
```

Primary Uses:
- **Systematic experimentation**: Generate comprehensive parameter combinations in main.py
- **Research evaluation**: Enable statistical analysis across parameter ranges
- **Sensitivity analysis**: Identify optimal parameter settings and their interactions
- **Algorithm comparison**: Compare performance across different algorithmic choices

Example Integration:
```python
def create_parameter_grid():
    # Uses grid variables to generate all combinations
    for pred_method in prediction_method:
        for ratio in selected_subproblems_ratio_grid:
            for feat_dim in feature_dim_grid:
                # Generate parameter combination dictionary
                params = {
                    'prediction_method': pred_method,  # Method includes selection strategy
                    'selected_subproblems_ratio': ratio,
                    'feature_dim': feat_dim,
                    # ... other parameters
                }
```

**Architectural Benefits**:

1. **Function Call Flexibility**:
   - Individual runs: `solve_master_problem(...)` uses defaults from individual variables
   - Experimental runs: `solve_master_problem(**params)` uses values from grid combinations

2. **Code Maintainability**:
   - Single source of truth for default values
   - Clear separation between default behavior and experimental exploration
   - Easy modification of defaults without affecting experimental grids

3. **Development Workflow**:
   - Researchers can quickly test with defaults during development
   - Comprehensive evaluation uses grid variables for publication-quality results
   - Debugging uses predictable default values rather than arbitrary grid selections

**Variable Naming Convention**:
- Base name: `parameter_name` (individual variable for defaults)
- Grid name: `parameter_name_grid` (list/array for experimental sweeps)

**Special Cases**:
Some individual variables are already lists (e.g., `prediction_method = ["KNN", "TREE", "DUAL_CLUSTERING"]`) because:
- They represent multiple valid options that may all be used as defaults
- The first element typically serves as the primary default
- They maintain compatibility with existing multi-option parameter handling

**Deprecation Strategy**: Clear marking of deprecated parameters
- Migration path from old single values to new dual system
- Backward compatibility maintenance during transition periods
- Clear documentation of preferred approaches for new development

**Grid Generation Logic**:
The main.py parameter grid generator creates configurations based on method architecture:
- **Exact methods**: Generate configurations with exact_callback_types (int_only, int_and_LP)
- **ML methods**: Generate configurations for each (ml_method_type, ml_callback_type, predictor_type) combination
  - ML_Augmented_BD × 3 predictors × 4 callback strategies
  - ML_Augmented_BD_with_duals × 3 predictors × 4 callback strategies
  - Covariate_Similarity × 3 predictors × 4 callback strategies
  - Uncertainty_Similarity × 3 predictors × 4 callback strategies  
  - Random_Sampling × 4 callback strategies (no predictor needed)
- **Optimization**: Exact methods don't iterate over predictor types (not used)

Feature Engineering Configuration:
----------------------------------

The feature generation system uses graph-based methods to capture scenario relationships:

- **n_walks**: Number of random walks per scenario for similarity computation
- **walk_length**: Length of each random walk in the scenario graph
- **feature_dim**: Dimensionality of Word2Vec embeddings for scenarios
- **window**: Context window size for Word2Vec training
- **min_count**: Minimum frequency threshold for Word2Vec vocabulary
- **sg**: Skip-gram vs CBOW model selection
- **seed**: Random seed for reproducible feature generation

Method Architecture:
--------------------

**6-Step Execution Pattern**:
All methods follow a structured 6-step pattern, with conditional execution:

Step 1 - Phase 0: Random first-stage solution generation (if m_trials > 0)
Step 2 - Phase 0 cut generation: Build cuts from Phase 0 duals
Step 3 - Phase 1: Standard Benders iterations for n_trials
Step 4 - Feature generation: Create features from collected duals
Step 5 - Partitioning and training: Select scenarios and train predictor
Step 6 - Phase 2: ML-augmented iterations

**Critical Architectural Distinction - Partition Strategy**:

The selected/unselected partition behavior is fundamentally different between methods:

FIXED PARTITION (Methods 1-4):
  - Selected and unselected scenarios are determined ONCE during training (Step 5)
  - These same selected scenarios remain the same throughout ALL Phase 2 iterations
  - The predictor's mappings (clustering_mappings, knn_neighbors, tree_rules) are trained on this fixed partition
  - All selected scenarios (regardless of ID, including scenario 0 if selected) are solved exactly in Phase 1
  - All scenarios in the predictor's results must be accessible from dual_cache when prediction accesses them
  - Scenario IDs are uniform: 0, 1, 2, ..., n-1 are all treated identically by clustering and KNN/TREE algorithms

DYNAMIC PARTITION (Random_Sampling only):
  - No fixed training occurs (Step 5 is skipped)
  - Selected scenarios are RANDOMLY chosen at EACH Phase 2 iteration (~25 random scenarios per iteration)
  - This means different scenarios are solved exactly and different scenarios are predicted at each iteration
  - Prediction is simple averaging of whichever scenarios were randomly sampled that iteration
  - No predictor_mappings exist - purely random selection + averaging strategy

**Design Rationale**:
  - Fixed partition: Enables sophisticated ML models (clustering, KNN, trees) that require consistent feature/dual space
  - Dynamic partition: Simpler, no training needed, but potentially less effective predictions (averaging vs learned mappings)

**Important Note on Scenario IDs**:
  - All scenario IDs (0 through n-1) are equivalent and should be handled uniformly
  - There is nothing special about scenario 0 - it participates in clustering, KNN selection, and TREE splits just like any other scenario
  - When scenario 0 is selected as a representative by clustering (or by any other method), it is solved exactly and cached exactly like other scenarios
  - The code must treat scenario ID 0 identically to scenario IDs 1, 2, etc. (this was the critical bug: Python falsy-zero condition check)

**Method Category: Exact**
- Executes: Step 3 only (continuous until convergence, no iteration cap)
- No steps 1,2,4,5,6 (no Phase 0, no features, no ML)
- Callback variants: int_only, int_and_LP

**Method Category: ML**

*Type 1: ML_Augmented_BD*
- Executes: Steps 1-6 (all steps, if Phase 0 configured)
- Input for Step 5: Features from Step 4
- Predictors: KNN, TREE, or CLUSTERING
- **Partition Strategy**: FIXED at end of Step 5 → Same 25 selected scenarios throughout ALL Phase 2 iterations

*Type 2: ML_Augmented_BD_with_duals*
- Executes: Steps 1,2,3,5,6 (skip Step 4 - feature generation)
- Input for Step 5: Duals directly from Steps 1-3
- Predictors: KNN, TREE, or CLUSTERING
- **Partition Strategy**: FIXED at end of Step 5 → Same selected scenarios throughout ALL Phase 2 iterations

*Type 3: Covariate_Similarity*
- Executes: Steps 5,6 only (skip Steps 1-4)
- Input for Step 5: Covariate vectors from problem data
- Predictors: KNN, TREE, or CLUSTERING
- **Partition Strategy**: FIXED at Step 5 (before Phase 2 begins) → Deterministic, unchanged across all Phase 2 iterations

*Type 4: Uncertainty_Similarity*
- Executes: Steps 5,6 only (skip Steps 1-4)
- Input for Step 5: Uncertainty vectors (demands/capacities) from problem data
- Predictors: KNN, TREE, or CLUSTERING
- **Partition Strategy**: FIXED at Step 5 (before Phase 2 begins) → Deterministic, unchanged across all Phase 2 iterations

*Type 5: Random_Sampling*
- Executes: Step 6 only (skip Steps 1-5)
- No training needed: **DYNAMIC random sampling at EACH Phase 2 iteration**
- Prediction: simple average of all sampled scenarios
- No predictor parameter
- **Partition Strategy**: **RANDOM at each iteration** → Different ~25 randomly-selected scenarios each Phase 2 callback call

Predictor Types and Parameters:
--------------------------------

**KNN (K-Nearest Neighbors)**:
- **n_neighbors**: Number of nearest neighbors for prediction
- **find_optimal_k**: Whether to optimize k during selection
- Works with features, duals, covariates, or uncertainty vectors

**TREE (Decision Tree)**:
- **min_impurity_decrease**: Minimum impurity decrease threshold
- **quantile_grid_size**: Number of quantiles for split candidates
- Adaptive implementation: grows naturally based on selection budget
- Works with features, duals, covariates, or uncertainty vectors

**CLUSTERING (K-means/K-medoids)**:
- **random_state**: Random seed for reproducibility
- **init_method**: Initialization method (k-means++, random)
- **max_iter**: Maximum iterations for convergence
- **tol**: Convergence tolerance
- Works with features, duals, covariates, or uncertainty vectors

Callback Strategies:
--------------------

**Exact Method Callbacks**:
- **int_only**: Register callbacks at integer solutions only
- **int_and_LP**: Register callbacks at integer + LP relaxations

**ML Method Callbacks** (4 variants for unselected scenarios):
- **ML_at_integer_only**: ML cuts at integer, no LP callbacks
- **ML_at_integer_exact_at_LP**: ML cuts at integer, exact at LP
- **ML_at_LP_exact_at_integer**: Exact cuts at integer, ML at LP
- **ML_at_both**: ML cuts at both integer and LP

Note: Sampled scenarios always use exact cuts in all variants

Switching Strategy Options:
---------------------------

- **no_switch**: Traditional ML-augmented Benders (default behavior)
- **alternate**: Temporary exact solving when ML fails, then return to ML
- **permanent_switch**: Permanent switch to exact solving after prediction failures

Experimental Design:
--------------------

The grid parameter system enables comprehensive experimental evaluation:

1. **Parameter Combinations**: Systematic exploration of parameter space
2. **Dependency Management**: Logical constraints between parameters
3. **Resource Planning**: Estimation of computational requirements
4. **Result Organization**: Structured output for analysis

Grid Parameter Dependencies:
----------------------------

**IMPORTANT**: Many parameters are IGNORED based on other parameter values. Setting a parameter
does NOT guarantee it will be used in experiments.

Summary of Key Dependencies:
- **Method Category = Exact**: Only exact_callback_types are relevant, all ML parameters IGNORED
- **Method Category = ML**: 
  - ml_method_types and ml_callback_types are relevant
  - Predictor types are relevant (except for Random_Sampling)
  - When predictor = KNN: Only KNN-specific parameters are used
  - When predictor = TREE: Only TREE-specific parameters are used
  - When predictor = CLUSTERING: Only CLUSTERING-specific parameters are used
- **ML Method Type Dependencies**:
  - ML_Augmented_BD: Requires feature_dim (Step 4 executed), uses predictor_ML_Augmented_BD_grid
  - ML_Augmented_BD_with_duals: feature_dim IGNORED (Step 4 skipped), uses predictor_ML_Augmented_BD_with_duals_grid
  - Covariate_Similarity: n_trials/m_trials IGNORED (Steps 1-3 skipped), uses predictor_Covariate_Similarity_grid
  - Uncertainty_Similarity: n_trials/m_trials IGNORED (Steps 1-3 skipped), uses predictor_Uncertainty_Similarity_grid
  - Random_Sampling: ALL predictor parameters IGNORED, no training parameters needed

**For Complete Dependency Rules**: See section "PARAMETER DEPENDENCIES AND IGNORED PARAMETERS" 
below (~line 540) which provides exhaustive details on what gets ignored in each configuration.

**Configuration Validation**: Use the `validate_config()` function (defined at end of this file)
to check your configuration and see warnings about parameters that will be ignored:

```python
from config import validate_config
validate_config()  # Prints detailed report of ignored parameters
```

This helps avoid wasting computational resources on parameter sweeps that won't be used.

Directory Structure:
--------------------

- **DATA_ROOT**: Input problem instances organized by problem class
- **FEATURES_ROOT**: Generated feature vectors and embeddings
- **RESULTS_ROOT**: Solution logs, timing analysis, and experimental results

Problem Instance Organization:
------------------------------

Instances are organized hierarchically:
```
DATA_ROOT/
├── UFL/
│   ├── MP1, MP2, ...
├── SCFLP/
│   ├── scenarios_s200_v1_mnormal/
│   │   ├── p1_stochastic.txt
│   │   └── ...
├── SSLP/
│   └── ...
```

Usage Patterns:
---------------

**Individual Experiments**:
```python
from config import method_categories_grid, exact_callback_types_grid, ml_method_types_grid
# Use grid values to specify which methods to run
```

**Systematic Evaluation**:
```python
from config import (
    method_categories_grid,           # ["Exact", "ML"]
    exact_callback_types_grid,        # For Exact: ["int_only", "int_and_LP"]
    ml_method_types_grid,             # For ML: 5 method types
    ml_callback_types_grid,           # For ML: 4 callback strategies
    predictor_ML_Augmented_BD_grid    # Predictor selection per method
)
# Use grid values for comprehensive experiments
```

**Parameter Validation**:
The configuration system includes validation logic to ensure:
- Compatible parameter combinations
- Valid ranges and types
- Dependency satisfaction

Extensibility:
--------------

The configuration system is designed for easy extension:
1. Add new parameters following existing patterns
2. Update grid parameters for new experimental factors
3. Maintain backward compatibility through deprecation strategy
4. Document parameter interactions and dependencies

Performance Considerations:
---------------------------

- Parameter grids can generate large numbers of combinations
- Memory and computational requirements scale with grid size
- Consider subset evaluation for initial parameter exploration
- Use dependency constraints to reduce infeasible combinations

Integration with Main Framework:
--------------------------------

The configuration system integrates with:
- **main.py**: Parameter grid generation and experimental orchestration
- **master_problem.py**: Problem-specific parameter passing
- **callbacks.py**: ML algorithm configuration
- **prediction.py**: Learning parameter specification
- **feature_generation.py**: Feature engineering configuration

Default Value Usage Pattern:
-----------------------------

Throughout the framework, functions use individual variables as default arguments:

```python
# In master_problem.py
def solve_master_problem(problem_type, data, selected_subproblems, 
                        switching_strategy=switching_strategy[0],    # "no_switch"
                        selected_subproblems_ratio=selected_subproblems_ratio):  # 0.2

# In callbacks.py or other modules  
def some_function(ratio=selected_subproblems_ratio,                 # 0.2
                 method=prediction_method[0]):                      # "KNN"
```

This pattern enables:
- **Zero-configuration usage**: Functions work with sensible defaults
- **Incremental customization**: Override only specific parameters as needed
- **Consistent behavior**: Same defaults across different function calls
- **Easy maintenance**: Change defaults in one place (config.py)

While experimental evaluation uses:
```python
# In main.py create_parameter_grid()
for method in prediction_method:                     # ["KNN", "TREE", "DUAL_CLUSTERING"]
    for ratio in selected_subproblems_ratio_grid:   # [0.1, 0.2, 0.3]
        # Generate all combinations for comprehensive testing
```

This dual approach balances ease of use with comprehensive experimental capability.

"""

import os
from pathlib import Path

# ========== DYNAMIC PROJECT ROOT DETECTION ==========
# Detect PROJECT_ROOT by going up from the src directory
# config.py is in src/, so PROJECT_ROOT is the parent of src/
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())

# Optional read-only input-data location for isolated Git worktrees.
# Unset by default: the original project layout and experiments are unchanged.
# The objective-guard pilot launcher sets this to the original Nibi DATA folder.
INPUT_DATA_ROOT = os.path.abspath(
    os.environ.get("LABD_DATA_ROOT") or os.path.join(PROJECT_ROOT, "DATA")
)

# ========== Root Directories (System-Agnostic, Based on PROJECT_ROOT) ==========
PROBLEM_CLASS_ROOTS = {
    "SCFLP": {
        "DATA_ROOT": INPUT_DATA_ROOT,
        "FEATURES_ROOT": os.path.join(PROJECT_ROOT, "RESULTS", "SCFLP_class", "FEATURES"),
        "RESULTS_ROOT": os.path.join(PROJECT_ROOT, "RESULTS", "SCFLP_class", "RESULTS"),
    },
    "SUMND": {
        "DATA_ROOT": INPUT_DATA_ROOT,
        "FEATURES_ROOT": os.path.join(PROJECT_ROOT, "RESULTS", "SUMND_class", "FEATURES"),
        "RESULTS_ROOT": os.path.join(PROJECT_ROOT, "RESULTS", "SUMND_class", "RESULTS"),
    },
    "SCMND": {
        "DATA_ROOT": INPUT_DATA_ROOT,
        "FEATURES_ROOT": os.path.join(PROJECT_ROOT, "RESULTS", "SCMND_class", "FEATURES"),
        "RESULTS_ROOT": os.path.join(PROJECT_ROOT, "RESULTS", "SCMND_class", "RESULTS"),
    },
}

DEFAULT_PROBLEM_CLASS_ROOTS = "SCFLP"


def get_problem_class_roots(problem_class: str):
    """Return (DATA_ROOT, FEATURES_ROOT, RESULTS_ROOT) for the given problem class."""
    normalized = problem_class.upper().replace("-", "")
    roots = PROBLEM_CLASS_ROOTS.get(normalized, PROBLEM_CLASS_ROOTS[DEFAULT_PROBLEM_CLASS_ROOTS])
    return roots["DATA_ROOT"], roots["FEATURES_ROOT"], roots["RESULTS_ROOT"]


# Backward-compatible defaults for modules that still import these globals.
DATA_ROOT, FEATURES_ROOT, RESULTS_ROOT = get_problem_class_roots(DEFAULT_PROBLEM_CLASS_ROOTS)
# ========== Problem Classes and Instances ==========
# Focus: SCFLP, NE-SP, SUMND, and SCMND - other problem classes have implementations stubs for future extension
# Other problem types ("UFL","UHL","CMND","MCFL","SSLP") exist as stubs for future extension
# Full SCMND comparison grid. Each data instance is evaluated under both the
# preserved implementation and the manuscript-envelope formulation selected
# by scmnd_formulation_grid below.
problem_classes_to_instances = {
    "SCMND": [
        "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r01.1.stoch.json",
        "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r01.1.stoch.json",
        "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r01.1.stoch.json",
        "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r01.2.stoch.json",
        "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r01.2.stoch.json",
        "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r01.2.stoch.json",
        "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r02.1.stoch.json",
        "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r02.1.stoch.json",
        "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r02.1.stoch.json",
        "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r02.2.stoch.json",
        "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r02.2.stoch.json",
        "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r02.2.stoch.json",
        "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r04.1.stoch.json",
        "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r04.1.stoch.json",
        "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r04.1.stoch.json",
        "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r04.2.stoch.json",
        "SUMND/R_stochastic/S300_sigma0.10-0.30_cov5/r04.2.stoch.json",
        "SUMND/R_stochastic/S500_sigma0.10-0.30_cov5/r04.2.stoch.json",
    ],
}

"""
              "scenarios_s1100_cov3_distarma(2,2)_with_covariate_generation/p5_stochastic.txt",
              "scenarios_s100_sigma0.1_0.3_cov3_merve_with_covariates/p5_stochastic.txt",
              "scenarios_s1300_cov3_distarma(2,2)_with_covariate_generation/p5_stochastic.txt"
              "scenarios_s500_cv0.1_distnormal/p5_stochastic.txt", "scenarios_s500_cv0.1_distnormal/p20_stochastic.txt", "scenarios_s500_cv0.1_distnormal/p50_stochastic.txt","scenarios_s300_cv0.1_distnormal/p51_stochastic.txt",
              "scenarios_s500_cv0.2_distnormal/p5_stochastic.txt", "scenarios_s500_cv0.2_distnormal/p20_stochastic.txt", "scenarios_s500_cv0.2_distnormal/p50_stochastic.txt","scenarios_s300_cv0.2_distnormal/p51_stochastic.txt",
              "scenarios_s500_cv0.3_distnormal/p5_stochastic.txt", "scenarios_s500_cv0.3_distnormal/p20_stochastic.txt", "scenarios_s500_cv0.3_distnormal/p50_stochastic.txt","scenarios_s500_cv0.3_distnormal/p51_stochastic.txt",
              "scenarios_s500_cv0.4_distnormal/p5_stochastic.txt", "scenarios_s500_cv0.4_distnormal/p20_stochastic.txt", "scenarios_s500_cv0.4_distnormal/p50_stochastic.txt","scenarios_s500_cv0.4_distnormal/p51_stochastic.txt",
              "scenarios_s500_cv0.5_distnormal/p5_stochastic.txt", "scenarios_s500_cv0.5_distnormal/p20_stochastic.txt", "scenarios_s500_cv0.5_distnormal/p50_stochastic.txt","scenarios_s500_cv0.5_distnormal/p51_stochastic.txt",
              "scenarios_s700_cv0.1_distnormal/p5_stochastic.txt", "scenarios_s500_cv0.1_distnormal/p20_stochastic.txt", "scenarios_s500_cv0.1_distnormal/p50_stochastic.txt","scenarios_s300_cv0.1_distnormal/p51_stochastic.txt",
              "scenarios_s700_cv0.2_distnormal/p5_stochastic.txt", "scenarios_s500_cv0.2_distnormal/p20_stochastic.txt", "scenarios_s500_cv0.2_distnormal/p50_stochastic.txt","scenarios_s300_cv0.2_distnormal/p51_stochastic.txt",
              "scenarios_s700_cv0.3_distnormal/p5_stochastic.txt", "scenarios_s700_cv0.3_distnormal/p20_stochastic.txt", "scenarios_s700_cv0.3_distnormal/p50_stochastic.txt","scenarios_s700_cv0.3_distnormal/p51_stochastic.txt",
              "scenarios_s700_cv0.4_distnormal/p5_stochastic.txt", "scenarios_s700_cv0.4_distnormal/p20_stochastic.txt", "scenarios_s700_cv0.4_distnormal/p50_stochastic.txt","scenarios_s700_cv0.4_distnormal/p51_stochastic.txt",
              "scenarios_s700_cv0.5_distnormal/p5_stochastic.txt", "scenarios_s700_cv0.5_distnormal/p20_stochastic.txt", "scenarios_s700_cv0.5_distnormal/p50_stochastic.txt","scenarios_s700_cv0.5_distnormal/p51_stochastic.txt",
              "scenarios_s100_sigma0.1_0.3_distmerve_normal", "scenarios_s300_sigma0.1_0.3_distmerve_normal",
"""

# ========== Phase Configuration ==========
# Phase 0: Random Feasible Solution Generation (Optional Warm-up)
# - Generates m_trials random feasible first-stage solutions
# - Solves all subproblems exactly for each solution to collect optimal duals
# - Collects all cuts and adds them to master problem before Phase 1 begins
# - Provides warm-start cuts to strengthen master problem relaxation
# - When m_trials = 0, Phase 0 is skipped (current two-phase LABD behavior)

# Phase 1: Traditional Benders Decomposition (Learning Phase)  
# - Runs classic Benders for n_trials iterations
# - Solves all subproblems exactly to collect high-quality dual training data
# - Duals from Phase 0 and Phase 1 are combined for feature generation in Phase 2

# ========== Feature Engineering Parameters ==========
# Individual feature generation parameters (used as function defaults)
n_walks = 2               # Number of random walks
walk_length = 5           # Length of each random walk
feature_dim = 3           # Embedding dimension (Word2Vec) #changed dimesion from 3 to 9 to see the effect on the performance of the model
window = 5                # Window size for Word2Vec
min_count = 1             # Minimum word frequency
sg = 1                    # Skip-gram model (1) vs CBOW (0)
seed = 42                 # Random seed for reproducibility
feature_transition_temperature = 1.0  # tau in exp(-Omega_ij/tau), with P_ii=0
feature_stability_epsilon = 1e-12      # denominator safeguard in normalized dual distances
feature_transition_rule = "MANUSCRIPT" # Legacy option: LEGACY_INVERSE_WITH_SELF_LOOPS

# ========== Method Category and Type Configuration ==========
# Individual parameters (used as function defaults)
selected_subproblems_ratio = 0.05  # Ratio of subproblems to select (0.0 to 1.0).

# ========== Method Categories Grid ==========
# Which broad algorithm categories to run
method_categories_grid = ["ML"]  # Standard BD plus the requested learning methods ["Exact", "ML"] 

# ========== Exact Method Configuration ==========
# For standard Benders decomposition (no ML, single type)#..
# When to add cuts for Exact methods:
exact_callback_types_grid = [
    #"int_only",              # Cuts only at integer solutions (classic single-tree Benders)
    "int_and_LP"             # Cuts at integer + LP relaxations (enhanced Benders)
]

# ========== ML Method Configuration ==========
# Which ML method types to run (5 distinct method types)
ml_method_types_grid = [
     "ML_Augmented_BD"         # TREE, CLUSTERING, and the two retained parametric predictors below
     #"ML_Augmented_BD_with_duals",     # Steps 1-3,5-6: Phase0 + Phase1 + Training(duals) + Phase2
     #"Covariate_Similarity",           # Steps 5-6: Training(covariates) + Phase2 (skip Phases 0,1)
     #"Uncertainty_Similarity",         # Steps 5-6: Training(uncertainty) + Phase2 (skip Phases 0,1)
     #"Random_Sampling"                 # Step 6 only: Dynamic random sampling in Phase2 (skip all training)
]

# ML callback strategies (when to use ML vs exact for unselected scenarios)
ml_callback_types_grid = [
    #"ML_at_integer_only",           # ML cuts for unselected at integer only, no LP cuts
     #"ML_at_integer_exact_at_LP",    # ML cuts for unselected at integer, exact at LP
     #"ML_at_LP_exact_at_integer"   # Exact cuts for unselected at integer, ML at LP
     "ML_at_both"                    # ML cuts for unselected at both integer and LP
]

# ========== Predictor Type Configuration for ML Methods ==========
# Each ML method type (except Random_Sampling) can use different predictors
# Predictor types retained for this grid: TREE, CLUSTERING, and two PARAMETRIC_* variants

# Predictors for ML_Augmented_BD (uses features from step 4)
predictor_ML_Augmented_BD_grid = [
    #"TREE",
    #"CLUSTERING",
    "PARAMETRIC_RIDGE_REPAIR"
    #"PARAMETRIC_RHS_RIDGE_REPAIR"
    #"PARAMETRIC_MANUSCRIPT",
    #"PARAMETRIC_PENALIZED_ERRORS"
]

# Predictors for ML_Augmented_BD_with_duals (uses duals directly, skip step 4)
predictor_ML_Augmented_BD_with_duals_grid = ["KNN", "TREE", "CLUSTERING"] #"KNN", "TREE", "CLUSTERING"

# Predictors for Covariate_Similarity (uses input covariates, skip steps 1-4)
predictor_Covariate_Similarity_grid = ["KNN", "TREE", "CLUSTERING"] #"KNN", "TREE", "CLUSTERING"

# Predictors for Uncertainty_Similarity (uses uncertainty vectors, skip steps 1-4)
predictor_Uncertainty_Similarity_grid = ["KNN", "TREE", "CLUSTERING"] #"KNN", "TREE", "CLUSTERING"

# Random_Sampling doesn't need predictor (uses simple averaging of all sampled scenarios)

# ========== PARAMETER DEPENDENCIES AND IGNORED PARAMETERS ==========
"""
Grid Parameter Dependency Rules:
================================

The parameter grid system has built-in dependencies where certain parameters are IGNORED
based on the values of other parameters. This ensures efficient experimentation by not
generating invalid or duplicate configurations.

CRITICAL: Setting a parameter value does NOT guarantee it will be used. Read dependencies below.

1. METHOD CATEGORY DEPENDENCIES
   -------------------------------
   
   IF method_categories_grid = ["Exact"] ONLY:
   → IGNORED: All ML-related parameters
     - ml_method_types_grid (not used)
     - ml_callback_types_grid (not used)
     - predictor_ML_Augmented_BD_grid (not used)
     - predictor_ML_Augmented_BD_with_duals_grid (not used)
     - predictor_Covariate_Similarity_grid (not used)
     - predictor_Uncertainty_Similarity_grid (not used)
     - n_neighbors_grid (not used)
     - knn_find_optimal_k_grid (not used)
     - tree_min_impurity_decrease_grid (not used)
     - tree_quantile_grid_size_grid (not used)
     - clustering_random_state_grid (not used)
     - clustering_init_method_grid (not used)
     - clustering_max_iter_grid (not used)
     - clustering_tol_grid (not used)
     - selected_subproblems_ratio_grid (limited relevance - no scenario selection)
     - use_covariates_grid (not used)
   → USED: Only Exact-related parameters
     - exact_callback_types_grid
     - n_trials_grid (Exact runs until convergence, but this can set Phase 1 equivalent)
     - use_stochastic_capacity_grid
   
   IF method_categories_grid = ["ML"] ONLY:
   → IGNORED: Exact-related parameters
     - exact_callback_types_grid (not used)
   → USED: All ML-related parameters based on ML method type (see below)

2. ML METHOD TYPE DEPENDENCIES
   ----------------------------
   
   IF ml_method_types_grid = ["ML_Augmented_BD"]:
   → USED:
     - predictor_ML_Augmented_BD_grid (determines which predictors to try)
     - ml_callback_types_grid (callback strategy)
     - feature_dim_grid (Step 4 creates features with this dimension)
     - n_trials_grid (Phase 1 iterations)
     - m_trials_grid (Phase 0 iterations, 0 = skip Phase 0)
     - selected_subproblems_ratio_grid (scenario selection ratio)
         - use_covariates_grid (feature source mode: features_only / features_plus_covariates / covariates_only)
   → IGNORED:
     - predictor_ML_Augmented_BD_with_duals_grid (other method)
     - predictor_Covariate_Similarity_grid (other method)
     - predictor_Uncertainty_Similarity_grid (other method)
   
   IF ml_method_types_grid = ["ML_Augmented_BD_with_duals"]:
   → USED:
     - predictor_ML_Augmented_BD_with_duals_grid (determines which predictors)
     - ml_callback_types_grid
     - n_trials_grid (Phase 1 iterations to collect duals)
     - m_trials_grid (Phase 0 iterations)
     - selected_subproblems_ratio_grid
   → IGNORED:
     - feature_dim_grid (Step 4 skipped - uses duals directly)
     - use_covariates_grid (not applicable - uses duals)
     - predictor_ML_Augmented_BD_grid (other method)
     - predictor_Covariate_Similarity_grid (other method)
     - predictor_Uncertainty_Similarity_grid (other method)
   
   IF ml_method_types_grid = ["Covariate_Similarity"]:
   → USED:
     - predictor_Covariate_Similarity_grid (determines which predictors)
     - ml_callback_types_grid
     - selected_subproblems_ratio_grid
   → IGNORED:
     - n_trials_grid (Steps 1-3 skipped - no Phase 0/1)
     - m_trials_grid (Steps 1-2 skipped - no Phase 0)
     - feature_dim_grid (Step 4 skipped - uses input covariates)
     - use_covariates_grid (uses covariates by definition)
     - predictor_ML_Augmented_BD_grid (other method)
     - predictor_ML_Augmented_BD_with_duals_grid (other method)
     - predictor_Uncertainty_Similarity_grid (other method)
   
   IF ml_method_types_grid = ["Uncertainty_Similarity"]:
   → USED:
     - predictor_Uncertainty_Similarity_grid (determines which predictors)
     - ml_callback_types_grid
     - selected_subproblems_ratio_grid
   → IGNORED:
     - n_trials_grid (Steps 1-3 skipped - no Phase 0/1)
     - m_trials_grid (Steps 1-2 skipped - no Phase 0)
     - feature_dim_grid (Step 4 skipped - uses uncertainty vectors)
     - use_covariates_grid (not applicable - uses uncertainty vectors)
     - predictor_ML_Augmented_BD_grid (other method)
     - predictor_ML_Augmented_BD_with_duals_grid (other method)
     - predictor_Covariate_Similarity_grid (other method)
   
   IF ml_method_types_grid = ["Random_Sampling"]:
   → USED:
     - ml_callback_types_grid (callback strategy)
     - selected_subproblems_ratio_grid (how many to sample each iteration)
   → IGNORED:
     - ALL predictor grids (no predictor used - simple averaging)
     - n_trials_grid (Steps 1-3 skipped - no Phase 0/1)
     - m_trials_grid (Steps 1-2 skipped - no Phase 0)
     - feature_dim_grid (Step 4 skipped - no feature generation)
     - use_covariates_grid (Step 5 skipped - no training)
     - n_neighbors_grid (no KNN predictor)
     - knn_find_optimal_k_grid (no KNN predictor)
     - tree_min_impurity_decrease_grid (no TREE predictor)
     - tree_quantile_grid_size_grid (no TREE predictor)
     - clustering_*_grid (no CLUSTERING predictor)

3. PREDICTOR TYPE DEPENDENCIES
   ----------------------------
   
   IF predictor_type in ["KNN"]:
   → USED:
     - n_neighbors_grid
     - knn_find_optimal_k_grid
   → IGNORED:
     - tree_min_impurity_decrease_grid (only for TREE predictor)
     - tree_quantile_grid_size_grid (only for TREE predictor)
     - clustering_random_state_grid (only for CLUSTERING predictor)
     - clustering_init_method_grid (only for CLUSTERING predictor)
     - clustering_max_iter_grid (only for CLUSTERING predictor)
     - clustering_tol_grid (only for CLUSTERING predictor)
   
   IF predictor_type in ["TREE"]:
   → USED:
     - tree_min_impurity_decrease_grid
     - tree_quantile_grid_size_grid
   → IGNORED:
     - n_neighbors_grid (only for KNN predictor)
     - knn_find_optimal_k_grid (only for KNN predictor)
     - clustering_random_state_grid (only for CLUSTERING predictor)
     - clustering_init_method_grid (only for CLUSTERING predictor)
     - clustering_max_iter_grid (only for CLUSTERING predictor)
     - clustering_tol_grid (only for CLUSTERING predictor)
   
   IF predictor_type in ["CLUSTERING"]:
   → USED:
     - clustering_random_state_grid
     - clustering_init_method_grid
     - clustering_max_iter_grid
     - clustering_tol_grid
   → IGNORED:
     - n_neighbors_grid (only for KNN predictor)
     - knn_find_optimal_k_grid (only for KNN predictor)
     - tree_min_impurity_decrease_grid (only for TREE predictor)
     - tree_quantile_grid_size_grid (only for TREE predictor)

4. COMPOUND DEPENDENCIES
   ----------------------
   
   Example 1: Exact-only configuration
   IF method_categories_grid = ["Exact"]:
     - Even if you set ml_method_types_grid = ["ML_Augmented_BD"], it's IGNORED
     - Even if you set predictor_ML_Augmented_BD_grid = ["KNN"], it's IGNORED
     - Even if you set n_neighbors_grid = [3, 5, 7], it's IGNORED
     - Only exact_callback_types_grid matters
   
   Example 2: Random Sampling configuration
   IF ml_method_types_grid = ["Random_Sampling"]:
     - Even if you set predictor_ML_Augmented_BD_grid = ["KNN"], it's IGNORED
     - Even if you set n_neighbors_grid = [3, 5, 7], it's IGNORED
     - No predictor parameters are used at all
   
   Example 3: Covariate Similarity with TREE
   IF ml_method_types_grid = ["Covariate_Similarity"]
   AND predictor_Covariate_Similarity_grid = ["TREE"]:
     - tree_min_impurity_decrease_grid and tree_quantile_grid_size_grid are USED
     - n_neighbors_grid and clustering_*_grid are IGNORED
     - n_trials_grid and m_trials_grid are IGNORED (no Phase 0/1)

CONFIGURATION VALIDATION TIPS:
------------------------------

1. **Start with method_categories_grid**: This is the ROOT decision
   - ["Exact"] → Ignore all ML parameters
   - ["ML"] → Ignore exact parameters
   - ["Exact", "ML"] → Both families active

2. **For ML, choose ml_method_types_grid**: This determines which steps execute
   - Controls which predictor grids are relevant
   - Controls which training parameters (n_trials, m_trials) matter

3. **For each ML method, choose predictors**: This determines algorithm parameters
   - KNN → set n_neighbors_grid, knn_find_optimal_k_grid
   - TREE → set tree_*_grid parameters
   - CLUSTERING → set clustering_*_grid parameters

4. **Check ignored parameters**: Review this list after configuration
   - Ensure you're not tuning parameters that will be ignored
   - Save computational resources by clearing irrelevant grids

EFFICIENCY RECOMMENDATIONS:
--------------------------

For best practice, explicitly set unused grids to empty or single default values:

```python
# Example: Exact-only configuration
method_categories_grid = ["Exact"]
ml_method_types_grid = []  # Not used - can set to empty
predictor_ML_Augmented_BD_grid = []  # Not used - can set to empty

# Example: Random Sampling configuration  
method_categories_grid = ["ML"]
ml_method_types_grid = ["Random_Sampling"]
predictor_ML_Augmented_BD_grid = []  # Not used
n_neighbors_grid = [3]  # Not used - but harmless to leave default
```

This makes your intent clear and documents which parameters you've intentionally left alone.
"""

use_stochastic_capacity = False  # If True, use scenario-specific capacities when available, if False, use first scenario capacities for all

# ========== Switching Strategy Parameters ==========
switching_strategy = ["no_switch"]  # Strategy for handling ML prediction when NO cuts added for unselected scenarios:
                                  # "no_switch": Continue with ML predictions (current behavior)
                                  # "alternate": Solve unselected scenarios exactly for this iteration, then continue ML
                                  # "permanent_switch": Permanently switch to exact solving for all remaining iterations

# ========== Feature Enhancement Parameters ==========
# Controls which vectors are used in Step 5 for ML_Augmented_BD:
# - "features_only": use Step 4 generated features only
# - "features_plus_covariates": concatenate generated features with covariates
# - "covariates_only": use only covariate vectors and skip Step 4 feature generation
# Note: This parameter only applies to ML_Augmented_BD; other ML method types use features_only
use_covariates_grid = ["features_plus_covariates"]  # Use the manuscript-aligned representation features
# ["features_only", "features_plus_covariates", "covariates_only"] 

# ========== Experimental Parameters for Grid Search ==========
# Grid parameters for comprehensive experimental evaluation
n_trials_grid = [5]  # Number of first-stage trial vectors (Phase 1 - classic Benders)  
m_trials_grid = [0]  # Number of Phase 0 warm-up random feasible solutions (0 = skip Phase 0)
selected_subproblems_ratio_grid = [0.05, 0.15, 0.25]
feature_dim_grid = [3]  # Manuscript feature dimension
use_stochastic_capacity_grid = [False]  # Whether to use scenario-specific capacities when available

# Author decision 2026-10-27: hard-capacity manuscript model only.
# Historical soft-capacity results belong to a separate experiment generation.
scmnd_formulation_grid = ["MANUSCRIPT_ENVELOPE"]

# ========== Predictor-Specific Grid Parameters ==========
# These parameters apply to specific predictor types across all ML methods

# KNN-specific grid parameters (applies to all methods using KNN predictor)
n_neighbors_grid = [3]  # Number of neighbors for KNN method
knn_find_optimal_k_grid = [False]  # Whether to enumerate k to find optimal (T,k) pair

# TREE-specific grid parameters (applies to all methods using TREE predictor)
# Adaptive Constraint-Free Implementation: tree grows naturally based on selection budget
tree_min_impurity_decrease_grid = [0.0]  # Minimum impurity decrease threshold (optional quality control)
tree_quantile_grid_size_grid = [10]  # Number of quantiles for split threshold candidates (optional granularity)

# CLUSTERING-specific grid parameters (applies to all methods using CLUSTERING predictor)
clustering_random_state_grid = [42]  # Random seed for clustering reproducibility
clustering_init_method_grid = ["k-means++"]  # Initialization method (k-means++ or random)
clustering_max_iter_grid = [300]  # Maximum iterations for convergence
clustering_tol_grid = [1e-4]  # Convergence tolerance

# ========== Parametric Predictor Parameters ==========
# All four variants use the same exact p-median partition.  The lists below
# are experiment grids; keeping one value in each list creates one run per
# active parametric predictor and outer experiment setting.
parametric_L_d_grid = [1e5]
parametric_L_v_grid = [1e5]
parametric_regularization_grid = [1e-6]
parametric_kappa_d_grid = [1.0]
parametric_kappa_v_grid = [1.0]
parametric_ridge_beta_grid = [1e-3]
parametric_rhs_dim_grid = [3]
parametric_rhs_ridge_beta_features_grid = [1e-3]
parametric_rhs_ridge_beta_rhs_grid = [1e-3]
parametric_repair_mode_grid = ["LOCAL"]  # LOCAL/GLOBAL line-segment or STRUCTURE_AWARE closed-form repair
parametric_feasibility_tolerance = 1e-7
parametric_repair_safety = 1e-10
parametric_objective_guard = False  # OFF reproduces existing published experiment runs
parametric_guard_gamma_zero_fastpath = False  # ON skips comparison if repaired dual is exactly its anchor
parametric_diagnostic_probe = False  # OFF by default; shadow objective probe in diagnostic runs
parametric_pmedian_solver = "MIP"         # Exact manuscript partition; GREEDY is test/pilot only
parametric_pmedian_time_limit = None
parametric_training_time_limit = None
parametric_solver_output = 0


# ========== Configuration Validation Helper ==========

def validate_config(verbose=True):
    """
    Validate the current configuration and report which parameters will be IGNORED.
    
    This function analyzes the configured parameter grids and reports which parameters
    will not be used based on the dependency rules. It helps identify:
    - Parameters you've set but that will be ignored
    - Potentially wasted computational effort on irrelevant parameter sweeps
    - Configuration mismatches and invalid combinations
    
    Args:
        verbose (bool): If True, print detailed warnings. If False, return warnings as list.
        
    Returns:
        list: List of warning messages about ignored parameters
        
    Example:
        >>> from config import validate_config
        >>> validate_config()
        
        Configuration Validation Report
        ================================
        
        Active Method Categories: ['Exact']
        
        IGNORED PARAMETERS (will not be used):
        - ml_method_types_grid: Not used for Exact-only configuration
        - predictor_ML_Augmented_BD_grid: Not used for Exact-only configuration
        ...
    """
    warnings = []
    
    # Check method categories
    has_exact = "Exact" in method_categories_grid
    has_ml = "ML" in method_categories_grid
    
    if verbose:
        print("\n" + "="*60)
        print("Configuration Validation Report")
        print("="*60)
        print(f"\nActive Method Categories: {method_categories_grid}")
        print()
    
    # If only Exact, warn about ML parameters
    if has_exact and not has_ml:
        warnings.append("EXACT-ONLY Mode: All ML parameters will be IGNORED")
        if ml_method_types_grid:
            warnings.append(f"  - ml_method_types_grid = {ml_method_types_grid} (not used)")
        if ml_callback_types_grid:
            warnings.append(f"  - ml_callback_types_grid = {ml_callback_types_grid} (not used)")
        if predictor_ML_Augmented_BD_grid:
            warnings.append(f"  - predictor_ML_Augmented_BD_grid = {predictor_ML_Augmented_BD_grid} (not used)")
        if predictor_ML_Augmented_BD_with_duals_grid:
            warnings.append(f"  - predictor_ML_Augmented_BD_with_duals_grid = {predictor_ML_Augmented_BD_with_duals_grid} (not used)")
        if predictor_Covariate_Similarity_grid:
            warnings.append(f"  - predictor_Covariate_Similarity_grid = {predictor_Covariate_Similarity_grid} (not used)")
        if predictor_Uncertainty_Similarity_grid:
            warnings.append(f"  - predictor_Uncertainty_Similarity_grid = {predictor_Uncertainty_Similarity_grid} (not used)")
        if len(n_neighbors_grid) > 1 or n_neighbors_grid[0] != 3:
            warnings.append(f"  - n_neighbors_grid = {n_neighbors_grid} (not used)")
        if selected_subproblems_ratio_grid:
            warnings.append(f"  - selected_subproblems_ratio_grid = {selected_subproblems_ratio_grid} (limited use)")
    
    # If only ML, warn about Exact parameters
    if has_ml and not has_exact:
        warnings.append("ML-ONLY Mode: Exact parameters will be IGNORED")
        if exact_callback_types_grid:
            warnings.append(f"  - exact_callback_types_grid = {exact_callback_types_grid} (not used)")
    
    # Check ML method types
    if has_ml:
        valid_use_covariates_modes = {"features_only", "features_plus_covariates", "covariates_only"}
        invalid_modes = [m for m in use_covariates_grid if m not in valid_use_covariates_modes]
        if invalid_modes:
            warnings.append(
                f"Invalid use_covariates_grid values: {invalid_modes}. "
                f"Valid values are {sorted(valid_use_covariates_modes)}"
            )

        for ml_method in ml_method_types_grid:
            method_warnings = []
            
            if ml_method == "ML_Augmented_BD":
                # Check if other predictor grids are set
                if predictor_ML_Augmented_BD_with_duals_grid and "ML_Augmented_BD_with_duals" not in ml_method_types_grid:
                    method_warnings.append(f"  - predictor_ML_Augmented_BD_with_duals_grid (not used for {ml_method})")
                if predictor_Covariate_Similarity_grid and "Covariate_Similarity" not in ml_method_types_grid:
                    method_warnings.append(f"  - predictor_Covariate_Similarity_grid (not used for {ml_method})")
                if predictor_Uncertainty_Similarity_grid and "Uncertainty_Similarity" not in ml_method_types_grid:
                    method_warnings.append(f"  - predictor_Uncertainty_Similarity_grid (not used for {ml_method})")
            
            elif ml_method == "ML_Augmented_BD_with_duals":
                # Check if feature_dim_grid has multiple values (Step 4 skipped)
                if len(feature_dim_grid) > 1:
                    method_warnings.append(f"  - feature_dim_grid = {feature_dim_grid} (Step 4 skipped, only first value may be used)")
                if use_covariates_grid != ["features_only"]:
                    method_warnings.append(f"  - use_covariates_grid = {use_covariates_grid} (not applicable, uses duals)")
            
            elif ml_method in ["Covariate_Similarity", "Uncertainty_Similarity"]:
                # Check if Phase 0/1 parameters have multiple values (Steps 1-3 skipped)
                if len(n_trials_grid) > 1 or n_trials_grid[0] != 5:
                    method_warnings.append(f"  - n_trials_grid = {n_trials_grid} (Steps 1-3 skipped)")
                if len(m_trials_grid) > 1 or m_trials_grid[0] != 0:
                    method_warnings.append(f"  - m_trials_grid = {m_trials_grid} (Steps 1-2 skipped)")
                if len(feature_dim_grid) > 1:
                    method_warnings.append(f"  - feature_dim_grid = {feature_dim_grid} (Step 4 skipped)")
            
            elif ml_method == "Random_Sampling":
                # Check if ANY predictor grids are set
                all_predictor_grids = [
                    predictor_ML_Augmented_BD_grid,
                    predictor_ML_Augmented_BD_with_duals_grid,
                    predictor_Covariate_Similarity_grid,
                    predictor_Uncertainty_Similarity_grid
                ]
                if any(all_predictor_grids):
                    method_warnings.append(f"  - ALL predictor grids (no predictor used for Random_Sampling)")
                if len(n_neighbors_grid) > 1 or len(clustering_random_state_grid) > 1:
                    method_warnings.append(f"  - ALL predictor parameter grids (no predictor used)")
                if len(n_trials_grid) > 1 or len(m_trials_grid) > 1 or len(feature_dim_grid) > 1:
                    method_warnings.append(f"  - n_trials_grid, m_trials_grid, feature_dim_grid (all training steps skipped)")
            
            if method_warnings:
                warnings.append(f"\nFor ML method '{ml_method}':")
                warnings.extend(method_warnings)
    
    # Check predictor-specific parameters
    if has_ml:
        all_predictors = set()
        for method in ml_method_types_grid:
            if method == "ML_Augmented_BD":
                all_predictors.update(predictor_ML_Augmented_BD_grid)
            elif method == "ML_Augmented_BD_with_duals":
                all_predictors.update(predictor_ML_Augmented_BD_with_duals_grid)
            elif method == "Covariate_Similarity":
                all_predictors.update(predictor_Covariate_Similarity_grid)
            elif method == "Uncertainty_Similarity":
                all_predictors.update(predictor_Uncertainty_Similarity_grid)
        
        predictor_warnings = []
        
        if "KNN" not in all_predictors:
            if len(n_neighbors_grid) > 1 or n_neighbors_grid[0] != 3:
                predictor_warnings.append(f"  - n_neighbors_grid = {n_neighbors_grid} (KNN not in active predictors)")
            if knn_find_optimal_k_grid != [False]:
                predictor_warnings.append(f"  - knn_find_optimal_k_grid = {knn_find_optimal_k_grid} (KNN not in active predictors)")
        
        if "TREE" not in all_predictors:
            if len(tree_min_impurity_decrease_grid) > 1 or tree_min_impurity_decrease_grid[0] != 0.0:
                predictor_warnings.append(f"  - tree_min_impurity_decrease_grid = {tree_min_impurity_decrease_grid} (TREE not in active predictors)")
            if len(tree_quantile_grid_size_grid) > 1 or tree_quantile_grid_size_grid[0] != 10:
                predictor_warnings.append(f"  - tree_quantile_grid_size_grid = {tree_quantile_grid_size_grid} (TREE not in active predictors)")
        
        if "CLUSTERING" not in all_predictors:
            if len(clustering_random_state_grid) > 1 or clustering_random_state_grid[0] != 42:
                predictor_warnings.append(f"  - clustering_random_state_grid = {clustering_random_state_grid} (CLUSTERING not in active predictors)")
            if clustering_init_method_grid != ["k-means++"]:
                predictor_warnings.append(f"  - clustering_init_method_grid = {clustering_init_method_grid} (CLUSTERING not in active predictors)")
        
        if predictor_warnings:
            warnings.append("\nPredictor-specific parameter warnings:")
            warnings.extend(predictor_warnings)
    
    # Print or return warnings
    if warnings:
        if verbose:
            print("IGNORED PARAMETERS (will not be used):")
            print("-" * 60)
            for warning in warnings:
                print(warning)
            print()
            print("TIP: Set unused grids to [] or single default values to clarify intent")
            print("="*60)
            print()
    else:
        if verbose:
            print("✓ Configuration looks good! All set parameters will be used.")
            print("="*60)
            print()
    
    return warnings


# Example usage (commented out - uncomment to run validation when importing config):
# if __name__ != "__main__":
#     validate_config(verbose=True)


# ========== Configuration Validation Usage Examples ==========
"""
How to Use Configuration Validation:
====================================

The validate_config() function helps you verify that your parameter grids are properly
configured and warns about parameters that will be ignored.

Example 1: Quick validation check before running experiments
------------------------------------------------------------
```python
# In your script or notebook
from config import validate_config

# Check configuration and print warnings
validate_config()
```

Example 2: Programmatic validation for automated workflows
----------------------------------------------------------
```python
from config import validate_config

# Get warnings without printing
warnings = validate_config(verbose=False)

if warnings:
    print(f"Warning: {len(warnings)} parameters will be ignored!")
    print("Consider adjusting your configuration.")
    # Optionally abort or prompt user
else:
    print("Configuration validated successfully!")
    # Proceed with experiments
```

Example 3: Validation in main.py before generating parameter grid
-----------------------------------------------------------------
```python
# At the start of main()
from config import validate_config

def main():
    # Validate configuration before expensive grid generation
    print("Validating configuration...")
    validate_config(verbose=True)
    
    # Proceed with experiments
    param_combinations = create_parameter_grid()
    # ...
```

Example 4: Testing different configurations
-------------------------------------------
```python
# Temporarily modify config for testing
import config

# Save original
original_categories = config.method_categories_grid

# Test Exact-only configuration
config.method_categories_grid = ["Exact"]
print("Testing Exact-only configuration:")
config.validate_config()

# Restore original
config.method_categories_grid = original_categories
```

Benefits of Using validate_config():
------------------------------------
1. **Catch configuration errors early**: Before expensive computation starts
2. **Understand parameter interactions**: See which grids actually matter
3. **Optimize computational resources**: Don't sweep over irrelevant parameters
4. **Document intent**: Makes configuration decisions explicit
5. **Educational**: Learn the dependency structure by seeing warnings

Recommended Workflow:
--------------------
1. Configure your parameter grids in this file
2. Run validate_config() to check for issues
3. Review warnings and adjust unnecessary parameter sweeps
4. Run experiments with validated configuration
5. Re-validate if you change configuration during development
"""
