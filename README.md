# ML-Augmented Benders Decomposition for Stochastic Optimization

An advanced ML-augmented Benders Decomposition algorithm that integrates machine learning techniques with classical optimization to accelerate convergence in large-scale stochastic optimization problems. Currently supports Stochastic Capacitated Facility Location Problems (SCFLP), Network Expansion with OD Shortest-Path Recourse (NE-SP), Stochastic Uncapacitated Multicommodity Network Design (SUMND), and Stochastic Capacitated Multicommodity Network Design (SCMND).

## Table of Contents

- [About the Project](#about-the-project)
- [General Two-Stage Optimization Pattern](#general-two-stage-optimization-pattern)
- [ML-Augmented Algorithm Overview](#ml-augmented-algorithm-overview)
- [System Architecture](#system-architecture)
- [Current Implementation Status](#current-implementation-status)
- [Project Outputs and Results](#project-outputs-and-results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Documentation Structure](#documentation-structure)
- [Built With](#built-with)
- [Future Extensions](#future-extensions)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About the Project

This research project implements a **three-phase ML-augmented Benders Decomposition algorithm** that dramatically reduces computational burden in large-scale stochastic optimization while maintaining solution quality. The core innovation integrates sophisticated machine learning prediction techniques into the classical Benders decomposition framework, enabling efficient solution of problems with hundreds or thousands of scenarios.

### 🎯 Current Research Focus

**As of March 2026, this implementation supports four problem classes:**
- **Stochastic Capacitated Facility Location Problems (SCFLP)**: Stochastic two-stage optimization with scenario-based subproblems
- **Network Expansion with OD Shortest-Path Recourse (NE-SP)**: Deterministic two-stage optimization with OD-pair subproblems
- **Stochastic Uncapacitated Multicommodity Network Design (SUMND)**: Stochastic two-stage optimization with scenario-based multicommodity recourse
- **Stochastic Capacitated Multicommodity Network Design (SCMND)**: Stochastic two-stage optimization with scenario-based capacitated multicommodity recourse

These problem classes follow the general two-stage stochastic programming pattern described below, enabling unified treatment within the ML-augmented Benders framework.

### Key Innovation

The project addresses the fundamental scalability limitation of traditional Benders decomposition: the requirement to solve all subproblems exactly in each iteration becomes computationally prohibitive for large-scale stochastic programs. Our ML-augmented approach solves only a strategically selected subset of subproblems exactly while predicting dual solutions for the remainder, achieving **60-80% computational speedup** while maintaining solution quality within 1-2% of exact methods.

## General Two-Stage Optimization Pattern

This project investigates the following general two-stage stochastic programming pattern with complete recourse:

### Problem Formulation

$$
\min_{x, \{y^m\}} \quad c^\top x + \sum_{m \in \mathcal{M}} \omega_m \cdot q^\top y^m
$$

$$
\begin{aligned}
\text{subject to:} \quad & A x \leq b && \text{(first-stage constraints)} \\
& W y^m \leq h_m - T x && \text{(second-stage constraints, } \forall m \in \mathcal{M}\text{)} \\
& x \in X && \text{(first-stage domain)} \\
& y^m \geq 0 && \text{(second-stage domain, } \forall m \in \mathcal{M}\text{)}
\end{aligned}
$$

**Where:**
- $x$: first-stage decision variables (e.g., facility locations, arc activations)
- $c$: first-stage costs
- $y^m$: second-stage recourse variables for subproblem $m$
- $q$: second-stage costs (common across subproblems)
- $\mathcal{M}$: index set of subproblems (scenarios or OD pairs)
- $\omega_m$: weight of subproblem $m$ (probability or demand weight)
- $W, T, h_m$: recourse constraint matrices and RHS

**Key Property:** When $W$, $T$, and $q$ are identical across all subproblems and only $h_m$ varies, all dual subproblems share the same feasible region. This enables efficient ML prediction of dual solutions across subproblems.

### Multi-Cut Benders Master Problem

Introduce recourse variables $\theta_m$ to represent $Q_m(x)$ in the master:

$$
\min_{x, \{\theta_m\}} \quad \sum_{(i,j) \in A} c^\top x + \sum_{m \in \mathcal{M}} \omega_m \cdot \theta_m
$$

$$
\begin{aligned}
\text{subject to:} \quad & A x \leq b \\
& x \in X \\
& \theta_m \geq \text{(optimality cuts)} && \forall m \in \mathcal{M}
\end{aligned}
$$

### Primal Subproblem for Each $m$

$$
Q_m(x) = \min_{y^m} \quad q^\top y^m
$$

$$
\begin{aligned}
\text{subject to:} \quad & W y^m \leq h_m - T x \\
& y^m \geq 0
\end{aligned}
$$

### Dual Subproblem for Each $m$

Let $\pi^m$ be the dual variables for the recourse constraints. The dual formulation is:

$$
Q_m(x) = \max_{\pi^m} \quad (h_m - T x)^\top \pi^m
$$

$$
\begin{aligned}
\text{subject to:} \quad & W^\top \pi^m \leq q \\
& \pi^m \geq 0
\end{aligned}
$$

**Dual Feasible Region:** The set of $\pi^m$ satisfying $W^\top \pi^m \leq q, \pi^m \geq 0$ is **identical** for all subproblems when $W$ and $q$ are constant. Only the objective $(h_m - Tx)^\top \pi^m$ varies with $m$ and $x$.

### Optimality Cuts

Given an optimal dual solution $\bar{\pi}^m$ at the current $\bar{x}$, add:

$$
\theta_m \geq (h_m - T \bar{x})^\top \bar{\pi}^m + (T x)^\top \bar{\pi}^m = h_m^\top \bar{\pi}^m + (T(x - \bar{x}))^\top \bar{\pi}^m
$$

Simplified:
$$
\theta_m \geq h_m^\top \bar{\pi}^m - (T x)^\top \bar{\pi}^m
$$

### Problem Classes in This Project

**The currently supported classes follow this general pattern:**

- **SCFLP (Stochastic Capacitated Facility Location)**: 
  - First-stage: Facility opening decisions
  - Subproblems: One per scenario (demand realization)
  - Complete recourse through sufficient facility capacity

- **NE-SP (Network Expansion with OD Shortest-Path Recourse)**:
  - First-stage: Arc activation decisions
  - Subproblems: One per OD pair (origin-destination pair)
  - Complete recourse through supernode penalty structure

- **SUMND (Stochastic Uncapacitated Multicommodity Network Design)**:
   - First-stage: Arc activation decisions
   - Subproblems: One per stochastic scenario
   - Recourse: Multicommodity network flow with uncapacitated semantics via capacity handling

- **SCMND (Stochastic Capacitated Multicommodity Network Design)**:
   - First-stage: Arc activation decisions
   - Subproblems: One per stochastic scenario
   - Recourse: Multicommodity network flow with scenario capacities and bounded dual multipliers

For detailed problem-specific formulations, see [Network Expansion with OD Shortest-Path Recourse.md](Network%20Expansion%20with%20OD%20Shortest-Path%20Recourse.md), [Stochastic Uncapacitated Multicommodity Network Design.md](Stochastic%20Uncapacitated%20Multicommodity%20Network%20Design.md), and the respective data structure implementations in src/Problem/data.py.

## ML-Augmented Algorithm Overview

### Three-Phase Framework

The algorithm implements a sophisticated **three-phase strategy** that progressively builds from random exploration to intelligent ML-guided optimization:

#### Phase 0: Intelligent Warm-Up (Optional)
- **Purpose**: Generate diverse, high-quality Benders cuts to strengthen the master problem
- **Method**: Pure Random but Filtered solution generation with intelligent feasibility repair
- **Output**: Multiple random feasible first-stage solutions with exact dual subproblem solving
- **Benefit**: Significantly improved LP relaxation bounds before learning begins
- **Configuration**: Controlled by `m_trials` parameter (0 = skip Phase 0)

#### Phase 1: Traditional Learning 
- **Purpose**: Execute classical Benders decomposition to collect high-quality training data
- **Method**: Exact solution of all dual subproblems for optimal first-stage decisions
- **Output**: Optimal dual vectors from structured Benders iterations
- **Training Data**: Combined Phase 0 + Phase 1 duals for comprehensive ML training dataset
- **Configuration**: Controlled by `n_trials` parameter (typically 15-30 iterations)

#### Phase 2: ML-Augmented Performance
- **Purpose**: Accelerate convergence through intelligent subproblem selection and dual prediction
- **Method**: Solve selected subproblems exactly, predict duals for remainder using trained ML models
- **Selection Strategies**: KNN, TREE, and clustering-based optimal scenario selection
- **Prediction**: Graph-based feature engineering with multiple ML prediction methods
- **Fallback**: Automatic switching to exact solving when ML predictions fail

### Method Taxonomy and 6-Step Pattern

The system implements a unified **6-step execution pattern** that enables systematic definition of diverse optimization methods. All methods (Exact and ML) are defined through this pattern, with each method executing specific steps based on its algorithmic requirements.

#### 6-Step Execution Pattern

**Step 1 - Phase 0 (Warm-up):** Generate m_trials random feasible first-stage solutions and solve all dual subproblems exactly to collect initial training data.

**Step 2 - Phase 0 Cut Generation:** Build optimality cuts from Phase 0 dual solutions (number of cuts = scenarios × m_trials iterations) and add to master problem.

**Step 3 - Phase 1 (Learning):** Execute standard Benders decomposition with exact solving of all dual subproblems for n_trials iterations to collect high-quality training data.

**Step 4 - Feature Generation:** Generate graph-based features from dual solutions collected in Phase 0 and Phase 1 using Word2Vec embeddings.

**Step 5 - Partitioning & Training:** Partition scenarios into selected (sampled) and unselected (unsampled) groups, then train non-parametric predictor to map unselected scenarios to selected scenarios.

**Step 6 - Phase 2 (ML-Augmented Execution):** Solve selected scenarios exactly, predict duals for unselected scenarios, and continue until convergence.

#### Method Categories

The system implements two broad categories encompassing multiple algorithm variants:

##### Category 1: Exact Methods

**Standard Benders Decomposition** with all dual subproblems solved exactly at every iteration.

**Steps Executed:** Step 3 only (Phase 1 runs until convergence)
- No iteration cap - continues until Gurobi convergence (optimal solution, time limit, etc.)
- No training data collection or ML prediction
- Provides ground truth solutions for comparison with ML methods

**Variants:**
- `int_exact`: Cuts added only at integer solutions
- `int_and_LP_exact`: Cuts added at both integer solutions and LP relaxations

##### Category 2: ML Methods

**Five ML-based algorithmic approaches** that leverage machine learning to accelerate convergence:

**1. ML_Augmented_BD** (Feature-Based ML Augmentation)

Leverages graph-based features generated from dual solutions for intelligent scenario selection and prediction.

**Steps Executed:** All 6 steps (Steps 1-6)
- Step 1-3: Collect dual solutions from Phase 0 and Phase 1
- Step 4: Generate Word2Vec graph embeddings from dual solutions
- Step 5: Use features for scenario partitioning and predictor training
- Step 6: Solve selected scenarios exactly, predict unselected using trained predictor

**Feature Source Modes (`use_covariates_grid`, ML_Augmented_BD only):**
- `features_only`: use Step 4 generated features
- `features_plus_covariates`: concatenate Step 4 features with scenario covariates
- `covariates_only`: skip Step 4 and use covariate vectors directly

For non-`ML_Augmented_BD` ML methods, this grid is fixed internally to `features_only` to avoid duplicate no-op experiment rows.

**Supported Predictors:** KNN, TREE, CLUSTERING
- `KNN`: Assigns each unselected scenario to k nearest neighbors in feature space
- `TREE`: Builds adaptive decision tree partitioning with proportional leaf selection
- `CLUSTERING`: K-means clustering in feature space with representative selection

**2. ML_Augmented_BD_with_duals** (Direct Dual-Based ML)

Works directly with dual solution vectors without intermediate feature generation.

**Steps Executed:** Steps 1, 2, 3, 5, 6 (skips Step 4)
- Step 1-3: Collect dual solutions from Phase 0 and Phase 1
- **Skips Step 4**: No feature generation - works directly with dual vectors
- Step 5: Use dual vectors directly for scenario partitioning and predictor training
- Step 6: Solve selected scenarios exactly, predict unselected using trained predictor

**Supported Predictors:** KNN, TREE, CLUSTERING
- `KNN`: Assigns to k nearest neighbors in dual solution space
- `TREE`: Builds decision tree on dual vector space
- `CLUSTERING`: K-medoids clustering directly on dual solutions

**3. Covariate_Similarity** (Input Covariate-Based)

Leverages problem-specific covariate vectors provided in input data for scenario similarity.

**Steps Executed:** Steps 5, 6 only (skips Steps 1-4)
- **Skips Steps 1-4**: No Phase 0, Phase 1, or feature generation needed
- Step 5: Use covariate vectors from input data for partitioning and training
- Step 6: Solve selected scenarios exactly, predict unselected using covariate-based predictor

**Input Requirement:** Each scenario must have associated covariate vector in data file

**Supported Predictors:** KNN, TREE, CLUSTERING
- `KNN`: Assigns to k nearest neighbors in covariate space
- `TREE`: Builds decision tree on covariate features
- `CLUSTERING`: K-means clustering on covariate vectors

**4. Uncertainty_Similarity** (Uncertain Parameter-Based)

Uses uncertainty realizations (e.g., demand vectors, capacity vectors) for scenario similarity.

**Steps Executed:** Steps 5, 6 only (skips Steps 1-4)
- **Skips Steps 1-4**: No Phase 0, Phase 1, or feature generation needed
- Step 5: Use uncertainty vectors (demands, capacities) for partitioning and training
- Step 6: Solve selected scenarios exactly, predict unselected using uncertainty-based predictor

**Input Requirement:** Access to scenario-specific uncertainty realizations from data

**Supported Predictors:** KNN, TREE, CLUSTERING
- `KNN`: Assigns to k nearest neighbors in uncertainty space
- `TREE`: Builds decision tree on uncertainty parameters
- `CLUSTERING`: K-means clustering on uncertainty vectors

**5. Random_Sampling** (Bertsimas Random Sampling)

Pure random scenario sampling with statistical averaging - no training or fixed partitioning needed.

**Steps Executed:** Step 6 only (skips Steps 1-5)
- **Skips Steps 1-5**: No Phase 0, Phase 1, feature generation, or training
- Step 6 (Modified): Randomly sample scenarios at each iteration
  - Selected scenarios solved exactly
  - Unselected scenarios: predicted dual = average of all selected scenario duals
  - Random partition changes at every iteration (not fixed)

**No Predictor Training Required:** Uses simple averaging over randomly sampled scenarios

#### Callback Type Variants (Cut Addition Strategy)

Each method (Exact or ML) can be configured with different strategies for when to add cuts:

**Exact Method Variants:**
- `int_exact`: Cuts added only at integer first-stage solutions
- `int_and_LP_exact`: Cuts added at both integer solutions and LP relaxations

**ML Method Variants** (4 variants each for all 5 ML methods):
- `int_ML_LP_ML` (both): ML cuts for unselected scenarios at both integer and LP relaxation points; exact cuts for selected scenarios at both
- `int_ML_LP_exact` (int_only): ML cuts for unselected scenarios at integer points only, exact cuts at LP relaxations; exact cuts for selected scenarios at both
- `int_exact_LP_ML` (LP_only): ML cuts for unselected scenarios at LP relaxations only, exact cuts at integer points; exact cuts for selected scenarios at both
- `int_ML` (int_only_no_LP): ML cuts for unselected scenarios at integer points; no cuts added at LP relaxations

### Method Configuration Structure

As of February 2026, the system uses a unified `method_config` dictionary for scalable method definition:

```python
method_config = {
    # Core method identification
    'method_category': 'ML',  # or 'Exact'
    
    # Method type (for ML category)
    'ml_method_type': 'ML_Augmented_BD',  # or ML_Augmented_BD_with_duals, 
                                          # Covariate_Similarity, 
                                          # Uncertainty_Similarity, Random_Sampling
    
    # Predictor type (for ML methods 1-4)
    'predictor_type': 'KNN',  # or 'TREE', 'CLUSTERING'
    
    # Callback type (cut addition strategy)
    'ml_callback_type': 'int_ML_LP_ML',  # or int_ML_LP_exact, int_exact_LP_ML, int_ML
    'exact_callback_type': 'int_exact',  # or int_and_LP_exact (for Exact methods)
    
    # Phase iteration counts
    'n_trials': 5,     # Phase 1 iterations
    'm_trials': 0,     # Phase 0 iterations (0 = skip Phase 0)
    
    # Predictor-specific parameters
    'n_neighbors': 3,  # for KNN
    'tree_min_impurity_decrease': 0.0,  # for TREE
    'tree_quantile_grid_size': 10,      # for TREE
    'clustering_random_state': 42,      # for CLUSTERING
    
    # Other configuration
    'switching_strategy': 'no_switch',
   # Tri-state mode (only relevant for ML_Augmented_BD)
   'use_covariates': 'features_only',  # or 'features_plus_covariates', 'covariates_only'
    'use_stochastic_capacity': True,
    'n_selected_subproblems': 20,
    'feature_params': {
        'n_walks': 10,
        'walk_length': 80,
        'feature_dim': 128
    }
}
```

This unified structure enables:
- **Scalable Method Addition**: New methods defined by updating method_config structure
- **Clean Code Architecture**: Single source of truth for method configuration
- **Easy Method Switching**: Change method by updating dictionary keys
- **Systematic Experimentation**: Grid search over method configurations handled cleanly

### Implementation Summary

The current implementation provides:

**Method Categories:** 2 (Exact, ML)
**ML Method Types:** 5 (ML_Augmented_BD, ML_Augmented_BD_with_duals, Covariate_Similarity, Uncertainty_Similarity, Random_Sampling)
**Predictor Types:** 3 (KNN, TREE, CLUSTERING) for applicable ML methods
**Callback Variants:** 2 for Exact, 4 for ML methods
**Problem Classes:** 4 currently implemented (SCFLP, NE-SP, SUMND, SCMND), architecture stubs for future extensions (UFL, UHL, SSLP, CMND, MCFL)

All methods follow the unified 6-step pattern with selective step execution based on algorithmic requirements, managed through the scalable `method_config` dictionary architecture implemented in February 2026.

## System Architecture

### Core System Organization

The project is organized into **four main architectural layers**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Execution Framework                              │
│  main.py (orchestrator) + config.py (configuration) + phase0_generator  │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────────────┐
│                      Problem Module (Core Algorithm)                    │
│   callbacks.py + data.py + master_problem.py + subproblem_dual.py      │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────────────┐
│                       ML Prediction System                             │
│          prediction.py + training_and_selection.py                     │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────────────┐
│                    Supporting Systems                                   │
│    feature_engineering + data_generation + utils + testing             │
└─────────────────────────────────────────────────────────────────────────┘
```

### File Roles and Interactions

**Execution Framework** (*detailed in [run_README.md](src/run_README.md)*):
- **`main.py`**: System orchestrator providing multi-trial experiment execution, result aggregation, and comprehensive Excel output generation
- **`config.py`**: Intelligent configuration management with method-specific parameter optimization and validation
- **`phase0_generator.py`**: Warm-up solution generator implementing Pure Random but Filtered method with feasibility repair

**Problem Module** (*detailed in [core.md](src/Problem/core.md)*):
- **`callbacks.py`**: Three-phase ML-augmented callback system integrating with Gurobi's branch-and-bound framework
- **`data.py`**: SCFLP data structures and parsing with support for stochastic scenarios and capacity variations
- **`master_problem.py`**: Master problem formulations and solution coordination with Benders cut management
- **`subproblem_dual.py`**: Exact dual subproblem solvers providing ground truth solutions for training and validation

**ML Prediction System** (*detailed in [training.md](src/Dual_Prediction/training.md)*):
- **`prediction.py`**: Unified dual prediction architecture supporting multiple ML methods
- **`training_and_selection.py`**: One-time integrated scenario selection and predictor training

**Supporting Systems**:
- **Feature Engineering** (*detailed in [training.md](src/Dual_Prediction/training.md)*): Graph-based feature generation from dual solutions
- **Data Generation** (*detailed in [data_generation.md](src/utils/data_generation.md)*): Comprehensive SCFLP instance generation with four evolutionary versions
- **Testing Framework**: Comprehensive test suites for validation and regression testing

### Key Integration Patterns

**Primary Data Flow**:
```
Problem Data (data.py) → Three-Phase Execution (callbacks.py) → 
{Phase 0: Random Solutions, Phase 1: Exact Solving, Phase 2: ML Prediction} → 
Master Problem Integration (master_problem.py) → Result Analysis (main.py)
```

**ML Training Flow**:
```
Phase 0+1 Dual Collection → Feature Engineering → 
Scenario Selection (training_and_selection.py) → 
ML Model Training → Iterative Prediction (prediction.py)
```

## Current Implementation Status

### Fully Implemented and Tested
- **✅ SCFLP Problem Class**: Complete implementation with stochastic scenarios and capacity variations
- **✅ Three-Phase Algorithm**: All phases fully implemented and integrated
- **✅ KNN Prediction Method**: Optimal k-NN selection with integrated mathematical objective
- **✅ Phase 0 Warm-Up**: Pure Random but Filtered method with intelligent repair algorithms
- **✅ Configuration Management**: Intelligent parameter optimization and validation
- **✅ Result Analysis**: Comprehensive Excel output with statistical aggregation
- **✅ Data Generation**: Four evolutionary versions of SCFLP instance generation

### Current Research Focus Areas
- **🔬 ML Method Comparison**: Systematic evaluation of KNN vs TREE vs clustering approaches
- **🔬 Stochastic Capacity Extensions**: Advanced modeling of facility capacity uncertainty
- **🔬 Feature Engineering Optimization**: Graph-based dual similarity analysis and embeddings
- **🔬 Performance Scaling**: Optimization for problems with 100+ scenarios and 50+ facilities

### Implementation Stubs (Future Extensions)
- **🚧 Multi-Problem Support**: UFL, UHL, SSLP, CMND, MCFL problem classes (architecture ready)
- **🚧 Advanced ML Methods**: Deep learning and ensemble prediction approaches
- **🚧 Distributed Computing**: Multi-node execution and parallel processing

## Project Outputs and Results

### Excel Result Files (RESULTS/ directory)

The system generates comprehensive Excel files with multiple sheets providing detailed experimental analysis:

**Trial-Level Results Sheet**:
- **Solution Quality**: Optimal objective values, optimality gaps, first-stage solutions
- **Performance Metrics**: Total solving time, iterations, phase-specific timing statistics
- **Algorithm Statistics**: Cut generation by phase and method, ML prediction accuracy
- **Method Comparison**: Exact vs ML solution quality, switching strategy performance

**Summary Statistics Sheet**:
- **Aggregated Metrics**: Mean, standard deviation, min/max across all trials
- **Statistical Analysis**: Confidence intervals, performance distributions
- **Convergence Analysis**: Phase-specific convergence rates and efficiency metrics
- **Quality Assessment**: Solution quality preservation with ML augmentation

**Configuration Sheet**:
- **Complete Parameter Record**: All algorithm parameters for reproducibility
- **Experimental Metadata**: Execution timestamp, random seeds, environment information
- **Method-Specific Settings**: Prediction method configurations and hyperparameters

**Cut Analysis Sheet**:
- **Cut Generation Statistics**: By phase, method, and scenario
- **Cut Quality Metrics**: Cut strength, violation patterns, effectiveness analysis
- **Phase Contribution**: Relative contribution of Phase 0/1/2 to overall performance

**ML Performance Sheet** (when applicable):
- **Prediction Accuracy**: Dual prediction quality vs exact solutions
- **Computational Efficiency**: Time savings, prediction speed, memory usage
- **Method Comparison**: Performance across different ML prediction approaches

### Log Files (logs/ directory)

**Execution Logs**:
- **Algorithm Progress**: Detailed iteration-by-iteration execution tracking
- **Phase Transitions**: Timing and performance metrics for each phase
- **Cut Generation**: Individual cut addition with violation and improvement metrics
- **ML Training**: Feature generation, model training, and prediction setup logging

**Performance Logs**:
- **Resource Usage**: Memory consumption, CPU utilization, wall-clock timing
- **Solver Integration**: Gurobi solver interaction and optimization statistics
- **Error Handling**: Exception tracking, recovery mechanisms, and diagnostic information

**Debug Logs**:
- **Data Flow Tracking**: Inter-component communication and data passing
- **Algorithm Validation**: Mathematical verification and consistency checks
- **Research Analysis**: Detailed algorithm behavior for research investigation

### Generated Problem Instances (DATA/ directory)

The data generation utilities create comprehensive SCFLP test instances:

**Directory Structure**: `DATA/SCFLP/scenarios_s{scenarios}_cv{coeff_var}_dist{distribution}/`

**Instance Files**: 
- **Problem Data**: Facilities, customers, scenarios, costs, capacities
- **Stochastic Parameters**: Scenario-specific demands, capacity variations
- **Metadata**: Generation parameters, statistical properties, validation metrics

**Formats**:
- **Text Format**: Human-readable with comprehensive headers and diagnostics
- **Pickle Format**: Binary Python objects for efficient loading
- **Excel Format**: Structured spreadsheets for analysis and visualization

### Research Datasets and Analysis

**Training Data Collections**:
- **Dual Solution Archives**: Complete dual vectors from Phase 0 and Phase 1 
- **Feature Vector Databases**: Graph-based scenario features for ML training
- **Performance Benchmarks**: Comprehensive algorithm comparison datasets

**Analysis Results**:
- **Method Comparison Studies**: Statistical analysis across prediction methods
- **Scaling Analysis**: Performance characteristics vs problem size
- **Quality Preservation**: ML prediction accuracy and solution quality maintenance

## Getting Started

### Prerequisites

**Required Software**:
- Python 3.8 or higher
- Gurobi Optimizer (academic or commercial license)
- Required Python packages (see `requirements.txt`)

**System Requirements**:
- Minimum 8GB RAM (16GB+ recommended for large instances)
- Multi-core processor for parallel execution support

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Benders_Integration_with_ML.git
   cd Benders_Integration_with_ML
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Gurobi license**:
   - Ensure Gurobi license is properly installed and accessible
   - Test Gurobi installation: `python -c "import gurobipy; print(gurobipy.gurobi.version())"`

4. **Verify installation**:
   ```bash
   cd src
   python -c "from Problem.data import SCFLPData; print('Installation successful')"
   ```

## Usage

### Basic Execution

**Standard SCFLP Experiment**:
```bash
cd src
python main.py --problem_class SCFLP --instance_name p1 --prediction_method KNN --n_trials 20 --m_trials 10
```

**Method Comparison Study**:
```bash
# Compare different ML methods
python main.py --problem_class SCFLP --instance_name p1 --prediction_method KNN --n_trials 20
python main.py --problem_class SCFLP --instance_name p1 --prediction_method TREE --n_trials 20
python main.py --problem_class SCFLP --instance_name p1 --prediction_method DUAL_CLUSTERING --n_trials 20
```

**Phase Analysis**:
```bash
# Phase 0 + Phase 1 + Phase 2
python main.py --m_trials 15 --n_trials 20 --prediction_method KNN

# Skip Phase 0 (traditional approach)
python main.py --m_trials 0 --n_trials 20 --prediction_method KNN

# Phase 0 only (warm-up analysis)
python main.py --m_trials 15 --n_trials 0
```

### Configuration Options

**Key Parameters**:
- `--problem_class`: Problem type (currently: SCFLP)
- `--prediction_method`: ML method (KNN, TREE, DUAL_CLUSTERING, etc.)
- `--n_trials`: Phase 1 learning iterations (default: 20)
- `--m_trials`: Phase 0 warm-up solutions (default: 10)
- `--switching_strategy`: ML failure handling (no_switch, alternate, permanent_switch)
- `--use_stochastic_capacity`: Enable stochastic capacity scenarios

**Advanced Configuration**:
- Custom configuration files supported via `--config_file`
- Method-specific parameters (e.g., `--n_neighbors` for KNN)
- Output directory specification via `--output_dir`

### Testing

**Run Comprehensive Tests**:
```bash
cd tests
python run_tests.py
```

**Specific Component Tests**:
```bash
python test_phase0_integration.py    # Phase 0 warm-up testing
python test_knn_integration.py       # KNN method validation  
python test_data_compatibility.py    # Data format verification
```

## Documentation Structure

This project provides comprehensive documentation across multiple specialized files:

### Technical Documentation

- **[run_README.md](src/run_README.md)**: Execution framework documentation covering `main.py`, `config.py`, and `phase0_generator.py` with detailed implementation analysis
- **[core.md](src/Problem/core.md)**: Problem module documentation covering `callbacks.py`, `data.py`, `master_problem.py`, and `subproblem_dual.py` with mathematical formulations
- **[training.md](src/Dual_Prediction/training.md)**: ML prediction system documentation covering `prediction.py` and `training_and_selection.py` with algorithmic details
- **[data_generation.md](src/utils/data_generation.md)**: Data generation utilities documentation covering all four versions of SCFLP instance generation

### Implementation Guides

- **[feature_generation.md](src/feature_engineering/feature_generation.md)**: Graph-based feature engineering system with mathematical foundations
- **Configuration Examples**: Sample configuration files for different experimental scenarios
- **Test Documentation**: Comprehensive testing framework documentation and validation procedures

### Usage Examples

- **Tutorial Notebooks**: Step-by-step guides for different use cases (when available)
- **API Documentation**: Detailed function and class documentation within code
- **Research Examples**: Complete experimental setups for common research scenarios

## Built With

**Core Technologies**:
- **[Python 3.8+](https://python.org)**: Primary programming language
- **[Gurobi Optimizer](https://gurobi.com)**: Commercial optimization solver
- **[NumPy](https://numpy.org)**: Numerical computing foundation
- **[Pandas](https://pandas.pydata.org)**: Data analysis and Excel output generation
- **[Scikit-learn](https://scikit-learn.org)**: Machine learning algorithms and utilities

**Specialized Libraries**:
- **[NetworkX](https://networkx.org)**: Graph-based feature engineering
- **[Gensim](https://radimrehurek.com/gensim/)**: Word2Vec embeddings for scenario features
- **[SciPy](https://scipy.org)**: Scientific computing and statistical analysis
- **[Matplotlib](https://matplotlib.org)**: Visualization and plotting support

**Development Tools**:
- **[Logging](https://docs.python.org/3/library/logging.html)**: Comprehensive execution monitoring
- **[Pytest](https://pytest.org)**: Testing framework and validation
- **[Git](https://git-scm.com)**: Version control and collaboration

## Future Extensions

### Immediate Development Priorities

**Multi-Problem Support**:
- Extend implementation to UFL, UHL, SSLP, CMND, and MCFL problem classes
- Generalize ML prediction methods for different problem structures
- Develop problem-specific feature engineering approaches

**Advanced ML Integration**:
- Deep learning models for dual prediction
- Ensemble methods combining multiple prediction approaches  
- Online learning and adaptive model updates
- Reinforcement learning for dynamic scenario selection

### Research Directions

**Theoretical Advances**:
- Convergence analysis for ML-augmented Benders decomposition
- Approximation quality bounds for ML predictions
- Optimal scenario selection theory and algorithms

**Computational Enhancements**:
- Distributed computing and parallel processing
- GPU acceleration for ML training and prediction
- Memory optimization for large-scale problems
- Real-time optimization with streaming data

**Application Domains**:
- Supply chain optimization under uncertainty
- Energy systems planning and operation
- Transportation network design
- Financial portfolio optimization

## Contributing

We welcome contributions to this research project! Please see our contribution guidelines:

**Development Process**:
1. Fork the repository and create a feature branch
2. Implement changes with comprehensive testing
3. Update documentation to reflect changes
4. Submit a pull request with detailed description

**Code Standards**:
- Follow PEP 8 Python style guidelines
- Include comprehensive docstrings and comments
- Add unit tests for new functionality
- Update relevant documentation files

**Research Contributions**:
- Novel ML prediction methods for optimization
- Extensions to new problem classes
- Theoretical analysis and convergence proofs
- Performance improvements and optimizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Research Team**:
- Primary Investigator: [Your Name] - [your.email@university.edu]
- Research Institution: [University/Organization Name]
- Project Page: [https://github.com/your-username/Benders_Integration_with_ML]

**Academic Reference**:
Please cite this work in academic publications as:
```
[Your Name] et al. "ML-Augmented Benders Decomposition for Large-Scale Stochastic Optimization," 
Research Institution, 2026.
```

**Support and Issues**:
- Technical issues: [GitHub Issues](https://github.com/your-username/Benders_Integration_with_ML/issues)
- Research collaboration: Contact primary investigator
- Documentation improvements: Submit pull requests

---

*For detailed technical implementation, mathematical formulations, and specific component documentation, please refer to the specialized documentation files linked throughout this README.*