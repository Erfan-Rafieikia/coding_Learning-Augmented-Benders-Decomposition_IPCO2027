# SCFLP Data Generation: Comprehensive Documentation

This document provides comprehensive documentation for the evolution of SCFLP (Stochastic Capacitated Facility Location Problem) data generation utilities developed for the ML-augmented Benders Decomposition research project. The data generation pipeline has evolved through four versions, each adding sophisticated features for creating realistic stochastic problem instances with varying degrees of complexity and covariates.

## Introduction

The SCFLP data generation modules serve as the foundation for creating test datasets used throughout the ML-augmented Benders Decomposition research. Starting from a simple variance-based approach, the pipeline has evolved to include coefficient of variation (CV) parameterization, multiple distribution methods, covariate generation for ML feature engineering, and comprehensive prediction quality evaluation.

Each version builds upon previous functionality while introducing new capabilities:
- **Version 1**: Basic variance-based stochastic scenario generation
- **Version 2**: CV-driven parameterization with advanced distributions and covariate generation
- **Version 3**: Enhanced prediction method evaluation and improved metrics
- **Version 4**: Capacity variation and comprehensive ML method assessment

## Table of Contents

### Section 1: SCFLP_data_generation.py (Original Version)
1.1. **Big Picture**: Variance-based stochastic scenario generation  
1.2. **Detailed Implementation**: Core functions, scenario generation methods, and file processing

### Section 2: SCFLP_data_generation_version2.py 
2.1. **Big Picture**: CV-driven parameterization and covariate generation  
2.2. **Detailed Implementation**: New distribution methods, PCA-based covariates, and ARMA feature generation

### Section 3: SCFLP_data_generation_version3.py
3.1. **Big Picture**: Enhanced prediction evaluation and multiple ML methods  
3.2. **Detailed Implementation**: Comprehensive prediction assessment and improved covariate analysis

### Section 4: SCFLP_data_generation_version4.py
4.1. **Big Picture**: Capacity variation and complete ML evaluation framework  
4.2. **Detailed Implementation**: Advanced capacity scenarios and comprehensive prediction metrics

### Section 5: Evolution Summary and Integration
5.1. **Version Relationships**: How each version builds on previous functionality  
5.2. **Method Comparison**: Complete overview of all data generation methods  
5.3. **Output Formats**: Directory structure and file naming conventions  
5.4. **Integration Guidance**: Usage recommendations for different research scenarios

---

## Section 1: SCFLP_data_generation.py (Original Version)

### 1.1 Big Picture

**Purpose and Context:**
The original SCFLP data generation script provides the foundation for converting deterministic Capacitated Facility Location Problem (CFLP) instances into stochastic variants by generating multiple demand scenarios with controlled variance. This module serves as the entry point for creating test datasets for the ML-augmented Benders decomposition algorithm.

**Input Context:**
- **Source Files**: Deterministic CFLP instances in text format (e.g., p1, p2, ..., p71)
- **Input Directory**: `/DATA/SCFLP/` containing standard CFLP benchmark instances
- **File Format**: Text files with structured data: `num_facilities num_customers capacities fixed_costs demands shipment_costs`

**Output Context:**
- **Generated Scenarios**: Stochastic demand vectors maintaining problem feasibility
- **Output Format**: Text files or pickle files with complete stochastic SCFLP instances
- **Directory Structure**: Organized subdirectories based on generation parameters
- **Naming Convention**: `scenarios_s{num_scenarios}_v{variance}_m{method}/`

**Core Functionality:**
Creates stochastic scenarios by applying Normal or Uniform distributions to deterministic demands, ensuring capacity feasibility through scaling, and packaging results in standardized formats for downstream ML algorithm testing.

### 1.2 Detailed Implementation

#### Data Structure Definition

```python
@dataclass
class Data:
    I: np.ndarray              # Customer indices 
    J: np.ndarray              # Facility indices
    S: np.ndarray              # Scenario indices
    demands: np.ndarray        # Original deterministic demands
    scenarios: np.ndarray      # Generated stochastic demand scenarios (S×I matrix)
    capacities: np.ndarray     # Facility capacities
    fixed_costs: np.ndarray    # Fixed facility opening costs
    shipment_costs: np.ndarray # Transportation costs (I×J matrix)
    max_demand_sum_over_scenario: float  # Maximum total demand across scenarios
```

#### Core Scenario Generation Logic

**generate_scenarios() Function:**

**Function Signature:**
```python
def generate_scenarios(demands, capacities, num_scenarios=10, 
                      variance_factor=0.2, method="normal", seed=42):
    """
    Generate demand scenarios ensuring feasibility.
    
    Args:
        demands: np.ndarray - Original demand vector (μᵢ)ᵢ∈I
        capacities: np.ndarray - Facility capacity vector  
        num_scenarios: int - Number of scenarios S to generate
        variance_factor: float - Variance factor σ for demand generation
        method: str - Distribution method ("normal" or "uniform")
        seed: int - Random seed for reproducibility
    
    Returns:
        np.ndarray - Feasible demand scenarios (S×I matrix)
    """
```

**Mathematical Framework:**
The scenario generation applies distributional variation to deterministic demands while maintaining feasibility:

**Normal Distribution Method:**
For each scenario $s = 1,\ldots,S$ and customer $i \in I$, a raw demand $\tilde{d}_{i,s}$ is drawn:
$$\tilde{d}_{i,s} \sim \mathcal{N}(\mu_i, (\sigma \cdot \mu_i)^2)$$

Then truncated to ensure non-negativity:
$$d_{i,s} = \max\{\tilde{d}_{i,s}, 0\}$$

**Uniform Distribution Method:**
For symmetric interval around mean with variance factor $\sigma$:
$$\tilde{d}_{i,s} \sim \mathcal{U}(\mu_i(1-\sigma), \mu_i(1+\sigma))$$
$$d_{i,s} = \max\{\tilde{d}_{i,s}, 0\}$$

Where:
- $d_{i,s}$: Final demand for customer $i$ in scenario $s$
- $\tilde{d}_{i,s}$: Raw demand before non-negativity enforcement
- $\mu_i$: Original deterministic demand for customer $i$
- $\sigma$: Variance factor controlling relative variability

**Feasibility Enforcement Algorithm:**
Critical constraint ensuring solvability:
$$\max_s \sum_i d_{i,s} \leq \sum_j c_j$$

Where $c_j$ represents facility $j$ capacity.

**Step-by-Step Process:**
1. **Compute Total Demands**: $D_s = \sum_{i} d_{i,s}$ for each scenario $s$
2. **Check Constraint**: If $\max_s D_s > C$ where $C = \sum_j c_j$
3. **Apply Scaling**: Calculate scaling factor $\alpha = 0.95 \cdot \frac{C}{\max_s D_s}$
4. **Scale All Scenarios**: $d_{i,s} \leftarrow \alpha \cdot d_{i,s}$ for all $i,s$
5. **Verify Result**: Ensure $\max_s \sum_i d_{i,s} \leq 0.95 \cdot C$ (5% safety buffer)

**Implementation Details:**

1. **Scenario Generation Process:**
```python
if method == "normal":
    scenarios = np.array([
        [max(0, np.random.normal(mu, variance_factor * mu)) for mu in demands]
        for _ in range(num_scenarios)
    ])
elif method == "uniform":
    scenarios = np.array([
        [max(0, np.random.uniform(mu * (1 - variance_factor), 
                                 mu * (1 + variance_factor))) for mu in demands]
        for _ in range(num_scenarios)
    ])
```

2. **Feasibility Scaling Algorithm:**
```python
max_scenario_demand = np.max(np.sum(scenarios, axis=1))
if max_scenario_demand > total_capacity:
    safety_factor = 0.95  # 5% safety buffer
    scaling_factor = (total_capacity / max_scenario_demand) * safety_factor
    scenarios = scenarios * scaling_factor
```

**Mathematical Guarantees:**
- **Non-negativity**: All demands ≥ 0 through `max(0, ...)` clamping
- **Feasibility**: Post-scaling ensures $\sum_i d_{i,s} \leq 0.95 \sum_j c_j$ for all scenarios
- **Reproducibility**: Fixed random seed ensures deterministic generation

#### File Processing Pipeline

**process_files() Function:**

**Input Processing:**
1. **File Discovery**: Locates deterministic CFLP files in specified directory
2. **Data Parsing**: Uses `word_reader()` generator for memory-efficient parsing
3. **Structure Validation**: Ensures proper CFLP format compliance

**Batch Processing Logic:**
```python
folder_name = f"scenarios_s{num_scenarios}_v{variance}_m{method}"
output_dir = os.path.join(input_dir, folder_name)
```

**Output Generation:**

**Complete Text Format Structure:**

**Version 1 Format:**
```
# Stochastic CFLP Instance generated from {filename}
# Parameters:
#   scenarios = {num_scenarios}
#   variance  = {variance_factor}
#   method    = {method}
#   seed      = {random_seed}

{num_facilities} {num_customers} {num_scenarios}

# Facility capacities
{capacity_1} {capacity_2} ... {capacity_J}

# Fixed costs
{fixed_cost_1} {fixed_cost_2} ... {fixed_cost_J}

# Original deterministic demands
{demand_1} {demand_2} ... {demand_I}

# Shipment costs (customers × facilities)
{cost_11} {cost_12} ... {cost_1J}
{cost_21} {cost_22} ... {cost_2J}
...
{cost_I1} {cost_I2} ... {cost_IJ}

# Demand scenarios
# Scenario 1
{d_1,1} {d_2,1} ... {d_I,1}
# Scenario 2
{d_1,2} {d_2,2} ... {d_I,2}
...
# Scenario S
{d_1,S} {d_2,S} ... {d_I,S}

# Max demand sum over scenarios: {max_total_demand}
```

**Enhanced Format (Version 2+):**
```
# Stochastic CFLP Instance with CV parameterization
# Generated from: {original_filename}
# Method: {generation_method}
# Target CV: {target_coefficient_of_variation}
# Random seed: {seed_value}
# Generation timestamp: {iso_timestamp}

{num_facilities} {num_customers} {num_scenarios}

# [Previous sections identical to Version 1]

# Empirical Statistics:
# Empirical CV mean: {cv_mean}
# Empirical CV std: {cv_std}
# Empirical CV range: [{cv_min}, {cv_max}]
# R_overall dispersion: {r_overall_value}
# Negatives clipped: {negative_count} / {total_draws}
# Scaling factor applied: {scaling_factor}

# Method-specific parameters:
# {method_specific_params}
```

**Covariate-Enhanced Format (Version 2+ with covariates):**
```
# [Previous sections]

# Scenario-level covariates ({num_covariates} dimensions)
# Covariate matrix ({num_scenarios} × {num_covariates})
{z_1,1} {z_1,2} ... {z_1,m}
{z_2,1} {z_2,2} ... {z_2,m}
...
{z_S,1} {z_S,2} ... {z_S,m}

# Covariate predictiveness metrics:
# Average training R²: {avg_train_r2}
# Average validation MSE: {avg_val_mse}
# Average validation MAPE: {avg_val_mape}

# Method-specific prediction performance:
# Linear Regression - MAPE₁: {reg_mape1}, MAPEₑ: {reg_mape_eps}
# k-NN (k=2) - MAPE₁: {knn_mape1}, MAPEₑ: {knn_mape_eps}
# Clustering - MAPE₁: {cluster_mape1}, MAPEₑ: {cluster_mape_eps}
# Decision Tree - MAPE₁: {tree_mape1}, MAPEₑ: {tree_mape_eps}
```

**Capacity Variation Format (Version 4):**
```
# [Previous sections]

# Capacity scenarios ({num_facilities} × {num_scenarios})
# Scenario 1 capacities (deterministic baseline)
{c_1,1} {c_2,1} ... {c_J,1}
# Scenario 2 capacities
{c_1,2} {c_2,2} ... {c_J,2}
...
# Scenario S capacities
{c_1,S} {c_2,S} ... {c_J,S}

# Capacity statistics:
# Max total capacity: {max_total_capacity}
# Min total capacity: {min_total_capacity}
# Capacity scaling iterations: {capacity_scaling_count}

# Combined prediction metrics (demand + capacity):
# [All four methods with dual MAPE variants]

# Demand-only prediction metrics:
# [All four methods for demand prediction only]

# Capacity-only prediction metrics:
# [All four methods for capacity prediction only]
```

**Pickle Format:**
Serialized `Data` object preserving all numerical precision and array structures for efficient loading in research workflows.

#### Configuration and Execution

**Parameter Ranges:**
- **Scenarios**: [100, 400, 800] - Multiple scales for testing scalability
- **Variance**: [0.1, 0.3, 0.5, 1.0, 2.0] - Range from low to high uncertainty  
- **Methods**: ["normal", "uniform"] - Different distributional assumptions
- **Instances**: p1-p71 - Comprehensive coverage of benchmark problems

**Quality Assurance:**
- Real-time feasibility monitoring with detailed logging
- Exception handling for malformed input files
- Comprehensive progress reporting during batch processing
- Validation of generated instances before output

**Computational Complexity:**
- **Time**: O(S × I) for scenario generation where S = scenarios, I = customers
- **Space**: O(S × I) for scenario storage
- **Scaling**: Linear with number of scenarios and customers

---

## Section 2: SCFLP_data_generation_version2.py

### 2.1 Big Picture

**Evolutionary Improvements:**
Version 2 represents a significant advancement from the original variance-based approach by introducing **Coefficient of Variation (CV) parameterization** as the primary control mechanism. This change provides more principled control over stochastic uncertainty and enables consistent comparison across different problem scales and demand magnitudes.

**New Conceptual Framework:**
- **CV-Driven Design**: All stochastic generation controlled by target CV = σ/μ rather than absolute variance
- **Distribution Expansion**: Addition of Gamma, Lognormal, and specialized Merve_Normal methods
- **Covariate Integration**: Introduction of PCA and ARMA-based covariate generation for ML feature engineering
- **Advanced Metrics**: Empirical CV tracking, R_overall dispersion measure, and predictiveness evaluation

**Enhanced Input Processing:**
- **Backward Compatibility**: Maintains support for original deterministic CFLP files
- **Parameter Richness**: Expanded configuration options for different research scenarios
- **Method Specialization**: Separate functions for different covariate generation approaches

**Sophisticated Output Structure:**
- **Extended Metadata**: Comprehensive tracking of generation statistics and quality metrics
- **Covariate Vectors**: Optional PCA or ARMA-based feature vectors for ML algorithm input
- **Predictiveness Analysis**: R², MSE, and MAPE metrics for covariate quality assessment
- **Excel Integration**: Detailed tracking spreadsheets for experimental analysis

### 2.2 Detailed Implementation

**Similarities with Version 1:**
- Core `Data` structure foundation (extended with additional fields)
- Basic `word_reader()` file parsing mechanism
- Text and pickle output format support
- Batch processing pipeline structure

**Key Architectural Changes:**

#### Enhanced Data Structure

```python
@dataclass  
class Data:
    # Original fields from Version 1
    I, J, S, demands, scenarios, capacities, fixed_costs, shipment_costs, max_demand_sum_over_scenario
    
    # New CV-based metadata
    distribution: str                    # Generation method identifier
    target_cv: float                    # Desired coefficient of variation
    empirical_cv_mean: float           # Achieved CV (mean across customers)
    empirical_cv_std: float            # CV variability measure
    empirical_cv_min: float            # Minimum CV achieved
    empirical_cv_max: float            # Maximum CV achieved
    num_negatives_clipped: int         # Count of negative values clamped to zero
    total_draws: int                   # Total random samples generated
    scaling_factor_applied: float      # Capacity feasibility scaling factor
    r_overall: float                   # Vector-level dispersion metric
    
    # Method-specific parameters
    sigma_lower: float = 0.0           # Merve_Normal lower bound parameter
    sigma_upper: float = 0.0           # Merve_Normal upper bound parameter
    
    # Covariate generation results
    covariate_vectors: Optional[np.ndarray] = None     # Generated feature vectors
    num_covariates: int = 0                           # Number of covariate dimensions
    arma_r2_scores: Optional[np.ndarray] = None       # Training R² per customer
    arma_validation_r2: Optional[np.ndarray] = None   # Validation R² per customer  
    arma_validation_mse: Optional[np.ndarray] = None  # Validation MSE per customer
    arma_validation_mape: Optional[np.ndarray] = None # Validation MAPE per customer
```

#### CV-Driven Scenario Generation

**Theoretical Foundation:**
Coefficient of Variation provides scale-invariant measure of relative variability:
$$CV = \frac{\sigma}{\mu}$$

This enables consistent uncertainty specification across problems with different demand scales.

**New Distribution Methods:**

**1. Normal with CV Control:**
For each scenario $s$ and customer $i$:
$$\tilde{d}_{i,s} \sim \mathcal{N}(\mu_i, (CV_{target} \cdot \mu_i)^2)$$
$$d_{i,s} = \max\{\tilde{d}_{i,s}, 0\}$$

**2. Uniform with CV Control:**
For uniform distribution with target CV, using mathematical relationship:
- Mean: $\mu = (a+b)/2$  
- Variance: $\sigma^2 = (b-a)^2/12$
- CV constraint: $CV = \sigma/\mu = (b-a)/(\sqrt{12} \cdot (a+b))$

Solving for symmetric interval parameters:
$$\delta_i = \mu_i \cdot CV_{target} \cdot \sqrt{3}$$
$$\tilde{d}_{i,s} \sim \mathcal{U}(\mu_i - \delta_i, \mu_i + \delta_i)$$
$$d_{i,s} = \max\{\tilde{d}_{i,s}, 0\}$$

**3. Gamma Distribution (Guaranteed Non-negative):**
Using shape-scale parameterization ensuring exact CV matching:
- Shape parameter: $k = \frac{1}{CV_{target}^2}$  
- Scale parameter: $\theta_i = \frac{\mu_i}{k} = \mu_i \cdot CV_{target}^2$

$$d_{i,s} \sim \text{Gamma}(k, \theta_i)$$

**Mathematical Verification:**
- $E[d_{i,s}] = k \cdot \theta_i = \mu_i$ ✓
- $\text{Var}[d_{i,s}] = k \cdot \theta_i^2 = (CV_{target} \cdot \mu_i)^2$ ✓
- $CV = \sqrt{\text{Var}}/E = CV_{target}$ ✓

**4. Lognormal Distribution:**
Using moment-based parameterization for exact CV control:
$$s^2 = \ln(1 + CV_{target}^2), \quad s = \sqrt{s^2}$$
$$m_i = \ln(\mu_i) - \frac{s^2}{2}$$
$$d_{i,s} \sim \text{Lognormal}(m_i, s)$$

**Mathematical Properties:**
- $E[d_{i,s}] = \exp(m_i + s^2/2) = \mu_i$ ✓
- $CV[d_{i,s}] = \sqrt{\exp(s^2) - 1} = CV_{target}$ ✓
- Always positive (no truncation needed)

**5. Merve_Normal Method (Heteroscedastic):**
Custom method with per-customer variance heterogeneity:

**Step 1:** Generate per-customer standard deviations
$$\sigma_i \sim \mathcal{U}(\sigma_{lower} \cdot \mu_i, \sigma_{upper} \cdot \mu_i)$$

**Step 2:** Generate demands with customer-specific variance
$$\tilde{d}_{i,s} \sim \mathcal{N}(\mu_i, \sigma_i^2)$$
$$d_{i,s} = \max\{\tilde{d}_{i,s}, 0\}$$

**Key Feature:** Each customer has different coefficient of variation:
$$CV_i = \frac{\sigma_i}{\mu_i} \in [\sigma_{lower}, \sigma_{upper}]$$

**Implementation Example:**
```python
def _draw_with_target_cv(means, target_cv, method, rng, sigma_lower=0.1, sigma_upper=0.3):
    if method == "normal":
        sigma = target_cv * means
        draws = rng.normal(loc=means, scale=sigma)
    elif method == "gamma":
        shape = 1.0 / (target_cv ** 2)
        scale = means * (target_cv ** 2)  
        draws = rng.gamma(shape=shape, scale=scale)
    # ... other methods
    
    neg_count = int(np.sum(draws < 0))
    draws = np.maximum(draws, 0.0)  # Ensure non-negativity
    return draws, neg_count
```

#### Advanced Covariate Generation

**PCA-Based Covariates (Merve_With_Covariates):**

**Complete Mathematical Framework:**
Given scenario-customer demand matrix $Y \in \mathbb{R}^{S \times I}$ where $Y_{s,i} = d_{i,s}$:

**Step 1: Data Centering**
Compute per-customer empirical means:
$$\bar{d}_i = \frac{1}{S}\sum_{s=1}^{S} Y_{s,i}$$

Center matrix across scenarios (per customer):
$$\tilde{Y}_{s,i} = Y_{s,i} - \bar{d}_i$$
$$\tilde{Y} = Y - \mathbf{1}_S \bar{d}^T \in \mathbb{R}^{S \times I}$$

**Step 2: Singular Value Decomposition**
$$\tilde{Y} = U\Sigma V^T$$
Where:
- $U \in \mathbb{R}^{S \times \min(S,I)}$: Left singular vectors (scenario loadings)
- $\Sigma \in \mathbb{R}^{\min(S,I) \times \min(S,I)}$: Singular values (ordered decreasing)
- $V \in \mathbb{R}^{I \times \min(S,I)}$: Right singular vectors (customer loadings)

**Step 3: Principal Component Selection**
Determine effective dimensionality:
$$m_{eff} = \min(m_{target}, \text{rank}(\tilde{Y}), S)$$

Extract scenario-level features:
$$Z = U_{[:, 1:m_{eff}]} \in \mathbb{R}^{S \times m_{eff}}$$

**Mathematical Properties:**
- **Orthonormality**: $Z^T Z = I_{m_{eff}}$ (uncorrelated features)
- **Variance Maximization**: Components ordered by explained variance $\sigma_j^2$
- **Dimensionality Reduction**: Projects $S \times I$ demand space to $S \times m_{eff}$ feature space
- **Scenario Interpretation**: Row $Z_s$ captures dominant demand patterns for scenario $s$

**Feature Interpretation:**
- **First Component**: Captures largest variation pattern across customers
- **Subsequent Components**: Capture orthogonal variation patterns in decreasing importance
- **Reconstruction**: $\tilde{Y} \approx Z \cdot (U_{[:, 1:m_{eff}]})^T$ (low-rank approximation)

**ARMA-Based Covariates:**

**Complete Mathematical Framework:**
Multivariate ARMA(2,2) process generating scenario-level features:
$$Z_t = \Phi_1 Z_{t-1} + \Phi_2 Z_{t-2} + U_t + \Theta_1 U_{t-1} + \Theta_2 U_{t-2}$$

**Parameter Specifications:**
- $Z_t \in \mathbb{R}^d$: Covariate vector at time $t$ (scenario $t$)
- $\Phi_1, \Phi_2 \in \mathbb{R}^{d \times d}$: Autoregressive coefficient matrices
- $\Theta_1, \Theta_2 \in \mathbb{R}^{d \times d}$: Moving average coefficient matrices  
- $U_t \sim \mathcal{N}(0, \Sigma_U)$: Innovation vectors with $\Sigma_U \in \mathbb{R}^{d \times d}$ positive definite

**Stability Conditions:**
For stationarity, eigenvalues of companion matrix must satisfy $|\lambda| < 1$:
$$C = \begin{bmatrix} \Phi_1 & \Phi_2 \\ I_d & 0_{d \times d} \end{bmatrix} \in \mathbb{R}^{2d \times 2d}$$

**Demand Generation from ARMA Features:**
Given generated covariates $\{Z_s\}_{s=1}^S$, demands are constructed as:

For each customer $i$ and scenario $s$:
1. **Linear Component**: $\text{linear}_{i,s} = A_i^T (Z_s + \delta_i/4)$
2. **Stochastic Component**: $\text{variance}_{i,s} = (B_i^T Z_s) \cdot \varepsilon_s$ where $\varepsilon_s \sim \mathcal{N}(0,1)$
3. **Raw Demand**: $\tilde{d}_{i,s} = \text{linear}_{i,s} + \text{variance}_{i,s}$
4. **Non-negative**: $d_{i,s} = \max\{\tilde{d}_{i,s}, 0\}$
5. **Mean Matching**: Scale to match deterministic mean $\mu_i$

Where $A_i, B_i, \delta_i \in \mathbb{R}^d$ are customer-specific random vectors.

**Implementation Details:**

1. **Matrix Construction:**
```python
def _create_stable_arma_matrices(d, rng):
    # Diagonal dominance for stability
    phi1_diag = rng.uniform(0.3, 0.5, d)
    phi2_diag = rng.uniform(0.1, 0.3, d)
    
    # Ensure stability: diagonal sums < 1
    total_diag = phi1_diag + phi2_diag
    scale_factor = np.where(total_diag >= 0.9, 0.8 / total_diag, 1.0)
    
    # Construct matrices with stability guarantee
    Phi1 = np.diag(phi1_diag * scale_factor)
    Phi2 = np.diag(phi2_diag * scale_factor)
```

2. **Covariate Generation:**
```python
def _generate_arma_features(num_scenarios, num_covariates, rng):
    Phi1, Phi2, Theta1, Theta2, Sigma_U = _create_stable_arma_matrices(num_covariates, rng)
    
    # Generate innovation sequence
    innovations = rng.multivariate_normal(
        mean=np.zeros(num_covariates), 
        cov=Sigma_U, 
        size=num_scenarios + 10  # Extra for burn-in
    )
    
    # ARMA process generation with proper initialization
    features = np.zeros((num_scenarios + 10, num_covariates))
    
    for t in range(2, num_scenarios + 10):
        features[t] = (Phi1 @ features[t-1] + Phi2 @ features[t-2] + 
                      innovations[t] + Theta1 @ innovations[t-1] + 
                      Theta2 @ innovations[t-2])
    
    return features[-num_scenarios:]  # Remove burn-in period
```

#### Quality Metrics and Validation

**Empirical CV Computation:**
$$CV_{empirical,i} = \frac{\sqrt{\frac{1}{S-1}\sum_{s=1}^S (d_{i,s} - \bar{d}_i)^2}}{\bar{d}_i}$$

where $\bar{d}_i = \frac{1}{S}\sum_{s=1}^S d_{i,s}$

**R_Overall Dispersion Metric:**
$$R_{overall} = \frac{1}{I}\sum_{i=1}^I \frac{\max_s d_{i,s} - \min_s d_{i,s}}{\bar{d}_i}$$

This provides a single scalar measure of demand variability across all customers.

**Predictiveness Evaluation:**
For each customer i, fit linear regression: $d_i = Z\beta_i + \epsilon_i$

Compute validation metrics:
- **R²**: $R^2_i = 1 - \frac{SS_{res}}{SS_{tot}}$
- **MSE**: $MSE_i = \frac{1}{n_{test}}\sum(y_{true} - y_{pred})^2$  
- **MAPE**: $MAPE_i = \frac{100\%}{n_{test}}\sum\frac{|y_{true} - y_{pred}|}{\max(|y_{true}|, 1.0)}$

**New Output Features:**
- Excel tracking files with comprehensive statistics
- Method-specific parameter logging
- Covariate quality assessment reports
- Comparative analysis across different CV levels

---

## Section 3: SCFLP_data_generation_version3.py

### 3.1 Big Picture

**Refinement Focus:**
Version 3 maintains the CV-driven architecture from Version 2 while significantly enhancing the **prediction evaluation framework**. The primary advancement lies in implementing comprehensive ML method assessment for covariate quality, moving beyond simple linear regression to include k-NN, clustering-based, and decision tree approaches.

**Enhanced Evaluation Philosophy:**
- **Multi-Method Assessment**: Evaluation using diverse ML approaches rather than linear regression only
- **Robust Metrics**: Introduction of dual MAPE formulations to handle different scales and edge cases
- **Clustering Intelligence**: Automatic cluster number selection using silhouette analysis
- **Comprehensive Comparison**: Side-by-side evaluation of multiple prediction paradigms

**Maintained Architectural Integrity:**
- **Core Generation Logic**: Unchanged CV-based scenario generation from Version 2
- **Distribution Methods**: Same five distribution options (normal, uniform, gamma, lognormal, merve_normal)
- **Covariate Framework**: Preserved PCA and ARMA covariate generation approaches
- **Output Compatibility**: Maintains backward compatibility with Version 2 output formats

### 3.2 Detailed Implementation

**Unchanged Components from Version 2:**
- `Data` structure definition (identical field structure)
- Core CV-based distribution functions (`_draw_with_target_cv`, `generate_scenarios_cv`)
- Merve_Normal generation logic (`generate_scenarios_merve_normal`)
- ARMA covariate generation (`generate_scenarios_arma_covariate`)
- File processing pipeline and output formatting

**Key Enhancements:**

#### Advanced MAPE Computation

**Dual MAPE Formulation:**
Version 3 introduces two MAPE variants to handle different scaling scenarios:

**MAPE₁ (Original):**
$$MAPE_1 = \frac{100\%}{n}\sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{\max(|y_i|, 1.0)}$$

**MAPEₑ (Epsilon-based):**
$$MAPE_\epsilon = \frac{100\%}{n}\sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{\max(|y_i|, \epsilon)}$$

**Mathematical Rationale:**
- **MAPE₁**: Prevents division by zero but may underestimate error for small demands
- **MAPEₑ**: More sensitive to prediction errors for small values (typically ε = 0.1)
- **Complementary Information**: Together provide comprehensive error assessment

**Implementation:**
```python
def _compute_two_mapes(y_true, y_pred, epsilon):
    absolute_errors = np.abs(y_true - y_pred)
    
    # Original: denominator clipped at 1.0
    denom1 = np.maximum(np.abs(y_true), 1.0)
    mape1 = np.mean(absolute_errors / denom1) * 100.0
    
    # Enhanced: denominator clipped at epsilon
    denom_eps = np.maximum(np.abs(y_true), epsilon)
    mape_eps = np.mean(absolute_errors / denom_eps) * 100.0
    
    return mape1, mape_eps
```

#### Intelligent Clustering Method

**Silhouette-Based Cluster Selection:**
Automatic determination of optimal cluster count K* through silhouette analysis:

$$S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$: Average distance to points in same cluster
- $b(i)$: Average distance to nearest neighboring cluster
- $S(i) \in [-1, 1]$: Silhouette coefficient for point i

**Algorithm:**
```python
def _select_optimal_kmeans(X_train, random_state, k_min=2, k_max=10):
    best_model = None
    best_score = -np.inf
    
    for k in range(k_min, min(k_max, len(X_train)-1) + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_train)
        
        if len(np.unique(labels)) > 1:  # Ensure multiple clusters
            score = silhouette_score(X_train, labels)
            if score > best_score:
                best_score = score
                best_model = kmeans
    
    return best_model
```

#### Comprehensive Prediction Method Evaluation

**Multi-Method Assessment Framework:**
Each customer's demand vector is evaluated using four distinct ML approaches:

**1. Linear Regression (Baseline Method):**
**Model Specification:**
$$\hat{d}_i = Z\beta_i + \epsilon_i$$

Where:
- $Z \in \mathbb{R}^{S \times m}$: Covariate matrix (scenarios × features)
- $\beta_i \in \mathbb{R}^m$: Customer-specific regression coefficients
- $\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$: Customer-specific error term

**Training Protocol:**
- **Method**: Ordinary Least Squares (OLS)
- **Train/Test Split**: 80% / 20% random split
- **Evaluation Metrics**: Training R², Test MSE, Test MAPE (dual formulations)

**2. k-Nearest Neighbors (k=2):**
**Algorithm Specification:**
$$\hat{d}_{i,s} = \frac{1}{k}\sum_{j \in N_k(Z_s)} d_{i,j}$$

Where:
- $N_k(Z_s)$: Set of $k$ nearest training scenarios to test scenario $s$ in covariate space
- Distance metric: Euclidean distance $||Z_s - Z_j||_2$
- $k = 2$: Fixed neighborhood size

**Implementation Details:**
- **Distance Computation**: $d(Z_s, Z_j) = \sqrt{\sum_{l=1}^m (Z_{s,l} - Z_{j,l})^2}$
- **Tie Breaking**: Deterministic ordering by scenario index
- **Prediction**: Simple average of $k$ nearest neighbors' demand values

**3. Clustering-Based Prediction with Automatic K Selection:**
**Optimal Cluster Selection:**
$$K^* = \arg\max_{K \in [2, K_{max}]} \text{SilhouetteScore}(\text{KMeans}(Z, K))$$

Where Silhouette Score for point $i$:
$$S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- $a(i)$: Average distance to points in same cluster
- $b(i)$: Average distance to nearest neighboring cluster
- $S(i) \in [-1, 1]$: Higher is better

**Prediction Algorithm:**
1. **Cluster Training Data**: Apply K-means with optimal $K^*$
2. **Compute Cluster Means**: $\bar{d}_{i,c} = \frac{1}{|C_c|} \sum_{j \in C_c} d_{i,j}$ for cluster $c$
3. **Test Assignment**: Assign test scenario to nearest cluster centroid
4. **Prediction**: $\hat{d}_{i,s} = \bar{d}_{i,c(s)}$ where $c(s)$ is assigned cluster

**4. Decision Tree Regression:**
**Model Configuration:**
$$\hat{d}_i = \text{DecisionTreeRegressor}(Z; \text{max\_depth}=5, \text{min\_samples\_split}=2, \text{random\_state}=42)$$

**Tree Construction:**
- **Splitting Criterion**: MSE minimization
- **Feature Selection**: Considers all $m$ covariate features
- **Pruning**: Controlled by max_depth and min_samples_split
- **Prediction**: Leaf mean for assigned region

**Advantage**: Captures non-linear relationships and feature interactions automatically through recursive partitioning.

**Comprehensive Evaluation Protocol:**
For each customer $i$ and each method:
1. **Data Split**: 80% training, 20% testing (stratified by scenario order)
2. **Model Training**: Fit method-specific model on training data
3. **Prediction**: Generate predictions for test scenarios
4. **Metric Computation**: Calculate dual MAPE formulations on test set
5. **Aggregation**: Average metrics across all customers for method comparison

**Evaluation Protocol:**
```python
def _evaluate_prediction_methods(features, demands, test_size=0.2, epsilon=0.1):
    X_train, X_test, y_train, y_test = train_test_split(
        features, demands, test_size=test_size, random_state=42
    )
    
    # Method 1: Linear Regression
    reg = LinearRegression().fit(X_train, y_train)
    reg_pred = reg.predict(X_test)
    reg_train_r2 = reg.score(X_train, y_train)
    reg_test_mse = mean_squared_error(y_test, reg_pred)
    reg_mape1, reg_mape_eps = _compute_two_mapes(y_test, reg_pred, epsilon)
    
    # Method 2: k-NN (k=2)
    knn = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_mape1, knn_mape_eps = _compute_two_mapes(y_test, knn_pred, epsilon)
    
    # Method 3: Clustering-based
    kmeans_model = _select_optimal_kmeans(X_train, random_state=42)
    if kmeans_model:
        train_clusters = kmeans_model.predict(X_train)
        test_clusters = kmeans_model.predict(X_test)
        cluster_means = np.array([y_train[train_clusters == k].mean() 
                                 for k in range(kmeans_model.n_clusters)])
        cluster_pred = cluster_means[test_clusters]
    else:
        cluster_pred = np.full(len(y_test), np.mean(y_train))
    
    cluster_mape1, cluster_mape_eps = _compute_two_mapes(y_test, cluster_pred, epsilon)
    
    # Method 4: Decision Tree
    tree = DecisionTreeRegressor(max_depth=5, min_samples_split=2, random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    tree_mape1, tree_mape_eps = _compute_two_mapes(y_test, tree_pred, epsilon)
    
    return (reg_train_r2, reg_test_mse, reg_mape1, reg_mape_eps,
            knn_mape1, knn_mape_eps, cluster_mape1, cluster_mape_eps,
            tree_mape1, tree_mape_eps)
```

#### Enhanced PCA Covariate Analysis

**Expanded Metrics Integration:**
The `generate_scenarios_merve_with_covariates` function now incorporates the comprehensive evaluation framework:

```python
def generate_scenarios_merve_with_covariates(demands, capacities, num_scenarios, 
                                           sigma_lower=0.1, sigma_upper=0.3,
                                           num_covariates=3, seed=42):
    # Step A: Generate scenarios (unchanged from Version 2)
    scenarios, merve_info = generate_scenarios_merve_normal(
        demands, capacities, num_scenarios, sigma_lower, sigma_upper, seed
    )
    
    # Step B: PCA covariate construction (unchanged from Version 2)
    Y = scenarios
    y_bar = np.mean(Y, axis=0)
    Y_tilde = Y - y_bar[np.newaxis, :]
    U, sigma, Vt = np.linalg.svd(Y_tilde, full_matrices=False)
    
    rank_Y_tilde = np.linalg.matrix_rank(Y_tilde)
    m_eff = min(num_covariates, rank_Y_tilde, num_scenarios)
    Z = U[:, :m_eff]
    
    # Step C: Enhanced evaluation using four methods
    all_metrics = []
    for i in range(len(demands)):
        customer_demands = scenarios[:, i]
        metrics = _evaluate_prediction_methods(Z, customer_demands, 
                                              test_size=0.2, epsilon=0.1)
        all_metrics.append(metrics)
    
    # Step D: Aggregate comprehensive statistics
    metrics_array = np.array(all_metrics)
    
    # Update info dictionary with all method results
    info = merve_info.copy()
    info.update({
        "num_covariates": m_eff,
        # Linear regression metrics
        "r2_mean": float(np.mean(metrics_array[:, 0])),
        "r2_std": float(np.std(metrics_array[:, 0])),
        "test_mse_mean": float(np.mean(metrics_array[:, 1])),
        "test_mse_std": float(np.std(metrics_array[:, 1])),
        
        # MAPE metrics for all methods (both MAPE variants)
        "reg_mape1_mean": float(np.mean(metrics_array[:, 2])),
        "reg_mape_eps_mean": float(np.mean(metrics_array[:, 3])),
        "knn_mape1_mean": float(np.mean(metrics_array[:, 4])),
        "knn_mape_eps_mean": float(np.mean(metrics_array[:, 5])),
        "cluster_mape1_mean": float(np.mean(metrics_array[:, 6])),
        "cluster_mape_eps_mean": float(np.mean(metrics_array[:, 7])),
        "tree_mape1_mean": float(np.mean(metrics_array[:, 8])),
        "tree_mape_eps_mean": float(np.mean(metrics_array[:, 9])),
        
        # Store per-customer detailed results for analysis
        "detailed_metrics": metrics_array  # For advanced analysis
    })
    
    return scenarios, Z, info
```

**Research Value:**
This enhanced evaluation provides researchers with comprehensive assessment of covariate quality across multiple ML paradigms, enabling informed decisions about which covariate generation method works best for specific problem characteristics.

---

## Section 4: SCFLP_data_generation_version4.py

### 4.1 Big Picture

**Advanced Integration:**
Version 4 represents the most sophisticated iteration, expanding beyond demand-only variation to include **stochastic capacity scenarios**. This version introduces comprehensive dual-variation (demand + capacity) modeling while maintaining all previous functionality and adding extensive ML evaluation frameworks for both individual and combined uncertainty sources.

**Dual-Source Stochasticity:**
- **Demand Variation**: Inherited from previous versions with all distribution methods
- **Capacity Variation**: New capability for generating stochastic facility capacities
- **Combined Analysis**: Joint evaluation of demand and capacity uncertainty effects
- **Comprehensive Metrics**: Separate evaluation of demand-only vs. combined prediction performance

**Enhanced Research Framework:**
- **Method Comparison**: Side-by-side analysis of different uncertainty modeling approaches
- **Prediction Assessment**: Comprehensive evaluation across linear, non-linear, and ensemble methods
- **Experimental Design**: Complete framework for systematic covariate generation studies
- **Output Sophistication**: Rich metadata and tracking for complex experimental scenarios

### 4.2 Detailed Implementation

**Preserved Functionality from Version 3:**
- All CV-based scenario generation methods (normal, uniform, gamma, lognormal, merve_normal)
- PCA and ARMA covariate generation frameworks
- Comprehensive four-method prediction evaluation (linear, k-NN, clustering, decision tree)
- Dual MAPE formulation for robust error assessment
- Silhouette-based automatic cluster selection

**Major Architectural Extensions:**

#### Extended Data Structure

```python
@dataclass
class Data:
    # All previous fields from Version 3, plus:
    
    # Capacity scenario support
    capacity_scenarios: Optional[np.ndarray] = None  # Shape: (S, J) for scenario-varying capacities
    
    # Comprehensive prediction metrics (combined demand+capacity analysis)
    pca_reg_mape1_mean: float = 0.0          # Linear regression, MAPE₁
    pca_reg_mape_eps_mean: float = 0.0       # Linear regression, MAPEₑ
    pca_knn_mape1_mean: float = 0.0          # k-NN, MAPE₁  
    pca_knn_mape_eps_mean: float = 0.0       # k-NN, MAPEₑ
    pca_cluster_mape1_mean: float = 0.0      # Clustering, MAPE₁
    pca_cluster_mape_eps_mean: float = 0.0   # Clustering, MAPEₑ
    pca_tree_mape1_mean: float = 0.0         # Decision tree, MAPE₁
    pca_tree_mape_eps_mean: float = 0.0      # Decision tree, MAPEₑ
    
    # Demand-only prediction metrics (isolation analysis)
    pca_demand_reg_mape1_mean: float = 0.0   # Demand-only linear regression
    pca_demand_reg_mape_eps_mean: float = 0.0
    pca_demand_knn_mape1_mean: float = 0.0   # Demand-only k-NN
    pca_demand_knn_mape_eps_mean: float = 0.0
    pca_demand_cluster_mape1_mean: float = 0.0  # Demand-only clustering
    pca_demand_cluster_mape_eps_mean: float = 0.0
    pca_demand_tree_mape1_mean: float = 0.0  # Demand-only decision tree
    pca_demand_tree_mape_eps_mean: float = 0.0
```

#### Stochastic Capacity Generation

**New Method: generate_scenarios_merve_with_covariates_and_capacities()**

**Theoretical Framework:**
Extends Merve_Normal approach to generate correlated demand-capacity scenarios:

**Demand Generation (Unchanged):**
$$d_{i,s} \sim \max(0, \mathcal{N}(\mu_i^{demand}, \sigma_i^{demand}))$$

**Stochastic Capacity Generation Algorithm:**

**Step 1: Per-Facility Standard Deviation Generation**
For each facility $j \in J$:
$$\sigma_j^{cap} \sim \mathcal{U}(\sigma_{lower}^{cap} \cdot c_j^{original}, \sigma_{upper}^{cap} \cdot c_j^{original})$$

**Step 2: Scenario-Specific Capacity Generation**
**Scenario 1 (Deterministic Baseline):**
$$c_{j,1} = c_j^{original} \quad \forall j \in J$$

**Scenarios 2 to S (Stochastic):**
For $s = 2, 3, \ldots, S$ and $j \in J$:
$$\tilde{c}_{j,s} \sim \mathcal{N}(c_j^{original}, (\sigma_j^{cap})^2)$$
$$c_{j,s} = \max\{\tilde{c}_{j,s}, c_{j,min}\}$$

Where:
- $c_j^{original}$: Deterministic capacity from input file
- $c_{j,min} = 0.5 \cdot c_j^{original}$: Minimum feasible capacity (50% of original)
- $\sigma_{lower}^{cap}, \sigma_{upper}^{cap}$: Capacity variability bounds (typically 0.05, 0.15)

**Step 3: Per-Scenario Feasibility Enforcement**
For each scenario $s$, ensure total capacity exceeds total demand:
$$\sum_{j \in J} c_{j,s} \geq \frac{\sum_{i \in I} d_{i,s}}{0.95}$$

If violated, apply capacity scaling:
$$\text{required\_capacity} = \frac{\sum_{i} d_{i,s}}{0.95}$$
$$\text{scaling\_factor} = \frac{\text{required\_capacity}}{\sum_{j} c_{j,s}}$$
$$c_{j,s} \leftarrow c_{j,s} \cdot \text{scaling\_factor} \quad \forall j \in J$$

**Mathematical Properties:**
- **Heteroscedasticity**: Each facility has different capacity variability
- **Correlation Structure**: Capacity variations independent across facilities
- **Feasibility Guarantee**: Each scenario individually feasible
- **Mean Preservation**: $E[c_{j,s}] = c_j^{original}$ for $s > 1$

**Feasibility Enforcement Algorithm:**
```python
def enforce_capacity_feasibility(demand_scenarios, capacity_scenarios, safety_factor=0.95):
    """
    Ensure each scenario remains feasible: sum(demands) ≤ sum(capacities)
    """
    S = len(demand_scenarios)
    
    for s in range(S):
        total_demand_s = np.sum(demand_scenarios[s])
        total_capacity_s = np.sum(capacity_scenarios[s])
        
        if total_demand_s > total_capacity_s * safety_factor:
            # Scale capacities up to ensure feasibility
            required_capacity = total_demand_s / safety_factor
            scaling_factor = required_capacity / total_capacity_s
            capacity_scenarios[s] = capacity_scenarios[s] * scaling_factor
            
    return capacity_scenarios
```

**Implementation Details:**
```python
def generate_scenarios_merve_with_covariates_and_capacities(
    demands, capacities, num_scenarios=10, sigma_lower=0.1, sigma_upper=0.3,
    capacity_sigma_lower=0.05, capacity_sigma_upper=0.15,  # Capacity-specific parameters
    num_covariates=3, seed=42
):
    rng = np.random.default_rng(seed)
    I, J = len(demands), len(capacities)
    
    # Step A: Generate demand scenarios (unchanged from previous versions)
    demand_scenarios, demand_info = generate_scenarios_merve_normal(
        demands, capacities, num_scenarios, sigma_lower, sigma_upper, seed
    )
    
    # Step B: Generate capacity scenarios (NEW FUNCTIONALITY)
    capacity_sigma = rng.uniform(
        low=capacity_sigma_lower * capacities,
        high=capacity_sigma_upper * capacities
    )
    
    capacity_scenarios = np.zeros((num_scenarios, J))
    for s in range(num_scenarios):
        # Generate stochastic capacities with minimum bounds
        capacity_scenarios[s] = np.maximum(
            rng.normal(capacities, capacity_sigma),
            0.5 * capacities  # Minimum 50% of original capacity
        )
    
    # Step C: Enforce global feasibility
    capacity_scenarios = enforce_capacity_feasibility(
        demand_scenarios, capacity_scenarios, safety_factor=0.95
    )
    
    # Step D: Combined PCA analysis
    # Concatenate demand and capacity data for unified covariate generation
    combined_data = np.concatenate([demand_scenarios, capacity_scenarios], axis=1)  # Shape: (S, I+J)
    
    # PCA on combined data
    combined_mean = np.mean(combined_data, axis=0)
    combined_centered = combined_data - combined_mean[np.newaxis, :]
    U_combined, _, _ = np.linalg.svd(combined_centered, full_matrices=False)
    
    m_eff = min(num_covariates, np.linalg.matrix_rank(combined_centered), num_scenarios)
    Z_combined = U_combined[:, :m_eff]
    
    # Step E: Comprehensive evaluation
    # 1. Combined analysis (demand + capacity features → demand prediction)
    combined_metrics = []
    for i in range(I):
        metrics = _evaluate_prediction_methods(
            Z_combined, demand_scenarios[:, i], test_size=0.2, epsilon=0.1
        )
        combined_metrics.append(metrics)
    
    # 2. Demand-only analysis (demand features only → demand prediction)  
    demand_only_mean = np.mean(demand_scenarios, axis=0)
    demand_centered = demand_scenarios - demand_only_mean[np.newaxis, :]
    U_demand, _, _ = np.linalg.svd(demand_centered, full_matrices=False)
    Z_demand = U_demand[:, :m_eff]
    
    demand_only_metrics = []
    for i in range(I):
        metrics = _evaluate_prediction_methods(
            Z_demand, demand_scenarios[:, i], test_size=0.2, epsilon=0.1
        )
        demand_only_metrics.append(metrics)
    
    # Step F: Comprehensive statistics aggregation
    combined_array = np.array(combined_metrics)
    demand_only_array = np.array(demand_only_metrics)
    
    info = demand_info.copy()
    info.update({
        "capacity_scenarios_generated": True,
        "num_covariates": m_eff,
        "capacity_sigma_lower": capacity_sigma_lower,
        "capacity_sigma_upper": capacity_sigma_upper,
        
        # Combined prediction metrics
        "pca_reg_mape1_mean": float(np.mean(combined_array[:, 2])),
        "pca_reg_mape_eps_mean": float(np.mean(combined_array[:, 3])),
        "pca_knn_mape1_mean": float(np.mean(combined_array[:, 4])),
        "pca_knn_mape_eps_mean": float(np.mean(combined_array[:, 5])),
        "pca_cluster_mape1_mean": float(np.mean(combined_array[:, 6])),
        "pca_cluster_mape_eps_mean": float(np.mean(combined_array[:, 7])),
        "pca_tree_mape1_mean": float(np.mean(combined_array[:, 8])),
        "pca_tree_mape_eps_mean": float(np.mean(combined_array[:, 9])),
        
        # Demand-only prediction metrics  
        "pca_demand_reg_mape1_mean": float(np.mean(demand_only_array[:, 2])),
        "pca_demand_reg_mape_eps_mean": float(np.mean(demand_only_array[:, 3])),
        "pca_demand_knn_mape1_mean": float(np.mean(demand_only_array[:, 4])),
        "pca_demand_knn_mape_eps_mean": float(np.mean(demand_only_array[:, 5])),
        "pca_demand_cluster_mape1_mean": float(np.mean(demand_only_array[:, 6])),
        "pca_demand_cluster_mape_eps_mean": float(np.mean(demand_only_array[:, 7])),
        "pca_demand_tree_mape1_mean": float(np.mean(demand_only_array[:, 8])),
        "pca_demand_tree_mape_eps_mean": float(np.mean(demand_only_array[:, 9])),
    })
    
    return demand_scenarios, Z_combined, capacity_scenarios, info
```

#### Advanced Research Metrics

**Comparative Analysis Framework:**
Version 4 enables sophisticated analysis of how different uncertainty sources affect ML prediction quality:

**Research Questions Enabled:**
1. **Isolation Effect**: How much does capacity uncertainty add beyond demand uncertainty?
2. **Combined Benefits**: Do combined features improve prediction over demand-only features?
3. **Method Sensitivity**: Which ML methods are most robust to different uncertainty sources?
4. **Scale Effects**: How do different CV levels affect prediction accuracy across methods?

**Metric Interpretation:**
- **Lower Combined MAPE vs. Demand-Only MAPE**: Indicates capacity information improves demand prediction
- **Method Ranking Consistency**: Stable ranking across demand-only and combined suggests robust method
- **MAPE₁ vs. MAPEₑ Patterns**: Reveals method sensitivity to small vs. large prediction errors

#### Enhanced Output Generation

**Enriched Text File Format:**
```
# Stochastic SCFLP Instance with Capacity Variation
# Generated from: {original_file}
# Method: merve_with_covariates_and_capacities
# Demand parameters: sigma_lower={σ_l}, sigma_upper={σ_u}
# Capacity parameters: capacity_sigma_lower={σ_cap_l}, capacity_sigma_upper={σ_cap_u}

{J} {I} {S}

# Original facility capacities
{original_capacities}

# Capacity scenarios (J facilities × S scenarios)
# Scenario 1 capacities
{capacity_scenario_1}
# Scenario 2 capacities
{capacity_scenario_2}
...

# Original demands
{original_demands}

# Demand scenarios
# Scenario 1 demands  
{demand_scenario_1}
# Scenario 2 demands
{demand_scenario_2}
...

# Combined PCA covariates (S scenarios × m features)
{covariate_matrix}

# Combined prediction metrics:
# Linear Regression - MAPE₁: {combined_reg_mape1}, MAPEₑ: {combined_reg_mape_eps}
# k-NN (k=2) - MAPE₁: {combined_knn_mape1}, MAPEₑ: {combined_knn_mape_eps}
# Clustering - MAPE₁: {combined_cluster_mape1}, MAPEₑ: {combined_cluster_mape_eps}
# Decision Tree - MAPE₁: {combined_tree_mape1}, MAPEₑ: {combined_tree_mape_eps}

# Demand-only prediction metrics:
# Linear Regression - MAPE₁: {demand_reg_mape1}, MAPEₑ: {demand_reg_mape_eps}
# k-NN (k=2) - MAPE₁: {demand_knn_mape1}, MAPEₑ: {demand_knn_mape_eps}
# Clustering - MAPE₁: {demand_cluster_mape1}, MAPEₑ: {demand_cluster_mape_eps}
# Decision Tree - MAPE₁: {demand_tree_mape1}, MAPEₑ: {demand_tree_mape_eps}
```

**Excel Integration:**
Comprehensive tracking spreadsheets with comparative analysis columns enabling systematic experimental studies across different parameter combinations and method configurations.

---

## Section 5: Evolution Summary and Integration

### 5.1 Version Relationships and Progression

**Architectural Evolution Timeline:**

**Version 1 → Version 2: Paradigm Shift**
- **Conceptual**: Variance-based → CV-based parameterization
- **Mathematical**: Simple distributional → principled statistical modeling  
- **Capability**: Demand-only → multi-distribution with covariate generation
- **Output**: Basic instances → comprehensive metadata and analysis

**Version 2 → Version 3: Evaluation Enhancement**
- **Assessment**: Linear regression only → four-method ML evaluation
- **Metrics**: Single MAPE → dual MAPE formulation for robustness
- **Intelligence**: Fixed clustering → automatic cluster selection via silhouette analysis
- **Research**: Basic validation → comprehensive prediction quality framework

**Version 3 → Version 4: Dual-Source Sophistication**
- **Scope**: Demand variation only → demand + capacity variation
- **Analysis**: Single-source evaluation → comparative dual-source assessment
- **Complexity**: Individual uncertainty → joint uncertainty modeling
- **Framework**: Method evaluation → comprehensive experimental design platform

**Cumulative Capability Matrix:**

| Feature | V1 | V2 | V3 | V4 |
|---------|----|----|----|----|
| Basic scenario generation | ✓ | ✓ | ✓ | ✓ |
| CV parameterization | ✗ | ✓ | ✓ | ✓ |
| Multiple distributions | ✗ | ✓ | ✓ | ✓ |
| PCA covariates | ✗ | ✓ | ✓ | ✓ |
| ARMA covariates | ✗ | ✓ | ✓ | ✓ |
| Multi-method evaluation | ✗ | ✗ | ✓ | ✓ |
| Dual MAPE metrics | ✗ | ✗ | ✓ | ✓ |
| Capacity variation | ✗ | ✗ | ✗ | ✓ |
| Comparative analysis | ✗ | ✗ | ✗ | ✓ |

### 5.2 Complete Method Comparison

**Data Generation Methods Summary:**

#### Basic Distribution Methods (All Versions)
1. **Normal**: $d_{i,s} \sim \max(0, \mathcal{N}(\mu_i, \sigma_i^2))$
2. **Uniform**: $d_{i,s} \sim \max(0, \mathcal{U}(\mu_i - \delta_i, \mu_i + \delta_i))$

#### Advanced Distribution Methods (V2+)
3. **Gamma**: $d_{i,s} \sim \text{Gamma}(k_i, \theta_i)$ with CV control
4. **Lognormal**: $d_{i,s} \sim \text{Lognormal}(\mu_i^{ln}, \sigma_i^{ln})$ with CV control
5. **Merve_Normal**: Heteroscedastic normal with per-customer variance

#### Covariate Generation Methods (V2+)
6. **ARMA Covariates**: Multivariate ARMA(2,2) time-series-based features
7. **PCA Covariates (Merve_With_Covariates)**: Principal component analysis on scenario-customer matrix

#### Advanced Uncertainty Methods (V4)
8. **Merve_With_Covariates_And_Capacities**: Joint demand-capacity variation with dual analysis

**Method Selection Guidance:**

| Method | Use Case | Strengths | Considerations |
|--------|----------|-----------|----------------|
| Normal/Uniform (V1) | Basic testing | Simple, fast | Limited realism |
| CV-driven (V2+) | Systematic studies | Scale-invariant | More complex |
| Gamma/Lognormal (V2+) | Non-negative guarantee | No clipping distortion | Parameter complexity |
| Merve_Normal (V2+) | Heteroscedastic demands | Customer-specific variance | Additional parameters |
| ARMA Covariates (V2+) | Time-series-like features | Temporal structure | Stationarity requirements |
| PCA Covariates (V2+) | Data-driven features | Natural dimensionality | Scenario-dependent |
| Dual-source (V4) | Comprehensive uncertainty | Realistic modeling | High complexity |

### 5.3 Output Formats and Directory Structure

**Directory Naming Conventions:**

**Version 1:**
```
scenarios_s{num_scenarios}_v{variance}_m{method}/
├── p1_stochastic.txt
├── p2_stochastic.txt
└── ...
```

**Version 2+:**
```
scenarios_s{num_scenarios}_cv{target_cv}_distrib{method}/
├── p1_stochastic.txt
├── p2_stochastic.txt
├── merve_covariates_tracking_{timestamp}.xlsx  (if applicable)
└── ...
```

**Version 4 (Capacity Variation):**
```
scenarios_s{num_scenarios}_cv{target_cv}_distrib{method}_capacities/
├── p1_stochastic_with_capacities.txt
├── p2_stochastic_with_capacities.txt
├── comparative_analysis_{timestamp}.xlsx
└── ...
```

**File Content Structure Evolution:**

**Basic Structure (All Versions):**
- Problem dimensions (J, I, S)
- Original problem data (capacities, fixed costs, demands, shipment costs)
- Generated demand scenarios
- Feasibility metadata

**Enhanced Structure (V2+):**
- Generation method and parameters
- Empirical CV statistics
- R_overall dispersion metric
- Method-specific parameters (sigma bounds, etc.)

**Covariate-Enhanced (V2+):**
- Covariate matrices (PCA or ARMA)
- Prediction quality metrics
- Per-customer evaluation results

**Comprehensive Structure (V4):**
- Capacity scenario matrices
- Dual-source covariate analysis
- Comparative prediction metrics
- Combined vs. demand-only evaluation

**Excel Tracking Files:**

**Version 2+:**
- Instance-level statistics
- Method parameter combinations
- Aggregated quality metrics
- Covariate generation diagnostics

**Version 4:**
- Comparative analysis tables
- Method ranking across uncertainty sources
- Parameter sensitivity analysis
- Experimental design templates

### 5.4 Integration Guidance for Research Applications

**Version Selection Strategy:**

**For Basic Algorithm Testing:**
- **Use**: Version 1 with normal/uniform distributions
- **Rationale**: Simple, fast generation for proof-of-concept studies
- **Configuration**: Small scenario counts (S=50-100), moderate variance

**For Systematic ML Studies:**
- **Use**: Version 2 with CV parameterization
- **Rationale**: Scale-invariant comparison, comprehensive metadata
- **Configuration**: CV ∈ {0.1, 0.3, 0.5, 1.0}, multiple distributions

**For Covariate Analysis:**
- **Use**: Version 3 with multi-method evaluation
- **Rationale**: Comprehensive assessment of feature quality
- **Configuration**: PCA/ARMA covariates, four-method evaluation framework

**For Advanced Uncertainty Modeling:**
- **Use**: Version 4 with capacity variation
- **Rationale**: Most realistic problem modeling, comprehensive analysis
- **Configuration**: Joint demand-capacity uncertainty, comparative metrics

**Integration with ML-Augmented Benders:**

1. **Feature Engineering**: Use PCA/ARMA covariates as input to `feature_engineering/feature_generation.py`
2. **Problem Loading**: Generated instances compatible with `Problem/` module SCFLP parsing
3. **Experimental Design**: Excel tracking enables systematic parameter studies
4. **Quality Assessment**: Prediction metrics guide covariate selection for ML methods

**Computational Considerations:**
- **Version 1**: O(S×I) generation time
- **Version 2-3**: O(S×I) + O(S×m²) for covariate analysis
- **Version 4**: O(S×(I+J)) + O(S×m²) + dual evaluation overhead

**Detailed Function Signatures by Version:**

**Version 1 Functions:**
```python
# Core generation function
def generate_scenarios(demands, capacities, num_scenarios=10, 
                      variance_factor=0.2, method="normal", seed=42):
    """Generate basic stochastic scenarios with variance control"""

# Batch processing function  
def process_files(input_dir, file_list, num_scenarios=10, variance=0.2, 
                 method="normal", output_format="txt", seed=42):
    """Process multiple CFLP files with batch generation"""

# Data parsing function
def read_dataset(file_path, num_scenarios=10, variance_factor=0.2, 
                method="normal", seed=42):
    """Parse CFLP file and generate stochastic instance"""
```

**Version 2+ Functions:**
```python
# CV-driven generation
def generate_scenarios_cv(demands, capacities, num_scenarios=10, 
                         target_cv=0.2, method="normal", seed=42):
    """Generate scenarios with coefficient of variation control"""

# Merve Normal method
def generate_scenarios_merve_normal(demands, capacities, num_scenarios=10,
                                   sigma_lower=0.1, sigma_upper=0.3, seed=42):
    """Generate heteroscedastic Normal scenarios"""

# ARMA covariate generation
def generate_scenarios_arma_covariate(demands, capacities, num_scenarios=10,
                                     num_covariates=4, seed=42):
    """Generate ARMA(2,2)-based features and scenarios"""

# PCA covariate generation (Version 3+)
def generate_scenarios_merve_with_covariates(demands, capacities, num_scenarios=10,
                                           sigma_lower=0.1, sigma_upper=0.3,
                                           num_covariates=3, seed=42):
    """Merve + PCA covariates + comprehensive ML evaluation"""

# Dual-source generation (Version 4)
def generate_scenarios_merve_with_covariates_and_capacities(
    demands, capacities, num_scenarios=10, sigma_lower=0.1, sigma_upper=0.3,
    capacity_sigma_lower=0.05, capacity_sigma_upper=0.15, 
    num_covariates=6, seed=42):
    """Joint demand-capacity variation with comprehensive analysis"""
```

**Usage Examples:**

**Basic Usage (Version 1):**
```python
from SCFLP_data_generation import process_files

# Generate basic stochastic instances
process_files(
    input_dir="/path/to/SCFLP/data",
    file_list=["p1", "p2", "p3"],
    num_scenarios=100,
    variance=0.3,
    method="normal",
    output_format="txt",
    seed=42
)
```

**Advanced Usage (Version 4):**
```python
from SCFLP_data_generation_version4 import process_files

# Generate comprehensive dual-source instances
process_files(
    input_dir="/path/to/SCFLP/data",
    file_list=["p1", "p2", "p3"],
    num_scenarios=500,
    target_cv=0.2,
    method="merve_with_covariate_with_random_demand_capacity",
    output_format="txt",
    seed=42,
    sigma_lower=0.1,
    sigma_upper=0.3,
    num_covariates=6
)
```

**Research Workflow Integration:**
1. **Method Selection**: Choose version based on research complexity requirements
2. **Parameter Configuration**: Set generation parameters for systematic studies
3. **Batch Generation**: Process entire problem sets with consistent parameters
4. **Quality Validation**: Review prediction metrics and feasibility diagnostics
5. **ML Algorithm Input**: Load generated instances into ML-augmented Benders pipeline
6. **Experimental Analysis**: Use Excel tracking for systematic comparison studies
7. **Results Interpretation**: Leverage comparative metrics for method optimization

**Integration with ML-Augmented Benders:**
```python
# Example integration workflow
from Problem.scflp import load_stochastic_instance
from feature_engineering.feature_generation import compute_scenario_features_from_duals

# Load generated instance
data = load_stochastic_instance("/path/to/p1_stochastic.txt")

# Extract covariates for ML features (if available)
if hasattr(data, 'covariate_vectors'):
    scenario_features = data.covariate_vectors  # Use generated PCA/ARMA features
else:
    scenario_features = compute_scenario_features_from_duals(...)  # Generate from duals

# Proceed with ML-augmented Benders decomposition
```

This comprehensive evolution provides a complete toolkit for SCFLP instance generation, from basic testing to sophisticated research applications, with detailed mathematical foundations, comprehensive evaluation metrics, and seamless integration with the ML-augmented Benders decomposition research framework.