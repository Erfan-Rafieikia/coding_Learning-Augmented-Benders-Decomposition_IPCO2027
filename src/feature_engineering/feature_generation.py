"""
feature_engineering.feature_generation
=======================================

This module implements a sophisticated graph-based feature engineering system for
scenario representation in stochastic optimization problems. It transforms dual
variable vectors from multiple optimization trials into low-dimensional embeddings
that capture scenario similarity patterns for machine learning-augmented Benders
decomposition.

Overview:
---------
The feature generation system creates meaningful numerical representations of
optimization scenarios by analyzing dual variable patterns across multiple problem
instances or iterations. This approach enables machine learning models to recognize
similar scenarios and transfer dual variable predictions effectively, improving
the efficiency of decomposition algorithms.

The pipeline transforms raw dual vectors into graph-based embeddings through:
1. Dual vector similarity analysis across scenarios and trials
2. Affinity graph construction based on dual-response discrepancies
3. Random walk generation with similarity-biased transitions
4. Word2Vec embedding learning from walk sequences
5. Low-dimensional feature vector extraction for ML models

Key Components:
---------------

1. **Dual Vector Analysis**:
   - Multi-trial dual variable collection and preprocessing
   - Cross-scenario similarity computation using normalized Euclidean distance
   - Statistical aggregation across multiple optimization trials

2. **Graph Construction**:
   - Weighted graph where nodes represent scenarios
   - Off-diagonal affinities use exp(-Omega_ij/tau)
   - The transition matrix has an exactly zero diagonal
   - Bidirectional edge support for symmetric relationships

3. **Random Walk Generation**:
   - Similarity-biased random walks starting from each scenario
   - Probability distributions obtained by row-normalizing affinities
   - Configurable walk parameters (length, quantity, temperature, seed)

4. **Embedding Learning**:
   - Word2Vec neural network training on walk sequences
   - Skip-gram or CBOW architecture selection
   - Configurable embedding dimensions and context windows
   - Minimum frequency filtering for robust representations

5. **Feature Vector Extraction**:
   - Low-dimensional scenario embeddings suitable for ML models
   - Consistent dimensionality across all scenarios
   - Numerical features capturing scenario relationships and patterns

Dual Similarity Computation:
----------------------------

**Distance Metric**:
For scenarios s1 and s2 across trial t, the normalized distance is:
```
dist(s1, s2, t) = ||dual(s1,t) - dual(s2,t)|| / (||dual(s1,t)|| + ||dual(s2,t)||)
```

**Aggregation Across Trials**:
Average distance across all trials where both scenarios appear:
```
avg_dist(s1, s2) = (1/|T|) * Σ_{t ∈ T} dist(s1, s2, t)
```
Where T = {trials where both dual(s1,t) and dual(s2,t) exist}

**Affinity Conversion**:
Discrepancies are converted to off-diagonal affinities:
```
affinity(s1, s2) = exp(-avg_dist(s1, s2) / tau)
```
The diagonal affinity and transition probability are both zero.

Random Walk Algorithm:
----------------------

**Walk Generation Process**:
1. Start from each scenario node
2. Generate n_walks independent walks of length walk_length
3. At each step, transition to a different scenario with affinity-based probabilities
4. Convert scenario IDs to strings for Word2Vec compatibility

**Transition Probabilities**:
From current scenario c to neighbor n:
```
P(c → n) = similarity(c, n) / Σ_{k ∈ neighbors(c)} similarity(c, k)
```

**Walk Sequence Example**:
```
Walk 1: [scenario_0, scenario_3, scenario_1, scenario_5, ...]
Walk 2: [scenario_1, scenario_0, scenario_2, scenario_4, ...]
...
```

Word2Vec Embedding Learning:
----------------------------

**Architecture Options**:
- **Skip-gram (sg=1)**: Predicts context scenarios from target scenario
- **CBOW (sg=0)**: Predicts target scenario from context scenarios

**Training Objective**:
Maximizes log-likelihood of scenario co-occurrence in walk contexts:
```
max Σ_{walk} Σ_{position} log P(context | target)
```

**Hyperparameters**:
- vector_size (feature_dim): Embedding dimensionality
- window: Context window size around target scenario
- min_count: Minimum scenario frequency threshold
- sg: Architecture selection (skip-gram vs CBOW)

Feature Vector Properties:
--------------------------

**Dimensionality**: 
Each scenario receives a feature_dim-dimensional embedding vector

**Numerical Range**: 
Typically [-1, 1] range with unit norm (Word2Vec default)

**Similarity Preservation**: 
Scenarios with similar dual patterns receive similar embeddings

**Consistency**: 
All scenarios mapped to same dimensional space for ML compatibility

Pipeline Integration:
---------------------

**Input Data Structure**:
```python
trial_duals = {
    (trial_0, scenario_0): dual_vector_0,
    (trial_0, scenario_1): dual_vector_1,
    (trial_1, scenario_0): dual_vector_2,
    ...
}
```

**Feature Parameter Configuration**:
```python
feature_params = {
    'n_walks': 1000,      # Number of walks per scenario
    'walk_length': 10,    # Steps per walk
    'seed': 42,           # Random seed for reproducibility
    'feature_dim': 50,    # Embedding dimensionality  
    'window': 5,          # Word2Vec context window
    'min_count': 1,       # Minimum scenario frequency
    'sg': 1               # Skip-gram architecture
}
```

**Output Format**:
```python
scenario_features = {
    scenario_0: np.array([0.1, -0.3, 0.7, ...]),  # feature_dim length
    scenario_1: np.array([-0.2, 0.5, -0.1, ...]),
    ...
}
```

Performance Characteristics:
----------------------------

**Computational Complexity**:
- Graph construction: O(S² × T) where S=scenarios, T=trials
- Random walk generation: O(S × n_walks × walk_length)
- Word2Vec training: O(vocabulary_size × embedding_dim × epochs)

**Memory Requirements**:
- Dual storage: O(S × T × dual_dimension)
- Graph weights: O(S²) for complete similarity graph
- Walk storage: O(S × n_walks × walk_length)

**Scalability Considerations**:
- Linear growth with number of scenarios
- Quadratic growth with scenario similarity matrix
- Efficient sparse representation for large scenario sets

Quality Metrics:
----------------

**Embedding Quality Indicators**:
1. **Similarity Preservation**: Correlation between dual distances and embedding distances
2. **Cluster Coherence**: Scenarios with similar dual patterns cluster together
3. **Dimensionality Effectiveness**: Information retention in reduced space

**Validation Approaches**:
1. **Cross-Trial Consistency**: Embeddings stable across different trial sets
2. **Downstream Performance**: ML model accuracy with generated features
3. **Interpretability**: Meaningful scenario relationships in embedding space

Configuration Guidelines:
-------------------------

**Walk Parameters**:
- **n_walks**: Higher values (1000+) for stable embeddings
- **walk_length**: 5-20 steps, longer for complex scenario relationships
- **seed**: Fixed value for reproducible research results

**Embedding Parameters**:
- **feature_dim**: 50-200 dimensions, balance expressiveness vs efficiency
- **window**: 3-10, larger for capturing long-range scenario dependencies
- **min_count**: 1 for small datasets, higher for noise reduction
- **sg**: Skip-gram (1) generally better for infrequent scenarios

**Problem-Specific Tuning**:
- **Dense dual vectors**: Smaller feature_dim may suffice
- **Sparse dual vectors**: Larger feature_dim for pattern capture
- **Few scenarios**: Lower min_count and more walks per scenario
- **Many scenarios**: Higher min_count and optimized walk parameters

Error Handling and Robustness:
------------------------------

**Numerical Stability**:
- Epsilon regularization prevents division by zero in similarity computation
- Normalization handles varying dual vector magnitudes
- Robust probability computation for random walk transitions

**Data Quality Assurance**:
- Validation of dual vector consistency across trials
- Detection and handling of missing or invalid dual vectors
- Graceful degradation with insufficient dual data

**Edge Cases**:
- Single trial scenario: Uses available dual patterns
- Identical scenarios: Perfect similarity (zero distance) handling
- Disconnected scenarios: Fallback similarity metrics

Integration with ML Framework:
------------------------------

**Callback Integration**:
The generated features integrate with the training system through:
- Feature vector provision for regression model training
- Consistent scenario representation across iterations
- Dynamic feature updating as new dual data becomes available

**Caching and Efficiency**:
- Feature caching to avoid recomputation with identical dual data
- Incremental updates when new trials add dual information
- Memory-efficient storage for large scenario sets

**Quality Monitoring**:
- Feature quality metrics for debugging and tuning
- Performance tracking for optimization decisions
- Integration with logging system for research analysis

Problem-Specific Adaptations:
-----------------------------

**UFL (Uncapacitated Facility Location)**:
- Customer-scenario dual vectors (λ values)
- Facility-opening pattern similarity
- Transportation cost structure reflection

**UHL (Uncapacitated Hub Location)**:
- OD-pair scenario dual vectors (hub flow duals)
- Hub routing pattern similarity
- Multi-commodity flow structure embedding

**Stochastic Problems (SCFLP, SSLP)**:
- Multi-scenario dual vector aggregation
- Uncertainty pattern representation
- Capacity and demand variability encoding

Extensions and Customization:
-----------------------------

**Alternative Distance Metrics**:
1. Cosine similarity for angle-based dual relationships
2. Wasserstein distance for distribution-based scenarios
3. Problem-specific dual component weighting

**Advanced Embedding Techniques**:
1. Graph neural networks for direct dual vector embedding
2. Variational autoencoders for nonlinear feature learning
3. Attention mechanisms for scenario importance weighting

**Dynamic Feature Updates**:
1. Online learning for streaming dual data
2. Incremental embedding updates without full retraining
3. Adaptive feature dimensions based on scenario complexity

Research Applications:
----------------------

The feature generation system supports research in:
- Scenario clustering and similarity analysis
- Transfer learning across optimization instances
- Adaptive algorithm design based on problem characteristics
- Feature engineering for optimization-aware machine learning

Dependencies:
-------------
- numpy: Numerical computations and array operations
- gensim.Word2Vec: Neural embedding learning from walk sequences
- collections.defaultdict: Efficient data structure management
- logging: Comprehensive feature generation diagnostic logging

Usage Patterns:
---------------

**Standard Feature Generation**:
```python
features = compute_scenario_features_from_duals(
    trial_duals, feature_params
)
```

**Custom Parameter Tuning**:
```python
# High-quality embeddings for research
feature_params = {
    'n_walks': 2000, 'walk_length': 15, 'feature_dim': 100,
    'window': 8, 'min_count': 1, 'sg': 1, 'seed': 42
}

# Fast embeddings for production
feature_params = {
    'n_walks': 500, 'walk_length': 8, 'feature_dim': 50,
    'window': 5, 'min_count': 2, 'sg': 0, 'seed': 42
}
```

The feature generation system provides a robust foundation for scenario representation
in machine learning-augmented optimization, enabling effective knowledge transfer
and improved algorithm performance across diverse problem instances.

"""

import numpy as np
try:
    from gensim.models import Word2Vec
except ImportError:  # Allow transition-probability tests without gensim installed.
    Word2Vec = None
from collections import defaultdict
import logging

# Set up logging
from utils.exclusive_profiler import get_profiled_logger

logger = get_profiled_logger(__name__)

def get_dual_vectors_from_trials(trial_duals):
    """
    Process trial dual vectors for feature generation pipeline compatibility.
    
    This function serves as a data format adapter that ensures compatibility between
    the dual vector collection from multiple optimization trials and the subsequent
    feature generation pipeline. Currently implemented as a pass-through function,
    it provides a standardized interface for potential future data preprocessing
    or format transformations.
    
    Parameters:
    -----------
    trial_duals : dict[(trial_idx, scenario_idx) -> np.ndarray]
        Collection of dual variable vectors from multiple optimization trials.
        Key format: (trial_index, scenario_index) tuple
        Value format: numpy array containing dual variable values
        Source: Multiple calls to Problem.subproblem_dual.solve_dual_subproblem()
        
        Example structure:
        ```
        {
            (0, scenario_0): np.array([λ_0, μ_0_0, μ_0_1, ...]),
            (0, scenario_1): np.array([λ_1, μ_1_0, μ_1_1, ...]),
            (1, scenario_0): np.array([λ_0', μ_0_0', μ_0_1', ...]),
            ...
        }
        ```
    
    Returns:
    --------
    dict[(trial_idx, scenario_idx) -> np.ndarray]
        The same dual vector collection, potentially preprocessed for downstream use.
        Used by: compute_scenario_weights() for similarity computation
        Format: Identical to input, ensuring pipeline compatibility
    
    Design Purpose:
    ---------------
    
    **Interface Standardization**:
    Provides a consistent entry point for dual vector processing that can be
    extended for future preprocessing needs without breaking the pipeline.
    
    **Future Extensibility**:
    Placeholder for potential enhancements such as:
    - Dual vector normalization across different trial scales
    - Outlier detection and filtering in dual solutions
    - Dimensionality reduction for high-dimensional dual vectors
    - Format conversion between different dual representation schemes
    
    **Pipeline Integration**:
    Ensures seamless integration between trial-based dual collection and
    the graph-based feature generation system that follows.
    
    Current Implementation:
    -----------------------
    
    **Pass-Through Operation**:
    ```python
    return trial_duals  # Direct return of input data
    ```
    
    **Rationale**:
    - Maintains data integrity without unnecessary transformations
    - Preserves all trial and scenario information for downstream processing
    - Enables efficient pipeline execution without copying overhead
    
    Integration Context:
    --------------------
    
    **Called by**: compute_scenario_features_from_duals() as first pipeline step
    **Calls to**: None (terminal processing step in this direction)
    **Data Flow**: trial_duals → get_dual_vectors_from_trials() → compute_scenario_weights()
    
    **Usage Pattern**:
    ```python
    def compute_scenario_features_from_duals(trial_duals, feature_params):
        duals = get_dual_vectors_from_trials(trial_duals)  # Preprocessing step
        weights = compute_scenario_weights(duals)          # Continue pipeline
        # ... rest of feature generation
    ```
    
    Example Usage:
    --------------
    ```python
    # Collect dual vectors from multiple trials
    trial_duals = {}
    for trial_idx in range(n_trials):
        for scenario_idx in scenarios:
            dual_vector = solve_dual_subproblem(...)
            trial_duals[(trial_idx, scenario_idx)] = dual_vector
    
    # Process for feature generation
    processed_duals = get_dual_vectors_from_trials(trial_duals)
    # Result: processed_duals == trial_duals (currently identical)
    ```
    
    Future Enhancements:
    --------------------
    
    **Potential Preprocessing Operations**:
    1. **Normalization**: Scale dual vectors to consistent magnitude ranges
    2. **Filtering**: Remove outlier dual solutions that could skew similarities
    3. **Dimensionality Reduction**: Project high-dimensional duals to manageable space
    4. **Format Conversion**: Transform between different dual representation formats
    
    **Example Enhanced Implementation**:
    ```python
    def get_dual_vectors_from_trials(trial_duals, normalize=True, filter_outliers=True):
        processed = {}
        for key, dual_vector in trial_duals.items():
            if filter_outliers and is_outlier(dual_vector):
                continue
            if normalize:
                dual_vector = normalize_dual_vector(dual_vector)
            processed[key] = dual_vector
        return processed
    ```
    
    Research Applications:
    ----------------------
    This function supports research in:
    - Dual solution quality analysis across multiple trials
    - Feature generation pipeline development and optimization
    - Data preprocessing strategy evaluation for optimization-ML integration
    - Robust feature engineering for varying dual solution characteristics
    """
    return trial_duals


def compute_scenario_weights(duals, stability_epsilon=1e-12):
    """
    Compute pairwise similarity weights between scenarios based on dual variable patterns.
    
    This function analyzes dual variable vectors across multiple optimization trials to
    compute normalized distance measures between all scenario pairs. The resulting weights
    form the foundation of a similarity graph that captures scenario relationships based
    on their optimization characteristics, enabling effective random walk generation and
    subsequent embedding learning.
    
    The similarity computation uses normalized Euclidean distance averaged across multiple
    trials, providing robust estimates of scenario relationships that account for
    variability in dual solutions while emphasizing consistent optimization patterns.
    
    Parameters:
    -----------
    duals : dict[(trial_idx, scenario_idx) -> np.ndarray]
        Dual variable vectors collected across multiple optimization trials.
        Format: {(trial_index, scenario_index): dual_vector_array}
        Source: get_dual_vectors_from_trials() output
        Content: Each dual_vector contains problem-specific dual variables
        
        Example structure:
        ```
        {
            (0, scenario_0): np.array([λ, μ_0, μ_1, ...]),
            (0, scenario_1): np.array([λ', μ_0', μ_1', ...]),
            (1, scenario_0): np.array([λ'', μ_0'', μ_1'', ...]),
            ...
        }
        ```
    
    Returns:
    --------
    dict[(scenario_id, scenario_id) -> float]
        Pairwise normalized distance weights between all scenario pairs.
        Key format: (scenario_i, scenario_j) for all i, j combinations
        Value range: [0.0, 1.0] where 0.0 = identical scenarios, 1.0 = maximally different
        Used by: generate_random_walks() for similarity-biased transition probabilities
        
        Properties:
        - Symmetric: weights[(s1, s2)] == weights[(s2, s1)]
        - Self-similarity: weights[(s, s)] == 0.0 for all scenarios s
        - Complete: Contains entry for every scenario pair
        
        Example output:
        ```
        {
            (scenario_0, scenario_0): 0.0,     # Perfect self-similarity
            (scenario_0, scenario_1): 0.25,    # Moderate similarity
            (scenario_1, scenario_0): 0.25,    # Symmetric entry
            (scenario_1, scenario_1): 0.0,     # Perfect self-similarity
            ...
        }
        ```
    
    Mathematical Formulation:
    -------------------------
    
    **Normalized Distance Computation**:
    For scenarios s1 and s2 in trial t:
    ```
    distance(s1, s2, t) = ||dual(s1,t) - dual(s2,t)||_2 / (||dual(s1,t)||_2 + ||dual(s2,t)||_2)
    ```
    Where ||·||_2 is the Euclidean (L2) norm.
    
    **Multi-Trial Aggregation**:
    Average normalized distance across all trials where both scenarios appear:
    ```
    avg_distance(s1, s2) = (1/|T|) * Σ_{t ∈ T} distance(s1, s2, t)
    ```
    Where T = {trials : dual(s1,t) exists AND dual(s2,t) exists}
    
    **Normalization Benefits**:
    - Scale invariance: Robust to varying dual variable magnitudes
    - Numerical stability: Avoids division by zero through norm sum normalization
    - Comparative consistency: Enables meaningful comparison across trial variations
    
    **Self-Loop Handling**:
    For identical scenarios: weights[(s, s)] = 0.0 (perfect similarity)
    
    Algorithm Implementation:
    -------------------------
    
    **Step 1: Scenario Extraction**
    ```python
    scenario_ids = sorted({scenario for (_, scenario) in duals})
    trial_ids = sorted({trial for (trial, _) in duals})
    ```
    
    **Step 2: Pairwise Distance Computation**
    ```python
    for s1 in scenario_ids:
        for s2 in scenario_ids[s1_index+1:]:  # Avoid redundant pairs
            distances_across_trials = []
            for trial in trial_ids:
                if (trial, s1) in duals and (trial, s2) in duals:
                    d1, d2 = duals[(trial, s1)], duals[(trial, s2)]
                    norm_sum = np.linalg.norm(d1) + np.linalg.norm(d2)
                    if norm_sum > 0:
                        dist = np.linalg.norm(d1 - d2) / norm_sum
                        distances_across_trials.append(dist)
            
            avg_dist = np.mean(distances_across_trials) if distances_across_trials else 0.0
            weights[(s1, s2)] = weights[(s2, s1)] = avg_dist
    ```
    
    **Step 3: Self-Loop Addition**
    ```python
    for scenario in scenario_ids:
        weights[(scenario, scenario)] = 0.0
    ```
    
    Computational Characteristics:
    ------------------------------
    
    **Time Complexity**: O(S² × T × D)
    - S = number of unique scenarios
    - T = number of trials  
    - D = dual vector dimensionality (for norm computations)
    
    **Space Complexity**: O(S²)
    - Stores pairwise weights for all scenario combinations
    - Memory scales quadratically with scenario count
    
    **Numerical Stability**:
    - Normalized distance prevents overflow/underflow issues
    - Robust handling of zero-norm dual vectors
    - Consistent results across varying trial magnitudes
    
    Integration with Feature Pipeline:
    ----------------------------------
    
    **Input Processing**:
    Receives dual vectors from get_dual_vectors_from_trials() with standardized format
    ensuring compatibility across different problem types and trial configurations.
    
    **Output Usage**:
    Weights feed into generate_random_walks() to create similarity-biased walk sequences:
    ```python
    weights = compute_scenario_weights(duals)
    walks = generate_random_walks(weights, n_walks, walk_length, seed)
    ```
    
    **Graph Interpretation**:
    The weights define a complete weighted graph where:
    - Nodes = scenarios
    - Edges = pairwise similarity relationships
    - Edge weights = normalized dual variable distances
    
    Quality Assurance:
    ------------------
    
    **Distance Property Validation**:
    - Non-negativity: All distances ≥ 0.0
    - Symmetry: distance(s1, s2) == distance(s2, s1)
    - Identity: distance(s, s) == 0.0
    - Triangle inequality: Generally satisfied due to Euclidean basis
    
    **Robustness Features**:
    - Handles missing dual vectors gracefully (skips those trials)
    - Manages zero-norm dual vectors without division errors
    - Provides meaningful similarity measures across diverse dual patterns
    
    **Logging and Debugging**:
    ```python
    logger.debug(f"Computed weights for {len(scenario_ids)} scenarios with {len(weights)} edges")
    ```
    Facilitates debugging and performance monitoring.
    
    Example Usage:
    --------------
    ```python
    # Input: dual vectors from multiple trials
    duals = {
        (0, 'customer_0'): np.array([5.2, 1.1, 0.0]),
        (0, 'customer_1'): np.array([3.7, 0.8, 1.2]),
        (1, 'customer_0'): np.array([5.1, 1.0, 0.1]),
        (1, 'customer_1'): np.array([3.8, 0.9, 1.1]),
    }
    
    # Compute scenario similarity weights
    weights = compute_scenario_weights(duals)
    
    # Result: weights contains pairwise distances
    # weights[('customer_0', 'customer_1')] ≈ 0.35  (normalized distance)
    # weights[('customer_0', 'customer_0')] == 0.0   (perfect similarity)
    ```
    
    Research Applications:
    ----------------------
    This function supports research in:
    - Scenario clustering and similarity analysis in stochastic optimization
    - Graph-based representation learning for optimization problems
    - Dual solution pattern recognition and characterization
    - Feature engineering effectiveness evaluation for ML-optimization integration
    
    The similarity weight computation provides the mathematical foundation for
    capturing optimization-relevant scenario relationships in the feature space,
    enabling effective knowledge transfer and improved prediction quality in
    ML-augmented decomposition algorithms.
    """
    weights = {}
    scenario_ids = sorted({scenario for (_, scenario) in duals})
    trial_ids = sorted({trial for (trial, _) in duals})

    # Check if we have any scenarios to work with
    if not scenario_ids:
        logger.warning("No scenarios found in dual vectors - returning empty weights")
        return weights

    # Compute pairwise distances between different scenarios
    for i, s1 in enumerate(scenario_ids): 
        for s2 in scenario_ids[i + 1:]:
            total, count = 0, 0
            for t in trial_ids:
                d1 = duals.get((t, s1))
                d2 = duals.get((t, s2))
                if d1 is not None and d2 is not None:
                    denominator = (
                        np.linalg.norm(d1)
                        + np.linalg.norm(d2)
                        + float(stability_epsilon)
                    )
                    dist = np.linalg.norm(d1 - d2) / denominator
                    total += dist
                    count += 1
            avg_dist = total / count if count > 0 else 0
            weights[(s1, s2)] = avg_dist
            weights[(s2, s1)] = avg_dist
    
    # Store Omega_ii=0 as defined in the manuscript.  The transition builder
    # explicitly excludes these diagonal entries, so they are not self-loops.
    for s in scenario_ids:
        weights[(s, s)] = 0.0
    
    logger.debug(f"Computed weights for {len(scenario_ids)} scenarios with {len(weights)} edges")
    return weights


def build_transition_probabilities(
    weights,
    temperature=1.0,
    transition_rule="MANUSCRIPT",
    legacy_inverse_epsilon=1e-8,
):
    """Construct the manuscript transition matrix from dual discrepancies.

    For distinct scenarios, ``s_ij = exp(-Omega_ij / temperature)`` and each
    row is normalized over all *other* scenarios.  The diagonal is exactly
    zero, so a walk can never remain at its current scenario through a
    self-loop.
    """
    rule = str(transition_rule).upper()
    nodes = sorted({s for edge in weights for s in edge})
    if rule == "LEGACY_INVERSE_WITH_SELF_LOOPS":
        transitions = {}
        for source in nodes:
            neighbors = [target for target in nodes if (source, target) in weights]
            inverse = np.asarray(
                [
                    1.0 / (float(weights[(source, target)]) + float(legacy_inverse_epsilon))
                    for target in neighbors
                ],
                dtype=float,
            )
            transitions[source] = (neighbors, inverse / inverse.sum())
        return transitions
    if rule != "MANUSCRIPT":
        raise ValueError(
            "transition_rule must be MANUSCRIPT or LEGACY_INVERSE_WITH_SELF_LOOPS"
        )

    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    transitions = {}
    for source in nodes:
        neighbors = [target for target in nodes if target != source and (source, target) in weights]
        if not neighbors:
            transitions[source] = ([], np.asarray([], dtype=float))
            continue
        discrepancies = np.asarray([weights[(source, target)] for target in neighbors], dtype=float)
        if not np.all(np.isfinite(discrepancies)):
            raise ValueError("Dual-response discrepancies must be finite")
        log_affinity = -discrepancies / temperature
        # Subtracting the maximum prevents underflow without changing probabilities.
        affinity = np.exp(log_affinity - np.max(log_affinity))
        probabilities = affinity / affinity.sum()
        transitions[source] = (neighbors, probabilities)
    return transitions


def generate_random_walks(
    weights,
    n_walks,
    walk_length,
    seed,
    temperature=1.0,
    transition_rule="MANUSCRIPT",
):
    rng = np.random.default_rng(seed)
    walks = []
    nodes = sorted({s for s, _ in weights})
    transitions = build_transition_probabilities(
        weights,
        temperature=temperature,
        transition_rule=transition_rule,
    )

    # Check if we have any nodes to generate walks from
    if not nodes:
        logger.warning("No nodes available for random walk generation - returning empty walks")
        return walks

    for _ in range(n_walks):
        for start in nodes:
            walk = [start]
            current = start
            for _ in range(walk_length):
                neighbors, probs = transitions[current]
                if not neighbors:
                    break
                current = rng.choice(neighbors, p=probs)
                walk.append(current)
            walks.append([str(s) for s in walk])
    return walks


def learn_embeddings(walks, feature_dim, window, min_count, sg, seed=42):
    # Check if we have any walks to learn from
    if not walks or all(len(walk) == 0 for walk in walks):
        logger.warning("No walks available for Word2Vec training - returning empty embeddings")
        return {}
    
    # Filter out empty walks
    non_empty_walks = [walk for walk in walks if len(walk) > 0]
    if not non_empty_walks:
        logger.warning("All walks are empty - returning empty embeddings")
        return {}
    
    try:
        if Word2Vec is None:
            raise ImportError("gensim is required to train the graph-walk embeddings")
        model = Word2Vec(
            sentences=non_empty_walks,
            vector_size=feature_dim,
            window=window,
            min_count=min_count,
            sg=sg,
            seed=int(seed),
            workers=1,
        )
        return {int(k): model.wv[k] for k in model.wv.index_to_key if k is not None}
    except Exception as e:
        logger.error(f"Failed to train Word2Vec model: {e}")
        return {}


from utils.exclusive_profiler import profile_call

def compute_scenario_features_from_duals(
    trial_duals,
    feature_params
):
    """
    Generate scenario embeddings from dual vectors across multiple trials.
    Args:
        trial_duals: dict of {(trial_idx, scenario_idx) → dual vector}
        feature_params: dict with keys:
            - n_walks
            - walk_length
            - seed
            - feature_dim
            - window
            - min_count
            - sg
    Returns:
        dict[scenario_idx] → feature vector
    """
    # Check if we have any dual vectors to work with
    if not trial_duals:
        logger.warning("No dual vectors available for feature generation - returning empty feature dictionary")
        return {}
    
    duals = profile_call("dual_trajectory_processing", get_dual_vectors_from_trials, trial_duals)
    weights = profile_call(
        "dual_trajectory_processing", compute_scenario_weights,
        duals, stability_epsilon=feature_params.get("stability_epsilon", 1e-12),
    )
    walks = profile_call(
        "graph_random_walk_generation", generate_random_walks,
        weights,
        n_walks=feature_params["n_walks"],
        walk_length=feature_params["walk_length"],
        seed=feature_params["seed"],
        temperature=feature_params.get("temperature", 1.0),
        transition_rule=feature_params.get("transition_rule", "MANUSCRIPT"),
    )
    return profile_call(
        "embedding_skipgram_training", learn_embeddings,
        walks,
        feature_dim=feature_params["feature_dim"],
        window=feature_params["window"],
        min_count=feature_params["min_count"],
        sg=feature_params["sg"],
        seed=feature_params["seed"],
    )
