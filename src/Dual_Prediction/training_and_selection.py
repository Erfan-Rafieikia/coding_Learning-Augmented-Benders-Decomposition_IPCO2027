"""
ML Training and Selection Module for SCFLP ML-Augmented Benders Decomposition

This module provides the ONE-TIME ML training and subproblem selection phase (Step 5 of 6-Step Pattern)
specifically for SCFLP (Stochastic Capacitated Facility Location) problems. It implements all ML method 
types with their respective predictor algorithms for optimal subproblem selection and dual prediction 
mapping creation.

NEW ARCHITECTURE (Feb 2026):
----------------------------
This module implements Step 5 (Partitioning & Training) of the 6-step execution pattern. The function
works with the restructured method architecture that separates:
- **Method Categories**: Exact vs ML  
- **ML Method Types**: ML_Augmented_BD, ML_Augmented_BD_with_duals, Covariate_Similarity, 
                        Uncertainty_Similarity, Random_Sampling
- **Predictor Types**: KNN, TREE, CLUSTERING (where applicable)

Each ML method type uses different input data for training:
- **ML_Augmented_BD**: Graph walk feature embeddings (from Step 4)
- **ML_Augmented_BD_with_duals**: Dual variable vectors (from Phase 1)
- **Covariate_Similarity**: Covariate vectors (problem instance parameters)
- **Uncertainty_Similarity**: Uncertainty vectors (scenario realizations)
- **Random_Sampling**: No training needed (random selection at each iteration)

Backward Compatibility:
-----------------------
The function currently accepts legacy prediction_method names (KNN, TREE, DUAL_CLUSTERING, etc.)  
for backward compatibility. The main.py orchestrator translates new method names to legacy names
before calling this module. Legacy method name mapping:
- KNN → ML_Augmented_BD + KNN predictor
- TREE → ML_Augmented_BD + TREE predictor  
- DUAL_CLUSTERING → ML_Augmented_BD + CLUSTERING predictor
- DUAL_CLUSTERING_WITH_DUAL → ML_Augmented_BD_with_duals + CLUSTERING predictor
- UNCERTAINTY_CLUSTERING → Uncertainty_Similarity + CLUSTERING predictor
- COVARIATE_CLUSTERING → Covariate_Similarity + CLUSTERING predictor
- BERTSIMAS_RANDOM_SAMPLING → Random_Sampling (no predictor)

Current Focus: SCFLP + All Prediction Methods
-------------------------------------------
**As of February 2026, this implementation supports SCFLP problems with all 5 ML method types
and 3 predictor algorithms** to provide comprehensive ML-augmented Benders decomposition approaches.

Architecture Role in 6-Step Pattern:
------------------------------------
- **Step 5 - Partitioning & Training**: This module implements this step
- **ONE-TIME OPERATION**: Called once at the start of Phase 2 for method-specific setup
- **Integrated Selection**: Implements optimal subproblem selection with prediction mapping
- **Unified Interface**: Supports all method types through consistent API
- **SCFLP Specialized**: Implementation optimized for SCFLP problem structure and constraints

Key Features:
- **k-NN Predictor**: Implements min_{T⊆S,|T|=p} Σ_{s∈S\\T} (1/k) Σ_{t∈T_k(s)} d_F(f^s, f^t)
- **TREE Predictor**: Greedy top-down tree construction with medoid-based dual reconstruction
- **CLUSTERING Predictor**: K-means/K-medoids clustering with representative selection
- **Joint Optimization**: Simultaneously optimizes scenario selection and prediction mappings
- **SCFLP-Optimized**: Handles SCFLP scenario feature vectors and dual variable structures
- **Fixed Mappings**: Creates stable assignments for consistent iterative prediction
- **Efficient Algorithms**: Greedy + local search for k-NN, CART-style for TREE, K-means for clustering

Method Implementations (SCFLP):
- **k-NN**: k-Nearest Neighbors with integrated optimal selection via paper's objective
- **TREE**: Decision tree-based medoid selection with dual space optimization  
- **CLUSTERING**: K-means/K-medoids clustering with centroid/medoid-based representative selection
- **Features**: SCFLP scenario feature vectors (demand patterns, costs, capacities) and uncertainty vectors
- **Mappings**: Method-specific neighbor/leaf/cluster assignments for unselected scenarios
- **Status**: All methods fully implemented and optimized for SCFLP problems

Integration:
- Called by DualPredictor._train_unified() from prediction.py for one-time method setup (Step 5)
- Returns selected scenarios and prediction mappings for iterative prediction (Step 6 - Phase 2)
- Replaces previous training.py and subproblem_selection.py modules
- Works exclusively with SCFLP problem instances
"""

import numpy as np
from scipy.spatial.distance import cdist
import logging
import heapq
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans

from utils.exclusive_profiler import get_profiled_logger

logger = get_profiled_logger(__name__)


from utils.exclusive_profiler import profiled, profile_section

def integrated_prediction_selection(feature_vectors, ml_method_type, predictor_type, p, method_params=None):
    """
    Unified function for ML prediction methods that jointly handle selection and mapping (Step 5).
    
    This function implements Step 5 (Partitioning & Training) of the 6-step execution pattern.
    Each prediction method determines its own optimal way to select subproblems and create 
    mappings for unselected scenarios using method-specific objectives and input data.
    
    Input Data by Method Type (New Architecture):
    ----------------------------------------------
    Different ML method types use different input data for training:
    
    1. **ML_Augmented_BD** methods:
       - Predictor types: KNN, TREE, CLUSTERING
       - Uses: Graph walk feature embeddings from Step 4
       - Data: feature_vectors parameter
       - Example: KNN finds k-nearest neighbors in feature space
       
    2. **ML_Augmented_BD_with_duals**:
       - Predictor types: KNN, TREE, CLUSTERING
       - Uses: Dual variable vectors collected during Phase 1 (Step 3)
       - Data: method_params['all_trial_duals']
       - Example: K-medoids clustering in dual space
       
    3. **Covariate_Similarity**:
       - Predictor types: KNN, TREE, CLUSTERING
       - Uses: Covariate vectors (problem instance parameters)
       - Data: method_params['covariate_vectors']
       - Example: K-means clustering on input covariates
       
    4. **Uncertainty_Similarity**:
       - Predictor types: KNN, TREE, CLUSTERING
       - Uses: Uncertainty vectors (scenario realizations)
       - Data: method_params['uncertainty_vectors']
       - Example: K-means clustering on demand/uncertainty vectors
       
    5. **Random_Sampling**:
       - Predictor type: None (not applicable)
       - Uses: None (random selection at each iteration)
       - Data: Returns empty mapping
       - Example: Dynamic random selection, no fixed training
    
    Args:
        feature_vectors (dict): Dictionary {subproblem_id: feature_vector}.
                               Used by ML_Augmented_BD methods.
                               May be None for methods that use alternative data sources.
        ml_method_type (str): Method type from new architecture:
                             'ML_Augmented_BD', 'ML_Augmented_BD_with_duals',
                             'Covariate_Similarity', 'Uncertainty_Similarity', 'Random_Sampling'
        predictor_type (str): Predictor type: 'KNN', 'TREE', 'CLUSTERING', or None for Random_Sampling
        p (int): Number of subproblems to select (cardinality of selected set)
        method_params (dict, optional): Method-specific parameters and alternative data sources:
            - 'k': int - Number of neighbors for KNN
            - 'find_optimal_k': bool - Whether to search for optimal k
            - 'all_trial_duals': dict - Dual vectors for ML_Augmented_BD_with_duals
            - 'uncertainty_vectors': dict - Uncertainty vectors for Uncertainty_Similarity
            - 'covariate_vectors': dict - Covariate vectors for Covariate_Similarity
            - Plus method-specific clustering/tree parameters
        
    Returns:
        tuple: (selected_subproblems, method_mapping)
            - selected_subproblems (list): List of selected subproblem IDs (size p)
            - method_mapping (dict): Method-specific mapping for unselected scenarios:
                * KNN: {scenario_id: {'neighbors': [ids], 'distances': [...], 'k': k}}
                * TREE: {scenario_id: {'leaf_id': int, 'medoid_id': int}}
                * CLUSTERING: {scenario_id: {'cluster_id': int, 'representative_id': int}}
                * Random: {} (empty - dynamic selection)
    
    Raises:
        ValueError: If required data is missing for the specified method
        NotImplementedError: If method combination is not supported
    
    Example:
        >>> # ML_Augmented_BD with KNN predictor using feature vectors
        >>> selected, mapping = integrated_prediction_selection(
        ...     feature_vectors={0: [1.0, 2.0], 1: [1.1, 2.1], 2: [3.0, 4.0]},
        ...     ml_method_type='ML_Augmented_BD',
        ...     predictor_type='KNN',
        ...     p=2,
        ...     method_params={'k': 1, 'find_optimal_k': False}
        ... )
        
        >>> # Uncertainty_Similarity with CLUSTERING predictor using uncertainty vectors
        >>> selected, mapping = integrated_prediction_selection(
        ...     feature_vectors=None,  # Not used for this method
        ...     ml_method_type='Uncertainty_Similarity',
        ...     predictor_type='CLUSTERING',
        ...     p=2,
        ...     method_params={'uncertainty_vectors': {0: [10, 20], 1: [11, 21], 2: [30, 40]}}
        ... )
    """
    # Determine which data source to use based on ML method type
    if ml_method_type == 'ML_Augmented_BD':
        # Uses feature_vectors from graph walk embeddings
        if not feature_vectors:
            raise ValueError(f"Feature vector dictionary is empty for method {ml_method_type} with predictor {predictor_type}.")
        input_vectors = feature_vectors
        
    elif ml_method_type == 'ML_Augmented_BD_with_duals':
        # Uses dual vectors collected during Phase 1
        if method_params is None or 'all_trial_duals' not in method_params:
            raise ValueError(f"{ml_method_type} requires 'all_trial_duals' in method_params")
        input_vectors = method_params['all_trial_duals']
        
    elif ml_method_type == 'Uncertainty_Similarity':
        # Uses uncertainty vectors (demand/capacity vectors)
        if method_params is None or 'uncertainty_vectors' not in method_params:
            raise ValueError(f"{ml_method_type} requires 'uncertainty_vectors' in method_params")
        input_vectors = method_params['uncertainty_vectors']
        
    elif ml_method_type == 'Covariate_Similarity':
        # Uses covariate vectors from problem data
        if method_params is None or 'covariate_vectors' not in method_params:
            raise ValueError(f"{ml_method_type} requires 'covariate_vectors' in method_params")
        input_vectors = method_params['covariate_vectors']
        
    elif ml_method_type == 'Random_Sampling':
        # No training needed, dynamic random selection
        return solve_bertsimas_random_sampling_selection(feature_vectors, p, method_params)
        
    else:
        raise NotImplementedError(f"ML method type '{ml_method_type}' is not implemented.")
    
    # Special handling for ML_Augmented_BD_with_duals: Convert dual dictionaries to numeric vectors
    if ml_method_type == 'ML_Augmented_BD_with_duals':
        # input_vectors contains dual dictionaries: {(scenario_id, iteration): {'mu': {...}, 'nu': {...}}}
        # Convert to numeric feature vectors for use with all predictor types
        logger.info(f"ML_Augmented_BD_with_duals: Converting {len(input_vectors)} dual dictionaries to numeric vectors")
        numeric_vectors = _build_stacked_dual_profiles(input_vectors)
        logger.info(f"Converted to {len(numeric_vectors)} numeric feature vectors")
        
        # IMPORTANT: Keep original all_trial_duals in method_params for TREE medoid computation
        # TREE needs both: numeric vectors for tree building, original duals for medoid selection
        if 'all_trial_duals' not in method_params:
            method_params['all_trial_duals'] = input_vectors  # Preserve original dual dictionaries
        
        # Use numeric vectors as input for all predictors
        input_vectors = numeric_vectors
    
    # Now route to appropriate predictor based on predictor_type
    if predictor_type == 'KNN':
        return solve_knn_integrated_selection(input_vectors, p, method_params)
    
    elif predictor_type == 'TREE':
        return solve_tree_integrated_selection(input_vectors, p, method_params)
    
    elif predictor_type == 'CLUSTERING':
        # Use standard clustering with (now numeric) feature vectors
        return solve_dual_clustering_selection(input_vectors, p, method_params)
    
    else:
        raise NotImplementedError(f"Predictor type '{predictor_type}' is not implemented.")


def solve_knn_integrated_selection(feature_vectors, p, knn_params=None):
    """
    Integrated KNN selection + mapping based on paper's k-NN objective.
    
    Solves: min_{{T subseteq S,|T|=p}} Sum_{{s in S\\T}} (1/k) Sum_{{t in T_k(s)}} d_F(f^s, f^t)
    
    Where T_k(s) are the k nearest neighbors of s in T.
    
    Args:
        feature_vectors (dict): Dictionary {scenario_id: feature_vector}
        p (int): Number of scenarios to select (|T|)
        knn_params (dict): Contains 'k' (neighbors) and 'find_optimal_k' (bool)
        
    Returns:
        selected_subproblems (list): Selected scenario IDs (T)
        knn_mapping (dict): For each unselected s, maps to {'neighbors': T_k(s), 
                                                           'distances': distances, 
                                                           'k': k}
    """
    if knn_params is None:
        knn_params = {'k': 3, 'find_optimal_k': False}
    
    scenario_ids = list(feature_vectors.keys())
    num_scenarios = len(scenario_ids)
    
    if p <= 0 or p > num_scenarios:
        raise ValueError(f"Invalid p={p}. Must be between 1 and {num_scenarios}")
    
    # Clamp k to valid range
    k = min(knn_params.get('k', 3), p)
    find_optimal_k = knn_params.get('find_optimal_k', False)
    
    logger.info(f"Starting integrated KNN selection with p={p}, initial k={k}")
    
    # Precompute distance matrix
    feature_matrix = np.array([feature_vectors[sid] for sid in scenario_ids])
    distance_matrix = cdist(feature_matrix, feature_matrix, metric="euclidean")
    
    best_T = None
    best_k = k
    best_objective = float('inf')
    
    # Enumerate k values if requested
    k_values = range(1, min(p, 26) + 1) if find_optimal_k else [k]
    
    for current_k in k_values:
        logger.info(f"Solving P({current_k}) with k={current_k}")
        
        # Solve P(k) for current k
        T, objective_value = _solve_knn_objective(
            scenario_ids, distance_matrix, p, current_k
        )
        
        logger.info(f"P({current_k}) objective: {objective_value:.6f}")
        
        if objective_value < best_objective:
            best_objective = objective_value
            best_T = T
            best_k = current_k
    
    logger.info(f"Best solution: T={len(best_T) if best_T else 0} scenarios, k={best_k}, objective={best_objective:.6f}")
    
    # Build k-NN mapping for the best solution
    knn_mapping = _build_knn_mapping(
        scenario_ids, feature_vectors, best_T, best_k, distance_matrix
    )
    
    # Add optimal k to the mapping for tracking
    knn_mapping['__optimal_k__'] = best_k
    
    return best_T, knn_mapping


def _solve_knn_objective(scenario_ids, distance_matrix, p, k):
    """
    Solve P(k) using greedy heuristic with local refinement.
    
    Returns:
        T (list): Selected scenario IDs
        objective_value (float): Objective value of P(k)
    """
    num_scenarios = len(scenario_ids)
    
    # Initialize with diversity seeding (farthest-first)
    T = _diversity_seeding(scenario_ids, distance_matrix, p)
    T_indices = [scenario_ids.index(sid) for sid in T]
    
    # Compute initial objective
    current_objective = _compute_knn_objective(
        list(range(num_scenarios)), T_indices, distance_matrix, k
    )
    
    # logger.debug(f"Initial objective after seeding: {current_objective:.6f}")
    
    # Greedy improvement
    improved = True
    while improved:
        improved = False
        best_swap = None
        best_improvement = 0
        
        for t_idx in range(len(T_indices)):
            for j in range(num_scenarios):
                if j in T_indices:
                    continue
                    
                # Try swap: remove T_indices[t_idx], add j
                new_T_indices = T_indices.copy()
                new_T_indices[t_idx] = j
                
                new_objective = _compute_knn_objective(
                    list(range(num_scenarios)), new_T_indices, distance_matrix, k
                )
                
                improvement = current_objective - new_objective
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = (t_idx, j)
        
        if best_swap is not None:
            t_idx, j = best_swap
            T_indices[t_idx] = j
            current_objective -= best_improvement
            improved = True
            # logger.debug(f"Improved objective to {current_objective:.6f}")
    
    # Convert back to scenario IDs
    final_T = [scenario_ids[idx] for idx in T_indices]
    
    return final_T, current_objective


def _diversity_seeding(scenario_ids, distance_matrix, p):
    """Farthest-first seeding for initial solution."""
    if p >= len(scenario_ids):
        return scenario_ids.copy()
    
    selected_indices = [0]  # Start with first scenario
    
    for _ in range(p - 1):
        max_min_distance = -1
        best_candidate = -1
        
        for candidate in range(len(scenario_ids)):
            if candidate in selected_indices:
                continue
                
            # Find minimum distance to any selected point
            min_distance = min(distance_matrix[candidate, selected] 
                             for selected in selected_indices)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = candidate
        
        if best_candidate != -1:
            selected_indices.append(best_candidate)
    
    return [scenario_ids[idx] for idx in selected_indices]


def _compute_knn_objective(all_indices, T_indices, distance_matrix, k):
    """
    Compute objective value for P(k): Sum_{{s in S\\T}} (1/k) Sum_{{t in T_k(s)}} d(s,t)
    """
    T_set = set(T_indices)
    unselected = [s for s in all_indices if s not in T_set]
    
    if not unselected:
        return 0.0
    
    total_cost = 0.0
    
    for s in unselected:
        # Get distances from s to all selected scenarios
        distances_to_T = [(distance_matrix[s, t], t) for t in T_indices]
        distances_to_T.sort()  # Sort by distance
        
        # Take k nearest (or all if k > |T|)
        k_nearest = distances_to_T[:min(k, len(distances_to_T))]
        
        # Add average distance to k nearest
        avg_distance = sum(dist for dist, _ in k_nearest) / len(k_nearest)
        total_cost += avg_distance
    
    return total_cost


@profiled("representative_assignment_nearest_weighted")
def _build_knn_mapping(scenario_ids, feature_vectors, T, k, distance_matrix):
    """
    Build k-NN mapping for unselected scenarios.
    
    Returns:
        knn_mapping (dict): Maps each unselected scenario to its k-NN info
    """
    all_scenarios = set(scenario_ids)
    selected_set = set(T)
    unselected = all_scenarios - selected_set
    
    T_indices = [scenario_ids.index(sid) for sid in T]
    
    knn_mapping = {}
    
    for s in unselected:
        s_idx = scenario_ids.index(s)
        
        # Get distances to all selected scenarios
        distances_to_T = [
            (distance_matrix[s_idx, t_idx], scenario_ids[t_idx]) 
            for t_idx in T_indices
        ]
        distances_to_T.sort()  # Sort by distance, deterministic tie-breaking by ID
        
        # Take k nearest
        k_nearest = distances_to_T[:min(k, len(distances_to_T))]
        
        neighbors = [sid for _, sid in k_nearest]
        distances = [dist for dist, _ in k_nearest]
        
        knn_mapping[s] = {
            'neighbors': neighbors,
            'distances': distances,
            'k': len(neighbors)
        }
    
    logger.info(f"Built k-NN mapping for {len(knn_mapping)} unselected scenarios")
    return knn_mapping


# ============================
# TREE Method Implementation
# ============================

class TreeNode:
    """Node in the decision tree for tree-medoid joint selection."""
    
    def __init__(self, scenario_ids: List[int], depth: int = 0, node_id: int = 0):
        self.scenario_ids = scenario_ids
        self.depth = depth
        self.node_id = node_id
        self.is_leaf = False
        self.feature_idx = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.medoid: Optional[int] = None
        self.loss = float('inf')


@profiled("representative_selection_tree")
def solve_tree_integrated_selection(feature_vectors, p, tree_params=None):
    """
    Adaptive Tree-Based Selection with Medoid Representatives.
    
    Builds decision tree adaptively until reaching at most p leaves (scenarios to select).
    Tree grows naturally without artificial depth or size constraints - only stopping 
    criterion is leaf count ≤ p to ensure selection feasibility.
    
    Computes dissimilarity consistently from input feature_vectors:
    - For ML_Augmented_BD: dissimilarity is Euclidean distance in feature space
    - For ML_Augmented_BD_with_duals: dissimilarity is Euclidean distance in converted dual space
    
    Selection Rules:
    - If leaves == p: Exactly 1 scenario selected per leaf
    - If leaves < p: Proportional selection ensuring at least 1 per leaf, total exactly p
    
    Args:
        feature_vectors (dict): Dictionary {scenario_id: feature_vector}.
                               Can be:
                               - Graph walk features (ML_Augmented_BD)
                               - Converted dual numeric vectors (ML_Augmented_BD_with_duals)
                               - Other scenario representations
        p (int): Total number of scenarios to select (becomes max leaves constraint)
        tree_params (dict): Optional parameters:
            - 'min_impurity_decrease': Minimum improvement threshold (default: 0.0)
            - 'quantile_grid_size': Split threshold candidates (default: 10)
            
    Returns:
        tuple: (selected_scenarios, tree_mapping)
            - selected_scenarios (list): Selected scenario IDs (exactly p scenarios)
            - tree_mapping (dict): Leaf assignments and tree diagnostics
    """
    if tree_params is None:
        tree_params = {}
    
    # Extract minimal parameters (no depth/leaf size constraints)
    min_impurity_decrease = tree_params.get('min_impurity_decrease', 0.0)
    quantile_grid_size = tree_params.get('quantile_grid_size', 10)
    
    scenario_ids = list(feature_vectors.keys())
    num_scenarios = len(scenario_ids)
    
    # Validate inputs
    if p <= 0 or p > num_scenarios:
        raise ValueError(f"Invalid p={p}, must be in range [1, {num_scenarios}]")
    
    # logger.info(f"Building adaptive tree from {num_scenarios} scenarios with max {p} leaves constraint")
    
    # Precompute matrices for tree building
    feature_matrix = np.array([feature_vectors[sid] for sid in scenario_ids])
    # Compute dissimilarity matrix directly from feature vectors (consistent for all approaches)
    feature_dissim_matrix = _compute_feature_dissimilarity_matrix(feature_matrix)
    
    # Build tree adaptively until we reach at most p leaves
    root = TreeNode(list(range(num_scenarios)), depth=0, node_id=0)
    tree_info = _build_tree_adaptive(
        root, feature_matrix, feature_dissim_matrix, p, 
        min_impurity_decrease, quantile_grid_size
    )
    
    # Extract final leaves
    leaves = _extract_leaves(root)
    num_leaves = len(leaves)
    
    # DEBUG: Log tree structure
    # logger.warning(f"TREE_DEBUG: Built tree with {num_leaves} leaves")
    # logger.warning(f"TREE_DEBUG: Leaf sizes: {[len(leaf.scenario_ids) for leaf in leaves]}")
    
    # Emergency fallback: If tree resulted in single leaf, create manual split
    if num_leaves == 1 and len(scenario_ids) >= 2 and p >= 2:
        # logger.warning(f"TREE_DEBUG: Emergency fallback triggered! Single leaf with {len(scenario_ids)} scenarios")
        logger.warning(f"Tree resulted in single leaf with {len(scenario_ids)} scenarios. Creating emergency 50-50 split.")
        
        # Get the single leaf
        single_leaf = leaves[0]
        scenarios_in_leaf = single_leaf.scenario_ids
        
        # Create manual 50-50 split
        mid_point = len(scenarios_in_leaf) // 2
        left_scenarios = scenarios_in_leaf[:mid_point]  
        right_scenarios = scenarios_in_leaf[mid_point:]
        
        # Create two new leaf nodes
        left_node = TreeNode(left_scenarios, depth=1, node_id=1)
        right_node = TreeNode(right_scenarios, depth=1, node_id=2)
        
        # Update root to have children
        root.is_leaf = False
        root.left_child = left_node
        root.right_child = right_node
        root.feature_idx = 0  # Arbitrary feature for manual split
        root.threshold = 0.5  # Arbitrary threshold
        
        # Update leaves list
        leaves = [left_node, right_node]
        num_leaves = 2
        
        logger.info(f"Emergency split created: {len(left_scenarios)} + {len(right_scenarios)} = {len(scenarios_in_leaf)} scenarios in 2 leaves")
    else:
        # logger.warning(f"TREE_DEBUG: No emergency fallback needed. num_leaves={num_leaves}, len(scenario_ids)={len(scenario_ids)}, p={p}")
        pass
    
    logger.info(f"Adaptive tree built with {num_leaves} leaves (max allowed: {p}). Selecting {p} scenarios.")
    
    # Handle selection based on leaf count vs selection budget
    if num_leaves == p:
        # Case 1: Leaves == p, select exactly 1 per leaf (medoid)
        logger.info("Equal leaves and selection budget: 1 scenario per leaf")
        selected_scenarios = []
        leaf_anchors = {}
        
        for i, leaf in enumerate(leaves):
            medoid_idx = _compute_medoid(leaf.scenario_ids, feature_dissim_matrix)
            selected_id = scenario_ids[medoid_idx]
            selected_scenarios.append(selected_id)
            leaf_anchors[i] = [selected_id]
            
    elif num_leaves < p:
        # Case 2: Leaves < p, proportional allocation ensuring at least 1 per leaf
        logger.info(f"More budget than leaves ({p} > {num_leaves}): proportional allocation with minimum 1 per leaf")
        leaf_selections = _compute_adaptive_proportional_selections(leaves, p)
        
        selected_scenarios = []
        leaf_anchors = {}
        
        for i, leaf in enumerate(leaves):
            num_to_select = leaf_selections[i]
            leaf_scenario_indices = leaf.scenario_ids
            
            if num_to_select == 1:
                # Single selection: use medoid
                medoid_idx = _compute_medoid(leaf_scenario_indices, feature_dissim_matrix)
                selected_scenarios.append(scenario_ids[medoid_idx])
                leaf_anchors[i] = [scenario_ids[medoid_idx]]
            else:
                # Multiple selections: medoid + diversity-based selection
                leaf_selected = _select_diverse_scenarios_from_leaf(
                    leaf_scenario_indices, num_to_select, feature_dissim_matrix
                )
                leaf_anchor_ids = [scenario_ids[idx] for idx in leaf_selected]
                selected_scenarios.extend(leaf_anchor_ids)
                leaf_anchors[i] = leaf_anchor_ids
    
    else:
        # This should never happen with adaptive building, but safety check
        raise RuntimeError(f"Tree building error: {num_leaves} leaves > {p} selection budget")
    
    with profile_section("representative_assignment_nearest_weighted"):
        # Build prediction mappings.  The public mapping for unselected scenarios is
        # preserved for the existing nonparametric TREE predictor.  The all-scenario
        # transfer mapping is additional metadata used by the parametric RHS-Ridge
        # cross-weight experiments so that warm-up residuals can be formed for every
        # scenario using exactly the same tree weights.
        tree_mapping = {}
        all_transfer_mapping = {}
    
        for s_idx, scenario_id in enumerate(scenario_ids):
            # Find which leaf this scenario belongs to.
            leaf_idx = _find_scenario_leaf(s_idx, root, feature_matrix)
            anchor_ids = list(leaf_anchors[leaf_idx])
            weights = [1.0 / len(anchor_ids)] * len(anchor_ids)
            all_transfer_mapping[scenario_id] = {
                'anchors': anchor_ids,
                'weights': weights,
                'leaf_id': leaf_idx,
            }
            if scenario_id not in selected_scenarios:
                tree_mapping[scenario_id] = {
                    'anchor': anchor_ids,
                    'leaf_id': leaf_idx
                }
    
        # Metadata keys are ignored by the existing TREE prediction pathway.
        tree_mapping['__all_transfer_mapping__'] = all_transfer_mapping

    # Add comprehensive tree diagnostics
    tree_mapping['__tree_info__'] = {
        'num_leaves': num_leaves,
        'actual_depth': _compute_tree_depth(root),
        'leaf_sizes': [len(leaf.scenario_ids) for leaf in leaves],
        'selection_per_leaf': [len(leaf_anchors[i]) for i in range(num_leaves)],
        'total_selected': len(selected_scenarios),
        'anchors': leaf_anchors,
        'adaptive_building': True,
        'constraint_respected': num_leaves <= p,
        'tree_structure': _serialize_tree_structure(root, scenario_ids)
    }
    
    # Verify selection count
    assert len(selected_scenarios) == p, f"Selection error: got {len(selected_scenarios)}, expected {p}"
    
    logger.info(f"Tree selection complete: {len(selected_scenarios)} selected scenarios, "
                f"{sum(isinstance(key, int) for key in tree_mapping)} unselected scenarios mapped")
    
    return selected_scenarios, tree_mapping


def _compute_feature_dissimilarity_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean dissimilarity matrix from feature vectors.
    
    Uses Euclidean L2 norm for dissimilarity computation, applicable to:
    - ML_Augmented_BD: Feature vectors from graph walk embeddings
    - ML_Augmented_BD_with_duals: Converted dual numeric vectors
    - Other approaches: Their respective feature representations
    
    Args:
        feature_matrix: (n_scenarios, n_features) numpy array where each row is a feature vector
        
    Returns:
        Dissimilarity matrix D of shape (n_scenarios, n_scenarios) where:
        D[i,j] = ||feature_matrix[i] - feature_matrix[j]||_2^2
    """
    n_scenarios = feature_matrix.shape[0]
    dissim_matrix = np.zeros((n_scenarios, n_scenarios))
    
    for i in range(n_scenarios):
        for j in range(i + 1, n_scenarios):
            diff = feature_matrix[i] - feature_matrix[j]
            dissim = np.sum(diff ** 2)
            dissim_matrix[i, j] = dissim
            dissim_matrix[j, i] = dissim
    
    return dissim_matrix


def _precompute_dual_dissimilarity(scenario_ids: List[int], all_trial_duals: Dict) -> np.ndarray:
    """
    Precompute dual dissimilarity matrix D_dual(i,j) = sum_u |lambda^{i,u} - lambda^{j,u}|_2^2
    
    Args:
        scenario_ids: List of scenario IDs
        all_trial_duals: Dictionary with keys (scenario_id, iteration) and values as structured dual dictionaries
    """
    n = len(scenario_ids)
    dual_dissim = np.zeros((n, n))
    
    if not all_trial_duals:
        logger.warning("No dual data provided, using identity dissimilarity matrix")
        return dual_dissim
    
    def _convert_scflp_dual_to_vector(dual_dict, data):
        """Convert SCFLP structured dual dictionary to flat numpy vector."""
        if dual_dict is None:
            return None
        try:
            # SCFLP format: {'mu': {i: value}, 'nu': {j: value}}
            mu_values = [dual_dict['mu'][i] for i in sorted(dual_dict['mu'].keys())]
            nu_values = [dual_dict['nu'][j] for j in sorted(dual_dict['nu'].keys())]
            return np.array(mu_values + nu_values)
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to convert dual dictionary to vector: {e}")
            return None
    
    # Extract dual vectors for each scenario (average across all iterations for that scenario)
    dual_vectors = {}
    for i, sid in enumerate(scenario_ids):
        scenario_dual_vectors = []
        
        # Collect all dual vectors for this scenario across iterations
        for key, dual_dict in all_trial_duals.items():
            if len(key) >= 2 and key[0] == sid:  # key is (scenario_id, iteration) 
                # Convert structured dual dictionary to flat numpy vector
                if isinstance(dual_dict, dict):
                    dual_vector = _convert_scflp_dual_to_vector(dual_dict, None)
                    if dual_vector is not None:
                        scenario_dual_vectors.append(dual_vector)
                elif isinstance(dual_dict, np.ndarray):
                    scenario_dual_vectors.append(dual_dict)
                else:
                    logger.warning(f"Unexpected dual data type for scenario {sid}: {type(dual_dict)}")
        
        # Average dual vectors across iterations for this scenario
        if scenario_dual_vectors:
            # Ensure all vectors have the same length
            lengths = [len(vec) for vec in scenario_dual_vectors]
            if len(set(lengths)) > 1:
                logger.warning(f"Inconsistent dual vector lengths for scenario {sid}: {lengths}")
                # Use the first vector as reference
                ref_length = lengths[0]
                scenario_dual_vectors = [vec for vec in scenario_dual_vectors if len(vec) == ref_length]
            
            if scenario_dual_vectors:
                stacked = np.vstack(scenario_dual_vectors)
                dual_vectors[i] = np.mean(stacked, axis=0)
            else:
                logger.warning(f"No valid dual vectors for scenario {sid}, using zero vector")
                dual_vectors[i] = np.array([0.0])
        else:
            # No dual data for this scenario - use zero vector
            logger.warning(f"No dual data found for scenario {sid}, using zero vector")
            dual_vectors[i] = np.array([0.0])
    
    # Determine common vector length
    vector_lengths = [len(vec) for vec in dual_vectors.values()]
    if len(set(vector_lengths)) > 1:
        logger.warning(f"Inconsistent vector lengths across scenarios: {set(vector_lengths)}")
        # Use the most common length
        from collections import Counter
        most_common_length = Counter(vector_lengths).most_common(1)[0][0]
        logger.info(f"Using most common vector length: {most_common_length}")
        
        # Resize all vectors to the common length
        for i in range(n):
            if len(dual_vectors[i]) != most_common_length:
                if len(dual_vectors[i]) == 1 and dual_vectors[i][0] == 0.0:
                    # Zero vector case - expand to common length
                    dual_vectors[i] = np.zeros(most_common_length)
                else:
                    # Truncate or pad as needed
                    if len(dual_vectors[i]) > most_common_length:
                        dual_vectors[i] = dual_vectors[i][:most_common_length]
                    else:
                        dual_vectors[i] = np.pad(dual_vectors[i], (0, most_common_length - len(dual_vectors[i])))
    
    # Compute pairwise dissimilarities
    for i in range(n):
        for j in range(i, n):
            if i == j:
                dual_dissim[i, j] = 0.0
            else:
                diff = dual_vectors[i] - dual_vectors[j]
                dissim = np.sum(diff ** 2)
                dual_dissim[i, j] = dissim
                dual_dissim[j, i] = dissim
    
    return dual_dissim


def _compute_medoid(scenario_indices: List[int], dual_dissim_matrix: np.ndarray) -> int:
    """Compute medoid (scenario minimizing sum of dual distances to all others in group)."""
    if len(scenario_indices) == 1:
        return scenario_indices[0]
    
    best_medoid = scenario_indices[0]
    best_loss = float('inf')
    
    for candidate in scenario_indices:
        loss = sum(dual_dissim_matrix[candidate, other] for other in scenario_indices)
        if loss < best_loss or (loss == best_loss and candidate < best_medoid):
            best_loss = loss
            best_medoid = candidate
    
    return best_medoid


def _local_medoid_refinement(scenario_indices: List[int], dual_dissim_matrix: np.ndarray, 
                           current_medoid: int, max_swaps: int = 2) -> int:
    """Perform local medoid refinement with limited swaps."""
    current_loss = sum(dual_dissim_matrix[current_medoid, other] for other in scenario_indices)
    best_medoid = current_medoid
    best_loss = current_loss
    
    for _ in range(max_swaps):
        improved = False
        for candidate in scenario_indices:
            if candidate == best_medoid:
                continue
            candidate_loss = sum(dual_dissim_matrix[candidate, other] for other in scenario_indices)  
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_medoid = candidate
                improved = True
        
        if not improved:
            break
    
    return best_medoid


def _build_tree_adaptive(root: TreeNode, feature_matrix: np.ndarray, dual_dissim_matrix: np.ndarray,
                         max_leaves: int, min_impurity_decrease: float, quantile_grid_size: int) -> Dict:
    """
    Build decision tree adaptively until reaching at most max_leaves.
    
    Uses priority queue to expand leaves with highest loss reduction potential.
    Stops when next split would exceed max_leaves or no beneficial splits remain.
    
    GUARANTEE: Always creates at least 2 leaves by forcing the first split even 
    if it doesn't meet min_impurity_decrease threshold.
    
    Args:
        root: Root node containing all scenarios
        feature_matrix: Scenario features for split decisions 
        dual_dissim_matrix: Dual dissimilarity for loss computation
        max_leaves: Maximum number of leaves allowed (= selection budget p)
        min_impurity_decrease: Minimum loss reduction required for splitting
        quantile_grid_size: Number of split threshold candidates per feature
    
    Returns:
        dict: Tree building statistics and diagnostics
    """
    import heapq
    
    # Initialize root loss
    root.loss = _compute_node_loss(root.scenario_ids, dual_dissim_matrix)
    
    # Priority queue: (-loss_reduction_potential, node_id, node)
    # Use negative loss for max heap behavior
    node_counter = 0
    pq = [(-root.loss, node_counter, root)]
    node_counter += 1
    
    current_leaves = 1
    total_splits = 0
    first_split_forced = False
    
    # logger.debug(f"Starting adaptive tree building: initial loss={root.loss:.4f}, max_leaves={max_leaves}")
    
    while pq and current_leaves < max_leaves:
        neg_loss_potential, _, current_node = heapq.heappop(pq)
        
        # Skip if node was already split (stale queue entry)
        if not current_node.is_leaf:
            continue
            
        # Skip if node has too few scenarios to split meaningfully
        if len(current_node.scenario_ids) < 2:
            continue
        
        # For the first split (to guarantee at least 2 leaves), relax criteria if needed
        effective_min_impurity = min_impurity_decrease
        force_any_split = False
        if total_splits == 0 and not first_split_forced:
            effective_min_impurity = -float('inf')  # Accept any split for first split
            
        # Attempt to find best split for this node
        best_split = _find_adaptive_split(
            current_node, feature_matrix, dual_dissim_matrix,
            effective_min_impurity, quantile_grid_size, force_split=False
        )
        
        if best_split is None:
            # No beneficial split found
            if total_splits == 0 and len(current_node.scenario_ids) >= 2:
                # Force a split for the first split to guarantee at least 2 leaves
                logger.info(f"No splits found with normal criteria - forcing aggressive split to guarantee at least 2 leaves")
                best_split = _find_adaptive_split(
                    current_node, feature_matrix, dual_dissim_matrix,
                    -float('inf'), quantile_grid_size, force_split=True  # Force any possible split
                )
                first_split_forced = True
            
            if best_split is None:
                continue
            
        feature_idx, threshold, left_indices, right_indices, loss_reduction = best_split
        
        # Verify split creates two non-empty children
        if len(left_indices) == 0 or len(right_indices) == 0:
            continue
            
        # Perform the split
        current_node.is_leaf = False
        current_node.feature_idx = feature_idx
        current_node.threshold = threshold
        current_node.loss_reduction = loss_reduction
        
        # Create child nodes
        left_node = TreeNode(left_indices, current_node.depth + 1, node_counter)
        right_node = TreeNode(right_indices, current_node.depth + 1, node_counter + 1)
        node_counter += 2
        
        current_node.left_child = left_node
        current_node.right_child = right_node
        
        # Compute child losses
        left_node.loss = _compute_node_loss(left_indices, dual_dissim_matrix)
        right_node.loss = _compute_node_loss(right_indices, dual_dissim_matrix)
        
        # Add children to priority queue for further expansion
        heapq.heappush(pq, (-left_node.loss, node_counter, left_node))
        heapq.heappush(pq, (-right_node.loss, node_counter + 1, right_node))
        
        current_leaves += 1  # Net gain of 1 leaf (1 split -> 2 children)
        total_splits += 1
        
        if first_split_forced and total_splits == 1:
            logger.info(f"First split completed (forced): feature {feature_idx}, threshold {threshold:.4f}, "
                       f"loss reduction {loss_reduction:.4f}, leaves now {current_leaves}")
        else:
            logger.debug(f"Split node {current_node.node_id}: feature {feature_idx}, threshold {threshold:.4f}, "
                        f"loss reduction {loss_reduction:.4f}, leaves now {current_leaves}")
    
    final_leaves = len(_extract_leaves(root))
    
    # Final validation: ensure at least 2 leaves
    if final_leaves < 2:
        logger.warning(f"Tree building resulted in {final_leaves} leaves, which is less than the minimum of 2")
    
    logger.info(f"Adaptive tree building complete: {total_splits} splits, {final_leaves} leaves, "
               f"depth {_compute_tree_depth(root)}")
    
    return {
        'total_splits': total_splits,
        'final_leaves': final_leaves,
        'final_depth': _compute_tree_depth(root),
        'constraint_satisfied': final_leaves <= max_leaves,
        'min_leaves_guaranteed': final_leaves >= 2,
        'first_split_forced': first_split_forced,
        'adaptive_method': True
    }


def _find_adaptive_split(node: TreeNode, feature_matrix: np.ndarray, dual_dissim_matrix: np.ndarray,
                        min_impurity_decrease: float, quantile_grid_size: int, force_split: bool = False) -> Optional[Tuple]:
    """
    Find best split for a node in adaptive tree building.
    
    Evaluates all features and quantile-based thresholds to find split that
    maximizes loss reduction (minimizes weighted sum of child losses).
    
    Args:
        node: TreeNode to split
        feature_matrix: Feature matrix for all scenarios
        dual_dissim_matrix: Dual dissimilarity matrix
        min_impurity_decrease: Minimum required loss reduction
        quantile_grid_size: Number of threshold candidates
        force_split: If True, relaxes strict criteria to force any possible split
    
    Returns:
        tuple: (feature_idx, threshold, left_indices, right_indices, loss_reduction) or None
    """
    if len(node.scenario_ids) < 2:
        return None
        
    node_scenarios = np.array(node.scenario_ids)
    node_features = feature_matrix[node_scenarios]
    num_features = node_features.shape[1]
    current_loss = _compute_node_loss(node.scenario_ids, dual_dissim_matrix)
    
    best_split = None
    best_loss_reduction = min_impurity_decrease
    
    # Relaxed thresholds for forced splitting
    variance_threshold = 1e-12 if force_split else 1e-8
    min_grid_size = 1 if force_split else 1
    
    # Try each feature dimension
    for f in range(num_features):
        feature_values = node_features[:, f]
        
        # Skip constant features (with relaxed threshold for forced splits)
        if np.std(feature_values) < variance_threshold:
            if force_split:
                # logger.debug(f"Feature {f}: std={np.std(feature_values):.2e} < {variance_threshold:.2e}, but forcing split attempt")
                # Continue anyway for forced splits
                pass
            else:
                continue
            
        # Generate quantile-based thresholds
        unique_values = np.unique(feature_values)
        if len(unique_values) < 2:
            if force_split and len(unique_values) == 1:
                # For forced splits, try median split even with single unique value
                # logger.debug(f"Feature {f}: Only {len(unique_values)} unique values, forcing median split")
                median_val = unique_values[0]
                # Create split at median + small epsilon
                thresholds = [median_val + 1e-10, median_val - 1e-10]
            else:
                continue
        else:
            # Use quantiles up to the grid size
            n_thresholds = min(quantile_grid_size, len(unique_values) - 1)
            if n_thresholds <= 0:
                if force_split:
                    n_thresholds = 1  # Force at least one threshold
                else:
                    continue
                    
            quantiles = np.linspace(0, 1, n_thresholds + 2)[1:-1]  # Exclude 0 and 1
            thresholds = np.quantile(unique_values, quantiles)
        
        # Evaluate each threshold
        for threshold in thresholds:
            # Split scenarios based on threshold
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            # Skip degenerate splits (but be more lenient for forced splits)
            if force_split:
                # For forced splits, accept even very unbalanced splits
                if (np.sum(left_mask) == 0 and np.sum(right_mask) == len(feature_values)) or \
                   (np.sum(right_mask) == 0 and np.sum(left_mask) == len(feature_values)):
                    continue  # Only skip completely degenerate splits
            else:
                # Normal strict criteria
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                    
            left_indices = node_scenarios[left_mask].tolist()
            right_indices = node_scenarios[right_mask].tolist()
            
            # Ensure we have valid splits
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            
            # Compute child losses
            left_loss = _compute_node_loss(left_indices, dual_dissim_matrix)
            right_loss = _compute_node_loss(right_indices, dual_dissim_matrix)
            
            # Weighted average of child losses
            left_weight = len(left_indices) / len(node.scenario_ids)
            right_weight = len(right_indices) / len(node.scenario_ids)
            weighted_child_loss = left_weight * left_loss + right_weight * right_loss
            
            # Loss reduction achieved by this split
            loss_reduction = current_loss - weighted_child_loss
            
            if loss_reduction > best_loss_reduction:
                best_loss_reduction = loss_reduction
                best_split = (f, threshold, left_indices, right_indices, loss_reduction)
                if force_split:
                    logger.debug(f"Forced split found: feature {f}, threshold {threshold:.6f}, "
                               f"left={len(left_indices)}, right={len(right_indices)}, loss_reduction={loss_reduction:.6f}")
    
    if force_split and best_split is None:
        logger.warning(f"Even forced splitting failed to find any valid split for node with {len(node.scenario_ids)} scenarios")
    
    return best_split


def _compute_adaptive_proportional_selections(leaves: List, p: int) -> List[int]:
    """
    Compute proportional scenario selections ensuring exactly p total and at least 1 per leaf.
    
    Improved version that handles fractional allocations more robustly.
    
    Args:
        leaves: List of TreeNode leaf objects
        p: Total number of scenarios to select
    
    Returns:
        list: Number of scenarios to select from each leaf (sums to exactly p)
    """
    num_leaves = len(leaves)
    leaf_sizes = [len(leaf.scenario_ids) for leaf in leaves]
    total_scenarios = sum(leaf_sizes)
    
    # Validate constraints
    if p < num_leaves:
        raise ValueError(f"Cannot select {p} scenarios with {num_leaves} leaves (minimum 1 per leaf required)")
        
    if p > total_scenarios:
        raise ValueError(f"Cannot select {p} scenarios from {total_scenarios} available scenarios")
    
    # Start with minimum allocation: 1 per leaf
    selections = [1] * num_leaves
    remaining = p - num_leaves
    
    if remaining == 0:
        return selections
    
    # Distribute remaining scenarios proportionally by leaf size
    # Use largest remainder method for exact allocation
    proportions = [size / total_scenarios for size in leaf_sizes]
    fractional_allocation = [remaining * prop for prop in proportions]
    
    # Integer parts
    integer_parts = [int(frac) for frac in fractional_allocation]
    remainders = [frac - int_part for frac, int_part in zip(fractional_allocation, integer_parts)]
    
    # Allocate integer parts (respecting leaf size limits)
    for i in range(num_leaves):
        max_additional = leaf_sizes[i] - 1  # Already allocated 1
        actual_allocation = min(integer_parts[i], max_additional)
        selections[i] += actual_allocation
        remaining -= actual_allocation
    
    # Distribute remaining using largest remainder method
    if remaining > 0:
        # Sort by remainder size (descending)
        remainder_indices = sorted(range(num_leaves), key=lambda i: remainders[i], reverse=True)
        
        allocated_in_remainder = 0
        for i in remainder_indices:
            if allocated_in_remainder >= remaining:
                break
                
            # Check if we can allocate one more to this leaf
            if selections[i] < leaf_sizes[i]:
                selections[i] += 1
                allocated_in_remainder += 1
    
    # Verify allocation
    assert sum(selections) == p, f"Allocation error: {sum(selections)} != {p}"
    assert all(sel >= 1 for sel in selections), f"Minimum constraint violated: {selections}"
    assert all(selections[i] <= leaf_sizes[i] for i in range(num_leaves)), "Leaf size exceeded"
    
    # DEBUG: Log the actual selection allocation
    # logger.warning(f"PROPORTIONAL_DEBUG: num_leaves={num_leaves}, p={p}")
    # logger.warning(f"PROPORTIONAL_DEBUG: leaf_sizes={leaf_sizes}")
    # logger.warning(f"PROPORTIONAL_DEBUG: final selections={selections}, total={sum(selections)}")
    
    return selections


def _select_diverse_scenarios_from_leaf(leaf_scenario_indices: List[int], num_to_select: int, 
                                      dual_dissim_matrix: np.ndarray) -> List[int]:
    """
    Select diverse scenarios from a leaf using medoid + max-min diversity strategy.
    
    Args:
        leaf_scenario_indices: Scenarios available in this leaf
        num_to_select: Number of scenarios to select
        dual_dissim_matrix: Dual dissimilarity matrix
    
    Returns:
        list: Selected scenario indices (in original scenario numbering)
    """
    if num_to_select <= 0:
        return []
    
    if num_to_select >= len(leaf_scenario_indices):
        return leaf_scenario_indices.copy()
    
    selected = []
    remaining = leaf_scenario_indices.copy()
    
    # First selection: medoid (most central scenario)
    medoid_idx = _compute_medoid(leaf_scenario_indices, dual_dissim_matrix)
    selected.append(medoid_idx)
    remaining.remove(medoid_idx)
    
    # Subsequent selections: maximize minimum distance to already selected
    while len(selected) < num_to_select and remaining:
        best_candidate = None
        max_min_distance = -1
        
        for candidate in remaining:
            # Find minimum distance to any already selected scenario
            min_dist = min(dual_dissim_matrix[candidate, sel] for sel in selected)
            
            if min_dist > max_min_distance:
                max_min_distance = min_dist
                best_candidate = candidate
        
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            # Safety: add random remaining scenario
            selected.append(remaining[0])
            remaining.pop(0)
    
    return selected


def _build_tree_natural(root: TreeNode, feature_matrix: np.ndarray, dual_dissim_matrix: np.ndarray,
                       D_max: int, min_impurity_decrease: float, quantile_grid_size: int) -> Dict:
    """Build decision tree naturally without targeting specific number of leaves."""
    
    # Priority queue: (negative_loss, node_id, node) - max heap by loss  
    node_counter = 0
    pq = []
    
    # Initialize root loss
    root.loss = _compute_node_loss(root.scenario_ids, dual_dissim_matrix)
    heapq.heappush(pq, (-root.loss, node_counter, root))
    node_counter += 1
    
    splits_made = 0
    
    while pq:
        neg_loss, _, node = heapq.heappop(pq)
        
        if node.is_leaf:
            continue
            
        # Try to split this node
        best_split = _find_best_split_natural(
            node, feature_matrix, dual_dissim_matrix,
            D_max, min_impurity_decrease, quantile_grid_size
        )
        
        if best_split is None:
            # Cannot split, mark as leaf
            node.is_leaf = True
            continue
        
        # Apply the split
        feature_idx, threshold, left_indices, right_indices, gain = best_split
        
        node.feature_idx = feature_idx
        node.threshold = threshold
        
        # Create children
        left_child = TreeNode(left_indices, node.depth + 1, node_counter)
        right_child = TreeNode(right_indices, node.depth + 1, node_counter + 1)
        
        left_child.loss = _compute_node_loss(left_indices, dual_dissim_matrix)
        right_child.loss = _compute_node_loss(right_indices, dual_dissim_matrix)
        
        node.left_child = left_child
        node.right_child = right_child
        
        # Add children to queue if they can potentially split
        if left_child.depth < D_max and len(left_indices) >= 2:
            heapq.heappush(pq, (-left_child.loss, node_counter, left_child))
        else:
            left_child.is_leaf = True
            
        node_counter += 1
            
        if right_child.depth < D_max and len(right_indices) >= 2:
            heapq.heappush(pq, (-right_child.loss, node_counter, right_child))
        else:
            right_child.is_leaf = True
            
        node_counter += 1
        splits_made += 1
    
    # Mark any remaining nodes in queue as leaves
    while pq:
        _, _, node = heapq.heappop(pq)
        node.is_leaf = True
    
    return {'splits_made': splits_made}


def _find_best_split_natural(node: TreeNode, feature_matrix: np.ndarray, dual_dissim_matrix: np.ndarray,
                            D_max: int, min_impurity_decrease: float, 
                            quantile_grid_size: int) -> Optional[Tuple]:
    """Find the best split for a node in natural tree building (no min_leaf_size constraint)."""
    
    if node.depth >= D_max or len(node.scenario_ids) < 2:
        return None
    
    scenario_indices = node.scenario_ids
    n_features = feature_matrix.shape[1]
    parent_loss = node.loss
    best_gain = -1
    best_split = None
    
    for feature_idx in range(n_features):
        # Get feature values for scenarios in this node
        feature_values = feature_matrix[scenario_indices, feature_idx]
        
        # Generate quantile-based thresholds
        if len(np.unique(feature_values)) <= 1:
            continue
            
        quantiles = np.linspace(0.1, 0.9, quantile_grid_size)
        thresholds = np.percentile(feature_values, quantiles * 100)
        thresholds = np.unique(thresholds)
        
        for threshold in thresholds:
            # Split scenarios
            left_mask = feature_values <= threshold
            left_indices = [scenario_indices[i] for i in range(len(scenario_indices)) if left_mask[i]]
            right_indices = [scenario_indices[i] for i in range(len(scenario_indices)) if not left_mask[i]]
            
            # Check validity (both sides must have at least 1 scenario)
            if len(left_indices) < 1 or len(right_indices) < 1:
                continue
            
            # Compute gain
            left_loss = _compute_node_loss(left_indices, dual_dissim_matrix)
            right_loss = _compute_node_loss(right_indices, dual_dissim_matrix)
            total_child_loss = left_loss + right_loss
            gain = parent_loss - total_child_loss
            
            # Update best split (deterministic tie-breaking)
            if (gain > best_gain or 
                (gain == best_gain and best_split is not None and feature_idx < best_split[0]) or
                (gain == best_gain and best_split is not None and feature_idx == best_split[0] and threshold < best_split[1])):
                best_gain = gain
                best_split = (feature_idx, threshold, left_indices, right_indices, gain)
    
    if best_gain < min_impurity_decrease:
        return None
    
    return best_split


def _build_tree_greedy(root: TreeNode, feature_matrix: np.ndarray, dual_dissim_matrix: np.ndarray,
                      target_leaves: int, D_max: int, n_min: int, min_impurity_decrease: float,
                      quantile_grid_size: int) -> Dict:
    """Build decision tree using greedy top-down approach."""
    
    # Priority queue: (negative_loss, node_id, node) - max heap by loss
    # Use unique node_id as tiebreaker to avoid TreeNode comparison
    node_counter = 0
    pq = []
    
    # Initialize root loss
    root.loss = _compute_node_loss(root.scenario_ids, dual_dissim_matrix)
    heapq.heappush(pq, (-root.loss, node_counter, root))
    node_counter += 1
    
    current_leaves = 1
    splits_made = 0
    max_splits_without_progress = 3  # Stop if we can't make progress
    
    while pq and current_leaves < target_leaves:
        neg_loss, _, node = heapq.heappop(pq)
        
        if node.is_leaf:
            continue
            
        # Try to split this node
        best_split = _find_best_split(
            node, feature_matrix, dual_dissim_matrix, 
            D_max, n_min, min_impurity_decrease, quantile_grid_size
        )
        
        if best_split is None:
            # Cannot split, mark as leaf
            node.is_leaf = True
            splits_made += 1
            if splits_made >= max_splits_without_progress:
                # logger.debug(f"Stopping tree construction: no valid splits found after {splits_made} attempts")
                break
            continue
        
        # Apply the split
        feature_idx, threshold, left_indices, right_indices, gain = best_split
        
        node.feature_idx = feature_idx
        node.threshold = threshold
        
        # Create children
        left_child = TreeNode(left_indices, node.depth + 1, node_counter)
        right_child = TreeNode(right_indices, node.depth + 1, node_counter + 1)
        
        left_child.loss = _compute_node_loss(left_indices, dual_dissim_matrix)
        right_child.loss = _compute_node_loss(right_indices, dual_dissim_matrix)
        
        node.left_child = left_child
        node.right_child = right_child
        
        # Add children to queue if they can potentially split
        if left_child.depth < D_max and len(left_indices) >= 2 * n_min:
            heapq.heappush(pq, (-left_child.loss, node_counter, left_child))
        else:
            left_child.is_leaf = True
            
        node_counter += 1  # Increment after using for left child
            
        if right_child.depth < D_max and len(right_indices) >= 2 * n_min:
            heapq.heappush(pq, (-right_child.loss, node_counter, right_child))
        else:
            right_child.is_leaf = True
            
        node_counter += 1  # Increment after using for right child
        
        current_leaves += 1  # Split adds one leaf
        splits_made = 0  # Reset counter on successful split
    
    # Mark remaining nodes in queue as leaves
    while pq:
        _, _, node = heapq.heappop(pq)
        node.is_leaf = True
    
    return {'final_leaves': current_leaves, 'target_leaves': target_leaves}


def _compute_node_loss(scenario_indices: List[int], dual_dissim_matrix: np.ndarray) -> float:
    """Compute medoid loss for a node."""
    if len(scenario_indices) <= 1:
        return 0.0
    
    medoid = _compute_medoid(scenario_indices, dual_dissim_matrix)
    return sum(dual_dissim_matrix[medoid, idx] for idx in scenario_indices)


def _find_best_split(node: TreeNode, feature_matrix: np.ndarray, dual_dissim_matrix: np.ndarray,
                    D_max: int, n_min: int, min_impurity_decrease: float, 
                    quantile_grid_size: int) -> Optional[Tuple]:
    """Find the best split for a node."""
    
    if node.depth >= D_max or len(node.scenario_ids) < 2 * n_min:
        return None
    
    scenario_indices = node.scenario_ids
    n_features = feature_matrix.shape[1]
    parent_loss = node.loss
    best_gain = -1
    best_split = None
    
    for feature_idx in range(n_features):
        # Get feature values for scenarios in this node
        feature_values = feature_matrix[scenario_indices, feature_idx]
        
        # Generate quantile-based thresholds
        if len(np.unique(feature_values)) <= 1:
            continue
            
        quantiles = np.linspace(0.1, 0.9, quantile_grid_size)
        thresholds = np.percentile(feature_values, quantiles * 100)
        thresholds = np.unique(thresholds)
        
        for threshold in thresholds:
            # Split scenarios
            left_mask = feature_values <= threshold
            left_indices = [scenario_indices[i] for i in range(len(scenario_indices)) if left_mask[i]]
            right_indices = [scenario_indices[i] for i in range(len(scenario_indices)) if not left_mask[i]]
            
            # Check validity
            if len(left_indices) < n_min or len(right_indices) < n_min:
                continue
            
            # Compute gain
            left_loss = _compute_node_loss(left_indices, dual_dissim_matrix)
            right_loss = _compute_node_loss(right_indices, dual_dissim_matrix)
            total_child_loss = left_loss + right_loss
            gain = parent_loss - total_child_loss
            
            # Update best split (deterministic tie-breaking)
            if (gain > best_gain or 
                (gain == best_gain and best_split is not None and feature_idx < best_split[0]) or
                (gain == best_gain and best_split is not None and feature_idx == best_split[0] and threshold < best_split[1])):
                best_gain = gain
                best_split = (feature_idx, threshold, left_indices, right_indices, gain)
    
    if best_gain < min_impurity_decrease:
        return None
    
    return best_split


def _extract_leaves(root: TreeNode) -> List[TreeNode]:
    """Extract all leaf nodes from tree."""
    leaves = []
    
    def traverse(node):
        if node.is_leaf or (node.left_child is None and node.right_child is None):
            leaves.append(node)
        else:
            if node.left_child:
                traverse(node.left_child)
            if node.right_child:
                traverse(node.right_child)
    
    traverse(root)
    return leaves


def _find_scenario_leaf(scenario_idx: int, root: TreeNode, feature_matrix: np.ndarray) -> int:
    """
    Find which leaf a scenario belongs to by checking leaf membership.
    
    This function directly checks which leaf contains the scenario, which is more
    reliable than tree traversal, especially for emergency manual splits where
    the feature/threshold split doesn't match the actual scenario assignment.
    
    Args:
        scenario_idx: Index of the scenario
        root: Root node of the tree
        feature_matrix: Feature matrix (not used in membership-based approach)
        
    Returns:
        Leaf index containing the scenario
    """
    # Get all leaves
    leaves = _extract_leaves(root)
    
    # Find which leaf contains this scenario
    for i, leaf in enumerate(leaves):
        if scenario_idx in leaf.scenario_ids:
            return i
    
    # Fallback: if not found in any leaf (shouldn't happen), use tree traversal
    logger.warning(f"Scenario {scenario_idx} not found in any leaf! Using tree traversal as fallback.")
    node = root
    
    while not node.is_leaf and node.left_child is not None and node.right_child is not None:
        feature_value = feature_matrix[scenario_idx, node.feature_idx]
        if feature_value <= node.threshold:
            node = node.left_child
        else:
            node = node.right_child
    
    # Find the leaf index
    for i, leaf in enumerate(leaves):
        if leaf is node:
            return i
    
    return 0  # Final fallback


def _compute_tree_depth(root: TreeNode) -> int:
    """Compute actual depth of the tree."""
    if root.is_leaf or (root.left_child is None and root.right_child is None):
        return root.depth
    
    max_depth = root.depth
    if root.left_child:
        max_depth = max(max_depth, _compute_tree_depth(root.left_child))
    if root.right_child:
        max_depth = max(max_depth, _compute_tree_depth(root.right_child))
    
    return max_depth


def _serialize_tree_structure(root: TreeNode, scenario_ids: List[int]) -> Dict:
    """Serialize tree structure for diagnostics."""
    def serialize_node(node):
        result = {
            'node_id': node.node_id,
            'depth': node.depth,
            'is_leaf': node.is_leaf,
            'num_scenarios': len(node.scenario_ids),
            'loss': node.loss
        }
        
        if not node.is_leaf and node.feature_idx is not None:
            result['feature_idx'] = node.feature_idx
            result['threshold'] = node.threshold
            
        if node.left_child:
            result['left_child'] = serialize_node(node.left_child)
        if node.right_child:
            result['right_child'] = serialize_node(node.right_child)
            
        if node.medoid is not None:
            result['medoid_scenario'] = scenario_ids[node.medoid]
        
        return result
    
    return serialize_node(root)


# ============================
# DUAL_CLUSTERING Method Implementation
# ============================

@profiled("representative_selection_clustering")
def solve_dual_clustering_selection(feature_vectors, p, clustering_params=None):
    """
    Dual clustering-based selection + mapping for scenario selection.
    
    This method creates clusters of scenarios based on their feature vectors and selects
    one representative scenario per cluster. The remaining scenarios in each cluster
    are predicted using the exact dual solution of their cluster representative.
    
    Algorithm:
    1. Apply K-means clustering to feature vectors with k = p clusters
    2. For each cluster, select the scenario closest to the cluster centroid as representative
    3. Create mapping from unselected scenarios to their cluster representative
    
    Args:
        feature_vectors (dict): Dictionary {scenario_id: feature_vector}
        p (int): Number of clusters (and thus selected scenarios)
        clustering_params (dict): Optional parameters for clustering
            - random_state (int): Random seed for reproducible clustering (default: 42)
            - init_method (str): Initialization method for K-means (default: 'k-means++')
            - max_iter (int): Maximum iterations for K-means (default: 300)
            - tol (float): Tolerance for convergence (default: 1e-4)
        
    Returns:
        tuple: (selected_subproblems, clustering_mapping)
            - selected_subproblems (list): List of selected scenario IDs (cluster representatives)
            - clustering_mapping (dict): For each unselected scenario, maps to:
                {'representative': cluster_representative_id, 'cluster_id': cluster_index}
    """
    if clustering_params is None:
        clustering_params = {}
    
    # Use actual scenario IDs from feature_vectors instead of assuming 0-based indexing
    scenario_ids = list(feature_vectors.keys())
    num_scenarios = len(scenario_ids)
    
    if p <= 0 or p > num_scenarios:
        raise ValueError(f"p must be between 1 and {num_scenarios}, got {p}")
    
    if p >= num_scenarios:
        logger.warning(f"p={p} >= num_scenarios={num_scenarios}, selecting all scenarios")
        return scenario_ids, {}
    
    # logger.info(f"Starting DUAL_CLUSTERING selection with {num_scenarios} scenarios, {p} clusters")
    
    # Convert feature vectors to matrix using actual scenario IDs
    feature_matrix = np.array([feature_vectors[sid] for sid in scenario_ids])
    
    # Set up clustering parameters
    random_state = clustering_params.get('random_state', 42)
    init_method = clustering_params.get('init_method', 'k-means++')
    max_iter = clustering_params.get('max_iter', 300)
    tol = clustering_params.get('tol', 1e-4)
    
    # Perform K-means clustering
    # logger.info(f"Performing K-means clustering with {p} clusters")
    kmeans = KMeans(
        n_clusters=p,
        random_state=random_state,
        init=init_method,
        max_iter=max_iter,
        tol=tol,
        n_init=10
    )
    
    cluster_labels = kmeans.fit_predict(feature_matrix)
    cluster_centers = kmeans.cluster_centers_
    
    # For each cluster, find the scenario closest to the centroid using actual scenario IDs
    selected_subproblems = []
    cluster_representatives = {}  # cluster_id -> representative_scenario_id
    
    for cluster_id in range(p):
        # Find all scenarios in this cluster using actual scenario IDs
        cluster_members_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        if not cluster_members_indices:
            logger.warning(f"Cluster {cluster_id} is empty, skipping")
            continue
        
        # Calculate distances from cluster members to centroid
        cluster_center = cluster_centers[cluster_id]
        min_distance = float('inf')
        representative_idx = None
        
        for member_idx in cluster_members_indices:
            distance = np.linalg.norm(feature_matrix[member_idx] - cluster_center)
            if distance < min_distance:
                min_distance = distance
                representative_idx = member_idx
        
        # Use actual scenario ID as representative
        representative_scenario_id = scenario_ids[representative_idx]
        selected_subproblems.append(representative_scenario_id)
        cluster_representatives[cluster_id] = representative_scenario_id
        
        logger.debug(f"Cluster {cluster_id}: {len(cluster_members_indices)} scenarios, "
                    f"representative: {representative_scenario_id}")
    
    with profile_section("representative_assignment_nearest_weighted"):
        # Create mapping for unselected scenarios using actual scenario IDs.
        # Also retain an all-scenario one-hot transfer map.  The latter lets the
        # parametric RHS-Ridge variant use exactly the same clustering weights as
        # the nonparametric predictor while still training on all warm-up scenarios.
        clustering_mapping = {}
        all_transfer_mapping = {}
        for i, scenario_id in enumerate(scenario_ids):
            cluster_id = cluster_labels[i]
            if cluster_id in cluster_representatives:
                representative_id = cluster_representatives[cluster_id]
            else:
                logger.warning(f"No representative found for cluster {cluster_id}, mapping scenario {scenario_id} to itself")
                representative_id = scenario_id

            all_transfer_mapping[scenario_id] = {
                'anchors': [representative_id],
                'weights': [1.0],
                'representative': representative_id,
                'cluster_id': cluster_id,
            }

            if scenario_id not in selected_subproblems:
                clustering_mapping[scenario_id] = {
                    'representative': representative_id,
                    'cluster_id': cluster_id
                }

        clustering_mapping['__all_transfer_mapping__'] = all_transfer_mapping
    
    logger.info(f"DUAL_CLUSTERING completed: selected {len(selected_subproblems)} representatives, "
                f"mapped {len(clustering_mapping)-1} unselected scenarios")
    
    # Log cluster statistics
    for cluster_id in range(p):
        cluster_size = sum(1 for label in cluster_labels if label == cluster_id)
        if cluster_id in cluster_representatives:
            rep_id = cluster_representatives[cluster_id]
            logger.debug(f"Cluster {cluster_id}: size={cluster_size}, representative={rep_id}")
    
    return selected_subproblems, clustering_mapping


# ============================
# DUAL_CLUSTERING_WITH_DUAL Method Implementation
# ============================

@profiled("representative_selection_clustering")
def solve_dual_clustering_with_dual_selection(all_trial_duals, p, clustering_params=None):
    """
    DUAL_CLUSTERING_WITH_DUAL selection + mapping for scenario selection.
    
    This method clusters scenarios based on their actual dual solutions from Phase 0 and Phase 1,
    rather than using feature vectors. It builds stacked, normalized dual profiles and uses
    K-means clustering (same as DUAL_CLUSTERING but on dual profiles instead of features).
    
    Algorithm:
    1. Build stacked dual profiles from all Phase 0 and Phase 1 dual solutions
    2. Normalize each per-iteration dual block to unit length
    3. Apply K-means clustering to dual profiles with k = p clusters
    4. For each cluster, select the scenario closest to the cluster centroid as representative
    5. Create mapping from unselected scenarios to their cluster representative
    
    Args:
        all_trial_duals (dict): Dictionary {(iteration, scenario_id): dual_dict} from Phase 0 and Phase 1
            Phase 0 iterations use negative numbers: (-1, sid), (-2, sid), etc.
            Phase 1 iterations use positive numbers: (sid, 1), (sid, 2), etc.
        p (int): Number of clusters (and thus selected scenarios)
        clustering_params (dict): Optional parameters for clustering
            - random_state (int): Random seed for reproducible clustering (default: 42)
            - init_method (str): Initialization method for K-means (default: 'k-means++')
            - max_iter (int): Maximum iterations for K-means (default: 300)
            - tol (float): Tolerance for convergence (default: 1e-4)
        
    Returns:
        tuple: (selected_subproblems, dual_clustering_mapping)
            - selected_subproblems (list): List of selected scenario IDs (cluster representatives)
            - dual_clustering_mapping (dict): For each unselected scenario, maps to:
                {'representative': representative_scenario_id, 'cluster_id': cluster_index}
    """
    if clustering_params is None:
        clustering_params = {}
    
    random_state = clustering_params.get('random_state', 42)
    init_method = clustering_params.get('init_method', 'k-means++')
    max_iter = clustering_params.get('max_iter', 300)
    tol = clustering_params.get('tol', 1e-4)
    
    logger.info(f"Starting DUAL_CLUSTERING_WITH_DUAL selection with p={p}")
    logger.info(f"all_trial_duals keys: {len(all_trial_duals)} entries")
    if len(all_trial_duals) > 0:
        logger.info(f"Sample key from all_trial_duals: {next(iter(all_trial_duals.keys()))}")
    logger.info(f"Clustering parameters: {clustering_params}")
    
    # Step 1: Build stacked dual profiles for each scenario
    logger.info("About to call _build_stacked_dual_profiles")
    dual_profiles = _build_stacked_dual_profiles(all_trial_duals)
    logger.info(f"Built {len(dual_profiles)} dual profiles")
    
    if len(dual_profiles) < p:
        logger.warning(f"Only {len(dual_profiles)} scenarios available, but {p} requested. Using all scenarios.")
        return list(dual_profiles.keys()), {}
    
    scenario_ids = list(dual_profiles.keys())
    logger.info(f"Built dual profiles for {len(scenario_ids)} scenarios")
    
    # Step 2: Convert dual profiles to matrix for K-means
    dual_profile_matrix = np.array([dual_profiles[sid] for sid in scenario_ids])
    logger.info(f"Dual profile matrix shape: {dual_profile_matrix.shape}")
    
    # Step 3: Apply K-means clustering (same as DUAL_CLUSTERING but on dual profiles)
    selected_subproblems, dual_clustering_mapping = _kmeans_clustering_on_duals(
        scenario_ids, dual_profile_matrix, p, random_state, init_method, max_iter, tol
    )
    
    logger.info(f"DUAL_CLUSTERING_WITH_DUAL completed: selected {len(selected_subproblems)} representatives, "
                f"mapped {len(dual_clustering_mapping)} unselected scenarios")
    
    return selected_subproblems, dual_clustering_mapping


def _build_stacked_dual_profiles(all_trial_duals):
    """
    Build stacked dual profiles for each scenario from Phase 0 and Phase 1 data.
    
    For each scenario, collect all dual vectors across iterations and stack them
    into a single profile with unit-length normalization per iteration block.
    
    Args:
        all_trial_duals (dict): {(iteration, scenario_id): dual_dict} or {(scenario_id, iteration): dual_dict}
    
    Returns:
        dict: {scenario_id: stacked_normalized_dual_vector}
    """
    # Group duals by scenario
    scenario_duals = defaultdict(list)
    
    for key, dual_dict in all_trial_duals.items():
        # Handle both (iteration, scenario_id) and (scenario_id, iteration) key formats
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], int) and key[0] < 0:  # Phase 0: (-iter, sid)
                _, scenario_id = key
                iteration = key[0]
            else:  # Phase 1: (sid, iter)
                scenario_id, iteration = key
        else:
            logger.warning(f"Unexpected key format in all_trial_duals: {key}")
            continue
            
        # Convert structured dual dictionary to a flat numeric vector.
        dual_vector = _dual_dict_to_vector(dual_dict)
        scenario_duals[scenario_id].append((iteration, dual_vector))
    
    # Build stacked profiles
    dual_profiles = {}
    for scenario_id, iteration_duals in scenario_duals.items():
        # Sort by iteration (Phase 0 has negative iterations, Phase 1 positive)
        iteration_duals.sort(key=lambda x: x[0])
        
        # Stack and normalize
        stacked_profile = []
        for iteration, dual_vector in iteration_duals:
            # Normalize to unit length to reduce scale effects
            norm = np.linalg.norm(dual_vector)
            if norm > 1e-12:  # Avoid division by zero
                normalized_vector = dual_vector / norm
            else:
                normalized_vector = dual_vector
            stacked_profile.extend(normalized_vector)
        
        dual_profiles[scenario_id] = np.array(stacked_profile)
    
    logger.debug(f"Built stacked dual profiles: {len(dual_profiles)} scenarios, "
                f"profile length: {len(next(iter(dual_profiles.values()))) if dual_profiles else 0}")

    # Defensive validation: dual profile vectors must contain at least one feature.
    # This guards against unsupported/incorrect dual dictionary formats silently
    # producing empty vectors, which later crashes KMeans with shape=(n, 0).
    if dual_profiles:
        profile_lengths = {sid: int(np.asarray(vec).size) for sid, vec in dual_profiles.items()}
        min_len = min(profile_lengths.values())
        max_len = max(profile_lengths.values())
        if min_len == 0:
            empty_sids = [sid for sid, ln in profile_lengths.items() if ln == 0][:10]
            raise ValueError(
                "DUAL_CLUSTERING_WITH_DUAL produced empty dual profile vectors "
                f"for scenarios {empty_sids}. This indicates an unsupported dual dictionary "
                "format or missing dual coefficients."
            )
        if min_len != max_len:
            raise ValueError(
                "Inconsistent stacked dual profile lengths across scenarios: "
                f"min={min_len}, max={max_len}."
            )
    
    return dual_profiles


def _dual_dict_to_vector(dual_dict):
    """
    Convert structured dual dictionary to flat numpy vector.
    
    For SCFLP: dual_dict = {'mu': {i: float}, 'nu': {j: float}}
    """
    vector_parts = []
    
    # Handle SCFLP format
    if 'mu' in dual_dict and 'nu' in dual_dict:
        # Mu values (sorted by key for consistency)
        mu_vals = dual_dict['mu']
        if isinstance(mu_vals, dict):
            mu_keys = sorted(mu_vals.keys())
            vector_parts.extend([mu_vals[k] for k in mu_keys])
        
        # Nu values (sorted by key for consistency) 
        nu_vals = dual_dict['nu']
        if isinstance(nu_vals, dict):
            nu_keys = sorted(nu_vals.keys())
            vector_parts.extend([nu_vals[k] for k in nu_keys])
    
    # Handle other formats (UFL, UHL, SSLP) if needed
    elif 'lambda' in dual_dict:  # UFL
        vector_parts.append(dual_dict['lambda'])
        mu_vals = dual_dict.get('mu', {})
        if isinstance(mu_vals, dict):
            mu_keys = sorted(mu_vals.keys())
            vector_parts.extend([mu_vals[k] for k in mu_keys])
    
    elif 'v' in dual_dict and 'u' in dual_dict:  # UHL
        v_vals = dual_dict['v']
        if isinstance(v_vals, dict):
            v_keys = sorted(v_vals.keys())
            vector_parts.extend([v_vals[k] for k in v_keys])
        
        u_vals = dual_dict['u']
        if isinstance(u_vals, dict):
            u_keys = sorted(u_vals.keys())
            vector_parts.extend([u_vals[k] for k in u_keys])
    
    elif 'pi' in dual_dict and 'gamma' in dual_dict:  # SSLP
        pi_vals = dual_dict['pi']
        if isinstance(pi_vals, dict):
            pi_keys = sorted(pi_vals.keys())
            vector_parts.extend([pi_vals[k] for k in pi_keys])
        
        gamma_vals = dual_dict['gamma']
        if isinstance(gamma_vals, dict):
            gamma_keys = sorted(gamma_vals.keys())
            vector_parts.extend([gamma_vals[k] for k in gamma_keys])

    elif 'pi' in dual_dict and 'alpha' in dual_dict:  # NE-SP / SUMND / SCMND
        pi_vals = dual_dict['pi']
        if isinstance(pi_vals, dict):
            pi_keys = sorted(pi_vals.keys())
            vector_parts.extend([pi_vals[k] for k in pi_keys])

        alpha_vals = dual_dict['alpha']
        if isinstance(alpha_vals, dict):
            alpha_keys = sorted(alpha_vals.keys())
            vector_parts.extend([alpha_vals[k] for k in alpha_keys])
    
    return np.array(vector_parts, dtype=float)


# ============================
# UNCERTAINTY_CLUSTERING Method Implementation
# ============================

@profiled("representative_selection_clustering")
def solve_uncertainty_clustering_selection(uncertainty_vectors, p, clustering_params=None):
    """
    Uncertainty-based clustering selection for SCFLP problems.
    
    This method clusters scenarios based on their uncertainty vectors (demand vectors,
    optionally combined with capacity vectors when use_stochastic_capacity=True)
    and selects one representative scenario from each cluster. The remaining scenarios
    in each cluster are mapped to their cluster representative for dual prediction.
    
    Key Features:
    - No Phase 0 or Phase 1 required (goes directly to Phase 2)
    - Uses uncertainty vectors for clustering, not feature vectors
    - Supports both demand-only and combined demand-capacity uncertainty vectors
    - Clusters scenarios using K-means on uncertainty patterns
    - Selects cluster centroids as representative scenarios
    - Maps unselected scenarios to their cluster representatives
    
    Args:
        uncertainty_vectors (dict): Dictionary {scenario_id: uncertainty_vector} where
                                  uncertainty_vector is demand or combined demand-capacity for that scenario
        p (int): Number of clusters/selected scenarios
        clustering_params (dict): Clustering parameters including:
            - random_state: Random seed for K-means
            - init_method: K-means initialization method
            - max_iter: Maximum iterations
            - tol: Convergence tolerance
            
    Returns:
        tuple: (selected_subproblems, uncertainty_mapping)
            - selected_subproblems (list): List of selected scenario IDs (cluster representatives)
            - uncertainty_mapping (dict): Maps unselected scenarios to selected representatives
                                         Format: {unselected_scenario_id: selected_representative_id}
    """
    if clustering_params is None:
        clustering_params = {}
        
    scenario_ids = list(uncertainty_vectors.keys())
    num_scenarios = len(scenario_ids)
    
    if p <= 0 or p > num_scenarios:
        raise ValueError(f"Invalid number of clusters p={p}. Must be between 1 and {num_scenarios}")
    
    if p == num_scenarios:
        # All scenarios selected, no clustering needed
        logger.info(f"All {num_scenarios} scenarios selected, no clustering performed")
        return scenario_ids, {}
    
    # Extract parameters
    random_state = clustering_params.get('random_state', 42)
    init_method = clustering_params.get('init_method', 'k-means++')
    max_iter = clustering_params.get('max_iter', 300)
    tol = clustering_params.get('tol', 1e-4)
    
    # logger.info(f"Starting uncertainty clustering with p={p} clusters on {num_scenarios} scenarios")
    
    # Convert uncertainty vectors to matrix
    uncertainty_matrix = np.array([uncertainty_vectors[sid] for sid in scenario_ids])
    
    # Perform K-means clustering
    kmeans = KMeans(
        n_clusters=p,
        init=init_method,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=10
    )
    
    cluster_labels = kmeans.fit_predict(uncertainty_matrix)
    cluster_centers = kmeans.cluster_centers_
    
    # Find representative scenario for each cluster (closest to cluster center)
    selected_scenarios = []
    uncertainty_mapping = {}
    
    for cluster_id in range(p):
        # Find scenarios in this cluster
        cluster_scenarios = [scenario_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        if not cluster_scenarios:
            logger.warning(f"Empty cluster {cluster_id}, skipping")
            continue
            
        # Find the scenario closest to cluster center
        cluster_center = cluster_centers[cluster_id]
        min_distance = float('inf')
        representative = None
        
        for scenario_id in cluster_scenarios:
            scenario_idx = scenario_ids.index(scenario_id)
            distance = np.linalg.norm(uncertainty_matrix[scenario_idx] - cluster_center)
            if distance < min_distance:
                min_distance = distance
                representative = scenario_id
                
        selected_scenarios.append(representative)
        
        # Map all other scenarios in this cluster to the representative
        for scenario_id in cluster_scenarios:
            if scenario_id != representative:
                uncertainty_mapping[scenario_id] = representative
    
    logger.info(f"Uncertainty clustering completed:")
    logger.info(f"  - Selected {len(selected_scenarios)} representative scenarios")
    logger.info(f"  - Mapped {len(uncertainty_mapping)} unselected scenarios to representatives")
    
    return selected_scenarios, uncertainty_mapping


# ============================
# COVARIATE_CLUSTERING Method Implementation
# ============================

@profiled("representative_selection_clustering")
def solve_covariate_clustering_selection(covariate_vectors, p, clustering_params=None):
    """
    Covariate-based clustering selection for SCFLP problems.
    
    This method clusters scenarios based on their covariate vectors (feature vectors from ARMA models)
    and selects one representative scenario from each cluster. The remaining scenarios
    in each cluster are mapped to their cluster representative for dual prediction.
    
    Key Features:
    - No Phase 0 or Phase 1 required (goes directly to Phase 2) 
    - Uses covariate vectors (feature vectors from ARMA data generation) for clustering
    - Clusters scenarios using K-means on covariate patterns
    - Selects cluster centroids as representative scenarios
    - Maps unselected scenarios to their cluster representatives
    
    Args:
        covariate_vectors (dict): Dictionary {scenario_id: covariate_vector} where
                                covariate_vector is the feature vector for that scenario
        p (int): Number of clusters/selected scenarios
        clustering_params (dict): Clustering parameters including:
            - random_state: Random seed for K-means
            - init_method: K-means initialization method
            - max_iter: Maximum iterations
            - tol: Convergence tolerance
            
    Returns:
        tuple: (selected_subproblems, covariate_mapping)
            - selected_subproblems (list): List of selected scenario IDs (cluster representatives)
            - covariate_mapping (dict): Maps unselected scenarios to selected representatives
                                       Format: {unselected_scenario_id: selected_representative_id}
    """
    if clustering_params is None:
        clustering_params = {}
        
    scenario_ids = list(covariate_vectors.keys())
    num_scenarios = len(scenario_ids)
    
    if p <= 0 or p > num_scenarios:
        raise ValueError(f"Invalid number of clusters p={p}. Must be between 1 and {num_scenarios}")
    
    if p == num_scenarios:
        # All scenarios selected, no clustering needed
        logger.info(f"All {num_scenarios} scenarios selected, no clustering performed")
        return scenario_ids, {}
    
    # Extract parameters
    random_state = clustering_params.get('random_state', 42)
    init_method = clustering_params.get('init_method', 'k-means++')
    max_iter = clustering_params.get('max_iter', 300)
    tol = clustering_params.get('tol', 1e-4)
    
    # logger.info(f"Starting covariate clustering with p={p} clusters on {num_scenarios} scenarios")
    
    # Convert covariate vectors to matrix
    covariate_matrix = np.array([covariate_vectors[sid] for sid in scenario_ids])
    
    # Perform K-means clustering
    kmeans = KMeans(
        n_clusters=p,
        init=init_method,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=10
    )
    
    cluster_labels = kmeans.fit_predict(covariate_matrix)
    cluster_centers = kmeans.cluster_centers_
    
    # Find representative scenario for each cluster (closest to cluster center)
    selected_scenarios = []
    covariate_mapping = {}
    
    for cluster_id in range(p):
        # Find scenarios in this cluster
        cluster_scenarios = [scenario_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        if not cluster_scenarios:
            logger.warning(f"Empty cluster {cluster_id}, skipping")
            continue
            
        # Find the scenario closest to cluster center
        cluster_center = cluster_centers[cluster_id]
        min_distance = float('inf')
        representative = None
        
        for scenario_id in cluster_scenarios:
            scenario_idx = scenario_ids.index(scenario_id)
            distance = np.linalg.norm(covariate_matrix[scenario_idx] - cluster_center)
            if distance < min_distance:
                min_distance = distance
                representative = scenario_id
                
        selected_scenarios.append(representative)
        
        # Map all other scenarios in this cluster to the representative
        for scenario_id in cluster_scenarios:
            if scenario_id != representative:
                covariate_mapping[scenario_id] = representative
    
    logger.info(f"Covariate clustering completed:")
    logger.info(f"  - Selected {len(selected_scenarios)} representative scenarios")
    logger.info(f"  - Mapped {len(covariate_mapping)} unselected scenarios to representatives")
    
    return selected_scenarios, covariate_mapping


# ============================
# BERTSIMAS_RANDOM_SAMPLING Method Implementation
# ============================

def solve_bertsimas_random_sampling_selection(feature_vectors, p, sampling_params=None):
    """
    Bertsimas random sampling method implementation.
    
    This method doesn't perform any fixed selection since scenarios are randomly sampled
    at each iteration. It returns an empty list of selected scenarios and empty mapping
    since the actual selection happens dynamically during each callback iteration.
    
    Args:
        feature_vectors (dict): Scenario feature vectors (ignored - no features needed)
        p (int): Number of scenarios to select at each iteration
        sampling_params (dict): Sampling parameters including random_state
        
    Returns:
        tuple: (empty_list, empty_mapping) since selection is done dynamically
    """
    if sampling_params is None:
        sampling_params = {}
    
    random_state = sampling_params.get('random_state', 42)
    
    # Get all scenario IDs
    scenario_ids = list(feature_vectors.keys()) if feature_vectors else []
    
    logger.info(f"BERTSIMAS_RANDOM_SAMPLING: No fixed selection needed, will sample {p} scenarios per iteration from {len(scenario_ids)} total scenarios")
    logger.info(f"Random state: {random_state}")
    
    # Return empty selection and mapping since selection is done dynamically
    # The actual sampling happens in the callback during each iteration
    return [], {
        'method': 'BERTSIMAS_RANDOM_SAMPLING',
        'p': p,
        'total_scenarios': len(scenario_ids),
        'random_state': random_state
    }


def _kmeans_clustering_on_duals(scenario_ids, dual_profile_matrix, p, random_state, init_method, max_iter, tol):
    """
    Perform K-means clustering on dual profiles (same logic as DUAL_CLUSTERING but on dual profiles).
    
    This function follows the exact same pattern as solve_dual_clustering_selection but operates
    on dual profiles instead of feature vectors.
    
    Args:
        scenario_ids (list): List of scenario IDs
        dual_profile_matrix (np.ndarray): Matrix of dual profiles (n_scenarios x dual_profile_length)
        p (int): Number of clusters
        random_state (int): Random seed
        init_method (str): K-means initialization method
        max_iter (int): Maximum iterations
        tol (float): Convergence tolerance
    
    Returns:
        tuple: (selected_subproblems, dual_clustering_mapping)
    """
    # Apply K-means clustering to dual profiles
    kmeans = KMeans(
        n_clusters=p,
        init=init_method,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=10
    )
    
    cluster_labels = kmeans.fit_predict(dual_profile_matrix)
    cluster_centers = kmeans.cluster_centers_
    
    # For each cluster, find the scenario closest to the centroid
    selected_subproblems = []
    cluster_representatives = {}  # cluster_id -> representative_scenario_id
    
    for cluster_id in range(p):
        # Find all scenarios in this cluster
        cluster_members_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        if not cluster_members_indices:
            logger.warning(f"Cluster {cluster_id} is empty, skipping")
            continue
        
        # Calculate distances from cluster members to centroid
        cluster_center = cluster_centers[cluster_id]
        min_distance = float('inf')
        representative_idx = None
        
        for member_idx in cluster_members_indices:
            distance = np.linalg.norm(dual_profile_matrix[member_idx] - cluster_center)
            if distance < min_distance:
                min_distance = distance
                representative_idx = member_idx
        
        representative_scenario_id = scenario_ids[representative_idx]
        selected_subproblems.append(representative_scenario_id)
        cluster_representatives[cluster_id] = representative_scenario_id
        
        logger.debug(f"Cluster {cluster_id}: {len(cluster_members_indices)} scenarios, "
                    f"representative: {representative_scenario_id}")
    
    # Create mapping for unselected scenarios (same as DUAL_CLUSTERING with error handling)
    dual_clustering_mapping = {}
    for i, scenario_id in enumerate(scenario_ids):
        if scenario_id not in selected_subproblems:
            cluster_id = cluster_labels[i]
            if cluster_id in cluster_representatives:
                representative_id = cluster_representatives[cluster_id]
                dual_clustering_mapping[scenario_id] = {
                    'representative': representative_id,
                    'cluster_id': cluster_id
                }
            else:
                logger.warning(f"No representative found for cluster {cluster_id}, mapping scenario {scenario_id} to itself")
                dual_clustering_mapping[scenario_id] = {
                    'representative': scenario_id,
                    'cluster_id': cluster_id
                }
    
    # Log cluster statistics (same as DUAL_CLUSTERING)
    for cluster_id in range(p):
        cluster_size = sum(1 for label in cluster_labels if label == cluster_id)
        if cluster_id in cluster_representatives:
            rep_id = cluster_representatives[cluster_id]
            logger.debug(f"Cluster {cluster_id}: size={cluster_size}, representative={rep_id}")
    
    return selected_subproblems, dual_clustering_mapping