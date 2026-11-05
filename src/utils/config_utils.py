"""
Utils module for ML-augmented Benders decomposition method configuration.

This module provides utility functions for the restructured method architecture with clear separation between:
- Method categories (Exact vs ML)
- ML method types (5 distinct types)
- Predictor types (KNN, TREE, CLUSTERING)
- Callback strategies (when to add cuts)

The 6-Step Execution Pattern:
------------------------------
Step 1 - Phase 0: Random first-stage solution generation (conditional)
Step 2 - Phase 0 cut generation from collected duals (conditional)
Step 3 - Phase 1: Standard Benders iterations (conditional)
Step 4 - Feature generation from duals (conditional)
Step 5 - Partitioning and predictor training (conditional)
Step 6 - Phase 2: ML-augmented iterations (conditional)

Each method has a specific subset of steps it executes based on its type.
"""

import numpy as np


PARAMETRIC_PREDICTOR_TYPES = [
    'PARAMETRIC_MANUSCRIPT',
    'PARAMETRIC_PENALIZED_ERRORS',
    'PARAMETRIC_RIDGE_REPAIR',
    'PARAMETRIC_RHS_RIDGE_REPAIR',
    'PARAMETRIC_RHS_RIDGE_TREE_ANCHOR',
    'PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR',
    'PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR',
]

PMEDIAN_WEIGHTED_TRANSFER = 'PMEDIAN_WEIGHTED_TRANSFER'
PMEDIAN_NEAREST_TRANSFER = 'PMEDIAN_NEAREST_TRANSFER'


# ============================================================================
# Core Method Configuration Helpers
# ============================================================================

def get_method_category(method_config):
    """
    Determine if method is Exact or ML category.
    
    Args:
        method_config (dict): Complete method configuration
        
    Returns:
        str: "Exact" or "ML"
    """
    return method_config.get('method_category', 'ML')


def get_ml_method_type(method_config):
    """
    Get specific ML method type.
    
    Args:
        method_config (dict): Complete method configuration
        
    Returns:
        str: ML method type or None if Exact method
            Options: "ML_Augmented_BD", "ML_Augmented_BD_with_duals",
                    "Covariate_Similarity", "Uncertainty_Similarity", "Random_Sampling"
    """
    if get_method_category(method_config) == 'Exact':
        return None
    return method_config.get('ml_method_type', 'ML_Augmented_BD')


def get_predictor_type(method_config):
    """
    Get predictor type for ML methods.
    
    Args:
        method_config (dict): Complete method configuration
        
    Returns:
        str: Predictor type or None if Exact or Random_Sampling
            Options: "KNN", "TREE", "CLUSTERING", "PMEDIAN_WEIGHTED_TRANSFER",
            or a PARAMETRIC_* variant inside ML_Augmented_BD.
    """
    ml_type = get_ml_method_type(method_config)
    if ml_type is None or ml_type == 'Random_Sampling':
        return None
    return method_config.get('predictor_type', 'KNN')


# ============================================================================
# 6-Step Execution Pattern
# ============================================================================

def should_execute_step(method_config, step_number):
    """
    Determine if a given step (1-6) should be executed for this method.
    
    Step definitions:
    1 - Phase 0: Random first-stage solution generation
    2 - Phase 0 cut generation
    3 - Phase 1: Standard Benders iterations
    4 - Feature generation from duals
    5 - Partitioning and predictor training
    6 - Phase 2: ML-augmented iterations
    
    Args:
        method_config (dict): Complete method configuration
        step_number (int): Step number (1-6)
        
    Returns:
        bool: True if step should be executed
    """
    method_category = get_method_category(method_config)
    
    if method_category == 'Exact':
        # Exact only has step 3 (continuous until convergence)
        return step_number == 3
    
    # ML methods
    ml_type = get_ml_method_type(method_config)
    m_trials = method_config.get('m_trials', 0)
    n_trials = method_config.get('n_trials', 0)
    
    step_map = {
        'ML_Augmented_BD': {
            1: m_trials > 0,  # Phase 0 if configured
            2: m_trials > 0,  # Phase 0 cuts if Phase 0 executed
            3: n_trials > 0,  # Phase 1
            4: True,          # Feature generation
            5: True,          # Partitioning/training
            6: True           # Phase 2
        },
        'ML_Augmented_BD_with_duals': {
            1: m_trials > 0,
            2: m_trials > 0,
            3: n_trials > 0,
            4: False,         # Skip feature generation (use duals directly)
            5: True,
            6: True
        },
        'Covariate_Similarity': {
            1: False,         # Skip Phase 0
            2: False,         # Skip Phase 0 cuts
            3: False,         # Skip Phase 1
            4: False,         # Skip feature generation (use covariates)
            5: True,          # Partitioning/training with covariates
            6: True           # Phase 2
        },
        'Uncertainty_Similarity': {
            1: False,
            2: False,
            3: False,
            4: False,         # Skip feature generation (use uncertainty vectors)
            5: True,          # Partitioning/training with uncertainty
            6: True
        },
        'Random_Sampling': {
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,         # Skip partitioning/training (random sampling each iteration)
            6: True           # Phase 2 with dynamic random sampling
        }
    }
    
    return step_map.get(ml_type, {}).get(step_number, False)


# ============================================================================
# Callback Registration Strategy
# ============================================================================

def should_register_lp_callbacks(method_config):
    """
    Determine if LP callbacks should be registered.
    
    Args:
        method_config (dict or str): Complete method configuration dict OR legacy use_prediction string
        
    Returns:
        bool: True if LP callbacks should be registered
    """
    # Handle legacy string format (use_prediction values like "int_and_LP", "int_only_ML_for_integer", etc.)
    if isinstance(method_config, str):
        # For legacy format, check if string contains "LP" or "and_LP"
        return 'and_LP' in method_config or '_LP' in method_config
    
    # Handle new method_config dict format
    method_category = get_method_category(method_config)
    
    if method_category == 'Exact':
        callback_type = method_config.get('exact_callback_type', 'int_only')
        return callback_type == 'int_and_LP'
    
    # ML methods - check if any LP callback types present
    ml_callback_type = method_config.get('ml_callback_type', 'ML_at_integer_only')
    # Register LP callbacks for any callback type that includes LP operations
    # All except ML_at_integer_only include LP callbacks
    return ml_callback_type in ['ML_at_integer_exact_at_LP', 'ML_at_LP_exact_at_integer', 'ML_at_both']


def should_use_ml_for_unselected(method_config, at_integer_solution):
    """
    Determine if ML should be used for unselected scenarios at current solution type.
    
    Only applies to ML methods in Phase 2. Always returns True for selected scenarios (exact cuts).
    
    Args:
        method_config (dict): Complete method configuration
        at_integer_solution (bool): True if at integer solution, False if at LP relaxation
        
    Returns:
        bool: True if ML cuts should be used for unselected scenarios
    """
    method_category = get_method_category(method_config)
    if method_category == 'Exact':
        return False  # Never use ML for Exact methods
    
    ml_callback_type = method_config.get('ml_callback_type', 'ML_at_integer_only')
    
    if at_integer_solution:
        # At integer solutions: use ML if method includes "integer" or "both"
        return ml_callback_type in ['ML_at_integer_only', 'ML_at_integer_exact_at_LP', 'ML_at_both']
    else:  # LP relaxation
        # At LP relaxations: use ML if method includes "LP" (but not "exact_at_LP") or "both"
        return ml_callback_type in ['ML_at_LP_exact_at_integer', 'ML_at_both']


# ============================================================================
# Input Data Selection for Step 5 (Partitioning/Training)
# ============================================================================

def get_input_data_for_step5(method_config, dual_vectors, feature_vectors, covariate_vectors, uncertainty_vectors):
    """
    Get appropriate input data for step 5 (partitioning/training) based on method type.
    
    Args:
        method_config (dict): Complete method configuration
        dual_vectors (dict): Dual vectors from Phase 0 and Phase 1
        feature_vectors (dict): Generated features from duals
        covariate_vectors (dict): Covariate vectors from problem data
        uncertainty_vectors (dict): Uncertainty vectors from problem data
        
    Returns:
        dict or None: Appropriate data structure based on ML method type
    """
    ml_type = get_ml_method_type(method_config)
    
    if ml_type == 'ML_Augmented_BD':
        return feature_vectors
    elif ml_type == 'ML_Augmented_BD_with_duals':
        return dual_vectors
    elif ml_type == 'Covariate_Similarity':
        return covariate_vectors
    elif ml_type == 'Uncertainty_Similarity':
        return uncertainty_vectors
    elif ml_type == 'Random_Sampling':
        return None  # Doesn't need input data
    
    return None


# ============================================================================
# Uncertainty Vector Extraction
# ============================================================================

def get_uncertainty_vectors_for_problem(problem_type, data, use_stochastic_capacity=False):
    """
    Extract uncertainty vectors from problem data for Uncertainty_Similarity method.
    
    Args:
        problem_type (str): Problem type (e.g., "SCFLP")
        data: Problem data object
        use_stochastic_capacity (bool): Whether to include stochastic capacities
        
    Returns:
        dict: Uncertainty vectors indexed by scenario ID
    """
    uncertainty_vectors = {}
    
    if str(problem_type).upper() == 'SCFLP':
        for sid in range(len(data.S)):
            if use_stochastic_capacity and hasattr(data, 'capacity_scenarios') and data.capacity_scenarios is not None:
                # Include both demands and capacities as uncertainty
                demand_vector = np.array(data.scenarios[sid])
                capacity_vector = np.array(data.capacity_scenarios[sid])
                uncertainty_vectors[sid] = np.concatenate([demand_vector, capacity_vector])
            else:
                # Only demands are uncertain
                uncertainty_vectors[sid] = np.array(data.scenarios[sid])
    
    # Add support for other problem types here as they are implemented
    
    return uncertainty_vectors


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_method_config(method_config):
    """
    Validate that method configuration is consistent and complete.
    
    Args:
        method_config (dict): Complete method configuration
        
    Returns:
        tuple: (is_valid, error_message)
            is_valid (bool): True if configuration is valid
            error_message (str): Explanation if invalid, empty string if valid
    """
    method_category = get_method_category(method_config)

    scmnd_formulation = str(
        method_config.get('scmnd_formulation', 'CURRENT_SOFT_CAPACITY')
    ).upper()
    if scmnd_formulation not in {
        'CURRENT_SOFT_CAPACITY',
        'MANUSCRIPT_ENVELOPE',
    }:
        return False, (
            "Invalid scmnd_formulation. Must be CURRENT_SOFT_CAPACITY or "
            "MANUSCRIPT_ENVELOPE"
        )
    
    # Validate method category
    if method_category not in ['Exact', 'ML']:
        return False, f"Invalid method_category: {method_category}. Must be 'Exact' or 'ML'"
    
    if method_category == 'Exact':
        # Validate Exact method configuration
        callback_type = method_config.get('exact_callback_type')
        if callback_type not in ['int_only', 'int_and_LP']:
            return False, f"Invalid exact_callback_type: {callback_type}. Must be 'int_only' or 'int_and_LP'"
    
    else:  # ML category
        # Validate ML method type
        ml_type = get_ml_method_type(method_config)
        valid_ml_types = ['ML_Augmented_BD', 'ML_Augmented_BD_with_duals', 
                         'Covariate_Similarity', 'Uncertainty_Similarity', 'Random_Sampling']
        if ml_type not in valid_ml_types:
            return False, f"Invalid ml_method_type: {ml_type}. Must be one of {valid_ml_types}"
        
        # Validate ML callback type
        ml_callback_type = method_config.get('ml_callback_type')
        valid_ml_callbacks = ['ML_at_integer_only', 'ML_at_integer_exact_at_LP', 
                             'ML_at_LP_exact_at_integer', 'ML_at_both']
        if ml_callback_type not in valid_ml_callbacks:
            return False, f"Invalid ml_callback_type: {ml_callback_type}. Must be one of {valid_ml_callbacks}"
        
        # Validate predictor type (except for Random_Sampling)
        if ml_type != 'Random_Sampling':
            predictor_type = get_predictor_type(method_config)
            valid_predictors = ['KNN', 'TREE', 'CLUSTERING', PMEDIAN_WEIGHTED_TRANSFER, PMEDIAN_NEAREST_TRANSFER] + PARAMETRIC_PREDICTOR_TYPES
            if predictor_type not in valid_predictors:
                return False, f"Invalid predictor_type: {predictor_type}. Must be one of {valid_predictors}"
            if predictor_type in PARAMETRIC_PREDICTOR_TYPES and ml_type != 'ML_Augmented_BD':
                return False, (
                    f"{predictor_type} is a predictor variant inside ML_Augmented_BD; "
                    f"it cannot be combined with {ml_type}."
                )
            if predictor_type in PARAMETRIC_PREDICTOR_TYPES:
                repair_mode = str(
                    method_config.get('parametric_repair_mode', 'LOCAL')
                ).upper()
                if repair_mode not in {'LOCAL', 'GLOBAL', 'STRUCTURE_AWARE'}:
                    return False, (
                        "parametric_repair_mode must be LOCAL, GLOBAL, or STRUCTURE_AWARE"
                    )

            # The p-median solver setting is required by the original RHS-Ridge
            # anchor and by its nonparametric control. Tree/cluster anchor variants
            # do not use this setting for representative selection.
            if predictor_type in {
                'PARAMETRIC_MANUSCRIPT',
                'PARAMETRIC_PENALIZED_ERRORS',
                'PARAMETRIC_RIDGE_REPAIR',
                'PARAMETRIC_RHS_RIDGE_REPAIR',
                'PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR',
                PMEDIAN_WEIGHTED_TRANSFER,
                PMEDIAN_NEAREST_TRANSFER,
            }:
                pmedian_solver = str(
                    method_config.get('parametric_pmedian_solver', 'MIP')
                ).upper()
                if pmedian_solver not in {'MIP', 'GREEDY', 'PAM'}:
                    return False, (
                        "parametric_pmedian_solver must be MIP, GREEDY, or PAM"
                    )
        
        # Validate Phase 1 requirement for ML_Augmented_BD methods
        if ml_type in ['ML_Augmented_BD', 'ML_Augmented_BD_with_duals']:
            n_trials = method_config.get('n_trials', 0)
            if n_trials == 0:
                return False, f"{ml_type} requires n_trials > 0 for training data collection"
    
    return True, ""


# ============================================================================
# Method Description Helpers
# ============================================================================

def get_method_description(method_config):
    """
    Get a human-readable description of the method configuration.
    
    Args:
        method_config (dict): Complete method configuration
        
    Returns:
        str: Method description
    """
    method_category = get_method_category(method_config)
    
    if method_category == 'Exact':
        callback_type = method_config.get('exact_callback_type', 'int_only')
        return f"Exact Benders ({callback_type})"
    
    # ML methods
    ml_type = get_ml_method_type(method_config)
    predictor_type = get_predictor_type(method_config)
    ml_callback_type = method_config.get('ml_callback_type', 'ML_at_integer_only')
    
    if ml_type == 'Random_Sampling':
        return f"Random_Sampling ({ml_callback_type})"
    
    return f"{ml_type} + {predictor_type} ({ml_callback_type})"


def get_steps_description(method_config):
    """
    Get a description of which steps (1-6) are executed for this method.
    
    Args:
        method_config (dict): Complete method configuration
        
    Returns:
        str: Steps description (e.g., "Steps: 1,2,3,4,5,6" or "Steps: 5,6")
    """
    executed_steps = [i for i in range(1, 7) if should_execute_step(method_config, i)]
    steps_str = ",".join(map(str, executed_steps))
    return f"Steps: {steps_str}"


# ============================================================================
# Backward Compatibility (Optional - for transitioning old code)
# ============================================================================

def legacy_use_prediction_to_new_config(use_prediction_str, prediction_method_str):
    """
    Convert legacy use_prediction and prediction_method to new method_config structure.
    
    This function helps transition from the old parameter system to the new one.
    
    Args:
        use_prediction_str (str): Old use_prediction value
        prediction_method_str (str): Old prediction_method value
        
    Returns:
        dict: Partial method_config dictionary (needs to be completed with other parameters)
    """
    # Map old use_prediction to new callback types
    callback_mapping = {
        'int_exact': ('Exact', 'int_only', None),
        'int_and_LP_ML_for_none': ('Exact', 'int_and_LP', None),
        'int_ML': ('ML', 'ML_at_integer_only', None),
        'int_and_LP_ML_for_int': ('ML', 'ML_at_integer_exact_at_LP', None),
        'int_and_LP_ML_for_LP': ('ML', 'ML_at_LP_exact_at_integer', None),
        'int_and_LP_ML_for_relax': ('ML', 'ML_at_LP_exact_at_integer', None),  # Legacy alias
        'int_and_LP_ML_for_both': ('ML', 'ML_at_both', None),
    }
    
    # Map old prediction_method to new ml_method_type and predictor_type
    method_mapping = {
        'KNN': ('ML_Augmented_BD', 'KNN'),
        'TREE': ('ML_Augmented_BD', 'TREE'),
        'DUAL_CLUSTERING': ('ML_Augmented_BD_with_duals', 'CLUSTERING'),  # Fixed: uses duals directly 
        'DUAL_CLUSTERING_WITH_DUAL': ('ML_Augmented_BD_with_duals', 'CLUSTERING'),
        'COVARIATE_CLUSTERING': ('Covariate_Similarity', 'CLUSTERING'),
        'UNCERTAINTY_CLUSTERING': ('Uncertainty_Similarity', 'CLUSTERING'),
        'BERTSIMAS_RANDOM_SAMPLING': ('Random_Sampling', None),
    }
    
    method_category, callback_type, _ = callback_mapping.get(use_prediction_str, ('ML', 'ML_at_integer_only', None))
    
    config = {
        'method_category': method_category
    }
    
    if method_category == 'Exact':
        config['exact_callback_type'] = callback_type
    else:
        ml_method_type, predictor_type = method_mapping.get(prediction_method_str.upper(), ('ML_Augmented_BD', 'KNN'))
        config['ml_method_type'] = ml_method_type
        config['ml_callback_type'] = callback_type or 'ML_at_integer_only'
        if predictor_type is not None:
            config['predictor_type'] = predictor_type
    
    return config


def normalize_use_prediction(use_prediction):
    """
    Normalize use_prediction to string format.
    
    Supports both old boolean format and new string format for backward compatibility.
    
    Args:
        use_prediction: Boolean, string, or list representation of prediction strategy
        
    Returns:
        str: Normalized string representation
    """
    if isinstance(use_prediction, bool):
        return 'int_and_LP_ML_for_both' if use_prediction else 'int_and_LP_ML_for_none'
    elif isinstance(use_prediction, str):
        return use_prediction
    elif isinstance(use_prediction, (list, tuple)) and len(use_prediction) > 0:
        return use_prediction[0]
    else:
        return 'int_and_LP_ML_for_both'  # Default


def convert_legacy_use_prediction(use_prediction):
    """
    Convert old boolean use_prediction format to new string grid format.
    
    Args:
        use_prediction: Boolean or list of booleans/strings
        
    Returns:
        list: List of normalized string values
    """
    if isinstance(use_prediction, bool):
        return ['int_and_LP_ML_for_both'] if use_prediction else ['int_and_LP_ML_for_none']
    elif isinstance(use_prediction, str):
        return [use_prediction]
    elif isinstance(use_prediction, (list, tuple)):
        result = []
        for val in use_prediction:
            if isinstance(val, bool):
                result.append('int_and_LP_ML_for_both' if val else 'int_and_LP_ML_for_none')
            else:
                result.append(str(val))
        return result
    else:
        return ['int_and_LP_ML_for_both']  # Default


def uses_ml_in_phase2(use_prediction_str):
    """
    Determine if a use_prediction configuration uses ML in Phase 2.
    
    Args:
        use_prediction_str (str): String representation of use_prediction
        
    Returns:
        bool: True if ML is used in Phase 2
    """
    ml_modes = [
        'int_and_LP_ML_for_both',
        'int_and_LP_ML_for_int', 
        'int_and_LP_ML_for_LP',
        'int_and_LP_ML_for_selected',
        'int_ML'
    ]
    return use_prediction_str in ml_modes


def should_skip_config(params):
    """
    Validate configuration compatibility. Legacy function for backward compatibility.
    
    Args:
        params (dict): Parameter dictionary
        
    Returns:
        tuple: (should_skip: bool, reason: str)
    """
    # Currently no validation rules, but kept for backward compatibility
    return (False, "")


def ml_enabled_for_context(method_config):
    """
    Check if ML is enabled for the given method configuration.
    
    This is a convenience function that checks if the method category is ML.
    Used for backward compatibility with legacy code that checks ML enablement.
    
    Args:
        method_config (dict or bool): Method configuration dictionary, or boolean for legacy support
        
    Returns:
        bool: True if ML is enabled, False otherwise
    """
    # Handle legacy boolean input
    if isinstance(method_config, bool):
        return method_config
    
    # Handle new method_config dictionary
    if isinstance(method_config, dict):
        return get_method_category(method_config) == 'ML'
    
    return False


def should_execute_phase0(approach_category):
    """
    DEPRECATED: Use should_execute_step(method_config, 1) instead.
    
    Determine if Phase 0 (random solution generation) should execute.
    
    This function is maintained for backward compatibility but is less accurate than
    should_execute_step() because it doesn't properly handle methods that skip Phase 0.
    
    Args:
        approach_category (str): Approach category (e.g., 'Exact', 'ML_Augmented_BD', etc.)
        
    Returns:
        bool: True if Phase 0 might execute (requires m_trials > 0 check separately)
    """
    # Only ML_Augmented_BD and ML_Augmented_BD_with_duals execute Phase 0
    # Other methods (Covariate_Similarity, Uncertainty_Similarity, Random_Sampling) skip it
    phase0_categories = [
        'ML_Augmented_BD',
        'ML_Augmented_BD_with_duals'
    ]
    return approach_category in phase0_categories


def should_execute_phase1(approach_category):
    """
    Determine if Phase 1 (standard Benders iterations) should execute.
    
    Phase 1 is applicable to:
    - All Exact methods
    - ML_Augmented_BD (needs trial iterations for dual collection)
    - ML_Augmented_BD_with_duals (needs trial iterations for dual collection)
    
    Phase 1 is NOT applicable to:
    - Covariate_Similarity (uses input covariates, no trials needed)
    - Uncertainty_Similarity (uses uncertainty vectors, no trials needed)
    - Random_Sampling (no training, random sampling only)
    
    Args:
        approach_category (str): Approach category (e.g., 'Exact', 'ML_Augmented_BD', etc.)
        
    Returns:
        bool: True if Phase 1 should execute, False otherwise
    """
    categories_with_phase1 = [
        'Exact',
        'ML_Augmented_BD',
        'ML_Augmented_BD_with_duals'
    ]
    return approach_category in categories_with_phase1


def should_generate_features_from_duals(approach_category):
    """
    Determine if features should be generated from collected duals.
    
    Feature generation from duals (using graph-based methods like Word2Vec) is only
    applicable to ML_Augmented_BD method. Other methods either:
    - Don't use features at all (Exact, Random_Sampling)
    - Use duals directly (ML_Augmented_BD_with_duals)
    - Use input vectors (Covariate_Similarity, Uncertainty_Similarity)
    
    Args:
        approach_category (str): Approach category (e.g., 'Exact', 'ML_Augmented_BD', etc.)
        
    Returns:
        bool: True if features should be generated from duals, False otherwise
    """
    return approach_category == 'ML_Augmented_BD'


def uses_direct_dual_vectors(approach_category):
    """
    Determine if method uses dual vectors directly without feature generation.
    
    Args:
        approach_category (str): Approach category
        
    Returns:
        bool: True if method uses duals directly (ML_Augmented_BD_with_duals)
    """
    return approach_category == 'ML_Augmented_BD_with_duals'


def uses_input_covariates(approach_category):
    """
    Determine if method uses input covariates from problem data.
    
    Args:
        approach_category (str): Approach category
        
    Returns:
        bool: True if method uses input covariates (Covariate_Similarity)
    """
    return approach_category == 'Covariate_Similarity'


def uses_uncertainty_vectors(approach_category):
    """
    Determine if method uses uncertainty vectors (demands/capacities) from problem data.
    
    Args:
        approach_category (str): Approach category
        
    Returns:
        bool: True if method uses uncertainty vectors (Uncertainty_Similarity)
    """
    return approach_category == 'Uncertainty_Similarity'


def uses_random_sampling(approach_category):
    """
    Determine if method uses random sampling without training.
    
    Args:
        approach_category (str): Approach category
        
    Returns:
        bool: True if method uses random sampling (Random_Sampling)
    """
    return approach_category == 'Random_Sampling'


def get_pred_or_exact(params):
    """
    Get prediction or exact classification for logging. Legacy compatibility function.
    
    Args:
        params (dict): Parameter dictionary with either old or new format
        
    Returns:
        str: "prediction" if ML is used, "exact" otherwise
    """
    if 'method_config' in params:
        method_category = get_method_category(params['method_config'])
        return "prediction" if method_category == 'ML' else "exact"
    elif 'use_prediction' in params:
        return "prediction" if params['use_prediction'] else "exact"
    else:
        return "exact"


def get_approach_category(params, prediction_method=None):
    """
    Get approach category for logging. Legacy compatibility function.
    
    Args:
        params (dict or bool or str): Parameter dictionary OR legacy use_prediction boolean/string
        prediction_method (str, optional): Legacy prediction method string (only if params is bool/str)
        
    Returns:
        str: Approach category description
    """
    # Handle legacy two-argument call: get_approach_category(use_prediction, prediction_method)
    if isinstance(params, (bool, str)) and prediction_method is not None:
        use_prediction = params
        
        # If not using prediction, it's Exact Benders
        if not use_prediction or use_prediction in ['int_exact', 'int_and_LP_ML_for_none']:
            return "Exact"
        
        # Map legacy prediction_method to new approach categories
        prediction_method_upper = prediction_method.upper() if prediction_method else ""
        
        # ML_Augmented_BD: Uses generated features from duals
        if prediction_method_upper in ['KNN', 'TREE']:
            return "ML_Augmented_BD"
        
        # ML_Augmented_BD_with_duals: Uses duals directly
        elif prediction_method_upper in ['DUAL_CLUSTERING', 'DUAL_CLUSTERING_WITH_DUAL']:
            return "ML_Augmented_BD_with_duals"
        
        # Covariate_Similarity: Uses input covariates
        elif prediction_method_upper == 'COVARIATE_CLUSTERING':
            return "Covariate_Similarity"
        
        # Uncertainty_Similarity: Uses uncertainty vectors
        elif prediction_method_upper == 'UNCERTAINTY_CLUSTERING':
            return "Uncertainty_Similarity"
        
        # Random_Sampling: No training, random sampling
        elif prediction_method_upper == 'BERTSIMAS_RANDOM_SAMPLING':
            return "Random_Sampling"
        
        # Default to ML_Augmented_BD for unknown prediction methods
        else:
            return "ML_Augmented_BD"
    
    # Handle new single-argument call: get_approach_category(params_dict)
    if 'method_config' in params:
        method_config = params['method_config']
        method_category = get_method_category(method_config)
        if method_category == 'ML':
            return method_config.get('ml_method_type', 'ML_Augmented_BD')
        else:
            return "Exact"
    elif 'prediction_method' in params and params.get('use_prediction'):
        return params['prediction_method']
    else:
        return "Exact"


def get_feature_generation_method(params):
    """
    Get feature generation method for logging. Legacy compatibility function.
    
    Args:
        params (dict): Parameter dictionary
        
    Returns:
        str: Feature generation method description
    """
    if 'method_config' in params:
        # For new format, return based on method type
        method_config = params['method_config']
        if get_method_category(method_config) == 'ML':
            ml_method_type = method_config.get('ml_method_type', '')
            if ml_method_type == 'Covariate_Similarity':
                return "covariates"
            elif ml_method_type == 'Uncertainty_Similarity':
                return "uncertainty"
            elif ml_method_type == 'Random_Sampling':
                return "none"
            else:
                # ML_Augmented_BD or ML_Augmented_BD_with_duals
                return "graph_walk"
        else:
            return "none"
    else:
        # Old format - check use_prediction
        if params.get('use_prediction'):
            return "graph_walk"
        else:
            return "none"
