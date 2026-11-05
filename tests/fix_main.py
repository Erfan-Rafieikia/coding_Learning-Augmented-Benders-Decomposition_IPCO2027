#!/usr/bin/env python3
"""
Script to fix the create_parameter_grid function in main.py by replacing it with 
a clean KNN-only version that doesn't reference removed REG/TREE parameters.
"""

def fix_main_py():
    # Read the current main.py file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'r') as f:
        content = f.read()
    
    # Find the start and end of the create_parameter_grid function
    start_marker = "def create_parameter_grid():"
    end_marker = "def save_experiment_result_incrementally("
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("Could not find function boundaries")
        return False
    
    # Extract the parts before and after the function
    before_function = content[:start_idx]
    after_function = content[end_idx:]
    
    # Define the new clean function
    new_function = '''def create_parameter_grid():
    """
    Generate parameter combinations for SCFLP + KNN evaluation.
    
    This function creates parameter combinations specifically optimized for the current
    SCFLP-focused KNN implementation. Since we now only support KNN prediction method
    with integrated subproblem selection, the grid is streamlined for the supported
    parameter space.
    
    Architecture:
    -------------
    The parameter grid generation follows a simplified structure:
    
    1. **Problem Class & Instance Selection**: Iterate through configured SCFLP 
       problem instances
    
    2. **Prediction Strategy**: Generate configurations for both traditional Benders 
       (use_prediction=False) and KNN-augmented Benders (use_prediction=True)
    
    3. **KNN-Specific Parameters**: Include only relevant hyperparameters:
       - n_neighbors: Critical for KNN performance
       - feature_dim: Word2Vec embedding dimension
       - selected_subproblems_ratio: Fraction of scenarios to select
    
    4. **Universal Parameters**: Include problem-agnostic settings like switching 
       strategies, cut generation modes, and trial configurations
    
    Returns:
    --------
    list of dict : Parameter combination grid
        Each dictionary represents a complete parameter configuration for one 
        experimental run, containing all necessary parameters for algorithm execution.
        
    Example Output Structure:
    -------------------------
    ```python
    [
        {
            'problem_class': 'SCFLP',
            'instance_list': ['scflp_10_10_10.txt', 'scflp_20_20_20.txt'],
            'use_prediction': True,
            'prediction_method': 'KNN',
            'n_neighbors': 5,
            'switching_strategy': 'no_switch',
            'add_cuts_at_relaxation': True,
            'n_trials': 20,
            'm_trials': 10,
            'selected_subproblems_ratio': 0.15,
            'feature_dim': 3,
            'fallback_method': 'solve'
        },
        # Additional parameter combinations...
    ]
    ```
    """
    param_combinations = []
    
    for pc_items in problem_classes_to_instances.items():
        problem_class, instance_list = pc_items
        
        for use_pred in use_prediction:
            if not use_pred:
                # Traditional Benders: Only basic parameters matter
                for add_cuts in add_cuts_at_relaxation:
                    for n_trial in n_trials_grid:
                        for m_trial in m_trials_grid:
                            param_combinations.append({
                                'problem_class': problem_class,
                                'instance_list': instance_list,
                                'use_prediction': use_pred,
                                'prediction_method': "KNN",  # Use KNN as default (only method available)
                                'n_neighbors': n_neighbors[0],  # Use first as default
                                'switching_strategy': "no_switch",  # Must be no_switch for traditional
                                'add_cuts_at_relaxation': add_cuts,
                                'n_trials': n_trial,
                                'm_trials': m_trial,
                                'selected_subproblems_ratio': selected_subproblems_ratio_grid[0],  # Irrelevant for traditional
                                'feature_dim': feature_dim_grid[0],  # Irrelevant for traditional
                                'fallback_method': fallback_method[0],  # Use default fallback
                            })
            else:
                # KNN-augmented Benders: Full parameter exploration for KNN method
                for pred_method in prediction_method:  # Should only be ["KNN"]
                    for n_trial in n_trials_grid:
                        for m_trial in m_trials_grid:
                            for ratio in selected_subproblems_ratio_grid:
                                for feat_dim in feature_dim_grid:
                                    for add_cuts in add_cuts_at_relaxation:
                                        for switch_strat in switching_strategy:
                                            for fallback in fallback_method:
                                                # KNN: n_neighbors is the critical parameter to explore
                                                for n_neigh in n_neighbors_grid:
                                                    param_combinations.append({
                                                        'problem_class': problem_class,
                                                        'instance_list': instance_list,
                                                        'use_prediction': use_pred,
                                                        'prediction_method': pred_method,
                                                        'n_neighbors': n_neigh,
                                                        'switching_strategy': switch_strat,
                                                        'add_cuts_at_relaxation': add_cuts,
                                                        'n_trials': n_trial,
                                                        'm_trials': m_trial,
                                                        'selected_subproblems_ratio': ratio,
                                                        'feature_dim': feat_dim,
                                                        'fallback_method': fallback,
                                                    })
    
    return param_combinations


'''
    
    # Combine the parts
    new_content = before_function + new_function + after_function
    
    # Write the new file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'w') as f:
        f.write(new_content)
    
    print("Successfully replaced create_parameter_grid function with clean KNN-only version")
    return True

if __name__ == "__main__":
    fix_main_py()