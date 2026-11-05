#!/usr/bin/env python3
"""
Script to carefully fix parameter access issues in main.py while preserving syntax.
"""

def fix_parameter_access_carefully():
    # Read the current main.py file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'r') as f:
        content = f.read()
    
    # Define replacements that preserve syntax
    replacements = [
        # Logging entries (these are in dictionary values, so no comma issues)
        ("\"dual_difference_pred_loss\": params['L_d']", "\"dual_difference_pred_loss\": params.get('L_d', None)  # REG parameter removed"),
        ("\"obj_difference_pred_loss\": params['L_v']", "\"obj_difference_pred_loss\": params.get('L_v', None)  # REG parameter removed"),
        ("\"subproblem_selection_method\": params['subproblem_selection_method']", "\"subproblem_selection_method\": params.get('subproblem_selection_method', 'integrated')  # Integrated in KNN"),
        
        # Feature dataframe assignments (these have no comma issues)
        ("feature_df[\"L_d\"] = params['L_d']", "feature_df[\"L_d\"] = params.get('L_d', None)  # REG parameter removed"),
        ("feature_df[\"L_v\"] = params['L_v']", "feature_df[\"L_v\"] = params.get('L_v', None)  # REG parameter removed"),
        ("feature_df[\"subproblem_selection_method\"] = params['subproblem_selection_method']", "feature_df[\"subproblem_selection_method\"] = params.get('subproblem_selection_method', 'integrated')  # Integrated in KNN"),
        
        # Function call parameters (these need careful handling of commas)
        ("L_d=params['L_d'],", "L_d=params.get('L_d', None),  # REG parameter removed"),
        ("L_v=params['L_v'],", "L_v=params.get('L_v', None),  # REG parameter removed"),
        ("p_d=params['p_d'],", "p_d=params.get('p_d', None),  # REG parameter removed"),
        ("p_v=params['p_v'],", "p_v=params.get('p_v', None),  # REG parameter removed"),
        ("subproblem_selection_method=params['subproblem_selection_method'],", "subproblem_selection_method=params.get('subproblem_selection_method', 'integrated'),  # Integrated in KNN"),
        ("max_depth=params['max_depth'],", "max_depth=params.get('max_depth', 10),  # TREE parameter removed"),
        ("min_samples_leaf=params['min_samples_leaf'],", "min_samples_leaf=params.get('min_samples_leaf', 5),  # TREE parameter removed"),
    ]
    
    # Apply all replacements
    original_content = content
    for old, new in replacements:
        content = content.replace(old, new)
        count = original_content.count(old)
        if count > 0:
            print(f"Replaced {count} occurrences of: {old}")
    
    # Write the updated file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'w') as f:
        f.write(content)
    
    print("Successfully fixed all parameter access issues while preserving syntax")
    return True

if __name__ == "__main__":
    fix_parameter_access_carefully()