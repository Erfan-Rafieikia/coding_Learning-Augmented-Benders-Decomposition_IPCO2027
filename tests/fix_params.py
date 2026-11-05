#!/usr/bin/env python3
"""
Script to fix all parameter access issues in main.py by replacing 
params['param'] with params.get('param', default_value) for removed parameters.
"""

def fix_parameter_access():
    # Read the current main.py file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'r') as f:
        content = f.read()
    
    # Define replacements for removed parameters
    replacements = [
        # REG parameters
        ("params['L_d']", "params.get('L_d', None)  # REG parameter removed - KNN only"),
        ("params['L_v']", "params.get('L_v', None)  # REG parameter removed - KNN only"),
        ("params['p_d']", "params.get('p_d', None)  # REG parameter removed - KNN only"),
        ("params['p_v']", "params.get('p_v', None)  # REG parameter removed - KNN only"),
        
        # TREE parameters
        ("params['max_depth']", "params.get('max_depth', 10)  # TREE parameter removed - KNN only"),
        ("params['min_samples_leaf']", "params.get('min_samples_leaf', 5)  # TREE parameter removed - KNN only"),
        
        # Subproblem selection method (now integrated)
        ("params['subproblem_selection_method']", "params.get('subproblem_selection_method', 'integrated')  # Integrated in KNN"),
    ]
    
    # Apply all replacements
    original_content = content
    for old, new in replacements:
        content = content.replace(old, new)
        count = original_content.count(old)
        if count > 0:
            print(f"Replaced {count} occurrences of {old}")
    
    # Write the updated file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'w') as f:
        f.write(content)
    
    print("Successfully fixed all parameter access issues in main.py")
    return True

if __name__ == "__main__":
    fix_parameter_access()