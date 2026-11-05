#!/usr/bin/env python3
"""
Simple script to fix only the KeyError issue by replacing params['subproblem_selection_method'] 
with params.get() calls.
"""

import re

def fix_keyerror_issue():
    # Read the current main.py file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'r') as f:
        content = f.read()
    
    # Simple replacement to fix KeyError without breaking syntax
    # Replace all params['param_name'] with params.get('param_name', default) for removed parameters
    content = content.replace(
        "params['subproblem_selection_method']", 
        "params.get('subproblem_selection_method', 'integrated')"
    )
    
    # Write the updated file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'w') as f:
        f.write(content)
    
    print("Fixed KeyError for subproblem_selection_method")
    return True

if __name__ == "__main__":
    fix_keyerror_issue()