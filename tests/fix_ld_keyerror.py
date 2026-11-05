#!/usr/bin/env python3
"""
Fix the L_d KeyError by making a simple targeted replacement.
"""

def fix_ld_keyerror():
    # Read the current main.py file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'r') as f:
        content = f.read()
    
    # Replace the specific line that's causing the error
    content = content.replace(
        '"dual_difference_pred_loss": params[\'L_d\']',
        '"dual_difference_pred_loss": params.get(\'L_d\', None)'
    )
    content = content.replace(
        '"obj_difference_pred_loss": params[\'L_v\']',
        '"obj_difference_pred_loss": params.get(\'L_v\', None)'
    )
    
    # Write the updated file
    with open('/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/main.py', 'w') as f:
        f.write(content)
    
    print("Fixed KeyError for L_d and L_v parameters")
    return True

if __name__ == "__main__":
    fix_ld_keyerror()