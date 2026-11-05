#!/usr/bin/env python3
"""
Test the new adaptive TREE method implementation.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Dual_Prediction.training_and_selection import solve_tree_integrated_selection

def test_adaptive_tree():
    """Test the adaptive tree method with synthetic data."""
    
    # Create synthetic feature vectors (5 scenarios, 3 features each)
    features = {
        0: np.array([1.0, 2.0, 0.5]),
        1: np.array([1.5, 2.5, 0.7]),
        2: np.array([3.0, 1.0, 1.2]),
        3: np.array([3.5, 1.5, 1.8]),
        4: np.array([2.0, 3.0, 0.9])
    }
    
    # Create synthetic dual solutions (empty dictionaries for simplicity)
    dual_solutions = [
        {'x1': 0.1, 'x2': 0.2},
        {'x1': 0.15, 'x2': 0.25}, 
        {'x1': 0.3, 'x2': 0.1},
        {'x1': 0.35, 'x2': 0.15},
        {'x1': 0.2, 'x2': 0.3}
    ]
    
    # Test parameters
    p = 3  # Select 3 out of 5 scenarios
    tree_params = {
        'all_trial_duals': dual_solutions,
        'min_impurity_decrease': 0.0,
        'quantile_grid_size': 5
    }
    
    try:
        print("Testing adaptive TREE method...")
        selected_scenarios, tree_mapping = solve_tree_integrated_selection(
            features, p, tree_params
        )
        
        print(f"✓ Success! Selected {len(selected_scenarios)} scenarios: {selected_scenarios}")
        print(f"✓ Tree mapping: {tree_mapping}")
        
        # Validate selection
        assert len(selected_scenarios) == p, f"Expected {p} scenarios, got {len(selected_scenarios)}"
        assert len(set(selected_scenarios)) == p, "Duplicate scenarios selected"
        assert all(0 <= s <= 4 for s in selected_scenarios), "Invalid scenario indices"
        
        print("✓ All validation checks passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_adaptive_tree()
    sys.exit(0 if success else 1)