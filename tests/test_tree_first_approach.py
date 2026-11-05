#!/usr/bin/env python3
"""
Test the Tree-First approach implementation to verify exact scenario selection counts.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from Dual_Prediction.training_and_selection import solve_tree_integrated_selection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tree_first_approach():
    """Test that Tree-First approach achieves exact scenario selection counts."""
    
    print("Testing Tree-First approach with proportional selection...")
    
    # Create test data with 100 scenarios
    num_scenarios = 100
    feature_dim = 5
    
    # Generate feature vectors
    np.random.seed(42)
    feature_vectors = {}
    for i in range(num_scenarios):
        feature_vectors[i] = np.random.randn(feature_dim)
    
    # Create mock dual solutions for dissimilarity computation
    all_trial_duals = {}
    for i in range(num_scenarios):
        # Use tuple keys as expected by the function: (iteration, scenario_id)
        all_trial_duals[(0, i)] = np.random.randn(10)  # Mock dual vector
        all_trial_duals[(1, i)] = np.random.randn(10)  # Second iteration
    
    # Test parameters
    tree_params = {
        'D_max': 4,
        'min_impurity_decrease': 0.0,
        'quantile_grid_size': 10,
        'all_trial_duals': all_trial_duals
    }
    
    # Test different selection sizes
    test_cases = [15, 20, 25, 30, 85]  # 15% selection ratio = 15 scenarios
    
    for p in test_cases:
        print(f"\n--- Testing p = {p} scenarios (ratio = {p/num_scenarios:.1%}) ---")
        
        try:
            selected_scenarios, tree_mapping = solve_tree_integrated_selection(
                feature_vectors, p, tree_params
            )
            
            # Verify exact selection count
            actual_count = len(selected_scenarios)
            print(f"Target: {p}, Actual: {actual_count}")
            
            if actual_count == p:
                print("✅ PASS: Exact scenario count achieved")
            else:
                print("❌ FAIL: Scenario count mismatch")
                
            # Verify tree info
            tree_info = tree_mapping.get('__tree_info__', {})
            num_leaves = tree_info.get('num_leaves', 0)
            leaf_sizes = tree_info.get('leaf_sizes', [])
            print(f"Tree structure: {num_leaves} leaves, sizes = {leaf_sizes}")
            
            # Verify each selected scenario is valid
            invalid_scenarios = [s for s in selected_scenarios if s not in feature_vectors]
            if invalid_scenarios:
                print(f"❌ FAIL: Invalid scenarios selected: {invalid_scenarios}")
            else:
                print("✅ PASS: All selected scenarios are valid")
                
            # Verify mapping coverage
            unselected_count = len([k for k in tree_mapping.keys() if k != '__tree_info__'])
            expected_unselected = num_scenarios - p
            print(f"Unselected scenarios mapped: {unselected_count}/{expected_unselected}")
            
            if unselected_count == expected_unselected:
                print("✅ PASS: All unselected scenarios mapped")
            else:
                print("❌ FAIL: Mapping count mismatch")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("Tree-First approach test complete!")

if __name__ == "__main__":
    test_tree_first_approach()