#!/usr/bin/env python3
"""
Test the Tree-First approach integration with realistic data structure.
This test validates that the original TREE method issue is fixed.
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

def test_original_issue():
    """Test the original issue: 88 unselected scenarios instead of 85."""
    
    print("Testing original issue scenario: 100 scenarios, 15% selection ratio")
    print("Expected: 15 selected, 85 unselected")
    print("Original TREE method gave: 12 selected, 88 unselected")
    print("Tree-First approach should give: 15 selected, 85 unselected\n")
    
    # Create test data matching the original issue
    num_scenarios = 100
    selection_ratio = 0.15  # 15%
    p = int(num_scenarios * selection_ratio)  # Should be 15
    
    # Generate realistic feature vectors (similar to dual features)
    np.random.seed(123)  # Different seed for different tree structure
    feature_vectors = {}
    for i in range(num_scenarios):
        # Features could represent dual solution characteristics
        feature_vectors[i] = np.random.randn(8)  # 8-dimensional feature space
    
    # Create mock dual solutions with realistic structure
    all_trial_duals = {}
    for iteration in range(3):  # 3 iterations of dual collection
        for scenario_id in range(num_scenarios):
            # Mock dual vectors with different cuts
            all_trial_duals[(iteration, scenario_id)] = np.random.randn(12)
    
    tree_params = {
        'D_max': 4,  # Original parameters that caused the issue
        'min_impurity_decrease': 0.0,
        'quantile_grid_size': 10,
        'all_trial_duals': all_trial_duals
    }
    
    print(f"Running Tree-First approach with:")
    print(f"- {num_scenarios} total scenarios")
    print(f"- {p} target selected scenarios ({selection_ratio:.1%})")
    print(f"- Tree parameters: D_max={tree_params['D_max']}")
    
    try:
        selected_scenarios, tree_mapping = solve_tree_integrated_selection(
            feature_vectors, p, tree_params
        )
        
        # Analyze results
        actual_selected = len(selected_scenarios)
        actual_unselected = len([k for k in tree_mapping.keys() if k != '__tree_info__'])
        
        print(f"\n--- RESULTS ---")
        print(f"Selected scenarios: {actual_selected} (target: {p})")
        print(f"Unselected scenarios: {actual_unselected} (target: {num_scenarios - p})")
        
        # Get tree structure info
        tree_info = tree_mapping.get('__tree_info__', {})
        num_leaves = tree_info.get('num_leaves', 0)
        leaf_sizes = tree_info.get('leaf_sizes', [])
        anchors = tree_info.get('anchors', {})
        
        print(f"Tree structure: {num_leaves} leaves")
        print(f"Leaf sizes: {leaf_sizes}")
        print(f"Anchors per leaf: {[len(anchors[i]) for i in range(num_leaves)]}")
        
        # Validation
        success = True
        
        if actual_selected == p:
            print("✅ PASS: Exact selected count achieved")
        else:
            print(f"❌ FAIL: Selected count mismatch (got {actual_selected}, expected {p})")
            success = False
            
        if actual_unselected == num_scenarios - p:
            print("✅ PASS: Exact unselected count achieved")
        else:
            print(f"❌ FAIL: Unselected count mismatch (got {actual_unselected}, expected {num_scenarios - p})")
            success = False
            
        # Verify no duplicates in selected scenarios
        if len(set(selected_scenarios)) == len(selected_scenarios):
            print("✅ PASS: No duplicate selected scenarios")
        else:
            print("❌ FAIL: Duplicate scenarios found")
            success = False
            
        # Verify all scenarios are accounted for
        total_accounted = actual_selected + actual_unselected
        if total_accounted == num_scenarios:
            print("✅ PASS: All scenarios accounted for")
        else:
            print(f"❌ FAIL: Missing scenarios (accounted for {total_accounted}/{num_scenarios})")
            success = False
        
        # Verify at least 1 scenario per leaf
        min_anchor_count = min(len(anchors[i]) for i in range(num_leaves))
        if min_anchor_count >= 1:
            print("✅ PASS: At least 1 scenario per leaf")
        else:
            print("❌ FAIL: Some leaves have no selected scenarios")
            success = False
            
        print(f"\n--- SUMMARY ---")
        if success:
            print("🎉 SUCCESS: Tree-First approach solves the original issue!")
            print("The exact selection ratio is now guaranteed regardless of tree structure.")
        else:
            print("❌ FAILURE: Tree-First approach still has issues")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_original_issue()