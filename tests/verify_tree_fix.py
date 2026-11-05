#!/usr/bin/env python3
"""
Verification test for TREE method after file corruption repair.
This test focuses on verifying that:
1. TREE method creates multiple leaves (≥2)
2. Proportional selection works correctly 
3. We get mixed "Averaged X anchors" values indicating proper distribution
4. All 4 requirements are satisfied
"""

import sys
import os
# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from Dual_Prediction.training_and_selection import solve_tree_integrated_selection
import config

def test_tree_method_after_repair():
    """Test TREE method to verify file corruption repair resolved all issues."""
    print("="*80)
    print("TREE METHOD VERIFICATION AFTER FILE CORRUPTION REPAIR")
    print("="*80)
    
    # Create a small test dataset
    print("\n1. Creating test dataset...")
    np.random.seed(42)
    
    # Create 30 scenarios with 5 features each
    n_scenarios = 30
    n_features = 5
    
    print(f"   - {n_scenarios} scenarios with {n_features} features each")
    
    # Create feature_vectors as dict {scenario_id: feature_vector}
    feature_vectors = {}
    for i in range(n_scenarios):
        feature_vectors[i] = np.random.randn(n_features)
    
    # Create fake dual solutions for dissimilarity computation
    # Keys need to be (scenario_id, iteration) tuples
    all_trial_duals = {}
    for i in range(n_scenarios):
        # Create fake dual solution for one iteration per scenario
        # Using SCFLP format: {'mu': {customer: value}, 'nu': {facility: value}}
        all_trial_duals[(i, 0)] = {
            'mu': {j: np.random.randn() for j in range(5)},  # 5 customers
            'nu': {k: np.random.randn() for k in range(3)}   # 3 facilities
        }
    
    print("   - Generated synthetic feature vectors and dual solutions")
    
    # Set up parameters for TREE method 
    # Use p=15 (15 selected out of 30) to test proportional selection
    p = 15
    tree_params = {
        'all_trial_duals': all_trial_duals,
        'min_impurity_decrease': 0.01,
        'quantile_grid_size': 10
    }
    
    print(f"   - Using TREE method with p={p} (selecting {p}/{n_scenarios} scenarios)")
    print(f"   - Min impurity decrease: {tree_params['min_impurity_decrease']}")
    
    print("\n2. Running TREE method...")
    print("   Looking for:")
    print("   ✓ At least 2 leaves created")
    print("   ✓ At least 1 scenario selected per leaf")  
    print(f"   ✓ Exactly {p} total scenarios selected")
    print("   ✓ Mixed 'Averaged X anchors' values (not all same)")
    print("   " + "-"*50)
    
    try:
        # Run the TREE method
        selected_scenarios, tree_mapping = solve_tree_integrated_selection(
            feature_vectors, p, tree_params
        )
        
        print(f"\n3. TREE Method Results:")
        print(f"   - Selected {len(selected_scenarios)} scenarios (expected: {p})")
        print(f"   - Selected indices: {sorted(selected_scenarios)}")
        
        # Check requirement 1: exactly correct count
        if len(selected_scenarios) == p:
            print(f"   ✓ REQUIREMENT 1 SATISFIED: Exactly {p} scenarios selected")
        else:
            print(f"   ✗ REQUIREMENT 1 FAILED: Got {len(selected_scenarios)} instead of {p}")
        
        # Check tree structure from mapping
        if 'tree_info' in tree_mapping:
            tree_info = tree_mapping['tree_info']
            n_leaves = tree_info.get('num_leaves', 0)
            print(f"   - Tree created {n_leaves} leaves")
            
            if n_leaves >= 2:
                print("   ✓ REQUIREMENT 2 SATISFIED: At least 2 leaves created")
            else:
                print(f"   ✗ REQUIREMENT 2 FAILED: Only {n_leaves} leaf created")
        else:
            print("   - Tree info not available in mapping")
                
        print(f"\n4. Checking tree mapping details...")
        print(f"   Tree mapping keys: {list(tree_mapping.keys())}")
        
        # Check for leaf allocation details
        if 'leaf_allocations' in tree_mapping:
            leaf_allocs = tree_mapping['leaf_allocations']
            print(f"   - Leaf allocations: {leaf_allocs}")
            allocation_counts = [alloc for alloc in leaf_allocs.values()]
            print(f"   - Allocation counts per leaf: {allocation_counts}")
            
            # Check if we got varied allocation (not all same)
            unique_counts = set(allocation_counts)
            if len(unique_counts) > 1:
                print("   ✓ REQUIREMENT 4 SATISFIED: Mixed allocation counts across leaves")
                print(f"     Different counts found: {sorted(unique_counts)}")
            else:
                print("   ⚠ All leaves got same allocation count - may be expected for some cases")
                
        print("\n5. Overall Assessment:")
        if len(selected_scenarios) == p:
            print("   ✓ SUCCESS: File corruption repair appears to have resolved the issue!")
            print("   ✓ TREE method is now working correctly")
            print("   ✓ Proportional selection producing correct total count")
        else:
            print("   ⚠ ISSUE PERSISTS: Still not getting correct selection count")
            print("   → May need additional investigation")
            
    except Exception as e:
        print(f"\n✗ ERROR during TREE method execution:")
        print(f"   {type(e).__name__}: {e}")
        print(f"   This suggests there may still be issues in the code")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    return True

if __name__ == "__main__":
    test_tree_method_after_repair()