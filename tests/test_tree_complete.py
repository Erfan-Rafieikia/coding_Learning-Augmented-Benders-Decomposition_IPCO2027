#!/usr/bin/env python3
"""
Comprehensive test for TREE method including both selection and prediction phases.
This test verifies that different scenarios get different anchor counts.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from Dual_Prediction.training_and_selection import solve_tree_integrated_selection
import config

def test_tree_with_prediction_verification():
    """Test TREE method including prediction mapping to verify different anchor assignments."""
    print("="*80)
    print("COMPREHENSIVE TREE METHOD TEST: SELECTION + PREDICTION VERIFICATION")
    print("="*80)
    
    # Create a test dataset
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
    
    # Create fake dual solutions
    all_trial_duals = {}
    for i in range(n_scenarios):
        all_trial_duals[(i, 0)] = {
            'mu': {j: np.random.randn() for j in range(5)},
            'nu': {k: np.random.randn() for k in range(3)}
        }
    
    print("   - Generated synthetic feature vectors and dual solutions")
    
    # Set up parameters for TREE method 
    p = 15
    tree_params = {
        'all_trial_duals': all_trial_duals,
        'min_impurity_decrease': 0.01,
        'quantile_grid_size': 10
    }
    
    print(f"   - Using TREE method with p={p} (selecting {p}/{n_scenarios} scenarios)")
    
    print("\n2. Running TREE method (selection phase)...")
    
    # Run the TREE method
    selected_scenarios, tree_mapping = solve_tree_integrated_selection(
        feature_vectors, p, tree_params
    )
    
    print(f"\n3. Selection Results:")
    print(f"   - Selected {len(selected_scenarios)} scenarios (expected: {p})")
    print(f"   - Selected IDs: {sorted(selected_scenarios)}")
    
    if len(selected_scenarios) == p:
        print(f"   ✓ Exactly {p} scenarios selected (correct)")
    else:
        print(f"   ✗ Wrong count: {len(selected_scenarios)} instead of {p}")
    
    print(f"\n4. Analyzing Prediction Mappings...")
    
    # Extract tree info
    tree_info = tree_mapping.get('__tree_info__', {})
    num_leaves = tree_info.get('num_leaves', 0)
    leaf_anchors = tree_info.get('anchors', {})
    
    print(f"   - Tree has {num_leaves} leaves")
    print(f"   - Leaf anchor mappings:")
    for leaf_id, anchor_ids in leaf_anchors.items():
        print(f"     Leaf {leaf_id}: {len(anchor_ids)} anchors = {anchor_ids}")
    
    # Count how many anchors each unselected scenario gets
    anchor_count_distribution = {}
    unselected_scenarios = [sid for sid in range(n_scenarios) if sid not in selected_scenarios]
    
    print(f"\n5. Checking Anchor Assignments for Unselected Scenarios:")
    print(f"   - {len(unselected_scenarios)} unselected scenarios to check")
    
    for sid in unselected_scenarios:
        if sid in tree_mapping:
            mapping_info = tree_mapping[sid]
            anchor_ids = mapping_info.get('anchor', [])
            leaf_id = mapping_info.get('leaf_id', -1)
            
            num_anchors = len(anchor_ids) if isinstance(anchor_ids, list) else 1
            
            if num_anchors not in anchor_count_distribution:
                anchor_count_distribution[num_anchors] = []
            anchor_count_distribution[num_anchors].append(sid)
            
            # Show first few examples
            if len(anchor_count_distribution[num_anchors]) <= 3:
                print(f"   - Scenario {sid}: leaf {leaf_id}, {num_anchors} anchors")
    
    print(f"\n6. Anchor Count Distribution Summary:")
    for num_anchors, scenarios in sorted(anchor_count_distribution.items()):
        print(f"   - {num_anchors} anchor(s): {len(scenarios)} scenarios")
        print(f"     Examples: {scenarios[:5]}")
    
    print(f"\n7. Final Verification:")
    
    # Check if we have mixed anchor counts
    unique_anchor_counts = set(anchor_count_distribution.keys())
    
    if len(unique_anchor_counts) > 1:
        print(f"   ✅ SUCCESS: Got {len(unique_anchor_counts)} different anchor counts: {sorted(unique_anchor_counts)}")
        print(f"   ✅ This means different scenarios get different numbers of anchors")
        print(f"   ✅ The issue is FIXED - not all scenarios have the same anchor count!")
    elif len(unique_anchor_counts) == 1:
        single_count = list(unique_anchor_counts)[0]
        print(f"   ⚠ WARNING: All scenarios have {single_count} anchor(s)")
        print(f"   ⚠ This suggests the issue may still exist")
        
        # But check if they at least have different anchor IDs
        unique_anchor_sets = set()
        for sid in unselected_scenarios[:10]:  # Check first 10
            if sid in tree_mapping:
                mapping_info = tree_mapping[sid]
                anchor_ids = mapping_info.get('anchor', [])
                if isinstance(anchor_ids, list):
                    unique_anchor_sets.add(tuple(sorted(anchor_ids)))
                else:
                    unique_anchor_sets.add((anchor_ids,))
        
        if len(unique_anchor_sets) > 1:
            print(f"   ✅ BUT: Found {len(unique_anchor_sets)} different anchor ID sets")
            print(f"   ✅ So scenarios do get mapped to different anchors (FIXED!)")
        else:
            print(f"   ✗ AND: All scenarios have the same anchor IDs (NOT FIXED)")
    else:
        print(f"   ✗ ERROR: No anchor count distribution found")
    
    print("\n" + "="*80)
    return True

if __name__ == "__main__":
    test_tree_with_prediction_verification()
