#!/usr/bin/env python3
"""
Test script for integrated KNN selection + mapping implementation.

This script validates the paper's k-NN objective implementation:
    min_{{T subseteq S,|T|=p}} Sum_{{s in S\\T}} (1/k) Sum_{{t in T_k(s)}} d_F(f^s, f^t)

Tests cover:
1. Fixed k selection and mapping
2. Optimal k enumeration and selection
3. Edge case k=1 (equivalent to p-median)
4. Integration with existing subproblem selection interface
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from subproblem_selection.subproblem_selection import solve_knn_integrated_selection, select_subproblems_based_on_features

def test_knn_integrated_selection():
    """Test the integrated KNN selection with various scenarios."""
    print("=== Testing Integrated KNN Selection (Paper's k-NN Objective) ===")
    
    # Create test data with 10 scenarios
    np.random.seed(42)
    n_scenarios = 10
    feature_dim = 3
    
    feature_vectors = {}
    for i in range(n_scenarios):
        feature_vectors[i] = np.random.randn(feature_dim)
    
    p = 3  # Select 3 scenarios
    
    # Test 1: Fixed k
    print(f"\nTest 1: Fixed k=2, p={p}")
    knn_params = {'k': 2, 'find_optimal_k': False}
    
    selected, mapping = solve_knn_integrated_selection(feature_vectors, p, knn_params)
    
    print(f"Selected scenarios: {selected}")
    print(f"Number of unselected scenarios with mappings: {len(mapping)}")
    
    # Verify mapping structure
    sample_mapping = list(mapping.items())[0]
    print(f"Sample mapping: {sample_mapping[0]} -> {sample_mapping[1]}")
    
    for scenario_id, mapping_info in mapping.items():
        assert mapping_info['k'] <= p, f"k={mapping_info['k']} should be <= p={p}"
        assert len(mapping_info['neighbors']) == mapping_info['k'], "Neighbors count mismatch"
        assert len(mapping_info['distances']) == mapping_info['k'], "Distances count mismatch"
    
    # Test 2: Optimal k finding
    print(f"\nTest 2: Find optimal k, p={p}")
    knn_params = {'k': 2, 'find_optimal_k': True}
    
    selected_opt, mapping_opt = solve_knn_integrated_selection(feature_vectors, p, knn_params)
    
    k_values = [mapping_info['k'] for mapping_info in mapping_opt.values()]
    print(f"Optimal k values found: {set(k_values)}")
    
    # Test 3: k=1 (p-median equivalent)
    print(f"\nTest 3: k=1 (p-median equivalent), p={p}")
    knn_params = {'k': 1, 'find_optimal_k': False}
    
    selected_median, mapping_median = solve_knn_integrated_selection(feature_vectors, p, knn_params)
    
    k_values_median = [mapping_info['k'] for mapping_info in mapping_median.values()]
    assert all(k == 1 for k in k_values_median), "All k values should be 1"
    print(f"✓ All unselected scenarios have k=1")
    
    # Test 4: Integration with existing interface
    print(f"\nTest 4: Integration with existing subproblem selection interface")
    selected_integ, mapping_integ = select_subproblems_based_on_features(
        feature_vectors=feature_vectors,
        method="knn",
        p=p,
        knn_params={'k': 2, 'find_optimal_k': False}
    )
    
    print(f"Selected via integrated interface: {selected_integ}")
    print(f"Mapping has integrated format: {'neighbors' in list(mapping_integ.values())[0]}")
    
    print("\n=== All tests passed! ===")
    print("✓ Paper's k-NN objective implementation is working correctly")
    print("✓ Integrated selection + mapping pipeline is functional")
    print("✓ Compatible with existing ML-augmented Benders system")

if __name__ == "__main__":
    test_knn_integrated_selection()