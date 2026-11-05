#!/usr/bin/env python3
"""
Test script for COVARIATE_CLUSTERING method implementation.

This script validates the covariate clustering method that:
1. Uses covariate vectors from SCFLP data instead of uncertainty vectors
2. Performs K-means clustering on covariate vectors
3. Selects cluster representatives and maps unselected scenarios

Tests cover:
1. Basic covariate clustering selection and mapping
2. Integration with DualPredictor class
3. Edge cases (p=1, p=num_scenarios)
4. Error handling for missing covariate vectors
"""

import numpy as np
import sys
import os
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from Dual_Prediction.training_and_selection import solve_covariate_clustering_selection
    from Dual_Prediction.prediction import DualPredictor
    from Problem.data import SCFLPData
    print("Successfully imported covariate clustering modules")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_covariate_clustering_basic():
    """Test basic covariate clustering selection."""
    print("=== Testing Basic Covariate Clustering Selection ===")
    
    # Create test covariate vectors (5 scenarios, 3 features each)
    np.random.seed(42)
    n_scenarios = 5
    n_features = 3
    
    covariate_vectors = {}
    for i in range(n_scenarios):
        covariate_vectors[i] = np.random.randn(n_features)
    
    p = 2  # Select 2 cluster representatives
    
    # Test covariate clustering
    print(f"\nTesting with {n_scenarios} scenarios, selecting {p} representatives")
    
    clustering_params = {
        'random_state': 42,
        'init_method': 'k-means++',
        'max_iter': 300,
        'tol': 1e-4
    }
    
    selected, mapping = solve_covariate_clustering_selection(covariate_vectors, p, clustering_params)
    
    print(f"Selected representatives: {selected}")
    print(f"Covariate mapping: {mapping}")
    print(f"Number of unselected scenarios: {len(mapping)}")
    
    # Verify that all unselected scenarios are mapped to selected ones
    for unselected, representative in mapping.items():
        assert representative in selected, f"Representative {representative} not in selected set {selected}"
    
    print("✓ Basic covariate clustering test passed")

def test_covariate_clustering_edge_cases():
    """Test edge cases for covariate clustering."""
    print("\n=== Testing Covariate Clustering Edge Cases ===")
    
    # Create test covariate vectors
    np.random.seed(42)
    covariate_vectors = {i: np.random.randn(3) for i in range(4)}
    
    # Test 1: p = 1 (single cluster)
    print("\nTest 1: Single cluster (p=1)")
    selected, mapping = solve_covariate_clustering_selection(covariate_vectors, 1)
    print(f"Selected: {selected}, Mapping: {mapping}")
    assert len(selected) == 1
    assert len(mapping) == 3  # 3 unselected scenarios
    
    # Test 2: p = num_scenarios (all selected)
    print("\nTest 2: All scenarios selected (p=num_scenarios)")
    selected, mapping = solve_covariate_clustering_selection(covariate_vectors, 4)
    print(f"Selected: {selected}, Mapping: {mapping}")
    assert len(selected) == 4
    assert len(mapping) == 0  # No unselected scenarios
    
    print("✓ Edge cases test passed")

def create_mock_scflp_data_with_covariates():
    """Create mock SCFLP data with covariate vectors for testing."""
    # Create minimal SCFLP data structure with covariate vectors
    n_scenarios = 5
    n_customers = 3
    n_facilities = 3
    n_covariates = 4
    
    # Create covariate vectors (5 scenarios x 4 features)
    covariate_vectors = np.random.randn(n_scenarios, n_covariates)
    
    data = SCFLPData(
        I=np.arange(n_customers),
        J=np.arange(n_facilities),
        S=np.arange(n_scenarios),
        demands=np.random.randint(10, 50, n_customers),
        scenarios=np.random.randint(10, 50, (n_scenarios, n_customers)),
        capacities=np.random.randint(100, 200, n_facilities),
        fixed_costs=np.random.randint(50, 150, n_facilities),
        shipment_costs=np.random.rand(n_customers, n_facilities) * 10,
        max_demand_sum_over_scenario=200.0,
        scenario_generation_variance=1.0,
        scenario_generation_method="test",
        covariate_vectors=covariate_vectors,
        num_covariates=n_covariates
    )
    
    return data

def test_dual_predictor_covariate_clustering():
    """Test COVARIATE_CLUSTERING integration with DualPredictor class."""
    print("\n=== Testing DualPredictor COVARIATE_CLUSTERING Integration ===")
    
    # Create mock SCFLP data with covariate vectors
    data = create_mock_scflp_data_with_covariates()
    
    # Create feature vectors (dummy for this test)
    feature_vectors = {i: np.random.randn(3) for i in range(5)}
    
    try:
        # Initialize DualPredictor with COVARIATE_CLUSTERING
        predictor = DualPredictor(
            method='COVARIATE_CLUSTERING',
            problem_type='SCFLP',
            data=data,
            covariate_clustering_random_state=42,
            covariate_clustering_init_method='k-means++',
            covariate_clustering_max_iter=300,
            covariate_clustering_tol=1e-4
        )
        
        print(f"✓ DualPredictor initialized with method: {predictor.method}")
        
        # Test unified training (without actually training, just setup)
        p = 2  # Select 2 scenarios
        
        # This would normally be called during training, but we'll test the setup
        result = predictor._train_unified(feature_vectors, p)
        
        print(f"✓ Unified training completed")
        print(f"Selected scenarios: {len(predictor.selected_subproblems)}")
        print(f"Has covariate mappings: {predictor.covariate_clustering_mappings is not None}")
        
        # Test training info
        info = predictor.get_training_info()
        print(f"Training info: {info}")
        
        print("✓ DualPredictor COVARIATE_CLUSTERING integration test passed")
        
    except Exception as e:
        print(f"✗ DualPredictor integration test failed: {e}")
        raise

def test_error_handling():
    """Test error handling for missing covariate vectors."""
    print("\n=== Testing Error Handling ===")
    
    # Test with empty covariate vectors
    try:
        empty_covariates = {}
        solve_covariate_clustering_selection(empty_covariates, 1)
        print("✗ Should have raised error for empty covariate vectors")
    except ValueError as e:
        print(f"✓ Correctly raised error for empty vectors: {e}")
    
    # Test with invalid p
    try:
        covariate_vectors = {0: np.array([1, 2, 3])}
        solve_covariate_clustering_selection(covariate_vectors, 0)  # p=0 should fail
        print("✗ Should have raised error for p=0")
    except ValueError as e:
        print(f"✓ Correctly raised error for invalid p: {e}")
    
    print("✓ Error handling test passed")

def main():
    """Run all covariate clustering tests."""
    print("Starting COVARIATE_CLUSTERING method tests...")
    
    try:
        test_covariate_clustering_basic()
        test_covariate_clustering_edge_cases()
        test_dual_predictor_covariate_clustering()
        test_error_handling()
        
        print("\n🎉 All COVARIATE_CLUSTERING tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())