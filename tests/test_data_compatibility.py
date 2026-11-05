#!/usr/bin/env python3
"""
Test script to verify SCFLP data reader compatibility across different generation methods.
This test checks that the updated SCFLPData class and reader can handle:
1. Legacy files (normal, uniform, gamma, etc.) without covariates
2. New ARMA files with feature vectors and validation metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Problem.data import read_scflp_data, SCFLPData
import numpy as np

def test_legacy_format():
    """Test reading a legacy normal distribution file (no covariates)"""
    print("Testing legacy format (normal distribution)...")
    
    file_path = "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP/scenarios_s100_cv0.1_distnormal/p1_stochastic.txt"
    
    data = read_scflp_data(file_path)
    
    # Basic structure checks
    assert isinstance(data, SCFLPData)
    assert len(data.I) == 50  # 50 customers
    assert len(data.J) == 10  # 10 facilities 
    assert len(data.S) == 100  # 100 scenarios
    assert data.scenarios.shape == (100, 50)  # 100 scenarios x 50 customers
    
    # Check legacy fields are populated
    assert data.distribution == "normal"
    assert data.target_cv == 0.1
    assert data.empirical_cv_mean > 0  # Should have some empirical CV
    
    # Check ARMA fields are None/default for legacy files
    assert data.feature_vectors is None
    assert data.num_covariates == 0
    assert data.validation_r_squared is None
    assert data.validation_mse is None
    assert data.validation_mape is None
    
    print(f"✓ Legacy file loaded successfully:")
    print(f"  - Distribution: {data.distribution}")
    print(f"  - Target CV: {data.target_cv}")
    print(f"  - Empirical CV: {data.empirical_cv_mean:.4f}")
    print(f"  - Covariate fields: None (as expected)")
    print()

def test_arma_format():
    """Test reading an ARMA file (with covariates and validation metrics)"""
    print("Testing ARMA format (with covariates)...")
    
    file_path = "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP/scenarios_s100_cov3_distarma(2,2)_with_covariate_generation/p1_stochastic.txt"
    
    data = read_scflp_data(file_path)
    
    # Basic structure checks
    assert isinstance(data, SCFLPData)
    assert len(data.I) == 50  # 50 customers
    assert len(data.J) == 10  # 10 facilities 
    assert len(data.S) == 100  # 100 scenarios
    assert data.scenarios.shape == (100, 50)  # 100 scenarios x 50 customers
    
    # Check ARMA-specific fields are populated
    assert data.distribution == "arma(2,2)_with_covariate_generation"
    assert data.num_covariates == 3
    assert data.feature_vectors is not None
    assert data.feature_vectors.shape == (100, 3)  # 100 scenarios x 3 covariates
    
    # Check validation metrics are present
    assert data.validation_r_squared is not None
    assert data.validation_mse is not None
    assert data.validation_mape is not None
    
    # Validation metrics should be reasonable (positive MSE, MAPE; R² can be negative)
    assert data.validation_mse > 0
    assert data.validation_mape > 0
    
    print(f"✓ ARMA file loaded successfully:")
    print(f"  - Distribution: {data.distribution}")
    print(f"  - Number of covariates: {data.num_covariates}")
    print(f"  - Feature vectors shape: {data.feature_vectors.shape}")
    print(f"  - Validation R²: {data.validation_r_squared:.6f}")
    print(f"  - Validation MSE: {data.validation_mse:.2f}")
    print(f"  - Validation MAPE: {data.validation_mape:.2f}%")
    print()

def test_backward_compatibility():
    """Test that both formats produce valid SCFLP data for optimization"""
    print("Testing backward compatibility for optimization use...")
    
    # Load both formats
    legacy_file = "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP/scenarios_s100_cv0.1_distnormal/p1_stochastic.txt"
    arma_file = "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP/scenarios_s100_cov3_distarma(2,2)_with_covariate_generation/p1_stochastic.txt"
    
    legacy_data = read_scflp_data(legacy_file)
    arma_data = read_scflp_data(arma_file)
    
    # Both should have the same core structure (same base problem instance)
    assert np.array_equal(legacy_data.capacities, arma_data.capacities)
    assert np.array_equal(legacy_data.fixed_costs, arma_data.fixed_costs)
    assert np.array_equal(legacy_data.demands, arma_data.demands)
    assert np.array_equal(legacy_data.shipment_costs, arma_data.shipment_costs)
    
    # Both should have valid optimization data
    for data, name in [(legacy_data, "Legacy"), (arma_data, "ARMA")]:
        assert data.max_demand_sum_over_scenario > 0, f"{name}: Invalid max demand"
        assert len(data.cost) == len(data.shipment_costs), f"{name}: Cost alias failed"
        assert len(data.capacity) == len(data.capacities), f"{name}: Capacity alias failed"
        
    print(f"✓ Both formats provide valid optimization data:")
    print(f"  - Same base problem structure: ✓")
    print(f"  - Valid demand bounds: ✓")  
    print(f"  - Property aliases work: ✓")
    print()

def main():
    """Run all compatibility tests"""
    print("=== SCFLP Data Reader Compatibility Test ===")
    print()
    
    try:
        test_legacy_format()
        test_arma_format()
        test_backward_compatibility()
        
        print("🎉 All tests passed! The SCFLP data reader is compatible across generation methods.")
        print()
        print("Summary:")
        print("- Legacy files (normal, uniform, etc.): ✓ Supported")
        print("- ARMA files (with covariates): ✓ Supported") 
        print("- Backward compatibility: ✓ Maintained")
        print("- Optimization readiness: ✓ Verified")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()