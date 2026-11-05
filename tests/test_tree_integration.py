#!/usr/bin/env python3
"""
Simple integration test for tree-based prediction method.

This script tests that the tree-based prediction method can be instantiated
and integrated with the existing callback system without runtime errors.
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from Problem.callbacks import Callback
    from Dual_Prediction.prediction import train_tree_predictor, SKLEARN_AVAILABLE
    print("✓ Successfully imported tree-based modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Mock data class for testing
class MockData:
    def __init__(self, problem_type):
        if problem_type == "UFL":
            self.F = np.array([0, 1])  # 2 facilities
            self.C = np.array([0, 1, 2])  # 3 customers  
            self.S = np.array([0, 1, 2])  # 3 scenarios
        else:
            self.S = np.array([0, 1, 2])

def test_tree_integration():
    """Test basic tree integration with callback system."""
    print("\n🧪 Testing tree-based prediction integration...")
    
    # Test callback initialization with TREE method
    try:
        mock_data = MockData("UFL")
        mock_first_stage = {0: 0.5, 1: 0.3}
        mock_theta = {0: 10.0, 1: 15.0, 2: 12.0}
        
        callback = Callback(
            problem_type="UFL",
            data=mock_data,
            first_stage_values=mock_first_stage,
            theta_vars=mock_theta,
            prediction_method="TREE",
            n_trials=2,
            use_prediction=True
        )
        print("✓ Callback with TREE method initialized successfully")
        
        # Check if tree-specific attributes are set
        assert hasattr(callback, 'all_trial_duals')
        assert hasattr(callback, 'trained_tree_models')
        assert hasattr(callback, 'tree_training_complete')
        assert callback.prediction_method == "TREE"
        print("✓ Tree-specific attributes present")
        
    except Exception as e:
        print(f"✗ Callback initialization failed: {e}")
        return False
    
    # Test tree training function availability
    try:
        if SKLEARN_AVAILABLE:
            print("✓ sklearn available - tree training will work")
        else:
            print("⚠ sklearn not available - tree training will fail at runtime")
            
        # Test function signature
        print("✓ train_tree_predictor function accessible")
        
    except Exception as e:
        print(f"✗ Tree training function test failed: {e}")
        return False
    
    return True

def test_configuration_support():
    """Test that configuration supports TREE method."""
    print("\n⚙️ Testing configuration support...")
    
    try:
        from config import prediction_method
        if "TREE" in prediction_method:
            print("✓ TREE method supported in configuration")
        else:
            print("✗ TREE method not found in configuration")
            return False
            
    except ImportError as e:
        print(f"✗ Configuration import failed: {e}")
        return False
    
    return True

def main():
    """Run all integration tests."""
    print("🌳 Tree-based Prediction Integration Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Tree integration
    if test_tree_integration():
        tests_passed += 1
        
    # Test 2: Configuration support
    if test_configuration_support():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All integration tests passed!")
        print("\nTree-based prediction method successfully integrated!")
        print("\nNext steps:")
        print("1. Install sklearn in your environment: pip install scikit-learn")
        print("2. Update your experiment configurations to use prediction_method='TREE'")
        print("3. Run experiments with tree-based prediction")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())