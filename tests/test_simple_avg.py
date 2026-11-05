#!/usr/bin/env python3
"""
Test script for Simple_Avg prediction method in the Benders decomposition framework.

This script demonstrates and tests the new SIMPLE_AVG prediction method by:
1. Creating a mock UFL problem setup
2. Testing the DualPredictor with SIMPLE_AVG method
3. Verifying that predictions are correct averages of selected scenarios
"""

import sys
import os
import numpy as np
from dataclasses import dataclass

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary modules
from Dual_Prediction.prediction import DualPredictor

# Mock UFL data class for testing
@dataclass 
class MockUFLData:
    F: np.ndarray  # Facilities
    S: np.ndarray  # Scenarios

def test_simple_avg_prediction():
    """Test the Simple_Avg prediction method."""
    print("Testing SIMPLE_AVG prediction method...")
    
    # Create mock data
    data = MockUFLData(
        F=np.array([0, 1, 2]),  # 3 facilities
        S=np.array([0, 1, 2, 3, 4])  # 5 scenarios
    )
    
    # Create feature vectors (not used by Simple_Avg but required by interface)
    feature_vectors = {
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([2.0, 3.0, 4.0]),
        2: np.array([3.0, 4.0, 5.0]),
        3: np.array([4.0, 5.0, 6.0]),
        4: np.array([5.0, 6.0, 7.0])
    }
    
    # Selected scenarios (will be used for averaging)
    selected_subproblems = {0, 1, 2}
    unselected_subproblems = [3, 4]
    
    # Create dual cache with mock dual solutions for selected scenarios
    dual_cache = {
        0: {
            'lambda': 10.0,
            'mu': {0: 1.0, 1: 2.0, 2: 3.0}
        },
        1: {
            'lambda': 20.0,
            'mu': {0: 4.0, 1: 5.0, 2: 6.0}
        },
        2: {
            'lambda': 30.0,
            'mu': {0: 7.0, 1: 8.0, 2: 9.0}
        }
    }
    
    # Expected averages
    expected_lambda = (10.0 + 20.0 + 30.0) / 3  # = 20.0
    expected_mu = {
        0: (1.0 + 4.0 + 7.0) / 3,  # = 4.0
        1: (2.0 + 5.0 + 8.0) / 3,  # = 5.0  
        2: (3.0 + 6.0 + 9.0) / 3   # = 6.0
    }
    
    print(f"Expected lambda: {expected_lambda}")
    print(f"Expected mu: {expected_mu}")
    
    # Initialize Simple_Avg predictor
    predictor = DualPredictor(
        method='SIMPLE_AVG',
        problem_type='UFL',
        data=data
    )
    
    # Train the predictor
    print("\nTraining SIMPLE_AVG predictor...")
    training_result = predictor.train(
        feature_vectors=feature_vectors,
        selected_subproblems=selected_subproblems
    )
    
    print(f"Training result: {training_result}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(
        unselected_subs=unselected_subproblems,
        dual_cache=dual_cache
    )
    
    print(f"Predictions: {predictions}")
    
    # Verify predictions
    print("\nVerifying predictions...")
    success = True
    
    for sid in unselected_subproblems:
        if sid not in predictions:
            print(f"ERROR: No prediction for scenario {sid}")
            success = False
            continue
            
        pred = predictions[sid]
        
        # Check lambda
        if abs(pred['lambda'] - expected_lambda) > 1e-10:
            print(f"ERROR: Lambda mismatch for scenario {sid}: {pred['lambda']} != {expected_lambda}")
            success = False
        
        # Check mu values
        for j in data.F:
            if abs(pred['mu'][j] - expected_mu[j]) > 1e-10:
                print(f"ERROR: Mu[{j}] mismatch for scenario {sid}: {pred['mu'][j]} != {expected_mu[j]}")
                success = False
    
    if success:
        print("✅ All tests passed! SIMPLE_AVG prediction method works correctly.")
    else:
        print("❌ Some tests failed!")
        
    # Test training info
    print("\nGetting training info...")
    training_info = predictor.get_training_info()
    print(f"Training info: {training_info}")
    
    return success

def test_predictor_initialization():
    """Test that SIMPLE_AVG is accepted as a valid method."""
    print("\nTesting predictor initialization...")
    
    try:
        # This should work
        predictor = DualPredictor(
            method='SIMPLE_AVG',
            problem_type='UFL',
            data=MockUFLData(F=np.array([0, 1]), S=np.array([0, 1]))
        )
        print("✅ SIMPLE_AVG method accepted successfully")
        return True
    except ValueError as e:
        print(f"❌ SIMPLE_AVG method rejected: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing SIMPLE_AVG Prediction Method Implementation")
    print("=" * 60)
    
    # Test 1: Initialization
    test1_passed = test_predictor_initialization()
    
    # Test 2: Prediction functionality
    test2_passed = test_simple_avg_prediction()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! SIMPLE_AVG method is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)

if __name__ == "__main__":
    main()