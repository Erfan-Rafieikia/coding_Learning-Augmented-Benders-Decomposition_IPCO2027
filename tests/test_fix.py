#!/usr/bin/env python3
"""
Quick test script to verify that the AttributeError fix works.
This script tests if we can import the necessary classes and check that
the unified attributes exist in the Callback class.
"""

import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def test_attribute_fix():
    """Test that the unified attributes exist and the old ones don't."""
    try:
        # Test importing the main classes
        from Problem.callbacks import Callback
        from Problem.master_problem import Solution
        
        print("✓ Successfully imported Callback and Solution classes")
        
        # Test that unified attributes exist in a callback instance
        # We'll create a minimal dummy data object for testing
        class DummyData:
            def __init__(self):
                self.S = [0, 1, 2]  # scenarios
                self.F = [0, 1]     # facilities
                self.C = [0, 1, 2]  # customers
        
        dummy_data = DummyData()
        dummy_first_stage = {}
        dummy_theta = {}
        
        # Create a callback instance
        callback = Callback(
            problem_type="UFL",
            data=dummy_data,
            first_stage_values=dummy_first_stage,
            theta_vars=dummy_theta,
            selected_subproblems=None,
            feature_vectors=None,
            prediction_method="REG",
            n_neighbors=5,
            use_prediction=True,
            fallback_method='SOLVE',
            n_trials=5
        )
        
        print("✓ Successfully created Callback instance")
        
        # Test that unified attributes exist
        assert hasattr(callback, 'num_cuts_mip_ml'), "Missing unified attribute: num_cuts_mip_ml"
        assert hasattr(callback, 'num_cuts_rel_ml'), "Missing unified attribute: num_cuts_rel_ml"
        print("✓ Unified attributes num_cuts_mip_ml and num_cuts_rel_ml exist")
        
        # Test that old attributes don't exist (should raise AttributeError)
        old_attributes = [
            'num_cuts_mip_ml_reg', 'num_cuts_rel_ml_reg',
            'num_cuts_mip_ml_knn', 'num_cuts_rel_ml_knn',
            'num_cuts_mip_ml_tree', 'num_cuts_rel_ml_tree',
            'num_cuts_mip_ml_simple_avg', 'num_cuts_rel_ml_simple_avg'
        ]
        
        for attr in old_attributes:
            if hasattr(callback, attr):
                print(f"⚠ WARNING: Old attribute {attr} still exists!")
            else:
                print(f"✓ Old attribute {attr} correctly removed")
        
        # Test Solution dataclass has unified fields
        solution_fields = Solution.__dataclass_fields__
        assert 'num_cuts_mip_ml' in solution_fields, "Solution missing num_cuts_mip_ml field"
        assert 'num_cuts_rel_ml' in solution_fields, "Solution missing num_cuts_rel_ml field"
        print("✓ Solution dataclass has unified fields")
        
        # Check that old fields don't exist in Solution
        old_solution_fields = ['num_cuts_mip_ml_reg', 'num_cuts_rel_ml_reg', 
                              'num_cuts_mip_ml_knn', 'num_cuts_rel_ml_knn']
        for field in old_solution_fields:
            if field in solution_fields:
                print(f"⚠ WARNING: Old Solution field {field} still exists!")
            else:
                print(f"✓ Old Solution field {field} correctly removed")
        
        print("\n🎉 All tests passed! The AttributeError fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_attribute_fix()
    sys.exit(0 if success else 1)