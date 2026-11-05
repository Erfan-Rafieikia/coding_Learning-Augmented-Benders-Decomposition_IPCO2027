#!/usr/bin/env python3
"""
Simple test to check that basic imports work and unified attributes exist.
"""

import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def test_basic_imports():
    """Test basic imports and attribute checking."""
    try:
        # Test importing the main classes
        from Problem.callbacks import Callback
        from Problem.master_problem import Solution
        
        print("✓ Successfully imported Callback and Solution classes")
        
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
        
        print("\n🎉 Basic tests passed! The AttributeError fix should work.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_imports()
    sys.exit(0 if success else 1)