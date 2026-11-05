#!/usr/bin/env python3
"""Minimal test callback to verify parameter passing."""

import sys
sys.path.append('src')

class TestCallback:
    def __init__(
        self,
        problem_type,
        data,
        first_stage_values,
        theta_vars,
        max_depth=5,
        min_samples_leaf=2,
    ):
        """Test callback with max_depth and min_samples_leaf parameters."""
        self.problem_type = problem_type
        self.data = data
        self.first_stage_values = first_stage_values
        self.theta_vars = theta_vars
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        print(f"✅ TestCallback created with max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")

if __name__ == "__main__":
    # Test this works
    test = TestCallback("UFL", None, {}, {}, max_depth=3, min_samples_leaf=1)
    
    # Now test the actual import
    try:
        from Problem.callbacks import Callback
        import inspect
        
        sig = inspect.signature(Callback.__init__)
        params = list(sig.parameters.keys())
        print(f"Actual Callback parameters: {params}")
        
        if 'max_depth' in params:
            print("✅ max_depth found in real Callback")
        else:
            print("❌ max_depth NOT found in real Callback")
            
    except Exception as e:
        print(f"Error: {e}")