#!/usr/bin/env python3
"""Quick test to check Callback parameters."""

import sys
import os
sys.path.append('src')

from Problem.callbacks import Callback
import inspect

def test_callback_signature():
    """Test the Callback constructor signature."""
    print("Testing Callback constructor signature...")
    
    # Get the signature
    sig = inspect.signature(Callback.__init__)
    params = list(sig.parameters.keys())
    
    print(f"Total parameters: {len(params)}")
    print(f"All parameters: {params}")
    
    # Check for specific parameters
    required_params = ['max_depth', 'min_samples_leaf']
    for param in required_params:
        if param in params:
            print(f"✅ {param} found in signature")
        else:
            print(f"❌ {param} NOT found in signature")
    
    # Try to create instance with parameters
    try:
        # Create a minimal data mock
        class MockData:
            pass
        
        mock_data = MockData()
        
        callback = Callback(
            problem_type="UFL",
            data=mock_data,
            first_stage_values={},
            theta_vars={},
            max_depth=3,
            min_samples_leaf=1
        )
        print("✅ Successfully created Callback with max_depth and min_samples_leaf")
        
    except TypeError as e:
        print(f"❌ TypeError creating Callback: {e}")
    except Exception as e:
        print(f"⚠️  Other error (expected): {e}")

if __name__ == "__main__":
    test_callback_signature()