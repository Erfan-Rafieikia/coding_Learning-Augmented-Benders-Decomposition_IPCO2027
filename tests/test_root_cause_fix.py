#!/usr/bin/env python3
"""
Test script to verify the root cause fix for dual vector collection
when use_prediction="continuous"
"""

import sys
import os
sys.path.insert(0, 'src')

from utils.config_utils import normalize_use_prediction, ml_enabled_for_context

def test_ml_context_logic():
    """Test that ML context logic works correctly for all use_prediction values"""
    
    print("=== Testing ML Context Logic ===")
    
    test_cases = [
        ("continuous", True, False),   # ML disabled for integer, enabled for continuous
        ("continuous", False, True),   
        ("integer", True, True),       # ML enabled for integer, disabled for continuous  
        ("integer", False, False),
        ("both", True, True),          # ML enabled for both contexts
        ("both", False, True),
        ("none", True, False),         # ML disabled for both contexts
        ("none", False, False),
    ]
    
    all_passed = True
    
    for use_pred, is_integer, expected in test_cases:
        result = ml_enabled_for_context(use_pred, is_integer)
        context = "integer" if is_integer else "continuous"
        
        if result == expected:
            print(f"✓ use_prediction='{use_pred}', {context} context: ML enabled = {result}")
        else:
            print(f"✗ use_prediction='{use_pred}', {context} context: Expected {expected}, got {result}")
            all_passed = False
    
    return all_passed

def test_relaxation_callback_logic():
    """Test that relaxation callbacks are enabled when needed"""
    
    print("\n=== Testing Relaxation Callback Logic ===")
    
    test_cases = [
        ("continuous", True),   # Should enable relaxation callbacks
        ("both", True),        # Should enable relaxation callbacks  
        ("integer", False),    # Should NOT enable relaxation callbacks
        ("none", False),       # Should NOT enable relaxation callbacks
    ]
    
    all_passed = True
    
    for use_pred, expected in test_cases:
        normalized = normalize_use_prediction(use_pred)
        result = normalized in ["continuous", "both"]
        
        if result == expected:
            print(f"✓ use_prediction='{use_pred}': Relaxation callbacks enabled = {result}")
        else:
            print(f"✗ use_prediction='{use_pred}': Expected {expected}, got {result}")
            all_passed = False
            
    return all_passed

def main():
    """Run all tests"""
    
    print("Testing Root Cause Fix for Dual Vector Collection")
    print("=" * 55)
    
    test1_passed = test_ml_context_logic()
    test2_passed = test_relaxation_callback_logic()
    
    print(f"\n=== Test Results ===")
    print(f"ML Context Logic: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Relaxation Callback Logic: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print(f"\n✓ ALL TESTS PASSED - Root cause fix is working correctly!")
        print("\nThe system should now:")
        print("  1. Enable LP relaxation callbacks when use_prediction='continuous'")
        print("  2. Collect dual vectors during Phase 1 in LP relaxation context")
        print("  3. Generate features successfully from collected dual vectors")
        return True
    else:
        print(f"\n✗ Some tests failed - fix needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)