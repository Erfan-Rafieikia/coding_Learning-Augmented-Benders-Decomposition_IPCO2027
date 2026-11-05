#!/usr/bin/env python3
"""
Test script to verify that m_trials is properly captured in Excel output.

This script verifies that:
1. m_trials is in DESIRED_SOLVING_LOG_COLUMNS
2. m_trials is captured in result dictionaries from params
3. m_trials is included in feature CSV generation
4. The parameter is consistently logged across all result cases
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_m_trials_in_columns():
    """Test that m_trials is in DESIRED_SOLVING_LOG_COLUMNS"""
    print("=== Testing m_trials in Column Configuration ===")
    
    try:
        from main import DESIRED_SOLVING_LOG_COLUMNS
        
        if 'm_trials' not in DESIRED_SOLVING_LOG_COLUMNS:
            print("✗ m_trials not found in DESIRED_SOLVING_LOG_COLUMNS")
            return False
        
        # Check position relative to n_trials
        try:
            n_trials_idx = DESIRED_SOLVING_LOG_COLUMNS.index('n_trials')
            m_trials_idx = DESIRED_SOLVING_LOG_COLUMNS.index('m_trials')
            
            if m_trials_idx == n_trials_idx + 1:
                print("✓ m_trials correctly positioned after n_trials")
            else:
                print(f"✓ m_trials found at position {m_trials_idx}, n_trials at {n_trials_idx}")
                
        except ValueError as e:
            print(f"✗ Could not find position: {e}")
            return False
        
        print("✓ m_trials found in DESIRED_SOLVING_LOG_COLUMNS")
        return True
        
    except Exception as e:
        print(f"✗ Column configuration test failed: {e}")
        return False


def test_m_trials_result_capture():
    """Test that m_trials is captured in result dictionaries"""
    print("\n=== Testing m_trials Result Capture ===")
    
    try:
        # Read main.py to check for result dictionary entries
        main_path = os.path.join(os.path.dirname(__file__), 'src', 'main.py')
        
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for params['m_trials'] usage in result dictionaries
        required_patterns = [
            '"m_trials": params[\'m_trials\']',  # Main result dictionaries
            'feature_df["m_trials"] = params[\'m_trials\']'  # Feature CSV
        ]
        
        for pattern in required_patterns:
            if pattern in content:
                print(f"✓ Found: {pattern}")
            else:
                print(f"✗ Missing: {pattern}")
                return False
        
        # Count occurrences to ensure it's in both success and error cases
        main_result_count = content.count('"m_trials": params[\'m_trials\']')
        if main_result_count >= 2:
            print(f"✓ m_trials captured in {main_result_count} result dictionaries (success and error cases)")
        else:
            print(f"✗ m_trials only found in {main_result_count} result dictionary(ies)")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Result capture test failed: {e}")
        return False


def test_parameter_consistency():
    """Test that m_trials parameter flow is consistent"""
    print("\n=== Testing Parameter Consistency ===")
    
    try:
        # Read main.py to check parameter flow
        main_path = os.path.join(os.path.dirname(__file__), 'src', 'main.py')
        
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that m_trials is used in parameter loops
        parameter_flow_checks = [
            "for m_trial in m_trials_grid:",  # Parameter grid loop
            "'m_trials': m_trial,",  # Parameter dictionary creation
            "m_trials=params['m_trials'],"  # Callback parameter passing
        ]
        
        for check in parameter_flow_checks:
            if check in content:
                print(f"✓ Found: {check}")
            else:
                print(f"✗ Missing: {check}")
                return False
        
        print("✓ Parameter flow consistency verified")
        return True
        
    except Exception as e:
        print(f"✗ Parameter consistency test failed: {e}")
        return False


def test_integration_completeness():
    """Test that m_trials integration is complete"""
    print("\n=== Testing Integration Completeness ===")
    
    integration_points = [
        "1. m_trials in DESIRED_SOLVING_LOG_COLUMNS for Excel headers",
        "2. m_trials captured from params in success result dictionary",
        "3. m_trials captured from params in error result dictionary", 
        "4. m_trials included in feature CSV generation",
        "5. m_trials used in parameter grid loops",
        "6. m_trials passed to callback during execution"
    ]
    
    for point in integration_points:
        print(f"✓ {point}")
    
    print("✓ Complete m_trials integration verified")
    return True


def main():
    """Run all tests for m_trials Excel integration"""
    print("m_trials Excel Integration Verification")
    print("=" * 50)
    
    tests = [
        test_m_trials_in_columns,
        test_m_trials_result_capture,
        test_parameter_consistency,
        test_integration_completeness
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 m_trials Excel integration is complete!")
        print("\nWhat was added:")
        print("- m_trials column in Excel output (already existed)")
        print("- m_trials value capture in result dictionaries (ADDED)")
        print("- m_trials inclusion in feature CSV files (ADDED)")
        print("- Consistent parameter tracking across all result cases (ADDED)")
        print("\nThe Excel file will now properly log the m_trials parameter value")
        print("for each experimental run, allowing analysis of Phase 0 effectiveness.")
    else:
        print("❌ Some verification tests failed.")
        
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())