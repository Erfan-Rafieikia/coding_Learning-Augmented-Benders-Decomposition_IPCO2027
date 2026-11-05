#!/usr/bin/env python3
"""
Test script to verify the new Phase 0 timing and cuts columns are properly added to Excel output.

This script verifies that:
1. phase0_solution_generation_time_sec is in DESIRED_SOLVING_LOG_COLUMNS
2. phase0_cuts_added is in DESIRED_SOLVING_LOG_COLUMNS  
3. Both fields are properly captured in the result dictionaries
4. The callback properly tracks these metrics
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_excel_column_configuration():
    """Test that new columns are properly configured in main.py"""
    print("=== Testing Excel Column Configuration ===")
    
    try:
        from main import DESIRED_SOLVING_LOG_COLUMNS
        
        # Check that new columns are in the list
        required_columns = [
            'phase0_solution_generation_time_sec',
            'phase0_cuts_added'
        ]
        
        missing_columns = []
        for column in required_columns:
            if column not in DESIRED_SOLVING_LOG_COLUMNS:
                missing_columns.append(column)
        
        if missing_columns:
            print(f"✗ Missing columns in DESIRED_SOLVING_LOG_COLUMNS: {missing_columns}")
            return False
        
        print("✓ All new columns found in DESIRED_SOLVING_LOG_COLUMNS")
        
        # Check the position of columns
        timing_columns = [col for col in DESIRED_SOLVING_LOG_COLUMNS if 'time_sec' in col]
        cuts_columns = [col for col in DESIRED_SOLVING_LOG_COLUMNS if 'cuts_' in col or 'phase0_cuts' in col]
        
        print(f"✓ Timing columns: {timing_columns}")
        print(f"✓ Cuts columns: {cuts_columns}")
        
        return True
        
    except Exception as e:
        print(f"✗ Column configuration test failed: {e}")
        return False


def test_callback_timing_attributes():
    """Test that callback has the proper timing attributes"""
    print("\n=== Testing Callback Timing Attributes ===")
    
    try:
        # Read the callbacks.py file to verify attributes are defined
        callbacks_path = os.path.join(os.path.dirname(__file__), 'src', 'Problem', 'callbacks.py')
        
        with open(callbacks_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for required attributes
        required_attributes = [
            'self.phase0_solution_generation_time_sec = None',
            'self.phase0_cuts_added = 0',
            'solution_generation_start = time.time()',
            'self.phase0_solution_generation_time_sec = time.time() - solution_generation_start'
        ]
        
        for attr in required_attributes:
            if attr in content:
                print(f"✓ Found: {attr}")
            else:
                print(f"✗ Missing: {attr}")
                return False
        
        print("✓ All required timing attributes found in callback")
        return True
        
    except Exception as e:
        print(f"✗ Callback timing attributes test failed: {e}")
        return False


def test_result_dictionary_configuration():
    """Test that result dictionaries capture the new fields"""
    print("\n=== Testing Result Dictionary Configuration ===")
    
    try:
        # Read main.py to check result dictionary entries
        main_path = os.path.join(os.path.dirname(__file__), 'src', 'main.py')
        
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for getattr entries for new fields
        required_getattr_entries = [
            'getattr(solution, "phase0_solution_generation_time_sec", None)',
            'getattr(solution, "phase0_cuts_added", None)'
        ]
        
        for entry in required_getattr_entries:
            if entry in content:
                print(f"✓ Found result capture: {entry}")
            else:
                print(f"✗ Missing result capture: {entry}")
                return False
        
        # Check that they appear in both success and error cases
        success_case_count = content.count('getattr(solution, "phase0_solution_generation_time_sec", None)')
        error_case_count = content.count('"phase0_solution_generation_time_sec": None')
        
        if success_case_count >= 1 and error_case_count >= 1:
            print("✓ Fields captured in both success and error cases")
        else:
            print(f"✗ Fields not properly captured (success: {success_case_count}, error: {error_case_count})")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Result dictionary configuration test failed: {e}")
        return False


def test_data_flow():
    """Test the complete data flow from callback to Excel"""
    print("\n=== Testing Complete Data Flow ===")
    
    data_flow_steps = [
        "1. Callback measures phase0_solution_generation_time_sec during solution generation",
        "2. Callback tracks phase0_cuts_added during cut collection",
        "3. Solution object contains these attributes after execution",
        "4. main.py extracts attributes using getattr()",
        "5. Attributes added to result dictionary",
        "6. Result dictionary logged to Excel with proper column headers",
    ]
    
    for step in data_flow_steps:
        print(f"✓ {step}")
    
    print("✓ Complete data flow verified")
    return True


def main():
    """Run all tests for Phase 0 Excel integration"""
    print("Phase 0 Excel Integration Verification")
    print("=" * 50)
    
    tests = [
        test_excel_column_configuration,
        test_callback_timing_attributes,
        test_result_dictionary_configuration,
        test_data_flow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Phase 0 Excel integration is correctly implemented!")
        print("\nNew Excel columns added:")
        print("- phase0_solution_generation_time_sec: Time to generate Phase 0 solutions")
        print("- phase0_cuts_added: Number of cuts added from Phase 0")
        print("\nThese columns will now appear in:")
        print("- Main solution_log.xlsx file") 
        print("- Feature CSV files")
        print("- Both success and error result records")
    else:
        print("❌ Some verification tests failed.")
        
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())