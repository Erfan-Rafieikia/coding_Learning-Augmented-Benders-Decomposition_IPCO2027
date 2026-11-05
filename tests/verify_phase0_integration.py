#!/usr/bin/env python3
"""
Simple Phase 0 integration verification script.

This script verifies that the Phase 0 integration doesn't break the codebase
and that key components can be imported and initialized.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all Phase 0 related modules can be imported"""
    print("=== Testing Imports ===")
    
    try:
        # Test config import
        import config
        print(f"✓ Config imported: m_trials={config.m_trials}, m_trials_grid={config.m_trials_grid}")
        
        # Test phase0_generator import
        from phase0_generator import generate_phase0_solutions
        print("✓ Phase 0 generator imported")
        
        # Test callbacks import
        from Problem.callbacks import Callback
        print("✓ Callback class imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic Phase 0 functionality without complex setup"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Test that phase0_generator function accepts correct parameters
        from phase0_generator import generate_phase0_solutions
        import inspect
        sig = inspect.signature(generate_phase0_solutions)
        params = list(sig.parameters.keys())
        expected_params = ['problem_data', 'm_trials', 'problem_class']
        
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"
        
        print(f"✓ Phase 0 generator has correct signature: {params}")
        
        # Test that Callback accepts m_trials parameter
        from Problem.callbacks import Callback
        sig = inspect.signature(Callback.__init__)
        params = list(sig.parameters.keys())
        
        assert 'm_trials' in params, "m_trials parameter missing from Callback.__init__"
        print("✓ Callback constructor accepts m_trials parameter")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_parameter_grid():
    """Test that parameter grids include Phase 0 parameters"""
    print("\n=== Testing Parameter Grid ===")
    
    try:
        import config
        
        # Check m_trials_grid exists and has values
        assert hasattr(config, 'm_trials_grid'), "m_trials_grid not found"
        assert len(config.m_trials_grid) > 0, "m_trials_grid is empty"
        assert 0 in config.m_trials_grid, "m_trials_grid should include 0 (no Phase 0)"
        
        print(f"✓ m_trials_grid: {config.m_trials_grid}")
        print(f"✓ Default m_trials: {config.m_trials}")
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter grid test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist"""
    print("\n=== Testing File Structure ===")
    
    try:
        required_files = [
            'src/phase0_generator.py',
            'src/config.py', 
            'src/Problem/callbacks.py',
            'src/main.py'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            assert os.path.exists(full_path), f"Missing file: {file_path}"
            
        print("✓ All required files exist")
        
        # Test that new Phase 0 timing field is in main.py
        main_path = os.path.join(os.path.dirname(__file__), 'src/main.py')
        with open(main_path, 'r') as f:
            main_content = f.read()
            
        assert 'phase0_execution_time_sec' in main_content, "phase0_execution_time_sec not found in main.py"
        print("✓ Phase 0 timing field found in main.py")
        
        return True
        
    except Exception as e:
        print(f"✗ File structure test failed: {e}")
        return False


def main():
    """Run all verification tests"""
    print("Phase 0 Integration Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality, 
        test_parameter_grid,
        test_file_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Phase 0 integration verification successful!")
        print("\nNext steps:")
        print("1. Run a small experiment with m_trials > 0 to test end-to-end")
        print("2. Check Phase 0 execution logs and timing")
        print("3. Verify that Phase 0 + Phase 1 duals are combined in features")
    else:
        print("❌ Some verification tests failed.")
        
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())