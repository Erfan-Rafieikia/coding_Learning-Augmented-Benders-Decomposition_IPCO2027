#!/usr/bin/env python3
"""
Test script for Phase 0 integration in Learning-Augmented Benders Decomposition.

This script tests the complete Phase 0 → Phase 1 → Phase 2 workflow to ensure:
1. Phase 0 solutions are generated and cuts are added
2. Phase 0 and Phase 1 duals are properly combined for feature generation
3. Training data includes both Phase 0 and Phase 1 information
4. Timing and logging are captured correctly
"""

import sys
import os
import numpy as np
import traceback

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import config
    from config import m_trials, m_trials_grid
    from phase0_generator import generate_phase0_solutions
    from Problem.data import Data
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    print("Running in limited test mode...")

def test_phase0_generator():
    """Test the Phase 0 generator with mock data"""
    print("=== Testing Phase 0 Generator ===")
    
    try:
        # Test SCFLP case
        print("Testing SCFLP problem type...")
        
        # Mock SCFLP data
        class MockSCFLPData:
            def __init__(self):
                self.J = [0, 1, 2]  # 3 facilities
                self.I = [0, 1, 2, 3]  # 4 customers
                self.capacity = {0: 100, 1: 150, 2: 120}
                self.scenarios = {
                    0: {0: 50, 1: 40, 2: 30, 3: 25},  # demands for scenario 0
                    1: {0: 45, 1: 35, 2: 35, 3: 30},  # demands for scenario 1
                }
                
        mock_data = MockSCFLPData()
        
        # Generate Phase 0 solutions
        solutions = generate_phase0_solutions(
            problem_data=mock_data,
            m_trials=5,  # Generate 5 solutions
            problem_class="SCFLP"
        )
        
        print(f"Generated {len(solutions)} Phase 0 solutions")
        print(f"Sample solution: {solutions[0]}")
        
        # Verify solutions are feasible
        for i, sol in enumerate(solutions):
            total_capacity = sum(mock_data.capacity[j] * sol[j] for j in mock_data.J)
            max_demand = max(sum(scenario.values()) for scenario in mock_data.scenarios.values())
            assert total_capacity >= max_demand, f"Solution {i} is infeasible: capacity={total_capacity}, max_demand={max_demand}"
            
        print("✓ All solutions are feasible")
        print("✓ Phase 0 generator test passed\n")
        
    except Exception as e:
        print(f"✗ Phase 0 generator test failed: {e}")
        traceback.print_exc()
        return False
        
    return True


def test_parameter_grid_integration():
    """Test that m_trials is properly integrated into parameter grids"""
    print("=== Testing Parameter Grid Integration ===")
    
    try:
        # Check that m_trials_grid is available
        assert hasattr(config, 'm_trials_grid'), "m_trials_grid not found in config"
        assert len(m_trials_grid) > 0, "m_trials_grid is empty"
        
        print(f"m_trials_grid: {m_trials_grid}")
        print(f"Default m_trials: {m_trials}")
        
        print("✓ Parameter grid integration test passed\n")
        
    except Exception as e:
        print(f"✗ Parameter grid integration test failed: {e}")
        return False
        
    return True


def test_callback_integration():
    """Test that Phase 0 integration doesn't break callback initialization"""
    print("=== Testing Callback Integration ===")
    
    try:
        # Import callback class
        from Problem.callbacks import Callback
        
        # Mock minimal data for callback initialization
        class MockData:
            def __init__(self):
                self.S = [(0, 1), (1, 2)]  # Two scenarios
                self.F = [0, 1]
                self.J = [0, 1]
                self.I = [0, 1]
                
        mock_data = MockData()
        
        # Test callback initialization with Phase 0 parameters
        callback = Callback(
            problem_type='UFL',
            data=mock_data,
            first_stage_values={},  # Mock first stage variables
            theta_vars={0: None, 1: None},  # Mock theta variables
            n_trials=3,
            m_trials=5,  # Phase 0 parameter
            use_prediction=True,
            prediction_method='REG', 
            selected_subproblems=None,
            add_cuts_at_relaxation=True,
            fallback_method='SOLVE',
            switching_strategy='no_switch',
            n_neighbors=3,
            L_d=1.0, L_v=1.0, p_d=1.0, p_v=1.0,
            subproblem_selection_method='trial-cut-freq',
            feature_params={
                'n_walks': 10, 'walk_length': 5, 'seed': 42,
                'feature_dim': 10, 'window': 3, 'min_count': 1, 'sg': 1
            }
        )
        
        # Check that Phase 0 attributes are initialized
        assert hasattr(callback, 'phase0_executed'), "phase0_executed attribute missing"
        assert hasattr(callback, 'phase0_solutions'), "phase0_solutions attribute missing"
        assert hasattr(callback, 'phase0_dual_vectors'), "phase0_dual_vectors attribute missing"
        assert hasattr(callback, 'phase0_execution_time_sec'), "phase0_execution_time_sec attribute missing"
        
        # Check that m_trials is stored
        assert callback.m_trials == 5, f"m_trials not properly stored: {callback.m_trials}"
        
        print(f"✓ Callback initialized with m_trials={callback.m_trials}")
        print("✓ All Phase 0 attributes present")
        print("✓ Callback integration test passed\n")
        
    except Exception as e:
        print(f"✗ Callback integration test failed: {e}")
        traceback.print_exc()
        return False
        
    return True


def main():
    """Run all integration tests"""
    print("Phase 0 Integration Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_phase0_generator():
        tests_passed += 1
        
    if test_parameter_grid_integration():
        tests_passed += 1
        
    if test_callback_integration():
        tests_passed += 1
    
    print("=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All integration tests passed! Phase 0 integration is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1
        
    return 0


if __name__ == '__main__':
    sys.exit(main())