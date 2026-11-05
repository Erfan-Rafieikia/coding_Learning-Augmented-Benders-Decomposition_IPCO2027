"""
Test script for SUMND (Stochastic Uncapacitated Multicommodity Network Design) integration.

This script verifies that the SUMND problem class has been properly integrated into
the ML-augmented Benders decomposition framework.

Tests:
1. Data loading from JSON files
2. Dual subproblem solver
3. Master problem formulation (basic structure check)
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Problem.data import read_sumnd_data
from Problem.subproblem_dual import solve_dual_subproblem

def test_data_loading():
    """Test that SUMND data can be loaded from JSON file."""
    print("=" * 70)
    print("TEST 1: Data Loading")
    print("=" * 70)
    
    # Path to a small test instance
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '../DATA/SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r01.1.stoch.json'
    )
    
    if not os.path.exists(data_path):
        print(f"ERROR: Test file not found: {data_path}")
        return False
    
    try:
        data = read_sumnd_data(data_path)
        
        print(f"✓ Successfully loaded SUMND data")
        print(f"  - Nodes: {data.n_nodes}")
        print(f"  - Arcs: {data.n_arcs}")
        print(f"  - Commodities: {data.n_commodities}")
        print(f"  - Scenarios: {data.n_scenarios}")
        print(f"  - Covariates: {data.num_covariates}")
        print(f"  - Generation method: {data.generation_method}")
        
        # Verify data structures
        assert len(data.A) == data.n_arcs, "Arc count mismatch"
        assert len(data.F) == data.n_arcs, "Fixed cost array size mismatch"
        assert data.C.shape == (data.n_arcs, data.n_commodities), "Cost matrix shape mismatch"
        assert len(data.origins) == data.n_commodities, "Origins array size mismatch"
        assert len(data.destinations) == data.n_commodities, "Destinations array size mismatch"
        assert data.demand_scenarios.shape == (data.n_scenarios, data.n_commodities), "Demand scenarios shape mismatch"
        assert data.capacity_scenarios.shape == (data.n_scenarios, data.n_arcs), "Capacity scenarios shape mismatch"
        assert len(data.probabilities) == data.n_scenarios, "Probabilities array size mismatch"
        
        print("✓ All data structure checks passed")
        return True, data
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_dual_subproblem(data):
    """Test that dual subproblems can be solved."""
    print("\n" + "=" * 70)
    print("TEST 2: Dual Subproblem Solver")
    print("=" * 70)
    
    try:
        # Create a simple first-stage solution (activate all arcs)
        first_stage_sol = {idx: 1.0 for idx in range(len(data.A))}
        
        # Solve scenario 0
        scenario_idx = 0
        obj_val, pi_vals, alpha_vals = solve_dual_subproblem(
            "SUMND", data, scenario_idx, first_stage_sol
        )
        
        print(f"✓ Successfully solved dual subproblem for scenario {scenario_idx}")
        print(f"  - Objective value: {obj_val:.2f}")
        print(f"  - Number of π variables: {len(pi_vals)}")
        print(f"  - Number of α variables: {len(alpha_vals)}")
        
        # Verify dual variable structure
        assert len(pi_vals) == data.n_nodes * data.n_commodities, "π variables count mismatch"
        assert len(alpha_vals) == data.n_arcs, "α variables count mismatch"
        
        # Verify all α values are non-negative (dual feasibility)
        for idx, val in alpha_vals.items():
            assert val >= -1e-6, f"α[{idx}] = {val} is negative (dual infeasibility)"
        
        print("✓ Dual solution structure verified")
        print("✓ Dual feasibility (α ≥ 0) verified")
        
        # Test with a different first-stage solution (activate only first 10 arcs)
        first_stage_sol_sparse = {idx: 1.0 if idx < 10 else 0.0 for idx in range(len(data.A))}
        obj_val2, _, _ = solve_dual_subproblem(
            "SUMND", data, scenario_idx, first_stage_sol_sparse
        )
        
        print(f"✓ Successfully solved with sparse solution")
        print(f"  - Objective value (sparse): {obj_val2:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dual subproblem solver failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_master_problem_structure(data):
    """Test that master problem can be formulated (structure only, not solved)."""
    print("\n" + "=" * 70)
    print("TEST 3: Master Problem Structure")
    print("=" * 70)
    
    try:
        from gurobipy import Model, GRB
        
        # Create a simple master problem structure
        with Model("SUMND_Test_Master") as m:
            m.Params.OutputFlag = 0
            
            # First-stage variables (arc activation)
            y = m.addVars(len(data.A), vtype=GRB.BINARY, name="y")
            
            # Second-stage variables (theta for each scenario)
            theta = m.addVars(data.S, lb=0.0, name="theta")
            
            # Objective
            activation_cost = sum(data.F[idx] * y[idx] for idx in range(len(data.A)))
            recourse_cost = sum(data.probabilities[s] * theta[s] for s in data.S)
            m.setObjective(activation_cost + recourse_cost, GRB.MINIMIZE)
            
            # At least one arc constraint
            m.addConstr(sum(y[idx] for idx in range(len(data.A))) >= 1)
            
            print(f"✓ Master problem formulated successfully")
            print(f"  - First-stage variables (y): {len(data.A)}")
            print(f"  - Second-stage variables (θ): {len(data.S)}")
            print(f"  - Constraints: {m.NumConstrs}")
            
        return True
        
    except Exception as e:
        print(f"✗ Master problem formulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SUMND INTEGRATION TEST SUITE")
    print("=" * 70)
    
    # Test 1: Data loading
    success_data, data = test_data_loading()
    if not success_data:
        print("\n✗ FAILED: Data loading test failed. Cannot proceed.")
        return False
    
    # Test 2: Dual subproblem
    success_dual = test_dual_subproblem(data)
    
    # Test 3: Master problem structure
    success_master = test_master_problem_structure(data)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Data Loading:       {'✓ PASS' if success_data else '✗ FAIL'}")
    print(f"Dual Subproblem:    {'✓ PASS' if success_dual else '✗ FAIL'}")
    print(f"Master Problem:     {'✓ PASS' if success_master else '✗ FAIL'}")
    print("=" * 70)
    
    all_passed = success_data and success_dual and success_master
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nSUMND integration is working correctly.")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease review the error messages above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
