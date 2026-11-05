#!/usr/bin/env python3
"""
Quick test script to verify the SCFLP objective value fix.
"""

import sys
import os
sys.path.insert(0, '/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src')

from Problem.master_problem import solve_master_problem
from Problem.data import SCFLPData
import logging

# Configure logging to see the objective values
logging.basicConfig(level=logging.INFO)

def test_scflp_objective():
    """Test that objective_value and original_Prob_Obj_Value are consistent after the fix."""
    
    # Load a small SCFLP instance
    data_path = "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP"
    instance_file = "scenarios_s100_cv0.1_distnormal/p5_stochastic.txt"
    
    try:
        data = SCFLPData.read_from_file(os.path.join(data_path, instance_file))
        print(f"Loaded SCFLP instance with {len(data.S)} scenarios")
        
        # Run with minimal parameters to speed up the test
        solution = solve_master_problem(
            problem_type="SCFLP",
            data=data,
            selected_subproblems=None,
            feature_vectors=None,
            n_neighbors=3,
            prediction_method='SIMPLE_AVG',
            use_prediction=True,  # This enables original_Prob_Obj_Value calculation
            n_trials=2,  # Minimal trials
            m_trials=0,  # No phase 0
            add_cuts_at_relaxation=False
        )
        
        if solution:
            print(f"\nRESULTS:")
            print(f"objective_value: {solution.objective_value}")
            print(f"original_Prob_Obj_Value: {solution.original_Prob_Obj_Value}")
            
            if solution.original_Prob_Obj_Value is not None:
                ratio = solution.objective_value / solution.original_Prob_Obj_Value
                print(f"Ratio (should be close to 1.0): {ratio:.6f}")
                
                if 0.95 <= ratio <= 1.05:
                    print("✅ PASS: Objective values are consistent!")
                else:
                    print("❌ FAIL: Objective values are still inconsistent!")
            else:
                print("❌ FAIL: original_Prob_Obj_Value is None")
        else:
            print("❌ FAIL: Solution is None")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scflp_objective()