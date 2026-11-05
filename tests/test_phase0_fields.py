#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, 'src')

try:
    from Problem.master_problem import Solution
    
    # Test creating a Solution object with Phase 0 timing fields
    solution = Solution(
        objective_value=100.0,
        first_stage_optimnal_values={},
        solution_time=1.0,
        feature_gen_time_sec=0.5,
        subproblem_selection_time_sec=0.2,
        predictor_training_time=0.3,
        KNN_closest_neighbor_time=0.1,
        num_cuts_mip_selected={},
        num_cuts_rel_selected={},
        num_cuts_mip_ml={},
        num_cuts_rel_ml={},
        num_cuts_mip_unselected={},
        num_cuts_rel_unselected={},
        num_cuts_mip_fallback={},
        num_cuts_rel_fallback={},
        num_cuts_mip_trial={},
        num_cuts_rel_trial={},
        num_cuts_mip_exact_switch={},
        num_cuts_rel_exact_switch={},
        feature_vectors={},
        iteration=5,
        phase0_execution_time_sec=2.5,
        phase0_solution_generation_time_sec=1.8,
        phase0_cuts_added=15
    )
    
    print("SUCCESS: Solution object created with Phase 0 timing fields")
    print(f"Phase 0 execution time: {solution.phase0_execution_time_sec}")
    print(f"Phase 0 solution generation time: {solution.phase0_solution_generation_time_sec}")
    print(f"Phase 0 cuts added: {solution.phase0_cuts_added}")
    
    # Test accessing the new fields
    if hasattr(solution, 'phase0_execution_time_sec'):
        print("✓ phase0_execution_time_sec field exists")
    if hasattr(solution, 'phase0_solution_generation_time_sec'):
        print("✓ phase0_solution_generation_time_sec field exists")
    if hasattr(solution, 'phase0_cuts_added'):
        print("✓ phase0_cuts_added field exists")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()