"""
Test module for stopping reason tracking functionality.

This module tests the stopping reason detection and classification system
that tracks Gurobi termination conditions and code errors.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestStoppingReasonDetection:
    """Test suite for stopping reason detection functionality."""
    
    def test_stopping_reason_function_import(self):
        """Test that the stopping reason function can be imported."""
        try:
            from Problem.master_problem import _determine_stopping_reason
            assert callable(_determine_stopping_reason)
        except ImportError:
            pytest.skip("Gurobi not available in test environment")
    
    def test_stopping_reason_mapping(self):
        """Test stopping reason status code mapping."""
        try:
            from Problem.master_problem import _determine_stopping_reason
            from gurobipy import GRB
            
            # Test known status codes
            test_cases = [
                (GRB.OPTIMAL, "OPTIMAL"),
                (GRB.INFEASIBLE, "INFEASIBLE"),
                (GRB.UNBOUNDED, "UNBOUNDED"),
                (GRB.TIME_LIMIT, "TIME_LIMIT"),
                (GRB.NODE_LIMIT, "NODE_LIMIT"),
                (GRB.ITERATION_LIMIT, "ITERATION_LIMIT"),
                (GRB.SOLUTION_LIMIT, "SOLUTION_LIMIT"),
                (GRB.INTERRUPTED, "USER_INTERRUPT"),
                (GRB.NUMERIC, "NUMERIC_ERROR"),
                (GRB.SUBOPTIMAL, "GAP_TOLERANCE"),
            ]
            
            for status, expected in test_cases:
                result = _determine_stopping_reason(status)
                assert result == expected, f"Status {status} should map to {expected}, got {result}"
                
        except ImportError:
            pytest.skip("Gurobi not available in test environment")
    
    def test_unknown_status_handling(self):
        """Test handling of unknown status codes."""
        try:
            from Problem.master_problem import _determine_stopping_reason
            
            # Test with an unknown status code
            result = _determine_stopping_reason(999)
            assert result.startswith("UNKNOWN"), f"Unknown status should start with UNKNOWN, got {result}"
            
        except ImportError:
            pytest.skip("Gurobi not available in test environment")


class TestSolutionDataclass:
    """Test suite for Solution dataclass enhancements."""
    
    def test_solution_has_stopping_reason_field(self):
        """Test that Solution dataclass has stopping_reason field."""
        try:
            from Problem.master_problem import Solution
            
            # Create minimal solution object
            solution = Solution(
                objective_value=100.0,
                first_stage_optimnal_values={},
                solution_time=5.0,
                feature_gen_time_sec=1.0,
                subproblem_selection_time_sec=0.5,
                predictor_training_time=2.0,
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
                iteration=5
            )
            
            # Test stopping_reason field exists and has default value
            assert hasattr(solution, 'stopping_reason')
            assert solution.stopping_reason == "OPTIMAL"
            
        except ImportError:
            pytest.skip("Dependencies not available in test environment")
    
    def test_solution_stopping_reason_assignment(self):
        """Test that stopping_reason can be assigned custom values."""
        try:
            from Problem.master_problem import Solution
            
            # Create solution with custom stopping reason
            solution = Solution(
                objective_value=100.0,
                first_stage_optimnal_values={},
                solution_time=5.0,
                feature_gen_time_sec=1.0,
                subproblem_selection_time_sec=0.5,
                predictor_training_time=2.0,
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
                stopping_reason="TIME_LIMIT"
            )
            
            assert solution.stopping_reason == "TIME_LIMIT"
            
        except ImportError:
            pytest.skip("Dependencies not available in test environment")


if __name__ == "__main__":
    pytest.main([__file__])
