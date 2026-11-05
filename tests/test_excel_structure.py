"""
Test module for Excel column structure and data export functionality.

This module tests the Excel output structure, column ordering, and
data logging functionality for the ML-augmented Benders framework.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestExcelStructure:
    """Test suite for Excel output structure and columns."""
    
    def test_desired_columns_import(self):
        """Test that DESIRED_SOLVING_LOG_COLUMNS can be imported."""
        try:
            from main import DESIRED_SOLVING_LOG_COLUMNS
            assert isinstance(DESIRED_SOLVING_LOG_COLUMNS, list)
            assert len(DESIRED_SOLVING_LOG_COLUMNS) > 0
        except ImportError:
            pytest.skip("Main module dependencies not available")
    
    def test_stopping_reason_column_exists(self):
        """Test that stopping_reason column is in the column list."""
        try:
            from main import DESIRED_SOLVING_LOG_COLUMNS
            assert "stopping_reason" in DESIRED_SOLVING_LOG_COLUMNS
        except ImportError:
            pytest.skip("Main module dependencies not available")
    
    def test_time_limit_column_exists(self):
        """Test that time_limit_sec column is in the column list."""
        try:
            from main import DESIRED_SOLVING_LOG_COLUMNS
            assert "time_limit_sec" in DESIRED_SOLVING_LOG_COLUMNS
        except ImportError:
            pytest.skip("Main module dependencies not available")
    
    def test_column_ordering(self):
        """Test that columns are in the correct order."""
        try:
            from main import DESIRED_SOLVING_LOG_COLUMNS
            
            # Check that time_limit_sec comes before stopping_reason
            time_limit_idx = DESIRED_SOLVING_LOG_COLUMNS.index("time_limit_sec")
            stopping_reason_idx = DESIRED_SOLVING_LOG_COLUMNS.index("stopping_reason")
            
            assert time_limit_idx < stopping_reason_idx, \
                "time_limit_sec should come before stopping_reason"
            
            # Check that stopping_reason comes before p_d and p_v
            if "p_d" in DESIRED_SOLVING_LOG_COLUMNS:
                p_d_idx = DESIRED_SOLVING_LOG_COLUMNS.index("p_d")
                assert stopping_reason_idx < p_d_idx, \
                    "stopping_reason should come before p_d"
            
            if "p_v" in DESIRED_SOLVING_LOG_COLUMNS:
                p_v_idx = DESIRED_SOLVING_LOG_COLUMNS.index("p_v")
                assert stopping_reason_idx < p_v_idx, \
                    "stopping_reason should come before p_v"
                    
        except ImportError:
            pytest.skip("Main module dependencies not available")
    
    def test_essential_columns_present(self):
        """Test that all essential columns are present."""
        try:
            from main import DESIRED_SOLVING_LOG_COLUMNS
            
            essential_columns = [
                "problem_class",
                "instance_filename", 
                "objective_value",
                "solution_time_sec",
                "stopping_reason",
                "time_limit_sec"
            ]
            
            for col in essential_columns:
                assert col in DESIRED_SOLVING_LOG_COLUMNS, \
                    f"Essential column '{col}' missing from DESIRED_SOLVING_LOG_COLUMNS"
                    
        except ImportError:
            pytest.skip("Main module dependencies not available")


class TestTimeLimitConfiguration:
    """Test suite for time limit configuration."""
    
    def test_gurobi_time_limit_setting(self):
        """Test that Gurobi TimeLimit parameter is set correctly."""
        # This test reads the source code to verify the parameter is set
        try:
            master_problem_path = os.path.join(
                os.path.dirname(__file__), '..', 'src', 'Problem', 'master_problem.py'
            )
            
            with open(master_problem_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for TimeLimit parameter setting
            assert 'mod.Params.TimeLimit = 3600' in content, \
                "TimeLimit parameter should be set to 3600 seconds"
                
        except FileNotFoundError:
            pytest.skip("Master problem file not found")
    
    def test_time_limit_value_consistency(self):
        """Test that time limit values are consistent across codebase."""
        try:
            # Check main.py for consistent time limit value
            main_path = os.path.join(
                os.path.dirname(__file__), '..', 'src', 'main.py'
            )
            
            with open(main_path, 'r') as f:
                content = f.read()
            
            # Should have consistent 3600 value in log entries
            time_limit_count = content.count('"time_limit_sec": 3600')
            assert time_limit_count >= 2, \
                "time_limit_sec should be set in both success and failure log entries"
                
        except FileNotFoundError:
            pytest.skip("Main file not found")


if __name__ == "__main__":
    pytest.main([__file__])
