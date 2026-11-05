"""
Test script to validate the new fine-grained use_prediction control implementation.

This script tests the backward compatibility and new functionality of the enhanced
use_prediction parameter that now supports context-aware ML control.
"""

import sys
import os

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from utils.config_utils import (
    normalize_use_prediction, 
    ml_enabled_for_context, 
    validate_config_compatibility,
    should_skip_config
)

def test_backward_compatibility():
    """Test that old boolean values are correctly converted."""
    print("Testing backward compatibility...")
    
    # Test boolean to string conversion
    assert normalize_use_prediction(True) == "both"
    assert normalize_use_prediction(False) == "none"
    
    # Test string values pass through
    assert normalize_use_prediction("none") == "none"
    assert normalize_use_prediction("integer") == "integer" 
    assert normalize_use_prediction("continuous") == "continuous"
    assert normalize_use_prediction("both") == "both"
    
    print("✓ Backward compatibility tests passed")

def test_context_logic():
    """Test the context-aware ML enablement logic."""
    print("Testing context-aware ML logic...")
    
    # Test "none" - no ML anywhere
    assert not ml_enabled_for_context("none", True)   # integer context
    assert not ml_enabled_for_context("none", False)  # continuous context
    
    # Test "both" - ML everywhere  
    assert ml_enabled_for_context("both", True)       # integer context
    assert ml_enabled_for_context("both", False)      # continuous context
    
    # Test "integer" - ML only at integer solutions
    assert ml_enabled_for_context("integer", True)    # integer context
    assert not ml_enabled_for_context("integer", False) # continuous context
    
    # Test "continuous" - ML only at continuous solutions
    assert not ml_enabled_for_context("continuous", True)  # integer context
    assert ml_enabled_for_context("continuous", False)     # continuous context
    
    # Test with legacy boolean values
    assert ml_enabled_for_context(True, True)    # True -> "both"
    assert ml_enabled_for_context(True, False)   # True -> "both"
    assert not ml_enabled_for_context(False, True)  # False -> "none"
    assert not ml_enabled_for_context(False, False) # False -> "none"
    
    print("✓ Context-aware ML logic tests passed")

def test_config_validation():
    """Test configuration validation logic.""" 
    print("Testing configuration validation...")
    
    # Valid configurations
    assert validate_config_compatibility("none", True) == (True, "")
    assert validate_config_compatibility("none", False) == (True, "")
    assert validate_config_compatibility("both", True) == (True, "")
    assert validate_config_compatibility("integer", True) == (True, "")
    assert validate_config_compatibility("integer", False) == (True, "")
    assert validate_config_compatibility("continuous", True) == (True, "")
    
    # Invalid configurations
    is_valid, reason = validate_config_compatibility("continuous", False)
    assert not is_valid
    assert "no continuous callbacks exist" in reason
    
    is_valid, reason = validate_config_compatibility("both", False)
    assert not is_valid
    assert "no continuous callbacks exist" in reason
    
    print("✓ Configuration validation tests passed")

def test_skip_config():
    """Test the parameter skipping logic."""
    print("Testing configuration skip logic...")
    
    # Valid configs should not be skipped
    assert not should_skip_config({'use_prediction': 'none', 'add_cuts_at_relaxation': True})[0]
    assert not should_skip_config({'use_prediction': 'integer', 'add_cuts_at_relaxation': False})[0]
    assert not should_skip_config({'use_prediction': True, 'add_cuts_at_relaxation': True})[0]  # legacy
    assert not should_skip_config({'use_prediction': False, 'add_cuts_at_relaxation': False})[0]  # legacy
    
    # Invalid configs should be skipped
    should_skip, reason = should_skip_config({'use_prediction': 'continuous', 'add_cuts_at_relaxation': False})
    assert should_skip
    assert "continuous callbacks exist" in reason
    
    should_skip, reason = should_skip_config({'use_prediction': 'both', 'add_cuts_at_relaxation': False})
    assert should_skip
    assert "continuous callbacks exist" in reason
    
    print("✓ Configuration skip logic tests passed")

def test_error_handling():
    """Test error handling for invalid inputs."""
    print("Testing error handling...")
    
    try:
        normalize_use_prediction("invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "use_prediction must be one of" in str(e)
    
    try:
        normalize_use_prediction(123)  # Invalid type
        assert False, "Should have raised ValueError" 
    except ValueError as e:
        assert "must be bool or str" in str(e)
    
    print("✓ Error handling tests passed")

if __name__ == "__main__":
    print("Running fine-grained use_prediction validation tests...")
    print("=" * 60)
    
    test_backward_compatibility()
    test_context_logic()
    test_config_validation()
    test_skip_config()
    test_error_handling()
    
    print("=" * 60)
    print("🎉 All tests passed! The fine-grained use_prediction implementation is working correctly.")
    print()
    print("Summary of new functionality:")
    print("- 'none': No ML anywhere (traditional Benders)")
    print("- 'integer': ML only at integer incumbent solutions")
    print("- 'continuous': ML only at LP-relaxation solutions")
    print("- 'both': ML at both integer and continuous solutions")
    print("- Backward compatibility: True -> 'both', False -> 'none'")
    print("- Invalid combinations are automatically skipped with clear logging")