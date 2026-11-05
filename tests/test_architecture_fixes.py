"""
Test script to validate architectural fixes for method_config system.

This test validates:
1. Callback type name mapping (canonical vs legacy)
2. Legacy string consistency (int_and_LP_ML_for_LP vs int_and_LP_ML_for_relax)
3. Phase 0 gating logic (should_execute_step vs should_execute_phase0)
4. Method config validation

Run with: python tests/test_architecture_fixes.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_utils import (
    should_execute_step,
    should_execute_phase0,
    should_register_lp_callbacks,
    should_use_ml_for_unselected,
    validate_method_config,
    legacy_use_prediction_to_new_config
)

def test_callback_type_mapping():
    """Test Issue #1: Verify canonical ml_callback_type names work correctly"""
    print("\n" + "="*70)
    print("TEST 1: Callback Type Name Mapping")
    print("="*70)
    
    test_cases = [
        ('ML_at_both', True, True),  # ML at both integer and LP
        ('ML_at_integer_only', False, True),  # ML at integer only, no LP callbacks
        ('ML_at_integer_exact_at_LP', True, True),  # ML at integer, exact at LP
        ('ML_at_LP_exact_at_integer', True, False),  # Exact at integer, ML at LP
    ]
    
    passed = 0
    failed = 0
    
    for ml_callback_type, expect_lp_callbacks, expect_ml_at_int in test_cases:
        method_config = {
            'method_category': 'ML',
            'ml_method_type': 'ML_Augmented_BD',
            'ml_callback_type': ml_callback_type,
            'predictor_type': 'KNN',
            'n_trials': 5
        }
        
        # Test LP callback registration
        has_lp_callbacks = should_register_lp_callbacks(method_config)
        # Test ML usage at integer
        uses_ml_at_int = should_use_ml_for_unselected(method_config, at_integer_solution=True)
        
        if has_lp_callbacks == expect_lp_callbacks and uses_ml_at_int == expect_ml_at_int:
            print(f"✓ {ml_callback_type}: LP={has_lp_callbacks}, ML@int={uses_ml_at_int}")
            passed += 1
        else:
            print(f"✗ {ml_callback_type}: Expected LP={expect_lp_callbacks}, ML@int={expect_ml_at_int}, "
                  f"Got LP={has_lp_callbacks}, ML@int={uses_ml_at_int}")
            failed += 1
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def test_legacy_string_consistency():
    """Test Issue #2: Verify int_and_LP_ML_for_LP is standard name"""
    print("\n" + "="*70)
    print("TEST 2: Legacy String Consistency")
    print("="*70)
    
    # Test that canonical name is used in compatibility mapping
    legacy_map = legacy_use_prediction_to_new_config('int_and_LP_ML_for_relax', 'KNN')
    
    if 'ml_callback_type' in legacy_map:
        callback_type = legacy_map['ml_callback_type']
        expected = 'ML_at_LP_exact_at_integer'  # This should map to the correct canonical name
        
        if callback_type == expected:
            print(f"✓ Legacy 'int_and_LP_ML_for_relax' correctly maps to '{expected}'")
            return True
        else:
            print(f"✗ Legacy mapping incorrect: got '{callback_type}', expected '{expected}'")
            return False
    else:
        print("✗ ml_callback_type not in legacy mapping")
        return False


def test_phase0_gating():
    """Test Issue #3: Verify Phase 0 executes only for correct methods"""
    print("\n" + "="*70)
    print("TEST 3: Phase 0 Gating Logic")
    print("="*70)
    
    test_cases = [
        # (ml_method_type, m_trials, should_execute_phase0)
        ('ML_Augmented_BD', 5, True),
        ('ML_Augmented_BD_with_duals', 5, True),
        ('Covariate_Similarity', 5, False),  # Should NOT execute Phase 0
        ('Uncertainty_Similarity', 5, False),  # Should NOT execute Phase 0
        ('Random_Sampling', 5, False),  # Should NOT execute Phase 0
        ('ML_Augmented_BD', 0, False),  # m_trials=0 means no Phase 0
    ]
    
    passed = 0
    failed = 0
    
    for ml_type, m_trials, expect_phase0 in test_cases:
        method_config = {
            'method_category': 'ML',
            'ml_method_type': ml_type,
            'ml_callback_type': 'ML_at_both',
            'predictor_type': 'KNN' if ml_type != 'Random_Sampling' else None,
            'n_trials': 5,
            'm_trials': m_trials
        }
        
        # Use step-based check (correct way)
        step_based = should_execute_step(method_config, 1)
        
        # Use category-based check (deprecated way - for comparison)
        category_based = should_execute_phase0(ml_type)
        
        if step_based == expect_phase0:
            status = "✓"
            passed += 1
        else:
            status = "✗"
            failed += 1
        
        print(f"{status} {ml_type} (m={m_trials}): step_based={step_based}, "
              f"category_based={category_based}, expect={expect_phase0}")
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def test_method_validation():
    """Test Issue #4: Verify method_config validation works correctly"""
    print("\n" + "="*70)
    print("TEST 4: Method Config Validation")
    print("="*70)
    
    valid_configs = [
        # Exact methods
        {
            'method_category': 'Exact',
            'exact_callback_type': 'int_only'
        },
        {
            'method_category': 'Exact',
            'exact_callback_type': 'int_and_LP'
        },
        # ML methods with all valid combinations
        {
            'method_category': 'ML',
            'ml_method_type': 'ML_Augmented_BD',
            'ml_callback_type': 'ML_at_both',
            'predictor_type': 'KNN',
            'n_trials': 5
        },
        {
            'method_category': 'ML',
            'ml_method_type': 'Covariate_Similarity',
            'ml_callback_type': 'ML_at_integer_only',
            'predictor_type': 'CLUSTERING'
        },
        {
            'method_category': 'ML',
            'ml_method_type': 'Random_Sampling',
            'ml_callback_type': 'ML_at_both'
        },
    ]
    
    invalid_configs = [
        # Invalid callback type names (should fail validation)
        {
            'method_category': 'ML',
            'ml_method_type': 'ML_Augmented_BD',
            'ml_callback_type': 'int_ML_LP_ML',  # Legacy name - should be ML_at_both
            'predictor_type': 'KNN',
            'n_trials': 5
        },
        # Missing n_trials for ML_Augmented_BD
        {
            'method_category': 'ML',
            'ml_method_type': 'ML_Augmented_BD',
            'ml_callback_type': 'ML_at_both',
            'predictor_type': 'KNN',
            'n_trials': 0
        },
    ]
    
    passed = 0
    failed = 0
    
    print("Valid configs:")
    for i, config in enumerate(valid_configs):
        is_valid, msg = validate_method_config(config)
        if is_valid:
            print(f"  ✓ Config {i+1}: {config.get('method_category')} - "
                  f"{config.get('ml_method_type', 'N/A')}")
            passed += 1
        else:
            print(f"  ✗ Config {i+1} incorrectly marked invalid: {msg}")
            failed += 1
    
    print("\nInvalid configs (should fail):")
    for i, config in enumerate(invalid_configs):
        is_valid, msg = validate_method_config(config)
        if not is_valid:
            print(f"  ✓ Config {i+1} correctly rejected: {msg}")
            passed += 1
        else:
            print(f"  ✗ Config {i+1} incorrectly marked valid")
            failed += 1
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def test_step_execution_patterns():
    """Test Issue #5: Verify correct step execution for each method type"""
    print("\n" + "="*70)
    print("TEST 5: Step Execution Patterns")
    print("="*70)
    
    # Expected step execution patterns
    test_cases = [
        ('Exact', None, [3]),  # Only step 3
        ('ML', 'ML_Augmented_BD', [1, 2, 3, 4, 5, 6]),  # All steps (with m_trials>0)
        ('ML', 'ML_Augmented_BD_with_duals', [1, 2, 3, 5, 6]),  # Skip step 4
        ('ML', 'Covariate_Similarity', [5, 6]),  # Only steps 5,6
        ('ML', 'Uncertainty_Similarity', [5, 6]),  # Only steps 5,6
        ('ML', 'Random_Sampling', [6]),  # Only step 6
    ]
    
    passed = 0
    failed = 0
    
    for method_category, ml_method_type, expected_steps in test_cases:
        if method_category == 'Exact':
            config = {
                'method_category': 'Exact',
                'exact_callback_type': 'int_only'
            }
        else:
            config = {
                'method_category': 'ML',
                'ml_method_type': ml_method_type,
                'ml_callback_type': 'ML_at_both',
                'predictor_type': 'KNN' if ml_method_type != 'Random_Sampling' else None,
                'm_trials': 3,
                'n_trials': 5
            }
        
        actual_steps = [i for i in range(1, 7) if should_execute_step(config, i)]
        
        if actual_steps == expected_steps:
            print(f"✓ {method_category}/{ml_method_type or 'N/A'}: {actual_steps}")
            passed += 1
        else:
            print(f"✗ {method_category}/{ml_method_type or 'N/A'}: "
                  f"Expected {expected_steps}, got {actual_steps}")
            failed += 1
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all validation tests"""
    print("\n" + "="*70)
    print("ARCHITECTURE FIX VALIDATION TESTS")
    print("="*70)
    
    all_passed = True
    
    all_passed &= test_callback_type_mapping()
    all_passed &= test_legacy_string_consistency()
    all_passed &= test_phase0_gating()
    all_passed &= test_method_validation()
    all_passed &= test_step_execution_patterns()
    
    print("\n" + "="*70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
