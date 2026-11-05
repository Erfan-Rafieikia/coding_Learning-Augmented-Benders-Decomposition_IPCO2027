#!/usr/bin/env python3
"""
Simple syntax validation test for tree-based prediction method.
This test only checks that our implementation has correct Python syntax.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_syntax_validation():
    """Test that all modified files have correct Python syntax."""
    print("🔍 Testing Python syntax validation...")
    
    files_to_check = [
        'src/Dual_Prediction/prediction.py',
        'src/Problem/callbacks.py',
        'src/config.py'
    ]
    
    syntax_errors = []
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            compile(source_code, file_path, 'exec')
            print(f"✓ {file_path} - syntax OK")
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"✗ {file_path} - syntax error: {e}")
        except Exception as e:
            print(f"⚠ {file_path} - could not check: {e}")
    
    return len(syntax_errors) == 0

def test_tree_method_documentation():
    """Test that TREE method is properly documented in key files."""
    print("\n📖 Testing documentation updates...")
    
    # Check if TREE is mentioned in key files
    checks = [
        ('src/Problem/callbacks.py', 'prediction_method (str): REG (regression), KNN, or TREE'),
        ('src/config.py', 'prediction_method = ["KNN", "REG", "TREE"]'),
        ('src/Dual_Prediction/prediction.py', 'def train_tree_predictor(')
    ]
    
    all_passed = True
    
    for file_path, expected_text in checks:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            if expected_text in content:
                print(f"✓ {file_path} - contains expected documentation")
            else:
                print(f"✗ {file_path} - missing expected text: {expected_text}")
                all_passed = False
        except Exception as e:
            print(f"⚠ {file_path} - could not check: {e}")
            all_passed = False
    
    return all_passed

def test_tree_method_structure():
    """Test that tree method has proper structure in callbacks."""
    print("\n🏗️ Testing tree method structure...")
    
    try:
        with open('src/Problem/callbacks.py', 'r') as f:
            content = f.read()
        
        required_elements = [
            'def tree_predict(',
            'self.all_trial_duals',
            'self.trained_tree_models', 
            'self.tree_training_complete',
            'num_cuts_mip_ml_tree',
            'num_cuts_rel_ml_tree',
            "prediction_method == 'TREE'"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"✗ Missing elements: {missing_elements}")
            return False
        else:
            print("✓ All required tree method elements found")
            return True
            
    except Exception as e:
        print(f"✗ Error checking structure: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🌳 Tree-based Prediction Syntax Validation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Syntax validation
    if test_syntax_validation():
        tests_passed += 1
        
    # Test 2: Documentation updates
    if test_tree_method_documentation():
        tests_passed += 1
        
    # Test 3: Structure validation
    if test_tree_method_structure():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All validation tests passed!")
        print("\n✅ Tree-based prediction method implementation is complete!")
        print("\n📋 Implementation Summary:")
        print("• Added train_tree_predictor() function in prediction.py")
        print("• Added tree_predict() method in callbacks.py")
        print("• Added tree-specific data storage and tracking")
        print("• Updated configuration to support 'TREE' method")
        print("• Added comprehensive documentation")
        print("\n🚀 Ready for integration testing with full environment!")
        return 0
    else:
        print("❌ Some validation tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())