#!/usr/bin/env python3
"""
Quick test to verify the TREE method improvements work.
"""

import sys
import os
sys.path.insert(0, '/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src')

import numpy as np
from Dual_Prediction.training_and_selection import _build_tree_adaptive, TreeNode

def test_tree_splitting():
    """Test tree splitting with guaranteed minimum leaves."""
    print("Testing tree splitting with forced minimum leaves...")
    
    # Create simple test data 
    scenario_ids = list(range(100))
    
    # Create low-variance feature matrix (this would cause single leaf issue)
    feature_matrix = np.random.normal(0, 0.001, (100, 3))  # Very low variance
    
    # Create dummy dual dissimilarity matrix
    dual_dissim_matrix = np.random.random((100, 100))
    
    # Create root node
    root = TreeNode(scenario_ids, 0, 0)
    
    # Test tree building
    max_leaves = 15
    min_impurity_decrease = 0.0
    quantile_grid_size = 10
    
    try:
        result = _build_tree_adaptive(
            root, feature_matrix, dual_dissim_matrix,
            max_leaves, min_impurity_decrease, quantile_grid_size
        )
        
        print(f"Tree building result: {result}")
        print(f"Final leaves: {result['final_leaves']}")
        print(f"Minimum leaves guaranteed: {result.get('min_leaves_guaranteed', False)}")
        
        if result['final_leaves'] >= 2:
            print("✅ SUCCESS: Tree has at least 2 leaves!")
        else:
            print("❌ FAILED: Tree still has only 1 leaf")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tree_splitting()