#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, 'src')

from Problem.callbacks import Callback

# Test basic instantiation
try:
    callback = Callback(
        problem_type="SCFLP",
        data=None,
        first_stage_values={},
        theta_vars={},
        m_trials=0
    )
    print("SUCCESS: Callback instantiated with m_trials parameter")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"ERROR TYPE: {type(e).__name__}")