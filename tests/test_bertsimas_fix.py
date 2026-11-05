#!/usr/bin/env python3
"""
Quick test to verify BERTSIMAS_RANDOM_SAMPLING works after validation fix.
"""
import sys
sys.path.insert(0, 'src')

from Problem.data import read_scflp_data
from Dual_Prediction.training_and_selection import solve_bertsimas_random_sampling_selection

print("=" * 60)
print("Testing BERTSIMAS_RANDOM_SAMPLING Fix")
print("=" * 60)

# Load data
data = read_scflp_data('DATA/SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p5_stochastic.txt')

print(f"\n1. Data Loading:")
print(f"   ✓ Number of scenarios: {len(data.S)}")

# Create dummy feature vectors (BERTSIMAS_RANDOM_SAMPLING doesn't use them)
feature_vectors = {s: [0.0] for s in range(len(data.S))}

print(f"\n2. Testing BERTSIMAS_RANDOM_SAMPLING Selection:")
p = 15  # Select 15 scenarios per iteration
sampling_params = {'random_state': 42}

selected_scenarios, mapping = solve_bertsimas_random_sampling_selection(
    feature_vectors, p, sampling_params
)

print(f"\n3. Selection Results:")
print(f"   ✓ Selected scenarios: {selected_scenarios}")
print(f"   ✓ Expected: [] (empty list - selection is dynamic)")
print(f"   ✓ Is empty: {len(selected_scenarios) == 0}")

print(f"\n4. Mapping Info:")
print(f"   ✓ Mapping keys: {list(mapping.keys())}")
print(f"   ✓ Contains: {mapping}")

# Verify expected behavior
if len(selected_scenarios) == 0 and 'method' in mapping:
    print(f"\n" + "=" * 60)
    print("✓ BERTSIMAS_RANDOM_SAMPLING Fix Verified Successfully!")
    print("  - Returns empty selection (as expected)")
    print("  - Selection happens dynamically at each iteration")
    print("=" * 60)
    sys.exit(0)
else:
    print(f"\n" + "=" * 60)
    print("✗ BERTSIMAS_RANDOM_SAMPLING Test Failed!")
    print("=" * 60)
    sys.exit(1)
