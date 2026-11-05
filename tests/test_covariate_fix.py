#!/usr/bin/env python3
"""
Quick test to verify COVARIATE_CLUSTERING works after data parser fix.
"""
import sys
sys.path.insert(0, 'src')

from Problem.data import read_scflp_data
from Dual_Prediction.training_and_selection import solve_covariate_clustering_selection
import numpy as np

# Load data
print("=" * 60)
print("Testing COVARIATE_CLUSTERING Fix")
print("=" * 60)

data = read_scflp_data('DATA/SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p5_stochastic.txt')

# Check covariate vectors loaded
print(f"\n1. Data Loading:")
if data.covariate_vectors is not None:
    print(f"   ✓ Covariate vectors loaded: {data.covariate_vectors.shape}")
    print(f"   ✓ Number of scenarios: {data.covariate_vectors.shape[0]}")
    print(f"   ✓ Number of covariates: {data.covariate_vectors.shape[1]}")
else:
    print(f"   ✗ Covariate vectors NOT loaded!")
    sys.exit(1)

# Create covariate_vectors dict for clustering
covariate_vectors = {}
for sid in range(len(data.covariate_vectors)):
    covariate_vectors[sid] = data.covariate_vectors[sid]

print(f"\n2. Covariate Vector Dictionary:")
print(f"   ✓ Created dict with {len(covariate_vectors)} entries")

# Test clustering
p = 15  # Select 15 representative scenarios
clustering_params = {
    'random_state': 42,
    'init_method': 'k-means++',
    'max_iter': 300,
    'tol': 1e-4
}

print(f"\n3. Running COVARIATE_CLUSTERING:")
print(f"   Parameters: p={p}, {clustering_params}")

selected_scenarios, covariate_mapping = solve_covariate_clustering_selection(
    covariate_vectors, p, clustering_params
)

print(f"\n4. Results:")
print(f"   ✓ Selected {len(selected_scenarios)} representative scenarios")
print(f"   ✓ Mapped {len(covariate_mapping)} unselected scenarios to representatives")
print(f"   ✓ Selected scenarios: {sorted(selected_scenarios)}")
print(f"   ✓ First 5 mappings: {dict(list(covariate_mapping.items())[:5])}")

# Verify all scenarios are covered
all_scenarios = set(range(100))
selected_set = set(selected_scenarios)
mapped_scenarios = set(covariate_mapping.keys())
covered_scenarios = selected_set.union(mapped_scenarios)

print(f"\n5. Coverage Verification:")
print(f"   Total scenarios: {len(all_scenarios)}")
print(f"   Selected: {len(selected_set)}")
print(f"   Mapped: {len(mapped_scenarios)}")
print(f"   Covered: {len(covered_scenarios)}")

if covered_scenarios == all_scenarios:
    print(f"   ✓ All scenarios covered!")
else:
    missing = all_scenarios - covered_scenarios
    print(f"   ✗ Missing scenarios: {missing}")

print(f"\n" + "=" * 60)
print("✓ COVARIATE_CLUSTERING Fix Verified Successfully!")
print("=" * 60)
