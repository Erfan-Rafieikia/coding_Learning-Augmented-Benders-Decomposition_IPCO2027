#!/usr/bin/env python3
"""
Test to verify DUAL_CLUSTERING_WITH_DUAL works after fixing feature_vectors check.
"""
import sys
sys.path.insert(0, 'src')

from Problem.data import read_scflp_data
from Dual_Prediction.training_and_selection import solve_dual_clustering_with_dual_selection

print("=" * 60)
print("Testing DUAL_CLUSTERING_WITH_DUAL Fix")
print("=" * 60)

# Load data
data = read_scflp_data('DATA/SCFLP/scenarios_s100_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p5_stochastic.txt')

print(f"\n1. Data Loading:")
print(f"   ✓ Number of scenarios: {len(data.S)}")

# Create dummy all_trial_duals (simulating Phase 0 + Phase 1 collection)
# Format: {(scenario_id, iteration): {'mu': {...}, 'nu': {...}}}
import numpy as np
np.random.seed(42)

all_trial_duals = {}
for sid in range(len(data.S)):
    for iteration in range(5):  # 5 iterations of Phase 1
        # Create realistic varying duals for this scenario-iteration pair
        # Each scenario has different dual values to enable proper clustering
        all_trial_duals[(sid, iteration)] = {
            'mu': {i: np.random.randn() + sid * 0.1 for i in range(len(data.I))},  # Varying mu duals
            'nu': {j: np.random.randn() + sid * 0.1 for j in range(len(data.J))}   # Varying nu duals
        }

print(f"\n2. Dual Collection Simulation:")
print(f"   ✓ Created {len(all_trial_duals)} trial duals")
print(f"   ✓ Format: (scenario_id, iteration) -> dual_dict")

print(f"\n3. Testing DUAL_CLUSTERING_WITH_DUAL Selection:")
p = 15  # Select 15 scenarios
clustering_params = {
    'random_state': 42,
    'max_iter': 300,
    'tol': 1e-6,
    'all_trial_duals': all_trial_duals
}

try:
    selected_scenarios, mapping = solve_dual_clustering_with_dual_selection(
        all_trial_duals, p, clustering_params
    )
    
    print(f"\n4. Selection Results:")
    print(f"   ✓ Selected scenarios: {len(selected_scenarios)}")
    print(f"   ✓ Expected: 15")
    print(f"   ✓ Match: {len(selected_scenarios) == 15}")
    print(f"   ✓ Selected IDs (first 5): {sorted(selected_scenarios)[:5]}")
    
    print(f"\n5. Mapping Results:")
    print(f"   ✓ Unselected scenarios mapped: {len(mapping)}")
    print(f"   ✓ Expected: {len(data.S) - 15} (85)")
    print(f"   ✓ Match: {len(mapping) == (len(data.S) - 15)}")
    
    # Verify mapping structure
    sample_unselected = [s for s in range(len(data.S)) if s not in selected_scenarios][0]
    print(f"   ✓ Sample mapping for scenario {sample_unselected}: {mapping.get(sample_unselected)}")
    
    # Verify all unselected scenarios are mapped
    all_mapped = all(s in mapping for s in range(len(data.S)) if s not in selected_scenarios)
    print(f"   ✓ All unselected scenarios mapped: {all_mapped}")
    
    # Verify coverage (selected + mapped = all scenarios)
    coverage = len(selected_scenarios) + len(mapping)
    print(f"   ✓ Coverage: {coverage} / {len(data.S)} scenarios")
    print(f"   ✓ Complete coverage: {coverage == len(data.S)}")
    
    if len(selected_scenarios) == 15 and len(mapping) == 85 and coverage == len(data.S):
        print(f"\n" + "=" * 60)
        print("✓ DUAL_CLUSTERING_WITH_DUAL Fix Verified Successfully!")
        print("  - Selection works without feature_vectors")
        print("  - Uses all_trial_duals directly")  
        print("  - Proper clustering in dual space")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"\n" + "=" * 60)
        print("✗ DUAL_CLUSTERING_WITH_DUAL Test Failed!")
        print(f"  - Selected: {len(selected_scenarios)} (expected 15)")
        print(f"  - Mapped: {len(mapping)} (expected 85)")
        print(f"  - Coverage: {coverage} (expected {len(data.S)})")
        print("=" * 60)
        sys.exit(1)
        
except Exception as e:
    print(f"\n" + "=" * 60)
    print("✗ DUAL_CLUSTERING_WITH_DUAL Test Failed with Exception!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 60)
    sys.exit(1)
