import pandas as pd

# Check the newest file (with all fixes applied - Feb 23, 2026)
file_path = '/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS/solution_log_20260223_133345.xlsx'

df = pd.read_excel(file_path)

print("Total columns:", len(df.columns))
print("\nLast 15 columns:")
for col in df.columns[-15:]:
    print(f"  - {col}")

print("\n" + "="*80)
print("Checking for diagnostic columns:")
print("="*80)

diagnostic_cols = [
    'total_unique_points', 'total_repeated_hits', 'repeated_hits_with_cuts',
    'mipsol_root', 'mipsol_nonroot', 'mipnode_root', 'mipnode_nonroot',
    'avg_mipsol_per_nonroot_node', 'avg_mipnode_per_nonroot_node',
    'num_solutions_reevaluated', 'max_objective_drift', 'mean_objective_drift', 'std_objective_drift'
]

available = [c for c in diagnostic_cols if c in df.columns]
missing = [c for c in diagnostic_cols if c not in df.columns]

print(f"\nFound {len(available)}/{len(diagnostic_cols)} diagnostic columns")

if available:
    print(f"\nAvailable columns: {available}")
    print("\nValues:")
    print(df[available].head(20))
    print("\nSummary statistics:")
    print(df[available].describe())
else:
    print("\nNO diagnostic columns found in the Excel file!")

if missing:
    print(f"\nMissing columns: {missing}")
