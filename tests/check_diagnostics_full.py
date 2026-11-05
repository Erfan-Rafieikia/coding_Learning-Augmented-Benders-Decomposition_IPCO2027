import pandas as pd

# Check the newest file (with all fixes applied - Feb 23, 2026)
file_path = '/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS/solution_log_20260223_133345.xlsx'

df = pd.read_excel(file_path)

print("="*80)
print("FULL DIAGNOSTIC DATA:")
print("="*80)

diagnostic_cols = [
    'total_unique_points', 'total_repeated_hits', 'repeated_hits_with_cuts',
    'mipsol_root', 'mipsol_nonroot', 'mipnode_root', 'mipnode_nonroot',
    'avg_mipsol_per_nonroot_node', 'avg_mipnode_per_nonroot_node',
    'num_solutions_reevaluated', 'max_objective_drift', 'mean_objective_drift', 'std_objective_drift'
]

available = [c for c in diagnostic_cols if c in df.columns]

if available:
    print(f"\nAll diagnostic columns:")
    for col in available:
        print(f"\n{col}:")
        print(df[col].values)
    
    print("\n" + "="*80)
    print("ROW-BY-ROW:")
    print("="*80)
    for idx, row in df[available].iterrows():
        print(f"\nRun {idx}:")
        for col in available:
            print(f"  {col}: {row[col]}")
