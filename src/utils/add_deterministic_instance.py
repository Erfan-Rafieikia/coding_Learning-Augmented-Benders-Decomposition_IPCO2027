"""
Quick script to add deterministic_instance column to Aggregate_of_experiments.xlsx
"""

import pandas as pd
from pathlib import Path

INPUT_FILE = r"/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS/Experiments - Digital Research Alliance of Canada/Aggregate_of_experiments.xlsx"

def extract_deterministic_instance(instance_filename):
    """
    Extract the deterministic instance name from instance_filename.
    Takes everything after the last backslash (\\) in the path.
    """
    if pd.isna(instance_filename):
        return instance_filename
    
    instance_str = str(instance_filename)
    
    # Split by backslash and take the last part
    if '\\' in instance_str:
        return instance_str.split('\\')[-1]
    # Also handle forward slash (in case data uses Unix paths)
    elif '/' in instance_str:
        return instance_str.split('/')[-1]
    
    return instance_str


print("Loading Aggregate_of_experiments.xlsx...")
df = pd.read_excel(INPUT_FILE)
print(f"Loaded {len(df)} rows")

# Add deterministic_instance column
print("Adding deterministic_instance column...")
df['deterministic_instance'] = df['instance_filename'].apply(extract_deterministic_instance)

print(f"Unique deterministic instances: {df['deterministic_instance'].nunique()}")
print(f"Deterministic instances found:")
for det_inst in sorted(df['deterministic_instance'].unique()):
    count = len(df[df['deterministic_instance'] == det_inst])
    print(f"  - {det_inst}: {count} rows")

# Save back to the same file
print(f"\nSaving updated file to {INPUT_FILE}...")
df.to_excel(INPUT_FILE, index=False, engine="openpyxl")

print("✓ Successfully updated Aggregate_of_experiments.xlsx")
print(f"  Total rows: {len(df)}")
print(f"  Total columns: {len(df.columns)}")
