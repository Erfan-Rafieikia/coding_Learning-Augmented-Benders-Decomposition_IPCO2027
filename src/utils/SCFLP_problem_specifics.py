# Re-import necessary modules after code execution environment reset
import os
from pathlib import Path
import pandas as pd

# Define the folder where the deterministic input files are stored
input_dir = "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP"
files_to_parse = [f"p{i}" for i in range(1, 72)]  # p1 to p71
file_paths = [os.path.join(input_dir, fname) for fname in files_to_parse]

# Initialize the output list
records = []

# Helper function to read number of facilities and customers
def read_basic_structure(file_path):
    try:
        with open(file_path, "r") as f:
            words = f.read().split()
            num_facilities = int(words[0])
            num_customers = int(words[1])
        return num_facilities, num_customers
    except Exception as e:
        return None, None

# Process each file
for fname in files_to_parse:
    full_path = os.path.join(input_dir, fname)
    if os.path.exists(full_path):
        num_f, num_c = read_basic_structure(full_path)
        records.append({
            "file_name": fname,
            "num_facilities": num_f,
            "num_customers": num_c
        })

# Create DataFrame and save as CSV
df_summary = pd.DataFrame(records)
csv_path = "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP/scflp_instance_summary.csv"
df_summary.to_csv(csv_path, index=False)

