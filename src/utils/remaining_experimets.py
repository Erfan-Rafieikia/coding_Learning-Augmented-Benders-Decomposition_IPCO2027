"""
remaining_experimets.py

Identify duplicate and missing experiments from aggregated experiment results.

This tool helps understand the status of your experimental runs by:
1. Loading the aggregated_of_experiments.xlsx
2. Extracting configuration grids from config.py snapshots
3. Comparing actual experiment results with expected complete experiment set
4. Identifying duplicates and missing experiment combinations

The goal is to ensure all instances of the same problem_class have been run 
with all configured method variations.

Usage:
    python remaining_experimets.py
    
    Then adjust import paths and output settings in the CONFIGURATION section below.
"""

import os
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
import json

# =========================
# CONFIGURATION
# =========================

# Set these paths according to your experiment setup
#INPUT_FILE = r"/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS/Experiments - Digital Research Alliance of Canada/Aggregate_of_experiments.xlsx"
#INPUT_FILE = r"/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS/Experiments - Digital Research Alliance of Canada/Output/cleaned_data.xlsx"
INPUT_FILE = r"/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS/Experiments - Digital Research Alliance of Canada/cleaned_data_analysis.xlsx"
OUTPUT_DIR = r"/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS/Experiments - Digital Research Alliance of Canada"
CONFIG_SNAPSHOT_DIR = r"/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS"

OUTPUT_REPORT_NAME = "experiment_status_report.xlsx"


# =========================
# HELPER FUNCTIONS
# =========================

def extract_deterministic_instance(instance_filename):
    """
    Extract the deterministic instance name from instance_filename.
    Takes everything after the last backslash (\\) in the path.
    
    Example:
        "SCFLP/scenarios_s100_sigma0.1_0.3_cov6/p51_stochastic.txt" -> "p51_stochastic.txt"
    
    Args:
        instance_filename: Full instance file path
    
    Returns:
        Filename after the last backslash, or original if no backslash found
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


def extract_config_from_snapshot(snapshot_dir):
    """
    Extract configuration grids from snapshot_config.py file in a run directory.
    
    Returns a dict with keys:
        - method_categories_grid
        - exact_callback_types_grid
        - ml_method_types_grid
        - ml_callback_types_grid
        - predictor_ML_Augmented_BD_grid
        - predictor_ML_Augmented_BD_with_duals_grid
        - predictor_Covariate_Similarity_grid
        - predictor_Uncertainty_Similarity_grid
        - selected_subproblems_ratio_grid
        - n_trials_grid (if exists)
        - m_trials_grid (if exists)
        - use_covariates_grid (if exists)
        - use_stochastic_capacity_grid (if exists)
    """
    config_path = Path(snapshot_dir) / "snapshot_config.py"
    
    if not config_path.exists():
        return None
    
    config_vars = {}
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Use exec to evaluate the config file in a controlled namespace
        namespace = {}
        exec(content, namespace)
        
        # Extract relevant grid variables
        for var_name in [
            "method_categories_grid",
            "exact_callback_types_grid",
            "ml_method_types_grid",
            "ml_callback_types_grid",
            "predictor_ML_Augmented_BD_grid",
            "predictor_ML_Augmented_BD_with_duals_grid",
            "predictor_Covariate_Similarity_grid",
            "predictor_Uncertainty_Similarity_grid",
            "selected_subproblems_ratio_grid",
            "n_trials_grid",
            "m_trials_grid",
            "use_covariates_grid",
            "use_stochastic_capacity_grid",
        ]:
            if var_name in namespace:
                config_vars[var_name] = namespace[var_name]
        
        return config_vars
    
    except Exception as e:
        print(f"Warning: Could not extract config from {config_path}: {e}")
        return None


def scan_run_directories(base_dir):
    """
    Scan the RESULTS directory structure for run directories with snapshot configs.
    
    Returns a dict:
        problem_class -> list of (run_dir, config_dict) tuples
    """
    runs_by_problem_class = defaultdict(list)
    base_path = Path(base_dir)
    
    # Look for pattern: RESULTS/{PROBLEM_CLASS}_class/RESULTS/{RUN_ID}_{TIMESTAMP}/snapshot_config.py
    for problem_class_dir in base_path.glob("*_class"):
        problem_class = problem_class_dir.name.replace("_class", "")
        results_subdir = problem_class_dir / "RESULTS"
        
        if results_subdir.exists():
            for run_dir in results_subdir.iterdir():
                if run_dir.is_dir():
                    config = extract_config_from_snapshot(run_dir)
                    if config:
                        runs_by_problem_class[problem_class].append((run_dir, config))
    
    return runs_by_problem_class




def build_expected_experiment_grid(df, config_by_problem_class):
    """
    Build the expected complete set of experiments based on:
    1. Instances present in the data (by problem_class)
    2. Configuration grids from snapshot configs
    
    Returns a dict:
        problem_class -> list of expected experiment dicts
    """
    expected_by_problem_class = defaultdict(list)
    
    # Group instances by problem_class
    instances_by_problem_class = defaultdict(set)
    for _, row in df.iterrows():
        problem_class = row.get("problem_class", "Unknown")
        instance_filename = row.get("instance_filename", "Unknown")
        instances_by_problem_class[problem_class].add(instance_filename)
    
    # Build expected grids for each problem class
    for problem_class, instances in instances_by_problem_class.items():
        
        # Find the most common config for this problem_class
        # (configurations should be consistent within a problem class)
        class_data = df[df["problem_class"] == problem_class]
        
        # Collect all configs from available configs
        # For now, use the actual runs in the data to infer what was intended
        configs = extract_configs_from_dataframe(class_data)
        
        if not configs:
            print(f"Warning: Could not determine configuration for {problem_class}")
            continue
        
        # Generate all combinations
        for instance in sorted(instances):
            for config in configs:
                expected_by_problem_class[problem_class].append({
                    "instance_filename": instance,
                    **config
                })
    
    return expected_by_problem_class


def extract_configs_from_dataframe(df):
    """
    Extract unique configuration combinations from actual test data.
    
    Returns a list of config dicts, each representing a unique method configuration.
    """
    configs = []
    
    # Extract unique combinations of configuration columns
    config_cols = [
        "method_category",
        "ml_method_type",
        "ml_callback_type",
        "exact_callback_type",
        "predictor_type",
        "selected_subproblems_ratio",
        "n_trials",
        "m_trials",
        "use_covariates",
        "use_stochastic_capacity",
        "time_limit_sec",
    ]
    
    # Filter to only columns that exist in the dataframe
    config_cols = [col for col in config_cols if col in df.columns]
    
    unique_configs = df[config_cols].drop_duplicates().reset_index(drop=True)
    
    configs = unique_configs.to_dict('records')
    
    return configs


def find_missing_experiments(actual_df, expected_per_problem_class):
    """
    Compare actual experiments with expected ones.
    
    Returns:
        missing_experiments: List of expected experiments not found in actual
        extra_experiments: List of actual experiments not in expected
        summary: Summary statistics
    """
    missing = []
    extra = []
    matched = []
    
    # Create a matching key function
    def make_key(exp_dict):
        """Create a hashable key from experiment dict"""
        key_cols = [
            "instance_filename",
            "method_category",
            "ml_method_type",
            "ml_callback_type",
            "exact_callback_type",
            "predictor_type",
            "selected_subproblems_ratio",
            "n_trials",
            "m_trials",
            "use_covariates",
            "use_stochastic_capacity",
            "time_limit_sec",
        ]
        key_vals = tuple(str(exp_dict.get(col, "NA")) for col in key_cols)
        return key_vals
    
    # Convert actual experiments to keys
    actual_keys = set()
    for _, row in actual_df.iterrows():
        key = make_key(row.to_dict())
        actual_keys.add(key)
    
    # Compare expected with actual
    for problem_class, expected_list in expected_per_problem_class.items():
        for expected in expected_list:
            key = make_key(expected)
            if key in actual_keys:
                matched.append((problem_class, expected))
            else:
                missing.append((problem_class, expected))
    
    # Find extra experiments (in actual but not expected)
    expected_keys = set()
    for expected_list in expected_per_problem_class.values():
        for expected in expected_list:
            key = make_key(expected)
            expected_keys.add(key)
    
    for _, row in actual_df.iterrows():
        key = make_key(row.to_dict())
        if key not in expected_keys:
            extra.append(row.to_dict())
    
    summary = {
        "total_expected": sum(len(lst) for lst in expected_per_problem_class.values()),
        "total_matched": len(matched),
        "total_missing": len(missing),
        "total_extra": len(extra),
        "completion_rate_%": 100 * len(matched) / sum(len(lst) for lst in expected_per_problem_class.values()) if sum(len(lst) for lst in expected_per_problem_class.values()) > 0 else 0
    }
    
    return missing, extra, matched, summary


def create_problem_class_completion_report(actual_df, expected_per_problem_class):
    """
    Create a summary report of completion status by problem class.
    
    Returns a DataFrame with columns:
        - problem_class
        - total_instances
        - total_configurations
        - total_expected
        - total_actual
        - completion_rate_%
    """
    report_data = []
    
    # Group actual by problem class
    actual_by_problem_class = defaultdict(lambda: {"count": 0, "instances": set(), "configs": set()})
    for _, row in actual_df.iterrows():
        pc = row.get("problem_class")
        instance = row.get("instance_filename")
        config_key = tuple(str(row.get(col, "NA")) for col in [
            "method_category", "ml_method_type", "ml_callback_type", 
            "exact_callback_type", "predictor_type", "selected_subproblems_ratio"
        ])
        
        actual_by_problem_class[pc]["count"] += 1
        actual_by_problem_class[pc]["instances"].add(instance)
        actual_by_problem_class[pc]["configs"].add(config_key)
    
    # Create report
    for problem_class, expected_list in expected_per_problem_class.items():
        actual = actual_by_problem_class[problem_class]
        
        expected_instances = set(exp["instance_filename"] for exp in expected_list)
        expected_configs = set(tuple(str(exp.get(col, "NA")) for col in [
            "method_category", "ml_method_type", "ml_callback_type", 
            "exact_callback_type", "predictor_type", "selected_subproblems_ratio"
        ]) for exp in expected_list)
        
        completion_rate = 100 * actual["count"] / len(expected_list) if expected_list else 0
        
        report_data.append({
            "problem_class": problem_class,
            "total_instances": len(expected_instances),
            "total_configurations": len(expected_configs),
            "total_expected": len(expected_list),
            "total_actual": actual["count"],
            "completion_rate_%": completion_rate,
            "unique_instances_actual": len(actual["instances"]),
            "unique_configs_actual": len(actual["configs"]),
        })
    
    return pd.DataFrame(report_data)


def create_missing_experiments_report(missing_experiments):
    """
    Create a detailed report of missing experiments.
    
    Returns a DataFrame with all missing experiments.
    """
    if not missing_experiments:
        return pd.DataFrame()
    
    data = []
    for problem_class, exp_dict in missing_experiments:
        row = {"problem_class": problem_class, **exp_dict}
        data.append(row)
    
    return pd.DataFrame(data)


def create_instance_configuration_matrix(actual_df):
    """
    Create a matrix showing which instances have been tested with which configurations.
    
    Returns a dict:
        problem_class -> DataFrame(instances x configurations)
    """
    matrices = {}
    
    for problem_class in actual_df["problem_class"].unique():
        class_data = actual_df[actual_df["problem_class"] == problem_class]
        
        # Get unique instances and configurations
        instances = sorted(class_data["instance_filename"].unique())
        
        # Define configuration signature
        def get_config_sig(row):
            return f"{row.get('method_category', '?')}_{row.get('ml_method_type', '?')}_{row.get('predictor_type', '?')}_{row.get('selected_subproblems_ratio', '?')}"
        
        configs = sorted(class_data.apply(get_config_sig, axis=1).unique())
        
        # Build matrix
        matrix = pd.DataFrame(0, index=instances, columns=configs)
        
        for _, row in class_data.iterrows():
            instance = row["instance_filename"]
            config_sig = get_config_sig(row)
            matrix.loc[instance, config_sig] += 1  # Count occurrences (should be 1 after dedup)
        
        matrices[problem_class] = matrix
    
    return matrices


def create_missing_by_instance_report(missing_experiments):
    """
    Create a detailed report showing which configurations are missing per instance.
    
    Returns a dict:
        instance_filename -> list of missing configurations
    """
    missing_by_instance = defaultdict(lambda: {"problem_class": None, "configs": []})
    
    for problem_class, exp_dict in missing_experiments:
        instance = exp_dict.get("instance_filename", "Unknown")
        missing_by_instance[instance]["problem_class"] = problem_class
        
        config_desc = f"{exp_dict.get('method_category', '?')}_{exp_dict.get('ml_method_type', '?')}_{exp_dict.get('predictor_type', '?')}_{exp_dict.get('selected_subproblems_ratio', '?')}"
        missing_by_instance[instance]["configs"].append(config_desc)
    
    return missing_by_instance


def generate_summary_statistics(actual_df, deduped_df, removed_duplicates, missing, summary):
    """
    Generate comprehensive summary statistics.
    """
    stats = {
        "Overall Statistics": {
            "Total rows in raw data": len(actual_df),
            "Total distinct experiments (after dedup)": len(deduped_df),
            "Total duplicate occurrences": len(removed_duplicates),
            "Total problem classes": actual_df["problem_class"].nunique(),
            "Total unique instances": actual_df["instance_filename"].nunique(),
        },
        "Method Distribution": {
            "Rows with method_category='Exact'": len(actual_df[actual_df["method_category"] == "Exact"]),
            "Rows with method_category='ML'": len(actual_df[actual_df["method_category"] == "ML"]),
        },
        "Expected vs Actual": summary,
    }
    
    return stats


# =========================
# MAIN ANALYSIS FLOW
# =========================

def main():
    """
    Main analysis pipeline.
    """
    print("\n" + "="*80)
    print("EXPERIMENT STATUS ANALYSIS")
    print("="*80 + "\n")
    
    # Step 1: Load aggregated data
    print("[Step 1] Loading aggregated experiment data...")
    if not Path(INPUT_FILE).exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return
    
    df = pd.read_excel(INPUT_FILE)
    print(f"✓ Loaded {len(df)} rows from {Path(INPUT_FILE).name}")
    print(f"  Problem classes: {sorted(df['problem_class'].unique())}")
    print(f"  Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
    
    # Add deterministic_instance column
    print("\n[Step 1a] Creating deterministic_instance column...")
    df['deterministic_instance'] = df['instance_filename'].apply(extract_deterministic_instance)
    print(f"✓ Added deterministic_instance column")
    print(f"  Unique deterministic instances: {df['deterministic_instance'].nunique()}")
    
    # Step 2: Keep all rows (no duplicate pruning)
    print("\n[Step 2] Skipping duplicate pruning (keeping all rows)...")
    deduped = df.copy()
    removed_dups = pd.DataFrame()
    dup_counts = {}
    
    # Step 3: Extract expected configurations from data
    print("\n[Step 3] Extracting configuration patterns from data...")
    expected_per_problem_class = build_expected_experiment_grid(df, {})
    
    for problem_class in sorted(expected_per_problem_class.keys()):
        num_expected = len(expected_per_problem_class[problem_class])
        print(f"  {problem_class}: {num_expected} expected experiment combinations")
    
    # Step 4: Find missing experiments
    print("\n[Step 4] Comparing actual vs expected experiments...")
    missing, extra, matched, summary = find_missing_experiments(deduped, expected_per_problem_class)
    print(f"✓ Matched experiments: {summary['total_matched']}")
    print(f"  Missing experiments: {summary['total_missing']}")
    print(f"  Extra/unexpected experiments: {summary['total_extra']}")
    print(f"  Completion rate: {summary['completion_rate_%']:.1f}%")
    
    # Step 5: Create completion report by problem class
    print("\n[Step 5] Creating completion report by problem class...")
    completion_report = create_problem_class_completion_report(deduped, expected_per_problem_class)
    print("Completion Report:")
    print(completion_report.to_string(index=False))
    
    # Step 6: Create instance-configuration matrix
    print("\n[Step 6] Creating instance-configuration matrix...")
    matrices = create_instance_configuration_matrix(deduped)
    for pc, matrix in matrices.items():
        print(f"\n{pc} (Instances x Configurations):")
        print(f"  Instances: {len(matrix.index)}")
        print(f"  Configurations: {len(matrix.columns)}")
        coverage = (matrix > 0).sum().sum() / matrix.size * 100
        print(f"  Coverage: {coverage:.1f}% of combinations tested")
    
    # Step 7: Summary statistics
    print("\n[Step 7] Summary statistics...")
    stats = generate_summary_statistics(df, deduped, removed_dups, missing, summary)
    for section, data in stats.items():
        print(f"\n{section}:")
        for key, value in data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Step 8: Analyze missing experiments by instance
    print("\n[Step 8] Analyzing missing experiments by instance...")
    missing_by_instance = create_missing_by_instance_report(missing)
    
    # Show top instances with most missing configurations
    sorted_instances = sorted(missing_by_instance.items(), key=lambda x: len(x[1]["configs"]), reverse=True)
    print(f"\nInstances with most missing configurations:")
    for instance, data in sorted_instances[:10]:
        print(f"  {instance}: {len(data['configs'])} missing configs")
        if len(data['configs']) <= 5:
            for config in data['configs'][:5]:
                print(f"    - {config}")
    
    # Step 9a: Create deterministic instance summary with num_subproblems pivot
    print("\n[Step 9a] Creating deterministic instance summary...")
    
    # Group by deterministic_instance and num_subproblems, count occurrences
    if 'num_subproblems' in deduped.columns:
        det_inst_summary = deduped.groupby(
            ['deterministic_instance', 'num_subproblems']
        ).size().reset_index(name='count')
        
        # Pivot to get num_subproblems as columns
        det_inst_pivot = det_inst_summary.pivot_table(
            index='deterministic_instance',
            columns='num_subproblems',
            values='count',
            fill_value=0,
            aggfunc='sum'
        ).reset_index()
        
        # Rename columns to make it clear
        det_inst_pivot.columns.name = None
        det_inst_pivot.columns = ['deterministic_instance'] + [f'num_subproblems_{col}' for col in det_inst_pivot.columns[1:]]
        
        print(f"✓ Created summary with {len(det_inst_pivot)} deterministic instances")
        print(f"  Unique num_subproblems values: {sorted(det_inst_summary['num_subproblems'].unique())}")
    else:
        det_inst_pivot = None
        print("  Note: 'num_subproblems' column not found")
    
    # Step 9: Write comprehensive output report
    print("\n[Step 9] Writing output report...")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    output_excel_path = output_path / OUTPUT_REPORT_NAME
    
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        # Sheet 1: Raw processed data (deduplicated)
        deduped.to_excel(writer, sheet_name="processed_data", index=False)
        
        # Sheet 1b: Deterministic instance summary (with num_subproblems pivot)
        if det_inst_pivot is not None:
            det_inst_pivot.to_excel(writer, sheet_name="det_instance_summary", index=False)
        
        # Sheet 2: Duplicates found
        if not removed_dups.empty:
            removed_dups.to_excel(writer, sheet_name="duplicates_removed", index=False)
        
        # Sheet 3: Completion report by problem class
        completion_report.to_excel(writer, sheet_name="completion_by_class", index=False)
        
        # Sheet 4: Missing experiments
        if missing:
            missing_df = create_missing_experiments_report(missing)
            missing_df.to_excel(writer, sheet_name="missing_experiments", index=False)
        
        # Sheet 5-N: Instance-configuration matrices
        for problem_class, matrix in matrices.items():
            safe_sheet_name = f"{problem_class}_matrix"[:31]  # Excel sheet name limit
            matrix.to_excel(writer, sheet_name=safe_sheet_name)
        
        # Sheet N+1: Summary statistics
        stats_data = []
        for section, data in stats.items():
            for key, value in data.items():
                stats_data.append({"Category": section, "Metric": key, "Value": str(value)})
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name="summary_stats", index=False)
    
    print(f"✓ Report saved to: {output_excel_path}")
    
    # Step 10: Generate CSV files for missing experiments
    if missing:
        missing_df = create_missing_experiments_report(missing)
        missing_csv = output_path / "missing_experiments.csv"
        missing_df.to_csv(missing_csv, index=False)
        print(f"✓ Missing experiments CSV: {missing_csv}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return {
        "status": "success",
        "raw_data": df,
        "deduped_data": deduped,
        "duplicates": removed_dups,
        "missing_experiments": missing,
        "completion_report": completion_report,
        "matrices": matrices,
        "summary": summary,
        "statistics": stats,
        "det_inst_summary": det_inst_pivot if 'det_inst_pivot' in locals() else None,
    }


if __name__ == "__main__":
    result = main()
