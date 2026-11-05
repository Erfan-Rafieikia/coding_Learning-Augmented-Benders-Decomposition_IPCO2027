"""
remaining_experimets.py - USAGE GUIDE

This document explains what the remaining_experimets.py script does and how to interpret its outputs.
"""

# =============================================================================
# WHAT THIS TOOL DOES
# =============================================================================

The remaining_experimets.py script analyzes your aggregated experiment results to:

1. DETECT DUPLICATES
   - Identifies experiment runs that were conducted multiple times (redundant rows)
   - Uses the duplicate detection rule defined in alliance.ipynb
   - Keeps the first occurrence and marks additional occurrences as duplicates

2. IDENTIFY MISSING EXPERIMENTS
   - Compares actual experiment results with the expected complete set
   - Expected set is inferred from your configuration grids
   - Reports which configurations were not completed in your runs

3. ASSESS COMPLETION STATUS
   - Shows overall completion percentage
   - Breaks down completion by problem class (SCFLP, SCMND, etc.)
   - Creates a matrix showing which instances have been tested with which configurations

4. GENERATE ACTIONABLE REPORTS
   - Excel workbook with multiple sheets for detailed analysis
   - CSV files listing missing experiments
   - Statistics on duplicates and completion rates


# =============================================================================
# KEY FINDINGS FROM YOUR DATA (as of current run)
# =============================================================================

DUPLICATE EXPERIMENTS:
  - Raw data: 1161 rows
  - Unique experiments: 762 rows
  - Duplicate occurrences: 399 rows
  - Most duplicated instance: 29 times
  → Action: These are redundant and should be removed before analysis

MISSING EXPERIMENTS:
  - Expected total combinations: 1155
  - Matched combinations: 762 (66%)
  - Missing combinations: 393 (34%)
  
  By Problem Class:
    - SCFLP: 75.2% complete (662 of 880)
    - SCMND: 36.4% complete (100 of 275)  ← This is the primary gap

INSTANCES WITH MOST MISSING WORK:
  - SUMND instances: 51 missing configs each (e.g., S300 and S500 variants)
  - SCFLP p50 variants: 21-22 missing configs each
  - SCFLP p4 variants: 12 missing configs each

INTERPRETATION:
  Your runs did not complete all planned experiments. The most likely reason is that some of your 
  SLURM job submissions stopped before finishing all queued tasks, particularly for SCMND with SUMND 
  instances (36% complete vs 75% for SCFLP).


# =============================================================================
# DUPLICATE DETECTION RULE
# =============================================================================

Two experiment rows are considered DUPLICATES if they match exactly on ALL of these columns:

  - instance_filename
  - method_category
  - ml_method_type
  - ml_callback_type
  - exact_callback_type
  - predictor_type
  - selected_subproblems_ratio
  - n_trials
  - m_trials
  - use_covariates
  - use_stochastic_capacity
  - time_limit_sec

If all these columns match, it means the exact same configuration was run on the same instance,
making it a redundant result. The script keeps the first occurrence and removes subsequent ones.

Why this matters: You may have re-run some jobs that completed in earlier runs. The aggregation 
process captured all results, so deduplication is needed for clean analysis.


# =============================================================================
# UNDERSTANDING THE OUTPUT REPORT
# =============================================================================

The script generates: experiment_status_report.xlsx

SHEETS INCLUDED:

1. processed_data
   - All deduplicated experiments (762 rows)
   - This is your clean dataset for analysis
   - Use this for creating performance metrics (Optimality_Gap%, Time_Ratio)

2. duplicates_removed
   - 399 rows that were identified as duplicates
   - Columns: All columns from processed_data, showing what was removed
   - Use to verify duplicates make sense

3. completion_by_class
   - Summary showing SCFLP: 75% complete, SCMND: 36% complete
   - Breakdown of expected vs actual by problem class
   - Helps identify which problem class needs priority for remaining runs

4. missing_experiments
   - List of all 393 missing experiment combinations
   - Each row shows what configuration was missing for which instance
   - Use to know exactly what needs to be re-run

5. SCFLP_matrix and SCMND_matrix
   - Rows = instances, Columns = configuration signatures
   - Values = 1 if experiment done, 0 if missing
   - Visual matrix showing coverage across instances x configurations
   - Darker blocks (1s) show completed, blank (0s) show missing

6. summary_stats
   - Overall statistics about your data
   - Method distribution (how many Exact vs ML runs)
   - Completion metrics


# =============================================================================
# RECOMMENDED NEXT STEPS
# =============================================================================

STEP 1: VERIFY DUPLICATES
  - Open duplicates_removed sheet
  - Visually check a few rows - do they represent actual re-runs from different job submissions?
  - If yes, your duplicate identification is correct
  
STEP 2: FOCUS ON SCMND (lowest completion at 36%)
  - Open missing_experiments sheet
  - Filter for problem_class = "SCMND"
  - Note which configurations are missing
  - These are candidates for immediate re-submission

STEP 3: PRIORITIZE INSTANCES
  - Open SCFLP_matrix and SCMND_matrix sheets
  - Identify instances with blank columns (missing configurations)
  - Target re-runs to fill the most gaps with fewest submissions

STEP 4: UPDATE SUBMIT.SH FOR RE-RUNS
  - Create a new submit.sh with only the missing configurations
  - Use missing_experiments.csv as a reference
  - This ensures you don't re-run completed work

STEP 5: RE-AGGREGATE AFTER NEW RUNS
  - Once new experiments complete, add their solution_log.xlsx to your aggregate
  - Re-run remaining_experimets.py to verify improved completion
  - Repeat until completion rate reaches 100% or acceptable threshold


# =============================================================================
# TROUBLESHOOTING COMMON QUESTIONS
# =============================================================================

Q: Why are there duplicates in my data?
A: Your runs may have restarted or some jobs were resubmitted. The aggregation process 
   captured all solution_log.xlsx files from all runs without deduplication.

Q: Why didn't all instances get all configurations?
A: Some of your SLURM jobs likely timed out or stopped partway through. The array task 
   queuing in submit.sh may have been interrupted, leaving some configurations incomplete.

Q: What should I do with the missing_experiments.csv?
A: Use it to create a new, targeted submit.sh that only runs the missing combinations. 
   This ensures efficient use of compute resources.

Q: How do I know if 66% completion is acceptable?
A: This depends on your research goals. If your goal is statistical analysis, aim for >90% 
   on all problem classes. For exploratory work, 66% may be sufficient. SCMND at 36% is 
   notably incomplete.

Q: What if duplicates have different objective values?
A: This shouldn't happen in deterministic experiments. If it does, you may have data quality 
   issues. Investigate the matching instances in duplicates_removed sheet.


# =============================================================================
# CONFIGURATION & CUSTOMIZATION
# =============================================================================

To modify the script for different analyses:

INPUT FILE:
  INPUT_FILE = r"path/to/your/Aggregate_of_experiments.xlsx"
  
OUTPUT DIRECTORY:
  OUTPUT_DIR = r"path/to/output"
  
DUPLICATE DETECTION COLUMNS:
  DUPLICATE_SUBSET = [
      "instance_filename",
      "method_category",
      ... (add/remove columns as needed)
  ]

All user-facing configurations are at the top of the script under "CONFIGURATION" section.


# =============================================================================
# INTEGRATION WITH alliance.ipynb
# =============================================================================

This script complements alliance.ipynb which handles:
  - Detailed performance metric calculations (Optimality_Gap%, Time_Ratio)
  - Advanced data analysis and visualization
  - Summary statistics by method type
  - Sensitivity analysis

WORKFLOW:
  1. remaining_experimets.py    → Identify what's done, what's missing, what's duplicate
  2. [Re-run missing experiments]
  3. Re-aggregate results
  4. Re-run remaining_experimets.py → Verify new completion status
  5. alliance.ipynb              → Once satisfied with completion, analyze performance


# =============================================================================
# KEY STATISTICS FOR YOUR PROJECT
# =============================================================================

Current Status (1161 raw rows):
  • 762 unique experiments (after deduplication)
  • 399 duplicate occurrences
  • 393 missing experiments
  • 66.0% overall completion
  • 75.2% SCFLP completion
  • 36.4% SCMND completion (priority for re-runs)

Problem Classes Present:
  • SCFLP: 16 unique instances, 39 configurations
  • SCMND: 5 unique instances, 31 configurations

Method Distribution (in processed_data):
  • Exact: 32 runs
  • ML: 1129 runs

Instances Needing Most Attention:
  • All SUMND instances (under SCMND): 35-51 missing configs each
  • SCFLP p50 variants: 21-22 missing configs each
  
"""
