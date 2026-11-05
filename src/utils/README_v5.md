LABD publication analysis — version 5

Purpose

This is the consolidated publication-analysis workflow for the LABD paper. A single Python script generates:

all established Version 5 tables and figures;

the revised versions of the existing boxplots under their original filenames;

violin-plot companions;

companion figures with sample sizes shown through the _plus_num_obs suffix;

the additional time, cut-count, and cut-share analyses;

figure-source CSV files;

analysis-ready datasets;

LaTeX table and section drafts;

diagnostics and an output manifest; and

a ZIP archive of the complete generated output folder.

The notebook does not contain a separate implementation. It runs the same consolidated publication_analysis_v5.py workflow, so the .py file is the single source of analysis logic.

Expected project locations

The files are designed for the following project structure:

Benders_Integration_with_ML/
├── RESULTS/
│   ├── Experiments - Digital Research Alliance of Canada/
│   │   └── Aggregate_of_experiments.xlsx
│   └── outputs_for_paper/
└── src/
    └── utils/
        ├── experiments_analysis_publication_v5.ipynb
        ├── publication_analysis_v5.py
        └── README_v5.md

The default paths are resolved relative to publication_analysis_v5.py. The workflow therefore does not depend on the current working directory and does not hard-code /mnt/c/Programming/....

Input workbook

By default, the script reads:

RESULTS/Experiments - Digital Research Alliance of Canada/Aggregate_of_experiments.xlsx

It uses worksheet:

Sheet1

Do not rename the worksheet unless EXCEL_SHEET_NAME is also updated in the script.

Optional input-path override

A different workbook can be supplied without editing the code by setting:

export LABD_EXCEL_INPUT_FILE="/absolute/path/to/Aggregate_of_experiments.xlsx"

How to run

Notebook

Open experiments_analysis_publication_v5.ipynb in VS Code or Jupyter and run all cells from a clean kernel.

Keep the notebook and publication_analysis_v5.py in the same folder. The notebook invokes the Python script directly and generates the complete output set.

Command line

From any directory:

python "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/src/utils/publication_analysis_v5.py"

Python packages

Install the required packages with:

pip install pandas numpy matplotlib openpyxl pillow jupyter ipython

No Gurobi installation is required because this workflow analyzes saved experiment results and does not rerun the optimization experiments.

Output location

Each run deletes and recreates only:

RESULTS/outputs_for_paper/publication_analysis_v5/

It does not delete the parent outputs_for_paper folder or any other analysis version.

A ZIP archive of the complete output folder is also created at:

RESULTS/outputs_for_paper/publication_analysis_v5_outputs.zip

Optional output-path override

A different output folder can be supplied without editing the script:

export LABD_OUTPUT_DIR="/absolute/path/to/publication_analysis_v5"

Generated folder structure

publication_analysis_v5/
├── 01_figures_for_overleaf/
│   └── all figure PDFs in one flat folder
├── 02_figures_by_problem/
│   ├── SCFLP/
│   │   ├── 01_main_results/
│   │   ├── 02_appendix_zoom_companions/
│   │   ├── 03_appendix_callback_policy/
│   │   ├── 04_appendix_feature_ablation/
│   │   ├── 05_feature_generation_time_share/
│   │   ├── 06_warm_up_time_share/
│   │   ├── 07_solution_time/
│   │   ├── 08_total_cuts/
│   │   └── 09_ml_cut_share/
│   └── SCMND/
│       └── the same nine analysis divisions
├── 03_figure_source_data/
│   ├── source CSVs for all figure families
│   └── by_problem/
│       ├── SCFLP/
│       └── SCMND/
├── 04_tables/
│   ├── primary, callback, and feature-ablation summaries
│   ├── additional-metric summary tables
│   ├── LaTeX table fragments
│   ├── zoom diagnostics and manifests
│   └── output_file_manifest_v5.csv
├── 05_latex/
│   ├── Results_section_draft_v5.tex
│   ├── Numerical_appendix_draft_v5.tex
│   ├── Manuscript_tables_v5.tex
│   └── Appendix_tables_v5.tex
├── 06_analysis_data/
│   ├── cleaned_deduplicated_data_v5.csv
│   ├── removed_duplicate_rows_v5.csv
│   ├── primary_analysis_data_v5.csv
│   ├── callback_robustness_data_v5.csv
│   └── feature_ablation_data_v5.csv
└── 07_diagnostics/
    ├── analysis_diagnostics_v5.json
    ├── runtime_policy_v5.txt
    ├── additional_metric_definitions_v5.txt
    └── visualization_rules_v5.txt

Existing and new figure families

The workflow preserves the original filenames of the established Version 5 figures so that they can replace the previous PDFs without changing existing LaTeX references.

Examples:

main__scflp__standard_bd_solved__num_subproblems_v5.pdf
main__scmnd__standard_bd_time_limited__selected_subproblems_ratio_v5.pdf
main__scmnd__standard_bd_solved__num_subproblems_zoom_v5.pdf
appendix_callback__scflp__standard_bd_solved__num_subproblems_v5.pdf
appendix_features__scmnd__standard_bd_time_limited__selected_subproblems_ratio_v5.pdf

Additional families use descriptive prefixes:

feature_generation_time_ratio__...
warm_up_time_ratio__...
solution_time__...
total_cuts__...
ml_cut_share__...

Use:

04_tables/output_file_manifest_v5.csv

to locate every generated figure and its organized copy.

Boxplot and violin-plot versions

For each applicable analysis, the workflow generates:

a boxplot version;

a violin-plot version;

a boxplot companion with _plus_num_obs; and

a violin-plot companion with _plus_num_obs.

For established boxplots, the original filename is preserved. Examples:

main__scflp__standard_bd_solved__num_subproblems_v5.pdf
main__scflp__standard_bd_solved__num_subproblems_v5_plus_num_obs.pdf
main__scflp__standard_bd_solved__num_subproblems_violin_v5.pdf
main__scflp__standard_bd_solved__num_subproblems_violin_v5_plus_num_obs.pdf

Visualization rules

The workflow applies the following rules consistently:

observation points are not overlaid on boxplots or violin plots;

all boxplot median lines are black;

methods and variants are distinguished by box or violin outline/fill colors rather than point colors or marker shapes;

y-axis numerical tick labels are explicitly enabled and adaptively formatted;

percentage signs in plot titles and axis labels appear as %, without a visible backslash;

redundant problem/status supertitles are removed;

each figure uses one complete title that states the metric and grouping dimension, such as:

Warm-up time share (%) by |M|
Feature-generation time share (%) by |M_e|/|M|
Solution time (sec) by method

the objective and runtime notation used in figures is Δ_a and τ_a, without the superscript BD;

full-range main figures are retained when extreme observations are present;

automatically flagged compressed figures receive zoomed appendix companions.

Additional metrics

Feature-generation time share

feature_generation_time_ratio
    = feature_gen_time_sec / solution_time_sec

The plotted percentage is:

feature_generation_time_ratio_pct
    = 100 × feature_generation_time_ratio

This analysis applies to LABD--Tree and LABD--Cluster.

Warm-up time share

warm_up_time_ratio
    = phase1_execution_time_sec / solution_time_sec

The plotted percentage is:

warm_up_time_ratio_pct
    = 100 × warm_up_time_ratio

This analysis applies to LABD--Tree and LABD--Cluster.

Solution time

Solution-time figures and tables use:

solution_time_sec

They are generated as separate figures. The existing two-panel objective/runtime figures remain two-panel figures; solution time is not added as a third subplot.

Total cuts

Blank cells in the component columns are treated as zero. The metric is:

total_cuts
    = phase0_cuts_added
    + cuts_mip_trial
    + cuts_rel_trial
    + cuts_mip_selected
    + cuts_rel_selected
    + cuts_mip_unselected_total
    + cuts_rel_unselected_total

Total-cut analyses include Standard BD, LABD--Tree, LABD--Cluster, and RND.

ML-generated cut share

ml_cut_share
    = (cuts_mip_ml + cuts_rel_ml) / total_cuts

The plotted percentage is:

ml_cut_share_pct
    = 100 × ml_cut_share

The ratio is left missing when total_cuts is zero. The analysis is generated for LABD--Tree, LABD--Cluster, and RND, as specified in the current workflow.

Treatment of time-limited runtimes

Do not replace solution_time_sec with 10,800 when:

stopping_reason = TIME_LIMIT

The 10,800-second limit applies to the solver optimization call. In contrast, solution_time_sec records end-to-end elapsed time and includes work performed outside the solver. It can therefore correctly exceed 10,800 seconds.

The workflow preserves the recorded values and writes a run-specific verification summary to:

07_diagnostics/runtime_policy_v5.txt

PDF export mode

Default: fast PDF generation

By default, the workflow creates high-resolution raster-backed PDF files. This is substantially faster when generating hundreds of figures and the resulting PDFs can be uploaded to Overleaf.

Direct vector-PDF export

To request direct vector PDFs, set the following environment variable before running the script or launching Jupyter from that terminal:

export LABD_FAST_RASTER_PDF=0

Vector export preserves the same filenames and directory structure but can take substantially longer.

Optional PNG export

PDF is the default publication format. To also create PNG copies, change:

EXPORT_PNG = False

to:

EXPORT_PNG = True

Overleaf use

The flat folder:

01_figures_for_overleaf/

contains every generated figure PDF.

The automatically generated LaTeX drafts under 05_latex/ currently reference:

Figures/Publication_v5/<figure-name>.pdf

Therefore, to use those generated drafts without editing their figure paths, copy all PDFs into:

Figures/Publication_v5/

If the current Overleaf manuscript already references figures through a different folder such as:

pdf/<figure-name>.pdf

copy the PDFs into pdf/ and retain the existing manuscript references. Do not simultaneously use both path conventions for the same figure set.

Important cautions

The version-specific output folder is overwritten on every run. Keep manual manuscript edits in Overleaf or in separately named files, not inside publication_analysis_v5/.

The primary analysis excludes SCFLP instance p51 and retains p4, p5, p10, p20, and p50.

The SCMND analysis excludes r5.2 and retains r1.1, r1.2, r2.1, r2.2, r4.1, and r4.2.

The retained values are |M| ∈ {100, 300, 500} and |M_e|/|M| ∈ {5%, 15%, 25%}.

The workflow does not include KNN or 700-subproblem experiments.

Before final submission, reconcile the exact SCMND RND dual-aggregation rule in the optimization source code with the manuscript description.

The notebook and script should remain in the same folder. Update the Python script only; the notebook simply invokes it.