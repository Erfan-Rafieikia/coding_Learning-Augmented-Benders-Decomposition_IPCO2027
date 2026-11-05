from __future__ import annotations

import json
import math
import re
import shutil
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# publication_analysis_v5.py is stored under <project>/src/utils, so parents[1]
# resolves to the project root regardless of the current working directory.
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXCEL_INPUT_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "Experiments - Digital Research Alliance of Canada"
    / "Aggregate_of_experiments.xlsx"
)
EXCEL_SHEET_NAME = "Sheet1"

EXPORT_PNG = False  # Set to True when raster copies are also needed.
OUTPUT_ROOT = PROJECT_ROOT / "RESULTS" / "outputs_for_paper"
# Only this version-specific folder is deleted and rebuilt on each run.
OUTPUT_DIR = OUTPUT_ROOT / "publication_analysis_v5"
FIGURE_DIR = OUTPUT_DIR / "01_figures_for_overleaf"
ORGANIZED_FIGURE_DIR = OUTPUT_DIR / "02_figures_by_problem"
FIGURE_DATA_DIR = OUTPUT_DIR / "03_figure_source_data"
TABLE_DIR = OUTPUT_DIR / "04_tables"
LATEX_DIR = OUTPUT_DIR / "05_latex"
ANALYSIS_DATA_DIR = OUTPUT_DIR / "06_analysis_data"
DIAGNOSTICS_DIR = OUTPUT_DIR / "07_diagnostics"

INSTANCES_TO_INCLUDE = [
    "p4_stochastic.txt", "p5_stochastic.txt", "p10_stochastic.txt",
    "p20_stochastic.txt", "p50_stochastic.txt",
    "r01.1.stoch.json", "r01.2.stoch.json", "r02.1.stoch.json",
    "r02.2.stoch.json", "r04.1.stoch.json", "r04.2.stoch.json",
]
DISPLAY_INSTANCE = {
    "p4_stochastic.txt": "p4", "p5_stochastic.txt": "p5",
    "p10_stochastic.txt": "p10", "p20_stochastic.txt": "p20",
    "p50_stochastic.txt": "p50",
    "r01.1.stoch.json": "r1.1", "r01.2.stoch.json": "r1.2",
    "r02.1.stoch.json": "r2.1", "r02.2.stoch.json": "r2.2",
    "r04.1.stoch.json": "r4.1", "r04.2.stoch.json": "r4.2",
}
SCFLP_METADATA = [
    ("SCFLP", "p4", 10, 50), ("SCFLP", "p5", 10, 50),
    ("SCFLP", "p10", 10, 50), ("SCFLP", "p20", 20, 50),
    ("SCFLP", "p50", 10, 100),
]
SCMND_METADATA = [
    ("SCMND", "r1.1", 10, 35, 10, 1, 1),
    ("SCMND", "r1.2", 10, 35, 10, 5, 1),
    ("SCMND", "r2.1", 10, 35, 25, 1, 1),
    ("SCMND", "r2.2", 10, 35, 25, 5, 1),
    ("SCMND", "r4.1", 10, 60, 10, 1, 1),
    ("SCMND", "r4.2", 10, 60, 10, 5, 1),
]

NUM_SUBPROBLEMS = [100, 300, 500]
EXACT_RATIOS = [0.05, 0.15, 0.25]
PRIMARY_CALLBACK = "ML_at_both"
SUPPLEMENTARY_CALLBACK = "ML_at_LP_exact_at_integer"
PRIMARY_FEATURE_SPEC = "features_only"
FEATURE_SPECS = ["features_only", "covariates_only", "features_plus_covariates"]
PREDICTORS = ["TREE", "CLUSTERING"]

DUPLICATE_SUBSET = [
    "problem_class", "deterministic_instance", "num_subproblems",
    "method_category", "ml_method_type", "ml_callback_type",
    "exact_callback_type", "predictor_type", "use_prediction",
    "prediction_method", "selected_subproblems_ratio", "n_trials",
    "m_trials", "n_walks", "walk_length", "feature_dim", "window",
    "min_count", "sg", "use_covariates",
]
DUPLICATE_RANDOM_STATE = 42
KEY_COLUMNS = ["problem_class", "deterministic_instance", "num_subproblems"]

METHOD_ORDER = ["Standard BD", "LABD--Tree", "LABD--Cluster", "RND"]
NONBASELINE_METHOD_ORDER = ["LABD--Tree", "LABD--Cluster", "RND"]
LABD_METHODS = ["LABD--Tree", "LABD--Cluster"]
METHOD_MARKERS = {"LABD--Tree": "o", "LABD--Cluster": "s", "RND": "^"}
STRATUM_ORDER = ["Standard BD solved", "Standard BD time-limited"]
INSTANCE_ORDER = {name: i for i, name in enumerate(["p4", "p5", "p10", "p20", "p50", "r1.1", "r1.2", "r2.1", "r2.2", "r4.1", "r4.2"])}
FEATURE_LABELS = {
    "features_only": "R",
    "covariates_only": "C",
    "features_plus_covariates": "R+C",
}
CALLBACK_LABELS = {
    PRIMARY_CALLBACK: "Predicted cuts at LP and integer solutions",
    SUPPLEMENTARY_CALLBACK: "Predicted cuts at LP; exact cuts at integer solutions",
}
CALLBACK_SHORT = {
    PRIMARY_CALLBACK: "ML at both",
    SUPPLEMENTARY_CALLBACK: "Exact at integer",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Robust diagnostic for figures whose central distribution is visually compressed
# by a small number of extreme observations. The full-range figure is always
# retained; flagged main-text figures receive an appendix zoom companion.
SQUASH_CENTRAL_RANGE_THRESHOLD = 0.22
SQUASH_TAIL_IQR_THRESHOLD = 5.0
ZOOM_FIGURES: dict[str, dict[str, object]] = {}
ZOOM_DIAGNOSTICS: list[dict[str, object]] = []


def _panel_diagnostic(values: pd.Series | np.ndarray, metric: str) -> dict[str, object]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    result: dict[str, object] = {"metric": metric, "n": int(len(arr)), "flagged": False, "zoom_lower": np.nan, "zoom_upper": np.nan}
    if len(arr) < 5:
        return result
    ymin, ymax = float(np.min(arr)), float(np.max(arr))
    full_range = ymax - ymin
    if not np.isfinite(full_range) or full_range <= 0:
        return result
    q05, q25, q50, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
    iqr = float(q75 - q25)
    central_fraction = float((q95 - q05) / full_range)
    denom = max(iqr, np.finfo(float).eps)
    upper_tail = float((ymax - q75) / denom)
    lower_tail = float((q25 - ymin) / denom)
    flagged = central_fraction < SQUASH_CENTRAL_RANGE_THRESHOLD and max(upper_tail, lower_tail) > SQUASH_TAIL_IQR_THRESHOLD
    result.update({
        "min": ymin, "max": ymax, "q05": float(q05), "q25": float(q25), "median": float(q50), "q75": float(q75), "q95": float(q95),
        "central_range_fraction": central_fraction, "upper_tail_iqr": upper_tail, "lower_tail_iqr": lower_tail, "flagged": bool(flagged),
    })
    if not flagged:
        return result
    if iqr > 0:
        lower = max(ymin, float(q25 - 1.75 * iqr))
        upper = min(ymax, float(q75 + 1.75 * iqr))
    else:
        lower, upper = float(q05), float(q95)
    # Include the Standard BD reference and a small visual margin.
    reference = 0.0 if metric == "relative_objective_difference_pct" else 1.0
    lower = min(lower, reference)
    upper = max(upper, reference)
    span = upper - lower
    if span <= 0:
        span = max(abs(reference), 1.0) * 0.1
    margin = 0.08 * span
    result["zoom_lower"] = lower - margin
    result["zoom_upper"] = upper + margin
    return result


def _zoom_label(stem: str) -> str:
    return "fig:zoom-" + stem.replace("_v5", "").replace("__", "-").replace("_", "-")

# =============================================================================
# Cleaning and classification
# =============================================================================
def extract_instance_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\\\\", "/", regex=False)
        .str.rsplit("/", n=1)
        .str[-1]
    )


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate the original notebook's documented duplicate-selection rules."""
    df = df.copy()
    duplicate_subset = [c for c in DUPLICATE_SUBSET if c in df.columns]
    duplicate_mask = df.duplicated(subset=duplicate_subset, keep=False)
    if not duplicate_mask.any():
        return df.reset_index(drop=True), df.iloc[0:0].copy()

    kept_indices: list[int] = []
    removed_indices: list[int] = []
    groups = df[duplicate_mask].groupby(duplicate_subset, dropna=False, sort=False)
    for _, group_df in groups:
        candidate_df = group_df

        if "time_limit_sec" in candidate_df.columns:
            time_limit = pd.to_numeric(candidate_df["time_limit_sec"], errors="coerce")
            rows_10800 = candidate_df[time_limit == 10800]
            rows_3600 = candidate_df[time_limit == 3600]
            rows_null = candidate_df[time_limit.isna()]
            if not rows_10800.empty:
                candidate_df = rows_10800
            elif not rows_3600.empty:
                candidate_df = rows_3600
            elif len(rows_null) == len(candidate_df):
                candidate_df = rows_null

        if "MIPGap_target" in candidate_df.columns:
            mipgap = pd.to_numeric(candidate_df["MIPGap_target"], errors="coerce")
            preferred = candidate_df[mipgap.notna() & (mipgap != 0)]
            if not preferred.empty:
                candidate_df = preferred

        kept_idx = candidate_df.sample(n=1, random_state=DUPLICATE_RANDOM_STATE).index[0]
        kept_indices.append(kept_idx)
        removed_indices.extend(idx for idx in group_df.index if idx != kept_idx)

    unique_rows = df[~duplicate_mask]
    kept_duplicate_rows = df.loc[kept_indices]
    deduplicated = pd.concat([unique_rows, kept_duplicate_rows]).sort_index().reset_index(drop=True)
    removed = df.loc[removed_indices].sort_index().reset_index(drop=True)
    return deduplicated, removed


def normalize_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Per the experiment log convention, missing stopping_reason means OPTIMAL.
    out["stopping_reason"] = out["stopping_reason"].fillna("OPTIMAL")
    out["is_time_limit"] = out["stopping_reason"].ne("OPTIMAL")
    unexpected = sorted(set(out.loc[out["is_time_limit"], "stopping_reason"].dropna()) - {"TIME_LIMIT"})
    if unexpected:
        raise ValueError(f"Unexpected stopping reasons after cleaning: {unexpected}")
    return out


def add_method_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method"] = np.select(
        [
            out["method_category"].eq("Exact"),
            out["ml_method_type"].eq("ML_Augmented_BD") & out["predictor_type"].eq("TREE"),
            out["ml_method_type"].eq("ML_Augmented_BD") & out["predictor_type"].eq("CLUSTERING"),
            out["ml_method_type"].eq("Random_Sampling"),
        ],
        ["Standard BD", "LABD--Tree", "LABD--Cluster", "RND"],
        default="Other",
    )
    out["instance"] = out["deterministic_instance"].map(DISPLAY_INSTANCE).fillna(out["deterministic_instance"])
    return out


def read_experiment_data() -> pd.DataFrame:
    """Read the authoritative aggregate experiment workbook directly."""
    if not EXCEL_INPUT_FILE.exists():
        raise FileNotFoundError(
            "The aggregate experiment workbook was not found at the expected path:\n"
            f"  {EXCEL_INPUT_FILE}\n"
            "Keep the project folder structure described in README_v5.md or update "
            "EXCEL_INPUT_FILE in the configuration block."
        )
    try:
        return pd.read_excel(
            EXCEL_INPUT_FILE,
            sheet_name=EXCEL_SHEET_NAME,
            engine="openpyxl",
        )
    except ValueError as exc:
        raise ValueError(
            f"Worksheet {EXCEL_SHEET_NAME!r} was not found in {EXCEL_INPUT_FILE}."
        ) from exc


def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_experiment_data()
    raw["deterministic_instance"] = extract_instance_name(raw["instance_filename"])
    selected = raw[raw["deterministic_instance"].isin(INSTANCES_TO_INCLUDE)].copy()
    deduplicated, removed = remove_duplicates(selected)
    deduplicated = deduplicated[deduplicated["num_subproblems"].isin(NUM_SUBPROBLEMS)].copy()
    deduplicated = normalize_status(deduplicated)
    deduplicated = add_method_label(deduplicated)
    deduplicated["selected_subproblems_ratio"] = pd.to_numeric(
        deduplicated["selected_subproblems_ratio"], errors="coerce"
    )
    return deduplicated, removed


def select_primary(df: pd.DataFrame) -> pd.DataFrame:
    exact = df["method_category"].eq("Exact")
    labd = (
        df["method_category"].eq("ML")
        & df["ml_method_type"].eq("ML_Augmented_BD")
        & df["ml_callback_type"].eq(PRIMARY_CALLBACK)
        & df["use_covariates"].eq(PRIMARY_FEATURE_SPEC)
        & df["predictor_type"].isin(PREDICTORS)
        & df["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    )
    rnd = (
        df["method_category"].eq("ML")
        & df["ml_method_type"].eq("Random_Sampling")
        & df["ml_callback_type"].eq(PRIMARY_CALLBACK)
        & df["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    )
    return df[exact | labd | rnd].copy()


def select_callback_robustness(df: pd.DataFrame) -> pd.DataFrame:
    exact = df["method_category"].eq("Exact")
    labd = (
        df["method_category"].eq("ML")
        & df["ml_method_type"].eq("ML_Augmented_BD")
        & df["ml_callback_type"].isin([PRIMARY_CALLBACK, SUPPLEMENTARY_CALLBACK])
        & df["use_covariates"].eq(PRIMARY_FEATURE_SPEC)
        & df["predictor_type"].isin(PREDICTORS)
        & df["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    )
    rnd = (
        df["method_category"].eq("ML")
        & df["ml_method_type"].eq("Random_Sampling")
        & df["ml_callback_type"].isin([PRIMARY_CALLBACK, SUPPLEMENTARY_CALLBACK])
        & df["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    )
    out = df[exact | labd | rnd].copy()
    out["callback_policy"] = out["ml_callback_type"].map(CALLBACK_LABELS)
    out["callback_short"] = out["ml_callback_type"].map(CALLBACK_SHORT)
    return out


def select_feature_ablation(df: pd.DataFrame) -> pd.DataFrame:
    exact = df["method_category"].eq("Exact")
    labd = (
        df["method_category"].eq("ML")
        & df["ml_method_type"].eq("ML_Augmented_BD")
        & df["ml_callback_type"].eq(PRIMARY_CALLBACK)
        & df["use_covariates"].isin(FEATURE_SPECS)
        & df["predictor_type"].isin(PREDICTORS)
        & df["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    )
    out = df[exact | labd].copy()
    out["feature_specification"] = out["use_covariates"].map(FEATURE_LABELS)
    return out


def attach_standard_bd_reference(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    baseline = out[out["method_category"].eq("Exact")].copy()
    if baseline.duplicated(KEY_COLUMNS, keep=False).any():
        raise ValueError("More than one Standard BD row remains for a classification key.")

    baseline = baseline[KEY_COLUMNS + [
        "original_Prob_Obj_Value", "solution_time_sec", "stopping_reason",
        "is_time_limit", "ObjBound_LB",
    ]].rename(columns={
        "original_Prob_Obj_Value": "standard_bd_objective",
        "solution_time_sec": "standard_bd_runtime_sec",
        "stopping_reason": "standard_bd_stopping_reason",
        "is_time_limit": "standard_bd_time_limit",
        "ObjBound_LB": "standard_bd_lower_bound",
    })
    out = out.merge(baseline, on=KEY_COLUMNS, how="left", validate="many_to_one")
    if out["standard_bd_runtime_sec"].isna().any():
        raise ValueError("At least one retained row has no matched Standard BD record.")

    out["baseline_stratum"] = np.where(
        out["standard_bd_time_limit"],
        STRATUM_ORDER[1],
        STRATUM_ORDER[0],
    )
    objective = pd.to_numeric(out["original_Prob_Obj_Value"], errors="coerce")
    bd_objective = pd.to_numeric(out["standard_bd_objective"], errors="coerce")
    runtime = pd.to_numeric(out["solution_time_sec"], errors="coerce")
    bd_runtime = pd.to_numeric(out["standard_bd_runtime_sec"], errors="coerce")

    out["relative_objective_difference_pct"] = np.where(
        objective.notna() & bd_objective.notna() & bd_objective.ne(0),
        100.0 * (objective - bd_objective) / bd_objective.abs(),
        np.nan,
    )
    out["relative_runtime"] = np.where(
        runtime.notna() & bd_runtime.notna() & bd_runtime.gt(0),
        runtime / bd_runtime,
        np.nan,
    )
    out["objective_available"] = objective.notna()
    return out

# =============================================================================
# Summary tables
# =============================================================================
def _stats(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {k: np.nan for k in ["min", "max", "mean", "median", "q1", "q3"]}
    return {
        "min": float(s.min()), "max": float(s.max()),
        "mean": float(s.mean()), "median": float(s.median()),
        "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)),
    }


def build_method_summary(df: pd.DataFrame, factor: str | None = None) -> pd.DataFrame:
    grouping = ["problem_class", "baseline_stratum", "method"]
    if factor:
        grouping.append(factor)
    rows: list[dict[str, object]] = []
    working = df[~df["method"].eq("Standard BD")].copy()
    for keys, group in working.groupby(grouping, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        obj = _stats(group["relative_objective_difference_pct"])
        tim = _stats(group["relative_runtime"])
        row.update({
            "records": len(group),
            "objective_values_available": int(group["objective_available"].sum()),
            "time_limit_rate_pct": 100.0 * float(group["is_time_limit"].mean()),
            "gap_min": obj["min"], "gap_max": obj["max"],
            "gap_mean": obj["mean"], "gap_median": obj["median"],
            "gap_q1": obj["q1"], "gap_q3": obj["q3"],
            "time_min": tim["min"], "time_max": tim["max"],
            "time_mean": tim["mean"], "time_median": tim["median"],
            "time_q1": tim["q1"], "time_q3": tim["q3"],
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    method_rank = {m: i for i, m in enumerate(NONBASELINE_METHOD_ORDER)}
    result["_method_rank"] = result["method"].map(method_rank).fillna(99)
    sort_cols = ["problem_class", "baseline_stratum", "_method_rank"] + ([factor] if factor else [])
    return result.sort_values(sort_cols).drop(columns="_method_rank").reset_index(drop=True)



def build_problem_coverage(df: pd.DataFrame, problem_class: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = df[df["problem_class"].eq(problem_class)].copy()

    def summarize(group: pd.DataFrame, instance: str, nsp: float | int | None) -> dict[str, object]:
        std = group[group["method"].eq("Standard BD")]
        labd = group[group["method"].isin(LABD_METHODS)]
        rnd = group[group["method"].eq("RND")]
        row: dict[str, object] = {"instance": instance, "num_subproblems": nsp}
        for prefix, g in [("standard_bd", std), ("labd", labd), ("rnd", rnd)]:
            row[f"{prefix}_records"] = len(g)
            row[f"{prefix}_time_limit_rate_pct"] = 100.0 * float(g["is_time_limit"].mean()) if len(g) else np.nan
            row[f"{prefix}_gap_median"] = g["relative_objective_difference_pct"].median() if len(g) else np.nan
            row[f"{prefix}_gap_mean"] = g["relative_objective_difference_pct"].mean() if len(g) else np.nan
            row[f"{prefix}_time_median"] = g["relative_runtime"].median() if len(g) else np.nan
        return row

    for (instance, nsp), group in subset.groupby(["instance", "num_subproblems"], sort=True):
        rows.append(summarize(group, instance, int(nsp)))
    rows.append(summarize(subset, "Total", np.nan))

    result = pd.DataFrame(rows)
    result["_instance_order"] = result["instance"].map(INSTANCE_ORDER).fillna(999)
    return result.sort_values(["_instance_order", "num_subproblems"], na_position="last").drop(columns="_instance_order").reset_index(drop=True)

def build_dimension_summary(df: pd.DataFrame, dimension: str, factor: str | None = None) -> pd.DataFrame:
    grouping = ["problem_class", "baseline_stratum", dimension, "method"]
    if factor:
        grouping.append(factor)
    rows = []
    working = df[~df["method"].eq("Standard BD")].copy()
    for keys, group in working.groupby(grouping, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        obj = _stats(group["relative_objective_difference_pct"])
        tim = _stats(group["relative_runtime"])
        row.update({
            "records": len(group),
            "objective_values_available": int(group["objective_available"].sum()),
            "time_limit_rate_pct": 100.0 * float(group["is_time_limit"].mean()),
            "gap_min": obj["min"], "gap_max": obj["max"],
            "gap_mean": obj["mean"], "gap_median": obj["median"],
            "time_min": tim["min"], "time_max": tim["max"],
            "time_mean": tim["mean"], "time_median": tim["median"],
        })
        rows.append(row)
    return pd.DataFrame(rows)

# =============================================================================
# Plotting
# =============================================================================
def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _dimension_values(subset: pd.DataFrame, dimension: str) -> list[float]:
    vals = sorted(pd.to_numeric(subset[dimension], errors="coerce").dropna().unique().tolist())
    return vals


def _format_dimension_ticks(values: list[float], dimension: str) -> list[str]:
    if dimension == "selected_subproblems_ratio":
        return [f"{100 * x:.0f}%" for x in values]
    return [str(int(x)) for x in values]



def _dimension_label(dimension: str) -> str:
    return r"$|\mathcal{M}_e|/|\mathcal{M}|$" if dimension == "selected_subproblems_ratio" else r"$|\mathcal{M}|$"


def _draw_grouped_boxplot(
    ax: plt.Axes,
    subset: pd.DataFrame,
    *,
    dimension: str,
    metric: str,
    categories: list[str],
    category_column: str,
    markers: dict[str, str],
    reference: float,
    y_label: str,
) -> list[Line2D]:
    values = _dimension_values(subset, dimension)
    centers = np.arange(len(values), dtype=float)
    ncat = max(len(categories), 1)
    offsets = np.linspace(-0.28, 0.28, ncat)
    width = min(0.18, 0.62 / ncat)
    rng = np.random.default_rng(42)
    legend_handles: list[Line2D] = []

    for cat_idx, category in enumerate(categories):
        arrays: list[np.ndarray] = []
        positions: list[float] = []
        point_x: list[float] = []
        point_y: list[float] = []
        for center, dim_value in zip(centers, values):
            vals = subset[
                subset[category_column].eq(category)
                & pd.to_numeric(subset[dimension], errors="coerce").eq(dim_value)
            ][metric].dropna().to_numpy(dtype=float)
            if len(vals):
                arrays.append(vals)
                pos = center + offsets[cat_idx]
                positions.append(pos)
                point_x.extend((np.full(len(vals), pos) + rng.uniform(-0.025, 0.025, len(vals))).tolist())
                point_y.extend(vals.tolist())
        if arrays:
            ax.boxplot(
                arrays,
                positions=positions,
                widths=width,
                manage_ticks=False,
                showfliers=False,
                patch_artist=False,
                medianprops={"linewidth": 1.6},
            )
            scatter = ax.scatter(
                point_x,
                point_y,
                marker=markers.get(category, "o"),
                s=20,
                alpha=0.72,
                label=category,
            )
            legend_handles.append(scatter)

    ax.axhline(reference, linestyle=":" if reference == 0 else "--", linewidth=1.15, zorder=0)
    ax.set_xticks(centers)
    ax.set_xticklabels(_format_dimension_ticks(values, dimension))
    ax.set_xlabel(_dimension_label(dimension))
    ax.set_ylabel(y_label)
    if metric == "relative_runtime":
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(axis="y", which="major", alpha=0.22, linewidth=0.65)
    else:
        ax.grid(axis="y", alpha=0.20, linewidth=0.65)
    return legend_handles


def _annotate_clipped_observations(
    ax: plt.Axes,
    original: pd.DataFrame,
    *,
    dimension: str,
    metric: str,
    categories: list[str],
    category_column: str,
    lower: float,
    upper: float,
) -> None:
    values = _dimension_values(original, dimension)
    centers = np.arange(len(values), dtype=float)
    offsets = np.linspace(-0.28, 0.28, max(len(categories), 1))
    span = upper - lower
    for cat_idx, category in enumerate(categories):
        for center, dim_value in zip(centers, values):
            vals = pd.to_numeric(original[
                original[category_column].eq(category)
                & pd.to_numeric(original[dimension], errors="coerce").eq(dim_value)
            ][metric], errors="coerce").dropna()
            upper_count = int((vals > upper).sum())
            lower_count = int((vals < lower).sum())
            x = center + offsets[cat_idx]
            if upper_count:
                ax.text(x, upper - 0.025 * span, rf"$\uparrow${upper_count}", ha="center", va="top", fontsize=7.5, clip_on=True)
            if lower_count:
                ax.text(x, lower + 0.025 * span, rf"$\downarrow${lower_count}", ha="center", va="bottom", fontsize=7.5, clip_on=True)


def _render_main_composite(
    subset: pd.DataFrame,
    *,
    problem_class: str,
    stratum: str,
    dimension: str,
    destination: Path,
    objective_limits: tuple[float, float] | None = None,
    runtime_limits: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    objective_plot = subset.copy()
    runtime_plot = subset.copy()
    if objective_limits is not None:
        lo, hi = objective_limits
        objective_plot.loc[~objective_plot["relative_objective_difference_pct"].between(lo, hi), "relative_objective_difference_pct"] = np.nan
    if runtime_limits is not None:
        lo, hi = runtime_limits
        runtime_plot.loc[~runtime_plot["relative_runtime"].between(lo, hi), "relative_runtime"] = np.nan
    h1 = _draw_grouped_boxplot(
        axes[0], objective_plot.dropna(subset=["relative_objective_difference_pct"]),
        dimension=dimension, metric="relative_objective_difference_pct",
        categories=NONBASELINE_METHOD_ORDER, category_column="method",
        markers=METHOD_MARKERS, reference=0.0,
        y_label=r"$\Delta_a^{\mathrm{BD}}$ (\%)",
    )
    _draw_grouped_boxplot(
        axes[1], runtime_plot.dropna(subset=["relative_runtime"]),
        dimension=dimension, metric="relative_runtime",
        categories=NONBASELINE_METHOD_ORDER, category_column="method",
        markers=METHOD_MARKERS, reference=1.0,
        y_label=r"$\tau_a^{\mathrm{BD}}$",
    )
    if objective_limits is not None:
        axes[0].set_ylim(*objective_limits)
        _annotate_clipped_observations(
            axes[0], subset, dimension=dimension, metric="relative_objective_difference_pct",
            categories=NONBASELINE_METHOD_ORDER, category_column="method",
            lower=objective_limits[0], upper=objective_limits[1],
        )
        axes[0].set_title("True objective value (zoomed)")
    else:
        axes[0].set_title("True objective value")
    if runtime_limits is not None:
        axes[1].set_ylim(*runtime_limits)
        _annotate_clipped_observations(
            axes[1], subset, dimension=dimension, metric="relative_runtime",
            categories=NONBASELINE_METHOD_ORDER, category_column="method",
            lower=runtime_limits[0], upper=runtime_limits[1],
        )
        axes[1].set_title("Computational time (zoomed)")
    else:
        axes[1].set_title("Computational time")
    title_status = "Standard BD solved" if stratum == STRATUM_ORDER[0] else "Standard BD time-limited"
    fig.suptitle(f"{problem_class}: {title_status}", y=1.02)
    if h1:
        ref = Line2D([], [], linestyle="--", label="Standard BD reference")
        fig.legend(handles=h1 + [ref], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(destination, bbox_inches="tight")
    if EXPORT_PNG:
        fig.savefig(destination.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def make_main_composite_figure(
    df: pd.DataFrame,
    *,
    problem_class: str,
    stratum: str,
    dimension: str,
) -> str | None:
    subset = df[
        df["problem_class"].eq(problem_class)
        & df["baseline_stratum"].eq(stratum)
        & df["method"].isin(NONBASELINE_METHOD_ORDER)
    ].copy()
    if subset.empty:
        return None

    stem = "__".join(["main", problem_class.lower(), _safe_slug(stratum), dimension]) + "_v5"
    cols = KEY_COLUMNS + [
        "instance", "baseline_stratum", "method", "selected_subproblems_ratio",
        "stopping_reason", "is_time_limit", "relative_objective_difference_pct",
        "relative_runtime", "objective_available",
    ]
    cols = list(dict.fromkeys(c for c in cols if c in subset.columns))
    subset[cols].to_csv(FIGURE_DATA_DIR / f"{stem}.csv", index=False)

    _render_main_composite(
        subset, problem_class=problem_class, stratum=stratum, dimension=dimension,
        destination=FIGURE_DIR / f"{stem}.pdf",
    )

    objective_diag = _panel_diagnostic(subset["relative_objective_difference_pct"], "relative_objective_difference_pct")
    runtime_diag = _panel_diagnostic(subset["relative_runtime"], "relative_runtime")
    for diag in [objective_diag, runtime_diag]:
        ZOOM_DIAGNOSTICS.append({"stem": stem, "problem_class": problem_class, "baseline_stratum": stratum, "dimension": dimension, **diag})
    if bool(objective_diag["flagged"]) or bool(runtime_diag["flagged"]):
        zoom_stem = stem.replace("_v5", "_zoom_v5")
        objective_limits = (float(objective_diag["zoom_lower"]), float(objective_diag["zoom_upper"])) if objective_diag["flagged"] else None
        runtime_limits = (float(runtime_diag["zoom_lower"]), float(runtime_diag["zoom_upper"])) if runtime_diag["flagged"] else None
        _render_main_composite(
            subset, problem_class=problem_class, stratum=stratum, dimension=dimension,
            destination=FIGURE_DIR / f"{zoom_stem}.pdf",
            objective_limits=objective_limits, runtime_limits=runtime_limits,
        )
        ZOOM_FIGURES[stem] = {
            "zoom_stem": zoom_stem,
            "label": _zoom_label(stem),
            "objective_zoomed": bool(objective_diag["flagged"]),
            "runtime_zoomed": bool(runtime_diag["flagged"]),
            "objective_limits": objective_limits,
            "runtime_limits": runtime_limits,
            "problem_class": problem_class,
            "baseline_stratum": stratum,
            "dimension": dimension,
        }
    return stem

def _draw_variant_panel(
    ax: plt.Axes,
    subset: pd.DataFrame,
    *,
    dimension: str,
    metric: str,
    variant_column: str,
    variants: list[str],
    variant_markers: dict[str, str],
    reference: float,
    y_label: str,
) -> list[Line2D]:
    return _draw_grouped_boxplot(
        ax,
        subset,
        dimension=dimension,
        metric=metric,
        categories=variants,
        category_column=variant_column,
        markers=variant_markers,
        reference=reference,
        y_label=y_label,
    )


def make_appendix_callback_figure(
    df: pd.DataFrame,
    *,
    problem_class: str,
    stratum: str,
    dimension: str,
) -> str | None:
    subset = df[
        df["problem_class"].eq(problem_class)
        & df["baseline_stratum"].eq(stratum)
        & df["method"].isin(NONBASELINE_METHOD_ORDER)
    ].copy()
    if subset.empty:
        return None
    stem = "__".join(["appendix_callback", problem_class.lower(), _safe_slug(stratum), dimension]) + "_v5"
    cols = KEY_COLUMNS + [
        "instance", "baseline_stratum", "method", "callback_policy", "callback_short",
        "selected_subproblems_ratio", "is_time_limit", "objective_available",
        "relative_objective_difference_pct", "relative_runtime",
    ]
    subset[cols].to_csv(FIGURE_DATA_DIR / f"{stem}.csv", index=False)

    methods = NONBASELINE_METHOD_ORDER
    variants = [CALLBACK_SHORT[PRIMARY_CALLBACK], CALLBACK_SHORT[SUPPLEMENTARY_CALLBACK]]
    markers = {variants[0]: "o", variants[1]: "x"}
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.9), sharex=False)
    legend_handles: list[Line2D] = []
    for col_idx, method in enumerate(methods):
        msub = subset[subset["method"].eq(method)]
        h = _draw_variant_panel(
            axes[0, col_idx], msub.dropna(subset=["relative_objective_difference_pct"]),
            dimension=dimension, metric="relative_objective_difference_pct",
            variant_column="callback_short", variants=variants, variant_markers=markers,
            reference=0.0,
            y_label=r"$\Delta_a^{\mathrm{BD}}$ (\%)" if col_idx == 0 else "",
        )
        _draw_variant_panel(
            axes[1, col_idx], msub.dropna(subset=["relative_runtime"]),
            dimension=dimension, metric="relative_runtime",
            variant_column="callback_short", variants=variants, variant_markers=markers,
            reference=1.0,
            y_label=r"$\tau_a^{\mathrm{BD}}$" if col_idx == 0 else "",
        )
        axes[0, col_idx].set_title(method)
        if h and not legend_handles:
            legend_handles = h
    fig.suptitle(f"Callback-policy robustness - {problem_class}: {stratum}", y=1.01)
    if legend_handles:
        ref = Line2D([], [], linestyle=":", label="Standard BD reference")
        fig.legend(handles=legend_handles + [ref], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    if EXPORT_PNG:
        fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    return stem


def make_appendix_feature_figure(
    df: pd.DataFrame,
    *,
    problem_class: str,
    stratum: str,
    dimension: str,
) -> str | None:
    subset = df[
        df["problem_class"].eq(problem_class)
        & df["baseline_stratum"].eq(stratum)
        & df["method"].isin(LABD_METHODS)
    ].copy()
    if subset.empty:
        return None
    stem = "__".join(["appendix_features", problem_class.lower(), _safe_slug(stratum), dimension]) + "_v5"
    cols = KEY_COLUMNS + [
        "instance", "baseline_stratum", "method", "feature_specification",
        "selected_subproblems_ratio", "is_time_limit", "objective_available",
        "relative_objective_difference_pct", "relative_runtime",
    ]
    subset[cols].to_csv(FIGURE_DATA_DIR / f"{stem}.csv", index=False)

    methods = LABD_METHODS
    variants = ["R", "C", "R+C"]
    markers = {"R": "o", "C": "s", "R+C": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharex=False)
    legend_handles: list[Line2D] = []
    for col_idx, method in enumerate(methods):
        msub = subset[subset["method"].eq(method)]
        h = _draw_variant_panel(
            axes[0, col_idx], msub.dropna(subset=["relative_objective_difference_pct"]),
            dimension=dimension, metric="relative_objective_difference_pct",
            variant_column="feature_specification", variants=variants, variant_markers=markers,
            reference=0.0,
            y_label=r"$\Delta_a^{\mathrm{BD}}$ (\%)" if col_idx == 0 else "",
        )
        _draw_variant_panel(
            axes[1, col_idx], msub.dropna(subset=["relative_runtime"]),
            dimension=dimension, metric="relative_runtime",
            variant_column="feature_specification", variants=variants, variant_markers=markers,
            reference=1.0,
            y_label=r"$\tau_a^{\mathrm{BD}}$" if col_idx == 0 else "",
        )
        axes[0, col_idx].set_title(method)
        if h and not legend_handles:
            legend_handles = h
    fig.suptitle(f"Feature-input robustness - {problem_class}: {stratum}", y=1.01)
    if legend_handles:
        ref = Line2D([], [], linestyle=":", label="Standard BD reference")
        fig.legend(handles=legend_handles + [ref], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    if EXPORT_PNG:
        fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    return stem

# =============================================================================
# LaTeX helpers
# =============================================================================
def _fmt(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def _escape(text: object) -> str:
    return str(text).replace("&", r"\&").replace("_", r"\_")



def latex_benchmark_table() -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Structural characteristics of the deterministic benchmark instances retained in the analysis.}",
        r"\label{tab:benchmark-instances}",
        r"\small",
        r"\textbf{(a) SCFLP instances}\par\smallskip",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Problem & Instance & $|I|$ & $|J|$ & $|\mathcal M|$ \\",
        r"\midrule",
    ]
    for problem, instance, i, j in SCFLP_METADATA:
        lines.append(f"{problem} & {instance} & {i} & {j} & 100, 300, 500 " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\textbf{(b) SCMND instances}\par\smallskip",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Problem & Instance & $|V|$ & $|A|$ & $|K|$ & Fixed cost & Capacity & $|\mathcal M|$ \\",
        r"\midrule",
    ]
    for problem, instance, v, a, k, f, c in SCMND_METADATA:
        lines.append(f"{problem} & {instance} & {v} & {a} & {k} & {f} & {c} & 100, 300, 500 " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_coverage_table(coverage: pd.DataFrame, problem: str, label: str, define_columns: bool = False) -> str:
    note = (
        r"Runs is the number of evaluated configurations and TL (\%) is the percentage reaching the time limit. "
        r"Each objective cell reports median/mean $\Delta_a^{\mathrm{BD}}$; runtime cells report median $\tau_a^{\mathrm{BD}}$. "
        r"LABD combines the Tree and Cluster predictors."
        if define_columns else
        r"Notation follows Table~\ref{tab:scflp-coverage}; LABD combines the Tree and Cluster predictors."
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Coverage, termination, and performance for the primary {problem} experiments. {note}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Standard BD} & \multicolumn{4}{c}{LABD} & \multicolumn{4}{c}{RND} " + r"\\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
        r"Instance & $|\mathcal M|$ & Runs & TL (\%) & Runs & TL (\%) & $\Delta_a^{\mathrm{BD}}$ Med./Mean (\%) & $\tau_a^{\mathrm{BD}}$ & Runs & TL (\%) & $\Delta_a^{\mathrm{BD}}$ Med./Mean (\%) & $\tau_a^{\mathrm{BD}}$ " + r"\\",
        r"\midrule",
    ]
    for _, row in coverage.iterrows():
        total = row["instance"] == "Total"
        first = r"\multicolumn{2}{l}{\textbf{Total}}" if total else f"{_escape(row['instance'])} & {int(row['num_subproblems'])}"
        labd_gap = f"{_fmt(row['labd_gap_median'])}/{_fmt(row['labd_gap_mean'])}"
        rnd_gap = f"{_fmt(row['rnd_gap_median'])}/{_fmt(row['rnd_gap_mean'])}"
        line = (
            f"{first} & {int(row['standard_bd_records'])} & {_fmt(row['standard_bd_time_limit_rate_pct'], 1)} & "
            f"{int(row['labd_records'])} & {_fmt(row['labd_time_limit_rate_pct'], 1)} & {labd_gap} & {_fmt(row['labd_time_median'], 3)} & "
            f"{int(row['rnd_records'])} & {_fmt(row['rnd_time_limit_rate_pct'], 1)} & {rnd_gap} & {_fmt(row['rnd_time_median'], 3)}"
        )
        if total:
            lines.append(r"\midrule")
        lines.append(line + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines)

def latex_method_summary_table(
    table: pd.DataFrame,
    *,
    problem: str,
    stratum: str,
    label: str,
    factor: str | None = None,
    factor_heading: str | None = None,
) -> str:
    subset = table[
        table["problem_class"].eq(problem)
        & table["baseline_stratum"].eq(stratum)
    ].copy()
    status = "solved by Standard BD" if stratum == STRATUM_ORDER[0] else "time-limited under Standard BD"
    factor_cols = "l" if factor else ""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Distributional summary for {problem} instances {status}.}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{l{factor_cols}rrrrrrrrrr}}",
        r"\toprule",
    ]
    left = "Method"
    if factor:
        left += f" & {factor_heading or factor}"
    delta_start = 5 if factor else 4
    delta_end = 8 if factor else 7
    tau_start = 9 if factor else 8
    tau_end = 12 if factor else 11
    lines += [
        f"{left} & $n$ & TL (\\%) & \\multicolumn{{4}}{{c}}{{$\\Delta_a^{{\\mathrm{{BD}}}}$ (\\%)}} & \\multicolumn{{4}}{{c}}{{$\\tau_a^{{\\mathrm{{BD}}}}$}} " + r"\\",
        rf"\cmidrule(lr){{{delta_start}-{delta_end}}}\cmidrule(lr){{{tau_start}-{tau_end}}}",
        f"{left} & & & Min & Max & Mean & Median & Min & Max & Mean & Median " + r"\\",
        r"\midrule",
    ]
    for _, row in subset.iterrows():
        cells = [_escape(row["method"])]
        if factor:
            cells.append(_escape(row[factor]))
        cells += [
            f"{int(row['records'])}",
            _fmt(row["time_limit_rate_pct"], 1),
            _fmt(row["gap_min"]), _fmt(row["gap_max"]),
            _fmt(row["gap_mean"]), _fmt(row["gap_median"]),
            _fmt(row["time_min"], 3), _fmt(row["time_max"], 3),
            _fmt(row["time_mean"], 3), _fmt(row["time_median"], 3),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines)


def latex_figure(stem: str, caption: str, label: str, width: str = "0.98\\linewidth") -> str:
    if stem in ZOOM_FIGURES:
        caption += rf" A zoomed view of the visually compressed panel(s) is provided in Appendix~\ref{{{ZOOM_FIGURES[stem]['label']}}}."
    return "\n".join([
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width={width}]{{Figures/Publication_v5/{stem}.pdf}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{figure}",
    ])


def latex_zoom_section() -> str:
    if not ZOOM_FIGURES:
        return ""
    lines = [
        r"\subsection{Zoomed Views of Visually Compressed Main-Text Figures}",
        r"\label{appendix:zoomed-main-figures}",
        "",
        r"The main text retains the full vertical range of every figure so that extreme observations remain visible. This subsection provides zoomed companions only for figures flagged by a prespecified robust diagnostic: the central 90\% of observations occupies less than 22\% of the full range and at least one tail extends more than five interquartile ranges beyond the adjacent quartile. For readability, the zoomed panel redraws the boxplots using observations inside the displayed range; arrows and counts identify observations outside that range. The full-range main-text figure retains every observation and remains the authoritative distributional display.",
        "",
    ]
    for stem, info in ZOOM_FIGURES.items():
        parts = []
        if info["objective_zoomed"]:
            parts.append(r"$\Delta_a^{\mathrm{BD}}$")
        if info["runtime_zoomed"]:
            parts.append(r"$\tau_a^{\mathrm{BD}}$")
        panels = " and ".join(parts)
        dimension = r"$|\mathcal M|$" if info["dimension"] == "num_subproblems" else r"$|\mathcal M_e|/|\mathcal M|$"
        status = "solved by" if info["baseline_stratum"] == STRATUM_ORDER[0] else "time-limited under"
        caption = f"Zoomed view of {panels} by {dimension} for {info['problem_class']} instances {status} Standard BD. The corresponding full-range figure is retained in the main text."
        lines += [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.98\linewidth]{{Figures/Publication_v5/{info['zoom_stem']}.pdf}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{info['label']}}}",
            r"\end{figure}",
            "",
        ]
    return "\n".join(lines)

def _dimension_row(table: pd.DataFrame, problem: str, stratum: str, method: str, dimension: str, value: float | int) -> pd.Series:
    match = table[
        table["problem_class"].eq(problem)
        & table["baseline_stratum"].eq(stratum)
        & table["method"].eq(method)
        & pd.to_numeric(table[dimension], errors="coerce").eq(value)
    ]
    if match.empty:
        raise KeyError((problem, stratum, method, dimension, value))
    return match.iloc[0]


def _coverage_total(coverage: pd.DataFrame) -> pd.Series:
    return coverage[coverage["instance"].eq("Total")].iloc[0]


def _callback_row(table: pd.DataFrame, problem: str, stratum: str, method: str, policy: str) -> pd.Series:
    return table[
        table["problem_class"].eq(problem)
        & table["baseline_stratum"].eq(stratum)
        & table["method"].eq(method)
        & table["callback_policy"].eq(policy)
    ].iloc[0]


def _feature_row(table: pd.DataFrame, problem: str, stratum: str, method: str, feature: str) -> pd.Series:
    return table[
        table["problem_class"].eq(problem)
        & table["baseline_stratum"].eq(stratum)
        & table["method"].eq(method)
        & table["feature_specification"].eq(feature)
    ].iloc[0]


# =============================================================================
# Manuscript generation
# =============================================================================
def _summary_lookup(summary: pd.DataFrame, problem: str, stratum: str, method: str) -> pd.Series:
    return summary[
        summary["problem_class"].eq(problem)
        & summary["baseline_stratum"].eq(stratum)
        & summary["method"].eq(method)
    ].iloc[0]



def build_results_draft(
    primary_summary: pd.DataFrame,
    primary_by_num: pd.DataFrame,
    primary_by_ratio: pd.DataFrame,
    coverage_scflp: pd.DataFrame,
    coverage_scmnd: pd.DataFrame,
    benchmark_table: str,
    coverage_scflp_tex: str,
    coverage_scmnd_tex: str,
    summary_tables: dict[tuple[str, str], str],
) -> str:
    sc_opt_tree = _summary_lookup(primary_summary, "SCFLP", STRATUM_ORDER[0], "LABD--Tree")
    sc_opt_clu = _summary_lookup(primary_summary, "SCFLP", STRATUM_ORDER[0], "LABD--Cluster")
    sc_opt_rnd = _summary_lookup(primary_summary, "SCFLP", STRATUM_ORDER[0], "RND")
    sc_tl_tree = _summary_lookup(primary_summary, "SCFLP", STRATUM_ORDER[1], "LABD--Tree")
    sc_tl_clu = _summary_lookup(primary_summary, "SCFLP", STRATUM_ORDER[1], "LABD--Cluster")
    sc_tl_rnd = _summary_lookup(primary_summary, "SCFLP", STRATUM_ORDER[1], "RND")
    nm_opt_tree = _summary_lookup(primary_summary, "SCMND", STRATUM_ORDER[0], "LABD--Tree")
    nm_opt_clu = _summary_lookup(primary_summary, "SCMND", STRATUM_ORDER[0], "LABD--Cluster")
    nm_opt_rnd = _summary_lookup(primary_summary, "SCMND", STRATUM_ORDER[0], "RND")
    nm_tl_tree = _summary_lookup(primary_summary, "SCMND", STRATUM_ORDER[1], "LABD--Tree")
    nm_tl_clu = _summary_lookup(primary_summary, "SCMND", STRATUM_ORDER[1], "LABD--Cluster")
    nm_tl_rnd = _summary_lookup(primary_summary, "SCMND", STRATUM_ORDER[1], "RND")

    sc_cov = _coverage_total(coverage_scflp)
    nm_cov = _coverage_total(coverage_scmnd)

    # Exact-share endpoints for the most interpretable trade-offs.
    sc_r05_tree = _dimension_row(primary_by_ratio, "SCFLP", STRATUM_ORDER[0], "LABD--Tree", "selected_subproblems_ratio", 0.05)
    sc_r25_tree = _dimension_row(primary_by_ratio, "SCFLP", STRATUM_ORDER[0], "LABD--Tree", "selected_subproblems_ratio", 0.25)
    sc_r05_clu = _dimension_row(primary_by_ratio, "SCFLP", STRATUM_ORDER[0], "LABD--Cluster", "selected_subproblems_ratio", 0.05)
    sc_r25_clu = _dimension_row(primary_by_ratio, "SCFLP", STRATUM_ORDER[0], "LABD--Cluster", "selected_subproblems_ratio", 0.25)
    sc_r05_rnd = _dimension_row(primary_by_ratio, "SCFLP", STRATUM_ORDER[0], "RND", "selected_subproblems_ratio", 0.05)
    sc_r25_rnd = _dimension_row(primary_by_ratio, "SCFLP", STRATUM_ORDER[0], "RND", "selected_subproblems_ratio", 0.25)

    nm_r05_tree = _dimension_row(primary_by_ratio, "SCMND", STRATUM_ORDER[1], "LABD--Tree", "selected_subproblems_ratio", 0.05)
    nm_r25_tree = _dimension_row(primary_by_ratio, "SCMND", STRATUM_ORDER[1], "LABD--Tree", "selected_subproblems_ratio", 0.25)
    nm_r05_clu = _dimension_row(primary_by_ratio, "SCMND", STRATUM_ORDER[1], "LABD--Cluster", "selected_subproblems_ratio", 0.05)
    nm_r25_clu = _dimension_row(primary_by_ratio, "SCMND", STRATUM_ORDER[1], "LABD--Cluster", "selected_subproblems_ratio", 0.25)

    scflp_solved_num_figure = latex_figure(
        "main__scflp__standard_bd_solved__num_subproblems_v5",
        "Effect of $|\\mathcal M|$ on $\\Delta_a^{\\mathrm{BD}}$ and $\\tau_a^{\\mathrm{BD}}$ for SCFLP instances solved by Standard BD. Boxes show interquartile ranges, center lines show medians, and markers show individual configurations.",
        "fig:scflp-solved-num",
    )
    scflp_solved_ratio_figure = latex_figure(
        "main__scflp__standard_bd_solved__selected_subproblems_ratio_v5",
        "Effect of $|\\mathcal M_e|/|\\mathcal M|$ on $\\Delta_a^{\\mathrm{BD}}$ and $\\tau_a^{\\mathrm{BD}}$ for SCFLP instances solved by Standard BD.",
        "fig:scflp-solved-ratio",
    )
    scflp_time_num_figure = latex_figure(
        "main__scflp__standard_bd_time_limited__num_subproblems_v5",
        "Effect of $|\\mathcal M|$ on incumbent quality and runtime for SCFLP instances time-limited under Standard BD.",
        "fig:scflp-time-num",
    )
    scflp_time_ratio_figure = latex_figure(
        "main__scflp__standard_bd_time_limited__selected_subproblems_ratio_v5",
        "Effect of $|\\mathcal M_e|/|\\mathcal M|$ on incumbent quality and runtime for SCFLP instances time-limited under Standard BD.",
        "fig:scflp-time-ratio",
    )
    scmnd_solved_num_figure = latex_figure(
        "main__scmnd__standard_bd_solved__num_subproblems_v5",
        "Distribution of $\\Delta_a^{\\mathrm{BD}}$ and $\\tau_a^{\\mathrm{BD}}$ for the 100-subproblem SCMND instances solved by Standard BD.",
        "fig:scmnd-solved-num",
    )
    scmnd_solved_ratio_figure = latex_figure(
        "main__scmnd__standard_bd_solved__selected_subproblems_ratio_v5",
        "Effect of $|\\mathcal M_e|/|\\mathcal M|$ on $\\Delta_a^{\\mathrm{BD}}$ and $\\tau_a^{\\mathrm{BD}}$ for SCMND instances solved by Standard BD.",
        "fig:scmnd-solved-ratio",
    )
    scmnd_time_num_figure = latex_figure(
        "main__scmnd__standard_bd_time_limited__num_subproblems_v5",
        "Effect of $|\\mathcal M|$ on incumbent quality and runtime for SCMND instances time-limited under Standard BD.",
        "fig:scmnd-time-num",
    )
    scmnd_time_ratio_figure = latex_figure(
        "main__scmnd__standard_bd_time_limited__selected_subproblems_ratio_v5",
        "Effect of $|\\mathcal M_e|/|\\mathcal M|$ on incumbent quality and runtime for SCMND instances time-limited under Standard BD.",
        "fig:scmnd-time-ratio",
    )

    text = rf"""
% -----------------------------------------------------------------------------
% Version 5 results draft generated from the cleaned experiment data.
% Primary filters: ML_at_both, features_only, |M| in {{100,300,500}},
% |M_e|/|M| in {{5%,15%,25%}}, predictors TREE and CLUSTERING.
% -----------------------------------------------------------------------------

\section{{Results}}
\label{{sec:results}}

All methods are implemented within the same single-tree multi-cut Benders
framework. The master MIP is solved through one branch-and-bound tree, and
Benders cuts are generated through callbacks at integer-feasible and
LP-relaxation solutions. Standard BD obtains every cut from exact subproblem
solutions. LABD and RND use the same master and callback structure but reduce
the number of exact subproblem solves through learning-guided and random
mechanisms, respectively. The primary LABD implementation generates predicted
cuts at both callback locations and uses the representation learned from the
warm-up dual trajectories. Appendix~\ref{{appendix:numerical-results}} studies
an alternative integer-point cut policy and contextual feature inputs.

\subsection{{Performance Metrics and Experimental Setup}}
\label{{subsec:performance-metrics}}

Standard BD solves every subproblem exactly when generating cuts. LABD solves
the subproblems in $\mathcal M_e$ exactly and predicts dual-feasible solutions
for $\mathcal M_p$. RND is an uninformed benchmark inspired by the
random-sampling approach of \citet{{bertsimas2024stochastic}}. It uses a random
exact-subproblem sample and forms the remaining dual estimates from the dual
information obtained from that sample. The comparison with RND separates the
benefit of learning-guided representation, selection, and prediction from the
benefit of reducing exact subproblem solves alone.
% TODO before final submission: verify the precise SCMND RND aggregation rule
% in the implementation and reconcile it with this description.

For each subproblem-reduction method $a\in\{{\text{{LABD--Tree}},
\text{{LABD--Cluster}},\text{{RND}}\}}$, let $V_a=V(\hat{{\mathbf x}}_a)$
denote the true objective value of its returned first-stage solution, obtained
by solving all subproblems exactly at $\hat{{\mathbf x}}_a$. Let
$V_{{\mathrm{{BD}}}}$ denote the corresponding true objective value returned by
Standard BD. We report
\begin{{equation}}
\label{{eq:relative-objective-difference}}
\Delta_a^{{\mathrm{{BD}}}}(\%)
=
100\frac{{V_a-V_{{\mathrm{{BD}}}}}}
{{\lvert V_{{\mathrm{{BD}}}}\rvert}}.
\end{{equation}}
When Standard BD solves the instance to the prescribed tolerance,
$\Delta_a^{{\mathrm{{BD}}}}$ is the objective gap relative to its solution. When
Standard BD reaches the time limit, the same measure compares the returned
incumbents; a negative value means that method $a$ finds a better feasible
solution. Runtime is summarized by
\begin{{equation}}
\label{{eq:relative-runtime}}
\tau_a^{{\mathrm{{BD}}}}
=
\frac{{T_a}}{{T_{{\mathrm{{BD}}}}}},
\end{{equation}}
where values below one indicate faster execution. We use TL to denote the
percentage of runs reaching the three-hour time limit.

The primary analysis uses $|\mathcal M|\in\{{100,300,500\}}$ and
$|\mathcal M_e|/|\mathcal M|\in\{{5\%,15\%,25\%\}}$. The warm-up length is
$K_{{\mathrm w}}=5$, and the learned representation has dimension three. The
algorithms were implemented in Python and solved with Gurobi~11.0 on a
high-performance computing cluster. Each run was allocated 32 CPU cores and
256 GB of memory, with a three-hour time limit and a target MIP gap of $1\%$.

The SCFLP analysis retains five deterministic facility-location instances. The
SCMND analysis uses six instances from the NetDesMMCF family introduced by
\citet{{crainic2001bundle}}. Table~\ref{{tab:benchmark-instances}} reports their
structural dimensions. The three values of $|\mathcal M|$ are evaluated for
each retained deterministic instance.

{benchmark_table}

\subsection{{Benchmarking on the SCFLP Instances}}
\label{{subsec:scflp-results}}

For each SCFLP instance and value of $|\mathcal M|$, Standard BD contributes
one run. LABD contributes six runs: the Tree and Cluster predictors, each at
$|\mathcal M_e|/|\mathcal M|\in\{{5\%,15\%,25\%\}}$. RND contributes three
runs, one at each of the same exact-solve shares. Table~\ref{{tab:scflp-coverage}}
reports the resulting coverage, termination frequency, and median performance.
The subsequent analyses distinguish the instances solved by Standard BD from
those for which it reaches the time limit because the interpretation of
$\Delta_a^{{\mathrm{{BD}}}}$ differs across these groups.

{coverage_scflp_tex}

Across all retained SCFLP experiments, Standard BD reaches the time limit in
{_fmt(sc_cov['standard_bd_time_limit_rate_pct'],1)}\% of its runs. The
corresponding rates are {_fmt(sc_cov['labd_time_limit_rate_pct'],1)}\% for LABD
and {_fmt(sc_cov['rnd_time_limit_rate_pct'],1)}\% for RND. Thus, both
subproblem-reduction approaches improve completion frequency, but LABD's rate
is less than half that of RND. The pooled median $\Delta_a^{{\mathrm{{BD}}}}$ is zero for both LABD and RND because at least half of the run-level objective differences equal zero. The accompanying means in Table~\ref{{tab:scflp-coverage}} retain the influence of the nonzero tails and should be read together with the medians. The principal overall distinction is therefore computational stability rather than the typical objective value.

\subsubsection{{Instances Solved by Standard BD}}
\label{{subsubsec:scflp-solved}}

This subsection considers the SCFLP instances that Standard BD solves to the
prescribed tolerance within the time limit. Table~\ref{{tab:scflp-solved-summary}}
provides the high-level performance distributions.

{summary_tables[("SCFLP", STRATUM_ORDER[0])]}

Both LABD predictors have median $\Delta_a^{{\mathrm{{BD}}}}=0$, while their
median runtime ratios are {_fmt(sc_opt_tree['time_median'],3)} for Tree and
{_fmt(sc_opt_clu['time_median'],3)} for Cluster. RND also has a zero median
objective difference, but its mean runtime ratio is {_fmt(sc_opt_rnd['time_mean'],3)}
and its maximum exceeds Standard BD by a factor of {_fmt(sc_opt_rnd['time_max'],2)}.
The contrast indicates that reducing the number of exact solves is not by
itself sufficient for reliable acceleration: the learned mechanisms preserve
the same median true objective while substantially reducing runtime dispersion.

{scflp_solved_num_figure}

As $|\mathcal M|$ increases, the LABD objective distributions remain tightly
centered at zero, while RND exhibits a wider positive tail. The median runtime
ratios of Tree and Cluster increase only moderately between 100 and 500
subproblems, remaining close to one quarter of Standard BD at the largest
level. RND's median ratio rises from roughly 0.39 at 100 subproblems to 0.73 at
500, and its mean exceeds one at both 300 and 500. The learning-guided methods
therefore scale more consistently with the number of subproblems.

{scflp_solved_ratio_figure}

Increasing $|\mathcal M_e|/|\mathcal M|$ from $5\%$ to $25\%$ leaves the median
objective difference at zero for every method but increases median runtime from
{_fmt(sc_r05_tree['time_median'],3)} to {_fmt(sc_r25_tree['time_median'],3)} for
Tree and from {_fmt(sc_r05_clu['time_median'],3)} to
{_fmt(sc_r25_clu['time_median'],3)} for Cluster. The increase is sharper for
RND, from {_fmt(sc_r05_rnd['time_median'],3)} to
{_fmt(sc_r25_rnd['time_median'],3)}. On these solved instances, a larger exact
set therefore adds computational cost without improving median solution
quality; the $5\%$ share provides the strongest observed runtime-quality
trade-off.

\subsubsection{{Instances Time-Limited under Standard BD}}
\label{{subsubsec:scflp-time-limited}}

This subsection considers the two SCFLP instances--the $p20$ cases with 300 and
500 subproblems--for which Standard BD reaches the time limit. Here,
$\Delta_a^{{\mathrm{{BD}}}}$ compares incumbent quality rather than an optimal
solution.

{summary_tables[("SCFLP", STRATUM_ORDER[1])]}

The median objective difference is strongly negative for both LABD predictors,
showing that they improve the Standard BD incumbent within the same time
budget. Tree and Cluster each reach the time limit in
{_fmt(sc_tl_tree['time_limit_rate_pct'],1)}\% of their configurations, compared
with {_fmt(sc_tl_rnd['time_limit_rate_pct'],1)}\% for RND. RND's positive
maximum $\Delta_a^{{\mathrm{{BD}}}}$ also shows that random reduction is less
reliable: it can improve the incumbent substantially, but it can also return a
worse one.

{scflp_time_num_figure}

At 300 subproblems, none of the accelerated configurations reaches the time
limit, and all three methods produce the same median incumbent improvement.
At 500 subproblems, the completion advantage narrows: one configuration of
each LABD predictor and two RND configurations reach the limit. LABD retains a
more favorable objective distribution than RND, indicating that the learned
selection and prediction mechanisms become more valuable as the instance
hardens.

{scflp_time_ratio_figure}

For both LABD predictors, the $5\%$ and $15\%$ exact-solve shares obtain the
same median incumbent improvement, whereas the $25\%$ share is slower and less
favorable in objective value. RND displays the same runtime direction but a
larger deterioration in incumbent quality. Under a binding time budget, using
a larger exact set can divert computation from master-search progress without
producing sufficiently stronger cuts.

\subsection{{Benchmarking on the SCMND Instances}}
\label{{subsec:scmnd-results}}

The SCMND design uses the same record structure as SCFLP. Table~\ref{{tab:scmnd-coverage}}
reports coverage, termination, and median performance by deterministic instance
and $|\mathcal M|$.

{coverage_scmnd_tex}

Standard BD reaches the time limit in {_fmt(nm_cov['standard_bd_time_limit_rate_pct'],1)}\%
of the SCMND runs. LABD reduces this rate to
{_fmt(nm_cov['labd_time_limit_rate_pct'],1)}\%, whereas RND remains at
{_fmt(nm_cov['rnd_time_limit_rate_pct'],1)}\%. The relative-runtime medians are
close to one because most difficult runs consume nearly the full common time
budget. Accordingly, the subsequent SCMND analysis places greater weight on
incumbent quality and termination frequency than on small differences in
runtime ratio.

\subsubsection{{Instances Solved by Standard BD}}
\label{{subsubsec:scmnd-solved}}

Only the $r1.1$ and $r1.2$ cases with 100 subproblems are solved by Standard BD.
The resulting summary is therefore informative about method behavior but does
not support a scaling conclusion across $|\mathcal M|$.

{summary_tables[("SCMND", STRATUM_ORDER[0])]}

LABD--Cluster has median $\Delta_a^{{\mathrm{{BD}}}}=0$, while LABD--Tree's
median is {_fmt(nm_opt_tree['gap_median'])}\%; both have median runtime ratios
near one quarter. RND also has a zero median objective difference, but one run
reaches the time limit, its maximum objective difference is
{_fmt(nm_opt_rnd['gap_max'])}\%, and its mean runtime ratio exceeds one. Thus,
the learned methods are markedly more stable even within this small solved
subset.

{scmnd_solved_num_figure}

Because only one value of $|\mathcal M|$ is represented, this figure should not
be read as a scalability result. It nevertheless confirms that LABD's runtime
distribution is concentrated below Standard BD, whereas RND spans both faster
and substantially slower outcomes.

{scmnd_solved_ratio_figure}

The LABD variants remain close to the Standard BD true objective at all three
exact-solve shares. Their runtime generally increases with the share, although
Cluster is more variable at $25\%$. RND is nonmonotone and reaches the time
limit at $25\%$, reinforcing that a larger random exact set does not guarantee
more useful cuts.

\subsubsection{{Instances Time-Limited under Standard BD}}
\label{{subsubsec:scmnd-time-limited}}

This group contains 16 of the 18 SCMND instance--subproblem-count combinations.
Negative values of $\Delta_a^{{\mathrm{{BD}}}}$ indicate that the accelerated
method finds a better incumbent than Standard BD within the three-hour budget.

{summary_tables[("SCMND", STRATUM_ORDER[1])]}

LABD--Tree and LABD--Cluster have median objective differences of
{_fmt(nm_tl_tree['gap_median'])}\% and {_fmt(nm_tl_clu['gap_median'])}\%,
respectively, compared with {_fmt(nm_tl_rnd['gap_median'])}\% for RND. Both
LABD predictors reach the time limit in {_fmt(nm_tl_tree['time_limit_rate_pct'],1)}\%
of their configurations, below RND's {_fmt(nm_tl_rnd['time_limit_rate_pct'],1)}\%.
RND also has a large positive upper tail, whereas the learned methods improve
the incumbent more consistently.

{scmnd_time_num_figure}

At 100 subproblems, all accelerated runs reach the time limit and RND has a
positive median objective difference, so reducing exact solves alone does not
improve the incumbent. At 300 and 500 subproblems, LABD develops increasingly
negative objective distributions; the median improvement is largest at 500.
RND also improves at 500 but remains more dispersed. Runtime ratios stay near
one, confirming that incumbent quality is the meaningful differentiator in
this censored regime.

{scmnd_time_ratio_figure}

Increasing the exact-solve share from $5\%$ to $25\%$ raises the LABD time-limit
rate from {_fmt(nm_r05_tree['time_limit_rate_pct'],1)}\% to
{_fmt(nm_r25_tree['time_limit_rate_pct'],1)}\% for Tree and from
{_fmt(nm_r05_clu['time_limit_rate_pct'],1)}\% to
{_fmt(nm_r25_clu['time_limit_rate_pct'],1)}\% for Cluster. The median incumbent
advantage simultaneously weakens: Tree changes from
{_fmt(nm_r05_tree['gap_median'])}\% to {_fmt(nm_r25_tree['gap_median'])}\%, and
Cluster from {_fmt(nm_r05_clu['gap_median'])}\% to
{_fmt(nm_r25_clu['gap_median'])}\%. This is the clearest evidence that, under a
binding time limit, more exact subproblem solves can worsen the overall
runtime-quality trade-off.

Overall, LABD is distinguished from RND by runtime stability on SCFLP and by
more reliable incumbent improvement on difficult SCMND instances. The appendix
shows how these conclusions change under the alternative callback policy and
with contextual feature inputs.
"""
    return textwrap.dedent(text).strip() + "\n"


def build_appendix_draft(
    callback_summary: pd.DataFrame,
    callback_by_num: pd.DataFrame,
    callback_by_ratio: pd.DataFrame,
    feature_summary: pd.DataFrame,
    feature_by_num: pd.DataFrame,
    feature_by_ratio: pd.DataFrame,
    callback_tables: dict[tuple[str, str], str],
    feature_tables: dict[tuple[str, str], str],
) -> str:
    sections: list[str] = [r"""
\section{Supplementary Numerical Results}
\label{appendix:numerical-results}

This appendix mirrors the principal analyses for two controlled implementation
variations. The notation $\Delta_a^{\mathrm{BD}}$, $\tau_a^{\mathrm{BD}}$, and
TL follows Section~\ref{subsec:performance-metrics}. The first analysis compares
the primary callback policy with a policy that solves the relevant
subproblems exactly at integer solutions. The second compares representation
features (R), contextual covariates (C), and their combination (R+C).
"""]

    zoom_section = latex_zoom_section()
    if zoom_section:
        sections.append(zoom_section)

    sections.append(r"""
\subsection{Alternative Integer-Point Cut Policy}
\label{appendix:callback-policy-results}

The primary policy generates predicted cuts at both LP-relaxation and integer
solutions. The alternative uses predicted cuts at LP points but exact
subproblem cuts at integer points. Representation-only input is held fixed, so
the comparison isolates the callback policy.
""")

    for problem in ["SCFLP", "SCMND"]:
        sections.append(f"\\subsubsection{{{problem}}}\n")
        for stratum in STRATUM_ORDER:
            solved = stratum == STRATUM_ORDER[0]
            title = "Instances Solved by Standard BD" if solved else "Instances Time-Limited under Standard BD"
            label_slug = "solved" if solved else "time-limited"
            sections.append(f"\\paragraph{{{title}.}}\n")
            sections.append(callback_tables[(problem, stratum)] + "\n\n")

            if problem == "SCFLP" and solved:
                p_tree = _callback_row(callback_summary, problem, stratum, "LABD--Tree", CALLBACK_LABELS[PRIMARY_CALLBACK])
                e_tree = _callback_row(callback_summary, problem, stratum, "LABD--Tree", CALLBACK_LABELS[SUPPLEMENTARY_CALLBACK])
                p_clu = _callback_row(callback_summary, problem, stratum, "LABD--Cluster", CALLBACK_LABELS[PRIMARY_CALLBACK])
                e_clu = _callback_row(callback_summary, problem, stratum, "LABD--Cluster", CALLBACK_LABELS[SUPPLEMENTARY_CALLBACK])
                sections.append(
                    f"Both policies retain median $\\Delta_a^{{\\mathrm{{BD}}}}=0$. "
                    f"For Tree, the primary policy lowers median $\\tau_a^{{\\mathrm{{BD}}}}$ from {_fmt(e_tree['time_median'],3)} to {_fmt(p_tree['time_median'],3)}; "
                    f"for Cluster, it lowers the median from {_fmt(e_clu['time_median'],3)} to {_fmt(p_clu['time_median'],3)}. "
                    "Exact integer-point cuts therefore add substantial computational work without improving median solution quality on the solved SCFLP instances.\n\n"
                )
            elif problem == "SCFLP":
                p_tree = _callback_row(callback_summary, problem, stratum, "LABD--Tree", CALLBACK_LABELS[PRIMARY_CALLBACK])
                e_tree = _callback_row(callback_summary, problem, stratum, "LABD--Tree", CALLBACK_LABELS[SUPPLEMENTARY_CALLBACK])
                sections.append(
                    f"On the time-limited SCFLP instances, the primary Tree policy reaches the limit in {_fmt(p_tree['time_limit_rate_pct'],1)}\\% of runs, compared with {_fmt(e_tree['time_limit_rate_pct'],1)}\\% under exact integer-point cuts. "
                    f"It also has a more negative median objective difference ({_fmt(p_tree['gap_median'])}\\% versus {_fmt(e_tree['gap_median'])}\\%). "
                    "The additional exact integer-point work therefore reduces both completion frequency and incumbent progress in this group.\n\n"
                )
            elif problem == "SCMND" and solved:
                p_tree = _callback_row(callback_summary, problem, stratum, "LABD--Tree", CALLBACK_LABELS[PRIMARY_CALLBACK])
                e_tree = _callback_row(callback_summary, problem, stratum, "LABD--Tree", CALLBACK_LABELS[SUPPLEMENTARY_CALLBACK])
                sections.append(
                    f"The alternative Tree policy attains a median objective difference of {_fmt(e_tree['gap_median'])}\\%, compared with {_fmt(p_tree['gap_median'])}\\% for the primary policy, but its median runtime ratio is approximately twice as large ({_fmt(e_tree['time_median'],3)} versus {_fmt(p_tree['time_median'],3)}). "
                    "This illustrates a modest solution-quality improvement purchased with a substantial runtime increase in the small solved SCMND subset.\n\n"
                )
            else:
                p_clu = _callback_row(callback_summary, problem, stratum, "LABD--Cluster", CALLBACK_LABELS[PRIMARY_CALLBACK])
                e_clu = _callback_row(callback_summary, problem, stratum, "LABD--Cluster", CALLBACK_LABELS[SUPPLEMENTARY_CALLBACK])
                p_tree = _callback_row(callback_summary, problem, stratum, "LABD--Tree", CALLBACK_LABELS[PRIMARY_CALLBACK])
                e_tree = _callback_row(callback_summary, problem, stratum, "LABD--Tree", CALLBACK_LABELS[SUPPLEMENTARY_CALLBACK])
                sections.append(
                    f"For time-limited SCMND instances, the primary policy yields more negative median objective differences for both Tree ({_fmt(p_tree['gap_median'])}\\% versus {_fmt(e_tree['gap_median'])}\\%) and Cluster ({_fmt(p_clu['gap_median'])}\\% versus {_fmt(e_clu['gap_median'])}\\%). "
                    "The Tree time-limit rate is also lower under the primary policy. These results suggest that repeated exact integer-point evaluation consumes time that can otherwise support master-search progress.\n\n"
                )

            for dimension, dlabel in [("num_subproblems", "$|\\mathcal M|$"), ("selected_subproblems_ratio", "$|\\mathcal M_e|/|\\mathcal M|$")]:
                stem = f"appendix_callback__{problem.lower()}__{_safe_slug(stratum)}__{dimension}_v5"
                figlabel = f"fig:appendix-callback-{problem.lower()}-{label_slug}-{'num' if dimension == 'num_subproblems' else 'ratio'}"
                caption = f"Callback-policy robustness by {dlabel} for {problem} instances {'solved by' if solved else 'time-limited under'} Standard BD."
                sections.append(latex_figure(stem, caption, figlabel) + "\n\n")
                if dimension == "num_subproblems":
                    sections.append(
                        "Across the displayed subproblem counts, the primary policy generally retains the lower runtime distribution. The difference widens where the alternative policy repeatedly invokes exact subproblems at integer solutions. Objective effects are smaller on solved instances but become consequential in the time-limited groups, where the primary policy more often maintains a negative incumbent difference.\n\n"
                    )
                else:
                    sections.append(
                        "The runtime penalty of exact integer-point cuts grows as $|\\mathcal M_e|/|\\mathcal M|$ increases because each callback evaluates a larger exact set. The objective distributions do not improve uniformly with this added effort, so the appendix evidence supports the primary policy used in the main analysis.\n\n"
                    )

    sections.append(r"""
\subsection{Representation and Contextual-Feature Ablation}
\label{appendix:feature-ablation-results}

The callback policy is fixed at predicted cuts at both LP-relaxation and integer
solutions. We compare representation-only input (R), contextual-only input
(C), and their combination (R+C), separately for Tree and Cluster.
""")

    for problem in ["SCFLP", "SCMND"]:
        sections.append(f"\\subsubsection{{{problem}}}\n")
        for stratum in STRATUM_ORDER:
            solved = stratum == STRATUM_ORDER[0]
            title = "Instances Solved by Standard BD" if solved else "Instances Time-Limited under Standard BD"
            label_slug = "solved" if solved else "time-limited"
            sections.append(f"\\paragraph{{{title}.}}\n")
            sections.append(feature_tables[(problem, stratum)] + "\n\n")

            if problem == "SCFLP" and solved:
                c_tree = _feature_row(feature_summary, problem, stratum, "LABD--Tree", "C")
                r_tree = _feature_row(feature_summary, problem, stratum, "LABD--Tree", "R")
                rc_tree = _feature_row(feature_summary, problem, stratum, "LABD--Tree", "R+C")
                sections.append(
                    f"All feature inputs have median $\\Delta_a^{{\\mathrm{{BD}}}}=0$. Contextual-only input gives the lowest Tree median runtime ({_fmt(c_tree['time_median'],3)}), while R+C ({_fmt(rc_tree['time_median'],3)}) and R ({_fmt(r_tree['time_median'],3)}) remain close. "
                    "The small runtime differences do not establish universal dominance, and representation-only input remains the most direct test of the proposed endogenous feature construction.\n\n"
                )
            elif problem == "SCFLP":
                r_tree = _feature_row(feature_summary, problem, stratum, "LABD--Tree", "R")
                c_tree = _feature_row(feature_summary, problem, stratum, "LABD--Tree", "C")
                sections.append(
                    f"Representation-only Tree has the lowest time-limit rate and median runtime in this group ({_fmt(r_tree['time_limit_rate_pct'],1)}\\% and {_fmt(r_tree['time_median'],3)}), compared with {_fmt(c_tree['time_limit_rate_pct'],1)}\\% and {_fmt(c_tree['time_median'],3)} for contextual-only input. "
                    "The objective differences remain favorable across all specifications, so the main distinction is computational efficiency.\n\n"
                )
            elif problem == "SCMND" and solved:
                rc_clu = _feature_row(feature_summary, problem, stratum, "LABD--Cluster", "R+C")
                c_clu = _feature_row(feature_summary, problem, stratum, "LABD--Cluster", "C")
                r_clu = _feature_row(feature_summary, problem, stratum, "LABD--Cluster", "R")
                sections.append(
                    f"For Cluster, R+C has the lowest median runtime ({_fmt(rc_clu['time_median'],3)}), followed by C ({_fmt(c_clu['time_median'],3)}) and R ({_fmt(r_clu['time_median'],3)}), while all three have median objective difference zero. "
                    "Because this group contains only two base instances, the ranking should be interpreted as descriptive rather than conclusive.\n\n"
                )
            else:
                c_clu = _feature_row(feature_summary, problem, stratum, "LABD--Cluster", "C")
                r_clu = _feature_row(feature_summary, problem, stratum, "LABD--Cluster", "R")
                rc_tree = _feature_row(feature_summary, problem, stratum, "LABD--Tree", "R+C")
                r_tree = _feature_row(feature_summary, problem, stratum, "LABD--Tree", "R")
                sections.append(
                    f"Contextual-only input improves Cluster's median objective difference from {_fmt(r_clu['gap_median'])}\\% under R to {_fmt(c_clu['gap_median'])}\\% and lowers its time-limit rate. For Tree, R+C improves the median objective difference relative to R ({_fmt(rc_tree['gap_median'])}\\% versus {_fmt(r_tree['gap_median'])}\\%) and modestly lowers runtime. "
                    "The benefit of contextual information is therefore predictor-dependent rather than uniform.\n\n"
                )

            for dimension, dlabel in [("num_subproblems", "$|\\mathcal M|$"), ("selected_subproblems_ratio", "$|\\mathcal M_e|/|\\mathcal M|$")]:
                stem = f"appendix_features__{problem.lower()}__{_safe_slug(stratum)}__{dimension}_v5"
                figlabel = f"fig:appendix-features-{problem.lower()}-{label_slug}-{'num' if dimension == 'num_subproblems' else 'ratio'}"
                caption = f"Feature-input robustness by {dlabel} for {problem} instances {'solved by' if solved else 'time-limited under'} Standard BD."
                sections.append(latex_figure(stem, caption, figlabel) + "\n\n")
                if dimension == "num_subproblems":
                    sections.append(
                        "The feature rankings vary with problem size and predictor. Contextual or combined inputs sometimes narrow the objective distribution at the larger subproblem counts, but the effect is not consistent enough to identify one universally dominant specification.\n\n"
                    )
                else:
                    sections.append(
                        "Changes in $|\\mathcal M_e|/|\\mathcal M|$ have a larger and more systematic effect on runtime than the choice among R, C, and R+C. Feature enrichment can improve particular incumbent distributions, but it does not eliminate the computational cost of increasing the exact-solve share.\n\n"
                    )

    sections.append(r"""
The ablation results support the representation-only specification used in the
main analysis as a transparent evaluation of LABD's endogenous feature-learning
mechanism. Contextual covariates can improve selected predictor--problem
combinations, but no input specification dominates across both problem classes,
termination groups, and prediction methods.
""")
    return "\n".join(sections).strip() + "\n"

# =============================================================================
# Output organization
# =============================================================================
def _figure_problem_and_category(stem: str) -> tuple[str, str]:
    lower = stem.lower()
    problem = "SCFLP" if "__scflp__" in lower else "SCMND" if "__scmnd__" in lower else "Other"
    if stem.startswith("main__") and "_zoom_v5" in stem:
        category = "02_appendix_zoom_companions"
    elif stem.startswith("main__"):
        category = "01_main_results"
    elif stem.startswith("appendix_callback__"):
        category = "03_appendix_callback_policy"
    elif stem.startswith("appendix_features__"):
        category = "04_appendix_feature_ablation"
    else:
        category = "05_other"
    return problem, category


def organize_output_files() -> pd.DataFrame:
    """Create problem- and analysis-specific copies while retaining a flat Overleaf set."""
    manifest_rows: list[dict[str, str]] = []

    for source in sorted(FIGURE_DIR.glob("*")):
        if source.suffix.lower() not in {".pdf", ".png"}:
            continue
        problem, category = _figure_problem_and_category(source.stem)
        destination_dir = ORGANIZED_FIGURE_DIR / problem / category
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        manifest_rows.append({
            "problem_class": problem,
            "analysis_category": category,
            "file_type": source.suffix.lower().lstrip("."),
            "file_name": source.name,
            "overleaf_ready_path": str(source.relative_to(OUTPUT_DIR)),
            "organized_path": str(destination.relative_to(OUTPUT_DIR)),
        })

    # Source-data CSVs are organized using the same naming convention.
    source_data_root = FIGURE_DATA_DIR / "by_problem"
    for source in sorted(FIGURE_DATA_DIR.glob("*.csv")):
        problem, category = _figure_problem_and_category(source.stem)
        destination_dir = source_data_root / problem / category
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / source.name)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(TABLE_DIR / "output_file_manifest_v5.csv", index=False)
    return manifest


# =============================================================================
# Main execution
# =============================================================================

def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for directory in [
        OUTPUT_DIR, FIGURE_DIR, ORGANIZED_FIGURE_DIR, FIGURE_DATA_DIR,
        TABLE_DIR, LATEX_DIR, ANALYSIS_DATA_DIR, DIAGNOSTICS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    cleaned, removed = load_and_clean()
    primary = attach_standard_bd_reference(select_primary(cleaned))
    callback = attach_standard_bd_reference(select_callback_robustness(cleaned))
    feature = attach_standard_bd_reference(select_feature_ablation(cleaned))

    standard_counts = primary[primary["method"].eq("Standard BD")].groupby(KEY_COLUMNS).size()
    if not standard_counts.eq(1).all():
        raise AssertionError("Primary analysis must contain one Standard BD row per key.")
    if not set(primary["stopping_reason"].unique()).issubset({"OPTIMAL", "TIME_LIMIT"}):
        raise AssertionError("Unexpected stopping status remains in the primary data.")
    if not primary["objective_available"].all():
        raise AssertionError("Every retained Version 5 primary record must have a valid true objective value.")

    cleaned.to_csv(ANALYSIS_DATA_DIR / "cleaned_deduplicated_data_v5.csv", index=False)
    removed.to_csv(ANALYSIS_DATA_DIR / "removed_duplicate_rows_v5.csv", index=False)
    primary.to_csv(ANALYSIS_DATA_DIR / "primary_analysis_data_v5.csv", index=False)
    callback.to_csv(ANALYSIS_DATA_DIR / "callback_robustness_data_v5.csv", index=False)
    feature.to_csv(ANALYSIS_DATA_DIR / "feature_ablation_data_v5.csv", index=False)

    primary_summary = build_method_summary(primary)
    callback_summary = build_method_summary(callback, factor="callback_policy")
    feature_summary = build_method_summary(feature, factor="feature_specification")
    coverage_scflp = build_problem_coverage(primary, "SCFLP")
    coverage_scmnd = build_problem_coverage(primary, "SCMND")

    tables = {
        "primary_summary": primary_summary,
        "callback_summary": callback_summary,
        "feature_summary": feature_summary,
        "coverage_scflp": coverage_scflp,
        "coverage_scmnd": coverage_scmnd,
        "primary_by_num": build_dimension_summary(primary, "num_subproblems"),
        "primary_by_ratio": build_dimension_summary(primary, "selected_subproblems_ratio"),
        "callback_by_num": build_dimension_summary(callback, "num_subproblems", "callback_policy"),
        "callback_by_ratio": build_dimension_summary(callback, "selected_subproblems_ratio", "callback_policy"),
        "feature_by_num": build_dimension_summary(feature, "num_subproblems", "feature_specification"),
        "feature_by_ratio": build_dimension_summary(feature, "selected_subproblems_ratio", "feature_specification"),
    }
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}_v5.csv", index=False)

    benchmark_table = latex_benchmark_table()
    coverage_scflp_tex = latex_coverage_table(coverage_scflp, "SCFLP", "tab:scflp-coverage", define_columns=True)
    coverage_scmnd_tex = latex_coverage_table(coverage_scmnd, "SCMND", "tab:scmnd-coverage", define_columns=False)
    summary_tables: dict[tuple[str, str], str] = {}
    callback_tables: dict[tuple[str, str], str] = {}
    feature_tables: dict[tuple[str, str], str] = {}
    for problem in ["SCFLP", "SCMND"]:
        for stratum in STRATUM_ORDER:
            slug = "solved" if stratum == STRATUM_ORDER[0] else "time-limited"
            summary_tables[(problem, stratum)] = latex_method_summary_table(
                primary_summary, problem=problem, stratum=stratum,
                label=f"tab:{problem.lower()}-{slug}-summary",
            )
            callback_tables[(problem, stratum)] = latex_method_summary_table(
                callback_summary, problem=problem, stratum=stratum,
                label=f"tab:appendix-callback-{problem.lower()}-{slug}",
                factor="callback_policy", factor_heading="Callback policy",
            )
            feature_tables[(problem, stratum)] = latex_method_summary_table(
                feature_summary, problem=problem, stratum=stratum,
                label=f"tab:appendix-features-{problem.lower()}-{slug}",
                factor="feature_specification", factor_heading="Feature input",
            )

    manuscript_tables = "\n\n".join([
        benchmark_table,
        coverage_scflp_tex,
        summary_tables[("SCFLP", STRATUM_ORDER[0])],
        summary_tables[("SCFLP", STRATUM_ORDER[1])],
        coverage_scmnd_tex,
        summary_tables[("SCMND", STRATUM_ORDER[0])],
        summary_tables[("SCMND", STRATUM_ORDER[1])],
    ]) + "\n"
    appendix_tables = "\n\n".join(
        [callback_tables[(p, st)] for p in ["SCFLP", "SCMND"] for st in STRATUM_ORDER]
        + [feature_tables[(p, st)] for p in ["SCFLP", "SCMND"] for st in STRATUM_ORDER]
    ) + "\n"
    (TABLE_DIR / "benchmark_instances_v5.tex").write_text(benchmark_table + "\n", encoding="utf-8")
    (TABLE_DIR / "scflp_coverage_v5.tex").write_text(coverage_scflp_tex + "\n", encoding="utf-8")
    (TABLE_DIR / "scmnd_coverage_v5.tex").write_text(coverage_scmnd_tex + "\n", encoding="utf-8")
    (LATEX_DIR / "Manuscript_tables_v5.tex").write_text(manuscript_tables, encoding="utf-8")
    (LATEX_DIR / "Appendix_tables_v5.tex").write_text(appendix_tables, encoding="utf-8")

    figure_manifest: list[dict[str, object]] = []
    for problem in ["SCFLP", "SCMND"]:
        for stratum in STRATUM_ORDER:
            for dimension in ["num_subproblems", "selected_subproblems_ratio"]:
                stem = make_main_composite_figure(primary, problem_class=problem, stratum=stratum, dimension=dimension)
                figure_manifest.append({"analysis": "main", "problem_class": problem, "baseline_stratum": stratum, "dimension": dimension, "stem": stem})
                cstem = make_appendix_callback_figure(callback, problem_class=problem, stratum=stratum, dimension=dimension)
                figure_manifest.append({"analysis": "appendix_callback", "problem_class": problem, "baseline_stratum": stratum, "dimension": dimension, "stem": cstem})
                fstem = make_appendix_feature_figure(feature, problem_class=problem, stratum=stratum, dimension=dimension)
                figure_manifest.append({"analysis": "appendix_features", "problem_class": problem, "baseline_stratum": stratum, "dimension": dimension, "stem": fstem})
    pd.DataFrame(figure_manifest).to_csv(TABLE_DIR / "figure_manifest_v5.csv", index=False)
    pd.DataFrame(ZOOM_DIAGNOSTICS).to_csv(TABLE_DIR / "zoom_diagnostics_v5.csv", index=False)
    zoom_manifest_rows = []
    for full_stem, info in ZOOM_FIGURES.items():
        zoom_manifest_rows.append({"full_stem": full_stem, **info})
    pd.DataFrame(zoom_manifest_rows).to_csv(TABLE_DIR / "zoom_manifest_v5.csv", index=False)
    output_file_manifest = organize_output_files()

    results_tex = build_results_draft(
        primary_summary, tables["primary_by_num"], tables["primary_by_ratio"],
        coverage_scflp, coverage_scmnd, benchmark_table,
        coverage_scflp_tex, coverage_scmnd_tex, summary_tables,
    )
    appendix_tex = build_appendix_draft(
        callback_summary, tables["callback_by_num"], tables["callback_by_ratio"],
        feature_summary, tables["feature_by_num"], tables["feature_by_ratio"],
        callback_tables, feature_tables,
    )
    (LATEX_DIR / "Results_section_draft_v5.tex").write_text(results_tex, encoding="utf-8")
    (LATEX_DIR / "Numerical_appendix_draft_v5.tex").write_text(appendix_tex, encoding="utf-8")

    diagnostics = {
        "input_workbook": str(EXCEL_INPUT_FILE),
        "input_worksheet": EXCEL_SHEET_NAME,
        "output_directory": str(OUTPUT_DIR),
        "organized_figure_files": int(len(output_file_manifest)),
        "raw_records": int(len(read_experiment_data())),
        "cleaned_deduplicated_records": int(len(cleaned)),
        "removed_duplicate_records": int(len(removed)),
        "primary_records": int(len(primary)),
        "standard_bd_classification_keys": int(primary[primary["method"].eq("Standard BD")][KEY_COLUMNS].drop_duplicates().shape[0]),
        "primary_time_limit_records": int(primary["is_time_limit"].sum()),
        "primary_missing_objective_records": 0,
        "main_figures_with_zoom_companions": int(len(ZOOM_FIGURES)),
        "zoom_detection_rule": {
            "central_90pct_range_fraction_threshold": SQUASH_CENTRAL_RANGE_THRESHOLD,
            "tail_iqr_threshold": SQUASH_TAIL_IQR_THRESHOLD,
        },
        "primary_filters": {
            "callback": PRIMARY_CALLBACK,
            "feature_input": PRIMARY_FEATURE_SPEC,
            "predictors": PREDICTORS,
            "exact_subproblem_ratios": EXACT_RATIOS,
            "numbers_of_subproblems": NUM_SUBPROBLEMS,
        },
        "appendix_analyses": {
            "callback_policies": [PRIMARY_CALLBACK, SUPPLEMENTARY_CALLBACK],
            "feature_inputs": FEATURE_SPECS,
        },
    }
    (DIAGNOSTICS_DIR / "analysis_diagnostics_v5.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(json.dumps(diagnostics, indent=2))
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

