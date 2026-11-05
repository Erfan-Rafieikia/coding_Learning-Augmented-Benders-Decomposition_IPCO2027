from __future__ import annotations

r"""Publication analysis for the expanded LABD experiment set.

Version 7 extends the publication_analysis_v6 workflow to:
  * Tree, Clustering, Random Sampling, and four parametric predictors;
  * SCFLP plus both SCMND formulations;
  * formulation-safe Standard-BD matching and deduplication;
  * configurable plot inclusion independent of experiment-completeness auditing;
  * conditional one-factor-at-a-time sensitivity outputs;
  * experiment-grid completeness, duplicate, corruption, and baseline audits;
  * parametric diagnostic plots/tables when diagnostic columns are populated;
  * a dedicated cross-weight comparison that separates the transfer-weight
    mechanism (Tree, Cluster, p-median) from the prediction mechanism
    (nonparametric transfer versus RHS Ridge);
  * legacy manuscript-facing figure filenames for SCFLP and MANUSCRIPT_ENVELOPE.

The established publication figure stems intentionally retain the historical ``_v5``
suffix. This is deliberate: existing manuscript \includegraphics references can be
reused by replacing the figure files rather than renaming LaTeX references.
"""

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, StrMethodFormatter

# =============================================================================
# Paths and run controls
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# Expected location: <project>/src/utils/publication_analysis_v7.py
try:
    PROJECT_ROOT = SCRIPT_DIR.parents[1]
except IndexError:  # defensive fallback for ad-hoc execution
    PROJECT_ROOT = SCRIPT_DIR

_DEFAULT_EXCEL_INPUT_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "Experiments - Digital Research Alliance of Canada"
    / "Aggregate_of_experiments.xlsx"
)
EXCEL_INPUT_FILE = Path(os.environ.get("LABD_EXCEL_INPUT_FILE", str(_DEFAULT_EXCEL_INPUT_FILE)))
# If this sheet does not exist, the script tries Merged_solution_log, Sheet1, then the first sheet.
EXCEL_SHEET_NAME = os.environ.get("LABD_EXCEL_SHEET", "").strip() or None

OUTPUT_ROOT = PROJECT_ROOT / "RESULTS" / "outputs_for_paper"
OUTPUT_DIR = Path(os.environ.get("LABD_OUTPUT_DIR", str(OUTPUT_ROOT / "publication_analysis_v7")))
FIGURE_DIR = OUTPUT_DIR / "01_figures_for_overleaf"
ORGANIZED_FIGURE_DIR = OUTPUT_DIR / "02_figures_by_problem"
FIGURE_DATA_DIR = OUTPUT_DIR / "03_figure_source_data"
TABLE_DIR = OUTPUT_DIR / "04_tables"
LATEX_DIR = OUTPUT_DIR / "05_latex"  # reserved for later manuscript-specific narrative generation
ANALYSIS_DATA_DIR = OUTPUT_DIR / "06_analysis_data"
DIAGNOSTICS_DIR = OUTPUT_DIR / "07_diagnostics"

EXPORT_PNG = os.environ.get("LABD_EXPORT_PNG", "0").strip().lower() in {"1", "true", "yes"}
SKIP_FIGURES = os.environ.get("LABD_SKIP_FIGURES", "0").strip().lower() in {"1", "true", "yes"}
SKIP_ARCHIVE = os.environ.get("LABD_SKIP_ARCHIVE", "0").strip().lower() in {"1", "true", "yes"}
GENERATE_PARAMETRIC_DIAGNOSTICS = True
GENERATE_SENSITIVITY_FIGURES = True

# Existing v6 used fast raster-backed PDFs to reduce batch export cost.
FAST_RASTER_PDF = os.environ.get("LABD_FAST_RASTER_PDF", "1").strip().lower() not in {"0", "false", "no"}
_ORIGINAL_FIGURE_SAVEFIG = Figure.savefig
if FAST_RASTER_PDF:
    try:
        from PIL import Image
    except ImportError:
        FAST_RASTER_PDF = False

if FAST_RASTER_PDF:
    def _publication_savefig(self, fname, *args, **kwargs):
        destination = Path(fname)
        if destination.suffix.lower() == ".pdf":
            temp_png = destination.with_suffix(destination.suffix + ".tmp.png")
            png_kwargs = dict(kwargs)
            png_kwargs.pop("format", None)
            png_kwargs.setdefault("dpi", 220)
            _ORIGINAL_FIGURE_SAVEFIG(self, temp_png, *args, format="png", **png_kwargs)
            with Image.open(temp_png) as image:
                image.convert("RGB").save(destination, "PDF", resolution=220.0)
            temp_png.unlink(missing_ok=True)
            return None
        return _ORIGINAL_FIGURE_SAVEFIG(self, fname, *args, **kwargs)

    Figure.savefig = _publication_savefig

# =============================================================================
# Experiment design
# =============================================================================
SCFLP_INSTANCES = [
    "p4_stochastic.txt", "p5_stochastic.txt", "p10_stochastic.txt",
    "p20_stochastic.txt", "p50_stochastic.txt",
]
SCMND_INSTANCES = [
    "r01.1.stoch.json", "r01.2.stoch.json", "r02.1.stoch.json",
    "r02.2.stoch.json", "r04.1.stoch.json", "r04.2.stoch.json",
]
INSTANCES_TO_INCLUDE = SCFLP_INSTANCES + SCMND_INSTANCES
DISPLAY_INSTANCE = {
    "p4_stochastic.txt": "p4", "p5_stochastic.txt": "p5",
    "p10_stochastic.txt": "p10", "p20_stochastic.txt": "p20",
    "p50_stochastic.txt": "p50",
    "r01.1.stoch.json": "r1.1", "r01.2.stoch.json": "r1.2",
    "r02.1.stoch.json": "r2.1", "r02.2.stoch.json": "r2.2",
    "r04.1.stoch.json": "r4.1", "r04.2.stoch.json": "r4.2",
}
INSTANCE_ORDER = {
    name: i for i, name in enumerate([
        "p4", "p5", "p10", "p20", "p50",
        "r1.1", "r1.2", "r2.1", "r2.2", "r4.1", "r4.2",
    ])
}

NUM_SUBPROBLEMS = [100, 300, 500]
EXACT_RATIOS = [0.05, 0.15, 0.25]
FEATURE_SPECS = ["features_only", "covariates_only", "features_plus_covariates"]
PRIMARY_FEATURE_SPEC = "features_only"
PRIMARY_CALLBACK = "ML_at_both"
SUPPLEMENTARY_CALLBACK = "ML_at_LP_exact_at_integer"
EXPECTED_CALLBACKS = [PRIMARY_CALLBACK, SUPPLEMENTARY_CALLBACK]

SCMND_MANUSCRIPT_FORMULATION = "MANUSCRIPT_ENVELOPE"
SCMND_ALTERNATIVE_FORMULATION = "CURRENT_SOFT_CAPACITY"
EXPECTED_SCMND_FORMULATIONS = [SCMND_MANUSCRIPT_FORMULATION, SCMND_ALTERNATIVE_FORMULATION]
SCFLP_GROUP = "SCFLP"
UNSPECIFIED_SCMND_GROUP = "SCMND_UNSPECIFIED_FORMULATION"

# Comment out any nonbaseline method here to remove it from figures/tables.
# Standard BD is always retained internally because Delta and tau require it.
PLOT_METHOD_KEYS = [
    "TREE",
    "CLUSTERING",
    "RANDOM_SAMPLING",
    "PARAMETRIC_RIDGE_REPAIR",
    "PARAMETRIC_RHS_RIDGE_REPAIR",
    "PARAMETRIC_MANUSCRIPT",
    "PARAMETRIC_PENALIZED_ERRORS",
]

# Completeness auditing is intentionally independent of PLOT_METHOD_KEYS.
EXPECTED_METHOD_KEYS = [
    "TREE",
    "CLUSTERING",
    "RANDOM_SAMPLING",
    "PARAMETRIC_RIDGE_REPAIR",
    "PARAMETRIC_RHS_RIDGE_REPAIR",
    "PARAMETRIC_MANUSCRIPT",
    "PARAMETRIC_PENALIZED_ERRORS",
]
PARAMETRIC_METHOD_KEYS = [
    "PARAMETRIC_RIDGE_REPAIR",
    "PARAMETRIC_RHS_RIDGE_REPAIR",
    "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR",
    "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR",
    "PARAMETRIC_MANUSCRIPT",
    "PARAMETRIC_PENALIZED_ERRORS",
]
FEATURE_BASED_METHOD_KEYS = [k for k in EXPECTED_METHOD_KEYS if k != "RANDOM_SAMPLING"]

# Dedicated factorial comparison requested after the manuscript separated the
# transfer-weight construction from the prediction mechanism.  These methods
# are deliberately NOT added to PLOT_METHOD_KEYS or EXPECTED_METHOD_KEYS, so
# the established paper figures and completeness audit remain unchanged.
WEIGHTING_CROSS_METHOD_KEYS = [
    "TREE",
    "CLUSTERING",
    "PMEDIAN_WEIGHTED_TRANSFER",
    "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR",
    "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR",
    "PARAMETRIC_RHS_RIDGE_REPAIR",
]
WEIGHTING_MECHANISM = {
    "TREE": "Tree",
    "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR": "Tree",
    "CLUSTERING": "Cluster",
    "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR": "Cluster",
    "PMEDIAN_WEIGHTED_TRANSFER": "P-Median",
    "PARAMETRIC_RHS_RIDGE_REPAIR": "P-Median",
}
PREDICTION_MECHANISM = {
    "TREE": "Nonparametric",
    "CLUSTERING": "Nonparametric",
    "PMEDIAN_WEIGHTED_TRANSFER": "Nonparametric",
    "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR": "RHS Ridge",
    "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR": "RHS Ridge",
    "PARAMETRIC_RHS_RIDGE_REPAIR": "RHS Ridge",
}
WEIGHTING_CROSS_LABELS = {
    "TREE": "Nonparametric--Tree",
    "CLUSTERING": "Nonparametric--Cluster",
    "PMEDIAN_WEIGHTED_TRANSFER": "Nonparametric--P-Median",
    "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR": "RHS Ridge--Tree",
    "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR": "RHS Ridge--Cluster",
    "PARAMETRIC_RHS_RIDGE_REPAIR": "RHS Ridge--P-Median",
}

METHOD_LABELS = {
    "STANDARD_BD": "Standard BD",
    "TREE": "LABD--Tree",
    "CLUSTERING": "LABD--Cluster",
    "RANDOM_SAMPLING": "RND",
    "PARAMETRIC_RIDGE_REPAIR": "Parametric--Ridge Repair",
    "PARAMETRIC_RHS_RIDGE_REPAIR": "Parametric--RHS Ridge Repair",
    "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR": "Parametric--RHS Ridge (Tree Anchor)",
    "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR": "Parametric--RHS Ridge (Cluster Anchor)",
    "PMEDIAN_WEIGHTED_TRANSFER": "LABD--P-Median Transfer",
    "PARAMETRIC_MANUSCRIPT": "Parametric--Manuscript",
    "PARAMETRIC_PENALIZED_ERRORS": "Parametric--Penalized Errors",
}
METHOD_COLORS = {
    "Standard BD": "#7f7f7f",
    "LABD--Tree": "#1f77b4",
    "LABD--Cluster": "#ff7f0e",
    "RND": "#2ca02c",
    "Parametric--Ridge Repair": "#d62728",
    "Parametric--RHS Ridge Repair": "#9467bd",
    "Parametric--Manuscript": "#8c564b",
    "Parametric--Penalized Errors": "#e377c2",
    "R": "#1f77b4",
    "C": "#ff7f0e",
    "R+C": "#2ca02c",
    "ML at both": "#1f77b4",
    "Exact at integer": "#d62728",
    "Nonparametric": "#1f77b4",
    "RHS Ridge": "#9467bd",
}

METHOD_ORDER = [METHOD_LABELS["STANDARD_BD"]] + [METHOD_LABELS[k] for k in PLOT_METHOD_KEYS]
NONBASELINE_METHOD_ORDER = [METHOD_LABELS[k] for k in PLOT_METHOD_KEYS]
LABD_METHODS = [METHOD_LABELS[k] for k in PLOT_METHOD_KEYS if k != "RANDOM_SAMPLING"]
PARAMETRIC_METHODS = [METHOD_LABELS[k] for k in PLOT_METHOD_KEYS if k in PARAMETRIC_METHOD_KEYS]

FEATURE_LABELS = {"features_only": "R", "covariates_only": "C", "features_plus_covariates": "R+C"}
CALLBACK_LABELS = {
    PRIMARY_CALLBACK: "Predicted cuts at LP and integer solutions",
    SUPPLEMENTARY_CALLBACK: "Predicted cuts at LP; exact cuts at integer solutions",
}
CALLBACK_SHORT = {PRIMARY_CALLBACK: "ML at both", SUPPLEMENTARY_CALLBACK: "Exact at integer"}
STRATUM_ORDER = ["Standard BD solved", "Standard BD time-limited"]

# Reference values used for the primary analysis and one-factor-at-a-time sensitivities.
REFERENCE_HYPERPARAMETERS = {
    "n_trials": 5,
    "m_trials": 0,
    "n_walks": 2,
    "walk_length": 5,
    "feature_dim": 3,
    "window": 5,
}
# Edit these lists when intentionally running new sensitivity levels. Values may also
# be discovered from the workbook; the explicit lists let the audit detect an entirely
# missing intended sensitivity level.
EXPECTED_HYPERPARAMETER_VALUES = {
    "n_trials": [5],
    "m_trials": [0],
    "n_walks": [2],
    "walk_length": [5],
    "feature_dim": [3],
    "window": [5],
}
SENSITIVITY_PARAMETERS = list(REFERENCE_HYPERPARAMETERS)

# =============================================================================
# Styling and metric definitions
# =============================================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CUT_COMPONENT_COLUMNS = [
    "phase0_cuts_added",
    "cuts_mip_trial",
    "cuts_rel_trial",
    "cuts_mip_selected",
    "cuts_rel_selected",
    "cuts_mip_unselected_total",
    "cuts_rel_unselected_total",
]
EXTRA_METRICS = {
    "feature_generation_time_ratio_pct": {
        "stem": "feature_generation_time_ratio",
        "label": "Feature-generation time share (%)",
        "method_keys": FEATURE_BASED_METHOD_KEYS,
    },
    "warm_up_time_ratio_pct": {
        "stem": "warm_up_time_ratio",
        "label": "Warm-up time share (%)",
        "method_keys": FEATURE_BASED_METHOD_KEYS,
    },
    "solution_time_sec": {
        "stem": "solution_time",
        "label": "Solution time (sec)",
        "method_keys": ["STANDARD_BD"] + PLOT_METHOD_KEYS,
    },
    "total_cuts": {
        "stem": "total_cuts",
        "label": "Total cuts",
        "method_keys": ["STANDARD_BD"] + PLOT_METHOD_KEYS,
    },
    "ml_cut_share_pct": {
        "stem": "ml_cut_share",
        "label": "ML-generated cut share (%)",
        "method_keys": PLOT_METHOD_KEYS,
    },
}
PARAMETRIC_DIAGNOSTICS = {
    "parametric_training_time_sec": "Parametric training time (sec)",
    "parametric_prediction_time_sec": "Parametric prediction time (sec)",
    "parametric_feasibility_time_sec": "Feasibility-check time (sec)",
    "parametric_repair_time_sec": "Repair time (sec)",
    "parametric_raw_infeasible_rate_pct": "Raw infeasible prediction rate (%)",
    "parametric_exact_fallback_rate_pct": "Exact-fallback rate (%)",
    "parametric_cut_addition_rate_pct": "Parametric cut-addition rate (%)",
    "parametric_max_condition_number": "Maximum condition number",
    "parametric_mean_repair_gamma": "Mean repair gamma",
}

# Robust zoom rule retained from v6.
SQUASH_CENTRAL_RANGE_THRESHOLD = 0.22
SQUASH_TAIL_IQR_THRESHOLD = 5.0
ZOOM_FIGURES: dict[str, dict[str, object]] = {}
ZOOM_DIAGNOSTICS: list[dict[str, object]] = []

# =============================================================================
# Generic helpers
# =============================================================================
def _safe_slug(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _extract_instance_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("\\\\", "/", regex=False).str.rsplit("/", n=1).str[-1]


def _canonical_json(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except Exception:
        return text


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _parametric_config_value(value: object, key: str) -> object:
    """Extract one field from the JSON parametric_config cell, if present."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return np.nan
    try:
        payload = json.loads(text)
    except Exception:
        return np.nan
    return payload.get(key, np.nan) if isinstance(payload, dict) else np.nan


def _first_existing_sheet(path: Path) -> str:
    xls = pd.ExcelFile(path, engine="openpyxl")
    if EXCEL_SHEET_NAME and EXCEL_SHEET_NAME in xls.sheet_names:
        return EXCEL_SHEET_NAME
    for preferred in ["Merged_solution_log", "Sheet1"]:
        if preferred in xls.sheet_names:
            return preferred
    if not xls.sheet_names:
        raise ValueError(f"No worksheets found in {path}")
    return xls.sheet_names[0]


def _analysis_group(problem_class: object, scmnd_formulation: object) -> str:
    if str(problem_class).upper() == "SCFLP":
        return SCFLP_GROUP
    formulation = str(scmnd_formulation).strip() if scmnd_formulation is not None else ""
    if not formulation or formulation.lower() in {"nan", "none"}:
        return UNSPECIFIED_SCMND_GROUP
    return f"SCMND_{formulation}"


def _group_problem_class(group: str) -> str:
    return "SCFLP" if group == SCFLP_GROUP else "SCMND"


def _group_formulation(group: str) -> str | None:
    if group == SCFLP_GROUP:
        return None
    if group.startswith("SCMND_"):
        return group[len("SCMND_"):]
    return None


def _group_file_suffix(group: str) -> str:
    """Legacy names for SCFLP and manuscript SCMND; suffix only alternative SCMND."""
    if group == f"SCMND_{SCMND_ALTERNATIVE_FORMULATION}":
        return "_current_soft_capacity"
    if group == UNSPECIFIED_SCMND_GROUP:
        return "_unspecified_scmnd_formulation"
    return ""


def _group_legacy_problem_token(group: str) -> str:
    return "scflp" if group == SCFLP_GROUP else "scmnd"


def _group_display(group: str) -> str:
    return {
        SCFLP_GROUP: "SCFLP",
        f"SCMND_{SCMND_MANUSCRIPT_FORMULATION}": "SCMND--Manuscript Envelope",
        f"SCMND_{SCMND_ALTERNATIVE_FORMULATION}": "SCMND--Current Soft Capacity",
        UNSPECIFIED_SCMND_GROUP: "SCMND--Unspecified Formulation",
    }.get(group, group.replace("_", " "))


def _method_key_from_row(row: pd.Series) -> str:
    if str(row.get("method_category", "")) == "Exact":
        return "STANDARD_BD"
    ml_type = str(row.get("ml_method_type", ""))
    predictor = str(row.get("predictor_type", ""))
    if ml_type == "Random_Sampling":
        return "RANDOM_SAMPLING"
    if ml_type == "ML_Augmented_BD" and predictor in METHOD_LABELS:
        return predictor
    return "OTHER"


def _is_valid_record_frame(df: pd.DataFrame) -> pd.Series:
    objective = pd.to_numeric(df.get("original_Prob_Obj_Value"), errors="coerce")
    runtime = pd.to_numeric(df.get("solution_time_sec"), errors="coerce")
    status = df.get("stopping_reason", pd.Series(index=df.index, dtype=object)).fillna("OPTIMAL")
    status_ok = status.isin(["OPTIMAL", "TIME_LIMIT"])
    return objective.notna() & runtime.notna() & runtime.ge(0) & status_ok


def _reference_hyperparameter_mask(df: pd.DataFrame, *, allow_missing: bool = True, exclude: str | None = None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, reference in REFERENCE_HYPERPARAMETERS.items():
        if column == exclude or column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if allow_missing:
            mask &= values.isna() | np.isclose(values, float(reference), atol=1e-12, rtol=0)
        else:
            mask &= np.isclose(values, float(reference), atol=1e-12, rtol=0)
    return mask


def _method_labels(keys: Sequence[str]) -> list[str]:
    return [METHOD_LABELS[k] for k in keys if k in METHOD_LABELS]

# =============================================================================
# Reading, cleaning, deduplication, classification
# =============================================================================
def read_experiment_data() -> tuple[pd.DataFrame, str]:
    if not EXCEL_INPUT_FILE.exists():
        raise FileNotFoundError(f"Experiment workbook not found: {EXCEL_INPUT_FILE}")
    sheet = _first_existing_sheet(EXCEL_INPUT_FILE)
    data = pd.read_excel(EXCEL_INPUT_FILE, sheet_name=sheet, engine="openpyxl")
    return data, sheet


def prepare_raw_data(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    required = [
        "problem_class", "instance_filename", "num_subproblems", "method_category",
        "ml_method_type", "ml_callback_type", "predictor_type",
        "selected_subproblems_ratio", "use_covariates",
        "original_Prob_Obj_Value", "solution_time_sec",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Input workbook is missing required columns: {missing}")

    if "scmnd_formulation" not in out.columns:
        out["scmnd_formulation"] = np.nan
    if "parametric_config" not in out.columns:
        out["parametric_config"] = np.nan

    out["deterministic_instance"] = _extract_instance_name(out["instance_filename"])
    out = out[out["deterministic_instance"].isin(INSTANCES_TO_INCLUDE)].copy()
    out = out[pd.to_numeric(out["num_subproblems"], errors="coerce").isin(NUM_SUBPROBLEMS)].copy()
    out["num_subproblems"] = pd.to_numeric(out["num_subproblems"], errors="coerce")
    out["selected_subproblems_ratio"] = pd.to_numeric(out["selected_subproblems_ratio"], errors="coerce")
    for column in SENSITIVITY_PARAMETERS + ["seed", "time_limit_sec", "MIPGap_target"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["scmnd_formulation"] = out["scmnd_formulation"].where(out["problem_class"].eq("SCMND"), np.nan)
    out["analysis_group"] = [
        _analysis_group(p, f) for p, f in zip(out["problem_class"], out["scmnd_formulation"])
    ]
    out["parametric_config_canonical"] = out["parametric_config"].map(_canonical_json)

    # Historical result workbooks store the p-median implementation choice only
    # inside the JSON parametric_config column.  Expose it explicitly so the
    # weighting-cross analysis can select the final GREEDY/build-and-swap
    # RHS-Ridge runs and exclude the earlier MIP diagnostic runs.
    if "parametric_pmedian_solver" not in out.columns:
        out["parametric_pmedian_solver"] = out["parametric_config"].map(
            lambda value: _parametric_config_value(value, "parametric_pmedian_solver")
        )

    out["method_key"] = out.apply(_method_key_from_row, axis=1)
    out["method"] = out["method_key"].map(METHOD_LABELS).fillna("Other")
    out["instance"] = out["deterministic_instance"].map(DISPLAY_INSTANCE).fillna(out["deterministic_instance"])
    return out


DUPLICATE_KEY_COLUMNS = [
    "analysis_group", "deterministic_instance", "num_subproblems",
    "method_category", "ml_method_type", "ml_callback_type", "exact_callback_type",
    "predictor_type", "use_prediction", "prediction_method",
    "selected_subproblems_ratio", "n_trials", "m_trials", "n_walks", "walk_length",
    "feature_dim", "window", "min_count", "sg", "use_covariates", "seed",
    "parametric_config_canonical",
]


def _quality_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    score += pd.to_numeric(df.get("original_Prob_Obj_Value"), errors="coerce").notna() * 100
    score += pd.to_numeric(df.get("solution_time_sec"), errors="coerce").notna() * 50
    status = df.get("stopping_reason", pd.Series(index=df.index, dtype=object)).fillna("OPTIMAL")
    score += status.isin(["OPTIMAL", "TIME_LIMIT"]) * 20
    if "time_limit_sec" in df.columns:
        t = pd.to_numeric(df["time_limit_sec"], errors="coerce")
        score += t.eq(10800) * 5 + t.eq(3600) * 2
    if "MIPGap_target" in df.columns:
        g = pd.to_numeric(df["MIPGap_target"], errors="coerce")
        score += (g.notna() & g.ne(0)) * 3
    # Deterministic tiebreaker: earlier original row wins after equal quality.
    return score


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    keys = [c for c in DUPLICATE_KEY_COLUMNS if c in out.columns]
    if not keys:
        return out.reset_index(drop=True), out.iloc[0:0].copy(), pd.DataFrame()
    duplicate_mask = out.duplicated(subset=keys, keep=False)
    if not duplicate_mask.any():
        return out.reset_index(drop=True), out.iloc[0:0].copy(), pd.DataFrame()

    kept: list[int] = []
    removed: list[int] = []
    report_rows: list[dict[str, object]] = []
    duplicate_rows = out[duplicate_mask]
    for group_id, (_, group) in enumerate(duplicate_rows.groupby(keys, dropna=False, sort=False), start=1):
        scores = _quality_score(group)
        best_score = scores.max()
        candidates = group.loc[scores.eq(best_score)]
        keep_idx = int(candidates.index.min())
        kept.append(keep_idx)
        removed_group = [int(i) for i in group.index if i != keep_idx]
        removed.extend(removed_group)
        report_rows.append({
            "duplicate_group_id": group_id,
            "records_in_group": int(len(group)),
            "kept_original_index": keep_idx,
            "removed_original_indices": ",".join(map(str, removed_group)),
            "kept_quality_score": float(best_score),
            **{c: group.iloc[0][c] for c in keys},
        })

    unique = out[~duplicate_mask]
    deduped = pd.concat([unique, out.loc[kept]], axis=0).sort_index().reset_index(drop=True)
    removed_df = out.loc[removed].sort_index().reset_index(drop=True)
    return deduped, removed_df, pd.DataFrame(report_rows)


def normalize_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stopping_reason"] = out["stopping_reason"].fillna("OPTIMAL")
    out["is_time_limit"] = out["stopping_reason"].eq("TIME_LIMIT")
    out["valid_record"] = _is_valid_record_frame(out)
    return out


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric = CUT_COMPONENT_COLUMNS + [
        "cuts_mip_ml", "cuts_rel_ml", "feature_gen_time_sec",
        "phase1_execution_time_sec", "solution_time_sec",
    ] + list(PARAMETRIC_DIAGNOSTICS)
    out = _coerce_numeric_columns(out, numeric)

    for column in CUT_COMPONENT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    for column in ["cuts_mip_ml", "cuts_rel_ml", "feature_gen_time_sec", "phase1_execution_time_sec"]:
        if column not in out.columns:
            out[column] = np.nan

    out["total_cuts"] = out[CUT_COMPONENT_COLUMNS].fillna(0.0).sum(axis=1)
    ml_cuts = out[["cuts_mip_ml", "cuts_rel_ml"]].fillna(0.0).sum(axis=1)
    out["ml_cut_share_pct"] = np.where(out["total_cuts"].gt(0), 100.0 * ml_cuts / out["total_cuts"], np.nan)
    out["feature_generation_time_ratio_pct"] = np.where(
        out["solution_time_sec"].gt(0), 100.0 * out["feature_gen_time_sec"] / out["solution_time_sec"], np.nan
    )
    out["warm_up_time_ratio_pct"] = np.where(
        out["solution_time_sec"].gt(0), 100.0 * out["phase1_execution_time_sec"] / out["solution_time_sec"], np.nan
    )

    for source, target in [
        ("parametric_raw_infeasible_rate", "parametric_raw_infeasible_rate_pct"),
        ("parametric_exact_fallback_rate", "parametric_exact_fallback_rate_pct"),
        ("parametric_cut_addition_rate", "parametric_cut_addition_rate_pct"),
    ]:
        if source in out.columns:
            out[target] = 100.0 * pd.to_numeric(out[source], errors="coerce")
        else:
            out[target] = np.nan
    return out


def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    raw, sheet = read_experiment_data()
    prepared = prepare_raw_data(raw)
    deduped, removed, duplicate_report = remove_duplicates(prepared)
    cleaned = add_derived_metrics(normalize_status(deduped))
    return cleaned, removed, duplicate_report, prepared, sheet

# =============================================================================
# Completeness and data-quality audit
# =============================================================================
def _expected_base_instance_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for instance in SCFLP_INSTANCES:
        for nsp in NUM_SUBPROBLEMS:
            rows.append({"analysis_group": SCFLP_GROUP, "deterministic_instance": instance, "num_subproblems": nsp})
    for formulation in EXPECTED_SCMND_FORMULATIONS:
        group = f"SCMND_{formulation}"
        for instance in SCMND_INSTANCES:
            for nsp in NUM_SUBPROBLEMS:
                rows.append({"analysis_group": group, "deterministic_instance": instance, "num_subproblems": nsp})
    return rows


def _expected_core_grid() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    refs = REFERENCE_HYPERPARAMETERS.copy()
    for base in _expected_base_instance_rows():
        rows.append({
            **base, "method_key": "STANDARD_BD", "ml_callback_type": "__NA__",
            "selected_subproblems_ratio": np.nan, "use_covariates": "__NA__", **refs,
        })
        for method_key in EXPECTED_METHOD_KEYS:
            callbacks = EXPECTED_CALLBACKS
            features = ["__NA__"] if method_key == "RANDOM_SAMPLING" else FEATURE_SPECS
            for callback in callbacks:
                for ratio in EXACT_RATIOS:
                    for feature in features:
                        rows.append({
                            **base, "method_key": method_key, "ml_callback_type": callback,
                            "selected_subproblems_ratio": ratio, "use_covariates": feature, **refs,
                        })
    return pd.DataFrame(rows)


def _normalize_audit_observed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["audit_callback"] = np.where(out["method_key"].eq("STANDARD_BD"), "__NA__", out["ml_callback_type"].fillna("__MISSING__"))
    out["audit_ratio"] = np.where(out["method_key"].eq("STANDARD_BD"), np.nan, out["selected_subproblems_ratio"])
    out["audit_feature"] = np.where(
        out["method_key"].isin(["STANDARD_BD", "RANDOM_SAMPLING"]),
        "__NA__",
        out["use_covariates"].fillna("__MISSING__"),
    )
    return out


def _audit_key_columns() -> list[str]:
    return [
        "analysis_group", "deterministic_instance", "num_subproblems", "method_key",
        "audit_callback", "audit_ratio", "audit_feature",
    ] + [f"audit_{p}" for p in SENSITIVITY_PARAMETERS]


def _with_audit_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_audit_observed(df)
    for p in SENSITIVITY_PARAMETERS:
        values = pd.to_numeric(out[p], errors="coerce") if p in out.columns else pd.Series(np.nan, index=out.index)
        # Exact and RND may legitimately not populate feature-generation hyperparameters.
        out[f"audit_{p}"] = np.where(out["method_key"].eq("STANDARD_BD"), REFERENCE_HYPERPARAMETERS[p], values)
        # Random Sampling does not use the LABD training/embedding hyperparameters,
        # so their spreadsheet values (often blank) must not affect completeness.
        out[f"audit_{p}"] = np.where(out["method_key"].eq("RANDOM_SAMPLING"), REFERENCE_HYPERPARAMETERS[p], out[f"audit_{p}"])
        out[f"audit_{p}"] = pd.to_numeric(out[f"audit_{p}"], errors="coerce")
    return out


def _expected_with_audit_keys() -> pd.DataFrame:
    expected = _expected_core_grid().rename(columns={
        "ml_callback_type": "audit_callback",
        "selected_subproblems_ratio": "audit_ratio",
        "use_covariates": "audit_feature",
    })
    for p in SENSITIVITY_PARAMETERS:
        expected[f"audit_{p}"] = pd.to_numeric(expected[p], errors="coerce")
    return expected


def audit_experiment_completeness(cleaned: pd.DataFrame, duplicate_report: pd.DataFrame) -> dict[str, pd.DataFrame]:
    expected = _expected_with_audit_keys()
    observed = _with_audit_keys(cleaned)
    keys = _audit_key_columns()

    # Reference-hyperparameter slice. Keep all methods at this stage so genuinely
    # unexpected methods/configurations are reported rather than silently discarded.
    observed_reference = observed[observed["analysis_group"].isin(expected["analysis_group"].unique())].copy()
    ref_mask = pd.Series(True, index=observed_reference.index)
    for p, ref in REFERENCE_HYPERPARAMETERS.items():
        ref_mask &= np.isclose(pd.to_numeric(observed_reference[f"audit_{p}"], errors="coerce"), float(ref), atol=1e-12, rtol=0)
    observed_reference = observed_reference[ref_mask]
    observed_core = observed_reference[observed_reference["method_key"].isin(["STANDARD_BD"] + EXPECTED_METHOD_KEYS)].copy()

    grouped = observed_core.groupby(keys, dropna=False, sort=False).agg(
        observed_records=("method_key", "size"),
        valid_records=("valid_record", "sum"),
    ).reset_index()

    merged = expected[keys].merge(grouped, on=keys, how="left")
    merged["observed_records"] = merged["observed_records"].fillna(0).astype(int)
    merged["valid_records"] = merged["valid_records"].fillna(0).astype(int)
    missing = merged[merged["observed_records"].eq(0)].copy()
    failed = merged[merged["observed_records"].gt(0) & merged["valid_records"].eq(0)].copy()

    # Unexpected core configurations (wrong callback/feature/ratio/method/group at reference hypers).
    expected_key_index = pd.MultiIndex.from_frame(expected[keys])
    observed_key_index = pd.MultiIndex.from_frame(observed_reference[keys])
    unexpected_mask = ~observed_key_index.isin(expected_key_index)
    unexpected = observed_reference.loc[unexpected_mask].copy()

    # Standard BD coverage for every analysis key present in nonbaseline data.
    base_keys = ["analysis_group", "deterministic_instance", "num_subproblems"]
    baseline = cleaned[cleaned["method_key"].eq("STANDARD_BD")][base_keys].drop_duplicates()
    nonbaseline_keys = cleaned[cleaned["method_key"].isin(EXPECTED_METHOD_KEYS)][base_keys].drop_duplicates()
    missing_baseline = nonbaseline_keys.merge(baseline.assign(has_standard_bd=True), on=base_keys, how="left")
    missing_baseline = missing_baseline[missing_baseline["has_standard_bd"].isna()].drop(columns="has_standard_bd")

    # Formulation validation.
    formulation_issues = cleaned[
        cleaned["problem_class"].eq("SCMND")
        & ~cleaned["analysis_group"].isin([f"SCMND_{x}" for x in EXPECTED_SCMND_FORMULATIONS])
    ].copy()

    summary = pd.DataFrame([{
        "expected_core_configurations": int(len(expected)),
        "observed_expected_configurations": int((merged["observed_records"] > 0).sum()),
        "valid_expected_configurations": int((merged["valid_records"] > 0).sum()),
        "missing_configurations": int(len(missing)),
        "present_but_invalid_configurations": int(len(failed)),
        "duplicate_groups": int(len(duplicate_report)),
        "unexpected_reference_configurations": int(len(unexpected)),
        "missing_standard_bd_keys": int(len(missing_baseline)),
        "scmnd_formulation_issue_records": int(len(formulation_issues)),
    }])

    return {
        "expected_core_grid": expected,
        "completeness_status": merged,
        "missing_configurations": missing,
        "failed_or_corrupted_configurations": failed,
        "unexpected_configurations": unexpected,
        "missing_standard_bd_references": missing_baseline,
        "scmnd_formulation_issues": formulation_issues,
        "summary": summary,
    }


def audit_sensitivity_levels(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    working = cleaned[cleaned["method_key"].isin(FEATURE_BASED_METHOD_KEYS)].copy()
    for parameter in SENSITIVITY_PARAMETERS:
        observed = sorted(pd.to_numeric(working.get(parameter), errors="coerce").dropna().unique().tolist())
        intended = sorted(set(float(v) for v in EXPECTED_HYPERPARAMETER_VALUES.get(parameter, [])))
        for value in sorted(set(observed) | set(intended)):
            rows.append({
                "parameter": parameter,
                "value": value,
                "declared_expected": value in intended,
                "observed": value in observed,
                "reference_value": float(REFERENCE_HYPERPARAMETERS[parameter]),
            })
    return pd.DataFrame(rows)

# =============================================================================
# Standard-BD reference attachment and analysis selections
# =============================================================================
def attach_standard_bd_reference(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    key_columns = ["analysis_group", "deterministic_instance", "num_subproblems"]
    baseline = out[out["method_key"].eq("STANDARD_BD")].copy()

    duplicate_baseline = baseline[baseline.duplicated(key_columns, keep=False)].copy()
    if not duplicate_baseline.empty:
        # Do not silently many-to-many merge. Keep the highest-quality baseline per key.
        baseline = baseline.assign(_quality=_quality_score(baseline))
        baseline = baseline.sort_values([*key_columns, "_quality"], ascending=[True, True, True, False])
        baseline = baseline.drop_duplicates(key_columns, keep="first").drop(columns="_quality")

    keep = key_columns + [
        "original_Prob_Obj_Value", "solution_time_sec", "stopping_reason", "is_time_limit", "ObjBound_LB",
    ]
    keep = [c for c in keep if c in baseline.columns]
    baseline = baseline[keep].rename(columns={
        "original_Prob_Obj_Value": "standard_bd_objective",
        "solution_time_sec": "standard_bd_runtime_sec",
        "stopping_reason": "standard_bd_stopping_reason",
        "is_time_limit": "standard_bd_time_limit",
        "ObjBound_LB": "standard_bd_lower_bound",
    })
    out = out.merge(baseline, on=key_columns, how="left", validate="many_to_one")

    out["baseline_stratum"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[out["standard_bd_time_limit"].eq(True), "baseline_stratum"] = STRATUM_ORDER[1]
    out.loc[out["standard_bd_time_limit"].eq(False), "baseline_stratum"] = STRATUM_ORDER[0]
    objective = pd.to_numeric(out["original_Prob_Obj_Value"], errors="coerce")
    bd_objective = pd.to_numeric(out["standard_bd_objective"], errors="coerce")
    runtime = pd.to_numeric(out["solution_time_sec"], errors="coerce")
    bd_runtime = pd.to_numeric(out["standard_bd_runtime_sec"], errors="coerce")
    out["relative_objective_difference_pct"] = np.where(
        objective.notna() & bd_objective.notna() & bd_objective.ne(0),
        100.0 * (objective - bd_objective) / bd_objective.abs(), np.nan,
    )
    out["relative_runtime"] = np.where(
        runtime.notna() & bd_runtime.notna() & bd_runtime.gt(0), runtime / bd_runtime, np.nan,
    )
    out["objective_available"] = objective.notna()
    missing_ref = out[out["method_key"].ne("STANDARD_BD") & out["standard_bd_runtime_sec"].isna()].copy()
    return out, missing_ref


def select_primary(cleaned: pd.DataFrame) -> pd.DataFrame:
    exact = cleaned["method_key"].eq("STANDARD_BD")
    allowed = cleaned["method_key"].isin(PLOT_METHOD_KEYS)
    callback = cleaned["ml_callback_type"].eq(PRIMARY_CALLBACK)
    ratio = cleaned["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    feature = (
        cleaned["method_key"].eq("RANDOM_SAMPLING")
        | cleaned["use_covariates"].eq(PRIMARY_FEATURE_SPEC)
    )
    hypers = _reference_hyperparameter_mask(cleaned, allow_missing=True)
    return cleaned[exact | (allowed & callback & ratio & feature & hypers)].copy()


def select_callback_robustness(cleaned: pd.DataFrame) -> pd.DataFrame:
    exact = cleaned["method_key"].eq("STANDARD_BD")
    allowed = cleaned["method_key"].isin(PLOT_METHOD_KEYS)
    callbacks = cleaned["ml_callback_type"].isin(EXPECTED_CALLBACKS)
    ratio = cleaned["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    feature = cleaned["method_key"].eq("RANDOM_SAMPLING") | cleaned["use_covariates"].eq(PRIMARY_FEATURE_SPEC)
    hypers = _reference_hyperparameter_mask(cleaned, allow_missing=True)
    out = cleaned[exact | (allowed & callbacks & ratio & feature & hypers)].copy()
    out["callback_policy"] = out["ml_callback_type"].map(CALLBACK_LABELS)
    out["callback_short"] = out["ml_callback_type"].map(CALLBACK_SHORT)
    return out


def select_feature_ablation(cleaned: pd.DataFrame) -> pd.DataFrame:
    exact = cleaned["method_key"].eq("STANDARD_BD")
    allowed = cleaned["method_key"].isin([k for k in PLOT_METHOD_KEYS if k != "RANDOM_SAMPLING"])
    callback = cleaned["ml_callback_type"].eq(PRIMARY_CALLBACK)
    ratio = cleaned["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    feature = cleaned["use_covariates"].isin(FEATURE_SPECS)
    hypers = _reference_hyperparameter_mask(cleaned, allow_missing=True)
    out = cleaned[exact | (allowed & callback & ratio & feature & hypers)].copy()
    out["feature_specification"] = out["use_covariates"].map(FEATURE_LABELS)
    return out


def _match_numeric_if_present(df: pd.DataFrame, column: str, value: float) -> pd.Series:
    """Match a numeric production setting, allowing legacy rows where it was not logged."""
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    values = pd.to_numeric(df[column], errors="coerce")
    return values.isna() | np.isclose(values, float(value), atol=1e-12, rtol=0)


def select_weighting_cross(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Select the apples-to-apples 3x2 transfer-weight experiment.

    Existing Tree, Cluster, and RHS-Ridge/P-Median rows are reused from the
    original experiment set.  The three new cross-combinations are selected
    under exactly the same paper-facing callback, feature, seed, time-limit,
    gap, warm-up, and exact-share settings.
    """
    exact = cleaned["method_key"].eq("STANDARD_BD")
    allowed = cleaned["method_key"].isin(WEIGHTING_CROSS_METHOD_KEYS)
    callback = cleaned["ml_callback_type"].eq(PRIMARY_CALLBACK)
    ratio = cleaned["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    feature = cleaned["use_covariates"].eq(PRIMARY_FEATURE_SPEC)
    hypers = _reference_hyperparameter_mask(cleaned, allow_missing=True)

    universal = pd.Series(True, index=cleaned.index)
    universal &= _match_numeric_if_present(cleaned, "seed", 42)
    universal &= _match_numeric_if_present(cleaned, "time_limit_sec", 10800)
    universal &= _match_numeric_if_present(cleaned, "MIPGap_target", 0.01)

    # Match the final production selector settings used by the existing methods.
    tree_rows = cleaned["method_key"].isin(["TREE", "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR"])
    tree_ok = (
        _match_numeric_if_present(cleaned, "tree_min_impurity_decrease", 0.0)
        & _match_numeric_if_present(cleaned, "tree_quantile_grid_size", 10)
    )
    cluster_rows = cleaned["method_key"].isin(["CLUSTERING", "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR"])
    cluster_ok = (
        _match_numeric_if_present(cleaned, "clustering_random_state", 42)
        & _match_numeric_if_present(cleaned, "clustering_max_iter", 300)
        & _match_numeric_if_present(cleaned, "clustering_tol", 1e-4)
    )
    if "clustering_init_method" in cleaned.columns:
        init = cleaned["clustering_init_method"].astype(str).str.strip().str.lower()
        cluster_ok &= cleaned["clustering_init_method"].isna() | init.eq("k-means++")

    selector_ok = (~tree_rows | tree_ok) & (~cluster_rows | cluster_ok)

    # Current paper-facing p-median runs use the deterministic GREEDY/build-and-swap
    # selector.  Historical MIP diagnostic rows are excluded when the solver field
    # is available.
    pmedian_rows = cleaned["method_key"].isin(
        ["PMEDIAN_WEIGHTED_TRANSFER", "PARAMETRIC_RHS_RIDGE_REPAIR"]
    )
    if "parametric_pmedian_solver" in cleaned.columns:
        solver = cleaned["parametric_pmedian_solver"].astype(str).str.strip().str.upper()
        pmedian_ok = cleaned["parametric_pmedian_solver"].isna() | solver.eq("GREEDY")
        selector_ok &= (~pmedian_rows | pmedian_ok)

    selected = cleaned[exact | (allowed & callback & ratio & feature & hypers & universal & selector_ok)].copy()
    selected["weighting_mechanism"] = selected["method_key"].map(WEIGHTING_MECHANISM)
    selected["prediction_mechanism"] = selected["method_key"].map(PREDICTION_MECHANISM)
    selected["cross_method"] = selected["method_key"].map(WEIGHTING_CROSS_LABELS)
    selected.loc[selected["method_key"].eq("STANDARD_BD"), "cross_method"] = "Standard BD"
    return selected


# =============================================================================
# Summary tables
# =============================================================================
def _stats(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {k: np.nan for k in ["min", "max", "mean", "median", "q1", "q3"]}
    return {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()), "median": float(values.median()),
        "q1": float(values.quantile(0.25)), "q3": float(values.quantile(0.75)),
    }


def build_method_summary(df: pd.DataFrame, factor: str | None = None) -> pd.DataFrame:
    grouping = ["analysis_group", "baseline_stratum", "method"] + ([factor] if factor else [])
    rows: list[dict[str, object]] = []
    working = df[df["method_key"].ne("STANDARD_BD")].copy()
    for keys, group in working.groupby(grouping, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        obj = _stats(group["relative_objective_difference_pct"])
        tim = _stats(group["relative_runtime"])
        row.update({
            "records": int(len(group)),
            "objective_values_available": int(group["objective_available"].sum()),
            "time_limit_rate_pct": 100.0 * float(group["is_time_limit"].mean()) if len(group) else np.nan,
            "gap_min": obj["min"], "gap_max": obj["max"], "gap_mean": obj["mean"], "gap_median": obj["median"],
            "gap_q1": obj["q1"], "gap_q3": obj["q3"],
            "time_min": tim["min"], "time_max": tim["max"], "time_mean": tim["mean"], "time_median": tim["median"],
            "time_q1": tim["q1"], "time_q3": tim["q3"],
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    rank = {m: i for i, m in enumerate(NONBASELINE_METHOD_ORDER)}
    result["_rank"] = result["method"].map(rank).fillna(999)
    sort_cols = ["analysis_group", "baseline_stratum", "_rank"] + ([factor] if factor else [])
    return result.sort_values(sort_cols).drop(columns="_rank").reset_index(drop=True)


def build_dimension_summary(df: pd.DataFrame, dimension: str, factor: str | None = None) -> pd.DataFrame:
    grouping = ["analysis_group", "baseline_stratum", dimension, "method"] + ([factor] if factor else [])
    rows: list[dict[str, object]] = []
    working = df[df["method_key"].ne("STANDARD_BD")].copy()
    for keys, group in working.groupby(grouping, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        obj = _stats(group["relative_objective_difference_pct"])
        tim = _stats(group["relative_runtime"])
        row.update({
            "records": int(len(group)),
            "time_limit_rate_pct": 100.0 * float(group["is_time_limit"].mean()) if len(group) else np.nan,
            "gap_min": obj["min"], "gap_max": obj["max"], "gap_mean": obj["mean"], "gap_median": obj["median"],
            "time_min": tim["min"], "time_max": tim["max"], "time_mean": tim["mean"], "time_median": tim["median"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_coverage(df: pd.DataFrame) -> pd.DataFrame:
    grouping = ["analysis_group", "instance", "num_subproblems", "method"]
    rows = []
    for keys, group in df.groupby(grouping, dropna=False, sort=False):
        analysis_group, instance, nsp, method = keys
        gap_values = pd.to_numeric(group["relative_objective_difference_pct"], errors="coerce").dropna()
        time_values = pd.to_numeric(group["relative_runtime"], errors="coerce").dropna()
        rows.append({
            "analysis_group": analysis_group,
            "instance": instance,
            "num_subproblems": nsp,
            "method": method,
            "records": int(len(group)),
            "time_limit_rate_pct": 100.0 * float(group["is_time_limit"].mean()),
            "gap_median": float(gap_values.median()) if len(gap_values) else np.nan,
            "gap_mean": float(gap_values.mean()) if len(gap_values) else np.nan,
            "time_median": float(time_values.median()) if len(time_values) else np.nan,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_instance_order"] = result["instance"].map(INSTANCE_ORDER).fillna(999)
        result = result.sort_values(["analysis_group", "_instance_order", "num_subproblems", "method"]).drop(columns="_instance_order")
    return result


def build_weighting_cross_summary(df: pd.DataFrame, factor: str | None = None) -> pd.DataFrame:
    grouping = [
        "analysis_group", "baseline_stratum", "weighting_mechanism",
        "prediction_mechanism", "cross_method",
    ] + ([factor] if factor else [])
    rows: list[dict[str, object]] = []
    working = df[df["method_key"].isin(WEIGHTING_CROSS_METHOD_KEYS)].copy()
    for keys, group in working.groupby(grouping, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        obj = _stats(group["relative_objective_difference_pct"])
        tim = _stats(group["relative_runtime"])
        row.update({
            "records": int(len(group)),
            "objective_values_available": int(group["objective_available"].sum()),
            "time_limit_rate_pct": 100.0 * float(group["is_time_limit"].mean()) if len(group) else np.nan,
            "gap_min": obj["min"], "gap_max": obj["max"], "gap_mean": obj["mean"],
            "gap_median": obj["median"], "gap_q1": obj["q1"], "gap_q3": obj["q3"],
            "time_min": tim["min"], "time_max": tim["max"], "time_mean": tim["mean"],
            "time_median": tim["median"], "time_q1": tim["q1"], "time_q3": tim["q3"],
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        weight_rank = {"Tree": 0, "Cluster": 1, "P-Median": 2}
        prediction_rank = {"Nonparametric": 0, "RHS Ridge": 1}
        result["_w"] = result["weighting_mechanism"].map(weight_rank).fillna(99)
        result["_p"] = result["prediction_mechanism"].map(prediction_rank).fillna(99)
        sort_cols = ["analysis_group", "baseline_stratum", "_w", "_p"] + ([factor] if factor else [])
        result = result.sort_values(sort_cols).drop(columns=["_w", "_p"]).reset_index(drop=True)
    return result


def build_weighting_cross_coverage(df: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "analysis_group", "instance", "num_subproblems",
        "weighting_mechanism", "prediction_mechanism", "cross_method",
    ]
    rows: list[dict[str, object]] = []
    working = df[df["method_key"].isin(WEIGHTING_CROSS_METHOD_KEYS)].copy()
    for keys, group in working.groupby(grouping, dropna=False, sort=False):
        row = dict(zip(grouping, keys))
        gaps = pd.to_numeric(group["relative_objective_difference_pct"], errors="coerce").dropna()
        times = pd.to_numeric(group["relative_runtime"], errors="coerce").dropna()
        row.update({
            "records": int(len(group)),
            "time_limit_rate_pct": 100.0 * float(group["is_time_limit"].mean()) if len(group) else np.nan,
            "gap_median": float(gaps.median()) if len(gaps) else np.nan,
            "gap_mean": float(gaps.mean()) if len(gaps) else np.nan,
            "time_median": float(times.median()) if len(times) else np.nan,
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_instance_order"] = result["instance"].map(INSTANCE_ORDER).fillna(999)
        result = result.sort_values(
            ["analysis_group", "_instance_order", "num_subproblems",
             "weighting_mechanism", "prediction_mechanism"]
        ).drop(columns="_instance_order")
    return result



def build_weighting_cross_completeness(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Audit the full 3x2 primary-design grid used for the weighting comparison."""
    expected_rows: list[dict[str, object]] = []
    group_instances = {
        SCFLP_GROUP: SCFLP_INSTANCES,
        f"SCMND_{SCMND_MANUSCRIPT_FORMULATION}": SCMND_INSTANCES,
        f"SCMND_{SCMND_ALTERNATIVE_FORMULATION}": SCMND_INSTANCES,
    }
    for analysis_group, instances in group_instances.items():
        for instance in instances:
            for nsp in NUM_SUBPROBLEMS:
                for ratio in EXACT_RATIOS:
                    for method_key in WEIGHTING_CROSS_METHOD_KEYS:
                        expected_rows.append({
                            "analysis_group": analysis_group,
                            "deterministic_instance": instance,
                            "num_subproblems": nsp,
                            "selected_subproblems_ratio": ratio,
                            "method_key": method_key,
                        })
    expected = pd.DataFrame(expected_rows)

    selected = select_weighting_cross(cleaned)
    selected = selected[selected["method_key"].isin(WEIGHTING_CROSS_METHOD_KEYS)].copy()
    keys = [
        "analysis_group", "deterministic_instance", "num_subproblems",
        "selected_subproblems_ratio", "method_key",
    ]
    observed = selected.groupby(keys, dropna=False, sort=False).agg(
        observed_records=("method_key", "size"),
        valid_records=("valid_record", "sum"),
    ).reset_index()

    result = expected.merge(observed, on=keys, how="left")
    result["observed_records"] = result["observed_records"].fillna(0).astype(int)
    result["valid_records"] = result["valid_records"].fillna(0).astype(int)
    result["status"] = np.select(
        [
            result["observed_records"].eq(0),
            result["valid_records"].eq(0),
            result["observed_records"].gt(1),
        ],
        ["MISSING", "INVALID", "MULTIPLE"],
        default="OK",
    )
    result["weighting_mechanism"] = result["method_key"].map(WEIGHTING_MECHANISM)
    result["prediction_mechanism"] = result["method_key"].map(PREDICTION_MECHANISM)
    return result


# =============================================================================
# Plotting helpers
# =============================================================================
def _dimension_values(subset: pd.DataFrame, dimension: str) -> list[float]:
    return sorted(pd.to_numeric(subset[dimension], errors="coerce").dropna().unique().tolist())


def _format_dimension_ticks(values: list[float], dimension: str) -> list[str]:
    if dimension == "selected_subproblems_ratio":
        return [f"{100*x:.0f}%" for x in values]
    if dimension in {"n_trials", "m_trials", "n_walks", "walk_length", "feature_dim", "window", "num_subproblems"}:
        return [str(int(x)) if float(x).is_integer() else f"{x:g}" for x in values]
    return [f"{x:g}" for x in values]


def _dimension_label(dimension: str) -> str:
    labels = {
        "selected_subproblems_ratio": r"$|\mathcal{M}_e|/|\mathcal{M}|$",
        "num_subproblems": r"$|\mathcal{M}|$",
        "n_trials": r"$n_{\mathrm{trial}}$",
        "m_trials": r"$m_{\mathrm{trial}}$",
        "n_walks": "Number of random walks",
        "walk_length": "Random-walk length",
        "feature_dim": "Feature dimension",
        "window": "Word2Vec window",
    }
    return labels.get(dimension, dimension)


def _dimension_title(dimension: str) -> str:
    return _dimension_label(dimension)


def _metric_axis_label(metric: str) -> str:
    return {
        "relative_objective_difference_pct": r"$\Delta_a$ (%)",
        "relative_runtime": r"$\tau_a$",
        "feature_generation_time_ratio_pct": "Feature-generation time share (%)",
        "warm_up_time_ratio_pct": "Warm-up time share (%)",
        "solution_time_sec": "Solution time (sec)",
        "total_cuts": "Total cuts",
        "ml_cut_share_pct": "ML-generated cut share (%)",
        **PARAMETRIC_DIAGNOSTICS,
    }.get(metric, metric)


def _adaptive_decimal_formatter(values: Iterable[float], *, percent: bool = False) -> FuncFormatter:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        digits = 1
    else:
        span = float(np.max(arr) - np.min(arr))
        max_abs = float(np.max(np.abs(arr)))
        if span < 0.1 and max_abs < 1:
            digits = 3
        elif span < 1 and max_abs < 10:
            digits = 2
        elif span < 10:
            digits = 1
        else:
            digits = 0 if percent else 1
    return FuncFormatter(lambda x, pos: f"{x:.{digits}f}")


def _format_y_axis(ax: plt.Axes, metric: str, values: Iterable[float]) -> None:
    ax.tick_params(axis="y", which="both", labelleft=True, left=True)
    ax.yaxis.set_visible(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
    if metric in {"solution_time_sec", "total_cuts", "parametric_max_condition_number"}:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    elif metric.endswith("_pct") or metric == "relative_objective_difference_pct":
        ax.yaxis.set_major_formatter(_adaptive_decimal_formatter(values, percent=True))
    elif metric == "relative_runtime":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
    else:
        ax.yaxis.set_major_formatter(_adaptive_decimal_formatter(values))
    ax.grid(axis="y", which="major", alpha=0.22, linewidth=0.65)


def _legend_handle(category: str, kind: str) -> Line2D | Patch:
    color = METHOD_COLORS.get(category, "#333333")
    if kind == "violin":
        return Patch(facecolor=color, edgecolor=color, alpha=0.25, label=category)
    return Line2D([], [], color=color, linewidth=1.8, label=category)


def _draw_grouped_distribution(
    ax: plt.Axes,
    subset: pd.DataFrame,
    *,
    dimension: str,
    metric: str,
    categories: list[str],
    category_column: str,
    reference: float | None,
    y_label: str,
    kind: str,
    show_num_obs: bool,
) -> list[Line2D | Patch]:
    values = _dimension_values(subset, dimension)
    centers = np.arange(len(values), dtype=float)
    active_categories = [c for c in categories if subset[category_column].eq(c).any()]
    ncat = max(len(active_categories), 1)
    offsets = np.linspace(-0.33, 0.33, ncat) if ncat > 1 else np.array([0.0])
    width = min(0.16, 0.66 / ncat)
    handles: list[Line2D | Patch] = []
    plotted_values: list[float] = []

    for cat_idx, category in enumerate(active_categories):
        arrays: list[np.ndarray] = []
        positions: list[float] = []
        counts: list[tuple[float, int]] = []
        for center, dim_value in zip(centers, values):
            vals = pd.to_numeric(
                subset[
                    subset[category_column].eq(category)
                    & np.isclose(pd.to_numeric(subset[dimension], errors="coerce"), float(dim_value), atol=1e-12, rtol=0)
                ][metric], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if len(vals):
                pos = center + offsets[cat_idx]
                arrays.append(vals)
                positions.append(pos)
                counts.append((pos, len(vals)))
                plotted_values.extend(vals.tolist())
        if not arrays:
            continue
        color = METHOD_COLORS.get(category, "#333333")
        if kind == "box":
            ax.boxplot(
                arrays, positions=positions, widths=width, manage_ticks=False,
                showfliers=False, patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.6},
                boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.45},
                whiskerprops={"color": color, "linewidth": 1.2},
                capprops={"color": color, "linewidth": 1.2},
            )
        elif kind == "violin":
            violin = ax.violinplot(arrays, positions=positions, widths=width * 1.45,
                                   showmeans=False, showmedians=False, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.24)
                body.set_linewidth(1.2)
            for pos, vals in zip(positions, arrays):
                q1, median, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
                ax.vlines(pos, q1, q3, color=color, linewidth=2.0)
                ax.hlines([q1, q3], pos - 0.25 * width, pos + 0.25 * width, color=color, linewidth=1.0)
                ax.hlines(median, pos - 0.34 * width, pos + 0.34 * width, color="black", linewidth=1.6)
        else:
            raise ValueError(f"Unsupported distribution kind: {kind}")
        handles.append(_legend_handle(category, kind))
        if show_num_obs:
            for pos, count in counts:
                ax.text(pos, -0.18, f"n={count}", ha="center", va="top", fontsize=6.3,
                        transform=ax.get_xaxis_transform(), clip_on=False)

    if reference is not None:
        ax.axhline(reference, linestyle=":" if reference == 0 else "--", color="#4c78a8", linewidth=1.15, zorder=0)
    ax.set_xticks(centers)
    ax.set_xticklabels(_format_dimension_ticks(values, dimension))
    ax.set_xlabel(_dimension_label(dimension))
    ax.set_ylabel(y_label)
    _format_y_axis(ax, metric, plotted_values)
    return handles


def _panel_diagnostic(values: pd.Series | np.ndarray, metric: str) -> dict[str, object]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    result: dict[str, object] = {"metric": metric, "n": int(len(arr)), "flagged": False, "zoom_lower": np.nan, "zoom_upper": np.nan}
    if len(arr) < 5:
        return result
    ymin, ymax = float(np.min(arr)), float(np.max(arr))
    full_range = ymax - ymin
    if not np.isfinite(full_range) or full_range <= 0:
        return result
    q05, q25, _, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
    iqr = float(q75 - q25)
    central_fraction = float((q95 - q05) / full_range)
    denom = max(iqr, np.finfo(float).eps)
    upper_tail = float((ymax - q75) / denom)
    lower_tail = float((q25 - ymin) / denom)
    flagged = central_fraction < SQUASH_CENTRAL_RANGE_THRESHOLD and max(upper_tail, lower_tail) > SQUASH_TAIL_IQR_THRESHOLD
    result.update({"central_range_fraction": central_fraction, "upper_tail_iqr": upper_tail, "lower_tail_iqr": lower_tail, "flagged": bool(flagged)})
    if not flagged:
        return result
    lower = max(ymin, float(q25 - 1.75 * iqr)) if iqr > 0 else float(q05)
    upper = min(ymax, float(q75 + 1.75 * iqr)) if iqr > 0 else float(q95)
    reference = 0.0 if metric == "relative_objective_difference_pct" else 1.0
    lower, upper = min(lower, reference), max(upper, reference)
    span = max(upper - lower, 0.1)
    result["zoom_lower"] = lower - 0.08 * span
    result["zoom_upper"] = upper + 0.08 * span
    return result


def _annotate_clipped_observations(ax: plt.Axes, original: pd.DataFrame, *, dimension: str, metric: str,
                                    categories: list[str], category_column: str, lower: float, upper: float) -> None:
    values = _dimension_values(original, dimension)
    centers = np.arange(len(values), dtype=float)
    active = [c for c in categories if original[category_column].eq(c).any()]
    offsets = np.linspace(-0.33, 0.33, max(len(active), 1)) if len(active) > 1 else np.array([0.0])
    span = upper - lower
    for ci, category in enumerate(active):
        for center, dim_value in zip(centers, values):
            vals = pd.to_numeric(original[
                original[category_column].eq(category)
                & np.isclose(pd.to_numeric(original[dimension], errors="coerce"), float(dim_value), atol=1e-12, rtol=0)
            ][metric], errors="coerce").dropna()
            x = center + offsets[ci]
            if (vals > upper).any():
                ax.text(x, upper - 0.025 * span, rf"$\uparrow${int((vals > upper).sum())}", ha="center", va="top", fontsize=7.0)
            if (vals < lower).any():
                ax.text(x, lower + 0.025 * span, rf"$\downarrow${int((vals < lower).sum())}", ha="center", va="bottom", fontsize=7.0)


def _save_figure(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    if EXPORT_PNG:
        fig.savefig(destination.with_suffix(".png"))
    plt.close(fig)


def _figure_source_path(stem: str) -> Path:
    return FIGURE_DATA_DIR / f"{stem}.csv"


def _main_stem(group: str, stratum: str, dimension: str) -> str:
    legacy = f"main__{_group_legacy_problem_token(group)}__{_safe_slug(stratum)}__{dimension}_v5"
    return legacy + _group_file_suffix(group)


def _appendix_stem(prefix: str, group: str, stratum: str, dimension: str) -> str:
    legacy = f"{prefix}__{_group_legacy_problem_token(group)}__{_safe_slug(stratum)}__{dimension}_v5"
    return legacy + _group_file_suffix(group)


def _violin_stem(stem: str) -> str:
    suffix = _group_file_suffix_from_stem(stem)
    core = stem[:-len(suffix)] if suffix else stem
    if core.endswith("_zoom_v5"):
        core = core[:-len("_zoom_v5")] + "_zoom_violin_v5"
    elif core.endswith("_v5"):
        core = core[:-len("_v5")] + "_violin_v5"
    else:
        core += "_violin"
    return core + suffix


def _group_file_suffix_from_stem(stem: str) -> str:
    for suffix in ["_current_soft_capacity", "_unspecified_scmnd_formulation"]:
        if stem.endswith(suffix):
            return suffix
    return ""


def _plus_num_obs_stem(stem: str) -> str:
    suffix = _group_file_suffix_from_stem(stem)
    core = stem[:-len(suffix)] if suffix else stem
    return core + "_plus_num_obs" + suffix


def _render_main_composite(subset: pd.DataFrame, *, dimension: str, destination: Path,
                           kind: str, show_num_obs: bool,
                           objective_limits: tuple[float, float] | None = None,
                           runtime_limits: tuple[float, float] | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    obj = subset.copy(); tim = subset.copy()
    if objective_limits is not None:
        lo, hi = objective_limits
        obj.loc[~obj["relative_objective_difference_pct"].between(lo, hi), "relative_objective_difference_pct"] = np.nan
    if runtime_limits is not None:
        lo, hi = runtime_limits
        tim.loc[~tim["relative_runtime"].between(lo, hi), "relative_runtime"] = np.nan
    handles = _draw_grouped_distribution(
        axes[0], obj.dropna(subset=["relative_objective_difference_pct"]),
        dimension=dimension, metric="relative_objective_difference_pct",
        categories=NONBASELINE_METHOD_ORDER, category_column="method", reference=0.0,
        y_label=r"$\Delta_a$ (%)", kind=kind, show_num_obs=show_num_obs,
    )
    _draw_grouped_distribution(
        axes[1], tim.dropna(subset=["relative_runtime"]),
        dimension=dimension, metric="relative_runtime",
        categories=NONBASELINE_METHOD_ORDER, category_column="method", reference=1.0,
        y_label=r"$\tau_a$", kind=kind, show_num_obs=show_num_obs,
    )
    title_dim = _dimension_title(dimension)
    axes[0].set_title(rf"$\Delta_a$ (%) by {title_dim}" + (" (zoomed)" if objective_limits else ""))
    axes[1].set_title(rf"$\tau_a$ by {title_dim}" + (" (zoomed)" if runtime_limits else ""))
    if objective_limits:
        axes[0].set_ylim(*objective_limits)
        _annotate_clipped_observations(axes[0], subset, dimension=dimension, metric="relative_objective_difference_pct",
                                       categories=NONBASELINE_METHOD_ORDER, category_column="method", lower=objective_limits[0], upper=objective_limits[1])
    if runtime_limits:
        axes[1].set_ylim(*runtime_limits)
        _annotate_clipped_observations(axes[1], subset, dimension=dimension, metric="relative_runtime",
                                       categories=NONBASELINE_METHOD_ORDER, category_column="method", lower=runtime_limits[0], upper=runtime_limits[1])
    if handles:
        ref = Line2D([], [], linestyle="--", color="#4c78a8", label="Standard BD reference")
        fig.legend(handles=handles + [ref], loc="lower center", ncol=min(4, len(handles) + 1), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.18 if show_num_obs else 0.08, 1, 1))
    _save_figure(fig, destination)


def make_main_figures(df: pd.DataFrame, *, group: str, stratum: str, dimension: str) -> list[str]:
    subset = df[
        df["analysis_group"].eq(group)
        & df["baseline_stratum"].eq(stratum)
        & df["method"].isin(NONBASELINE_METHOD_ORDER)
    ].copy()
    if subset.empty:
        return []
    base = _main_stem(group, stratum, dimension)
    source_cols = [
        "analysis_group", "instance", "num_subproblems", "baseline_stratum", "method", "method_key",
        "selected_subproblems_ratio", "stopping_reason", "is_time_limit",
        "relative_objective_difference_pct", "relative_runtime", "objective_available",
    ]
    subset[[c for c in source_cols if c in subset.columns]].to_csv(_figure_source_path(base), index=False)

    stems = [base, _plus_num_obs_stem(base), _violin_stem(base), _plus_num_obs_stem(_violin_stem(base))]
    _render_main_composite(subset, dimension=dimension, destination=FIGURE_DIR / f"{stems[0]}.pdf", kind="box", show_num_obs=False)
    _render_main_composite(subset, dimension=dimension, destination=FIGURE_DIR / f"{stems[1]}.pdf", kind="box", show_num_obs=True)
    _render_main_composite(subset, dimension=dimension, destination=FIGURE_DIR / f"{stems[2]}.pdf", kind="violin", show_num_obs=False)
    _render_main_composite(subset, dimension=dimension, destination=FIGURE_DIR / f"{stems[3]}.pdf", kind="violin", show_num_obs=True)

    obj_diag = _panel_diagnostic(subset["relative_objective_difference_pct"], "relative_objective_difference_pct")
    tim_diag = _panel_diagnostic(subset["relative_runtime"], "relative_runtime")
    for diag in [obj_diag, tim_diag]:
        ZOOM_DIAGNOSTICS.append({"stem": base, "analysis_group": group, "baseline_stratum": stratum, "dimension": dimension, **diag})
    if obj_diag["flagged"] or tim_diag["flagged"]:
        suffix = _group_file_suffix(group)
        core = base[:-len(suffix)] if suffix else base
        zoom = core.replace("_v5", "_zoom_v5") + suffix
        obj_limits = (float(obj_diag["zoom_lower"]), float(obj_diag["zoom_upper"])) if obj_diag["flagged"] else None
        tim_limits = (float(tim_diag["zoom_lower"]), float(tim_diag["zoom_upper"])) if tim_diag["flagged"] else None
        for zstem, kind, counts in [
            (zoom, "box", False), (_plus_num_obs_stem(zoom), "box", True),
            (_violin_stem(zoom), "violin", False), (_plus_num_obs_stem(_violin_stem(zoom)), "violin", True),
        ]:
            _render_main_composite(subset, dimension=dimension, destination=FIGURE_DIR / f"{zstem}.pdf", kind=kind,
                                   show_num_obs=counts, objective_limits=obj_limits, runtime_limits=tim_limits)
            stems.append(zstem)
        ZOOM_FIGURES[base] = {"zoom_stem": zoom, "analysis_group": group, "baseline_stratum": stratum,
                              "dimension": dimension, "objective_limits": obj_limits, "runtime_limits": tim_limits}
    return stems


def _variant_figure(subset: pd.DataFrame, *, stem: str, methods: list[str], variants: list[str],
                    variant_column: str, dimension: str, kind: str, show_num_obs: bool) -> None:
    active_methods = [m for m in methods if subset["method"].eq(m).any()]
    if not active_methods:
        return
    cols = len(active_methods)
    fig, axes = plt.subplots(2, cols, figsize=(max(9.0, 3.25 * cols), 6.8), squeeze=False)
    legend_handles: list[Line2D | Patch] = []
    for ci, method in enumerate(active_methods):
        msub = subset[subset["method"].eq(method)]
        h = _draw_grouped_distribution(
            axes[0, ci], msub.dropna(subset=["relative_objective_difference_pct"]),
            dimension=dimension, metric="relative_objective_difference_pct", categories=variants,
            category_column=variant_column, reference=0.0, y_label=r"$\Delta_a$ (%)" if ci == 0 else "",
            kind=kind, show_num_obs=show_num_obs,
        )
        _draw_grouped_distribution(
            axes[1, ci], msub.dropna(subset=["relative_runtime"]),
            dimension=dimension, metric="relative_runtime", categories=variants,
            category_column=variant_column, reference=1.0, y_label=r"$\tau_a$" if ci == 0 else "",
            kind=kind, show_num_obs=show_num_obs,
        )
        axes[0, ci].set_title(rf"{method}: $\Delta_a$ (%) by {_dimension_title(dimension)}")
        axes[1, ci].set_title(rf"{method}: $\tau_a$ by {_dimension_title(dimension)}")
        if h and not legend_handles:
            legend_handles = h
    if legend_handles:
        ref = Line2D([], [], linestyle="--", color="#4c78a8", label="Standard BD reference")
        fig.legend(handles=legend_handles + [ref], loc="lower center", ncol=min(4, len(legend_handles) + 1), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.17 if show_num_obs else 0.07, 1, 1))
    _save_figure(fig, FIGURE_DIR / f"{stem}.pdf")


def make_variant_figures(df: pd.DataFrame, *, group: str, stratum: str, dimension: str,
                         prefix: str, methods: list[str], variants: list[str], variant_column: str) -> list[str]:
    subset = df[
        df["analysis_group"].eq(group)
        & df["baseline_stratum"].eq(stratum)
        & df["method"].isin(methods)
    ].copy()
    if subset.empty:
        return []
    base = _appendix_stem(prefix, group, stratum, dimension)
    subset.to_csv(_figure_source_path(base), index=False)
    stems = [base, _plus_num_obs_stem(base), _violin_stem(base), _plus_num_obs_stem(_violin_stem(base))]
    for stem, kind, counts in [
        (stems[0], "box", False), (stems[1], "box", True),
        (stems[2], "violin", False), (stems[3], "violin", True),
    ]:
        _variant_figure(subset, stem=stem, methods=methods, variants=variants, variant_column=variant_column,
                        dimension=dimension, kind=kind, show_num_obs=counts)
    return stems


def _draw_overall_distribution(ax: plt.Axes, subset: pd.DataFrame, *, metric: str, methods: list[str], kind: str, show_num_obs: bool) -> list[Line2D | Patch]:
    active = [m for m in methods if subset["method"].eq(m).any()]
    arrays, positions, handles, plotted = [], [], [], []
    for i, method in enumerate(active):
        vals = pd.to_numeric(subset[subset["method"].eq(method)][metric], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(vals):
            continue
        arrays.append(vals); positions.append(float(i)); plotted.extend(vals.tolist())
        color = METHOD_COLORS.get(method, "#333333")
        if kind == "box":
            ax.boxplot([vals], positions=[i], widths=0.45, manage_ticks=False, showfliers=False, patch_artist=True,
                       medianprops={"color": "black", "linewidth": 1.6},
                       boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.45},
                       whiskerprops={"color": color}, capprops={"color": color})
        else:
            vp = ax.violinplot([vals], positions=[i], widths=0.6, showmeans=False, showmedians=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.24)
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            ax.vlines(i, q1, q3, color=color, linewidth=2.0); ax.hlines(med, i - 0.12, i + 0.12, color="black", linewidth=1.6)
        handles.append(_legend_handle(method, kind))
        if show_num_obs:
            ax.text(i, -0.14, f"n={len(vals)}", ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xticks(range(len(active))); ax.set_xticklabels(active, rotation=20, ha="right")
    ax.set_ylabel(_metric_axis_label(metric)); _format_y_axis(ax, metric, plotted)
    return handles


def _extra_metric_stem(metric_stem: str, group: str, stratum: str, dimension: str | None, kind: str, counts: bool) -> str:
    # Match the established v6 filename exactly for SCFLP and manuscript SCMND:
    # <metric>__<problem>__<status>__<grouping>__boxplot_v5[_plus_num_obs].
    dimension_token = "overall" if dimension is None else dimension
    legacy = f"{metric_stem}__{_group_legacy_problem_token(group)}__{_safe_slug(stratum)}__{dimension_token}__{kind}plot_v5"
    if counts:
        legacy += "_plus_num_obs"
    return legacy + _group_file_suffix(group)


def make_extra_metric_figure(df: pd.DataFrame, *, metric: str, metric_stem: str, methods: list[str],
                             group: str, stratum: str, dimension: str | None, kind: str, counts: bool) -> str | None:
    subset = df[
        df["analysis_group"].eq(group)
        & df["baseline_stratum"].eq(stratum)
        & df["method"].isin(methods)
    ].copy()
    if metric not in subset.columns or pd.to_numeric(subset[metric], errors="coerce").dropna().empty:
        return None
    stem = _extra_metric_stem(metric_stem, group, stratum, dimension, kind, counts)
    source_cols = ["analysis_group", "instance", "num_subproblems", "method", "selected_subproblems_ratio", metric]
    # Retain v6's source-data stem for the canonical boxplot, while also saving
    # figure-specific source data for other companions.
    if kind == "box" and not counts:
        grouping = "overall" if dimension is None else dimension
        legacy_source = f"{metric_stem}__{_group_legacy_problem_token(group)}__{_safe_slug(stratum)}__{grouping}_v5" + _group_file_suffix(group)
        subset[[c for c in source_cols if c in subset.columns]].to_csv(_figure_source_path(legacy_source), index=False)
    else:
        subset[[c for c in source_cols if c in subset.columns]].to_csv(_figure_source_path(stem), index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    if dimension is None:
        handles = _draw_overall_distribution(ax, subset, metric=metric, methods=methods, kind=kind, show_num_obs=counts)
        title = f"{_metric_axis_label(metric)} by method"
    else:
        handles = _draw_grouped_distribution(ax, subset, dimension=dimension, metric=metric, categories=methods,
                                             category_column="method", reference=None, y_label=_metric_axis_label(metric),
                                             kind=kind, show_num_obs=counts)
        title = f"{_metric_axis_label(metric)} by {_dimension_title(dimension)}"
    ax.set_title(title)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.18 if counts else 0.08, 1, 1))
    _save_figure(fig, FIGURE_DIR / f"{stem}.pdf")
    return stem

# =============================================================================
# Hyperparameter sensitivities
# =============================================================================
def _sensitivity_values(cleaned: pd.DataFrame, parameter: str) -> list[float]:
    observed = pd.to_numeric(cleaned.get(parameter), errors="coerce").dropna().unique().tolist()
    declared = [float(v) for v in EXPECTED_HYPERPARAMETER_VALUES.get(parameter, [])]
    return sorted(set(float(v) for v in observed) | set(declared))


def select_sensitivity_data(cleaned: pd.DataFrame, parameter: str, callback: str) -> pd.DataFrame:
    method_mask = cleaned["method_key"].isin([k for k in PLOT_METHOD_KEYS if k != "RANDOM_SAMPLING"])
    callback_mask = cleaned["ml_callback_type"].eq(callback)
    ratio_mask = cleaned["selected_subproblems_ratio"].isin(EXACT_RATIOS)
    # Walk/embedding parameters have no meaning for covariates-only inputs.
    if parameter in {"n_walks", "walk_length", "feature_dim", "window"}:
        feature_mask = cleaned["use_covariates"].isin(["features_only", "features_plus_covariates"])
    else:
        feature_mask = cleaned["use_covariates"].eq(PRIMARY_FEATURE_SPEC)
    other_refs = _reference_hyperparameter_mask(cleaned, allow_missing=False, exclude=parameter)
    return cleaned[method_mask & callback_mask & ratio_mask & feature_mask & other_refs].copy()


def make_sensitivity_figures(df_with_refs: pd.DataFrame, *, parameter: str, callback: str, group: str, stratum: str) -> list[str]:
    subset = df_with_refs[
        df_with_refs["analysis_group"].eq(group)
        & df_with_refs["baseline_stratum"].eq(stratum)
        & df_with_refs["method"].isin(LABD_METHODS)
    ].copy()
    if subset.empty or pd.to_numeric(subset[parameter], errors="coerce").nunique(dropna=True) <= 1:
        return []
    callback_slug = _safe_slug(callback)
    base = f"sensitivity_{parameter}__{_group_legacy_problem_token(group)}__{_safe_slug(stratum)}__{callback_slug}_v7" + _group_file_suffix(group)
    subset.to_csv(_figure_source_path(base), index=False)
    stems = [base, _plus_num_obs_stem(base), _violin_stem(base), _plus_num_obs_stem(_violin_stem(base))]
    for stem, kind, counts in [
        (stems[0], "box", False), (stems[1], "box", True),
        (stems[2], "violin", False), (stems[3], "violin", True),
    ]:
        _render_main_composite(subset, dimension=parameter, destination=FIGURE_DIR / f"{stem}.pdf", kind=kind, show_num_obs=counts)
    return stems

# =============================================================================
# Parametric diagnostics
# =============================================================================
def generate_parametric_diagnostics(primary: pd.DataFrame, groups: list[str]) -> list[str]:
    stems: list[str] = []
    if not PARAMETRIC_METHODS:
        return stems
    for metric, label in PARAMETRIC_DIAGNOSTICS.items():
        if metric not in primary.columns or pd.to_numeric(primary[metric], errors="coerce").dropna().empty:
            continue
        for group in groups:
            for stratum in STRATUM_ORDER:
                for dimension in [None, "num_subproblems", "selected_subproblems_ratio"]:
                    for kind in ["box", "violin"]:
                        for counts in [False, True]:
                            stem = make_extra_metric_figure(
                                primary, metric=metric, metric_stem=_safe_slug(metric), methods=PARAMETRIC_METHODS,
                                group=group, stratum=stratum, dimension=dimension, kind=kind, counts=counts,
                            )
                            if stem:
                                stems.append(stem)
        table = metric_summary_table(primary, metric=metric, methods=PARAMETRIC_METHODS, factor=None)
        table.to_csv(TABLE_DIR / f"parametric_{_safe_slug(metric)}_summary_v7.csv", index=False)
    return stems


def metric_summary_table(df: pd.DataFrame, *, metric: str, methods: list[str], factor: str | None) -> pd.DataFrame:
    grouping = ["analysis_group", "baseline_stratum", "method"] + ([factor] if factor else [])
    rows = []
    working = df[df["method"].isin(methods)].copy()
    for keys, group in working.groupby(grouping, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        row.update({
            "records": int(len(group)), "available_values": int(len(values)),
            "minimum": float(values.min()) if len(values) else np.nan,
            "first_quartile": float(values.quantile(0.25)) if len(values) else np.nan,
            "median": float(values.median()) if len(values) else np.nan,
            "mean": float(values.mean()) if len(values) else np.nan,
            "third_quartile": float(values.quantile(0.75)) if len(values) else np.nan,
            "maximum": float(values.max()) if len(values) else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)

# =============================================================================
# Output organization and documentation
# =============================================================================
def _infer_group_from_stem(stem: str) -> str:
    if "__scflp__" in stem.lower():
        return SCFLP_GROUP
    if stem.endswith("_current_soft_capacity"):
        return f"SCMND_{SCMND_ALTERNATIVE_FORMULATION}"
    if stem.endswith("_unspecified_scmnd_formulation"):
        return UNSPECIFIED_SCMND_GROUP
    if "__scmnd__" in stem.lower():
        return f"SCMND_{SCMND_MANUSCRIPT_FORMULATION}"
    return "Other"


def _infer_callback_from_stem(stem: str) -> str:
    lower = stem.lower()
    if _safe_slug(SUPPLEMENTARY_CALLBACK) in lower:
        return SUPPLEMENTARY_CALLBACK
    # Legacy main/feature/extra figures use the primary callback.
    return PRIMARY_CALLBACK


def organize_output_files() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for source in sorted(FIGURE_DIR.glob("*")):
        if source.suffix.lower() not in {".pdf", ".png"}:
            continue
        group = _infer_group_from_stem(source.stem)
        callback = _infer_callback_from_stem(source.stem)
        destination_dir = ORGANIZED_FIGURE_DIR / group / callback
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        rows.append({
            "analysis_group": group,
            "callback": callback,
            "file_name": source.name,
            "flat_path": str(source.relative_to(OUTPUT_DIR)),
            "organized_path": str(destination.relative_to(OUTPUT_DIR)),
        })
    source_root = FIGURE_DATA_DIR / "by_problem"
    for source in sorted(FIGURE_DATA_DIR.glob("*.csv")):
        group = _infer_group_from_stem(source.stem)
        callback = _infer_callback_from_stem(source.stem)
        destination_dir = source_root / group / callback
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / source.name)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(TABLE_DIR / "output_file_manifest_v7.csv", index=False)
    return manifest


def write_diagnostics(cleaned: pd.DataFrame, removed: pd.DataFrame, duplicate_report: pd.DataFrame,
                      audit: dict[str, pd.DataFrame], sheet: str, missing_ref_after_merge: pd.DataFrame,
                      sensitivity_levels: pd.DataFrame, generated_stems: list[str]) -> None:
    removed.to_csv(ANALYSIS_DATA_DIR / "removed_duplicate_rows_v7.csv", index=False)
    duplicate_report.to_csv(DIAGNOSTICS_DIR / "duplicate_configurations_v7.csv", index=False)
    for name, table in audit.items():
        table.to_csv(DIAGNOSTICS_DIR / f"{name}_v7.csv", index=False)
    missing_ref_after_merge.to_csv(DIAGNOSTICS_DIR / "missing_standard_bd_after_reference_merge_v7.csv", index=False)
    sensitivity_levels.to_csv(DIAGNOSTICS_DIR / "sensitivity_levels_v7.csv", index=False)
    pd.DataFrame(ZOOM_DIAGNOSTICS).to_csv(DIAGNOSTICS_DIR / "zoom_diagnostics_v7.csv", index=False)

    parametric_configs = cleaned[cleaned["method_key"].isin(PARAMETRIC_METHOD_KEYS)][
        [c for c in ["method_key", "parametric_config_canonical"] if c in cleaned.columns]
    ].drop_duplicates()
    parametric_configs.to_csv(DIAGNOSTICS_DIR / "parametric_configurations_observed_v7.csv", index=False)

    diagnostics = {
        "input_workbook": str(EXCEL_INPUT_FILE),
        "input_worksheet": sheet,
        "output_directory": str(OUTPUT_DIR),
        "cleaned_deduplicated_records": int(len(cleaned)),
        "removed_duplicate_records": int(len(removed)),
        "duplicate_groups": int(len(duplicate_report)),
        "valid_records": int(cleaned["valid_record"].sum()),
        "invalid_records": int((~cleaned["valid_record"]).sum()),
        "plot_method_keys": PLOT_METHOD_KEYS,
        "expected_method_keys": EXPECTED_METHOD_KEYS,
        "weighting_cross_method_keys": WEIGHTING_CROSS_METHOD_KEYS,
        "primary_callback": PRIMARY_CALLBACK,
        "primary_feature_spec": PRIMARY_FEATURE_SPEC,
        "reference_hyperparameters": REFERENCE_HYPERPARAMETERS,
        "expected_hyperparameter_values": EXPECTED_HYPERPARAMETER_VALUES,
        "legacy_filename_policy": {
            "SCFLP": "unchanged legacy v5 figure stems",
            "SCMND_MANUSCRIPT_ENVELOPE": "unchanged legacy SCMND v5 figure stems",
            "SCMND_CURRENT_SOFT_CAPACITY": "legacy SCMND stem plus _current_soft_capacity suffix",
        },
        "generated_figure_stems": int(len(set(generated_stems))),
        "main_figures_with_zoom_companions": int(len(ZOOM_FIGURES)),
        "completeness_summary": audit["summary"].iloc[0].to_dict() if not audit["summary"].empty else {},
    }
    (DIAGNOSTICS_DIR / "analysis_diagnostics_v7.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")

    notes = f"""Publication analysis v7 assumptions and decisions
=============================================

1. Standard BD is always retained internally, even when all nonbaseline methods are
   commented out of PLOT_METHOD_KEYS.
2. Completeness auditing uses EXPECTED_METHOD_KEYS, not PLOT_METHOD_KEYS.
3. The primary figures use {PRIMARY_CALLBACK}, {PRIMARY_FEATURE_SPEC}, exact-subproblem
   ratios {EXACT_RATIOS}, and the reference hyperparameters {REFERENCE_HYPERPARAMETERS}.
4. SCFLP ignores scmnd_formulation. SCMND baselines are matched only within the same
   scmnd_formulation through analysis_group.
5. MANUSCRIPT_ENVELOPE uses the established SCMND manuscript-facing filenames.
   CURRENT_SOFT_CAPACITY receives the suffix _current_soft_capacity.
6. Duplicate identity includes scmnd formulation (via analysis_group), seed,
   parametric_config, and all feature-generation hyperparameters. Duplicate groups are
   retained deterministically using a data-quality score and fully reported.
7. A record is flagged invalid/corrupted when true reevaluated objective or runtime is
   missing, runtime is negative, or stopping_reason is outside OPTIMAL/TIME_LIMIT.
8. Hyperparameter sensitivity figures are generated only when more than one observed
   value is present after holding the other sensitivity hyperparameters at their
   reference values. The four walk/embedding sensitivities exclude covariates_only.
9. Automatic LaTeX narrative generation from v6 is intentionally not carried forward,
   because the set of plotted parametric methods is configurable. CSV tables, source
   data, diagnostics, and figure files are generated instead.
10. The dedicated weighting-cross outputs form a 3x2 design: Tree, Cluster, and
    P-Median transfer structures crossed with direct nonparametric transfer and
    RHS-Ridge residual prediction. These new methods are not inserted into the
    established manuscript-facing PLOT_METHOD_KEYS or legacy completeness audit.
11. weighting_cross_completeness_v7.csv separately audits every expected primary
    3x2 comparison cell across SCFLP and both SCMND formulations.
"""
    (DIAGNOSTICS_DIR / "analysis_policy_v7.txt").write_text(notes, encoding="utf-8")


def finalize_archive() -> Path | None:
    if SKIP_ARCHIVE:
        return None
    archive_base = OUTPUT_ROOT / "publication_analysis_v7_outputs"
    archive_path = archive_base.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(str(archive_base), "zip", OUTPUT_DIR)
    return archive_path

# =============================================================================
# Main
# =============================================================================
def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for directory in [OUTPUT_DIR, FIGURE_DIR, ORGANIZED_FIGURE_DIR, FIGURE_DATA_DIR,
                      TABLE_DIR, LATEX_DIR, ANALYSIS_DATA_DIR, DIAGNOSTICS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    cleaned, removed, duplicate_report, prepared, sheet = load_and_clean()
    cleaned.to_csv(ANALYSIS_DATA_DIR / "cleaned_deduplicated_data_v7.csv", index=False)
    prepared.to_csv(ANALYSIS_DATA_DIR / "prepared_before_deduplication_v7.csv", index=False)

    audit = audit_experiment_completeness(cleaned, duplicate_report)
    sensitivity_levels = audit_sensitivity_levels(cleaned)

    primary_raw = select_primary(cleaned)
    callback_raw = select_callback_robustness(cleaned)
    feature_raw = select_feature_ablation(cleaned)
    weighting_cross_raw = select_weighting_cross(cleaned)
    primary, missing_primary_ref = attach_standard_bd_reference(primary_raw)
    callback, missing_callback_ref = attach_standard_bd_reference(callback_raw)
    feature, missing_feature_ref = attach_standard_bd_reference(feature_raw)
    weighting_cross, missing_weighting_cross_ref = attach_standard_bd_reference(weighting_cross_raw)

    # For the dedicated 3x2 figures, columns represent the weighting mechanism
    # (Tree, Cluster, P-Median) and the plotted variants are the prediction
    # mechanisms (direct nonparametric transfer versus RHS Ridge).
    weighting_cross_plot = weighting_cross.copy()
    weighting_cross_plot["method"] = weighting_cross_plot["weighting_mechanism"]

    # attach_standard_bd_reference preserves the cross-design columns carried by
    # the nonbaseline rows. Standard-BD rows are used only as references.
    missing_ref_after_merge = pd.concat(
        [missing_primary_ref, missing_callback_ref, missing_feature_ref, missing_weighting_cross_ref],
        ignore_index=True,
    ).drop_duplicates()

    primary.to_csv(ANALYSIS_DATA_DIR / "primary_analysis_data_v7.csv", index=False)
    callback.to_csv(ANALYSIS_DATA_DIR / "callback_robustness_data_v7.csv", index=False)
    feature.to_csv(ANALYSIS_DATA_DIR / "feature_ablation_data_v7.csv", index=False)
    weighting_cross.to_csv(ANALYSIS_DATA_DIR / "weighting_cross_analysis_data_v7.csv", index=False)

    tables = {
        "primary_summary": build_method_summary(primary),
        "primary_by_num_subproblems": build_dimension_summary(primary, "num_subproblems"),
        "primary_by_selected_subproblems_ratio": build_dimension_summary(primary, "selected_subproblems_ratio"),
        "callback_summary": build_method_summary(callback, "callback_policy"),
        "callback_by_num_subproblems": build_dimension_summary(callback, "num_subproblems", "callback_policy"),
        "callback_by_selected_subproblems_ratio": build_dimension_summary(callback, "selected_subproblems_ratio", "callback_policy"),
        "feature_summary": build_method_summary(feature, "feature_specification"),
        "feature_by_num_subproblems": build_dimension_summary(feature, "num_subproblems", "feature_specification"),
        "feature_by_selected_subproblems_ratio": build_dimension_summary(feature, "selected_subproblems_ratio", "feature_specification"),
        "coverage": build_coverage(primary),
        "weighting_cross_summary": build_weighting_cross_summary(weighting_cross),
        "weighting_cross_by_num_subproblems": build_weighting_cross_summary(weighting_cross, "num_subproblems"),
        "weighting_cross_by_selected_subproblems_ratio": build_weighting_cross_summary(weighting_cross, "selected_subproblems_ratio"),
        "weighting_cross_coverage": build_weighting_cross_coverage(weighting_cross),
        "weighting_cross_completeness": build_weighting_cross_completeness(cleaned),
    }
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}_v7.csv", index=False)

    groups_present = [
        g for g in [SCFLP_GROUP, f"SCMND_{SCMND_MANUSCRIPT_FORMULATION}", f"SCMND_{SCMND_ALTERNATIVE_FORMULATION}", UNSPECIFIED_SCMND_GROUP]
        if cleaned["analysis_group"].eq(g).any()
    ]
    generated_stems: list[str] = []

    if not SKIP_FIGURES:
        for group in groups_present:
            for stratum in STRATUM_ORDER:
                for dimension in ["num_subproblems", "selected_subproblems_ratio"]:
                    generated_stems += make_main_figures(primary, group=group, stratum=stratum, dimension=dimension)
                    generated_stems += make_variant_figures(
                        callback, group=group, stratum=stratum, dimension=dimension,
                        prefix="appendix_callback", methods=NONBASELINE_METHOD_ORDER,
                        variants=[CALLBACK_SHORT[PRIMARY_CALLBACK], CALLBACK_SHORT[SUPPLEMENTARY_CALLBACK]],
                        variant_column="callback_short",
                    )
                    generated_stems += make_variant_figures(
                        feature, group=group, stratum=stratum, dimension=dimension,
                        prefix="appendix_features", methods=LABD_METHODS,
                        variants=["R", "C", "R+C"], variant_column="feature_specification",
                    )
                    generated_stems += make_variant_figures(
                        weighting_cross_plot,
                        group=group,
                        stratum=stratum,
                        dimension=dimension,
                        prefix="weighting_cross",
                        methods=["Tree", "Cluster", "P-Median"],
                        variants=["Nonparametric", "RHS Ridge"],
                        variant_column="prediction_mechanism",
                    )

        # Established extra metrics.
        for metric, config in EXTRA_METRICS.items():
            methods = _method_labels(config["method_keys"])
            for group in groups_present:
                for stratum in STRATUM_ORDER:
                    for dimension in [None, "num_subproblems", "selected_subproblems_ratio"]:
                        for kind in ["box", "violin"]:
                            for counts in [False, True]:
                                stem = make_extra_metric_figure(
                                    primary, metric=metric, metric_stem=config["stem"], methods=methods,
                                    group=group, stratum=stratum, dimension=dimension, kind=kind, counts=counts,
                                )
                                if stem:
                                    generated_stems.append(stem)
            for factor in [None, "num_subproblems", "selected_subproblems_ratio"]:
                table = metric_summary_table(primary, metric=metric, methods=methods, factor=factor)
                suffix = "" if factor is None else f"_by_{factor}"
                table.to_csv(TABLE_DIR / f"{config['stem']}_summary{suffix}_v7.csv", index=False)

        # Conditional one-factor-at-a-time sensitivities.
        if GENERATE_SENSITIVITY_FIGURES:
            for parameter in SENSITIVITY_PARAMETERS:
                if len(_sensitivity_values(cleaned, parameter)) <= 1:
                    continue
                for cb in EXPECTED_CALLBACKS:
                    sens_raw = select_sensitivity_data(cleaned, parameter, cb)
                    sens, _ = attach_standard_bd_reference(pd.concat([
                        cleaned[cleaned["method_key"].eq("STANDARD_BD")], sens_raw
                    ], ignore_index=True))
                    for group in groups_present:
                        for stratum in STRATUM_ORDER:
                            generated_stems += make_sensitivity_figures(
                                sens, parameter=parameter, callback=cb, group=group, stratum=stratum
                            )
                # Table pooled across callbacks for traceability.
                sens_all = pd.concat([select_sensitivity_data(cleaned, parameter, cb) for cb in EXPECTED_CALLBACKS], ignore_index=True)
                sens_all.to_csv(ANALYSIS_DATA_DIR / f"sensitivity_{parameter}_data_v7.csv", index=False)

        if GENERATE_PARAMETRIC_DIAGNOSTICS:
            generated_stems += generate_parametric_diagnostics(primary, groups_present)

        organize_output_files()

    write_diagnostics(cleaned, removed, duplicate_report, audit, sheet, missing_ref_after_merge,
                      sensitivity_levels, generated_stems)
    archive_path = finalize_archive()

    summary = audit["summary"].iloc[0].to_dict() if not audit["summary"].empty else {}
    print(json.dumps({
        "input_workbook": str(EXCEL_INPUT_FILE),
        "input_sheet": sheet,
        "output_directory": str(OUTPUT_DIR),
        "cleaned_records": int(len(cleaned)),
        "figures_skipped": bool(SKIP_FIGURES),
        "pdf_figures": int(len(list(FIGURE_DIR.glob("*.pdf")))),
        "audit": summary,
        "archive": str(archive_path) if archive_path else None,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
