"""Gurobi-observed master bounds; no retrospective candidate evaluation.

For inexact methods these are surrogate/master bounds. The final exact original
objective is a separate workbook diagnostic and is never inserted into a trace.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

TRACE_SCHEMA_VERSION = "gurobi_master_v3_events"
LEGACY_MASTER_TRACE_VERSION = "gurobi_master_v2"
TRACE_EVENTS = {
    "solver_initial", "solver_bound_change", "solver_cut_change",
    "solver_iteration_start", "solver_iteration_end", "solver_final",
}
SUBMISSION_FIELDS = (
    "submitted_exact_cuts", "submitted_ml_cuts",
    "submitted_lazy_cuts", "submitted_user_cuts",
)
TRACE_FIELDS = (
    "elapsed_seconds", "iteration", "node_count", "event",
    "lower_bound", "master_upper_bound", "master_relative_gap",
    "lb_ub_inconsistent", "callback_context", "bound_changed",
    "gurobi_mip_cut_count", "gurobi_cut_count_changed",
    *SUBMISSION_FIELDS, "trace_schema_version",
)


def _finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # Gurobi's +/-1e100 sentinels mean unavailable, not finite bounds.
    return f if math.isfinite(f) and abs(f) < 1e100 else None


def verified_relative_gap(lower_bound, upper_bound):
    """Signed final-original gap; negative values expose inconsistent bounds."""
    difference = upper_bound - lower_bound
    if upper_bound == 0:
        return 0.0 if difference == 0 else math.copysign(math.inf, difference)
    return difference / abs(upper_bound)


def master_relative_gap(lower_bound, upper_bound):
    """Gurobi's absolute relative MIP gap from its observed bounds."""
    return abs(verified_relative_gap(lower_bound, upper_bound))


def build_solver_bound_trace(solver_points):
    """Preserve each observed pair and timestamp without smoothing or backdating.

    Keep observed bound/count changes, Benders callback iteration boundaries,
    and the terminal observation. Gurobi-internal changes between callback
    invocations are not observable. Never replace an incumbent with MIPSOL_OBJ.
    """
    output = []
    previous = -1.0
    for point in solver_points or ():
        stamp = _finite(point.get("elapsed_seconds"))
        if stamp is None or stamp < 0 or stamp < previous:
            raise ValueError("Invalid or nonmonotone solver timestamp")
        previous = stamp
        lb = _finite(point.get("lower_bound"))
        ub = _finite(point.get("master_upper_bound"))
        gap = None
        bad = False
        if lb is not None and ub is not None:
            gap = master_relative_gap(lb, ub)
            bad = lb > ub + 1e-6 * max(1.0, abs(ub))
        output.append({
            "elapsed_seconds": stamp,
            "iteration": point.get("iteration"),
            "node_count": point.get("node_count"),
            "event": point.get("event", "solver_bound_change"),
            "lower_bound": lb,
            "master_upper_bound": ub,
            "master_relative_gap": gap,
            "lb_ub_inconsistent": int(bad),
            "callback_context": point.get("callback_context"),
            "bound_changed": point.get("bound_changed"),
            "gurobi_mip_cut_count": point.get("gurobi_mip_cut_count"),
            "gurobi_cut_count_changed": point.get("gurobi_cut_count_changed"),
            **{key: point.get(key) for key in SUBMISSION_FIELDS},
            "trace_schema_version": TRACE_SCHEMA_VERSION,
        })
    return output


def write_bound_trace(rows, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)
