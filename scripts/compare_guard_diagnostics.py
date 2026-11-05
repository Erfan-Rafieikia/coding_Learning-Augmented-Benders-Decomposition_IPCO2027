#!/usr/bin/env python3
"""Read-only analysis for a matched three-arm parametric diagnostic experiment.

Usage: python scripts/compare_guard_diagnostics.py [DIAG_MASTER_ROOT]
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
import pandas as pd

RESULTS = Path.home() / "scratch/Benders_Integration_with_ML/RESULTS"
POINTER = RESULTS / "MASTER_RUNS/latest_parametric_guard_diag.txt"

FIELDS = (
    "solution_time_sec", "phase1_execution_time_sec", "phase2_execution_time_sec",
    "iteration", "bnb_nodes", "nodes_explored", "cuts_mip_selected",
    "cuts_rel_selected", "cuts_mip_ml", "cuts_rel_ml",
    "objective_value", "original_Prob_Obj_Value",
    "parametric_raw_predictions", "parametric_raw_infeasible_predictions",
    "parametric_mean_repair_gamma", "parametric_min_repair_gamma",
    "parametric_repair_time_sec", "parametric_prediction_time_sec",
    "parametric_feasibility_time_sec",
    "parametric_algorithm_exact_subproblem_solves",
    "parametric_algorithm_exact_subproblem_time_sec",
    "parametric_reevaluation_exact_subproblem_time_sec",
    "parametric_diag_gamma_zero", "parametric_diag_gamma_positive_count",
    "parametric_diag_gamma_positive_mean", "parametric_diag_gamma_max",
    "parametric_diag_gamma_gt_1e12", "parametric_diag_gamma_gt_1e8",
    "parametric_diag_gamma_gt_1e6", "parametric_diag_gamma_gt_1e3",
    "parametric_diag_gamma_gt_01",
    "parametric_diag_probe_checks", "parametric_diag_probe_improved",
    "parametric_diag_probe_negative", "parametric_diag_probe_tied",
    "parametric_diag_probe_positive_total", "parametric_diag_probe_negative_total",
    "parametric_diag_probe_max_positive",
    "parametric_diag_probe_anchor_check_time_sec",
    "parametric_diag_probe_objective_time_sec",
    "parametric_diag_probe_total_time_sec",
    "parametric_objective_guard_checks",
    "parametric_objective_guard_accepted",
    "parametric_objective_guard_rejected",
    "parametric_objective_guard_rejected_negative",
    "parametric_objective_guard_total_improvement",
)


def read_task(root, problem, task):
    folder = RESULTS / f"{problem}_class/RESULTS" / root.name
    files = list(folder.glob(f"task_{task}_*/solution_log.xlsx"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one Excel file for {folder} / task {task}: {files}")
    rows = pd.read_excel(files[0])
    if len(rows) != 1:
        raise RuntimeError(f"Expected one experiment row in {files[0]}, got {len(rows)}")
    return rows.iloc[0], str(files[0])


def val(row, key):
    v = row.get(key)
    if v is None or pd.isna(v):
        return None
    try:
        v = float(v)
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None


def assert_match(rows, target):
    non, off, on = (rows[x][0] for x in ("NONPARAM", "OFF", "ON"))
    for key in ("instance_filename", "selected_subproblems_ratio",
                "scmnd_formulation", "num_subproblems"):
        if key == "selected_subproblems_ratio":
            # These three methods must use the same representative budget.
            a = [str(rows[m][0].get(key)) for m in ("NONPARAM", "OFF", "ON")]
        else:
            a = [str(rows[m][0].get(key)) for m in ("NONPARAM", "OFF", "ON")]
        if len(set(a)) != 1:
            raise RuntimeError(f"Pair {target}: mismatched {key}: {a}")
    if str(off.get("predictor_type")) != str(on.get("predictor_type")):
        raise RuntimeError(f"Pair {target}: predictor variant mismatch")
    for method in ("OFF", "ON"):
        row = rows[method][0]
        c = json.loads(row["parametric_config"])
        if c.get("parametric_diagnostic_probe") is not True:
            raise RuntimeError(f"Pair {target} {method}: diagnostic flag missing")
        if c.get("parametric_objective_guard") is not (method == "ON"):
            raise RuntimeError(f"Pair {target} {method}: wrong guard condition")
        checks = val(row, "parametric_diag_probe_checks")
        predicted = val(row, "parametric_raw_predictions")
        if checks != predicted:
            raise RuntimeError(
                f"Pair {target} {method}: objective probe counts {checks} "
                f"do not match raw prediction count {predicted}"
            )
        zero = val(row, "parametric_diag_gamma_zero")
        positive = val(row, "parametric_diag_gamma_positive_count")
        if zero is None or positive is None or zero + positive != predicted:
            raise RuntimeError(
                f"Pair {target} {method}: gamma counts do not sum to predictions"
            )
        improve = val(row, "parametric_diag_probe_improved")
        negative = val(row, "parametric_diag_probe_negative")
        ties = val(row, "parametric_diag_probe_tied")
        if improve + negative + ties != checks:
            raise RuntimeError(f"Pair {target} {method}: probe outcomes inconsistent")
        actual_checks = val(row, "parametric_objective_guard_checks")
        if actual_checks != (predicted if method == "ON" else 0):
            raise RuntimeError(
                f"Pair {target} {method}: objective guard executed unexpectedly"
            )


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else POINTER.read_text().strip())
    with (root / "task_pairs.tsv").open() as f:
        task_pairs = list(csv.DictReader(f, delimiter="\t"))
    by_pair = {}
    for p in task_pairs:
        target = int(p["pair_parametric_task_id"])
        method = p["method"]
        by_pair.setdefault(target, {})[method] = read_task(
            root, p["problem_class"], int(p["new_task_id"])
        )

    wide = []
    print("Metrics measured on ONE instrumentation branch; objective gaps at current x.")
    for target, rows in sorted(by_pair.items()):
        if set(rows) != {"NONPARAM", "OFF", "ON"}:
            raise RuntimeError(f"Pair {target}: incomplete methods {sorted(rows)}")
        assert_match(rows, target)
        baseline, off, on = (rows[m][0] for m in ("NONPARAM", "OFF", "ON"))
        p = dict(target=target, problem=str(on["problem_class"]))
        for method, row in (("nonparam", baseline), ("off", off), ("on", on)):
            for key in FIELDS:
                p[f"{method}_{key}"] = val(row, key)
            p[f"{method}_stopping_reason"] = str(row.get("stopping_reason"))
            p[f"{method}_file"] = rows[method.upper()][1]
        p["on_minus_off_runtime_sec"] = (
            val(on, "solution_time_sec") - val(off, "solution_time_sec")
        )
        p["on_vs_off_runtime_pct"] = 100 * (
            val(on, "solution_time_sec") / val(off, "solution_time_sec") - 1
        )
        wide.append(p)
        print(f"\nPAIR {target} {p['problem']}:")
        print("  runtime nonparam / OFF / ON:",
              *(round(val(x, "solution_time_sec"), 3) for x in (baseline, off, on)))
        for name, row in (("OFF", off), ("ON", on)):
            print(
                f"  {name}: gamma==0={int(val(row,'parametric_diag_gamma_zero')):,}, "
                f"gamma>1e-12={int(val(row,'parametric_diag_gamma_gt_1e12')):,}, "
                f"gamma>1e-6={int(val(row,'parametric_diag_gamma_gt_1e6')):,}, "
                f"max gamma={val(row,'parametric_diag_gamma_max'):.12g}"
            )
            print(
                f"       improved={int(val(row,'parametric_diag_probe_improved')):,}, "
                f"negative={int(val(row,'parametric_diag_probe_negative')):,}, "
                f"non-strict/ties={int(val(row,'parametric_diag_probe_tied')):,}, "
                f"positive objective gain={val(row,'parametric_diag_probe_positive_total'):.10g}"
            )
            print(
                f"       probe time={val(row,'parametric_diag_probe_total_time_sec'):.3f}s "
                f"(anchor feasibility={val(row,'parametric_diag_probe_anchor_check_time_sec'):.3f}s, "
                f"objective calculation={val(row,'parametric_diag_probe_objective_time_sec'):.3f}s)"
            )
        for key in ("iteration", "bnb_nodes", "cuts_mip_ml", "cuts_rel_ml"):
            print(f"  {key}: OFF={val(off,key)} ON={val(on,key)}")
    path = root / "matched_diagnostic_comparison.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(wide[0]))
        writer.writeheader()
        writer.writerows(wide)
    print(f"\nCompleted {len(wide)} three-arm matched comparisons.")
    print(f"Full component comparison (new file, never original Excel): {path}")


if __name__ == "__main__":
    main()
