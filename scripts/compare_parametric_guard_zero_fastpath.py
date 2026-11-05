#!/usr/bin/env python3
"""Compare four matched methods in the zero-gamma fastpath experiment.

Read-only on source Excel results. Writes a separate comparison CSV under
the new experimental master directory.

Usage:
    python scripts/compare_parametric_guard_zero_fastpath.py [RUN_ROOT]
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd


RESULTS = Path.home() / "scratch/Benders_Integration_with_ML/RESULTS"
POINTER = RESULTS / "MASTER_RUNS/latest_parametric_guard_zero_fastpath.txt"
ARMS = ("NONPARAM", "OFF", "GUARD", "FAST")
TRACK = (
    "solution_time_sec",
    "phase1_execution_time_sec",
    "phase2_execution_time_sec",
    "objective_value",
    "original_Prob_Obj_Value",
    "iteration",
    "bnb_nodes",
    "nodes_explored",
    "cuts_mip_ml",
    "cuts_rel_ml",
    "cuts_mip_selected",
    "cuts_rel_selected",
    "parametric_raw_predictions",
    "parametric_mean_repair_gamma",
    "parametric_objective_guard_checks",
    "parametric_objective_guard_accepted",
    "parametric_objective_guard_rejected",
    "parametric_objective_guard_evaluated",
    "parametric_objective_guard_skipped_zero",
    "parametric_objective_guard_time_sec",
    "parametric_prediction_time_sec",
    "parametric_feasibility_time_sec",
    "parametric_repair_time_sec",
    "parametric_algorithm_exact_subproblem_time_sec",
    "parametric_reevaluation_exact_subproblem_time_sec",
)


def read_result(run_root, task_id, problem):
    pattern = (
        RESULTS / f"{problem}_class" / "RESULTS" / run_root.name
    ).glob(f"task_{task_id}_*/solution_log.xlsx")
    paths = list(pattern)
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one Excel for {problem} task {task_id}: {paths}")
    df = pd.read_excel(paths[0])
    if len(df) != 1:
        raise ValueError(f"Expected one Excel row: {paths[0]}")
    return df.iloc[0], str(paths[0])


def metric(row, key):
    x = row.get(key)
    if x is None or pd.isna(x):
        return None
    try:
        value = float(x)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def equal(a, b):
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6)


def guard_counts(row, arm, target):
    config = json.loads(row["parametric_config"])
    expected = arm in ("GUARD", "FAST")
    if config.get("parametric_objective_guard") is not expected:
        raise ValueError(f"Task {target} / {arm}: wrong guard setting")
    if config.get("parametric_guard_gamma_zero_fastpath") is not (arm == "FAST"):
        raise ValueError(f"Task {target} / {arm}: wrong fastpath setting")
    if config.get("parametric_diagnostic_probe") is not False:
        raise ValueError(f"Task {target} / {arm}: shadow objective probe must be OFF")

    predictions = metric(row, "parametric_raw_predictions")
    checks = metric(row, "parametric_objective_guard_checks")
    evaluated = metric(row, "parametric_objective_guard_evaluated")
    skipped = metric(row, "parametric_objective_guard_skipped_zero")
    accepted = metric(row, "parametric_objective_guard_accepted")
    rejected = metric(row, "parametric_objective_guard_rejected")
    expected_checks = predictions if expected else 0
    if not equal(checks, expected_checks):
        raise ValueError(f"Task {target} / {arm}: unexpected guard check count")
    if not equal(evaluated + skipped, checks):
        raise ValueError(f"Task {target} / {arm}: evaluated + skipped != checks")
    if not equal(accepted + rejected, checks):
        raise ValueError(f"Task {target} / {arm}: accepted + rejected != checks")
    if arm == "GUARD" and skipped != 0:
        raise ValueError(f"Task {target}: standard guard unexpectedly skipped checks")
    if arm == "OFF" and skipped != 0:
        raise ValueError(f"Task {target}: guard OFF unexpectedly skipped checks")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else POINTER.read_text().strip())
    with (root / "task_pairs.tsv").open() as f:
        pairs = list(csv.DictReader(f, delimiter="\t"))
    by_target = {}
    for entry in pairs:
        target = int(entry["pair_parametric_task_id"])
        arm = entry["method"]
        task = int(entry["new_task_id"])
        if arm in by_target.setdefault(target, {}):
            raise ValueError(f"Duplicate arm for task {target}/{arm}")
        by_target[target][arm] = read_result(root, task, entry["problem_class"])

    records = []
    print("Matched four-arm experiment: original baseline, unguarded, standard guard, zero-gamma guard")
    for target, batch in sorted(by_target.items()):
        if set(batch) != set(ARMS):
            raise ValueError(f"Task {target}: missing arms; got {sorted(batch)}")
        rows = {arm: batch[arm][0] for arm in ARMS}
        for key in ("instance_filename", "selected_subproblems_ratio", "scmnd_formulation", "num_subproblems"):
            values = {str(rows[arm].get(key)) for arm in ARMS}
            if len(values) != 1:
                raise ValueError(f"Task {target}: unmatched {key}: {values}")
        for arm in ("OFF", "GUARD", "FAST"):
            if str(rows["GUARD"].get("predictor_type")) != str(rows[arm].get("predictor_type")):
                raise ValueError(f"Task {target}: predictor mismatch {arm}")
            guard_counts(rows[arm], arm, target)
        if any(
            str(rows[arm].get("stopping_reason", "")).upper().startswith(("CODE_ERROR", "SOLUTION_NULL"))
            for arm in ARMS
        ):
            raise ValueError(f"Task {target}: some runs are invalid")

        row = {"task_id": target, "problem_class": str(rows["FAST"].get("problem_class"))}
        for arm in ARMS:
            for key in TRACK:
                row[f"{arm.lower()}_{key}"] = metric(rows[arm], key)
            row[f"{arm.lower()}_stopping_reason"] = str(rows[arm].get("stopping_reason"))
            row[f"{arm.lower()}_source_excel"] = batch[arm][1]
        ref, fast = rows["GUARD"], rows["FAST"]
        ref_time, fast_time = metric(ref, "solution_time_sec"), metric(fast, "solution_time_sec")
        row["fast_minus_standard_runtime_sec"] = fast_time - ref_time
        row["fast_runtime_change_pct"] = 100 * (fast_time/ref_time - 1)
        row["fast_minus_standard_guard_time_sec"] = (
            metric(fast, "parametric_objective_guard_time_sec")
            - metric(ref, "parametric_objective_guard_time_sec")
        )
        match_keys = (
            "objective_value", "original_Prob_Obj_Value",
            "iteration", "bnb_nodes", "nodes_explored",
            "cuts_mip_ml", "cuts_rel_ml", "cuts_mip_selected", "cuts_rel_selected",
        )
        matched = all(
            equal(metric(ref, key), metric(fast, key))
            for key in match_keys
        )
        row["standard_vs_fast_solver_metrics_match"] = matched
        records.append(row)

        print(f"\nTASK {target} {row['problem_class']} (seconds):")
        print(
            "  Nonparametric={:.3f}; Guard OFF={:.3f}; Guard standard={:.3f}; Guard fast={:.3f}"
            .format(*(metric(rows[arm], "solution_time_sec") for arm in ARMS))
        )
        print(
            f"  Shortcut: {int(metric(fast,'parametric_objective_guard_skipped_zero')):,} "
            f"of {int(metric(fast,'parametric_objective_guard_checks')):,} checks avoided; "
            f"non-skipped objective evaluations={int(metric(fast,'parametric_objective_guard_evaluated')):,}"
        )
        print(
            "  Guard time: standard={:.4f}s vs fast={:.4f}s; difference={:+.4f}s"
            .format(
                metric(ref, "parametric_objective_guard_time_sec"),
                metric(fast, "parametric_objective_guard_time_sec"),
                row["fast_minus_standard_guard_time_sec"],
            )
        )
        print(
            f"  Total runtime difference fast-standard={row['fast_minus_standard_runtime_sec']:+.3f}s "
            f"({row['fast_runtime_change_pct']:+.2f}%)"
        )
        print(
            f"  Identical final objectives / solver counters / cut counts: {matched}"
        )
    output = root / "matched_zero_gamma_fastpath_comparison.csv"
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nMatched comparisons: {len(records)}.")
    print(f"New comparison CSV (original experiment files untouched): {output}")
    print("Guard time is a directly measured subcomponent; overall runtimes can also reflect solver variability.")


if __name__ == "__main__":
    main()
