#!/usr/bin/env python3
"""Read-only component analysis of five completed guard pilots.

Compare original nonparametric, guard-OFF parametric and guard-ON parametric
using existing per-task solution_log.xlsx files on Nibi. No solver runs,
recomputation, or modification of experiment files.

Usage: python scripts/analyze_guard_runtime_components.py [PILOT_ROOT]
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
import pandas as pd

RESULTS = Path.home() / "scratch/Benders_Integration_with_ML/RESULTS"
ORIGINAL = "pmedian_nearest_transfer_20261016_160602"
PILOTS = (46, 66, 76, 199, 232)

COMPONENTS = (
    "solution_time_sec",
    "phase1_execution_time_sec",
    "phase2_execution_time_sec",
    "feature_gen_time_sec",
    "unified_selection_training_time_sec",
    "parametric_training_time_sec",
    "parametric_prediction_time_sec",
    "parametric_feasibility_time_sec",
    "parametric_repair_time_sec",
    "parametric_algorithm_exact_subproblem_time_sec",
    "parametric_reevaluation_exact_subproblem_time_sec",
)
TRAJECTORY = (
    "objective_value",
    "original_Prob_Obj_Value",
    "iteration",
    "num_predictions",
    "bnb_nodes",
    "nodes_explored",
    "cuts_mip_selected",
    "cuts_rel_selected",
    "cuts_mip_ml",
    "cuts_rel_ml",
    "parametric_raw_predictions",
    "parametric_raw_infeasible_predictions",
    "parametric_mean_repair_gamma",
    "parametric_repaired_predictions",
    "parametric_algorithm_exact_subproblem_solves",
    "parametric_cuts_added",
)
METRICS = COMPONENTS + TRAJECTORY + ("stopping_reason", "parametric_config")


def one(log_root: Path, task: int) -> dict:
    matches = list(log_root.glob(f"task_{task}_*/solution_log.xlsx"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one result for task {task} under {log_root}, "
            f"got {len(matches)}"
        )
    df = pd.read_excel(matches[0])
    if len(df) != 1:
        raise RuntimeError(f"Expected one row in {matches[0]}, got {len(df)}")
    return df.iloc[0].to_dict()


def number(row: dict, col: str):
    value = row.get(col)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def formatted(v, digits=2):
    return "-" if v is None else f"{v:,.{digits}f}"


def difference(a, b):
    return None if a is None or b is None else b - a


def same_numeric(a, b, col):
    x, y = number(a, col), number(b, col)
    if x is None or y is None:
        return None
    return math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-7)


def main():
    if len(sys.argv) >= 2:
        root = Path(sys.argv[1]).expanduser()
    else:
        root = Path(
            (RESULTS / "MASTER_RUNS/latest_parametric_guard_pilot.txt")
            .read_text().strip()
        )
    old_run = ORIGINAL
    new_run = root.name
    output = []

    for task in PILOTS:
        problem = "SCFLP" if task <= 90 else "SCMND"
        original_dir = RESULTS / f"{problem}_class/RESULTS" / old_run
        guarded_dir = RESULTS / f"{problem}_class/RESULTS" / new_run
        baseline_id = task - (45 if problem == "SCFLP" else 108)
        rows = {
            "nonparam": one(original_dir, baseline_id),
            "off": one(original_dir, task),
            "on": one(guarded_dir, task),
        }
        conf = rows["on"].get("parametric_config")
        try:
            flag = json.loads(conf or "{}").get("parametric_objective_guard")
        except (TypeError, ValueError):
            flag = None
        if flag is not True:
            raise RuntimeError(f"Task {task}: pilot configuration not guard-ON.")

        for metric in METRICS:
            if metric in ("stopping_reason", "parametric_config"):
                continue
            data = dict(
                task_id=task, problem=problem, metric=metric,
                nonparam=number(rows["nonparam"], metric),
                off=number(rows["off"], metric),
                on=number(rows["on"], metric),
            )
            data["on_minus_off"] = difference(data["off"], data["on"])
            output.append(data)

        def n(method, key):
            return number(rows[method], key)

        oldt, newt = n("off", "solution_time_sec"), n("on", "solution_time_sec")
        raw_on = n("on", "parametric_raw_predictions")
        print(f"\n===== TASK {task} / {problem} =====")
        print(
            f"runtime: nonparam {formatted(n('nonparam','solution_time_sec'))}s"
            f" | off {formatted(oldt)}s | on {formatted(newt)}s"
            f" | ON-OFF {formatted(newt-oldt)}s ({100*(newt/oldt-1):+.2f}%)"
        )
        print(
            f"same trajectory OFF vs ON: master_obj={same_numeric(rows['off'],rows['on'],'objective_value')}, "
            f"iterations={same_numeric(rows['off'],rows['on'],'iteration')}, "
            f"nodes={same_numeric(rows['off'],rows['on'],'bnb_nodes')}, "
            f"ML-MIP-cuts={same_numeric(rows['off'],rows['on'],'cuts_mip_ml')}, "
            f"ML-LP-cuts={same_numeric(rows['off'],rows['on'],'cuts_rel_ml')}"
        )
        print(
            f"number of raw dual predictions: OFF {formatted(n('off','parametric_raw_predictions'),0)}"
            f" ON {formatted(raw_on,0)}"
            f" | repair gamma mean: OFF {formatted(n('off','parametric_mean_repair_gamma'),6)}"
            f" ON {formatted(n('on','parametric_mean_repair_gamma'),6)}"
            f" | raw-infeasible ON {formatted(n('on','parametric_raw_infeasible_predictions'),0)}"
        )
        print(
            f"effective wall-time difference per ON raw dual: "
            f"{formatted(1000*(newt-oldt)/raw_on if raw_on else None,5)} ms "
            "(NOT a measured guard timer)"
        )
        print("Timing decomposition (seconds; ON minus OFF):")
        for key in COMPONENTS[1:]:
            off, on = n("off", key), n("on", key)
            if off is not None and on is not None:
                print(
                    f"  {key:<50} OFF {off:>9.3f}"
                    f"  ON {on:>9.3f}  difference {on-off:>+9.3f}"
                )
        print(
            "CAUTION: timing components, especially exact-solve totals and "
            "phase times, can overlap and must NOT be summed."
        )

    out = root / "guard_existing_results_component_comparison.csv"
    with out.open("w", newline="") as file:
        wr = csv.DictWriter(file, fieldnames=list(output[0]))
        wr.writeheader()
        wr.writerows(output)
    print(f"\nFull component comparison saved to: {out}")
    print(
        "No experiments were submitted; all source workbooks were read without modification."
    )


if __name__ == "__main__":
    main()
