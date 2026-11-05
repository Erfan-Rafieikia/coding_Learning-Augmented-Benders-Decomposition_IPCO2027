#!/usr/bin/env python3
"""Compare guarded P-median pilots to matched unguarded and nonparametric runs.

Supports the original pilot Excel schema (guard counts survived in
parametric_iteration_records and the flag in parametric_config), as well as
newer Excel files with dedicated guard-diagnostic columns.

Usage:
    python scripts/compare_parametric_guard_pilots.py [PILOT_ROOT]
Requires pandas/openpyxl (bd_ml environment). Does not modify existing logs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


RESULTS = Path.home() / "scratch/Benders_Integration_with_ML/RESULTS"
DEFAULT_POINTER = RESULTS / "MASTER_RUNS/latest_parametric_guard_pilot.txt"
PILOT_IDS = (46, 66, 76, 199, 232)
OLD_RUN = "pmedian_nearest_transfer_20261016_160602"


def find_one(root: Path, task_id: int) -> Path:
    files = sorted(root.glob(f"task_{task_id}_*/solution_log.xlsx"))
    if len(files) != 1:
        raise ValueError(
            f"Task {task_id}: expected one Excel result under {root}, found {len(files)}"
        )
    return files[0]


def read_one(file: Path):
    df = pd.read_excel(file)
    if len(df) != 1:
        raise ValueError(f"{file}: expected one result row, found {len(df)}")
    return df.iloc[0]


def json_field(row, name, expected_type):
    """Read structured metadata retained by the fixed-schema Excel exporter."""
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        return expected_type()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {name}: {exc}") from exc
    if not isinstance(parsed, expected_type):
        raise ValueError(f"Expected {expected_type.__name__} in {name}")
    return parsed


def guard_metrics(row):
    """Recover pilot guard status without changing or rerunning prior results."""
    config = json_field(row, "parametric_config", dict)
    if config.get("parametric_objective_guard") is not True:
        raise ValueError(
            "Objective guard not confirmed in saved parametric_config; "
            f"observed value={config.get('parametric_objective_guard')!r}"
        )
    records = None
    records_truncated = False
    try:
        records = json_field(row, "parametric_iteration_records", list)
    except ValueError:
        # Excel cells are limited to 32,767 characters. Longer JSON may have
        # been truncated by the first pilot's exporter. Do not fabricate totals.
        raw_records = row.get("parametric_iteration_records")
        if isinstance(raw_records, str) and len(raw_records) >= 32000:
            records_truncated = True
        else:
            raise
    if records and not all(
        isinstance(r, dict) and r.get("objective_guard_enabled") is True
        for r in records
    ):
        raise ValueError("Saved iteration diagnostics do not consistently enable guard")

    dedicated = (
        "parametric_objective_guard_checks",
        "parametric_objective_guard_accepted",
        "parametric_objective_guard_rejected",
    )
    if all(k in row.index and pd.notna(row[k]) for k in dedicated):
        checks, accepted, rejected = (int(row[k]) for k in dedicated)
        negative = row.get("parametric_objective_guard_rejected_negative")
        improvement = row.get("parametric_objective_guard_total_improvement")
        negative = int(negative) if pd.notna(negative) else None
        improvement = float(improvement) if pd.notna(improvement) else None
        source = "excel_dedicated_columns"
    else:
        # First pilot's exporter dropped new columns from its fixed schema.
        # Recover counts from retained per-iteration JSON where complete.
        negative = None  # Not stored in the first pilot's iteration records
        improvement = None  # Not stored in the first pilot's iteration records
        if records_truncated:
            checks = accepted = rejected = None
            source = "excel_iteration_records_truncated"
        else:
            checks = sum(int(r.get("objective_guard_checked", 0)) for r in records)
            accepted = sum(int(r.get("objective_guard_accepted", 0)) for r in records)
            rejected = sum(int(r.get("objective_guard_rejected", 0)) for r in records)
            source = "excel_iteration_records"
            opportunities = row.get("parametric_prediction_opportunities")
            if not records and pd.notna(opportunities) and float(opportunities) > 0:
                raise ValueError(
                    "Prediction opportunities exist but no retained iteration "
                    "records are available to verify safeguard execution."
                )

    if checks is not None and (
        checks < 0 or accepted < 0 or rejected < 0
        or checks != accepted + rejected
    ):
        raise ValueError(
            f"Inconsistent guard counts: checks={checks}, "
            f"accepted={accepted}, rejected={rejected}"
        )
    return dict(
        guard_checks=checks,
        guard_accepted=accepted,
        guard_rejected=rejected,
        guard_rejected_negative=negative,
        guard_total_improvement=improvement,
        guard_diagnostic_source=source,
    )


def compare():
    root = Path(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_POINTER.read_text().strip()
    ).expanduser()
    if not (root / "code_commit.txt").is_file():
        raise ValueError(f"Not a prepared objective-guard pilot root: {root}")
    run = root.name
    records = []

    for task in PILOT_IDS:
        problem = "SCFLP" if task <= 90 else "SCMND"
        original_dir = RESULTS / f"{problem}_class/RESULTS" / OLD_RUN
        new_dir = RESULTS / f"{problem}_class/RESULTS" / run
        old_file = find_one(original_dir, task)
        new_file = find_one(new_dir, task)
        nonparam_task = task - (45 if problem == "SCFLP" else 108)
        nonparam_file = find_one(original_dir, nonparam_task)
        old, new, nonparam = (
            read_one(old_file), read_one(new_file), read_one(nonparam_file)
        )

        for col in ("instance_filename", "predictor_type", "selected_subproblems_ratio"):
            if str(old[col]) != str(new[col]):
                raise ValueError(
                    f"Task {task}: different {col}: {old[col]!r} vs {new[col]!r}"
                )
        for col in ("instance_filename", "selected_subproblems_ratio"):
            if str(nonparam[col]) != str(new[col]):
                raise ValueError(f"Task {task}: nonparametric {col} mismatch")
        if problem == "SCMND":
            for col in ("scmnd_formulation",):
                if str(old[col]) != str(new[col]) or str(nonparam[col]) != str(new[col]):
                    raise ValueError(f"Task {task}: SCMND formulation mismatch")

        old_config = json_field(old, "parametric_config", dict)
        new_config = json_field(new, "parametric_config", dict)
        old_config.pop("parametric_objective_guard", None)
        new_config.pop("parametric_objective_guard", None)
        if old_config != new_config:
            raise ValueError(
                f"Task {task}: other parametric settings changed between paired runs"
            )
        try:
            guard = guard_metrics(new)
        except ValueError as exc:
            raise ValueError(f"Task {task}: {exc}") from exc

        reason = str(new.get("stopping_reason", "")).strip()
        if not reason or reason.upper().startswith(("CODE_ERROR", "SOLUTION_NULL")):
            raise ValueError(f"Task {task}: invalid stopping reason {reason!r}")

        required = ("objective_value", "original_Prob_Obj_Value", "solution_time_sec")
        for key in required:
            if not math.isfinite(float(new[key])):
                raise ValueError(f"Task {task}: nonfinite {key}: {new[key]!r}")
        old_time = float(old["solution_time_sec"])
        new_time = float(new["solution_time_sec"])

        records.append(dict(
            task_id=task,
            nonparametric_task_id=nonparam_task,
            problem_class=problem,
            instance_filename=str(new["instance_filename"]),
            predictor_type=str(new["predictor_type"]),
            selected_subproblems_ratio=float(new["selected_subproblems_ratio"]),
            nonparametric_stopping_reason=str(nonparam["stopping_reason"]),
            old_stopping_reason=str(old["stopping_reason"]),
            new_stopping_reason=reason,
            nonparametric_runtime_sec=float(nonparam["solution_time_sec"]),
            old_runtime_sec=old_time,
            new_runtime_sec=new_time,
            new_over_old_runtime_ratio=(
                new_time / old_time if old_time > 0 else float("nan")
            ),
            nonparametric_master_obj=float(nonparam["objective_value"]),
            old_master_obj=float(old["objective_value"]),
            new_master_obj=float(new["objective_value"]),
            nonparametric_exact_obj=float(nonparam["original_Prob_Obj_Value"]),
            old_exact_obj=float(old["original_Prob_Obj_Value"]),
            new_exact_obj=float(new["original_Prob_Obj_Value"]),
            **guard,
            nonparametric_excel=str(nonparam_file),
            old_excel=str(old_file),
            new_excel=str(new_file),
        ))

    df = pd.DataFrame(records)
    path = root / "matched_guard_comparison.xlsx"
    df.to_excel(path, index=False)
    cols = [
        "task_id", "problem_class", "nonparametric_runtime_sec",
        "old_runtime_sec", "new_runtime_sec", "nonparametric_exact_obj",
        "old_exact_obj", "new_exact_obj", "guard_checks", "guard_accepted",
        "guard_rejected",
    ]
    print(df[cols].to_string(index=False))
    print(f"PASS: compared {len(records)} matched guarded pilots.")
    print(f"Comparison workbook: {path}")
    if df["guard_checks"].isna().any():
        print(
            "WARNING: Some pilot Excel cells contain truncated iteration JSON; "
            "their guard counts cannot be recovered reliably. Guard configuration "
            "and final performance remain available in the comparison."
        )
    elif df["guard_checks"].sum() == 0:
        print("WARNING: guard was configured, but no guard evaluations were recorded.")
    if df["guard_total_improvement"].isna().all():
        print(
            "NOTE: Original pilot Excel did not preserve aggregate rejected-negative "
            "counts or positive improvement totals; no runtime rerun is required "
            "to recover the recorded acceptance/rejection counts."
        )
    print(
        "The guard guarantees local dual-objective dominance against its own "
        "nonparametric anchor; final runtime and solution quality are empirical."
    )
    return df


if __name__ == "__main__":
    compare()
