#!/usr/bin/env python3
"""Read-only three-arm comparison of solver-free structure-aware repair.

Uses newly completed matched tasks and preserves original per-task Excel
workbooks. A recovery root may replace only the failed STRUCT arms if the
retry launcher's strict numerical-code-equivalence checks passed.
"""
import csv
import json
import math
import sys
from pathlib import Path
import pandas as pd

RESULTS = Path.home() / "scratch/Benders_Integration_with_ML/RESULTS"
POINTER = RESULTS / "MASTER_RUNS/latest_parametric_structure_repair.txt"
ARMS = ("NONPARAM", "LINE", "STRUCT")
FIELDS = (
    "solution_time_sec", "phase1_execution_time_sec", "phase2_execution_time_sec",
    "iteration", "bnb_nodes", "nodes_explored", "objective_value",
    "original_Prob_Obj_Value", "ObjBound_LB", "cuts_mip_ml", "cuts_rel_ml",
    "cuts_mip_selected", "cuts_rel_selected", "num_predictions",
    "parametric_raw_predictions", "parametric_raw_infeasible_predictions",
    "parametric_mean_repair_gamma", "parametric_repair_time_sec",
    "parametric_feasibility_time_sec", "parametric_prediction_time_sec",
    "parametric_objective_guard_checks", "parametric_objective_guard_accepted",
    "parametric_objective_guard_rejected",
    "parametric_objective_guard_skipped_zero",
    "parametric_objective_guard_skipped_equal_anchor",
    "parametric_objective_guard_evaluated",
    "parametric_objective_guard_time_sec",
    "parametric_objective_guard_total_improvement",
    "parametric_structure_repair_attempts",
    "parametric_structure_repair_feasible_candidates",
    "parametric_structure_repair_anchor_fallbacks",
    "parametric_structure_repair_nonanchor_candidates",
    "parametric_structure_repair_positive_objectives",
    "parametric_structure_repair_positive_gain_total",
    "parametric_algorithm_exact_subproblem_time_sec",
    "parametric_reevaluation_exact_subproblem_time_sec",
)


def number(row, field):
    v = row.get(field)
    if v is None or pd.isna(v):
        return None
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def load(root, task_id, problem):
    files = list(
        (RESULTS / f"{problem}_class" / "RESULTS" / root.name)
        .glob(f"task_{task_id}_*/solution_log.xlsx")
    )
    if len(files) != 1:
        raise ValueError(f"Task {task_id}: expected one workbook, got {files}")
    df = pd.read_excel(files[0])
    if len(df) != 1:
        raise ValueError(f"Task {task_id}: expected one workbook row")
    return df.iloc[0], str(files[0])


def config_check(row, arm, target):
    config = json.loads(row["parametric_config"])
    if config.get("parametric_objective_guard") is not True:
        raise ValueError(f"{target}/{arm}: objective guard must be ON")
    if config.get("parametric_guard_gamma_zero_fastpath") is not True:
        raise ValueError(f"{target}/{arm}: fastpath must be ON")
    if config.get("parametric_diagnostic_probe") is not False:
        raise ValueError(f"{target}/{arm}: diagnostic probe must be OFF")
    expected_mode = "LOCAL" if arm == "LINE" else "STRUCTURE_AWARE"
    if config.get("parametric_repair_mode") != expected_mode:
        raise ValueError(f"{target}/{arm}: expected mode {expected_mode}, got {config.get('parametric_repair_mode')}")
    if row.get("parametric_repair_mode_actual") != expected_mode:
        raise ValueError(f"{target}/{arm}: execution used unexpected repair mode")

    predictions = number(row, "parametric_raw_predictions")
    checks = number(row, "parametric_objective_guard_checks")
    accepted = number(row, "parametric_objective_guard_accepted")
    rejected = number(row, "parametric_objective_guard_rejected")
    evaluated = number(row, "parametric_objective_guard_evaluated")
    skipped = number(row, "parametric_objective_guard_skipped_zero")
    equal_skipped = number(row, "parametric_objective_guard_skipped_equal_anchor")
    if any(v is None for v in (
        predictions, checks, accepted, rejected, evaluated, skipped, equal_skipped,
    )):
        raise ValueError(f"{target}/{arm}: missing guard statistics")
    if not (
        predictions == checks == accepted + rejected
        == evaluated + skipped + equal_skipped
    ):
        raise ValueError(f"{target}/{arm}: inconsistent prediction/guard counts")
    if arm == "STRUCT":
        attempts = number(row, "parametric_structure_repair_attempts")
        fallback = number(row, "parametric_structure_repair_anchor_fallbacks")
        if attempts != predictions or fallback is None:
            raise ValueError(f"{target}/{arm}: missing structure repair statistics")
        if skipped:
            raise ValueError(f"{target}/{arm}: gamma shortcut must not run without a gamma")
    elif equal_skipped:
        raise ValueError(f"{target}/{arm}: equality shortcut is structure-only")


def main():
    if len(sys.argv) > 3:
        raise ValueError("Usage: compare.py ORIGINAL_ROOT [RECOVERY_ROOT]")
    root = Path(sys.argv[1] if len(sys.argv) > 1 else POINTER.read_text().strip())
    recovery = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    recovery_map = {}
    if recovery is not None:
        if (
            (recovery / "comparison_original_root.txt").read_text().strip()
            != str(root)
        ):
            raise ValueError("Recovery root does not identify this original run")
        if (recovery / "config.py").read_bytes() != (root / "config.py").read_bytes():
            raise ValueError("Original and retry root master configs differ")
        with (recovery / "task_pairs.tsv").open() as f:
            for p in csv.DictReader(f, delimiter="\t"):
                if p["method"] != "STRUCT":
                    raise ValueError("Recovery must contain STRUCT arms only")
                target = int(p["pair_parametric_task_id"])
                if target in recovery_map:
                    raise ValueError(f"Duplicate recovered target {target}")
                recovery_map[target] = p
        if set(recovery_map) != {46, 199}:
            raise ValueError(f"Wrong recovery target set: {set(recovery_map)}")
        print("Combining original successful NONPARAM/LINE with fresh STRUCT-only recovery")
        print("Original commit:", (root / "code_commit.txt").read_text().strip())
        print("Recovery commit:", (recovery / "code_commit.txt").read_text().strip())
        print("Comparison is across revisions: validator-only change verified by retry launcher.")

    with (root / "task_pairs.tsv").open() as f:
        pairs = list(csv.DictReader(f, delimiter="\t"))
    rows_by_target = {}
    for p in pairs:
        target = int(p["pair_parametric_task_id"])
        arm = p["method"]
        d = rows_by_target.setdefault(target, {})
        if arm in d:
            raise ValueError(f"Duplicate task {target}/{arm}")
        if arm == "STRUCT" and recovery is not None:
            r = recovery_map[target]
            if p["problem_class"] != r["problem_class"]:
                raise ValueError(f"Problem class mismatch in recovery for {target}")
            d[arm] = load(recovery, int(r["new_task_id"]), r["problem_class"])
        else:
            d[arm] = load(root, int(p["new_task_id"]), p["problem_class"])
    output = []
    print("Matched three-arm comparison: nonparametric / guarded line repair / guarded structure-aware repair")
    for target, group in sorted(rows_by_target.items()):
        if set(group) != set(ARMS):
            raise ValueError(f"Incomplete target {target}: {set(group)}")
        rows = {k: group[k][0] for k in ARMS}
        for key in ("instance_filename", "selected_subproblems_ratio",
                    "num_subproblems", "scmnd_formulation"):
            vals = {str(rows[a].get(key)) for a in ARMS}
            if len(vals) != 1:
                raise ValueError(f"Mismatch at {target}/{key}: {vals}")
        config_check(rows["LINE"], "LINE", target)
        config_check(rows["STRUCT"], "STRUCT", target)
        if str(rows["LINE"].get("predictor_type")) != str(rows["STRUCT"].get("predictor_type")):
            raise ValueError(f"Predictor mismatch at {target}")
        for arm in ARMS:
            stop = str(rows[arm].get("stopping_reason", ""))
            if "ERROR" in stop.upper():
                raise ValueError(f"Error stopping reason at {target}/{arm}: {stop}")
        entry = dict(task_id=target, problem=str(rows["LINE"].get("problem_class")))
        for arm in ARMS:
            for field in FIELDS:
                entry[f"{arm.lower()}_{field}"] = number(rows[arm], field)
            entry[f"{arm.lower()}_stopping_reason"] = str(rows[arm].get("stopping_reason"))
            entry[f"{arm.lower()}_result_path"] = group[arm][1]
        entry["struct_minus_line_runtime_sec"] = (
            number(rows["STRUCT"], "solution_time_sec")
            - number(rows["LINE"], "solution_time_sec")
        )
        entry["struct_runtime_change_pct"] = 100 * (
            number(rows["STRUCT"], "solution_time_sec")
            / number(rows["LINE"], "solution_time_sec") - 1
        )
        old_obj, new_obj = (
            number(rows[a], "original_Prob_Obj_Value") for a in ("LINE", "STRUCT")
        )
        entry["same_original_exact_objective"] = (
            math.isclose(old_obj, new_obj, abs_tol=1e-3, rel_tol=1e-7)
            if old_obj is not None and new_obj is not None else False
        )
        output.append(entry)
        struct = rows["STRUCT"]
        print(f"\nTASK {target} {entry['problem']}:")
        print(
            "  runtimes s: nonparam {:.2f}; line {:.2f}; structure {:.2f} ({:+.2f}%)".format(
                *(number(rows[a], "solution_time_sec") for a in ARMS),
                entry["struct_runtime_change_pct"],
            )
        )
        print(
            "  structure attempts: {:,}; feasible candidate count: {:,}; anchor fallbacks: {:,}".format(
                *(int(number(struct, f)) for f in (
                    "parametric_structure_repair_attempts",
                    "parametric_structure_repair_feasible_candidates",
                    "parametric_structure_repair_anchor_fallbacks",
                ))
            )
        )
        print(f"  structure identical-anchor guard checks skipped: {int(number(struct, 'parametric_objective_guard_skipped_equal_anchor')):,}")
        print(
            "  nonanchor chosen before guard: {:,}; strict positive before guard: {:,}; "
            "guard accepted: {:,}; positive gain before guard: {:.6g}".format(
                int(number(struct, "parametric_structure_repair_nonanchor_candidates")),
                int(number(struct, "parametric_structure_repair_positive_objectives")),
                int(number(struct, "parametric_objective_guard_accepted")),
                number(struct, "parametric_structure_repair_positive_gain_total"),
            )
        )
        print(
            f"  exact final objectives line/struct: {old_obj} / {new_obj}; "
            f"matched={entry['same_original_exact_objective']}"
        )
        print(
            "  iterations line/struct: {} / {}; ML MIP cuts: {} / {}; ML LP cuts: {} / {}".format(
                number(rows["LINE"], "iteration"), number(struct, "iteration"),
                number(rows["LINE"], "cuts_mip_ml"), number(struct, "cuts_mip_ml"),
                number(rows["LINE"], "cuts_rel_ml"), number(struct, "cuts_rel_ml"),
            )
        )
    output_path = (recovery or root) / "matched_structure_repair_comparison.csv"
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"\nMatched problem pairs: {len(output)}")
    print(f"Saved: {output_path}")
    print("This is a pilot, not a general speedup claim. Only new matched runs are compared.")


if __name__ == "__main__":
    main()
