#!/usr/bin/env python3
"""Read-only audit of completed LABD convergence pilots on Nibi.

Checks saved experiment workbooks, gap targets, additive nonnegative timing,
Gurobi master-bound traces, and the separate final original-objective report.
Legacy v1 workbooks retain their original inclusive accounting interpretation. Never submits jobs, edits any experiment, or changes input datasets.

Example:
 python scripts/audit_convergence_pilots.py --pilot-root /scratch/.../labd_convergence_pilot_20261024_014715 --job-id 24905941
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from openpyxl import load_workbook

# Import only the lightweight schema module, not the optimization stack.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.exclusive_profiler import (TIMING_CATEGORIES, TIMING_PHASE_TAGS,
    TIMING_SCHEMA_VERSION, SOLUTION_TIME_METRIC_VERSION, CUT_CONSTRUCTION_TIMING_VERSION)
from utils.convergence_diagnostics import (verified_relative_gap, master_relative_gap,
    TRACE_SCHEMA_VERSION, LEGACY_MASTER_TRACE_VERSION, TRACE_EVENTS, SUBMISSION_FIELDS, _finite)
from utils.gurobi_cut_diagnostics import CUT_COUNT_SCHEMA_VERSION


def number(value):
    try:
        v = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def fmt(value, places=2):
    n = number(value)
    return f"{n:.{places}f}" if n is not None else "N/A"


def read_result(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.active
        rows = ws.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            raise ValueError("Empty workbook")
        keys = [str(x).strip() if x is not None else "" for x in header]
        entries = []
        for row in rows:
            if any(x is not None for x in row):
                entries.append(dict(zip(keys, row)))
        if len(entries) != 1:
            raise ValueError(f"Expected 1 experiment, found {len(entries)}")
        return entries[0]
    finally:
        wb.close()


def read_trace(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit_cut_construction(result, components):
    """Check optional construction details; old workbooks remain auditable."""
    keys = [f"{origin}_cut_construction_{suffix}" for origin in ("exact", "ml")
            for suffix in ("time_sec", "count")]
    version = result.get("cut_construction_timing_version")
    if version in (None, "") and all(result.get(k) in (None, "") for k in keys):
        return []  # Not measured in earlier workbooks; do not infer zero times.
    failures = []
    if version != CUT_CONSTRUCTION_TIMING_VERSION:
        failures.append("wrong/missing cut construction timing version")
    values = {k: number(result.get(k)) for k in keys}
    if any(v is None or v < 0 for v in values.values()):
        return failures + ["invalid/missing cut construction details"]
    for origin in ("exact", "ml"):
        count = values[f"{origin}_cut_construction_count"]
        seconds = values[f"{origin}_cut_construction_time_sec"]
        if count != int(count) or isinstance(result[f"{origin}_cut_construction_count"], bool):
            failures.append("invalid cut construction count")
        if seconds > 0 and count == 0:
            failures.append("cut construction time without a count")
    seconds = sum(values[f"{origin}_cut_construction_time_sec"] for origin in ("exact", "ml"))
    parent = number(components.get("cuts_evaluation_generation"))
    if parent is None or seconds > parent + max(1e-8, abs(parent) * 1e-9):
        failures.append("cut construction details exceed their parent timing category")
    try:
        counts = json.loads(result.get("end_to_end_component_counts_json") or "{}")
        submitted = number(counts.get("cuts_insertion"))
    except (ValueError, TypeError, AttributeError):
        submitted = None
    if submitted is None or sum(values[f"{origin}_cut_construction_count"] for origin in ("exact", "ml")) != submitted:
        failures.append("cut construction counts do not match insertion calls")
    return failures


def _count(value):
    n = number(value)
    return int(n) if n is not None and n >= 0 and n == int(n) and not isinstance(value, bool) else None


def audit_event_trace(trace, result):
    """Validate complete callback rounds and distinguish solver counts from submissions."""
    failures = []
    previous_pair, previous_submitted = None, dict.fromkeys(SUBMISSION_FIELDS, 0)
    previous_cut_count = None
    iteration, open_round, mip_counts, mip_times = 0, None, [], []
    for row in trace:
        event, context = row.get('event'), row.get('callback_context')
        it = _count(row.get('iteration'))
        if event not in TRACE_EVENTS or context not in {'MIP', 'MIPSOL', 'MIPNODE', 'TERMINAL'}:
            failures.append('invalid event/context in bound trace')
        if event == 'solver_iteration_start':
            if open_round is not None or it != iteration + 1 or context not in {'MIPSOL', 'MIPNODE'}:
                failures.append('invalid iteration-start record')
            iteration = it if it is not None else iteration
            open_round = (it, context)
        elif event == 'solver_iteration_end':
            if open_round != (it, context):
                failures.append('unpaired iteration-end record')
            open_round = None
        elif it != iteration:
            failures.append('iteration changed without a start record')
        pair = (_finite(row.get('lower_bound')), _finite(row.get('master_upper_bound')))
        changed = int(pair != previous_pair)
        if _count(row.get('bound_changed')) != changed:
            failures.append('bound-change flag disagrees with observations')
        previous_pair = pair
        if event == 'solver_bound_change' and not changed:
            failures.append('bound-change event without changed bounds')
        submitted = {key: _count(row.get(key)) for key in SUBMISSION_FIELDS}
        if any(n is None for n in submitted.values()):
            failures.append('missing/invalid cumulative cut submission count')
        else:
            if submitted['submitted_exact_cuts'] + submitted['submitted_ml_cuts'] != submitted['submitted_lazy_cuts'] + submitted['submitted_user_cuts']:
                failures.append('cut submission partitions disagree')
            if any(submitted[k] < previous_submitted[k] for k in SUBMISSION_FIELDS):
                failures.append('cut submission count decreased')
            previous_submitted = submitted
        cut = _count(row.get('gurobi_mip_cut_count'))
        cut_changed = _count(row.get('gurobi_cut_count_changed'))
        if context == 'MIP':
            if cut is None or cut_changed != int(cut != previous_cut_count):
                failures.append('invalid Gurobi cut-count observation/change flag')
            if cut is not None:
                mip_counts.append(cut); mip_times.append(number(row.get('elapsed_seconds')))
            previous_cut_count = cut
        elif row.get('gurobi_mip_cut_count') not in (None, '') or cut_changed != 0:
            failures.append('Gurobi MIP cut count queried/carried outside MIP context')
        if event == 'solver_cut_change' and (not cut_changed or changed):
            failures.append('invalid cut-only change event')
    if open_round is not None or _count(result.get('iteration')) != iteration:
        failures.append('missing callback iteration boundary records')
    if not trace or trace[-1].get('callback_context') != 'TERMINAL':
        failures.append('missing terminal trace context')
    for origin in ('exact', 'ml'):
        expected = _count(result.get(f'{origin}_cut_construction_count'))
        if expected != previous_submitted[f'submitted_{origin}_cuts']:
            failures.append(f'terminal {origin} submission count disagrees with workbook')
    if result.get('gurobi_cut_count_schema_version') != CUT_COUNT_SCHEMA_VERSION:
        failures.append('wrong/missing Gurobi cut-count schema version')
    observations = _count(result.get('gurobi_mip_cut_count_observation_count'))
    last = _count(result.get('gurobi_mip_cut_count_last_observed'))
    peak = _count(result.get('gurobi_mip_cut_count_peak_observed'))
    stamp = number(result.get('gurobi_mip_cut_count_last_observed_time_sec'))
    runtime = number(result.get('gurobi_solver_runtime_sec'))
    if mip_counts:
        if observations is None or observations < len(mip_counts) or last != mip_counts[-1] or peak != max(mip_counts):
            failures.append('Gurobi cut-count summary disagrees with trace')
        if stamp is None or mip_times[-1] is None or stamp < mip_times[-1] - 1e-8 or runtime is None or stamp > runtime + .02:
            failures.append('invalid last-observed Gurobi cut-count timestamp')
    elif observations != 0 or any(v is not None for v in (last, peak, stamp)):
        failures.append('Gurobi cut-count summary without MIP observations')
    status = result.get('gurobi_last_lp_cut_summary_status')
    fields = ['gurobi_last_lp_total_cut_count', 'gurobi_last_lp_user_cut_count', 'gurobi_last_lp_lazy_constraint_count']
    if status == 'reported':
        try:
            counts = json.loads(result.get('gurobi_last_lp_cut_counts_json') or 'null')
            if not isinstance(counts, dict) or any(_count(v) is None for v in counts.values()):
                raise ValueError('invalid counts')
            expected = [sum(counts.values()), counts.get('User', 0), counts.get('Lazy constraints', 0)]
            if [_count(result.get(k)) for k in fields] != expected:
                failures.append('last-LP cut summary fields disagree')
        except (ValueError, TypeError):
            failures.append('invalid last-LP cut summary JSON')
    elif status in {'not_reported', 'incomplete'}:
        if any(result.get(k) not in (None, '') for k in fields + ['gurobi_last_lp_cut_counts_json']):
            failures.append('unavailable last-LP cut summary filled with counts')
    else:
        failures.append('missing/invalid last-LP cut-summary status')
    return sorted(set(failures))


def audit_one(record, result, trace_path, target, require_33=False, require_current=False):
    failures = []
    warnings = []
    version = result.get("timing_schema_version")
    current = version == TIMING_SCHEMA_VERSION
    if require_current and not current:
        failures.append("current final-only reporting policy required")
    solution_time = number(result.get("solution_time_sec"))
    t = number(result.get("end_to_end_wall_time_sec")) if current else solution_time
    t_sum = number(result.get("end_to_end_component_sum_sec"))
    t_error = number(result.get("end_to_end_component_reconciliation_error_sec"))
    gurobi_time = number(result.get("gurobi_solver_runtime_sec"))
    if t is None or t <= 0:
        failures.append("missing end-to-end wall time")
    if t_sum is None or (t is not None and abs(t_sum - t) > max(0.02, 1e-4 * t)):
        failures.append("timing sum != inclusive wall time")
    if t_error is None or abs(t_error) > 0.02:
        failures.append("timing reconciliation error")
    if gurobi_time is None or gurobi_time < 0:
        failures.append("missing Gurobi-only time")
    if t is not None and gurobi_time is not None and gurobi_time > t + 1.0:
        failures.append("Gurobi time exceeds end-to-end solution time")
    residual = number(result.get("model_build_and_other_time_sec"))
    if residual is None:
        failures.append("missing unclassified timing remainder")
    elif residual < -0.02:
        failures.append(f"NEGATIVE timing remainder {residual:.3f}s (profiling overlap)")
    elif t is not None and residual > max(10.0, 0.1 * t):
        warnings.append(f"large unclassified timing remainder {residual:.1f}s")
    try:
        components = json.loads(result.get("end_to_end_component_breakdown_json") or "{}")
        if not isinstance(components, dict) or not components:
            raise ValueError("not a populated dictionary")
        failures.extend(audit_cut_construction(result, components))
        if require_33:
            expected = set(TIMING_CATEGORIES)
            missing = sorted(expected - set(components))
            extra = sorted(set(components) - expected)
            if missing or extra:
                failures.append(
                    f"33-category schema mismatch: missing={missing}, extra={extra}"
                )
            if version not in {"exclusive_33_v1", TIMING_SCHEMA_VERSION}:
                failures.append("wrong/missing timing schema version")
            try:
                phase_tags = json.loads(result.get("timing_phase_tags_json") or "{}")
                if phase_tags != TIMING_PHASE_TAGS:
                    failures.append("33-category phase-tag mapping mismatch")
            except (ValueError, TypeError, json.JSONDecodeError):
                failures.append("invalid phase-tag JSON")
            counts = {}
            try:
                counts = json.loads(
                    result.get("end_to_end_component_counts_json") or "{}"
                )
                if not isinstance(counts, dict) or set(counts) != expected:
                    failures.append("operation-count schema mismatch")
                elif any(
                    type(v) is not int or v < 0 for v in counts.values()
                ):
                    failures.append("invalid 33-category operation counts")
            except (ValueError, TypeError, json.JSONDecodeError):
                failures.append("invalid 33-category operation-count JSON")
            if isinstance(counts, dict) and set(counts) == expected:
                for category, value in components.items():
                    if number(value) is not None and float(value) > 0 and counts.get(category, 0) == 0:
                        failures.append(f"nonzero {category} has zero operation count")
                exact_categories = [k for k in expected if "exact_subproblem_solving" in k]
                if _finite(result.get("objective_value")) is not None and not any(type(counts.get(k)) is int and counts[k] > 0 for k in exact_categories):
                    failures.append("no measured exact scenario solves")
                method = record.get("method", "")
                if method in {"Standard_BD", "Random_Sampling"}:
                    forbidden = list(TIMING_CATEGORIES[9:19]) + list(TIMING_CATEGORIES[20:25])
                    if method == "Standard_BD":
                        forbidden.append("nonparametric_dual_transfer")
                    for category in forbidden:
                        if counts.get(category, 0) or number(components.get(category)) != 0:
                            failures.append(f"{method} unexpectedly measured {category}")
                phase2 = sum(counts.get(k, 0) for k in (
                    "phase2_selected_exact_subproblem_solving", "phase2_fallback_exact_subproblem_solving"
                ) if type(counts.get(k, 0)) is int)
                if phase2 and method not in {"Standard_BD", "Random_Sampling"}:
                    required = ["dual_trajectory_processing", "graph_random_walk_generation", "embedding_skipgram_training"]
                    selector = ("representative_selection_tree" if "TREE" in method else
                                "representative_selection_clustering" if "CLUSTER" in method else
                                "representative_selection_pmedian")
                    required += [selector, "representative_assignment_nearest_weighted"]
                    if method.startswith("PARAMETRIC_"):
                        required += ["regression_weight_training", "parametric_residual_prediction", "feasibility_check"]
                    for category in required:
                        if not counts.get(category, 0):
                            failures.append(f"phase 2 ran without measured {category}")
                if t is not None and t > 0:
                    other_callback = number(components.get("callback_other_logic"))
                    if other_callback is not None and other_callback > 0.1 * t:
                        warnings.append(f"callback residual {other_callback/t:.1%} of inclusive time")
            optimize_wall = number(result.get("master_optimize_wall_time_sec"))
            solver_sum = number(result.get("solver_component_sum_sec"))
            if optimize_wall is None or optimize_wall < 0:
                failures.append("missing measured optimize wall interval")
            elif solver_sum is None or abs(solver_sum-optimize_wall) > max(1e-6, 1e-9*optimize_wall):
                failures.append("solver components do not reconcile to optimize wall interval")
            if t is not None and t > 0:
                other = number(components.get("remaining_unclassified_overhead"))
                if other is None or other < 0:
                    failures.append("invalid remaining_unclassified_overhead")
                elif other > 0.10 * t:
                    failures.append(
                        f"remaining unclassified {other/t:.1%} exceeds 10% limit"
                    )
        bad = [(k, v) for k, v in components.items()
               if number(v) is None or number(v) < 0]
        if bad:
            failures.append(f"invalid or negative timing categories: {bad[:3]}")
        component_total = sum(float(v) for v in components.values())
        if t is not None and abs(component_total - t) > max(0.02, 1e-4 * t):
            failures.append("JSON timing components do not reconcile")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        failures.append(f"bad timing JSON ({exc})")
    achieved_target = number(result.get("MIPGap_target"))
    if achieved_target is None or abs(achieved_target - target) > 1e-8:
        failures.append(f"configured Gurobi gap {achieved_target} != {target}")
    final_gurobi_gap = number(result.get("MIPGap_Stop"))
    if final_gurobi_gap is None:
        warnings.append("final Gurobi gap unavailable")
    elif final_gurobi_gap > target * 1.01 + 1e-7:
        warnings.append(f"Gurobi stopped above target (gap={final_gurobi_gap:.5g})")
    certified = number(result.get("verified_final_relative_gap"))
    if certified is None:
        warnings.append("final original-objective gap unavailable")
    elif certified > target * 1.01 + 1e-7:
        warnings.append(f"verified original gap above target ({certified:.5g})")
    elif certified < -1e-5:
        failures.append(f"negative verified gap ({certified:.5g})")
    if trace_path is None:
        failures.append("convergence CSV not present")
    else:
        try:
            trace = read_trace(trace_path)
            if not trace:
                failures.append("empty convergence CSV")
            else:
                lb = [number(x.get("lower_bound")) for x in trace]
                ub_key = "master_upper_bound" if current else "verified_upper_bound"
                gap_key = "master_relative_gap" if current else "relative_gap"
                ub = [_finite(x.get(ub_key)) for x in trace]
                if current:
                    trace_version = result.get("convergence_trace_schema_version")
                    if any(x.get("trace_schema_version") != trace_version for x in trace):
                        failures.append("wrong/missing Gurobi master trace schema")
                    events = TRACE_EVENTS if trace_version == TRACE_SCHEMA_VERSION else {"solver", "solver_final"}
                    if any(x.get("event") not in events for x in trace):
                        failures.append("non-solver event in Gurobi bound history")
                    if trace_version == TRACE_SCHEMA_VERSION:
                        failures.extend(audit_event_trace(trace, result))
                    if any("verified_upper_bound" in x for x in trace):
                        failures.append("retrospective UB column in current trace")
                    terminal = trace[-1]
                    if terminal.get("event") != "solver_final":
                        failures.append("missing terminal Gurobi observation")
                    for field, workbook_field in (("master_upper_bound", "objective_value"), ("lower_bound", "ObjBound_LB")):
                        observed, expected = _finite(terminal.get(field)), _finite(result.get(workbook_field))
                        if expected is not None and (observed is None or abs(observed-expected) > 1e-8*max(1, abs(expected))):
                            failures.append(f"terminal {field} disagrees with workbook")
                    terminal_time = number(terminal.get("elapsed_seconds"))
                    if gurobi_time is not None and (terminal_time is None or abs(terminal_time-gurobi_time) > .02):
                        failures.append("terminal timestamp differs from Gurobi runtime")
                times = [number(x.get("elapsed_seconds")) for x in trace]
                if any(n is not None and n < -1e-5 for n in times):
                    failures.append("negative convergence timestamp")
                if any(a is not None and b is not None and b + 1e-5 < a
                       for a, b in zip(times, times[1:])):
                    failures.append("nonmonotone convergence timestamps")
                if not any(n is not None for n in lb):
                    warnings.append("no solver lower-bound samples")
                if not any(n is not None for n in ub):
                    warnings.append("no upper-bound samples for this trace schema")
                for row, lower, upper in zip(trace, lb, ub):
                    if lower is not None and upper is not None:
                        if lower > upper + 1e-6 * max(1.0, abs(upper)):
                            failures.append("solver LB exceeds UB in CSV")
                            break
                        gap = number(row.get(gap_key))
                        expected_gap = (master_relative_gap if current else verified_relative_gap)(lower, upper)
                        if math.isfinite(expected_gap) and (gap is None or abs(gap-expected_gap) > 1e-8):
                            failures.append("CSV gap disagrees with bounds")
                            break
                if any(n is None for n in times):
                    failures.append("missing/nonfinite convergence timestamp")
                if gurobi_time is not None and any(n is not None and n > gurobi_time + 0.02 for n in times):
                    failures.append("trace extends beyond Gurobi solver time")
                if any(str(x.get("lb_ub_inconsistent", "0")) == "1" for x in trace):
                    failures.append("LB exceeds UB at some point")
        except (OSError, ValueError, csv.Error) as exc:
            failures.append(f"cannot parse convergence CSV: {exc}")
    if current:
        final_time = number(result.get("final_exact_evaluation_time_sec"))
        legacy_final_time = number(result.get("final_objective_validation_time_sec"))
        if result.get("solution_time_metric_version") != SOLUTION_TIME_METRIC_VERSION:
            failures.append("wrong/missing solution-time metric version")
        supported_trace_versions = {TRACE_SCHEMA_VERSION} if require_current else {TRACE_SCHEMA_VERSION, LEGACY_MASTER_TRACE_VERSION}
        if result.get("convergence_trace_schema_version") not in supported_trace_versions:
            failures.append("wrong/missing workbook trace version")
        if solution_time is None or solution_time <= 0 or final_time is None or final_time < 0:
            failures.append("invalid algorithm or final-reporting time")
        elif t is None or abs(solution_time + final_time - t) > max(1e-6, 1e-9*t):
            failures.append("solution time includes reporting or does not partition wall time")
        if final_time is None or legacy_final_time != final_time:
            failures.append("final reporting time aliases disagree")
        solution_sum = number(result.get("solution_component_sum_sec"))
        if solution_sum is None or solution_time is None or abs(solution_sum-solution_time) > 1e-6:
            failures.append("solution components do not reconcile")
        method = result.get("method_category")
        exact = method == "Exact" or record.get("method") in {"Standard_BD", "STANDARD_BD"}
        has_incumbent = _finite(result.get("objective_value")) is not None
        expected_passes = int(has_incumbent and not exact)
        if number(result.get("final_exact_evaluation_count")) != expected_passes:
            failures.append("wrong number of final decision evaluations")
        expected_solves = number(result.get("num_subproblems")) if expected_passes else 0
        if expected_solves is None or number(result.get("final_exact_subproblem_solves")) != expected_solves:
            failures.append("wrong number of final scenario solves")
        status = result.get("final_exact_evaluation_status")
        expected_status = "no_incumbent" if not has_incumbent else ("reused_gurobi_objective" if exact else "evaluated")
        if status != expected_status:
            failures.append("final original-objective report failed or has wrong status")
        original = _finite(result.get("original_Prob_Obj_Value"))
        objective = _finite(result.get("objective_value"))
        if has_incumbent and (original is None or (exact and original != objective)):
            failures.append("missing/incorrect final true objective")
        if exact and final_time != 0:
            failures.append("standard Benders incurred final-reporting time")
        for key in ("candidate_reevaluation_time_sec", "num_solutions_reevaluated"):
            if number(result.get(key)) != 0:
                failures.append("historical candidate reevaluation is not disabled")
        try:
            components = json.loads(result.get("end_to_end_component_breakdown_json") or "{}")
            counts = json.loads(result.get("end_to_end_component_counts_json") or "{}")
            if components.get("all_candidate_expost_reevaluation") != 0 or counts.get("all_candidate_expost_reevaluation") != 0:
                failures.append("retired reevaluation timing/count must be zero")
            measured_final = number(components.get("terminal_exact_original_objective_validation"))
            if (measured_final is None or final_time is None
                    or not math.isclose(measured_final, final_time, rel_tol=1e-12, abs_tol=1e-9)
                    or counts.get("terminal_exact_original_objective_validation") != expected_passes):
                failures.append("final reporting component/count disagrees")
        except (ValueError, TypeError):
            failures.append("invalid final-reporting component JSON")
        if (record.get("problem") or result.get("problem_class")) == "SCMND" and result.get("scmnd_formulation") != "MANUSCRIPT_ENVELOPE":
            failures.append("current SCMND campaign must use hard capacity")
        lower = _finite(result.get("ObjBound_LB"))
        if lower is not None and original is not None:
            expected = verified_relative_gap(lower, original)
            if math.isfinite(expected) and (certified is None or abs(expected-certified) > 1e-8):
                failures.append("final verified gap disagrees with final true objective")
    return failures, warnings, solution_time, gurobi_time, certified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", type=Path, required=True)
    ap.add_argument("--job-id", required=True, help="Slurm array job ID")
    ap.add_argument("--require-33", action="store_true",
                    help="enforce complete v1/v2 exclusive timing and operation-count schema")
    ap.add_argument("--require-current", action="store_true", help="require final-only v2 reporting and hard SCMND")
    ap.add_argument("--detail", action="store_true",
                    help="show solver stop reason, both gaps and exclusive time components")
    args = ap.parse_args()
    root = args.pilot_root.expanduser().resolve()
    manifest = root / "pilot_manifest.tsv"
    if not manifest.is_file():
        ap.error(f"pilot manifest missing: {manifest}")
    if not args.job_id.isdigit():
        ap.error("--job-id must be digits")
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 20 or sorted(int(r["task_id"]) for r in rows) != list(range(1, 21)):
        ap.error("expected exactly 20 tasks with IDs 1..20")
    results_root = root.parents[1]
    run_id = root.name
    failed_tasks = []
    all_warnings = []
    print(f"Auditing {run_id}, job {args.job_id}; source workbooks are read-only.")
    print("ID  Gap    Problem Method                            Solve(s) Gurobi(s) FinalGap  Status")
    print("-" * 102)
    for rec in rows:
        tid = int(rec["task_id"])
        problem = rec["problem"]
        # SLURM_JOB_ID is the array ELEMENT job ID on Nibi, while
        # --job-id refers to the ARRAY parent. Discover each element's
        # independent task_<array_index>_<element_job_id> folder.
        class_root = results_root / f"{problem}_class" / "RESULTS" / run_id
        candidates = sorted(
            path for path in class_root.glob(f"task_{tid}_*")
            if path.is_dir()
            and path.name.startswith(f"task_{tid}_")
            and path.name[len(f"task_{tid}_"):].isdigit()
        )
        expected_gap = number(rec["gap"])
        failures, warnings = [], []
        total = solver = verified = None
        result = None
        if not candidates:
            failures = [
                f"no task_{tid}_<element job id> directory under {class_root}"
            ]
        elif len(candidates) > 1:
            failures = [
                f"ambiguous task directories for {tid}: "
                + ", ".join(str(x) for x in candidates)
            ]
        else:
            out_dir = candidates[0]
            workbook = out_dir / "solution_log.xlsx"
            if not workbook.is_file():
                failures = [f"missing solution_log.xlsx: {workbook}"]
            else:
                try:
                    result = read_result(workbook)
                    filename = result.get("convergence_trace_path")
                    trace_path = None
                    if isinstance(filename, str) and filename.strip():
                        candidate = Path(filename)
                        trace_path = candidate if candidate.is_absolute() else out_dir / candidate
                        if not trace_path.is_file():
                            trace_path = None
                    failures, warnings, total, solver, verified = audit_one(
                        rec, result, trace_path, expected_gap, args.require_33 or args.require_current, args.require_current
                    )
                    status = str(result.get("stopping_reason") or "N/A")
                    if status.startswith("CODE_ERROR") or status == "SOLUTION_NULL":
                        failures.append(f"optimization status {status}")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    failures = [f"cannot read workbook: {exc}"]
        label = "FAIL" if failures else ("WARN" if warnings else "OK")
        print(f"{tid:2d}  {expected_gap:<6g} {problem:<7s} {rec['method'][:33]:<33s} "
              f"{fmt(total, 0):>8s}  {fmt(solver, 0):>8s}  "
              f"{fmt(verified, 5):>9s}  {label}")
        for msg in failures:
            print("      FAIL: " + msg)
        for msg in warnings:
            print("      WARN: " + msg)
        if args.detail and result is not None:
            print(
                "      STOP:", result.get("stopping_reason"),
                "master MIP gap:", fmt(result.get("MIPGap_Stop"), 6),
                "target:", fmt(result.get("MIPGap_target"), 6),
                "final evaluations:", result.get("final_exact_evaluation_count"),
            )
            print(
                "      TIME sec:",
                "load=", fmt(result.get("data_loading_time_sec"), 1),
                "final validation=", fmt(result.get("final_objective_validation_time_sec"), 1),
                "inclusive total=", fmt(result.get("end_to_end_wall_time_sec", result.get("solution_time_sec")), 1),
                "unclassified=", fmt(result.get("model_build_and_other_time_sec"), 1),
            )
            try:
                times = json.loads(result.get("end_to_end_component_breakdown_json") or "{}")
                tracked = [
                    (key, number(value)) for key, value in times.items()
                    if number(value) is not None
                ]
                prominent = sorted(tracked, key=lambda kv: kv[1], reverse=True)[:6]
                print(
                    "      TOP COMPONENTS:",
                    ", ".join(f"{name}={value:.1f}s" for name, value in prominent)
                )
                counts = json.loads(result.get("end_to_end_component_counts_json") or "{}")
                for name in TIMING_CATEGORIES:
                    value = number(times.get(name))
                    if value is not None and (value or counts.get(name, 0)):
                        denominator = number(result.get("end_to_end_wall_time_sec", total))
                        share = 100 * value / denominator if denominator else 0
                        print(f"        {name}: {value:.6f}s ({share:.2f}%), count={counts.get(name, 'missing')}")
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if failures:
            failed_tasks.append(tid)
        if warnings:
            all_warnings.append(tid)
    print("-" * 102)
    print(f"Audited {len(rows)} runs; FAILED={len(failed_tasks)}; WARNED={len(all_warnings)}.")
    print("Warnings for verified gap mean Gurobi's master gap is not automatically")
    print("a certificate on the true stochastic objective. Review before final sweep.")
    print("The timings exclude result-file export by the existing accounting convention.")
    if failed_tasks:
        print("Failing task IDs:", ", ".join(map(str, failed_tasks)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
