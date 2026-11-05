"""Deterministic accounting tests and real operation-path attribution checks."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from gurobipy import GRB
from utils.exclusive_profiler import (
    ExclusiveProfiler, TIMING_CATEGORIES, TIMING_PHASE_TAGS,
    profile_call, reconcile_end_to_end,
)
from utils.convergence_diagnostics import build_solver_bound_trace, verified_relative_gap, write_bound_trace
from Problem.callbacks import Callback
from Dual_Prediction.parametric import (
    ParametricDualPredictor, PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
    solve_parametric_pmedian_selection, _RightHandSideReducer,
)
from Dual_Prediction.training_and_selection import integrated_prediction_selection
from scripts.audit_convergence_pilots import audit_one


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


class TimingAttributionTests(unittest.TestCase):
    def test_exact_nested_durations_and_same_label(self):
        clock = Clock()
        profiler = ExclusiveProfiler(clock=clock)
        with profiler.measure("callback_other_logic"):
            clock.advance(2)
            with profiler.measure("cuts_evaluation_generation"):
                clock.advance(3)
                with profiler.measure("cuts_insertion"):
                    clock.advance(5)
            with profiler.measure("callback_other_logic"):
                clock.advance(7)
        self.assertEqual(profiler.totals["callback_other_logic"], 9)
        self.assertEqual(profiler.totals["cuts_evaluation_generation"], 3)
        self.assertEqual(profiler.totals["cuts_insertion"], 5)
        self.assertEqual(sum(profiler.totals.values()), 17)
        self.assertEqual(profiler.counts["callback_other_logic"], 2)

    def test_exception_records_elapsed_and_restores_context(self):
        clock = Clock()
        profiler = ExclusiveProfiler(clock=clock)
        with self.assertRaises(ValueError):
            with profiler.measure("callback_other_logic"):
                clock.advance(1)
                with profiler.measure("feasibility_check"):
                    clock.advance(2)
                    raise ValueError("intentional")
        self.assertEqual(sum(profiler.totals.values()), 3)
        self.assertEqual(profiler._stack, [])
        profile_call("cuts_insertion", lambda: None)
        self.assertEqual(profiler.count_schema()["cuts_insertion"], 0)

    def test_unknown_labels_and_overlap_are_not_silently_hidden(self):
        profiler = ExclusiveProfiler()
        with profiler.measure("typo"): pass
        with self.assertRaisesRegex(ValueError, "Unknown"):
            profiler.snapshot_schema()
        with self.assertRaisesRegex(ValueError, "overlap"):
            reconcile_end_to_end(5, {"master_solver_internal": 6}, {"master_solver_internal": 1})
        with self.assertRaisesRegex(ValueError, "without an operation"):
            reconcile_end_to_end(5, {"cuts_insertion": 1}, {})
        with self.assertRaisesRegex(ValueError, "counts"):
            reconcile_end_to_end(5, {}, {"cuts_insertion": True})

    def test_reconciliation_records_residual_and_preserves_measured_leaves(self):
        values, counts = reconcile_end_to_end(10, {"master_solver_internal": 4, "input_data_load_validation": 2},
                                             {"master_solver_internal": 1, "input_data_load_validation": 1})
        self.assertEqual(len(values), 33)
        self.assertEqual(sum(values.values()), 10)
        self.assertEqual(values["remaining_unclassified_overhead"], 4)
        self.assertEqual(counts["remaining_unclassified_overhead"], 1)
        self.assertEqual(values["regression_weight_training"], 0)

    def test_exact_phase_attribution_for_both_problem_classes(self):
        for problem in ("SCFLP", "SCMND"):
            cb = Callback.__new__(Callback)
            cb.problem_type = problem
            cb.approach_category = "ML_Augmented_BD"
            cb._phase0_running = False
            cb.n_trials = 5
            cb.iteration = 2
            self.assertEqual(cb._exact_profile_label("algorithm"), "phase1_exact_subproblem_solving")
            cb._phase0_running = True
            self.assertEqual(cb._exact_profile_label("algorithm"), "phase0_exact_subproblem_solving")
            cb._phase0_running = False
            cb.iteration = 6
            self.assertEqual(cb._exact_profile_label("algorithm"), "phase2_selected_exact_subproblem_solving")
            for context in ("fallback", "switch", "phase2_exact"):
                self.assertEqual(cb._exact_profile_label(context), "phase2_fallback_exact_subproblem_solving")
            for baseline in ("Exact", "Random_Sampling"):
                cb.approach_category = baseline
                self.assertEqual(cb._exact_profile_label("algorithm"), "standard_or_random_exact_subproblem_solving")
                self.assertEqual(cb._exact_profile_label("reevaluation"), "all_candidate_expost_reevaluation")

    def test_callback_failure_terminates_and_invalidates_run(self):
        cb = Callback.__new__(Callback)
        cb.exclusive_profiler = ExclusiveProfiler()
        cb.callback_wall_time_sec = 0
        cb._callback_exception = None
        cb._dispatch_callback = Mock(side_effect=ValueError("broken"))
        model = Mock()
        cb(model, GRB.Callback.MIP)
        model.terminate.assert_called_once()
        self.assertGreater(cb.callback_wall_time_sec, 0)
        self.assertEqual(cb.exclusive_profiler.counts["callback_other_logic"], 1)
        with self.assertRaisesRegex(RuntimeError, "run is invalid"):
            cb.raise_if_failed()

    def test_postsolve_nested_logging_does_not_contaminate_callback_totals(self):
        cb = Callback.__new__(Callback)
        cb.problem_type = "SCFLP"
        cb.data = None
        cb.use_stochastic_capacity = False
        cb.exclusive_profiler = ExclusiveProfiler()
        cb.reevaluation_exact_subproblem_time_sec = 0
        cb.reevaluation_exact_subproblem_solves = 0
        def exact(*args, **kwargs):
            profile_call("diagnostics_logging_file_io", lambda: None)
            return (1, {}, {})
        with patch("Problem.callbacks.solve_dual_subproblem", side_effect=exact):
            cb._solve_dual_exact(0, {}, context="reevaluation")
        self.assertEqual(sum(cb.exclusive_profiler.count_schema().values()), 0)
        self.assertEqual(cb.reevaluation_exact_subproblem_solves, 1)

    def test_bound_and_cut_changes_are_not_throttled_to_five_seconds(self):
        from utils.gurobi_cut_diagnostics import GurobiCutStatistics
        cb = Callback.__new__(Callback)
        cb.bound_trace = []
        cb.iteration = 7
        cb._last_bound_pair = None
        cb._submitted_cut_counts = dict.fromkeys(("exact_lazy", "exact_user", "ml_lazy", "ml_user"), 0)
        cb.gurobi_cut_statistics = GurobiCutStatistics()
        model = Mock()
        values = {GRB.Callback.RUNTIME: 1, GRB.Callback.MIP_NODCNT: 0,
                  GRB.Callback.MIP_OBJBND: 10, GRB.Callback.MIP_OBJBST: 11,
                  GRB.Callback.MIP_CUTCNT: 0}
        model.cbGet.side_effect = values.__getitem__
        for second, lb, cuts in ((1, 10, 0), (1.01, 10, 0), (1.02, 10.5, 0),
                                 (1.03, 10.5, 2), (1.04, 10.5, 1)):
            values.update({GRB.Callback.RUNTIME: second, GRB.Callback.MIP_OBJBND: lb,
                           GRB.Callback.MIP_CUTCNT: cuts})
            cb._record_solver_bounds(model)
        self.assertEqual([x["elapsed_seconds"] for x in cb.bound_trace], [1, 1.02, 1.03, 1.04])
        self.assertEqual([x["event"] for x in cb.bound_trace],
                         ["solver_initial", "solver_bound_change", "solver_cut_change", "solver_cut_change"])
        self.assertEqual([x["gurobi_mip_cut_count"] for x in cb.bound_trace], [0, 0, 2, 1])
        self.assertEqual(cb.gurobi_cut_statistics.peak_count, 2)
        self.assertEqual(cb.gurobi_cut_statistics.observation_count, 5)

    def test_solver_residual_rejects_non_nested_intervals(self):
        cb = Callback.__new__(Callback)
        cb.callback_wall_time_sec = 10
        cb.exact_subproblem_time_sec = 4
        with self.assertRaisesRegex(RuntimeError, "not nested"):
            cb.solver_timing_buckets(9)
        cb.exclusive_profiler = ExclusiveProfiler()
        cb.exclusive_profiler.totals.update({"callback_other_logic": 3, "standard_or_random_exact_subproblem_solving": 4,
                                             "all_candidate_expost_reevaluation": 20})
        values = cb.solver_component_breakdown(15)
        self.assertEqual(values["all_candidate_expost_reevaluation"], 0)
        self.assertEqual(values["master_solver_internal"], 5)
        self.assertEqual(values["callback_other_logic"], 6)
        self.assertEqual(sum(values.values()), 15)

    def test_selection_leaves_and_assignment_are_distinct(self):
        features = {i: np.array([float(i), float(i*i)]) for i in range(6)}
        for method, label in (("TREE", "representative_selection_tree"),
                              ("CLUSTERING", "representative_selection_clustering"),
                              ("PMEDIAN", "representative_selection_pmedian")):
            profiler = ExclusiveProfiler()
            with profiler.measure("callback_other_logic"):
                if method == "PMEDIAN":
                    solve_parametric_pmedian_selection(features, 2, {"parametric_pmedian_solver": "GREEDY"})
                else:
                    integrated_prediction_selection(features, "ML_Augmented_BD", method, 2, {})
            counts = profiler.count_schema()
            self.assertGreater(counts[label], 0)
            self.assertGreater(counts["representative_assignment_nearest_weighted"], 0)
            for other in ("representative_selection_tree", "representative_selection_clustering", "representative_selection_pmedian"):
                if other != label: self.assertEqual(counts[other], 0)

    def test_rhs_projection_counts_fitting_and_prediction_transform(self):
        profiler = ExclusiveProfiler()
        reducer = _RightHandSideReducer(1)
        with profiler.measure("callback_other_logic"):
            reducer.fit_transform(np.array([[1., 2.], [3., 4.]]))
            reducer.transform([2., 3.])
        self.assertEqual(profiler.counts["rhs_projection_dimensionality_reduction"], 2)

    def test_guard_fastpath_and_both_repairs_are_actually_measured(self):
        data = SimpleNamespace(I=[0], J=[0], cost=[[20.]], scenarios=[[10.], [10.]], capacity=[20.])
        for mode, label in (("LOCAL", "feasibility_repair_line_segment"),
                            ("STRUCTURE_AWARE", "feasibility_repair_structure_aware")):
            predictor = ParametricDualPredictor(PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR, "SCFLP", data,
                parameters={"parametric_repair_mode": mode, "parametric_objective_guard": True,
                            "parametric_guard_gamma_zero_fastpath": True})
            predictor.selected = [0]
            predictor.mapping = {1: {"representative": 0}}
            predictor.counts = {0: 1.}
            predictor._raw_predictions = lambda ids, model, x: {sid: np.array([22., -3.]) for sid in ids}
            profiler = ExclusiveProfiler()
            with profiler.measure("callback_other_logic"):
                self.assertTrue(predictor.prepare_iteration({0: {"mu": {0: 20.}, "nu": {0: 0.}}}, {0: .2}, [1], 1))
            self.assertGreater(profiler.counts[label], 0)
            self.assertGreater(profiler.totals[label], 0)
            self.assertGreater(profiler.counts["objective_dominance_guard"], 0)
            if mode == "LOCAL": self.assertEqual(predictor.objective_guard_skipped_zero, 1)

    def test_gap_small_zero_and_unavailable_bounds(self):
        self.assertEqual(verified_relative_gap(.25, .5), .5)
        self.assertEqual(verified_relative_gap(0, 0), 0)
        self.assertEqual(verified_relative_gap(-1, 0), float("inf"))
        rows = build_solver_bound_trace([{"elapsed_seconds": 0, "lower_bound": -1e100, "master_upper_bound": 1e100}])
        self.assertIsNone(rows[0]["lower_bound"])
        self.assertIsNone(rows[0]["master_upper_bound"])
        with self.assertRaisesRegex(ValueError, "timestamp"):
            build_solver_bound_trace([{"lower_bound": 12}])

    def test_audit_rejects_zero_counts_and_forged_consistency_flag(self):
        components = dict.fromkeys(TIMING_CATEGORIES, 0.)
        components["standard_or_random_exact_subproblem_solving"] = 8
        components["master_solver_internal"] = 2
        result = {"solution_time_sec": 10, "end_to_end_component_sum_sec": 10,
                  "end_to_end_component_reconciliation_error_sec": 0,
                  "gurobi_solver_runtime_sec": 9, "model_build_and_other_time_sec": 0,
                  "end_to_end_component_breakdown_json": json.dumps(components),
                  "end_to_end_component_counts_json": json.dumps(dict.fromkeys(TIMING_CATEGORIES, 0)),
                  "timing_phase_tags_json": json.dumps(TIMING_PHASE_TAGS),
                  "timing_schema_version": "exclusive_33_v1", "MIPGap_target": .01}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"trace.csv"
            path.write_text("elapsed_seconds,lower_bound,verified_upper_bound,relative_gap,lb_ub_inconsistent\n1,20,10,-1,0\n")
            failures, *_ = audit_one({"method": "Standard_BD"}, result, path, .01, True)
        self.assertTrue(any("zero operation count" in x for x in failures))
        self.assertTrue(any("LB exceeds" in x for x in failures))


if __name__ == "__main__": unittest.main()
