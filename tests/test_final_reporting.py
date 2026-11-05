"""Behavioral regressions for final-only reporting and manuscript-only SCMND."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from Problem.master_problem import _final_objective_report
from Problem.callbacks import Callback
from scripts.prepare_convergence_pilots import hard_capacity_override
from scripts.audit_convergence_pilots import audit_one
from utils.exclusive_profiler import (
    solution_time_excluding_reporting, TIMING_CATEGORIES, TIMING_PHASE_TAGS,
    TIMING_SCHEMA_VERSION, SOLUTION_TIME_METRIC_VERSION,
)
from utils.convergence_diagnostics import build_solver_bound_trace, write_bound_trace, TRACE_SCHEMA_VERSION


class FinalReportingTests(unittest.TestCase):
    def test_exact_modes_never_call_postsolve_evaluator(self):
        for mode in (False, "none", "int_exact", "int_and_LP_ML_for_none"):
            with self.subTest(mode=mode), patch("Problem.master_problem._calculate_true_objective_value") as evaluate:
                report = _final_objective_report("SCFLP", None, {0: 1.}, 17.123456789, mode)
                evaluate.assert_not_called()
                self.assertEqual(report["original_Prob_Obj_Value"], 17.123456789)
                self.assertEqual(report["final_objective_validation_time_sec"], 0)
                self.assertEqual(report["final_exact_evaluation_count"], 0)
                self.assertEqual(report["final_exact_subproblem_solves"], 0)

    def test_inexact_evaluates_one_returned_decision_once_per_scenario(self):
        data = SimpleNamespace(A=[(0, 1)], S=[0, 1, 2], F=[7.], probabilities=[.2, .3, .5])
        returned = {0: 1.}
        with patch("Problem.subproblem_dual.solve_dual_subproblem", side_effect=[(10., {}, {}), (20., {}, {}), (30., {}, {})]) as exact:
            report = _final_objective_report("SCMND", data, returned, 1., True)
        self.assertEqual(exact.call_count, 3)
        self.assertEqual([call.args[2] for call in exact.call_args_list], [0, 1, 2])
        self.assertTrue(all(call.args[3] is returned for call in exact.call_args_list))
        self.assertAlmostEqual(report["original_Prob_Obj_Value"], 30.)
        self.assertEqual(report["final_exact_evaluation_count"], 1)
        self.assertEqual(report["final_exact_subproblem_solves"], 3)
        self.assertEqual(report["final_exact_evaluation_status"], "evaluated")
        self.assertGreater(report["final_objective_validation_time_sec"], 0)

    def test_no_incumbent_does_not_evaluate(self):
        with patch("Problem.master_problem._calculate_true_objective_value") as evaluate:
            report = _final_objective_report("SCFLP", None, None, np.inf, True)
        evaluate.assert_not_called()
        self.assertIsNone(report["original_Prob_Obj_Value"])
        self.assertEqual(report["final_exact_evaluation_count"], 0)
        self.assertEqual(report["final_objective_validation_time_sec"], 0)
        self.assertEqual(report["final_exact_evaluation_status"], "no_incumbent")

    def test_failed_evaluation_cannot_look_like_a_true_objective(self):
        data = SimpleNamespace(A=[(0, 1)], S=[0], F=[7.], probabilities=[1.])
        with patch("Problem.subproblem_dual.solve_dual_subproblem", side_effect=RuntimeError("infeasible recourse")):
            report = _final_objective_report("SCMND", data, {0: 1.}, 10., True)
        self.assertIsNone(report["original_Prob_Obj_Value"])
        self.assertEqual(report["final_exact_evaluation_status"], "failed")
        self.assertEqual(report["final_exact_subproblem_solves"], 1)

    def test_solution_time_excludes_only_final_reporting(self):
        self.assertEqual(solution_time_excluding_reporting(150, 40), 110)
        self.assertEqual(solution_time_excluding_reporting(150, 0), 150)
        for total, final in [(1, 2), (1, -1), (np.inf, 2)]:
            with self.assertRaises(ValueError):
                solution_time_excluding_reporting(total, final)

    def test_plot_uses_master_values_even_for_old_workbooks(self):
        from scripts.plot_labd_convergence import rows_from
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"legacy.csv"
            path.write_text("elapsed_seconds,event,lower_bound,master_incumbent_proxy,verified_upper_bound,relative_gap\n"
                            "0,solver,10,11,99,0.9\n1,verified_candidate,10,11,20,0.5\n")
            self.assertEqual(rows_from(path, "time", "master_upper_bound"), [(0., 11.)])
            self.assertEqual(rows_from(path, "time", "master_relative_gap"), [(0., 1/11)])

    def test_historical_candidate_evaluator_is_removed(self):
        self.assertFalse(hasattr(Callback, "reevaluate_integer_solutions"))
        self.assertFalse(hasattr(Callback, "_snapshot_integer_solution"))

    def test_pilot_copy_changes_only_formulation_and_preserves_original(self):
        source = '# historical\nscmnd_formulation_grid = ["CURRENT_SOFT_CAPACITY"]\nother = 42\n'
        copied = hard_capacity_override(source, "SCMND")
        self.assertEqual(copied, source.replace('CURRENT_SOFT_CAPACITY', 'MANUSCRIPT_ENVELOPE'))
        self.assertIn('CURRENT_SOFT_CAPACITY', source)
        self.assertEqual(hard_capacity_override(source, "SCFLP"), source)
        self.assertEqual(hard_capacity_override(copied, "SCMND"), copied)
        with self.assertRaises(ValueError):
            hard_capacity_override('other = 42\n', "SCMND")

    def test_audit_enforces_time_partition_counts_and_master_trace(self):
        components = dict.fromkeys(TIMING_CATEGORIES, 0.)
        counts = dict.fromkeys(TIMING_CATEGORIES, 0)
        components.update(standard_or_random_exact_subproblem_solving=8., master_solver_internal=2.,
                          terminal_exact_original_objective_validation=3.)
        counts.update(standard_or_random_exact_subproblem_solving=8, master_solver_internal=1,
                      terminal_exact_original_objective_validation=1)
        row = dict(solution_time_sec=10., end_to_end_wall_time_sec=13., solution_component_sum_sec=10.,
                   end_to_end_component_sum_sec=13., end_to_end_component_reconciliation_error_sec=0.,
                   gurobi_solver_runtime_sec=10., master_optimize_wall_time_sec=10., solver_component_sum_sec=10.,
                   model_build_and_other_time_sec=0., final_objective_validation_time_sec=3.,
                   final_exact_evaluation_time_sec=3., final_exact_evaluation_count=1, final_exact_subproblem_solves=8,
                   final_exact_evaluation_status="evaluated", num_solutions_reevaluated=0, candidate_reevaluation_time_sec=0.,
                   num_subproblems=8, method_category="ML", problem_class="SCMND", scmnd_formulation="MANUSCRIPT_ENVELOPE",
                   objective_value=11., original_Prob_Obj_Value=12., ObjBound_LB=10., verified_final_relative_gap=2/12,
                   MIPGap_target=.01, MIPGap_Stop=1/11, timing_schema_version=TIMING_SCHEMA_VERSION,
                   solution_time_metric_version=SOLUTION_TIME_METRIC_VERSION, convergence_trace_schema_version=TRACE_SCHEMA_VERSION,
                   end_to_end_component_breakdown_json=json.dumps(components), end_to_end_component_counts_json=json.dumps(counts),
                   timing_phase_tags_json=json.dumps(TIMING_PHASE_TAGS))
        from utils.gurobi_cut_diagnostics import GurobiCutStatistics
        row.update(GurobiCutStatistics().export())
        row.update(iteration=0, exact_cut_construction_count=0, ml_cut_construction_count=0)
        row.update(exact_cut_construction_time_sec=0, ml_cut_construction_time_sec=0,
                   cut_construction_timing_version="exact_ml_construction_v1")
        trace = build_solver_bound_trace([dict(elapsed_seconds=10, event="solver_final", lower_bound=10., master_upper_bound=11.,
            iteration=0, callback_context="TERMINAL", bound_changed=1, gurobi_cut_count_changed=0,
            submitted_exact_cuts=0, submitted_ml_cuts=0, submitted_lazy_cuts=0, submitted_user_cuts=0)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'trace.csv'
            write_bound_trace(trace, path)
            def failures(result):
                return audit_one({'method': 'Random_Sampling', 'problem': 'SCMND'}, result, path, .01, True, True)[0]
            self.assertEqual(failures(row), [])
            for bad in (dict(solution_time_sec=13.), dict(final_exact_evaluation_count=2),
                        dict(final_exact_subproblem_solves=16), dict(candidate_reevaluation_time_sec=1.),
                        dict(scmnd_formulation="CURRENT_SOFT_CAPACITY"), dict(objective_value=12.)):
                with self.subTest(bad=bad):
                    self.assertTrue(failures(dict(row, **bad)))

            # Existing v2 workbooks remain auditable without the current-schema gate.
            legacy_row = dict(row, convergence_trace_schema_version='gurobi_master_v2')
            legacy_fields = ['elapsed_seconds', 'iteration', 'node_count', 'event', 'lower_bound',
                             'master_upper_bound', 'master_relative_gap', 'lb_ub_inconsistent', 'trace_schema_version']
            legacy_trace = {k: trace[0][k] for k in legacy_fields}
            legacy_trace['trace_schema_version'] = 'gurobi_master_v2'
            with path.open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=legacy_fields)
                writer.writeheader(); writer.writerow(legacy_trace)
            self.assertEqual(audit_one({'method': 'Random_Sampling', 'problem': 'SCMND'},
                                      legacy_row, path, .01, True, False)[0], [])
            self.assertTrue(failures(legacy_row))
