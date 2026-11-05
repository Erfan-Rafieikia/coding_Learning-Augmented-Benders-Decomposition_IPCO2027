"""Opt-in tiny Gurobi integration checks; these are NOT the Nibi pilot.

LABD_RUN_SOLVER_SMOKE=1 PYTHONPATH=src python -m pytest -q tests/test_timing_solver_smoke.py
"""
import contextlib
import io
import logging
import os
import time
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from Problem import master_problem
from utils.exclusive_profiler import reconcile_end_to_end, TIMING_CATEGORIES, CUT_CONSTRUCTION_TIMING_VERSION
from utils.convergence_diagnostics import build_solver_bound_trace
from scripts.audit_convergence_pilots import audit_event_trace


METHODS = [("STANDARD_BD", "LOCAL"), ("RANDOM", "LOCAL"), ("TREE", "LOCAL"),
           ("CLUSTERING", "LOCAL"), ("PMEDIAN_WEIGHTED_TRANSFER", "LOCAL"),
           ("PMEDIAN_NEAREST_TRANSFER", "LOCAL"), ("PARAMETRIC_RHS_RIDGE_REPAIR", "LOCAL")]
METHODS += [(m, repair) for m in ("PARAMETRIC_RHS_RIDGE_TREE_ANCHOR", "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR",
                                 "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR") for repair in ("LOCAL", "STRUCTURE_AWARE")]


def tiny_data(problem):
    rng = np.random.default_rng(42)
    if problem == "SCFLP":
        demand = rng.uniform(1, 4, size=(8, 4))
        capacities = np.array([8., 7., 10., 9.])
        return SimpleNamespace(I=np.arange(4), J=np.arange(4), S=np.arange(8),
            cost=rng.uniform(1, 9, size=(4, 4)), fixed_costs=np.array([5., 7., 9., 6.]),
            scenarios=demand, capacities=capacities, capacity=capacities,
            capacity_scenarios=None, max_demand_sum_over_scenario=demand.sum(axis=1).max())
    return SimpleNamespace(N=np.arange(3), K=np.array([0, 1]), S=np.arange(8),
        A=[(0, 1), (1, 2), (0, 2)], C=np.array([[3., 5.], [4., 2.], [9., 10.]]),
        F=np.array([5., 6., 8.]), origins=np.array([0, 0]), destinations=np.array([2, 1]),
        u_base=np.array([20., 20., 20.]), demand_scenarios=rng.uniform(1, 9, size=(8, 2)),
        capacity_scenarios=rng.uniform(2, 8, size=(8, 3)), probabilities=np.full(8, 1/8),
        slack_penalty_multiplier=2., scmnd_formulation="MANUSCRIPT_ENVELOPE")


def method_config(method, repair):
    c = dict(method_category="ML", ml_method_type="ML_Augmented_BD", ml_callback_type="ML_at_both",
             predictor_type=method, n_trials=1, m_trials=0, n_selected_subproblems=2,
             switching_strategy="no_switch", use_covariates="features_only", use_stochastic_capacity=False,
             scmnd_formulation="MANUSCRIPT_ENVELOPE", parametric_pmedian_solver="GREEDY",
             parametric_repair_mode=repair, parametric_objective_guard=True,
             parametric_guard_gamma_zero_fastpath=True,
             feature_params=dict(n_walks=2, walk_length=5, feature_dim=2, window=5, min_count=1, sg=1, seed=42))
    if method == "STANDARD_BD":
        c.update(method_category="Exact", exact_callback_type="int_and_LP")
        c.pop("ml_method_type"); c.pop("predictor_type")
    elif method == "RANDOM":
        c.update(ml_method_type="Random_Sampling", predictor_type="RANDOM_SAMPLING")
    return c


@unittest.skipUnless(os.environ.get("LABD_RUN_SOLVER_SMOKE") == "1", "opt-in tiny solver checks")
class TinySolverTimingTests(unittest.TestCase):
    def test_all_campaign_methods_both_classes_and_gaps(self):
        from feature_engineering.feature_generation import Word2Vec
        self.assertIsNotNone(Word2Vec, "Working gensim and its dependencies are required for this smoke check")
        original_params = master_problem._set_params
        active = set()
        def small_params(model):
            original_params(model)
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = 20
        with patch.object(master_problem, "_set_params", side_effect=small_params):
            for problem in ("SCFLP", "SCMND"):
                for gap in ("0.01", "0.001"):
                    for method, repair in METHODS:
                        with self.subTest(problem=problem, gap=gap, method=method, repair=repair):
                            with patch.dict(os.environ, LABD_MIP_GAP_TARGET=gap), contextlib.redirect_stdout(io.StringIO()):
                                start = time.perf_counter()
                                data_start = time.perf_counter()
                                data = tiny_data(problem)
                                load_time = time.perf_counter() - data_start
                                solution = master_problem.solve_master_problem(problem, data, [], {}, method_config(method, repair))
                                total = time.perf_counter() - start
                            values = dict(solution.solver_component_breakdown)
                            counts = dict(solution.timing_component_counts)
                            additions = {"input_data_load_validation": load_time,
                                         "master_model_construction": solution.master_model_construction_time_sec,
                                         "terminal_exact_original_objective_validation": solution.final_objective_validation_time_sec,
                                         "all_candidate_expost_reevaluation": solution.candidate_reevaluation_time_sec}
                            values.update(additions)
                            counts.update({k: 1 for k in additions})
                            counts["all_candidate_expost_reevaluation"] = 0
                            counts["terminal_exact_original_objective_validation"] = solution.final_exact_evaluation_count
                            values, counts = reconcile_end_to_end(total, values, counts)
                            self.assertAlmostEqual(sum(values.values()), total, places=8)
                            self.assertEqual(set(values), set(TIMING_CATEGORIES))
                            detail = solution.cut_construction_breakdown
                            self.assertEqual(detail['cut_construction_timing_version'], CUT_CONSTRUCTION_TIMING_VERSION)
                            exact_cuts = solution.phase0_cuts_added + sum(
                                sum(getattr(solution, f'num_cuts_{context}_{source}').values())
                                for context in ('mip', 'rel')
                                for source in ('trial', 'selected', 'fallback', 'exact_switch'))
                            ml_cuts = sum(sum(getattr(solution, f'num_cuts_{context}_ml').values())
                                          for context in ('mip', 'rel'))
                            self.assertEqual(detail['exact_cut_construction_count'], exact_cuts)
                            self.assertEqual(detail['ml_cut_construction_count'], ml_cuts)
                            self.assertEqual(exact_cuts + ml_cuts, counts['cuts_insertion'])
                            construction_time = sum(detail[f'{origin}_cut_construction_time_sec'] for origin in ('exact', 'ml'))
                            self.assertLessEqual(construction_time, values['cuts_evaluation_generation'] + 1e-9)
                            for origin, count in (('exact', exact_cuts), ('ml', ml_cuts)):
                                self.assertEqual(detail[f'{origin}_cut_construction_time_sec'] > 0, count > 0)
                            if method == 'STANDARD_BD':
                                self.assertEqual(ml_cuts, 0)
                            self.assertTrue(all(v >= 0 for v in values.values()))
                            self.assertAlmostEqual(sum(solution.solver_component_breakdown.values()), solution.master_optimize_wall_time_sec, places=6)
                            self.assertEqual(solution.mip_gap_target, float(gap))
                            self.assertGreater(solution.original_Prob_Obj_Value, 0)
                            self.assertEqual(solution.final_exact_evaluation_count, int(method != "STANDARD_BD"))
                            self.assertEqual(solution.final_exact_subproblem_solves, 0 if method == "STANDARD_BD" else len(data.S))
                            self.assertEqual(solution.candidate_reevaluation_time_sec, 0)
                            self.assertIsNone(solution.objective_reevaluation_results)
                            if method == "STANDARD_BD":
                                self.assertEqual(solution.final_objective_validation_time_sec, 0)
                                self.assertEqual(solution.original_Prob_Obj_Value, solution.objective_value)
                            trace = build_solver_bound_trace(solution.bound_trace)
                            self.assertTrue(trace)
                            starts = [r for r in trace if r['event'] == 'solver_iteration_start']
                            ends = [r for r in trace if r['event'] == 'solver_iteration_end']
                            self.assertEqual([r['iteration'] for r in starts], list(range(1, solution.iteration+1)))
                            self.assertEqual([r['iteration'] for r in ends], list(range(1, solution.iteration+1)))
                            self.assertEqual(trace[-1]['submitted_exact_cuts'], exact_cuts)
                            self.assertEqual(trace[-1]['submitted_ml_cuts'], ml_cuts)
                            self.assertIn(solution.gurobi_cut_statistics['gurobi_last_lp_cut_summary_status'], ('reported', 'not_reported'))
                            self.assertEqual(audit_event_trace(trace, dict(
                                iteration=solution.iteration, gurobi_solver_runtime_sec=solution.solution_time,
                                **detail, **solution.gurobi_cut_statistics)), [])
                            self.assertFalse(any(row["lb_ub_inconsistent"] for row in trace))
                            active.update(k for k, n in counts.items() if n)
                            if method == "STANDARD_BD":
                                self.assertEqual(counts["regression_weight_training"], 0)
                                self.assertEqual(counts["nonparametric_dual_transfer"], 0)
                            if method in ("STANDARD_BD", "RANDOM"):
                                self.assertGreater(counts["standard_or_random_exact_subproblem_solving"], 0)
        # These leaves must have genuinely run somewhere in the 52 tiny solves.
        for category in ("representative_selection_tree", "representative_selection_clustering",
                         "representative_selection_pmedian", "regression_weight_training",
                         "graph_random_walk_generation", "embedding_skipgram_training",
                         "feasibility_repair_line_segment", "feasibility_repair_structure_aware",
                         "objective_dominance_guard", "cuts_insertion"):
            self.assertIn(category, active)

    def test_hard_capacity_matches_known_objective_and_rejects_overload(self):
        original = master_problem._set_params
        def small(model):
            original(model)
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
        for capacity in (9., 2.):
            data = SimpleNamespace(N=np.array([0, 1]), K=np.array([0]), S=np.array([0, 1]),
                A=[(0, 1)], C=np.array([[5.]]), F=np.array([3.]), origins=np.array([0]), destinations=np.array([1]),
                u_base=np.array([capacity]), demand_scenarios=np.array([[2.], [3.]]),
                capacity_scenarios=np.array([[100.], [100.]]), probabilities=np.array([.5, .5]),
                slack_penalty_multiplier=2., scmnd_formulation="MANUSCRIPT_ENVELOPE")
            with patch.object(master_problem, "_set_params", side_effect=small), contextlib.redirect_stdout(io.StringIO()):
                sol = master_problem.solve_master_problem("SCMND", data, [], {}, method_config("STANDARD_BD", "LOCAL"))
            if capacity == 9:
                self.assertAlmostEqual(sol.objective_value, 15.5)
                self.assertEqual(sol.original_Prob_Obj_Value, sol.objective_value)
            else:
                self.assertEqual(sol.stopping_reason, "INFEASIBLE")
                self.assertIsNone(sol.first_stage_optimnal_values)
                self.assertIsNone(sol.original_Prob_Obj_Value)
            self.assertEqual(sol.final_exact_evaluation_count, 0)
            self.assertEqual(sol.final_objective_validation_time_sec, 0)

    def test_main_exports_auditable_workbook_and_trace(self):
        for problem in ("SCFLP", "SCMND"):
            for method in ("STANDARD_BD", "RANDOM", "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"):
                with self.subTest(problem=problem, method=method):
                    self._check_export(problem, method)

    def _check_export(self, problem, method):
        from scripts.audit_convergence_pilots import audit_one, read_result
        from utils.exclusive_profiler import SOLUTION_TIME_METRIC_VERSION
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder/"config.py").write_text((root/"src/config.py").read_text())
            settings = dict(problem_classes_to_instances={problem: ['synthetic.txt']},
                method_categories_grid=['Exact' if method == 'STANDARD_BD' else 'ML'],
                exact_callback_types_grid=['int_and_LP'],
                ml_method_types_grid=['Random_Sampling' if method == 'RANDOM' else 'ML_Augmented_BD'],
                predictor_ML_Augmented_BD_grid=['PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR'],
                scmnd_formulation_grid=['MANUSCRIPT_ENVELOPE'],
                selected_subproblems_ratio_grid=[.25], n_trials_grid=[1], m_trials_grid=[0],
                parametric_repair_mode_grid=['LOCAL'], switching_strategy=['no_switch'],
                use_covariates_grid=['features_only'], parametric_pmedian_solver='GREEDY')
            (folder/"config_override.py").write_text("""
import config as _config
import sys as _sys
_values = {k: [v[0]] for k, v in vars(_config).items() if k.endswith('_grid') and isinstance(v, list) and v}
_values.update(""" + repr(settings) + """)
for _key, _value in _values.items():
    globals()[_key] = _value
    setattr(_config, _key, _value)
    setattr(_sys.modules['__main__'], _key, _value)
""")
            env = dict(os.environ, RUN_DIRECTORY=str(folder/"output"), RUN_IDENTIFIER="tiny_timing_export",
                       RUN_PROBLEM_CLASS=problem, LABD_DATA_ROOT=str(folder),
                       RESULTS_ROOT_OVERRIDE=str(folder/"results"), LABD_MIP_GAP_TARGET="0.001")
            env["PYTHONPATH"] = os.pathsep.join([str(folder), str(root/"src"), str(root), env.get("PYTHONPATH", "")])
            program = """
import config, runpy, os
from pathlib import Path
from tests.test_timing_solver_smoke import tiny_data
from Problem import data, master_problem
assert Path(config.__file__).parent == Path.cwd()
data.read_scflp_data = lambda *a, **kw: tiny_data('SCFLP')
data.read_scmnd_data = lambda *a, **kw: tiny_data('SCMND')
original = master_problem._set_params
def small(model):
    original(model)
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = 20
master_problem._set_params = small
runpy.run_path(os.environ['SMOKE_MAIN_PATH'], run_name='__main__')
"""
            env["SMOKE_MAIN_PATH"] = str(root/"src/main.py")
            result = subprocess.run([sys.executable, '-c', program], cwd=folder, env=env,
                                    capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:]+result.stderr[-3000:])
            row = read_result(folder/"output/solution_log.xlsx")
            record_method = {'STANDARD_BD': 'Standard_BD', 'RANDOM': 'Random_Sampling'}.get(method, method)
            failures, *_ = audit_one({'method': record_method, 'problem': problem},
                                     row, Path(row['convergence_trace_path']), .001, True, True)
            self.assertEqual(failures, [])
            self.assertEqual(row['solution_time_metric_version'], SOLUTION_TIME_METRIC_VERSION)
            self.assertEqual(row['cut_construction_timing_version'], CUT_CONSTRUCTION_TIMING_VERSION)
            self.assertEqual(row['gurobi_last_lp_cut_summary_status'], 'reported')
            self.assertGreater(row['gurobi_last_lp_total_cut_count'], 0)
            self.assertGreater(row['exact_cut_construction_time_sec'], 0)
            self.assertGreater(row['exact_cut_construction_count'], 0)
            self.assertEqual(row['ml_cut_construction_time_sec'] > 0, row['ml_cut_construction_count'] > 0)
            self.assertAlmostEqual(row['solution_time_sec'] + row['final_exact_evaluation_time_sec'], row['end_to_end_wall_time_sec'])
            self.assertEqual(row['final_exact_evaluation_count'], int(method != 'STANDARD_BD'))
            self.assertEqual(row['final_exact_subproblem_solves'], 0 if method == 'STANDARD_BD' else 8)
            self.assertEqual(row['num_solutions_reevaluated'], 0)
            if method == 'STANDARD_BD':
                self.assertEqual(row['objective_value'], row['original_Prob_Obj_Value'])
                self.assertEqual(row['final_exact_evaluation_time_sec'], 0)
