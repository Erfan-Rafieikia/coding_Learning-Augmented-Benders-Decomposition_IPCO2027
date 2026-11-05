"""Callback-context, iteration completeness and solver cut-summary regressions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from gurobipy import GRB
from Problem.callbacks import Callback
from scripts.audit_convergence_pilots import audit_event_trace
from scripts.plot_labd_convergence import rows_from
from utils.convergence_diagnostics import build_solver_bound_trace, write_bound_trace
from utils.gurobi_cut_diagnostics import GurobiCutStatistics


def callback_fixture():
    cb = Callback.__new__(Callback)
    cb.bound_trace = []
    cb.iteration = 0
    cb._last_bound_pair = None
    cb._submitted_cut_counts = dict.fromkeys(('exact_lazy', 'exact_user', 'ml_lazy', 'ml_user'), 0)
    cb.gurobi_cut_statistics = GurobiCutStatistics()
    cb.callback_counts = dict.fromkeys(('mipsol_root', 'mipsol_nonroot', 'mipnode_root', 'mipnode_nonroot'), 0)
    cb.total_nodes_explored = 0
    cb._should_handle_relaxation_callbacks = lambda: True
    cb.handle_callback = Mock()
    return cb


def context_model(context, *, elapsed=1., lower=10., upper=20., cuts=0, status=GRB.OPTIMAL):
    # Illegal query codes fail rather than returning a fabricated generic value.
    values = {GRB.Callback.RUNTIME: elapsed,
              getattr(GRB.Callback, context+'_OBJBND'): lower,
              getattr(GRB.Callback, context+'_OBJBST'): upper,
              getattr(GRB.Callback, context+'_NODCNT'): 0.}
    if context == 'MIP':
        values[GRB.Callback.MIP_CUTCNT] = cuts
    if context == 'MIPNODE':
        values[GRB.Callback.MIPNODE_STATUS] = status
    model = Mock()
    model.cbGet.side_effect = values.__getitem__
    return model, values


def complete_fixture():
    cb = callback_fixture()
    mip, _ = context_model('MIP')
    cb._dispatch_callback(mip, GRB.Callback.MIP)
    sol, values = context_model('MIPSOL', elapsed=1.1)
    def submit(*args, **kwargs):
        cb._submitted_cut_counts['exact_lazy'] += 2
        values[GRB.Callback.RUNTIME] = 1.2
    cb.handle_callback.side_effect = submit
    cb._dispatch_callback(sol, GRB.Callback.MIPSOL)
    mip, _ = context_model('MIP', elapsed=1.3, lower=12., cuts=2)
    cb._dispatch_callback(mip, GRB.Callback.MIP)
    cb.append_terminal_bounds(SimpleNamespace(Runtime=1.4, ObjBound=12., ObjVal=20., SolCount=1, NodeCount=0))
    row = dict(iteration=1, gurobi_solver_runtime_sec=1.4,
               exact_cut_construction_count=2, ml_cut_construction_count=0,
               **cb.gurobi_cut_statistics.export())
    return build_solver_bound_trace(cb.bound_trace), row


class SolverEventTraceTests(unittest.TestCase):
    def test_candidate_is_not_incumbent_and_zero_cut_round_has_both_boundaries(self):
        cb = callback_fixture()
        model, values = context_model('MIPSOL', upper=1e100)
        values[GRB.Callback.MIPSOL_OBJ] = 0.  # A candidate that may be rejected.
        cb._dispatch_callback(model, GRB.Callback.MIPSOL)
        self.assertEqual([r['event'] for r in cb.bound_trace], ['solver_iteration_start', 'solver_iteration_end'])
        self.assertEqual([r['iteration'] for r in cb.bound_trace], [1, 1])
        self.assertEqual([r['master_upper_bound'] for r in cb.bound_trace], [None, None])
        self.assertTrue(all(r['submitted_exact_cuts'] == 0 for r in cb.bound_trace))
        self.assertNotIn((GRB.Callback.MIPSOL_OBJ,), [c.args for c in model.cbGet.call_args_list])
        cb.handle_callback.assert_called_once_with(model, integer_solution=True)

    def test_lp_bounds_observed_even_when_no_benders_round_runs(self):
        for skip_by_policy, status in ((True, GRB.OPTIMAL), (False, GRB.INFEASIBLE)):
            with self.subTest(skip_by_policy=skip_by_policy, status=status):
                cb = callback_fixture()
                cb._should_handle_relaxation_callbacks = lambda: not skip_by_policy
                model, values = context_model('MIPNODE', status=status)
                cb._dispatch_callback(model, GRB.Callback.MIPNODE)
                values[GRB.Callback.MIPNODE_OBJBND] = 11.
                values[GRB.Callback.RUNTIME] = 1.01
                cb._dispatch_callback(model, GRB.Callback.MIPNODE)
                cb.handle_callback.assert_not_called()
                self.assertEqual(cb.iteration, 0)
                self.assertEqual([r['lower_bound'] for r in cb.bound_trace], [10., 11.])
                self.assertTrue(all(r['gurobi_mip_cut_count'] is None for r in cb.bound_trace))

    def test_eligible_lp_round_counts_user_submissions_and_keeps_visible_bounds(self):
        cb = callback_fixture()
        model, _ = context_model('MIPNODE')
        def submit(*args, **kwargs):
            cb._submitted_cut_counts['ml_user'] += 3
        cb.handle_callback.side_effect = submit
        cb._dispatch_callback(model, GRB.Callback.MIPNODE)
        start, end = cb.bound_trace
        self.assertEqual((start['submitted_ml_cuts'], end['submitted_ml_cuts']), (0, 3))
        self.assertEqual(end['submitted_user_cuts'], 3)
        self.assertEqual((start['lower_bound'], end['lower_bound']), (10., 10.))
        self.assertEqual(end['bound_changed'], 0)
        cb.handle_callback.assert_called_once_with(model, integer_solution=False)

    def test_complete_trace_round_trip_and_distinct_time_iteration_views(self):
        trace, row = complete_fixture()
        self.assertEqual(audit_event_trace(trace, row), [])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'trace.csv'
            write_bound_trace(trace, path)
            self.assertEqual(rows_from(path, 'time', 'lower_bound'),
                             [(1., 10.), (1.1, 10.), (1.2, 10.), (1.3, 12.), (1.4, 12.)])
            self.assertEqual(rows_from(path, 'iteration', 'lower_bound'), [(1., 10.), (1., 12.)])

    def test_audit_rejects_missing_boundaries_or_mislabelled_counts(self):
        trace, row = complete_fixture()
        mutations = [
            lambda t, r: t.pop(2),
            lambda t, r: t[1].update(gurobi_mip_cut_count=2),
            lambda t, r: t[2].update(submitted_ml_cuts=1),
            lambda t, r: t[3].update(bound_changed=0),
            lambda t, r: r.update(gurobi_mip_cut_count_peak_observed=99),
            lambda t, r: r.update(gurobi_last_lp_total_cut_count=0),
            lambda t, r: r.update(exact_cut_construction_count=99),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                changed_trace, changed_row = copy.deepcopy(trace), dict(row)
                mutate(changed_trace, changed_row)
                self.assertTrue(audit_event_trace(changed_trace, changed_row))


class GurobiCutStatisticsTests(unittest.TestCase):
    def test_streamed_last_lp_summary_separates_user_lazy_and_builtin(self):
        stats = GurobiCutStatistics()
        text = 'ordinary log line\nCutting planes:\n  Gomory: 2\n  User: 3\n  Lazy constraints: 4\n\nExplored 5 nodes\n'
        for start in range(0, len(text), 7):
            stats.feed_message(text[start:start+7])
        stats.finalize()
        result = stats.export()
        self.assertEqual(result['gurobi_last_lp_cut_summary_status'], 'reported')
        self.assertEqual(json.loads(result['gurobi_last_lp_cut_counts_json']), {'Gomory': 2, 'User': 3, 'Lazy constraints': 4})
        self.assertEqual(result['gurobi_last_lp_total_cut_count'], 9)
        self.assertEqual(result['gurobi_last_lp_user_cut_count'], 3)
        self.assertEqual(result['gurobi_last_lp_lazy_constraint_count'], 4)

    def test_missing_or_interrupted_summary_is_unknown_not_zero(self):
        for log, status in (('no cuts section\n', 'not_reported'),
                            ('Cutting planes:\n  User: 2\n', 'incomplete'),
                            ('Cutting planes:\n  Unsupported format\n', 'incomplete')):
            with self.subTest(status=status, log=log):
                stats = GurobiCutStatistics()
                stats.feed_message(log)
                stats.finalize()
                result = stats.export()
                self.assertEqual(result['gurobi_last_lp_cut_summary_status'], status)
                for field in ('cut_counts_json', 'total_cut_count', 'user_cut_count', 'lazy_constraint_count'):
                    self.assertIsNone(result['gurobi_last_lp_'+field])

    def test_repeated_section_uses_only_last_report_and_final_partial_line(self):
        stats = GurobiCutStatistics()
        stats.feed_message('Cutting planes:\n  User: 10\n\nCutting planes:\n  Lazy constraints: 2\nExplored 1 nodes')
        stats.finalize()
        result = stats.export()
        self.assertEqual(result['gurobi_last_lp_total_cut_count'], 2)
        self.assertEqual(result['gurobi_last_lp_user_cut_count'], 0)

    def test_current_cut_count_can_decrease_and_is_not_cumulative(self):
        stats = GurobiCutStatistics()
        changes = [stats.observe(n, t) for t, n in enumerate((0, 4, 4, 2))]
        self.assertEqual(changes, [True, True, False, True])
        result = stats.export()
        self.assertEqual(result['gurobi_mip_cut_count_last_observed'], 2)
        self.assertEqual(result['gurobi_mip_cut_count_peak_observed'], 4)
        self.assertEqual(result['gurobi_mip_cut_count_observation_count'], 4)
        self.assertEqual(result['gurobi_mip_cut_count_last_observed_time_sec'], 3.)
        for count, stamp in ((-1, 0), (1.5, 0), (float('inf'), 0), (1, -1), (1, float('nan'))):
            with self.subTest(count=count, stamp=stamp), self.assertRaises(ValueError):
                stats.observe(count, stamp)


if __name__ == '__main__':
    unittest.main()
