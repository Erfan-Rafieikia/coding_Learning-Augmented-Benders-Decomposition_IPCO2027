"""Cut-origin attribution, including nested insertion and both fallback paths."""
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import json

from Problem.callbacks import Callback
from utils.exclusive_profiler import ExclusiveProfiler
from scripts.audit_convergence_pilots import audit_cut_construction


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


class CutConstructionTimingTests(unittest.TestCase):
    def test_auditor_checks_new_details_without_reinterpreting_old_workbooks(self):
        self.assertEqual(audit_cut_construction({}, {}), [])
        detail = ExclusiveProfiler().cut_construction_breakdown()
        detail.update(exact_cut_construction_time_sec=3, exact_cut_construction_count=2,
                      end_to_end_component_counts_json=json.dumps({'cuts_insertion': 2}))
        parent = {'cuts_evaluation_generation': 5}
        self.assertEqual(audit_cut_construction(detail, parent), [])
        for change in ({'ml_cut_construction_time_sec': -1}, {'exact_cut_construction_time_sec': 6},
                       {'exact_cut_construction_count': 1}, {'exact_cut_construction_count': 2.5},
                       {'cut_construction_timing_version': None}, {'ml_cut_construction_time_sec': None}):
            with self.subTest(change=change):
                self.assertTrue(audit_cut_construction({**detail, **change}, parent))

    def test_origin_details_exclude_insertion_and_do_not_double_count(self):
        clock = Clock()
        cb = Callback.__new__(Callback)
        cb.exclusive_profiler = ExclusiveProfiler(clock)
        cb._submitted_cut_counts = dict.fromkeys(("exact_lazy", "exact_user", "ml_lazy", "ml_user"), 0)
        def construct(*args):
            clock.advance(3)
            with cb.exclusive_profiler.measure("cuts_insertion"):
                clock.advance(5)
        cb._add_optimality_cut_impl = Mock(side_effect=construct)
        with cb.exclusive_profiler.measure("callback_other_logic"):
            clock.advance(7)
            # Evaluation shares the parent category but is not construction.
            with cb.exclusive_profiler.measure("cuts_evaluation_generation"):
                clock.advance(11)
            cb.add_optimality_cut(None, {}, 0, True)
            cb.add_optimality_cut(None, {}, 1, False, cut_origin="ml")
        detail = cb.exclusive_profiler.cut_construction_breakdown()
        self.assertEqual(detail["exact_cut_construction_time_sec"], 3)
        self.assertEqual(detail["ml_cut_construction_time_sec"], 3)
        self.assertEqual(detail["exact_cut_construction_count"], 1)
        self.assertEqual(detail["ml_cut_construction_count"], 1)
        self.assertEqual(cb.exclusive_profiler.totals["cuts_evaluation_generation"], 17)
        self.assertEqual(cb.exclusive_profiler.totals["cuts_insertion"], 10)
        self.assertEqual(sum(cb.exclusive_profiler.snapshot_schema().values()), 34)
        self.assertEqual(cb._add_optimality_cut_impl.call_args_list[1].args, (None, {}, 1, False))
        with self.assertRaisesRegex(ValueError, "cut origin"):
            cb.add_optimality_cut(None, {}, 0, cut_origin="typo")

    def test_exception_records_detail_and_unwinds_timers(self):
        clock = Clock()
        p = ExclusiveProfiler(clock)
        with self.assertRaises(RuntimeError):
            with p.measure("cuts_evaluation_generation", detail="ml"):
                clock.advance(2)
                with p.measure("cuts_insertion"):
                    clock.advance(5)
                    raise RuntimeError("submission failed")
        self.assertEqual(p.cut_construction_breakdown()["ml_cut_construction_time_sec"], 2)
        self.assertEqual(sum(p.snapshot_schema().values()), 7)
        self.assertEqual(p._stack, [])

    def test_solve_and_knn_fallback_use_actual_dual_origin(self):
        for method, origin in (("SOLVE", "exact"), ("KNN", "ml")):
            for integer in (True, False):
                with self.subTest(method=method, integer=integer):
                    cb = Callback.__new__(Callback)
                    cb.fallback_method = method
                    cb.exclusive_profiler = ExclusiveProfiler()
                    cb._submitted_cut_counts = dict.fromkeys(("exact_lazy", "exact_user", "ml_lazy", "ml_user"), 0)
                    cb.EPS = 1e-6
                    cb.theta_keys = [0]
                    cb._solve_dual_exact = Mock(return_value="solved")
                    cb.structure_duals = Mock(return_value={"dual": 1})
                    cb.dual_predictor = SimpleNamespace(predictor_type="KNN", is_trained=True,
                        predict=Mock(return_value={0: {"dual": 1}}))
                    cb.compute_dual_obj = Mock(return_value=10)
                    cb._add_optimality_cut_impl = Mock()
                    for context in ("mip", "rel"):
                        for source in ("fallback", "ml"):
                            setattr(cb, f"num_cuts_{context}_{source}", defaultdict(int))
                    cb._handle_fallback(None, [0], {}, {0: 0}, {}, integer)
                    detail = cb.exclusive_profiler.cut_construction_breakdown()
                    self.assertEqual(detail[f"{origin}_cut_construction_count"], 1)
                    other = "ml" if origin == "exact" else "exact"
                    self.assertEqual(detail[f"{other}_cut_construction_count"], 0)
                    cb._add_optimality_cut_impl.assert_called_once_with(None, {"dual": 1}, 0, integer)
                    # A nonviolated candidate must not construct or submit a cut.
                    cb._handle_fallback(None, [0], {}, {0: 10}, {}, integer)
                    self.assertEqual(cb.exclusive_profiler.cut_construction_breakdown(), detail)


if __name__ == "__main__":
    unittest.main()
