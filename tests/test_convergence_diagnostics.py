"""Unit tests for the Gurobi-observed master bound series."""
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.convergence_diagnostics import build_solver_bound_trace, write_bound_trace
from utils.exclusive_profiler import (
    ExclusiveProfiler, TIMING_CATEGORIES, TIMING_PHASE_TAGS, profile_call, profiled,
)

class ConvergenceDiagnosticsTests(unittest.TestCase):
    def test_only_observed_master_values_are_exported(self):
        solver = [
            {"elapsed_seconds": 0., "iteration": 0, "lower_bound": 10, "master_upper_bound": 11},
            {"elapsed_seconds": 2., "iteration": 2, "lower_bound": 9, "master_upper_bound": 12, "event": "solver_final"},
        ]
        data = build_solver_bound_trace(solver)
        self.assertEqual([r["elapsed_seconds"] for r in data], [0, 2])
        self.assertEqual(data[-1]["lower_bound"], 9)  # Do not smooth observations.
        self.assertEqual(data[-1]["master_upper_bound"], 12)
        self.assertAlmostEqual(data[-1]["master_relative_gap"], .25)
        self.assertNotIn("verified_upper_bound", data[-1])
        self.assertEqual(data[-1]["event"], "solver_final")
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "trace.csv"
            write_bound_trace(data, file)
            self.assertEqual(len(file.read_text().splitlines()), 3)
            self.assertNotIn("verified_upper_bound", file.read_text())

    def test_bad_timestamps_are_rejected_without_sorting(self):
        for points in ([{}], [{"elapsed_seconds": -1}],
                       [{"elapsed_seconds": 2}, {"elapsed_seconds": 1}]):
            with self.assertRaisesRegex(ValueError, "timestamp"):
                build_solver_bound_trace(points)

    def test_nested_exclusive_profiler(self):
        profiler = ExclusiveProfiler()
        with profiler.measure("callback"):
            with profiler.measure("exact"):
                _ = sum(range(1000))
            with profiler.measure("feature"):
                _ = sum(range(2000))
        times = profiler.snapshot()
        self.assertGreaterEqual(times["callback"], 0)
        self.assertGreaterEqual(times["exact"], 0)
        self.assertGreaterEqual(times["feature"], 0)
        self.assertAlmostEqual(
            sum(times.values()),
            profiler.totals["callback"] + profiler.totals["exact"] + profiler.totals["feature"],
            places=8
        )

    def test_all_33_categories_and_absent_method_zeros(self):
        self.assertEqual(len(TIMING_CATEGORIES), 33)
        self.assertEqual(len(set(TIMING_CATEGORIES)), 33)
        self.assertEqual(set(TIMING_PHASE_TAGS), set(TIMING_CATEGORIES))
        self.assertEqual(TIMING_PHASE_TAGS["phase0_exact_subproblem_solving"], "phase0")
        self.assertEqual(TIMING_PHASE_TAGS["all_candidate_expost_reevaluation"], "postsolve")
        profiler = ExclusiveProfiler()
        self.assertEqual(set(profiler.snapshot_schema()), set(TIMING_CATEGORIES))
        self.assertTrue(all(value == 0.0 for value in profiler.snapshot_schema().values()))
        self.assertTrue(all(value == 0 for value in profiler.count_schema().values()))
        with profiler.measure("callback_other_logic"):
            self.assertEqual(
                profile_call("graph_random_walk_generation", lambda x: x + 1, 2),
                3,
            )
            self.assertEqual(
                profile_call("embedding_skipgram_training", lambda: 10), 10
            )
        self.assertEqual(profiler.count_schema()["callback_other_logic"], 1)
        self.assertEqual(profiler.count_schema()["graph_random_walk_generation"], 1)
        self.assertEqual(profiler.count_schema()["embedding_skipgram_training"], 1)
        self.assertEqual(profiler.count_schema()["feasibility_repair_line_segment"], 0)
        self.assertAlmostEqual(
            sum(profiler.snapshot_schema().values()),
            sum(profiler.snapshot().values()),
            delta=1e-7,
        )
        self.assertGreaterEqual(min(profiler.snapshot_schema().values()), 0.0)

    def test_profiled_decorators_and_nested_exclusive_time(self):
        @profiled("regression_weight_training")
        def fit():
            return sum(range(10000))

        profiler = ExclusiveProfiler()
        # No callback profiler active: this call must not record anything.
        self.assertEqual(fit(), 49995000)
        with profiler.measure("callback_other_logic"):
            self.assertEqual(fit(), 49995000)
        self.assertEqual(profiler.count_schema()["regression_weight_training"], 1)
        self.assertLess(
            profiler.snapshot_schema()["regression_weight_training"],
            sum(profiler.snapshot_schema().values()) + 1e-9,
        )

    def test_profile_context_restored_on_exception(self):
        profiler = ExclusiveProfiler()
        with self.assertRaisesRegex(ValueError, "intentional"):
            with profiler.measure("callback_other_logic"):
                profile_call("feasibility_check",
                             lambda: (_ for _ in ()).throw(ValueError("intentional")))
        self.assertEqual(profiler._stack, [])
        self.assertEqual(profiler.count_schema()["feasibility_check"], 1)
        # A later unprofiled method must not attach to the old callback.
        profile_call("graph_random_walk_generation", lambda: None)
        self.assertEqual(profiler.count_schema()["graph_random_walk_generation"], 0)

    def test_missing_events(self):
        self.assertEqual(build_solver_bound_trace([]), [])

if __name__ == "__main__":
    unittest.main()
