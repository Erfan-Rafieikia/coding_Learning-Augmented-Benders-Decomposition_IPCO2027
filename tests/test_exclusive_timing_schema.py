"""Regression checks for exclusive 33-category LABD timing accounting.

No Gurobi license, data files, or numerical optimization are required.
Run with: PYTHONPATH=src python -m unittest tests.test_exclusive_timing_schema -v
"""
import time
import unittest

from utils.exclusive_profiler import (
    ExclusiveProfiler,
    TIMING_CATEGORIES,
    TIMING_PHASE_TAGS,
    profile_call,
    profile_section,
)


class ExclusiveTimingSchemaTests(unittest.TestCase):
    def test_complete_unique_category_schema(self):
        self.assertEqual(33, len(TIMING_CATEGORIES))
        self.assertEqual(33, len(set(TIMING_CATEGORIES)))
        self.assertEqual(set(TIMING_CATEGORIES), set(TIMING_PHASE_TAGS))
        profiler = ExclusiveProfiler()
        self.assertEqual(set(TIMING_CATEGORIES), set(profiler.snapshot_schema()))
        self.assertTrue(all(v == 0 for v in profiler.snapshot_schema().values()))
        self.assertTrue(all(v == 0 for v in profiler.count_schema().values()))

    def test_nested_spans_are_exclusive(self):
        profiler = ExclusiveProfiler()
        with profiler.measure("callback_other_logic"):
            time.sleep(0.004)
            with profile_section("representative_selection_pmedian"):
                time.sleep(0.004)
            time.sleep(0.004)
        totals = profiler.snapshot_schema()
        self.assertGreater(totals["callback_other_logic"], 0)
        self.assertGreater(totals["representative_selection_pmedian"], 0)
        self.assertEqual(1, profiler.count_schema()["callback_other_logic"])
        self.assertEqual(1, profiler.count_schema()["representative_selection_pmedian"])
        # Exclusive durations should approximate elapsed wall-clock time;
        # double-counting a nested interval would violate this bound.
        self.assertLess(sum(totals.values()), 0.06)

    def test_nested_calls_preserve_operation_counts(self):
        profiler = ExclusiveProfiler()
        with profiler.measure("callback_other_logic"):
            for _ in range(3):
                profile_call("cuts_insertion", lambda: None)
        self.assertEqual(3, profiler.count_schema()["cuts_insertion"])
        self.assertEqual(1, profiler.count_schema()["callback_other_logic"])

    def test_unprofiled_calls_do_not_create_fake_measurements(self):
        with profile_section("representative_selection_tree"):
            result = profile_call("cuts_insertion", lambda x: x + 1, 5)
        self.assertEqual(6, result)


if __name__ == "__main__":
    unittest.main()
