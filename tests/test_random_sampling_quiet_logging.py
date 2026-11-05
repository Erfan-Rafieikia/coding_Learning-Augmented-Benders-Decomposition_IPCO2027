"""Regression checks: Random baseline values unchanged, SCMND log bounded.

PYTHONPATH=src python -m unittest tests.test_random_sampling_quiet_logging -v
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Dual_Prediction.prediction import DualPredictor


class QuietRandomSamplingTests(unittest.TestCase):
    def test_scmnd_first_selected_dual_and_log_once(self):
        predictor = SimpleNamespace(problem_type="SCMND")
        first = {"pi": {(0, 0): 2.0}, "alpha": {0: 1.0}}
        second = {"pi": {(0, 0): 9.0}, "alpha": {0: 3.0}}
        cache = {1: first, 2: second}
        with patch("Dual_Prediction.prediction.logger") as logger:
            for _ in range(3):
                predictions = DualPredictor._predict_bertsimas_random_sampling(
                    predictor, list(range(100, 1000)), cache
                )
                self.assertEqual(900, len(predictions))
                self.assertEqual(first, predictions[555])
                self.assertEqual(first, predictions[999])
            self.assertEqual(1, logger.info.call_count)
            self.assertEqual(0, logger.warning.call_count)

    def test_scflp_average_unchanged_and_no_fallback_notice(self):
        predictor = SimpleNamespace(problem_type="SCFLP")
        cache = {
            1: {"mu": {0: 2.0}, "nu": {0: 6.0}},
            2: {"mu": {0: 4.0}, "nu": {0: 10.0}},
        }
        with patch("Dual_Prediction.prediction.logger") as logger:
            predictions = DualPredictor._predict_bertsimas_random_sampling(
                predictor, [10, 11], cache
            )
            self.assertEqual(
                {10: {"mu": {0: 3.0}, "nu": {0: 8.0}},
                 11: {"mu": {0: 3.0}, "nu": {0: 8.0}}},
                predictions,
            )
            logger.info.assert_not_called()
            logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
