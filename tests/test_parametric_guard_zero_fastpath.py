"""Regression tests for optional gamma==0 dominance fast path."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Dual_Prediction.parametric import (
    ParametricDualPredictor, PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR
)


class ZeroGammaFastpathTests(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(
            I=[0], J=[0], cost=[[20.0]],
            scenarios=[[10.0], [10.0]],
            capacity=[20.0], capacity_scenarios=[[20.0], [20.0]],
        )

    def run_predictor(self, fastpath, anchor_mu, raw_mu,
                      guard=True, patch_repair=None):
        predictor = ParametricDualPredictor(
            PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
            "SCFLP", self.data,
            parameters={
                "parametric_objective_guard": guard,
                "parametric_guard_gamma_zero_fastpath": fastpath,
                "parametric_diagnostic_probe": False,
            },
        )
        predictor.selected = [0]
        predictor.mapping = {1: {"representative": 0}}
        predictor.counts = {0: 1.0}
        predictor._raw_predictions = lambda ids, model, x: {
            int(sid): np.array([float(raw_mu), 0.0]) for sid in ids
        }
        if patch_repair:
            predictor.codec.repair_along_segment = patch_repair
        self.assertTrue(predictor.prepare_iteration(
            {0: {"mu": {0: anchor_mu}, "nu": {0: 0.0}}},
            {0: 0.0}, [1], iteration=1,
        ))
        return predictor

    def test_exact_zero_skips_comparison_and_preserves_same_dual(self):
        regular = self.run_predictor(False, 20.0, 30.0)
        shortcut = self.run_predictor(True, 20.0, 30.0)
        self.assertEqual(regular.predict([1]), shortcut.predict([1]))
        self.assertEqual(regular.diagnostics()["objective_guard_checks"], 1)
        self.assertEqual(shortcut.diagnostics()["objective_guard_checks"], 1)
        self.assertEqual(regular.diagnostics()["objective_guard_evaluated"], 1)
        self.assertEqual(shortcut.diagnostics()["objective_guard_evaluated"], 0)
        self.assertEqual(shortcut.diagnostics()["objective_guard_skipped_zero"], 1)
        self.assertEqual(shortcut.diagnostics()["objective_guard_rejected"], 1)
        self.assertEqual(shortcut.diagnostics()["objective_guard_accepted"], 0)

    def test_nonzero_gamma_better_candidate_is_still_evaluated(self):
        regular = self.run_predictor(False, 10.0, 12.0)
        shortcut = self.run_predictor(True, 10.0, 12.0)
        self.assertEqual(regular.predict([1]), shortcut.predict([1]))
        self.assertEqual(shortcut.diagnostics()["objective_guard_evaluated"], 1)
        self.assertEqual(shortcut.diagnostics()["objective_guard_skipped_zero"], 0)
        self.assertEqual(shortcut.diagnostics()["objective_guard_accepted"], 1)

    def test_nonzero_gamma_worse_candidate_is_rejected(self):
        regular = self.run_predictor(False, 20.0, 5.0)
        shortcut = self.run_predictor(True, 20.0, 5.0)
        self.assertEqual(regular.predict([1]), shortcut.predict([1]))
        self.assertEqual(shortcut.diagnostics()["objective_guard_evaluated"], 1)
        self.assertEqual(shortcut.diagnostics()["objective_guard_rejected_negative"], 1)

    def test_zero_gamma_but_different_dual_must_not_skip(self):
        def altered_repair(base, raw, **kwargs):
            return np.array([12.0, 0.0]), 0.0, False, False
        regular = self.run_predictor(False, 10.0, 15.0,
                                     patch_repair=altered_repair)
        shortcut = self.run_predictor(True, 10.0, 15.0,
                                      patch_repair=altered_repair)
        self.assertEqual(regular.predict([1]), shortcut.predict([1]))
        self.assertEqual(shortcut.diagnostics()["objective_guard_skipped_zero"], 0)
        self.assertEqual(shortcut.diagnostics()["objective_guard_evaluated"], 1)

    def test_zero_gamma_fastpath_preserves_scmnd_both_formulations(self):
        data = SimpleNamespace(
            N=[0, 1], K=[0], A=[(0, 1)],
            C=[[9.0]], origins={0: 0}, destinations={0: 1},
            demand_scenarios=[[10.0], [10.0]],
            u_base=[20.0], capacity_scenarios=[[15.0], [15.0]],
            slack_penalty_multiplier=2.0,
        )
        for form in ("MANUSCRIPT_ENVELOPE", "CURRENT_SOFT_CAPACITY"):
            predictors = []
            for fast in (False, True):
                p = ParametricDualPredictor(
                    PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
                    "SCMND", data,
                    parameters={
                        "scmnd_formulation": form,
                        "parametric_objective_guard": True,
                        "parametric_guard_gamma_zero_fastpath": fast,
                    },
                )
                p.selected = [0]
                p.mapping = {1: {"representative": 0}}
                p.counts = {0: 1.0}
                p._raw_predictions = lambda ids, model, x: {
                    int(sid): np.array([0.0, -20.0, 0.0]) for sid in ids
                }
                self.assertTrue(p.prepare_iteration(
                    {0: {"pi": {(0, 0): 0.0, (1, 0): -9.0}, "alpha": {0: 0.0}}},
                    {0: 1.0}, [1], iteration=1,
                ))
                predictors.append(p)
            regular, optimized = predictors
            self.assertEqual(regular.predict([1]), optimized.predict([1]))
            self.assertEqual(optimized.diagnostics()["objective_guard_skipped_zero"], 1)
            self.assertEqual(optimized.diagnostics()["objective_guard_evaluated"], 0)

    def test_global_zero_gamma_fastpath_matches_standard(self):
        for fast in (False, True):
            p = ParametricDualPredictor(
                PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
                "SCFLP", self.data,
                parameters={
                    "parametric_objective_guard": True,
                    "parametric_guard_gamma_zero_fastpath": fast,
                    "parametric_repair_mode": "GLOBAL",
                },
            )
            p.selected = [0]
            p.mapping = {1: {"representative": 0}}
            p.counts = {0: 1.0}
            p._raw_predictions = lambda ids, model, x: {
                int(sid): np.array([30.0, 0.0]) for sid in ids
            }
            self.assertTrue(p.prepare_iteration(
                {0: {"mu": {0: 20.0}, "nu": {0: 0.0}}},
                {0: 0.0}, [1], iteration=1,
            ))
            self.assertEqual(p.predict([1])[1]["mu"][0], 20.0)
            self.assertEqual(p.diagnostics()["objective_guard_skipped_zero"], 1 if fast else 0)

    def test_guard_off_ignores_fastpath_flag(self):
        off = self.run_predictor(True, 20.0, 30.0, guard=False)
        self.assertEqual(off.diagnostics()["objective_guard_checks"], 0)
        self.assertEqual(off.diagnostics()["objective_guard_skipped_zero"], 0)

    def test_wrong_flag_type_rejected(self):
        with self.assertRaisesRegex(ValueError, "fastpath must be True or False"):
            ParametricDualPredictor(
                PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
                "SCFLP", self.data,
                parameters={"parametric_guard_gamma_zero_fastpath": "True"},
            )

    def test_count_accounting_guard_on(self):
        for fast in (False, True):
            pred = self.run_predictor(fast, 20.0, 30.0)
            d = pred.diagnostics()
            self.assertEqual(
                d["objective_guard_checks"],
                d["objective_guard_accepted"] + d["objective_guard_rejected"],
            )
            self.assertEqual(
                d["objective_guard_checks"],
                d["objective_guard_evaluated"] + d["objective_guard_skipped_zero"],
            )


if __name__ == "__main__":
    unittest.main()
