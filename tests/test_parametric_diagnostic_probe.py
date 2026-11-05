"""Regression tests for matched guard ON/OFF diagnostic instrumentation."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Dual_Prediction.parametric import (
    ParametricDualPredictor,
    PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
)


class DiagnosticProbeTests(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(
            I=[0], J=[0], cost=[[20.0]],
            scenarios=[[10.0], [10.0]],
            capacity=[20.0], capacity_scenarios=[[20.0], [20.0]],
        )
        self.duals = {0: {"mu": {0: 20.0}, "nu": {0: 0.0}}}

    def predictor(self, guard, raw_mu):
        p = ParametricDualPredictor(
            PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
            "SCFLP", self.data,
            parameters={
                "parametric_objective_guard": guard,
                "parametric_diagnostic_probe": True,
            },
        )
        p.selected = [0]
        p.mapping = {1: {"representative": 0}}
        p.counts = {0: 1.0}
        p._raw_predictions = lambda ids, model, x: {
            int(sid): np.array([float(raw_mu), 0.0]) for sid in ids
        }
        self.assertTrue(p.prepare_iteration(
            self.duals, {0: 0.0}, [1], iteration=1
        ))
        return p

    def test_boundary_repair_is_exactly_zero_and_ties(self):
        for guard in (False, True):
            p = self.predictor(guard, 30.0)
            d = p.diagnostics()
            self.assertEqual(d["diag_gamma_zero"], 1)
            self.assertEqual(d["diag_gamma_positive_count"], 0)
            self.assertEqual(d["diag_probe_checks"], 1)
            self.assertEqual(d["diag_probe_tied"], 1)
            self.assertEqual(d["diag_probe_improved"], 0)
            self.assertEqual(p.predict([1])[1]["mu"][0], 20.0)
            self.assertEqual(d["objective_guard_checks"], 1 if guard else 0)

    def test_shadow_probe_never_changes_guard_off_predictions(self):
        # Lower mu is dual feasible but loses objective value.
        # OFF keeps the smaller dual, ON reverts to anchor.
        off = self.predictor(False, 5.0)
        on = self.predictor(True, 5.0)
        self.assertEqual(off.predict([1])[1]["mu"][0], 5.0)
        self.assertEqual(on.predict([1])[1]["mu"][0], 20.0)
        self.assertEqual(off.diagnostics()["diag_probe_negative"], 1)
        self.assertEqual(on.diagnostics()["diag_probe_negative"], 1)
        self.assertEqual(off.diagnostics()["objective_guard_checks"], 0)
        self.assertEqual(on.diagnostics()["objective_guard_rejected_negative"], 1)

    def test_positive_gamma_and_beneficial_correction(self):
        # Start from an interior feasible anchor so repair can move.
        self.duals[0]["mu"][0] = 10.0
        for guard in (False, True):
            p = self.predictor(guard, 12.0)
            d = p.diagnostics()
            self.assertEqual(d["diag_gamma_zero"], 0)
            self.assertEqual(d["diag_gamma_gt_01"], 1)
            self.assertEqual(d["diag_probe_improved"], 1)
            self.assertEqual(d["diag_probe_positive_total"], 20.0)
            self.assertEqual(p.predict([1])[1]["mu"][0], 12.0)
            self.assertEqual(d["objective_guard_accepted"], 1 if guard else 0)


if __name__ == "__main__":
    unittest.main()
