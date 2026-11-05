"""Tests for the IPCO 2027 direct-dual ridge predictor with a mean-of-exact-duals anchor."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Dual_Prediction.parametric import (  # noqa: E402
    DIRECT_RIDGE_PREDICTORS,
    PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN,
    PARAMETRIC_PREDICTORS,
    RHS_RIDGE_PREDICTORS,
    ParametricDualPredictor,
)


def _data(n_scen=8):
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        I=[0, 1, 2], J=[0, 1],
        cost=[[4.0, 6.0], [5.0, 3.0], [7.0, 2.0]],
        scenarios=[list(rng.uniform(5, 15, size=3)) for _ in range(n_scen)],
        capacity=[20.0, 20.0],
    )


def _feasible_duals(rng, data):
    # mu_i <= min_j cost_ij with nu = 0 is dual feasible for SCFLP.
    cmin = np.min(np.asarray(data.cost), axis=1)
    mu = rng.uniform(0.0, 1.0, size=len(data.I)) * cmin
    return {"mu": {i: float(mu[k]) for k, i in enumerate(data.I)},
            "nu": {j: 0.0 for j in data.J}}


class DirectRidgeTests(unittest.TestCase):
    def setUp(self):
        self.data = _data()
        self.rng = np.random.default_rng(1)
        self.ids = list(range(len(self.data.scenarios)))
        self.features = {sid: self.rng.normal(size=2) for sid in self.ids}
        self.params = {
            "parametric_pmedian_solver": "GREEDY",
            "parametric_objective_guard": True,
            "parametric_repair_mode": "LOCAL",
            "parametric_rhs_dim": 2,
        }

    def _trained(self):
        pred = ParametricDualPredictor(PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN, "SCFLP", self.data,
                                       parameters=self.params)
        pred.select(self.features, 3)
        warm, xs = {}, {}
        for it in (1, 2, 3):
            xs[it] = {0: 0.5 * it / 3, 1: 1.0 - 0.3 * it / 3}
            for sid in self.ids:
                warm[(sid, it)] = _feasible_duals(self.rng, self.data)
        pred.fit_warmup_rhs_model(warm, xs)
        return pred

    def test_registered(self):
        self.assertIn(PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN, DIRECT_RIDGE_PREDICTORS)
        self.assertIn(PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN, RHS_RIDGE_PREDICTORS)
        self.assertIn(PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN, PARAMETRIC_PREDICTORS)

    def test_no_intercept_no_standardization_no_baseline(self):
        pred = self._trained()
        self.assertTrue(pred.is_direct)
        self.assertEqual(pred.anchor_method, "MEAN_EXACT")
        # No intercept: 2 static features + 2 RHS components, no constant column.
        self.assertEqual(pred.warmup_coefficient.shape[1], 2 + 2)
        # No standardization of z or g.
        np.testing.assert_array_equal(pred.feature_scaler.mean, np.zeros(2))
        np.testing.assert_array_equal(pred.feature_scaler.scale, np.ones(2))
        np.testing.assert_array_equal(pred.rhs_reducer.mean, np.zeros(3 + 2))
        np.testing.assert_array_equal(pred.rhs_reducer.scale, np.ones(3 + 2))
        pred.current_bases = {}
        pred.current_base = None  # direct raw prediction must not need a baseline
        raw = pred._raw_predictions([0, 1], None, {0: 0.4, 1: 0.6})
        self.assertEqual(set(raw), {0, 1})

    def test_mean_anchor_feasible_guard_and_certified_predictions(self):
        pred = self._trained()
        exact = {sid: _feasible_duals(self.rng, self.data) for sid in pred.selected}
        x = {0: 0.4, 1: 0.6}
        predicted = [sid for sid in self.ids if sid not in pred.selected]
        self.assertTrue(pred.prepare_iteration(exact, x, predicted, iteration=5))
        mean = np.mean([pred.codec.flatten(exact[s]) for s in pred.selected], axis=0)
        for sid in predicted:
            np.testing.assert_allclose(pred.current_bases[sid], mean)
            vec = pred.codec.flatten(pred.predict([sid])[sid])
            self.assertTrue(pred.codec.is_feasible(vec, 1e-7))
            g = pred.codec.objective_coefficients(sid, x)
            # Objective guard: never weaker than the mean anchor at the current x.
            self.assertGreaterEqual(float(g @ vec), float(g @ mean) - 1e-9)


if __name__ == "__main__":
    unittest.main()
