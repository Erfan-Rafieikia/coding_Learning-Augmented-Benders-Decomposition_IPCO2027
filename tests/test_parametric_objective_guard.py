"""Dependency-light tests for the optional dual-objective dominance safeguard."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Dual_Prediction.parametric import (
    DualVectorCodec,
    apply_objective_dominance_guard,
    SCMND_MANUSCRIPT,
    SCMND_CURRENT,
    ParametricDualPredictor,
    PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
)


class DominanceGuardTests(unittest.TestCase):
    def setUp(self):
        self.facilities = SimpleNamespace(
            I=[0], J=[0],
            cost=[[20.0]],
            scenarios=[[10.0]],
            capacity=[20.0],
            capacity_scenarios=[[25.0]],
        )
        self.codec = DualVectorCodec("SCFLP", self.facilities)
        self.x = {0: 1.0}

    def test_switch_defaults_off_and_accepts_true(self):
        p0 = ParametricDualPredictor(
            PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
            "SCFLP", self.facilities
        )
        p1 = ParametricDualPredictor(
            PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
            "SCFLP", self.facilities,
            parameters={"parametric_objective_guard": True}
        )
        self.assertFalse(p0.objective_guard_enabled)
        self.assertTrue(p1.objective_guard_enabled)
        with self.assertRaises(ValueError):
            ParametricDualPredictor(
                PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
                "SCFLP", self.facilities,
                parameters={"parametric_objective_guard": "False"}
            )

    def test_better_candidate_retained_scflp(self):
        baseline = np.array([1., 0.])
        candidate = np.array([2., 0.])
        self.assertTrue(self.codec.is_feasible(baseline))
        self.assertTrue(self.codec.is_feasible(candidate))
        vector, accepted, delta = apply_objective_dominance_guard(
            self.codec, 0, self.x, candidate, baseline
        )
        self.assertTrue(accepted)
        self.assertAlmostEqual(delta, 10)
        np.testing.assert_array_equal(vector, candidate)

    def test_worse_candidate_reverts_scflp(self):
        baseline = np.array([2., 0.])
        candidate = np.array([1., 0.])
        vector, accepted, delta = apply_objective_dominance_guard(
            self.codec, 0, self.x, candidate, baseline
        )
        self.assertFalse(accepted)
        self.assertLess(delta, 0)
        np.testing.assert_array_equal(vector, baseline)

    def test_tie_returns_baseline(self):
        baseline = np.array([2., 0.])
        vector, accepted, delta = apply_objective_dominance_guard(
            self.codec, 0, self.x, baseline, baseline
        )
        self.assertFalse(accepted)
        self.assertEqual(delta, 0)
        np.testing.assert_array_equal(vector, baseline)

    def test_uses_current_first_stage_rhs(self):
        baseline = np.array([2., 0.])
        candidate = np.array([3., 1.])
        # At x=0 the extra dual mu is valuable; at x=1 it is not.
        good, ok0, _ = apply_objective_dominance_guard(
            self.codec, 0, {0: 0.0}, candidate, baseline
        )
        bad, ok1, _ = apply_objective_dominance_guard(
            self.codec, 0, {0: 1.0}, candidate, baseline
        )
        self.assertTrue(ok0)
        self.assertFalse(ok1)
        np.testing.assert_array_equal(good, candidate)
        np.testing.assert_array_equal(bad, baseline)

    def test_stochastic_capacity_changes_dominance(self):
        stochastic = DualVectorCodec("SCFLP", self.facilities, use_stochastic_capacity=True)
        baseline = np.array([2., 0.])
        candidate = np.array([3., 0.5])
        _, fixed_ok, _ = apply_objective_dominance_guard(
            self.codec, 0, self.x, candidate, baseline
        )
        _, stoch_ok, _ = apply_objective_dominance_guard(
            stochastic, 0, self.x, candidate, baseline
        )
        self.assertFalse(fixed_ok)  # delta 10 - 20*.5 = 0
        self.assertFalse(stoch_ok)  # delta 10 - 25*.5 = -2.5

    def test_scmnd_envelope_and_current(self):
        data = SimpleNamespace(
            N=[0, 1], K=[0], A=[(0, 1)],
            C=[[9.0]], origins={0: 0}, destinations={0: 1},
            demand_scenarios=[[10.0]],
            u_base=[20.0],
            capacity_scenarios=[[15.0]],
            slack_penalty_multiplier=2.0,
        )
        for form in (SCMND_MANUSCRIPT, SCMND_CURRENT):
            codec = DualVectorCodec("SCMND", data, scmnd_formulation=form)
            baseline = np.array([0., -1., 0.])
            candidate = np.array([0., -2., 0.])
            self.assertTrue(codec.is_feasible(baseline))
            self.assertTrue(codec.is_feasible(candidate))
            chosen, accepted, delta = apply_objective_dominance_guard(
                codec, 0, {0: 1.0}, candidate, baseline
            )
            self.assertTrue(accepted)
            self.assertAlmostEqual(delta, 10.0)
            np.testing.assert_array_equal(chosen, candidate)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            apply_objective_dominance_guard(
                self.codec, 0, self.x,
                np.array([float("nan"), 0.]), np.array([1., 0.])
            )


if __name__ == "__main__":
    unittest.main()
