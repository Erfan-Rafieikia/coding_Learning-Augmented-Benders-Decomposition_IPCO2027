"""Pure-data tests for the two selectable SCMND formulations."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Dual_Prediction.parametric import DualVectorCodec  # noqa: E402
from Problem.data import (  # noqa: E402
    SCMND_FORMULATION_CURRENT,
    SCMND_FORMULATION_MANUSCRIPT,
    normalize_scmnd_formulation,
    scmnd_capacity_vector,
    scmnd_envelope_demands,
)


def network_data(formulation):
    return SimpleNamespace(
        N=np.asarray([1, 2]),
        A=[(1, 2)],
        K=np.asarray([0]),
        S=np.asarray([0, 1]),
        C=np.asarray([[5.0]]),
        F=np.asarray([3.0]),
        u_base=np.asarray([9.0]),
        origins=np.asarray([1]),
        destinations=np.asarray([2]),
        demand_scenarios=np.asarray([[2.0], [3.0]]),
        capacity_scenarios=np.asarray([[7.0], [11.0]]),
        slack_penalty_multiplier=2.0,
        scmnd_formulation=formulation,
    )


class SCMNDFormulationSwitchTests(unittest.TestCase):
    def test_aliases_and_capacity_sources(self):
        self.assertEqual(
            normalize_scmnd_formulation("current"), SCMND_FORMULATION_CURRENT
        )
        self.assertEqual(
            normalize_scmnd_formulation("manuscript"), SCMND_FORMULATION_MANUSCRIPT
        )
        current = network_data(SCMND_FORMULATION_CURRENT)
        manuscript = network_data(SCMND_FORMULATION_MANUSCRIPT)
        np.testing.assert_allclose(scmnd_capacity_vector(current, 1), [11.0])
        np.testing.assert_allclose(scmnd_capacity_vector(manuscript, 1), [9.0])
        np.testing.assert_allclose(scmnd_envelope_demands(manuscript), [3.0])

    def test_dual_region_and_objective_coefficients_change_together(self):
        current_data = network_data(SCMND_FORMULATION_CURRENT)
        manuscript_data = network_data(SCMND_FORMULATION_MANUSCRIPT)
        current = DualVectorCodec(
            "SCMND", current_data, scmnd_formulation=SCMND_FORMULATION_CURRENT
        )
        manuscript = DualVectorCodec(
            "SCMND", manuscript_data, scmnd_formulation=SCMND_FORMULATION_MANUSCRIPT
        )

        # The current soft-capacity dual has alpha <= 2*max(c)=10;
        # the manuscript hard-capacity dual has no such upper bound.
        dual = {"pi": {(1, 0): 3.0, (2, 0): 1.0}, "alpha": {0: 11.0}}
        current_vector = current.flatten(dual)
        manuscript_vector = manuscript.flatten(dual)
        self.assertFalse(current.is_feasible(current_vector))
        self.assertTrue(manuscript.is_feasible(manuscript_vector))

        current_g = current.objective_coefficients(0, {0: 1.0})
        manuscript_g = manuscript.objective_coefficients(0, {0: 1.0})
        self.assertEqual(current_g[current.alpha_pos[0]], -7.0)
        self.assertEqual(manuscript_g[manuscript.alpha_pos[0]], -9.0)

        # Commodity-specific potential normalization is objective-invariant.
        normalized = manuscript.canonicalize(manuscript_vector)
        self.assertAlmostEqual(normalized[manuscript.pi_pos[(1, 0)]], 0.0)
        self.assertAlmostEqual(
            float(manuscript_vector @ manuscript_g),
            float(normalized @ manuscript_g),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
