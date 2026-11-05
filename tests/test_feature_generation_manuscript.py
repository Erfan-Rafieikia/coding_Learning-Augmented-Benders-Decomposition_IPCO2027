"""Tests for the manuscript graph-walk transition rule."""

import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering.feature_generation import (  # noqa: E402
    build_transition_probabilities,
    compute_scenario_weights,
    generate_random_walks,
)


class ManuscriptFeatureTransitionTests(unittest.TestCase):
    def setUp(self):
        self.weights = {
            (0, 0): 0.0,
            (1, 1): 0.0,
            (2, 2): 0.0,
            (0, 1): 0.2,
            (1, 0): 0.2,
            (0, 2): 0.8,
            (2, 0): 0.8,
            (1, 2): 0.4,
            (2, 1): 0.4,
        }

    def test_diagonal_is_excluded_and_rows_sum_to_one(self):
        transitions = build_transition_probabilities(self.weights, temperature=0.5)
        for source, (neighbors, probabilities) in transitions.items():
            self.assertNotIn(source, neighbors)
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        neighbors, probabilities = transitions[0]
        by_neighbor = dict(zip(neighbors, probabilities))
        expected_ratio = np.exp(-0.2 / 0.5) / np.exp(-0.8 / 0.5)
        self.assertAlmostEqual(
            by_neighbor[1] / by_neighbor[2], expected_ratio, places=12
        )

    def test_walks_never_take_a_self_loop_and_are_reproducible(self):
        first = generate_random_walks(self.weights, 4, 6, seed=17, temperature=1.0)
        second = generate_random_walks(self.weights, 4, 6, seed=17, temperature=1.0)
        self.assertEqual(first, second)
        for walk in first:
            self.assertTrue(all(a != b for a, b in zip(walk, walk[1:])))

    def test_legacy_transition_rule_remains_explicitly_available(self):
        transitions = build_transition_probabilities(
            self.weights,
            transition_rule="LEGACY_INVERSE_WITH_SELF_LOOPS",
        )
        neighbors, probabilities = transitions[0]
        probability = dict(zip(neighbors, probabilities))
        self.assertIn(0, neighbors)
        self.assertGreater(probability[0], 0.999999)

    def test_discrepancy_matches_manuscript_normalization(self):
        duals = {
            (0, 0): np.asarray([1.0, 0.0]),
            (0, 1): np.asarray([0.0, 1.0]),
            (1, 0): np.asarray([2.0, 0.0]),
            (1, 1): np.asarray([0.0, 2.0]),
        }
        epsilon = 1e-12
        weights = compute_scenario_weights(duals, stability_epsilon=epsilon)
        expected = 0.5 * (
            np.sqrt(2.0) / (2.0 + epsilon)
            + np.sqrt(8.0) / (4.0 + epsilon)
        )
        self.assertAlmostEqual(weights[(0, 1)], expected, places=12)
        self.assertEqual(weights[(0, 0)], 0.0)


if __name__ == "__main__":
    unittest.main()
