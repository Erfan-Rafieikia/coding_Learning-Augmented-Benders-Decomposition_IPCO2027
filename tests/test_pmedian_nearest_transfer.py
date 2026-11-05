"""Regression checks for the additive, scenario-specific p-median variants.

Run from repository root with: python -m pytest -q tests/test_pmedian_nearest_transfer.py
These tests do not solve full optimization instances.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Dual_Prediction.prediction import DualPredictor
from Dual_Prediction.parametric import ParametricDualPredictor


def test_nonparametric_nearest_transfer_follows_mapping_and_copies_duals():
    predictor = DualPredictor.__new__(DualPredictor)
    predictor.predictor_mappings = {
        2: {"representative": 0, "distance": 0.2},
        3: {"representative": 1, "distance": 0.1},
    }
    exact = {
        0: {"mu": {0: 10.0}, "nu": {0: 2.0}},
        1: {"mu": {0: 30.0}, "nu": {0: 4.0}},
    }

    predicted = predictor._predict_pmedian_nearest([2, 3], exact)
    assert predicted[2] == exact[0]
    assert predicted[3] == exact[1]
    assert predicted[2] != predicted[3]

    # Returned dicts are copied; prediction mutations cannot alter exact cache.
    predicted[2]["mu"][0] = 999.0
    assert exact[0]["mu"][0] == 10.0


def test_nonparametric_nearest_transfer_rejects_missing_representative():
    predictor = DualPredictor.__new__(DualPredictor)
    predictor.predictor_mappings = {2: {"representative": 7}}
    with pytest.raises(KeyError):
        predictor._predict_pmedian_nearest([2], {0: {"mu": {0: 1.0}}})


class DummyCodec:
    def flatten(self, dual):
        return np.asarray(dual, dtype=float)

    def canonicalize(self, vector):
        return np.asarray(vector, dtype=float)

    def is_feasible(self, vector, tolerance):
        return True


def test_parametric_nearest_anchor_is_scenario_specific():
    predictor = ParametricDualPredictor.__new__(ParametricDualPredictor)
    predictor.anchor_method = "PMEDIAN_NEAREST"
    predictor.selected = [0, 1]
    predictor.mapping = {
        2: {"representative": 0},
        3: {"representative": 1},
    }
    predictor.codec = DummyCodec()
    exact = {
        0: np.asarray([10.0, 2.0]),
        1: np.asarray([30.0, 4.0]),
    }

    np.testing.assert_allclose(predictor._feasible_anchor(exact, 2), exact[0])
    np.testing.assert_allclose(predictor._feasible_anchor(exact, 3), exact[1])
    np.testing.assert_allclose(predictor._feasible_anchor(exact, 0), exact[0])


def test_parametric_nearest_anchor_rejects_missing_representative():
    predictor = ParametricDualPredictor.__new__(ParametricDualPredictor)
    predictor.anchor_method = "PMEDIAN_NEAREST"
    predictor.selected = [0]
    predictor.mapping = {2: {"representative": 1}}
    predictor.codec = DummyCodec()
    with pytest.raises(KeyError):
        predictor._feasible_anchor({0: np.asarray([1.0])}, 2)
