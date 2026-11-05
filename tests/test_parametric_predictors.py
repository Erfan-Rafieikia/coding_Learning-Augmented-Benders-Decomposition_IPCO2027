"""Deterministic numerical tests for parametric and cross-weight predictors."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Dual_Prediction.parametric import (  # noqa: E402
    PARAMETRIC_MANUSCRIPT,
    PARAMETRIC_PENALIZED_ERRORS,
    PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR,
    PARAMETRIC_RHS_RIDGE_REPAIR,
    PARAMETRIC_RHS_RIDGE_TREE_ANCHOR,
    PARAMETRIC_RIDGE_REPAIR,
    DualVectorCodec,
    ParametricDualPredictor,
    gp,
    solve_parametric_pmedian_selection,
    weighted_feasible_average,
)
from Dual_Prediction.prediction import DualPredictor  # noqa: E402
from utils.config_utils import PMEDIAN_WEIGHTED_TRANSFER, validate_method_config  # noqa: E402


def scflp_data():
    return SimpleNamespace(
        I=np.asarray([0, 1]),
        J=np.asarray([0, 1]),
        S=np.asarray([0, 1, 2, 3]),
        cost=np.asarray([[4.0, 5.0], [5.0, 4.0]]),
        scenarios=np.asarray(
            [[2.0, 1.0], [1.0, 2.0], [2.5, 1.5], [1.5, 2.5]]
        ),
        capacity=np.asarray([5.0, 6.0]),
        capacity_scenarios=None,
    )


def scflp_dual(mu0, mu1, nu0=0.0, nu1=0.0):
    return {
        "mu": {0: float(mu0), 1: float(mu1)},
        "nu": {0: float(nu0), 1: float(nu1)},
    }


FEATURES = {
    0: np.asarray([-1.0, -0.5]),
    1: np.asarray([-0.8, 0.4]),
    2: np.asarray([0.7, -0.3]),
    3: np.asarray([1.0, 0.6]),
}

DUALS = {
    0: scflp_dual(1.0, 1.5),
    1: scflp_dual(1.2, 1.8),
    2: scflp_dual(2.0, 1.0),
    3: scflp_dual(2.2, 1.2),
}


class ParametricNumericalTests(unittest.TestCase):
    def test_parametric_variants_are_valid_only_inside_ml_augmented_bd(self):
        for variant in (
            PARAMETRIC_MANUSCRIPT,
            PARAMETRIC_PENALIZED_ERRORS,
            PARAMETRIC_RIDGE_REPAIR,
            PARAMETRIC_RHS_RIDGE_REPAIR,
            PARAMETRIC_RHS_RIDGE_TREE_ANCHOR,
            PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR,
        ):
            valid, message = validate_method_config(
                {
                    "method_category": "ML",
                    "ml_method_type": "ML_Augmented_BD",
                    "ml_callback_type": "ML_at_both",
                    "predictor_type": variant,
                    "n_trials": 5,
                    "scmnd_formulation": "CURRENT_SOFT_CAPACITY",
                }
            )
            self.assertTrue(valid, message)

            valid, _ = validate_method_config(
                {
                    "method_category": "ML",
                    "ml_method_type": "Covariate_Similarity",
                    "ml_callback_type": "ML_at_both",
                    "predictor_type": variant,
                    "n_trials": 5,
                }
            )
            self.assertFalse(valid)

    def test_shared_pmedian_partition_is_deterministic(self):
        selected_1, mapping_1 = solve_parametric_pmedian_selection(
            FEATURES, 2, {"parametric_pmedian_solver": "GREEDY"}
        )
        selected_2, mapping_2 = solve_parametric_pmedian_selection(
            FEATURES, 2, {"parametric_pmedian_solver": "GREEDY"}
        )
        self.assertEqual(selected_1, selected_2)
        self.assertEqual(set(mapping_1), set(mapping_2))
        self.assertEqual(len(selected_1), 2)
        self.assertEqual(
            {sid for sid in FEATURES if sid not in selected_1},
            {sid for sid in mapping_1 if isinstance(sid, int)},
        )

    def test_all_four_variants_use_the_same_partition(self):
        variants = [
            PARAMETRIC_MANUSCRIPT,
            PARAMETRIC_PENALIZED_ERRORS,
            PARAMETRIC_RIDGE_REPAIR,
            PARAMETRIC_RHS_RIDGE_REPAIR,
        ]
        selections = []
        mappings = []
        for variant in variants:
            predictor = ParametricDualPredictor(
                variant,
                "SCFLP",
                scflp_data(),
                {"parametric_pmedian_solver": "GREEDY"},
            )
            selected, mapping = predictor.select(FEATURES, 2)
            selections.append(selected)
            mappings.append(
                {
                    sid: info["representative"]
                    for sid, info in mapping.items()
                    if isinstance(sid, int)
                }
            )
        self.assertTrue(all(value == selections[0] for value in selections[1:]))
        self.assertTrue(all(value == mappings[0] for value in mappings[1:]))

    def test_cross_weight_predictors_are_valid_configurations(self):
        valid, message = validate_method_config(
            {
                "method_category": "ML",
                "ml_method_type": "ML_Augmented_BD",
                "ml_callback_type": "ML_at_both",
                "predictor_type": PMEDIAN_WEIGHTED_TRANSFER,
                "n_trials": 5,
                "parametric_pmedian_solver": "GREEDY",
            }
        )
        self.assertTrue(valid, message)

    def test_pmedian_nonparametric_control_matches_rhs_ridge_anchor(self):
        control = DualPredictor(
            "ML_Augmented_BD",
            PMEDIAN_WEIGHTED_TRANSFER,
            "SCFLP",
            scflp_data(),
            use_stochastic_capacity=False,
            parametric_config={"parametric_pmedian_solver": "GREEDY"},
        )
        control._train_unified(FEATURES, 2, {})
        selected = sorted(control.selected_subproblems)
        predicted = sorted(set(FEATURES) - set(selected))
        current = {sid: DUALS[sid] for sid in selected}
        control_predictions = control.predict(predicted, current)

        rhs = ParametricDualPredictor(
            PARAMETRIC_RHS_RIDGE_REPAIR,
            "SCFLP",
            scflp_data(),
            {"parametric_pmedian_solver": "GREEDY"},
        )
        rhs_selected, _ = rhs.select(FEATURES, 2)
        self.assertEqual(selected, sorted(rhs_selected))
        expected_anchor = weighted_feasible_average(
            rhs.codec, current, rhs.counts
        )

        for sid in predicted:
            np.testing.assert_allclose(
                rhs.codec.flatten(control_predictions[sid]),
                expected_anchor,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_tree_and_cluster_anchor_variants_reuse_convex_transfer_weights(self):
        cases = [
            (
                PARAMETRIC_RHS_RIDGE_TREE_ANCHOR,
                "TREE",
                {"tree_min_impurity_decrease": 0.0, "tree_quantile_grid_size": 10},
            ),
            (
                PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR,
                "CLUSTERING",
                {
                    "clustering_random_state": 42,
                    "clustering_init_method": "k-means++",
                    "clustering_max_iter": 300,
                    "clustering_tol": 1e-4,
                },
            ),
        ]
        for variant, anchor_method, extra in cases:
            params = {
                "parametric_rhs_dim": 2,
                "parametric_repair_mode": "LOCAL",
                **extra,
            }
            predictor = ParametricDualPredictor(
                variant, "SCFLP", scflp_data(), params
            )
            selected, _ = predictor.select(FEATURES, 2)
            self.assertEqual(predictor.anchor_method, anchor_method)
            self.assertEqual(set(predictor.all_transfer_mapping), set(FEATURES))
            for info in predictor.all_transfer_mapping.values():
                self.assertGreater(len(info["anchors"]), 0)
                self.assertAlmostEqual(sum(info["weights"]), 1.0, places=12)
                self.assertTrue(set(info["anchors"]).issubset(set(selected)))

    def test_tree_and_cluster_rhs_ridge_models_train_once_and_remain_feasible(self):
        variants = [
            PARAMETRIC_RHS_RIDGE_TREE_ANCHOR,
            PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR,
        ]
        history = {}
        for iteration in (1, 2):
            for sid in FEATURES:
                offset = 0.05 * iteration
                history[(sid, iteration)] = scflp_dual(
                    DUALS[sid]["mu"][0] + offset,
                    DUALS[sid]["mu"][1] + offset,
                )
        first_stages = {
            1: {0: 1.0, 1: 0.0},
            2: {0: 1.0, 1: 1.0},
        }

        for variant in variants:
            predictor = ParametricDualPredictor(
                variant,
                "SCFLP",
                scflp_data(),
                {
                    "parametric_rhs_dim": 2,
                    "parametric_repair_mode": "LOCAL",
                    "clustering_random_state": 42,
                    "clustering_init_method": "k-means++",
                    "clustering_max_iter": 300,
                    "clustering_tol": 1e-4,
                    "tree_min_impurity_decrease": 0.0,
                    "tree_quantile_grid_size": 10,
                },
            )
            selected, _ = predictor.select(FEATURES, 2)
            predictor.fit_warmup_rhs_model(history, first_stages)
            self.assertEqual(predictor.training_calls, 1)

            predicted = sorted(set(FEATURES) - set(selected))
            current = {sid: DUALS[sid] for sid in selected}
            self.assertTrue(
                predictor.prepare_iteration(
                    current, {0: 0.0, 1: 1.0}, predicted, iteration=3
                )
            )
            self.assertEqual(predictor.training_calls, 1)
            self.assertTrue(
                all(
                    predictor.validate_structured_dual(value)
                    for value in predictor.predict(predicted).values()
                )
            )

    def test_feasible_average_and_analytical_repair(self):
        codec = DualVectorCodec("SCFLP", scflp_data())
        average = weighted_feasible_average(codec, {0: DUALS[0], 1: DUALS[1]})
        self.assertTrue(codec.is_feasible(average))

        raw = average.copy()
        raw[codec.customer_pos[0]] = 10.0
        repaired, gamma, raw_feasible, reverted = codec.repair_along_segment(
            average, raw
        )
        self.assertFalse(raw_feasible)
        self.assertFalse(reverted)
        self.assertGreaterEqual(gamma, 0.0)
        self.assertLess(gamma, 1.0)
        self.assertTrue(codec.is_feasible(repaired))

    def test_repair_accepts_anchor_within_gurobi_tolerance(self):
        """Regression for campaign tasks 361/368/396/406 (p20_s100).

        An anchor built from Gurobi duals can violate mu_i - nu_j <= c_ij by
        more than 1e-7 but less than 1e-6. It passes anchor certification
        (1e-6) and must not make the line-segment repair raise.
        """
        codec = DualVectorCodec("SCFLP", scflp_data())
        anchor = weighted_feasible_average(codec, {0: DUALS[0], 1: DUALS[1]})
        i0 = codec.customer_pos[0]
        nus = anchor[[codec.facility_pos[j] for j in (0, 1)]]
        slack = float(np.min(codec._scflp_cost[0, :] + nus) - anchor[i0])
        loose = anchor.copy()
        loose[i0] += slack + 5e-7  # violation = 5e-7
        self.assertFalse(codec.is_feasible(loose, 1e-7))
        self.assertTrue(codec.is_feasible(loose, 1e-6))

        raw = loose.copy()
        raw[i0] = 10.0
        repaired, gamma, raw_feasible, reverted = codec.repair_along_segment(loose, raw)
        self.assertFalse(raw_feasible)
        self.assertTrue(codec.is_feasible(repaired, 1e-6))

        # Anchors beyond the Gurobi tolerance are still rejected.
        bad = anchor.copy()
        bad[i0] += slack + 1e-4
        with self.assertRaises(ValueError):
            codec.repair_along_segment(bad, raw)

    def test_iteration_ridge_predictions_are_certified(self):
        predictor = ParametricDualPredictor(
            PARAMETRIC_RIDGE_REPAIR,
            "SCFLP",
            scflp_data(),
            {
                "parametric_pmedian_solver": "GREEDY",
                "parametric_ridge_beta": 1e-3,
                "parametric_repair_mode": "LOCAL",
            },
        )
        selected, _ = predictor.select(FEATURES, 2)
        predicted = sorted(set(FEATURES) - set(selected))
        solved = {sid: DUALS[sid] for sid in selected}
        self.assertTrue(
            predictor.prepare_iteration(
                solved, {0: 1.0, 1: 1.0}, predicted, iteration=6
            )
        )
        results = predictor.predict(predicted)
        self.assertEqual(set(results), set(predicted))
        self.assertTrue(all(predictor.validate_structured_dual(v) for v in results.values()))
        diagnostics = predictor.diagnostics()
        self.assertEqual(diagnostics["raw_predictions"], len(predicted))
        self.assertEqual(diagnostics["training_calls"], 1)
        self.assertTrue(np.isfinite(diagnostics["max_condition_number"]))

    def test_rhs_model_is_trained_once_and_reused(self):
        predictor = ParametricDualPredictor(
            PARAMETRIC_RHS_RIDGE_REPAIR,
            "SCFLP",
            scflp_data(),
            {
                "parametric_pmedian_solver": "GREEDY",
                "parametric_rhs_dim": 2,
                "parametric_repair_mode": "LOCAL",
            },
        )
        selected, _ = predictor.select(FEATURES, 2)
        history = {}
        for iteration in (1, 2):
            for sid in FEATURES:
                offset = 0.05 * iteration
                history[(sid, iteration)] = scflp_dual(
                    DUALS[sid]["mu"][0] + offset,
                    DUALS[sid]["mu"][1] + offset,
                )
        first_stages = {
            1: {0: 1.0, 1: 0.0},
            2: {0: 1.0, 1: 1.0},
        }
        predictor.fit_warmup_rhs_model(history, first_stages)
        self.assertEqual(predictor.training_calls, 1)

        predicted = sorted(set(FEATURES) - set(selected))
        current = {sid: DUALS[sid] for sid in selected}
        self.assertTrue(
            predictor.prepare_iteration(
                current, {0: 0.0, 1: 1.0}, predicted, iteration=3
            )
        )
        self.assertEqual(predictor.training_calls, 1)
        self.assertTrue(
            all(
                predictor.validate_structured_dual(value)
                for value in predictor.predict(predicted).values()
            )
        )


@unittest.skipUnless(gp is not None, "Gurobi is unavailable")
class ParametricConstrainedModelTests(unittest.TestCase):
    def _run_variant(self, variant):
        predictor = ParametricDualPredictor(
            variant,
            "SCFLP",
            scflp_data(),
            {
                "parametric_pmedian_solver": "GREEDY",
                "parametric_L_d": 1e6,
                "parametric_L_v": 1e6,
                "parametric_kappa_d": 1.0,
                "parametric_kappa_v": 1.0,
            },
        )
        selected, _ = predictor.select(FEATURES, 2)
        predicted = sorted(set(FEATURES) - set(selected))
        solved = {sid: DUALS[sid] for sid in selected}
        self.assertTrue(
            predictor.prepare_iteration(
                solved, {0: 1.0, 1: 1.0}, predicted, iteration=6
            )
        )
        self.assertTrue(
            all(
                predictor.validate_structured_dual(value)
                for value in predictor.predict(predicted).values()
            )
        )

    def test_manuscript_constrained_model(self):
        self._run_variant(PARAMETRIC_MANUSCRIPT)

    def test_penalized_errors_constrained_model(self):
        self._run_variant(PARAMETRIC_PENALIZED_ERRORS)


if __name__ == "__main__":
    unittest.main()
