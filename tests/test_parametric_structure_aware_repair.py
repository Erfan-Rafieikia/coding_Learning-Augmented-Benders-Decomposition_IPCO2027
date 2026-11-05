"""Structure-aware closed-form dual repair feasibility and guard regression tests."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Dual_Prediction.parametric import (
    DualVectorCodec, ParametricDualPredictor,
    PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
    SCMND_MANUSCRIPT, SCMND_CURRENT,
)


class StructureAwareDualTests(unittest.TestCase):
    def setUp(self):
        self.scflp = SimpleNamespace(
            I=[0, 1], J=[0, 1],
            cost=[[5.0, 8.0], [7.0, 4.0]],
            scenarios=[[10.0, 8.0], [10.0, 8.0]],
            capacity=[10.0, 12.0],
        )
        self.network = SimpleNamespace(
            N=[0, 1, 2], K=[0, 1],
            A=[(0, 1), (1, 2), (0, 2)],
            C=[[9.0, 6.0], [4.0, 7.0], [8.0, 10.0]],
            origins={0: 0, 1: 0},
            destinations={0: 2, 1: 1},
            demand_scenarios=[[10.0, 8.0], [10.0, 8.0]],
            u_base=[20.0, 17.0, 30.0],
            capacity_scenarios=[[15.0, 14.0, 22.0], [15.0, 14.0, 22.0]],
            slack_penalty_multiplier=2.0,
        )

    def test_scflp_both_candidates_dual_feasible(self):
        codec = DualVectorCodec("SCFLP", self.scflp)
        vectors = [
            [25.0, -3.0, -2.0, 2.0],
            [-5.0, 35.0, 100.0, -4.0],
            [1e4, -1e4, 0.0, -40.0],
        ]
        for raw in vectors:
            candidates = codec.structure_aware_candidates(raw)
            self.assertEqual(set(candidates), {"scflp_lift_nu", "scflp_clip_mu"})
            for vector in candidates.values():
                self.assertTrue(codec.is_feasible(vector), vector)

    def test_scmnd_both_formulations_feasible(self):
        for form in (SCMND_MANUSCRIPT, SCMND_CURRENT):
            codec = DualVectorCodec("SCMND", self.network, scmnd_formulation=form)
            raws = [
                [5., -40., 15., -10., 31., 9., -6., 2., -9.],
                [-8., 20., 120., 50., -90., 0., -1., -3., -8.],
            ]
            for raw in raws:
                candidates = codec.structure_aware_candidates(raw)
                self.assertEqual(
                    len(candidates), 1 if form == SCMND_MANUSCRIPT else 2
                )
                for label, candidate in candidates.items():
                    self.assertTrue(codec.is_feasible(candidate), (form, label, candidate))

    def test_scmnd_soft_capacity_alpha_capped(self):
        codec = DualVectorCodec("SCMND", self.network, scmnd_formulation=SCMND_CURRENT)
        raw = np.array([0., -100., -200., 0., -100., -200., -50., 1000., -20.])
        for v in codec.structure_aware_candidates(raw).values():
            self.assertTrue(codec.is_feasible(v))
            alpha = v[-codec._n_arcs:]
            self.assertTrue(np.all(alpha >= 0))
            self.assertTrue(np.all(alpha <= codec.rho + 1e-10))

    def test_exact_zero_gamma_can_still_have_structural_improvement_scflp(self):
        # Anchor at mu=20, nu=0 is at a dual boundary. Raw nu<0 forces
        # gamma=0, yet lifting nu from ML raw produces a better feasible
        # dual at a small first-stage facility activation x.
        data = SimpleNamespace(
            I=[0], J=[0], cost=[[20.0]],
            scenarios=[[10.0], [10.0]],
            capacity=[20.0], capacity_scenarios=[[20.0], [20.0]],
        )
        raw = np.array([22.0, -3.0])
        kwargs = {
            "variant": PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
            "problem_type": "SCFLP", "data": data,
        }
        pred = {}
        for mode in ("LOCAL", "STRUCTURE_AWARE"):
            p = ParametricDualPredictor(
                **kwargs,
                parameters={
                    "parametric_repair_mode": mode,
                    "parametric_objective_guard": True,
                    "parametric_guard_gamma_zero_fastpath": True,
                },
            )
            p.selected = [0]
            p.mapping = {1: {"representative": 0}}
            p.counts = {0: 1.0}
            p._raw_predictions = lambda ids, model, x: {
                int(sid): raw.copy() for sid in ids
            }
            self.assertTrue(p.prepare_iteration(
                {0: {"mu": {0: 20.0}, "nu": {0: 0.0}}},
                {0: 0.2}, [1], iteration=1,
            ))
            pred[mode] = p
        self.assertEqual(pred["LOCAL"].diagnostics()["full_reversions"], 1)
        self.assertEqual(pred["LOCAL"].predict([1])[1]["mu"][0], 20.0)
        repaired = pred["STRUCTURE_AWARE"].predict([1])[1]
        self.assertEqual(repaired["mu"][0], 22.0)
        self.assertEqual(repaired["nu"][0], 2.0)
        self.assertEqual(
            pred["STRUCTURE_AWARE"].diagnostics()["objective_guard_accepted"], 1
        )
        self.assertEqual(
            pred["STRUCTURE_AWARE"].diagnostics()["structure_repair_positive_objectives"], 1
        )

    def test_exact_zero_gamma_can_still_have_structural_improvement_scmnd(self):
        data = SimpleNamespace(
            N=[0, 1], K=[0], A=[(0, 1)],
            C=[[9.0]], origins={0: 0}, destinations={0: 1},
            demand_scenarios=[[10.0], [10.0]],
            u_base=[20.0], capacity_scenarios=[[15.0], [15.0]],
            slack_penalty_multiplier=2.0,
        )
        for form in (SCMND_MANUSCRIPT, SCMND_CURRENT):
            for mode in ("LOCAL", "STRUCTURE_AWARE"):
                p = ParametricDualPredictor(
                    PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
                    "SCMND", data,
                    parameters={
                        "scmnd_formulation": form,
                        "parametric_repair_mode": mode,
                        "parametric_objective_guard": True,
                        "parametric_guard_gamma_zero_fastpath": True,
                    },
                )
                p.selected = [0]
                p.mapping = {1: {"representative": 0}}
                p.counts = {0: 1.0}
                p._raw_predictions = lambda ids, model, x: {
                    int(sid): np.array([0.0, -14.0, -1.0]) for sid in ids
                }
                self.assertTrue(p.prepare_iteration(
                    {0: {"pi": {(0, 0): 0.0, (1, 0): -9.0}, "alpha": {0: 0.0}}},
                    {0: 0.0}, [1], iteration=1,
                ))
                if mode == "LOCAL":
                    self.assertEqual(p.diagnostics()["full_reversions"], 1)
                    self.assertEqual(p.diagnostics()["objective_guard_accepted"], 0)
                else:
                    self.assertGreater(
                        p.diagnostics()["structure_repair_positive_objectives"], 0
                    )
                    self.assertEqual(p.diagnostics()["objective_guard_accepted"], 1)
                    self.assertTrue(p.codec.is_feasible(
                        p.codec.flatten(p.predict([1])[1])
                    ))

    def test_feasible_predictions_remain_feasible_when_strengthened(self):
        sc = DualVectorCodec("SCFLP", self.scflp)
        # Feasible: mu_i=2, nu_j=0, costs >= 4.
        sc_raw = np.array([2.0, 2.0, 0.0, 0.0])
        self.assertTrue(sc.is_feasible(sc_raw))
        for candidate in sc.structure_aware_candidates(sc_raw).values():
            self.assertTrue(sc.is_feasible(candidate))
            self.assertGreaterEqual(
                np.dot(sc.objective_coefficients(0, {0: 0.2, 1: 0.2}), candidate),
                np.dot(sc.objective_coefficients(0, {0: 0.2, 1: 0.2}), sc_raw),
            )

        for form in (SCMND_MANUSCRIPT, SCMND_CURRENT):
            nw = DualVectorCodec("SCMND", self.network, scmnd_formulation=form)
            # Identically zero potentials and multipliers are feasible
            # because all test network arc costs are nonnegative.
            raw = np.zeros(nw.dimension)
            self.assertTrue(nw.is_feasible(raw))
            for candidate in nw.structure_aware_candidates(raw).values():
                np.testing.assert_allclose(candidate, raw)

    def test_scflp_completion_dominates_partial_repair(self):
        codec = DualVectorCodec("SCFLP", self.scflp)
        raw = np.array([2.0, 2.0, 0.0, 0.0])
        g = codec.objective_coefficients(0, {0: 0.2, 1: 0.2})
        result = codec.structure_aware_candidates(raw)
        self.assertTrue(codec.is_feasible(raw))
        self.assertTrue(codec.is_feasible(result["scflp_clip_mu"]))
        self.assertGreater(
            np.dot(g, result["scflp_clip_mu"]), np.dot(g, raw)
        )
        np.testing.assert_allclose(
            result["scflp_clip_mu"][:2], [5.0, 4.0]
        )

    def test_scmnd_multiplier_completion_dominates_surplus(self):
        for form in (SCMND_MANUSCRIPT, SCMND_CURRENT):
            codec = DualVectorCodec("SCMND", self.network, scmnd_formulation=form)
            raw = np.concatenate([
                np.zeros(codec._n_commodities * codec._n_nodes),
                np.full(codec._n_arcs, 10.0),
            ])
            g = codec.objective_coefficients(0, {a: 1.0 for a in range(codec._n_arcs)})
            self.assertTrue(codec.is_feasible(raw))
            for v in codec.structure_aware_candidates(raw).values():
                self.assertTrue(codec.is_feasible(v))
                np.testing.assert_allclose(v[-codec._n_arcs:], 0.0)
                self.assertGreater(np.dot(g, v), np.dot(g, raw))

    def test_closure_dominates_original_candidates_on_randomized_duals(self):
        rng = np.random.default_rng(20261019)
        sc = DualVectorCodec("SCFLP", self.scflp)
        g_sc = sc.objective_coefficients(0, {0: 0.3, 1: 0.7})
        for _ in range(50):
            raw = rng.normal(0, 12, sc.dimension)
            mu = raw[:sc._n_customers]
            nu = raw[sc._n_customers:]
            lifted_nu = np.maximum.reduce([
                np.maximum(nu, 0),
                np.max(np.maximum(mu, 0)[:, None] - sc._scflp_cost, axis=0),
            ])
            old_lift = np.concatenate([np.maximum(mu, 0), lifted_nu])
            kept_nu = np.maximum(nu, 0)
            old_clip = np.concatenate([
                np.minimum(
                    np.maximum(mu, 0),
                    np.min(sc._scflp_cost + kept_nu[None, :], axis=1),
                ),
                kept_nu,
            ])
            new = sc.structure_aware_candidates(raw)
            for label, old in (
                ("scflp_lift_nu", old_lift),
                ("scflp_clip_mu", old_clip),
            ):
                self.assertTrue(sc.is_feasible(new[label]))
                self.assertGreaterEqual(
                    float(np.dot(g_sc, new[label])) + 1e-8,
                    float(np.dot(g_sc, old)),
                )

        nw = DualVectorCodec("SCMND", self.network, scmnd_formulation=SCMND_MANUSCRIPT)
        g_nw = nw.objective_coefficients(0, {a: 0.6 for a in range(nw._n_arcs)})
        for _ in range(50):
            raw = nw.canonicalize(rng.normal(0, 12, nw.dimension))
            pi_end = nw._n_nodes * nw._n_commodities
            pi = raw[:pi_end].reshape(nw._n_commodities, nw._n_nodes)
            required = np.max(
                (
                    pi[:, nw._tail_index] - pi[:, nw._head_index]
                ).T - nw._network_cost,
                axis=1,
            )
            old_alpha = np.maximum.reduce([
                np.zeros(nw._n_arcs), raw[pi_end:], required,
            ])
            old = np.concatenate([raw[:pi_end], old_alpha])
            new = nw.structure_aware_candidates(raw)["scmnd_lift_alpha"]
            self.assertTrue(nw.is_feasible(new))
            self.assertGreaterEqual(
                float(np.dot(g_nw, new)) + 1e-8, float(np.dot(g_nw, old))
            )

    def test_randomized_nonnegative_cost_feasibility(self):
        rng = np.random.default_rng(150)
        sc = DualVectorCodec("SCFLP", self.scflp)
        for _ in range(70):
            raw = rng.normal(0, 25, sc.dimension)
            for v in sc.structure_aware_candidates(raw).values():
                self.assertTrue(sc.is_feasible(v), v)
        for form in (SCMND_MANUSCRIPT, SCMND_CURRENT):
            nw = DualVectorCodec("SCMND", self.network, scmnd_formulation=form)
            for _ in range(70):
                raw = rng.normal(0, 25, nw.dimension)
                for v in nw.structure_aware_candidates(raw).values():
                    self.assertTrue(nw.is_feasible(v), (form, v))

    def test_structure_identity_shortcut_retains_same_certified_anchor(self):
        # A feasible predicted dual that equals the selected anchor needs no
        # repeated objective evaluation, and must not count as gamma=0.
        data = SimpleNamespace(
            I=[0], J=[0], cost=[[20.0]],
            scenarios=[[10.0], [10.0]], capacity=[20.0],
        )
        for fast in (False, True):
            p = ParametricDualPredictor(
                PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
                "SCFLP", data,
                parameters={
                    "parametric_repair_mode": "STRUCTURE_AWARE",
                    "parametric_objective_guard": True,
                    "parametric_guard_gamma_zero_fastpath": fast,
                },
            )
            p.selected = [0]
            p.mapping = {1: {"representative": 0}}
            p.counts = {0: 1.0}
            p._raw_predictions = lambda ids, model, x: {
                int(sid): np.array([20.0, 0.0]) for sid in ids
            }
            self.assertTrue(p.prepare_iteration(
                {0: {"mu": {0: 20.0}, "nu": {0: 0.0}}},
                {0: 0.2}, [1], iteration=1,
            ))
            self.assertEqual(p.predict([1])[1]["mu"][0], 20.0)
            d = p.diagnostics()
            self.assertEqual(d["objective_guard_checks"], 1)
            self.assertEqual(d["objective_guard_accepted"], 0)
            self.assertEqual(d["objective_guard_skipped_zero"], 0)
            self.assertEqual(d["objective_guard_skipped_equal_anchor"], int(fast))
            self.assertEqual(d["objective_guard_evaluated"], int(not fast))

    def test_nonfinite_raw_candidate_generation_falls_back(self):
        codec = DualVectorCodec("SCFLP", self.scflp)
        self.assertEqual(codec.structure_aware_candidates([np.nan]*4), {})

    def test_invalid_repair_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "parametric_repair_mode must be"):
            ParametricDualPredictor(
                PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
                "SCFLP", self.scflp,
                parameters={"parametric_repair_mode": "PROJECT_QP"},
            )


if __name__ == "__main__":
    unittest.main()
