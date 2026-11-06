"""Parametric dual predictors for Learning-Augmented Benders Decomposition.

The predictors in this module are predictor choices inside
``ML_Augmented_BD``.  The original parametric variants use a feature-space
p-median partition.  The cross-weight RHS-Ridge variants reuse the same
Tree/Clustering transfer structures as the nonparametric methods so the
weighting mechanism can be separated experimentally from the residual model.

Supported predictor names
-------------------------
``PARAMETRIC_MANUSCRIPT``
    The affine model in Model ``parametric-training-main``: hard value- and
    dual-error budgets and hard feasibility for every prediction subproblem.

``PARAMETRIC_PENALIZED_ERRORS``
    A fixed feasible intercept, errors in the objective, and hard feasibility.

``PARAMETRIC_RIDGE_REPAIR``
    Iteration-specific ridge regression followed by analytical line-segment
    feasibility repair and explicit certification.

``PARAMETRIC_RHS_RIDGE_REPAIR``
    A warm-up-trained ridge model using both static features and the current
    dual-objective right-hand side, followed by the same repair/certification.

``PARAMETRIC_RHS_RIDGE_TREE_ANCHOR``
    The same warm-up-trained RHS-Ridge residual model using the TREE transfer
    weights as its feasible, scenario-specific anchor.

``PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR``
    The same warm-up-trained RHS-Ridge residual model using the CLUSTERING
    transfer weights as its feasible, scenario-specific anchor.

``PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN`` (IPCO 2027 version)
    A warm-up-trained ridge model f_theta(z_m, g_m(x)) that predicts the dual
    vector itself (no transfer baseline, no residual target). The map is linear
    with no intercept, on unstandardized z_m and an uncentred low-dimensional
    projection of g_m(x). A raw prediction is made feasible along the segment
    from the mean of the current exact duals in M_e (equal weights) toward the
    raw prediction, using the same closed-form step and objective guard.
    M_e is chosen by the feature-space p-median.

The numerical ridge and feasibility routines do not require Gurobi.  Gurobi
is required only for the p-median MIP and the two constrained training models.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Keep numerical unit tests runnable without a local Gurobi installation.
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised on development machines without Gurobi.
    gp = None
    GRB = None


from utils.exclusive_profiler import profiled, profile_section

from utils.exclusive_profiler import get_profiled_logger

logger = get_profiled_logger(__name__)


# Exact duals come from Gurobi's dual LP, which satisfies its constraints only
# up to Gurobi's default FeasibilityTol (1e-6). Anchors built from those duals
# are therefore certified at this tolerance everywhere, including when an
# anchor is re-checked inside the line-segment repair and the objective guard
# (those two re-checks previously used the stricter 1e-7 and could raise on an
# anchor that had already passed certification). Only anchors whose violation
# lies in (1e-7, 1e-6] are affected; every run that completed before this
# change is unaffected because its anchors passed the stricter check.
ANCHOR_FEASIBILITY_TOLERANCE = 1e-6

PARAMETRIC_MANUSCRIPT = "PARAMETRIC_MANUSCRIPT"
PARAMETRIC_PENALIZED_ERRORS = "PARAMETRIC_PENALIZED_ERRORS"
PARAMETRIC_RIDGE_REPAIR = "PARAMETRIC_RIDGE_REPAIR"
PARAMETRIC_RHS_RIDGE_REPAIR = "PARAMETRIC_RHS_RIDGE_REPAIR"
PARAMETRIC_RHS_RIDGE_TREE_ANCHOR = "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR"
PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR = "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR"
PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR = "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"
# IPCO 2027: direct dual prediction with a mean-of-exact-duals reference point.
PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN = "PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN"
# Variant (not adopted; branch variant/random-Me): M_e drawn uniformly at random.
PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN = "PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN"

# Comparison variants (not in the IPCO paper): no repair; an infeasible raw
# prediction is solved exactly instead (the callback's per-subproblem fallback).
PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_EXACTFB = "PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_EXACTFB"
PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN_EXACTFB = "PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN_EXACTFB"
EXACT_FALLBACK_PREDICTORS = frozenset(
    {PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_EXACTFB, PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN_EXACTFB}
)
# First IPCO version, kept for comparison: repair starts at the mean itself and
# negative entries are not clipped (eps = 0, no clipping).
PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_V0 = "PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_V0"
PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN_V0 = "PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN_V0"
START_AT_MEAN_PREDICTORS = frozenset(
    {PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_V0, PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN_V0}
)
RANDOM_SELECTION_PREDICTORS = frozenset(
    {PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN, PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN_EXACTFB,
     PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN_V0}
)

DIRECT_RIDGE_PREDICTORS = frozenset(
    {PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN, PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN,
     *EXACT_FALLBACK_PREDICTORS, *START_AT_MEAN_PREDICTORS}
)

RHS_RIDGE_PREDICTORS = frozenset(
    {
        PARAMETRIC_RHS_RIDGE_REPAIR,
        PARAMETRIC_RHS_RIDGE_TREE_ANCHOR,
        PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR,
        PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR,
        *DIRECT_RIDGE_PREDICTORS,
    }
)

PARAMETRIC_PREDICTORS = frozenset(
    {
        PARAMETRIC_MANUSCRIPT,
        PARAMETRIC_PENALIZED_ERRORS,
        PARAMETRIC_RIDGE_REPAIR,
        *RHS_RIDGE_PREDICTORS,
    }
)

SCMND_CURRENT = "CURRENT_SOFT_CAPACITY"
SCMND_MANUSCRIPT = "MANUSCRIPT_ENVELOPE"
SCMND_FORMULATIONS = frozenset({SCMND_CURRENT, SCMND_MANUSCRIPT})


def is_parametric_predictor(name: Optional[str]) -> bool:
    return str(name).upper() in PARAMETRIC_PREDICTORS if name is not None else False


def normalize_scmnd_formulation(value: Optional[str]) -> str:
    value = str(value or SCMND_CURRENT).upper()
    aliases = {
        "CURRENT": SCMND_CURRENT,
        "CURRENT_SOFT_CAPACITY": SCMND_CURRENT,
        "SOFT_CAPACITY": SCMND_CURRENT,
        "MANUSCRIPT": SCMND_MANUSCRIPT,
        "MANUSCRIPT_ENVELOPE": SCMND_MANUSCRIPT,
        "ENVELOPE": SCMND_MANUSCRIPT,
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown SCMND formulation '{value}'. Expected one of "
            f"{sorted(SCMND_FORMULATIONS)}."
        )
    return aliases[value]


@dataclass
class FeatureScaler:
    """Deterministic weighted centering/scaling of the manuscript features."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    @profiled("feature_scaling_transformation")
    def fit(
        cls,
        feature_vectors: Mapping[int, Sequence[float]],
        ids: Optional[Sequence[int]] = None,
        weights: Optional[Mapping[int, float]] = None,
    ) -> "FeatureScaler":
        ids = list(ids if ids is not None else sorted(feature_vectors))
        if not ids:
            raise ValueError("Cannot fit a feature scaler without observations.")
        matrix = np.asarray([feature_vectors[sid] for sid in ids], dtype=float)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("Feature vectors must form a finite two-dimensional matrix.")
        raw_weights = np.asarray(
            [float((weights or {}).get(sid, 1.0)) for sid in ids], dtype=float
        )
        raw_weights = np.maximum(raw_weights, 0.0)
        if raw_weights.sum() <= 0.0:
            raw_weights[:] = 1.0
        raw_weights /= raw_weights.sum()
        mean = raw_weights @ matrix
        variance = raw_weights @ np.square(matrix - mean)
        scale = np.sqrt(np.maximum(variance, 0.0))
        scale[scale <= 1e-12] = 1.0
        return cls(mean=np.asarray(mean, dtype=float), scale=np.asarray(scale, dtype=float))

    @profiled("feature_scaling_transformation")
    def transform(self, value: Sequence[float]) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        return (value - self.mean) / self.scale


class DualVectorCodec:
    """Flatten, rebuild, normalize, evaluate, and certify structured duals.

    SCFLP and SCMND both have a scenario-independent feasible region in the
    implementations supported here. Scenario information and the current
    first-stage point enter only through the dual-objective coefficient vector.

    Feasibility checking and analytical line-segment repair deliberately use
    vectorized NumPy expressions.  This changes only how the same inequalities
    are evaluated; the feasible region, tolerance, repair gamma, and final
    certification are unchanged.
    """

    def __init__(
        self,
        problem_type: str,
        data,
        use_stochastic_capacity: bool = False,
        scmnd_formulation: str = SCMND_CURRENT,
    ):
        self.problem_type = str(problem_type).upper()
        self.data = data
        self.use_stochastic_capacity = bool(use_stochastic_capacity)
        self.scmnd_formulation = normalize_scmnd_formulation(scmnd_formulation)

        if self.problem_type == "SCFLP":
            self.customers = [int(i) for i in data.I]
            self.facilities = [int(j) for j in data.J]
            self.customer_pos = {i: pos for pos, i in enumerate(self.customers)}
            self.facility_pos = {
                j: len(self.customers) + pos for pos, j in enumerate(self.facilities)
            }
            self.dimension = len(self.customers) + len(self.facilities)
            self._n_customers = len(self.customers)
            self._n_facilities = len(self.facilities)
            self._scflp_cost = np.asarray(self.data.cost, dtype=float)
        elif self.problem_type in {"SUMND", "SCMND"}:
            self.nodes = [int(i) for i in data.N]
            self.commodities = [int(k) for k in data.K]
            self.pi_pos = {}
            pos = 0
            for k in self.commodities:
                for i in self.nodes:
                    self.pi_pos[(i, k)] = pos
                    pos += 1
            self.alpha_pos = {a: pos + a for a in range(len(data.A))}
            self.dimension = pos + len(data.A)
            self.rho = None
            if self.problem_type == "SCMND" and self.scmnd_formulation == SCMND_CURRENT:
                multiplier = float(getattr(data, "slack_penalty_multiplier", 2.0))
                self.rho = multiplier * np.max(np.asarray(data.C, dtype=float), axis=1)
            self._n_nodes = len(self.nodes)
            self._n_commodities = len(self.commodities)
            self._n_arcs = len(self.data.A)
            node_index = {node: idx for idx, node in enumerate(self.nodes)}
            self._tail_index = np.asarray(
                [node_index[int(tail)] for tail, _ in self.data.A], dtype=int
            )
            self._head_index = np.asarray(
                [node_index[int(head)] for _, head in self.data.A], dtype=int
            )
            self._origin_index = np.asarray(
                [node_index[int(self.data.origins[k])] for k in self.commodities], dtype=int
            )
            self._network_cost = np.asarray(self.data.C, dtype=float)
        else:
            raise NotImplementedError(
                "Parametric dual prediction is currently implemented for SCFLP and SCMND."
            )

    def flatten(self, duals: Mapping, canonicalize: bool = True) -> np.ndarray:
        if self.problem_type == "SCFLP":
            vector = np.empty(self.dimension, dtype=float)
            for i in self.customers:
                vector[self.customer_pos[i]] = float(duals["mu"][i])
            for j in self.facilities:
                vector[self.facility_pos[j]] = float(duals["nu"][j])
        else:
            vector = np.empty(self.dimension, dtype=float)
            for key, pos in self.pi_pos.items():
                vector[pos] = float(duals["pi"][key])
            for arc_idx, pos in self.alpha_pos.items():
                vector[pos] = float(duals["alpha"][arc_idx])
        if not np.all(np.isfinite(vector)):
            raise ValueError("Dual vector contains a non-finite coordinate.")
        return self.canonicalize(vector) if canonicalize else vector

    def unflatten(self, vector: Sequence[float]) -> Dict:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Expected a dual vector of length {self.dimension}, got {vector.shape}."
            )
        if self.problem_type == "SCFLP":
            return {
                "mu": {i: float(vector[self.customer_pos[i]]) for i in self.customers},
                "nu": {j: float(vector[self.facility_pos[j]]) for j in self.facilities},
            }
        return {
            "pi": {key: float(vector[pos]) for key, pos in self.pi_pos.items()},
            "alpha": {a: float(vector[pos]) for a, pos in self.alpha_pos.items()},
        }

    def canonicalize(self, vector: Sequence[float]) -> np.ndarray:
        """Remove the harmless commodity-wise additive potential freedom."""
        result = np.asarray(vector, dtype=float).copy()
        if self.problem_type in {"SUMND", "SCMND"}:
            pi = result[: self._n_commodities * self._n_nodes].reshape(
                self._n_commodities, self._n_nodes
            )
            pi -= pi[np.arange(self._n_commodities), self._origin_index][:, None]
        return result

    def objective_coefficients(self, scenario_id: int, first_stage_values) -> np.ndarray:
        """Return g so that the code's dual objective equals ``lambda @ g``."""
        sid = int(scenario_id)
        g = np.zeros(self.dimension, dtype=float)
        if self.problem_type == "SCFLP":
            demand = np.asarray(self.data.scenarios[sid], dtype=float)
            if (
                self.use_stochastic_capacity
                and getattr(self.data, "capacity_scenarios", None) is not None
                and sid < len(self.data.capacity_scenarios)
            ):
                capacities = np.asarray(self.data.capacity_scenarios[sid], dtype=float)
            else:
                capacities = np.asarray(self.data.capacity, dtype=float)
            for i in self.customers:
                g[self.customer_pos[i]] = demand[i]
            for j in self.facilities:
                g[self.facility_pos[j]] = -capacities[j] * float(first_stage_values[j])
            return g

        demand = np.asarray(self.data.demand_scenarios[sid], dtype=float)
        for k in self.commodities:
            origin = int(self.data.origins[k])
            destination = int(self.data.destinations[k])
            g[self.pi_pos[(origin, k)]] += demand[k]
            g[self.pi_pos[(destination, k)]] -= demand[k]
        if self.problem_type == "SCMND" and self.scmnd_formulation == SCMND_MANUSCRIPT:
            capacities = np.asarray(self.data.u_base, dtype=float)
        else:
            capacities = np.asarray(self.data.capacity_scenarios[sid], dtype=float)
        for arc_idx in range(len(self.data.A)):
            g[self.alpha_pos[arc_idx]] = -capacities[arc_idx] * float(
                first_stage_values[arc_idx]
            )
        return g

    @staticmethod
    def _max_positive(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return 0.0
        return max(0.0, float(np.max(values)))

    @profiled("feasibility_check")
    def max_violation(self, vector: Sequence[float]) -> float:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            return math.inf

        if self.problem_type == "SCFLP":
            mu = vector[: self._n_customers]
            nu = vector[self._n_customers : self._n_customers + self._n_facilities]
            nonneg_violation = max(
                self._max_positive(-mu),
                self._max_positive(-nu),
            )
            arc_violation = self._max_positive(
                mu[:, None] - nu[None, :] - self._scflp_cost
            )
            return max(nonneg_violation, arc_violation)

        pi = vector[: self._n_commodities * self._n_nodes].reshape(
            self._n_commodities, self._n_nodes
        )
        alpha = vector[
            self._n_commodities * self._n_nodes :
            self._n_commodities * self._n_nodes + self._n_arcs
        ]
        violation = self._max_positive(-alpha)
        if self.rho is not None:
            violation = max(violation, self._max_positive(alpha - self.rho))
        arc_lhs = (
            (pi[:, self._tail_index] - pi[:, self._head_index]).T
            - alpha[:, None]
            - self._network_cost
        )
        return max(violation, self._max_positive(arc_lhs))

    @profiled("feasibility_repair_structure_aware")
    def structure_aware_candidates(
        self, raw_vector: Sequence[float],
    ) -> Dict[str, np.ndarray]:
        """Construct feasible objective-completed candidates with no solver.

        Feasibility is certified by the caller, including in cases where
        negative network costs invalidate the nonnegative-cost scaling
        construction. These operators exploit the particular SCFLP and SCMND
        dual polyhedra; they are not Euclidean projections.
        """
        raw = self.canonicalize(raw_vector)
        if not np.all(np.isfinite(raw)):
            return {}

        if self.problem_type == "SCFLP":
            mu = raw[:self._n_customers]
            nu = raw[self._n_customers:]
            # Candidate A: keep positive customer potentials, lift facility
            # multipliers just enough to satisfy every linking inequality.
            keep_mu = np.maximum(mu, 0.0)
            lift_nu = np.maximum(
                np.maximum(nu, 0.0),
                np.max(keep_mu[:, None] - self._scflp_cost, axis=0),
            )
            # Candidate B: keep nonnegative raw facility multipliers;
            # complete the corresponding customer potentials at their caps.
            keep_nu = np.maximum(nu, 0.0)
            # Objective-aware closure: with nu fixed, mu_i has a
            # nonnegative scenario-demand coefficient and is best set to
            # its largest feasible value. The prior candidates leave mu
            # below this bound even when doing so cannot help the objective.
            # These completions weakly dominate the prior candidates for
            # nonnegative demand; the final feasibility and Borzou
            # objective-dominance guards remain independent.
            complete_lift_mu = np.maximum(
                0.0, np.min(self._scflp_cost + lift_nu[None, :], axis=1)
            )
            complete_clip_mu = np.maximum(
                0.0, np.min(self._scflp_cost + keep_nu[None, :], axis=1)
            )
            return {
                "scflp_lift_nu": np.concatenate([complete_lift_mu, lift_nu]),
                "scflp_clip_mu": np.concatenate([complete_clip_mu, keep_nu]),
            }

        # SCMND has free commodity potentials pi and nonnegative arc alpha.
        pi_end = self._n_commodities * self._n_nodes
        pi = raw[:pi_end].reshape(self._n_commodities, self._n_nodes)
        alpha_raw = raw[pi_end:pi_end + self._n_arcs]
        delta = (pi[:, self._tail_index] - pi[:, self._head_index]).T
        required_alpha = np.max(delta - self._network_cost, axis=1)
        lift = np.maximum.reduce([
            np.zeros_like(alpha_raw), alpha_raw, required_alpha,
        ])

        if self.rho is None:
            # Manuscript dual: alpha has no upper bound. A single
            # arc-wise maximum restores every linking inequality.
            # With pi fixed, alpha_a need only cover the maximum arc
            # violation. Any larger alpha has a nonpositive objective
            # coefficient (-capacity_a * x_a), so retain no surplus.
            tight_alpha = np.maximum(0.0, required_alpha)
            return {
                "scmnd_lift_alpha": self.canonicalize(
                    np.concatenate([pi.reshape(-1), tight_alpha])
                )
            }

        # Current soft-capacity dual: alpha has 0 <= alpha <= rho.
        # With nonnegative arc costs the zero-potential vector is feasible,
        # so a commodity-specific s_k in [0,1] yields feasible potentials:
        # s_k * (pi_tail,k - pi_head,k) <= C[a,k] + alpha[a].
        # We validate the result regardless of the cost sign.
        def scaled_with_alpha(alpha: np.ndarray) -> np.ndarray:
            allowance = self._network_cost + alpha[:, None]
            active = delta > 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(active, allowance / delta, np.inf)
            scales = np.clip(np.minimum(1.0, np.min(ratios, axis=0)), 0.0, 1.0)
            adjusted_pi = pi * scales[:, None]
            adjusted_delta = (
                adjusted_pi[:, self._tail_index]
                - adjusted_pi[:, self._head_index]
            ).T
            # The original alpha makes adjusted_pi feasible whenever
            # this candidate is valid. Tightening to the smallest
            # necessary alpha preserves its linking constraints and
            # weakly improves the dual objective for nonnegative
            # first-stage capacities.
            tight_alpha = np.maximum(
                0.0, np.max(adjusted_delta - self._network_cost, axis=1)
            )
            return self.canonicalize(
                np.concatenate([adjusted_pi.reshape(-1), tight_alpha])
            )

        clipped = np.clip(alpha_raw, 0.0, self.rho)
        lifted = np.minimum(lift, self.rho)
        return {
            "scmnd_clip_scale": scaled_with_alpha(clipped),
            "scmnd_lift_scale": scaled_with_alpha(lifted),
        }

    @profiled("feasibility_check")
    def is_feasible(self, vector: Sequence[float], tolerance: float = 1e-7) -> bool:
        return self.max_violation(vector) <= float(tolerance)

    @staticmethod
    def _tighten_step(bound: float, safety: float) -> float:
        if not np.isfinite(bound):
            return 1.0
        return max(0.0, min(1.0, float(bound) - float(safety)))

    @classmethod
    def _vectorized_gamma_bound(
        cls,
        slack: np.ndarray,
        change: np.ndarray,
        safety: float,
    ) -> float:
        slack = np.asarray(slack, dtype=float)
        change = np.asarray(change, dtype=float)
        active = change > 1e-14
        if not np.any(active):
            return 1.0
        ratio = slack[active] / change[active]
        finite = ratio[np.isfinite(ratio)]
        if finite.size == 0:
            return 1.0
        return cls._tighten_step(float(np.min(finite)), safety)

    @profiled("feasibility_repair_line_segment")
    def repair_along_segment(
        self,
        feasible_vector: Sequence[float],
        raw_vector: Sequence[float],
        feasibility_tolerance: float = 1e-7,
        safety: float = 1e-10,
    ) -> Tuple[np.ndarray, float, bool, bool]:
        """Move from a feasible vector toward a raw prediction as far as possible.

        This is the same analytical line-segment repair as before, evaluated
        with vectorized array operations instead of Python loops.

        Returns ``(repaired, gamma, raw_was_feasible, reverted_to_feasible_vector)``.
        """
        base = self.canonicalize(feasible_vector)
        raw = self.canonicalize(raw_vector)
        start_tolerance = max(float(feasibility_tolerance), ANCHOR_FEASIBILITY_TOLERANCE)
        if not self.is_feasible(base, start_tolerance):
            raise ValueError("The line-segment starting vector is not dual feasible.")
        raw_feasible = self.is_feasible(raw, feasibility_tolerance)
        if raw_feasible:
            return raw, 1.0, True, False

        direction = raw - base
        gamma = 1.0

        if self.problem_type == "SCFLP":
            base_mu = base[: self._n_customers]
            base_nu = base[
                self._n_customers : self._n_customers + self._n_facilities
            ]
            dir_mu = direction[: self._n_customers]
            dir_nu = direction[
                self._n_customers : self._n_customers + self._n_facilities
            ]
            gamma = min(
                gamma,
                self._vectorized_gamma_bound(base_mu, -dir_mu, safety),
                self._vectorized_gamma_bound(base_nu, -dir_nu, safety),
            )
            lhs_base = base_mu[:, None] - base_nu[None, :]
            lhs_change = dir_mu[:, None] - dir_nu[None, :]
            gamma = min(
                gamma,
                self._vectorized_gamma_bound(
                    self._scflp_cost - lhs_base,
                    lhs_change,
                    safety,
                ),
            )
        else:
            pi_end = self._n_commodities * self._n_nodes
            base_pi = base[:pi_end].reshape(self._n_commodities, self._n_nodes)
            dir_pi = direction[:pi_end].reshape(self._n_commodities, self._n_nodes)
            base_alpha = base[pi_end : pi_end + self._n_arcs]
            dir_alpha = direction[pi_end : pi_end + self._n_arcs]
            gamma = min(
                gamma,
                self._vectorized_gamma_bound(base_alpha, -dir_alpha, safety),
            )
            if self.rho is not None:
                gamma = min(
                    gamma,
                    self._vectorized_gamma_bound(
                        self.rho - base_alpha,
                        dir_alpha,
                        safety,
                    ),
                )
            lhs_base = (
                base_pi[:, self._tail_index] - base_pi[:, self._head_index]
            ).T - base_alpha[:, None]
            lhs_change = (
                dir_pi[:, self._tail_index] - dir_pi[:, self._head_index]
            ).T - dir_alpha[:, None]
            gamma = min(
                gamma,
                self._vectorized_gamma_bound(
                    self._network_cost - lhs_base,
                    lhs_change,
                    safety,
                ),
            )

        gamma = max(0.0, min(1.0, gamma))
        repaired = self.canonicalize(base + gamma * direction)
        if self.is_feasible(repaired, feasibility_tolerance):
            return repaired, gamma, False, False
        return base, 0.0, False, True

    def add_gurobi_feasibility_constraints(self, model, expressions, name_prefix: str):
        if gp is None:
            raise RuntimeError("Gurobi is required for constrained parametric training.")
        if self.problem_type == "SCFLP":
            for i in self.customers:
                model.addConstr(expressions[self.customer_pos[i]] >= 0.0,
                                name=f"{name_prefix}_mu_nonneg_{i}")
            for j in self.facilities:
                model.addConstr(expressions[self.facility_pos[j]] >= 0.0,
                                name=f"{name_prefix}_nu_nonneg_{j}")
            costs = np.asarray(self.data.cost, dtype=float)
            for i in self.customers:
                for j in self.facilities:
                    model.addConstr(
                        expressions[self.customer_pos[i]]
                        - expressions[self.facility_pos[j]] <= costs[i, j],
                        name=f"{name_prefix}_arc_{i}_{j}",
                    )
            return

        for arc_idx in range(len(self.data.A)):
            alpha = expressions[self.alpha_pos[arc_idx]]
            model.addConstr(alpha >= 0.0, name=f"{name_prefix}_alpha_nonneg_{arc_idx}")
            if self.rho is not None:
                model.addConstr(alpha <= self.rho[arc_idx],
                                name=f"{name_prefix}_alpha_upper_{arc_idx}")
        costs = np.asarray(self.data.C, dtype=float)
        for arc_idx, (tail, head) in enumerate(self.data.A):
            tail = int(tail)
            head = int(head)
            for k in self.commodities:
                model.addConstr(
                    expressions[self.pi_pos[(tail, k)]]
                    - expressions[self.pi_pos[(head, k)]]
                    - expressions[self.alpha_pos[arc_idx]] <= costs[arc_idx, k],
                    name=f"{name_prefix}_arc_{arc_idx}_{k}",
                )
        for k in self.commodities:
            origin = int(self.data.origins[k])
            model.addConstr(
                expressions[self.pi_pos[(origin, k)]] == 0.0,
                name=f"{name_prefix}_normalization_{k}",
            )


@profiled("representative_assignment_nearest_weighted")
def representative_counts(
    selected: Iterable[int], mapping: Mapping[int, Mapping]
) -> Dict[int, int]:
    counts = {int(sid): 0 for sid in selected}
    for info in mapping.values():
        if not isinstance(info, Mapping) or "representative" not in info:
            continue
        representative = int(info["representative"])
        if representative in counts:
            counts[representative] += 1
    return counts


@profiled("representative_assignment_nearest_weighted")
def weighted_feasible_average(
    codec: DualVectorCodec,
    solved_duals: Mapping[int, Mapping],
    counts: Optional[Mapping[int, float]] = None,
) -> np.ndarray:
    if not solved_duals:
        raise ValueError("A feasible average requires at least one exact dual solution.")
    ids = sorted(int(sid) for sid in solved_duals)
    weights = np.asarray([max(0.0, float((counts or {}).get(sid, 1.0))) for sid in ids])
    if weights.sum() <= 0.0:
        weights[:] = 1.0
    weights /= weights.sum()
    matrix = np.asarray([codec.flatten(solved_duals[sid]) for sid in ids], dtype=float)
    average = codec.canonicalize(weights @ matrix)
    if not codec.is_feasible(average, ANCHOR_FEASIBILITY_TOLERANCE):
        raise ValueError("The weighted average of exact duals failed feasibility certification.")
    return average


def _distance_matrix(feature_vectors: Mapping[int, Sequence[float]], ids: Sequence[int]) -> np.ndarray:
    matrix = np.asarray([feature_vectors[sid] for sid in ids], dtype=float)
    difference = matrix[:, None, :] - matrix[None, :, :]
    return np.linalg.norm(difference, axis=2)


def _greedy_pmedian(distance: np.ndarray, p: int) -> Sequence[int]:
    """Deterministic build-and-swap heuristic, used only when explicitly requested."""
    n = distance.shape[0]
    selected = []
    current = np.full(n, np.inf)
    for _ in range(p):
        candidates = [j for j in range(n) if j not in selected]
        best = min(
            candidates,
            key=lambda j: (float(np.minimum(current, distance[:, j]).sum()), j),
        )
        selected.append(best)
        current = np.minimum(current, distance[:, best])

    improved = True
    while improved:
        improved = False
        incumbent = float(np.min(distance[:, selected], axis=1).sum())
        best_swap = None
        for old in list(selected):
            retained = [j for j in selected if j != old]
            for new in range(n):
                if new in selected:
                    continue
                trial = retained + [new]
                value = float(np.min(distance[:, trial], axis=1).sum())
                key = (value, old, new)
                if value + 1e-12 < incumbent and (best_swap is None or key < best_swap[0]):
                    best_swap = (key, old, new)
        if best_swap is not None:
            _, old, new = best_swap
            selected[selected.index(old)] = new
            selected.sort()
            improved = True
    return selected


@profiled("representative_selection_pmedian")
def solve_parametric_pmedian_selection(
    feature_vectors: Mapping[int, Sequence[float]],
    p: int,
    parameters: Optional[Mapping] = None,
) -> Tuple[Sequence[int], Dict]:
    """Shared manuscript partition for all four parametric predictors."""
    parameters = dict(parameters or {})
    ids = sorted(int(sid) for sid in feature_vectors)
    n = len(ids)
    if p <= 0 or p > n:
        raise ValueError(f"p must be in [1, {n}], got {p}.")
    if p == n:
        return ids, {"__pmedian_info__": {"status": "ALL_SELECTED", "objective": 0.0}}
    distance = _distance_matrix(feature_vectors, ids)
    solver = str(parameters.get("parametric_pmedian_solver", "MIP")).upper()
    start = time.perf_counter()
    status = ""
    mip_gap = None

    if solver in {"GREEDY", "PAM"}:
        selected_positions = list(_greedy_pmedian(distance, p))
        status = "GREEDY_SWAP"
    else:
        if gp is None:
            raise RuntimeError(
                "The manuscript p-median partition requires Gurobi. "
                "Use parametric_pmedian_solver='GREEDY' only for tests or an explicitly "
                "approved approximate pilot."
            )
        model = gp.Model("PARAMETRIC_PMEDIAN")
        model.Params.OutputFlag = int(parameters.get("parametric_solver_output", 0))
        time_limit = parameters.get("parametric_pmedian_time_limit")
        if time_limit not in (None, "", 0, 0.0):
            model.Params.TimeLimit = float(time_limit)
        y = model.addVars(n, vtype=GRB.BINARY, name="selected")
        assignment = model.addVars(n, n, lb=0.0, ub=1.0, name="assignment")
        model.addConstr(gp.quicksum(y[j] for j in range(n)) == p, name="budget")
        model.addConstrs(
            (gp.quicksum(assignment[i, j] for j in range(n)) == 1.0 for i in range(n)),
            name="assign_once",
        )
        model.addConstrs(
            (assignment[i, j] <= y[j] for i in range(n) for j in range(n)),
            name="open_link",
        )
        model.setObjective(
            gp.quicksum(distance[i, j] * assignment[i, j] for i in range(n) for j in range(n)),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.SolCount <= 0:
            raise RuntimeError(f"The p-median model returned no incumbent (status={model.Status}).")
        selected_positions = [j for j in range(n) if y[j].X > 0.5]
        if len(selected_positions) != p:
            selected_positions = sorted(range(n), key=lambda j: (-y[j].X, j))[:p]
        status = str(model.Status)
        try:
            mip_gap = float(model.MIPGap)
        except Exception:
            mip_gap = None

    with profile_section("representative_assignment_nearest_weighted"):
        selected = [ids[pos] for pos in sorted(selected_positions)]
        selected_set = set(selected)
        position = {sid: pos for pos, sid in enumerate(ids)}
        mapping = {}
        for row, sid in enumerate(ids):
            if sid in selected_set:
                continue
            representative = min(
                selected, key=lambda t: (distance[row, position[t]], t)
            )
            mapping[sid] = {
                "representative": representative,
                "distance": float(distance[row, position[representative]]),
            }
    objective = sum(info["distance"] for info in mapping.values())
    mapping["__pmedian_info__"] = {
        "status": status,
        "mip_gap": mip_gap,
        "objective": float(objective),
        "runtime_sec": time.perf_counter() - start,
        "selected": list(selected),
    }
    return selected, mapping


def random_exact_selection(
    feature_vectors: Mapping[int, Sequence[float]],
    p: int,
    parameters: Optional[Mapping] = None,
) -> Tuple[Sequence[int], Dict]:
    """Draw M_e uniformly at random (seeded). The nearest-representative mapping is
    kept only so the feature scaler and diagnostics see the same structure as the
    p-median version; the mean-of-exact-duals anchor does not use it."""
    parameters = dict(parameters or {})
    ids = sorted(int(sid) for sid in feature_vectors)
    n = len(ids)
    if p <= 0 or p > n:
        raise ValueError(f"p must be in [1, {n}], got {p}.")
    seed = int(parameters.get("parametric_random_selection_seed", 42))
    rng = np.random.default_rng(seed)
    selected = sorted(int(v) for v in rng.choice(ids, size=p, replace=False))
    selected_set = set(selected)
    distance = _distance_matrix(feature_vectors, ids)
    position = {sid: pos for pos, sid in enumerate(ids)}
    mapping = {}
    for row, sid in enumerate(ids):
        if sid in selected_set:
            continue
        representative = min(selected, key=lambda t: (distance[row, position[t]], t))
        mapping[sid] = {
            "representative": representative,
            "distance": float(distance[row, position[representative]]),
        }
    mapping["__random_selection_info__"] = {"seed": seed, "selected": list(selected)}
    return selected, mapping


@profiled("regression_weight_training")
def _ridge_fit(
    design: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    regularization: np.ndarray,
) -> Tuple[np.ndarray, float]:
    design = np.asarray(design, dtype=float)
    targets = np.asarray(targets, dtype=float)
    weights = np.asarray(sample_weights, dtype=float).reshape(-1)
    if design.shape[0] != targets.shape[0] or design.shape[0] != weights.size:
        raise ValueError("Ridge design, targets, and weights have inconsistent row counts.")
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0.0:
        weights[:] = 1.0
    root_w = np.sqrt(weights)[:, None]
    weighted_design = design * root_w
    weighted_targets = targets * root_w
    gram = weighted_design.T @ weighted_design + np.diag(regularization)
    rhs = weighted_design.T @ weighted_targets
    condition = float(np.linalg.cond(gram))
    try:
        coefficient_transpose = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficient_transpose = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return coefficient_transpose.T, condition


class _RightHandSideReducer:
    def __init__(self, dimension: int, standardize: bool = True):
        self.dimension = int(dimension)
        self.standardize = bool(standardize)
        self.mean = None
        self.scale = None
        self.components = None

    @profiled("rhs_projection_dimensionality_reduction")
    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if self.standardize:
            self.mean = matrix.mean(axis=0)
            self.scale = matrix.std(axis=0)
            self.scale[self.scale <= 1e-12] = 1.0
        else:
            # IPCO direct model: no centring or scaling, so the projected
            # state keeps its level and the linear map needs no intercept.
            self.mean = np.zeros(matrix.shape[1])
            self.scale = np.ones(matrix.shape[1])
        standardized = (matrix - self.mean) / self.scale
        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        q = max(1, min(self.dimension, vt.shape[0], vt.shape[1]))
        self.components = vt[:q]
        return standardized @ self.components.T

    @profiled("rhs_projection_dimensionality_reduction")
    def transform(self, vector: Sequence[float]) -> np.ndarray:
        if self.components is None:
            raise ValueError("The right-hand-side reducer has not been fitted.")
        standardized = (np.asarray(vector, dtype=float) - self.mean) / self.scale
        return standardized @ self.components.T



@profiled("objective_dominance_guard")
def apply_objective_dominance_guard(
    codec: DualVectorCodec,
    scenario_id: int,
    first_stage_values,
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    relative_tolerance: float = 1e-12,
) -> Tuple[np.ndarray, bool, float]:
    """Accept a feasible candidate only when its dual objective improves on the anchor.

    Both inputs must be dual feasible for the same subproblem. The baseline is
    the nonparametric transfer anchor built from the exact selected duals at
    this iteration. Returns (chosen vector, accepted, candidate-minus-base
    objective). Comparison uses the *current* subproblem RHS g_m(x), rather
    than training values or dual-vector distances. No subproblem is solved.
    """
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    g = codec.objective_coefficients(scenario_id, first_stage_values)
    baseline_value = float(np.dot(g, baseline))
    candidate_value = float(np.dot(g, candidate))
    improvement = float(np.dot(g, candidate - baseline))
    if not (
        np.isfinite(baseline_value)
        and np.isfinite(candidate_value)
        and np.isfinite(improvement)
    ):
        raise ValueError("Nonfinite dual objective in objective-dominance guard.")
    # Ties revert to the baseline; numerical noise must not be recorded as
    # improvement. The guard never accepts a candidate that is worse.
    threshold = relative_tolerance * max(
        1.0, abs(baseline_value), abs(candidate_value)
    )
    if improvement > threshold and candidate_value >= baseline_value:
        return candidate.copy(), True, improvement
    return baseline.copy(), False, improvement


class ParametricDualPredictor:
    """Lifecycle manager used by :class:`DualPredictor` and the Benders callback."""

    def __init__(
        self,
        variant: str,
        problem_type: str,
        data,
        parameters: Optional[Mapping] = None,
        use_stochastic_capacity: bool = False,
    ):
        self.variant = str(variant).upper()
        if self.variant not in PARAMETRIC_PREDICTORS:
            raise ValueError(f"Unknown parametric predictor: {variant}")
        self.parameters = dict(parameters or {})
        guard_value = self.parameters.get("parametric_objective_guard", False)
        if not isinstance(guard_value, (bool, np.bool_)):
            raise ValueError("parametric_objective_guard must be True or False.")
        self.objective_guard_enabled = bool(guard_value)
        fastpath = self.parameters.get("parametric_guard_gamma_zero_fastpath", False)
        if not isinstance(fastpath, (bool, np.bool_)):
            raise ValueError("parametric_guard_gamma_zero_fastpath must be True or False.")
        self.guard_gamma_zero_fastpath = bool(fastpath)
        self.repair_mode = str(
            self.parameters.get("parametric_repair_mode", "LOCAL")
        ).upper()
        if self.repair_mode not in {"LOCAL", "GLOBAL", "STRUCTURE_AWARE"}:
            raise ValueError("parametric_repair_mode must be LOCAL, GLOBAL or STRUCTURE_AWARE.")
        self.diagnostic_probe_enabled = bool(self.parameters.get("parametric_diagnostic_probe", False))
        self.codec = DualVectorCodec(
            problem_type,
            data,
            use_stochastic_capacity=use_stochastic_capacity,
            scmnd_formulation=self.parameters.get("scmnd_formulation", SCMND_CURRENT),
        )
        self.feature_vectors = None
        self.feature_scaler = None
        self.selected = []
        self.mapping = {}
        self.counts = {}
        self.current_predictions = {}
        self.current_base = None
        self.current_bases = {}
        self.current_model = None
        self.rhs_reducer = None
        self.all_transfer_mapping = {}
        self.is_direct = self.variant in DIRECT_RIDGE_PREDICTORS
        self.exact_fallback = self.variant in EXACT_FALLBACK_PREDICTORS
        self.direct_exact_fallback_requested = 0
        # IPCO repair settings (used only by the direct predictors). The
        # exact-fallback variant applies no closed-form fix at all by default.
        self.start_at_mean = self.variant in START_AT_MEAN_PREDICTORS
        clip = self.parameters.get(
            "parametric_direct_clip_negative",
            not (self.exact_fallback or self.start_at_mean),
        )
        if not isinstance(clip, (bool, np.bool_)):
            raise ValueError("parametric_direct_clip_negative must be True or False.")
        self.direct_clip_negative = bool(clip)
        self.direct_anchor_shrink = float(
            self.parameters.get(
                "parametric_direct_anchor_shrink", 0.0 if self.start_at_mean else 0.05
            )
        )
        if not 0.0 <= self.direct_anchor_shrink < 1.0:
            raise ValueError("parametric_direct_anchor_shrink must be in [0, 1).")
        self.current_guard_bases = {}
        self.direct_negative_clipped = 0
        self.direct_gamma_positive = 0
        self.direct_gamma_evaluations = 0
        self.anchor_method = (
            "MEAN_EXACT"
            if self.is_direct
            else "TREE"
            if self.variant == PARAMETRIC_RHS_RIDGE_TREE_ANCHOR
            else "CLUSTERING"
            if self.variant == PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR
            else "PMEDIAN_NEAREST"
            if self.variant == PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR
            else "PMEDIAN_COUNT"
        )
        self.warmup_coefficient = None
        self.training_time_total = 0.0
        self.prediction_time_total = 0.0
        self.feasibility_time_total = 0.0
        self.repair_time_total = 0.0
        self.training_calls = 0
        self.training_failures = 0
        self.raw_predictions = 0
        self.prediction_opportunities = 0
        self.raw_infeasible = 0
        self.repaired_predictions = 0
        self.full_reversions = 0
        self.certification_failures = 0
        self.objective_guard_checks = 0
        self.objective_guard_accepted = 0
        self.objective_guard_rejected = 0
        self.objective_guard_rejected_negative = 0
        self.objective_guard_total_improvement = 0.0
        self.objective_guard_skipped_zero = 0
        self.objective_guard_skipped_equal_anchor = 0
        self.objective_guard_evaluated = 0
        self.objective_guard_time_sec = 0.0
        self.structure_repair_attempts = 0
        self.structure_repair_feasible_candidates = 0
        self.structure_repair_anchor_fallbacks = 0
        self.structure_repair_nonanchor_candidates = 0
        self.structure_repair_positive_objectives = 0
        self.structure_repair_selected_kinds = {}
        self.structure_repair_positive_gain_total = 0.0
        # Aggregate scalar diagnostics: do not enlarge per-iteration Excel JSON
        # and do not store an extra array for each of millions of predictions.
        self.diag_gamma_zero = 0
        self.diag_gamma_gt_1e12 = 0
        self.diag_gamma_gt_1e8 = 0
        self.diag_gamma_gt_1e6 = 0
        self.diag_gamma_gt_1e3 = 0
        self.diag_gamma_gt_01 = 0
        self.diag_gamma_max = 0.0
        self.diag_gamma_positive_sum = 0.0
        self.diag_gamma_positive_count = 0
        self.diag_probe_checks = 0
        self.diag_probe_improved = 0
        self.diag_probe_negative = 0
        self.diag_probe_tied = 0
        self.diag_probe_positive_total = 0.0
        self.diag_probe_negative_total = 0.0
        self.diag_probe_max_positive = 0.0
        self.diag_probe_anchor_check_time_sec = 0.0
        self.diag_probe_objective_time_sec = 0.0
        self.diag_probe_total_time_sec = 0.0
        self.exact_fallback_subproblems = 0
        self.selected_exact_subproblems = 0
        self.cut_evaluations = 0
        self.parametric_cuts_added = 0
        self.cut_violations = []
        self.condition_numbers = []
        self.gamma_values = []
        self.iteration_records = []

    def select(self, feature_vectors: Mapping[int, Sequence[float]], p: int):
        """Select exact representatives and construct the requested anchor weights.

        The original RHS-Ridge implementation uses feature-space p-median selection
        and one common count-weighted anchor. The two cross-weight variants reuse
        the exact TREE or CLUSTERING transfer structures used by the corresponding
        nonparametric predictors.
        """
        self.feature_vectors = {
            int(sid): np.asarray(value, dtype=float).copy()
            for sid, value in feature_vectors.items()
        }

        if self.variant == PARAMETRIC_RHS_RIDGE_TREE_ANCHOR:
            from Dual_Prediction.training_and_selection import integrated_prediction_selection

            method_params = {
                "min_impurity_decrease": float(
                    self.parameters.get("tree_min_impurity_decrease", 0.0)
                ),
                "quantile_grid_size": int(
                    self.parameters.get("tree_quantile_grid_size", 10)
                ),
            }
            selected, mapping = integrated_prediction_selection(
                self.feature_vectors,
                "ML_Augmented_BD",
                "TREE",
                p,
                method_params,
            )
            self.all_transfer_mapping = dict(
                mapping.get("__all_transfer_mapping__", {})
            )
        elif self.variant == PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR:
            from Dual_Prediction.training_and_selection import integrated_prediction_selection

            method_params = {
                "random_state": int(
                    self.parameters.get("clustering_random_state", 42)
                ),
                "init_method": str(
                    self.parameters.get("clustering_init_method", "k-means++")
                ),
                "max_iter": int(
                    self.parameters.get("clustering_max_iter", 300)
                ),
                "tol": float(self.parameters.get("clustering_tol", 1e-4)),
            }
            selected, mapping = integrated_prediction_selection(
                self.feature_vectors,
                "ML_Augmented_BD",
                "CLUSTERING",
                p,
                method_params,
            )
            self.all_transfer_mapping = dict(
                mapping.get("__all_transfer_mapping__", {})
            )
        elif self.variant in RANDOM_SELECTION_PREDICTORS:
            selected, mapping = random_exact_selection(
                self.feature_vectors, p, self.parameters
            )
            self.all_transfer_mapping = {}
        else:
            selected, mapping = solve_parametric_pmedian_selection(
                self.feature_vectors, p, self.parameters
            )
            self.all_transfer_mapping = {}

        self.selected = list(selected)
        self.mapping = mapping

        if self.anchor_method in {"PMEDIAN_COUNT", "PMEDIAN_NEAREST"}:
            self.counts = representative_counts(self.selected, mapping)
        else:
            # Representative mass induced by the same m-specific transfer
            # weights used by TREE/CLUSTERING. Count only prediction scenarios
            # so this is analogous to representative_counts() for p-median.
            selected_set = set(self.selected)
            self.counts = {int(sid): 0.0 for sid in self.selected}
            for sid, info in self.all_transfer_mapping.items():
                sid = int(sid)
                if sid in selected_set:
                    continue
                anchors = [int(a) for a in info.get("anchors", [])]
                weights = [float(w) for w in info.get("weights", [])]
                for anchor, weight in zip(anchors, weights):
                    if anchor in self.counts:
                        self.counts[anchor] += weight

        if self.is_direct:
            # IPCO direct model: raw (unstandardized) representations.
            dim = len(next(iter(self.feature_vectors.values())))
            self.feature_scaler = FeatureScaler(
                mean=np.zeros(dim), scale=np.ones(dim)
            )
        else:
            self.feature_scaler = FeatureScaler.fit(
                self.feature_vectors, self.selected, self.counts
            )
        return selected, mapping

    @profiled("representative_assignment_nearest_weighted")
    def _feasible_anchor(self, solved_duals: Mapping[int, Mapping], sid: int) -> np.ndarray:
        """Return the current feasible anchor for one scenario."""
        if self.anchor_method == "MEAN_EXACT":
            return self._mean_exact_anchor(solved_duals)

        if self.anchor_method == "PMEDIAN_COUNT":
            return weighted_feasible_average(self.codec, solved_duals, self.counts)

        if self.anchor_method == "PMEDIAN_NEAREST":
            # Selected scenarios represent themselves; prediction scenarios
            # use their nearest selected p-median representative.
            if int(sid) in self.selected:
                representative = int(sid)
            else:
                entry = self.mapping.get(int(sid))
                if not isinstance(entry, Mapping) or "representative" not in entry:
                    raise KeyError(f"No p-median representative for scenario {sid}.")
                representative = int(entry["representative"])
            if representative not in solved_duals:
                raise KeyError(
                    f"Exact dual for p-median representative {representative} "
                    f"of scenario {sid} is unavailable."
                )
            anchor = self.codec.canonicalize(
                self.codec.flatten(solved_duals[representative])
            )
            if not self.codec.is_feasible(anchor, ANCHOR_FEASIBILITY_TOLERANCE):
                raise ValueError(
                    f"P-median representative {representative} is not dual feasible."
                )
            return anchor

        info = self.all_transfer_mapping.get(int(sid))
        if not info:
            raise KeyError(
                f"No {self.anchor_method} transfer weights are available for scenario {sid}."
            )
        anchors = [int(a) for a in info.get("anchors", [])]
        weights = np.asarray(info.get("weights", []), dtype=float)
        if not anchors or len(anchors) != weights.size:
            raise ValueError(
                f"Invalid {self.anchor_method} transfer mapping for scenario {sid}."
            )
        if any(anchor not in solved_duals for anchor in anchors):
            missing = [anchor for anchor in anchors if anchor not in solved_duals]
            raise KeyError(
                f"Exact anchor duals {missing} are unavailable for scenario {sid}."
            )
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0.0:
            raise ValueError(f"Transfer weights for scenario {sid} do not have positive mass.")
        weights /= weights.sum()
        matrix = np.asarray(
            [self.codec.flatten(solved_duals[anchor]) for anchor in anchors],
            dtype=float,
        )
        anchor = self.codec.canonicalize(weights @ matrix)
        if not self.codec.is_feasible(anchor, ANCHOR_FEASIBILITY_TOLERANCE):
            raise ValueError(
                f"The {self.anchor_method} convex anchor for scenario {sid} is not dual feasible."
            )
        return anchor

    def _mean_exact_anchor(self, solved_duals: Mapping[int, Mapping]) -> np.ndarray:
        """Equal-weight mean of the current exact duals of M_e (feasible: D is common and convex)."""
        missing = [sid for sid in self.selected if sid not in solved_duals]
        if missing:
            raise KeyError(f"Exact duals of selected scenarios {missing[:5]} are unavailable.")
        exact = {int(sid): solved_duals[sid] for sid in self.selected}
        return weighted_feasible_average(self.codec, exact, None)

    def _configure_gurobi(self, model):
        model.Params.OutputFlag = int(self.parameters.get("parametric_solver_output", 0))
        time_limit = self.parameters.get("parametric_training_time_limit")
        if time_limit not in (None, "", 0, 0.0):
            model.Params.TimeLimit = float(time_limit)

    def _feature(self, sid: int) -> np.ndarray:
        return self.feature_scaler.transform(self.feature_vectors[int(sid)])

    @profiled("regression_weight_training")
    def _fit_manuscript(self, solved_duals, first_stage_values, predicted_ids):
        if gp is None:
            raise RuntimeError("Gurobi is required for PARAMETRIC_MANUSCRIPT.")
        model = gp.Model("PARAMETRIC_MANUSCRIPT")
        self._configure_gurobi(model)
        r = self.codec.dimension
        d = len(self._feature(self.selected[0]))
        theta = model.addVars(r, d + 1, lb=-GRB.INFINITY, name="theta")
        regularization = float(self.parameters.get("parametric_regularization", 1e-6))
        objective = gp.LinExpr()
        if regularization > 0.0:
            absolute_theta = model.addVars(r, d + 1, lb=0.0, name="abs_theta")
            for j in range(r):
                for q in range(d + 1):
                    model.addConstr(absolute_theta[j, q] >= theta[j, q])
                    model.addConstr(absolute_theta[j, q] >= -theta[j, q])
                    objective += regularization * absolute_theta[j, q]
        model.setObjective(objective, GRB.MINIMIZE)

        dual_error = gp.LinExpr()
        value_error = gp.LinExpr()
        for sid in self.selected:
            x = np.concatenate(([1.0], self._feature(sid)))
            target = self.codec.flatten(solved_duals[sid])
            prediction = [gp.quicksum(theta[j, q] * x[q] for q in range(d + 1)) for j in range(r)]
            weight = float(self.counts.get(sid, 0))
            g = self.codec.objective_coefficients(sid, first_stage_values)
            coordinate_error = []
            for j in range(r):
                error = model.addVar(lb=0.0, name=f"dual_error_{sid}_{j}")
                model.addConstr(error >= prediction[j] - target[j])
                model.addConstr(error >= target[j] - prediction[j])
                coordinate_error.append(error)
                dual_error += weight * error
            cut_error_expression = gp.quicksum(
                g[j] * (prediction[j] - target[j]) for j in range(r)
            )
            cut_error = model.addVar(lb=0.0, name=f"value_error_{sid}")
            model.addConstr(cut_error >= cut_error_expression)
            model.addConstr(cut_error >= -cut_error_expression)
            value_error += weight * cut_error

        model.addConstr(
            dual_error <= float(self.parameters.get("parametric_L_d", 1e5)),
            name="dual_error_budget",
        )
        model.addConstr(
            value_error <= float(self.parameters.get("parametric_L_v", 1e5)),
            name="value_error_budget",
        )
        for sid in predicted_ids:
            x = np.concatenate(([1.0], self._feature(sid)))
            prediction = [gp.quicksum(theta[j, q] * x[q] for q in range(d + 1)) for j in range(r)]
            self.codec.add_gurobi_feasibility_constraints(model, prediction, f"prediction_{sid}")

        model.optimize()
        # Every incumbent satisfies the hard budgets and feasibility constraints.
        # Thus, a time-limited feasible incumbent is safe for cut generation even
        # if the regularization objective has not been proved optimal.
        if model.SolCount <= 0:
            return None, {"status": int(model.Status)}
        coefficient = np.asarray(
            [[theta[j, q].X for q in range(d + 1)] for j in range(r)], dtype=float
        )
        return {"coefficient": coefficient}, {"status": int(model.Status)}

    @profiled("regression_weight_training")
    def _fit_penalized(self, solved_duals, first_stage_values, predicted_ids, base):
        if gp is None:
            raise RuntimeError("Gurobi is required for PARAMETRIC_PENALIZED_ERRORS.")
        model = gp.Model("PARAMETRIC_PENALIZED_ERRORS")
        self._configure_gurobi(model)
        r = self.codec.dimension
        d = len(self._feature(self.selected[0]))
        slope = model.addVars(r, d, lb=-GRB.INFINITY, name="slope")
        kappa_d = float(self.parameters.get("parametric_kappa_d", 1.0))
        kappa_v = float(self.parameters.get("parametric_kappa_v", 1.0))
        regularization = float(self.parameters.get("parametric_regularization", 1e-6))
        objective = gp.LinExpr()

        if regularization > 0.0:
            absolute_slope = model.addVars(r, d, lb=0.0, name="abs_slope")
            for j in range(r):
                for q in range(d):
                    model.addConstr(absolute_slope[j, q] >= slope[j, q])
                    model.addConstr(absolute_slope[j, q] >= -slope[j, q])
                    objective += regularization * absolute_slope[j, q]

        for sid in self.selected:
            x = self._feature(sid)
            target = self.codec.flatten(solved_duals[sid])
            prediction = [base[j] + gp.quicksum(slope[j, q] * x[q] for q in range(d)) for j in range(r)]
            weight = float(self.counts.get(sid, 0))
            g = self.codec.objective_coefficients(sid, first_stage_values)
            for j in range(r):
                error = model.addVar(lb=0.0, name=f"dual_error_{sid}_{j}")
                model.addConstr(error >= prediction[j] - target[j])
                model.addConstr(error >= target[j] - prediction[j])
                objective += kappa_d * weight * error
            cut_expression = gp.quicksum(g[j] * (prediction[j] - target[j]) for j in range(r))
            cut_error = model.addVar(lb=0.0, name=f"value_error_{sid}")
            model.addConstr(cut_error >= cut_expression)
            model.addConstr(cut_error >= -cut_expression)
            objective += kappa_v * weight * cut_error

        for sid in predicted_ids:
            x = self._feature(sid)
            prediction = [base[j] + gp.quicksum(slope[j, q] * x[q] for q in range(d)) for j in range(r)]
            self.codec.add_gurobi_feasibility_constraints(model, prediction, f"prediction_{sid}")

        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        if model.SolCount <= 0:
            # B=0 is a certified feasible solution; use it after solver/numerical failure.
            return {"base": base, "coefficient": np.zeros((r, d))}, {
                "status": int(model.Status),
                "constant_fallback": True,
            }
        coefficient = np.asarray(
            [[slope[j, q].X for q in range(d)] for j in range(r)], dtype=float
        )
        return {"base": base, "coefficient": coefficient}, {"status": int(model.Status)}

    @profiled("regression_weight_training")
    def _fit_iteration_ridge(self, solved_duals, base):
        ids = list(self.selected)
        design = np.asarray([self._feature(sid) for sid in ids], dtype=float)
        targets = np.asarray(
            [self.codec.flatten(solved_duals[sid]) - base for sid in ids], dtype=float
        )
        weights = np.asarray([float(self.counts.get(sid, 0)) for sid in ids], dtype=float)
        beta = float(self.parameters.get("parametric_ridge_beta", 1e-3))
        coefficient, condition = _ridge_fit(
            design,
            targets,
            weights,
            np.full(design.shape[1], beta, dtype=float),
        )
        self.condition_numbers.append(condition)
        return {"base": base, "coefficient": coefficient}, {
            "status": "CLOSED_FORM",
            "condition_number": condition,
        }

    @profiled("regression_weight_training")
    def fit_warmup_rhs_model(self, all_trial_duals, first_stage_by_iteration):
        if self.variant not in RHS_RIDGE_PREDICTORS:
            return
        start = time.perf_counter()
        scenario_ids = sorted(self.feature_vectors)
        grouped = {}
        for key, duals in (all_trial_duals or {}).items():
            if not isinstance(key, tuple) or len(key) != 2 or not isinstance(duals, Mapping):
                continue
            sid, iteration = int(key[0]), int(key[1])
            if sid in self.feature_vectors and iteration > 0:
                grouped.setdefault(iteration, {})[sid] = duals
        complete_iterations = [
            iteration
            for iteration in sorted(grouped)
            if all(sid in grouped[iteration] for sid in scenario_ids)
            and iteration in first_stage_by_iteration
        ]
        if not complete_iterations:
            raise ValueError("No complete warm-up round is available for RHS-aware training.")

        raw_g = []
        residual_targets = []
        static_design = []
        observation_weights = []
        for iteration in complete_iterations:
            round_duals = grouped[iteration]
            exact_round = {sid: round_duals[sid] for sid in self.selected}
            first_stage = first_stage_by_iteration[iteration]
            for sid in scenario_ids:
                static_design.append(self._feature(sid))
                g = self.codec.objective_coefficients(sid, first_stage)
                raw_g.append(g)
                if self.is_direct:
                    # IPCO variant: learn the dual itself (no baseline).
                    residual_targets.append(self.codec.flatten(round_duals[sid]))
                else:
                    base = self._feasible_anchor(exact_round, sid)
                    residual_targets.append(self.codec.flatten(round_duals[sid]) - base)
                # Every warm-up subproblem is observed directly. Transfer weights
                # enter through the feasible anchor; the regression observations
                # themselves receive equal statistical weight.
                observation_weights.append(1.0)

        rhs_dimension = int(self.parameters.get("parametric_rhs_dim", 3))
        self.rhs_reducer = _RightHandSideReducer(
            rhs_dimension, standardize=not self.is_direct
        )
        rhs_design = self.rhs_reducer.fit_transform(np.asarray(raw_g, dtype=float))
        static_design = np.asarray(static_design, dtype=float)
        design = np.concatenate([static_design, rhs_design], axis=1)
        beta_f = float(self.parameters.get("parametric_rhs_ridge_beta_features", 1e-3))
        beta_g = float(self.parameters.get("parametric_rhs_ridge_beta_rhs", 1e-3))
        regularization = np.concatenate(
            [
                np.full(static_design.shape[1], beta_f),
                np.full(rhs_design.shape[1], beta_g),
            ]
        )

        coefficient, condition = _ridge_fit(
            design,
            np.asarray(residual_targets, dtype=float),
            np.asarray(observation_weights, dtype=float),
            regularization,
        )
        self.warmup_coefficient = coefficient
        self.condition_numbers.append(condition)
        elapsed = time.perf_counter() - start
        self.training_time_total += elapsed
        self.training_calls += 1
        self.iteration_records.append(
            {
                "stage": "warmup_rhs_training",
                "training_time_sec": elapsed,
                "condition_number": condition,
                "observations": int(design.shape[0]),
                "predictors": int(design.shape[1]),
            }
        )

    @profiled("parametric_residual_prediction")
    def _raw_predictions(self, predicted_ids, model, first_stage_values):
        predictions = {}
        if self.variant == PARAMETRIC_MANUSCRIPT:
            theta = model["coefficient"]
            for sid in predicted_ids:
                x = np.concatenate(([1.0], self._feature(sid)))
                predictions[sid] = self.codec.canonicalize(theta @ x)
        elif self.variant in {PARAMETRIC_PENALIZED_ERRORS, PARAMETRIC_RIDGE_REPAIR}:
            for sid in predicted_ids:
                predictions[sid] = self.codec.canonicalize(
                    model["base"] + model["coefficient"] @ self._feature(sid)
                )
        else:
            if self.warmup_coefficient is None or self.rhs_reducer is None:
                raise ValueError("The RHS-aware ridge model was not trained after warm-up.")
            for sid in predicted_ids:
                rhs_feature = self.rhs_reducer.transform(
                    self.codec.objective_coefficients(sid, first_stage_values)
                )
                design = np.concatenate([self._feature(sid), rhs_feature])
                if self.is_direct:
                    # Linear map with no intercept: lambda = Theta [z_m; s_m(x)].
                    vector = self.warmup_coefficient @ design
                    if self.direct_clip_negative and self.codec.problem_type == "SCFLP":
                        # Projection onto lambda >= 0 (closed form). For SCFLP every
                        # dual coordinate is sign-constrained, and with c >= 0 raising a
                        # negative mu_i to 0 or a negative nu_j to 0 cannot create a new
                        # violation of mu_i - nu_j <= c_ij; it only removes sign violations.
                        if np.any(vector < 0.0):
                            self.direct_negative_clipped += 1
                            vector = np.maximum(vector, 0.0)
                    predictions[sid] = self.codec.canonicalize(vector)
                    continue
                base = self.current_bases.get(int(sid), self.current_base)
                if base is None:
                    raise KeyError(f"No current feasible anchor is available for scenario {sid}.")
                predictions[sid] = self.codec.canonicalize(
                    base + self.warmup_coefficient @ design
                )
        return predictions

    def _record_diagnostic_gamma(self, gamma: float) -> None:
        """Count the distribution of repair steps without retaining extra vectors."""
        if not self.diagnostic_probe_enabled:
            return
        value = float(gamma)
        if value == 0.0:
            self.diag_gamma_zero += 1
        else:
            self.diag_gamma_positive_sum += value
            self.diag_gamma_positive_count += 1
            self.diag_gamma_max = max(self.diag_gamma_max, value)
        if value > 1e-12:
            self.diag_gamma_gt_1e12 += 1
        if value > 1e-8:
            self.diag_gamma_gt_1e8 += 1
        if value > 1e-6:
            self.diag_gamma_gt_1e6 += 1
        if value > 1e-3:
            self.diag_gamma_gt_1e3 += 1
        if value > 0.1:
            self.diag_gamma_gt_01 += 1

    def prepare_iteration(
        self,
        solved_duals: Mapping[int, Mapping],
        first_stage_values,
        predicted_ids: Iterable[int],
        iteration: Optional[int] = None,
    ) -> bool:
        predicted_ids = sorted(int(sid) for sid in predicted_ids)
        self.prediction_opportunities += len(predicted_ids)
        start = time.perf_counter()
        prediction_time_before = self.prediction_time_total
        feasibility_time_before = self.feasibility_time_total
        repair_time_before = self.repair_time_total
        is_rhs_ridge = self.variant in RHS_RIDGE_PREDICTORS
        self.training_calls += 0 if is_rhs_ridge else 1
        try:
            if is_rhs_ridge:
                # Build anchors for prediction scenarios and selected representatives.
                # The latter are used only by diagnostics; Phase-2 predictions are
                # still produced only for the prediction set.
                anchor_ids = sorted(set(predicted_ids).union(self.selected))
                if self.anchor_method == "MEAN_EXACT":
                    # One common reference point for all scenarios; compute once.
                    mean_anchor = self._mean_exact_anchor(solved_duals)
                    # The repair starts from (1 - eps) * mean, which is feasible because
                    # 0 is in D (q >= 0) and D is convex, and lies off the faces on which
                    # the mean sits. The objective guard still compares with the mean.
                    start = self.codec.canonicalize((1.0 - self.direct_anchor_shrink) * mean_anchor)
                    if not self.codec.is_feasible(start, ANCHOR_FEASIBILITY_TOLERANCE):
                        start = mean_anchor
                    self.current_bases = {sid: start for sid in anchor_ids}
                    self.current_guard_bases = {sid: mean_anchor for sid in anchor_ids}
                else:
                    self.current_bases = {
                        sid: self._feasible_anchor(solved_duals, sid)
                        for sid in anchor_ids
                    }
                first_anchor_sid = anchor_ids[0] if anchor_ids else None
                base = (
                    self.current_bases[first_anchor_sid]
                    if first_anchor_sid is not None
                    else weighted_feasible_average(self.codec, solved_duals, self.counts)
                )
                self.current_base = base
            else:
                base = weighted_feasible_average(self.codec, solved_duals, self.counts)
                self.current_base = base
                self.current_bases = {}
            if self.variant == PARAMETRIC_MANUSCRIPT:
                model, info = self._fit_manuscript(
                    solved_duals, first_stage_values, predicted_ids
                )
                if model is None:
                    self.training_failures += 1
                    failure_elapsed = time.perf_counter() - start
                    self.iteration_records.append(
                        {
                            "iteration": iteration,
                            "training_failed": True,
                            "training_time_sec": failure_elapsed,
                            "preparation_time_sec": failure_elapsed,
                            **info,
                        }
                    )
                    return False
            elif self.variant == PARAMETRIC_PENALIZED_ERRORS:
                model, info = self._fit_penalized(
                    solved_duals, first_stage_values, predicted_ids, base
                )
            elif self.variant == PARAMETRIC_RIDGE_REPAIR:
                model, info = self._fit_iteration_ridge(solved_duals, base)
            else:
                model, info = {"base": base}, {"status": "REUSE_WARMUP_MODEL"}
            self.current_model = model
        except Exception:
            self.training_failures += 1
            logger.exception("Parametric training failed at iteration %s", iteration)
            return False
        finally:
            elapsed = time.perf_counter() - start
            if not is_rhs_ridge:
                self.training_time_total += elapsed

        prediction_start = time.perf_counter()
        raw = self._raw_predictions(predicted_ids, self.current_model, first_stage_values)
        self.prediction_time_total += time.perf_counter() - prediction_start
        self.raw_predictions += len(raw)

        feasibility_start = time.perf_counter()
        raw_feasible = {
            sid: self.codec.is_feasible(
                vector, float(self.parameters.get("parametric_feasibility_tolerance", 1e-7))
            )
            for sid, vector in raw.items()
        }
        self.feasibility_time_total += time.perf_counter() - feasibility_start
        self.raw_infeasible += sum(not value for value in raw_feasible.values())

        zero_repair_ids = set()
        fallback_ids = set()
        if self.exact_fallback:
            # No repair: infeasible raw predictions are returned unchanged so the
            # callback's certification rejects them and solves those subproblems
            # exactly. Feasible raw predictions still go through the guard.
            fallback_ids = {sid for sid, ok in raw_feasible.items() if not ok}
            self.direct_exact_fallback_requested += len(fallback_ids)
        needs_repair = (
            self.variant == PARAMETRIC_RIDGE_REPAIR
            or self.variant in RHS_RIDGE_PREDICTORS
        ) and not self.exact_fallback
        if needs_repair:
            repair_start = time.perf_counter()
            tolerance = float(self.parameters.get("parametric_feasibility_tolerance", 1e-7))
            safety = float(self.parameters.get("parametric_repair_safety", 1e-10))
            repaired = {}
            if self.repair_mode == "STRUCTURE_AWARE":
                with profile_section("feasibility_repair_structure_aware"):
                    # Generate and certify closed-form candidates. The current
                    # scenario objective selects among them; Borzou's independent
                    # dominance guard still decides candidate versus anchor.
                    for sid, vector in raw.items():
                        self.structure_repair_attempts += 1
                        scenario_base = self.current_bases.get(int(sid), base)
                        rhs = self.codec.objective_coefficients(sid, first_stage_values)
                        chosen = None
                        chosen_kind = None
                        chosen_obj = -np.inf
                        for kind, candidate in self.codec.structure_aware_candidates(vector).items():
                            if not self.codec.is_feasible(candidate, tolerance):
                                continue
                            self.structure_repair_feasible_candidates += 1
                            score = float(np.dot(rhs, candidate))
                            if np.isfinite(score) and score > chosen_obj:
                                chosen = candidate
                                chosen_obj = score
                                chosen_kind = kind
                        if chosen is None:
                            self.structure_repair_anchor_fallbacks += 1
                            chosen = scenario_base.copy()
                        else:
                            self.structure_repair_selected_kinds[chosen_kind] = (
                                self.structure_repair_selected_kinds.get(chosen_kind, 0) + 1
                            )
                        repaired[sid] = chosen
                        if not np.array_equal(chosen, scenario_base):
                            self.structure_repair_nonanchor_candidates += 1
                        # This measures strict raw structure-repair improvement
                        # BEFORE Borzou's independent objective-dominance guard.
                        anchor_obj = float(np.dot(rhs, scenario_base))
                        gain = float(np.dot(rhs, chosen - scenario_base))
                        threshold = 1e-12 * max(1.0, abs(anchor_obj), abs(chosen_obj))
                        if gain > threshold and chosen_obj >= anchor_obj:
                            self.structure_repair_positive_objectives += 1
                            self.structure_repair_positive_gain_total += gain
                        if not raw_feasible[sid]:
                            self.repaired_predictions += 1
            else:
                local_results = {}
                mode = self.repair_mode
                for sid, vector in raw.items():
                    scenario_base = self.current_bases.get(int(sid), base)
                    local_results[sid] = self.codec.repair_along_segment(
                        scenario_base,
                        vector,
                        feasibility_tolerance=tolerance,
                        safety=safety,
                    )
                if mode == "GLOBAL" and local_results:
                    global_gamma = min(result[1] for result in local_results.values())
                    for sid, vector in raw.items():
                        scenario_base = self.current_bases.get(int(sid), base)
                        repaired[sid] = self.codec.canonicalize(
                            scenario_base + global_gamma * (vector - scenario_base)
                        )
                        self.gamma_values.append(global_gamma)
                        self._record_diagnostic_gamma(global_gamma)
                        if self.objective_guard_enabled and self.guard_gamma_zero_fastpath and global_gamma == 0.0:
                            zero_repair_ids.add(int(sid))
                        if global_gamma < 1.0 - 1e-12:
                            self.repaired_predictions += 1
                        if global_gamma <= 1e-12:
                            self.full_reversions += 1
                else:
                    for sid, (vector, gamma, was_feasible, reverted) in local_results.items():
                        repaired[sid] = vector
                        if self.is_direct:
                            self.direct_gamma_evaluations += 1
                            if gamma > 0.0:
                                self.direct_gamma_positive += 1
                        self.gamma_values.append(gamma)
                        self._record_diagnostic_gamma(gamma)
                        if self.objective_guard_enabled and self.guard_gamma_zero_fastpath and gamma == 0.0:
                            zero_repair_ids.add(int(sid))
                        if not was_feasible:
                            self.repaired_predictions += 1
                        if reverted or gamma <= 1e-12:
                            self.full_reversions += 1
            self.repair_time_total += time.perf_counter() - repair_start
            predictions = repaired
        else:
            predictions = {sid: v for sid, v in raw.items() if sid not in fallback_ids}

        certification_start = time.perf_counter()
        for sid, vector in list(predictions.items()):
            if not self.codec.is_feasible(
                vector, float(self.parameters.get("parametric_feasibility_tolerance", 1e-7))
            ):
                self.certification_failures += 1
                if needs_repair or self.variant == PARAMETRIC_PENALIZED_ERRORS:
                    fallback = (
                        self.current_guard_bases.get(int(sid))
                        if self.is_direct else None
                    )
                    if fallback is None:
                        fallback = self.current_bases.get(int(sid), base)
                    predictions[sid] = fallback.copy()
                else:
                    self.feasibility_time_total += time.perf_counter() - certification_start
                    return False
        self.feasibility_time_total += time.perf_counter() - certification_start

        # In diagnostic mode, both guard OFF and ON perform the exact same
        # objective probe at the current first-stage solution. Only guard ON
        # substitutes the chosen dual. Timings separately measure feasibility
        # validation and objective-score evaluation.
        guard_checked = guard_accepted = guard_rejected = 0
        if self.objective_guard_enabled or self.diagnostic_probe_enabled:
            with profile_section("objective_dominance_guard"):
                tolerance = float(self.parameters.get("parametric_feasibility_tolerance", 1e-7))
                probe_start = time.perf_counter()
                for sid, vector in list(predictions.items()):
                    scenario_base = (
                        self.current_guard_bases.get(int(sid))
                        if self.is_direct
                        else self.current_bases.get(int(sid), self.current_base)
                    )
                    # If exact line-segment repair produced gamma=0, its certified
                    # result is the anchor itself. A numerical equality check
                    # protects against canonicalization or fallback edge cases.
                    # All other cases retain the previous feasibility/objective
                    # comparison behavior verbatim.
                    skip = (
                        self.objective_guard_enabled
                        and not self.is_direct
                        and self.guard_gamma_zero_fastpath
                        and int(sid) in zero_repair_ids
                        and scenario_base is not None
                        and np.array_equal(vector, scenario_base)
                    )
                    # Structure-aware repair has no gamma. It can nonetheless
                    # reproduce its certified anchor exactly; comparing two
                    # identical objective vectors would be redundant. Preserve
                    # the old line-segment path and distinguish this case in logs.
                    skip_equal_anchor = (
                        self.objective_guard_enabled
                        and not self.is_direct
                        and self.guard_gamma_zero_fastpath
                        and self.repair_mode == "STRUCTURE_AWARE"
                        and not skip
                        and scenario_base is not None
                        and np.array_equal(vector, scenario_base)
                    )
                    if skip or skip_equal_anchor:
                        predictions[sid] = scenario_base.copy()
                        guard_checked += 1
                        guard_rejected += 1  # an objective tie by construction
                        self.objective_guard_checks += 1
                        self.objective_guard_rejected += 1
                        if skip:
                            self.objective_guard_skipped_zero += 1
                        else:
                            self.objective_guard_skipped_equal_anchor += 1
                        if self.diagnostic_probe_enabled:
                            self.diag_probe_checks += 1
                            self.diag_probe_tied += 1
                        continue
                    anchor_start = time.perf_counter()
                    if scenario_base is None or not self.codec.is_feasible(
                        scenario_base, max(tolerance, ANCHOR_FEASIBILITY_TOLERANCE)
                    ):
                        raise ValueError(f"No feasible nonparametric anchor for scenario {sid}.")
                    if self.diagnostic_probe_enabled:
                        self.diag_probe_anchor_check_time_sec += (
                            time.perf_counter() - anchor_start
                        )
                    objective_start = time.perf_counter()
                    guarded, accepted, improvement = apply_objective_dominance_guard(
                        self.codec, sid, first_stage_values, vector, scenario_base
                    )
                    if self.objective_guard_enabled:
                        self.objective_guard_evaluated += 1
                    if self.diagnostic_probe_enabled:
                        self.diag_probe_objective_time_sec += (
                            time.perf_counter() - objective_start
                        )
                        self.diag_probe_checks += 1
                        if accepted:
                            self.diag_probe_improved += 1
                            self.diag_probe_positive_total += improvement
                            self.diag_probe_max_positive = max(
                                self.diag_probe_max_positive, improvement
                            )
                        elif improvement < 0:
                            self.diag_probe_negative += 1
                            self.diag_probe_negative_total += improvement
                        else:
                            self.diag_probe_tied += 1
                    # Shadow probe on guard OFF must not modify its prediction.
                    if self.objective_guard_enabled:
                        predictions[sid] = guarded
                        guard_checked += 1
                        self.objective_guard_checks += 1
                        if accepted:
                            guard_accepted += 1
                            self.objective_guard_accepted += 1
                            self.objective_guard_total_improvement += improvement
                        else:
                            guard_rejected += 1
                            self.objective_guard_rejected += 1
                            if improvement < 0:
                                self.objective_guard_rejected_negative += 1
                if self.objective_guard_enabled:
                    self.objective_guard_time_sec += time.perf_counter() - probe_start
                if self.diagnostic_probe_enabled:
                    self.diag_probe_total_time_sec += time.perf_counter() - probe_start
        self.current_predictions = {
            sid: self.codec.unflatten(vector) for sid, vector in predictions.items()
        }
        for sid in fallback_ids:
            # Deliberately uncertified: the callback solves these exactly.
            self.current_predictions[sid] = self.codec.unflatten(raw[sid])

        # Report the two manuscript training-error quantities for every fitted
        # model.  These are diagnostics, not additional constraints.
        training_raw = self._raw_predictions(
            self.selected, self.current_model, first_stage_values
        )
        weighted_dual_error = 0.0
        weighted_value_error = 0.0
        for sid in self.selected:
            error = training_raw[sid] - self.codec.flatten(solved_duals[sid])
            weight = float(self.counts.get(sid, 0.0))
            weighted_dual_error += weight * float(np.linalg.norm(error, ord=1))
            weighted_value_error += weight * abs(
                float(
                    error
                    @ self.codec.objective_coefficients(sid, first_stage_values)
                )
            )
        self.iteration_records.append(
            {
                "iteration": iteration,
                "training_time_sec": (
                    0.0
                    if is_rhs_ridge
                    else elapsed
                ),
                "preparation_time_sec": elapsed,
                "prediction_time_sec": (
                    self.prediction_time_total - prediction_time_before
                ),
                "feasibility_time_sec": (
                    self.feasibility_time_total - feasibility_time_before
                ),
                "repair_time_sec": self.repair_time_total - repair_time_before,
                "num_predictions": len(predicted_ids),
                "objective_guard_enabled": self.objective_guard_enabled,
                "objective_guard_checked": guard_checked,
                "objective_guard_accepted": guard_accepted,
                "objective_guard_rejected": guard_rejected,
                "raw_infeasible": sum(not value for value in raw_feasible.values()),
                "weighted_training_dual_l1": weighted_dual_error,
                "weighted_training_value_l1": weighted_value_error,
                **info,
            }
        )
        return True

    def predict(self, predicted_ids: Iterable[int]) -> Dict[int, Dict]:
        missing = set(int(sid) for sid in predicted_ids) - set(self.current_predictions)
        if missing:
            raise KeyError(f"Parametric predictions were not prepared for scenarios {sorted(missing)}")
        return {int(sid): self.current_predictions[int(sid)] for sid in predicted_ids}

    @profiled("feasibility_check")
    def validate_structured_dual(self, duals: Mapping) -> bool:
        start = time.perf_counter()
        result = self.codec.is_feasible(
            self.codec.flatten(duals),
            float(self.parameters.get("parametric_feasibility_tolerance", 1e-7)),
        )
        self.feasibility_time_total += time.perf_counter() - start
        return result

    def record_exact_fallback(self, count: int) -> None:
        """Record predictions replaced by exact subproblem solves."""
        self.exact_fallback_subproblems += max(0, int(count))

    def record_selected_exact_solves(self, count: int) -> None:
        self.selected_exact_subproblems += max(0, int(count))

    def record_cut_evaluation(self, violation: float, added: bool) -> None:
        self.cut_evaluations += 1
        self.cut_violations.append(float(violation))
        if added:
            self.parametric_cuts_added += 1

    def diagnostics(self) -> Dict:
        gamma = np.asarray(self.gamma_values, dtype=float)
        conditions = np.asarray(self.condition_numbers, dtype=float)
        cut_violations = np.asarray(self.cut_violations, dtype=float)
        return {
            "variant": self.variant,
            "anchor_method": self.anchor_method,
            "direct_clip_negative": self.direct_clip_negative,
            "direct_anchor_shrink": self.direct_anchor_shrink,
            "direct_negative_clipped": self.direct_negative_clipped,
            "direct_gamma_evaluations": self.direct_gamma_evaluations,
            "direct_gamma_positive": self.direct_gamma_positive,
            "direct_exact_fallback_requested": self.direct_exact_fallback_requested,
            "repair_mode": self.repair_mode,
            "structure_repair_attempts": self.structure_repair_attempts,
            "structure_repair_feasible_candidates": self.structure_repair_feasible_candidates,
            "structure_repair_anchor_fallbacks": self.structure_repair_anchor_fallbacks,
            "structure_repair_nonanchor_candidates": self.structure_repair_nonanchor_candidates,
            "structure_repair_positive_objectives": self.structure_repair_positive_objectives,
            "structure_repair_positive_gain_total": self.structure_repair_positive_gain_total,
            "structure_repair_selected_kinds": dict(self.structure_repair_selected_kinds),
            "objective_guard_enabled": self.objective_guard_enabled,
            "objective_guard_gamma_zero_fastpath": self.guard_gamma_zero_fastpath,
            "objective_guard_skipped_zero": self.objective_guard_skipped_zero,
            "objective_guard_skipped_equal_anchor": self.objective_guard_skipped_equal_anchor,
            "objective_guard_evaluated": self.objective_guard_evaluated,
            "objective_guard_time_sec": self.objective_guard_time_sec,
            "objective_guard_checks": self.objective_guard_checks,
            "objective_guard_accepted": self.objective_guard_accepted,
            "objective_guard_rejected": self.objective_guard_rejected,
            "objective_guard_rejected_negative": self.objective_guard_rejected_negative,
            "objective_guard_total_improvement": self.objective_guard_total_improvement,
            "diagnostic_probe_enabled": self.diagnostic_probe_enabled,
            "diag_gamma_zero": self.diag_gamma_zero,
            "diag_gamma_positive_count": self.diag_gamma_positive_count,
            "diag_gamma_positive_mean": (
                self.diag_gamma_positive_sum / self.diag_gamma_positive_count
                if self.diag_gamma_positive_count else 0.0
            ),
            "diag_gamma_max": self.diag_gamma_max,
            "diag_gamma_gt_1e12": self.diag_gamma_gt_1e12,
            "diag_gamma_gt_1e8": self.diag_gamma_gt_1e8,
            "diag_gamma_gt_1e6": self.diag_gamma_gt_1e6,
            "diag_gamma_gt_1e3": self.diag_gamma_gt_1e3,
            "diag_gamma_gt_01": self.diag_gamma_gt_01,
            "diag_probe_checks": self.diag_probe_checks,
            "diag_probe_improved": self.diag_probe_improved,
            "diag_probe_negative": self.diag_probe_negative,
            "diag_probe_tied": self.diag_probe_tied,
            "diag_probe_positive_total": self.diag_probe_positive_total,
            "diag_probe_negative_total": self.diag_probe_negative_total,
            "diag_probe_max_positive": self.diag_probe_max_positive,
            "diag_probe_anchor_check_time_sec": self.diag_probe_anchor_check_time_sec,
            "diag_probe_objective_time_sec": self.diag_probe_objective_time_sec,
            "diag_probe_total_time_sec": self.diag_probe_total_time_sec,
            "training_calls": self.training_calls,
            "training_failures": self.training_failures,
            "training_time_total_sec": self.training_time_total,
            "prediction_time_total_sec": self.prediction_time_total,
            "feasibility_time_total_sec": self.feasibility_time_total,
            "repair_time_total_sec": self.repair_time_total,
            "raw_predictions": self.raw_predictions,
            "prediction_opportunities": self.prediction_opportunities,
            "raw_infeasible_predictions": self.raw_infeasible,
            "raw_infeasible_rate": (
                self.raw_infeasible / self.raw_predictions if self.raw_predictions else 0.0
            ),
            "repaired_predictions": self.repaired_predictions,
            "full_reversions": self.full_reversions,
            "certification_failures": self.certification_failures,
            "exact_fallback_subproblems": self.exact_fallback_subproblems,
            "exact_fallback_rate": (
                self.exact_fallback_subproblems / self.prediction_opportunities
                if self.prediction_opportunities
                else 0.0
            ),
            "selected_exact_subproblems": self.selected_exact_subproblems,
            "total_exact_subproblems": (
                self.selected_exact_subproblems + self.exact_fallback_subproblems
            ),
            "cut_evaluations": self.cut_evaluations,
            "parametric_cuts_added": self.parametric_cuts_added,
            "parametric_cut_addition_rate": (
                self.parametric_cuts_added / self.cut_evaluations
                if self.cut_evaluations
                else 0.0
            ),
            "mean_cut_violation": (
                float(cut_violations.mean()) if cut_violations.size else None
            ),
            "max_cut_violation": (
                float(cut_violations.max()) if cut_violations.size else None
            ),
            "mean_repair_gamma": float(gamma.mean()) if gamma.size else None,
            "min_repair_gamma": float(gamma.min()) if gamma.size else None,
            "max_condition_number": float(conditions.max()) if conditions.size else None,
            "pmedian": self.mapping.get("__pmedian_info__", {}),
            "iteration_records": list(self.iteration_records),
        }
