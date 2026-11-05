"""
Dual_Prediction.prediction
===========================

ML dual prediction module for ITERATIVE ML-augmented Benders decomposition (Step 6 of 6-Step Pattern)
focused exclusively on SCFLP (Stochastic Capacitated Facility Location) problems. This module handles 
the iterative prediction phase using pre-established mappings from the one-time training_and_selection.py 
setup phase (Step 5).

NEW ARCHITECTURE (Feb 2026):
----------------------------
This module implements Step 6 (Phase 2 - ML-Augmented Iterations) of the restructured 6-step
execution pattern. It supports all 5 ML method types with their respective predictors:
- **ML_Augmented_BD**: Uses KNN/TREE/CLUSTERING on graph walk features
- **ML_Augmented_BD_with_duals**: Uses CLUSTERING on dual vectors
- **Covariate_Similarity**: Uses CLUSTERING on covariate vectors
- **Uncertainty_Similarity**: Uses CLUSTERING on uncertainty vectors
- **Random_Sampling**: Dynamic random selection (no fixed training)

For backward compatibility, this module accepts legacy prediction_method names (KNN, TREE,
DUAL_CLUSTERING, etc.). The main orchestrator translates new method types to legacy names.

Current Focus: SCFLP + All Prediction Methods
--------------------------------------------
**As of February 2026, this implementation supports SCFLP problems with all 5 ML method types
and 3 predictor algorithms** to provide comprehensive ML-augmented Benders decomposition approaches.

Architecture Role
-----------------
- **ITERATIVE OPERATIONS**: Used repeatedly during Phase 2 Benders iterations
- **Pre-computed Mappings**: Uses method-specific mappings established during one-time training phase
- **Fast Prediction**: Optimized for repeated prediction calls with minimal overhead
- **SCFLP Specialized**: Implementation optimized for SCFLP dual structure and constraints

The `DualPredictor` class serves as the unified interface for all prediction methods:
- **Unified Interface**: Provides `_train_unified()` and `predict()` methods for all methods
- **Training Delegation**: Delegates one-time setup to training_and_selection.py
- **Method-Specific Prediction**: Fast prediction using established method mappings
- **SCFLP Only**: Non-SCFLP problem implementations raise NotImplementedError

Method Implementations (SCFLP)
-------------------------------
**K-Nearest Neighbors (KNN)**: Iterative prediction using pre-computed k-NN mappings
- **Training**: Delegates to training_and_selection.py for integrated k-NN selection
- **Selection**: Uses paper's mathematical objective for joint selection and mapping
- **Prediction**: Fast lookup and uniform averaging using established neighbor mappings

**Tree-Based Medoid Selection (TREE)**: Decision tree-based dual reconstruction
- **Training**: Delegates to training_and_selection.py for tree construction and medoid selection
- **Selection**: Greedy top-down tree construction with dual space optimization
- **Prediction**: Fast leaf lookup and medoid-based dual reconstruction

**Dual Clustering Selection (DUAL_CLUSTERING)**: K-means clustering-based dual reconstruction
- **Training**: Delegates to training_and_selection.py for clustering and representative selection
- **Selection**: K-means clustering in feature space with centroid-based representative selection
- **Prediction**: Fast cluster lookup and direct dual copying from cluster representatives

**Dual Space Clustering (DUAL_CLUSTERING_WITH_DUAL)**: K-medoids clustering in dual space
- **Training**: Delegates to training_and_selection.py for dual space clustering and medoid selection
- **Selection**: K-medoids clustering in dual solution space with medoid-based representative selection
- **Prediction**: Fast cluster lookup and direct dual copying from cluster medoids

**Uncertainty Clustering (UNCERTAINTY_CLUSTERING)**: K-means clustering on uncertainty vectors
- **Training**: Delegates to training_and_selection.py for uncertainty vector clustering
- **Selection**: K-means clustering on demand vectors with Phase 0/1 skipping
- **Prediction**: Fast cluster lookup and direct dual copying from cluster representatives

**Covariate Clustering (COVARIATE_CLUSTERING)**: K-means clustering on covariate vectors
- **Training**: Delegates to training_and_selection.py for covariate vector clustering
- **Selection**: K-means clustering on covariate vectors with Phase 0/1 skipping
- **Prediction**: Fast cluster lookup and direct dual copying from cluster representatives

All six methods are fully implemented and optimized for SCFLP problems.

Usage Examples
--------------
Basic usage with the unified DualPredictor class:

```python
# Example: Create DualPredictor instance

# Initialize predictor for SCFLP (other problem types removed)
predictor = DualPredictor(method='KNN', problem_type='SCFLP', n_neighbors=3)
# or
predictor = DualPredictor(method='DUAL_CLUSTERING', problem_type='SCFLP')
```

# Train the predictor (all methods use same interface)
predictor.train(
    all_trial_duals=trial_duals,
    feature_vectors=feature_vectors,
    selected_subproblems=selected_scenarios
)

# Generate predictions (all methods use same interface)
predictions = predictor.predict(
    dual_cache=dual_cache,
    feature_vectors=feature_vectors,
    unselected_subproblems=unselected_scenarios
)

# Get training information
info = predictor.get_training_info()
```

Integration with Callbacks (SCFLP Only)
---------------------------------------
The `DualPredictor` class integrates seamlessly with the unified callback system:

1. **Initialization**: Create predictor instance in callback `__init__` (SCFLP only)
2. **One-Time Setup**: Call `predictor._train_unified()` which delegates to training_and_selection.py
3. **Iterative Prediction**: Call `predictor.predict()` during Phase 2 using pre-established mappings
4. **Unified Architecture**: Selection and training handled by training_and_selection.py
5. **SCFLP Focus**: Non-SCFLP problem types raise NotImplementedError

Primary Entry Points
--------------------
- **DualPredictor**: Main unified class for all prediction methods with integrated selection
- **_train_unified()**: Core method that handles both subproblem selection and predictor training
- **integrated_prediction_selection()**: Backend function that implements method-specific selection strategies

Mathematical Foundations
-----------------------

**KNN Method (Integrated Selection + Prediction):**
- Selection: min_{T⊆S,|T|=p} Σ_{s∈S\\T} (1/k) Σ_{t∈T_k(s)} d_F(f^s, f^t)
  Where T_k(s) are k nearest neighbors of scenario s in selected set T
- Prediction: π̂_s = (1/k) * Σ_{j∈N_k(s)} π_j for unselected scenario s
- Mapping: Fixed neighbors determined during selection phase

**Future Methods (To Be Implemented):**
- **TREE**: Will combine tree-based partitioning with optimal representative selection
- **REG**: Will integrate regression-specific selection with linear prediction
- **SIMPLE_AVG**: Will use appropriate selection strategy with simple averaging

Unified Selection Strategy:
1. Each prediction method determines its own optimal subproblem selection
2. Selection and prediction are performed in single integrated step
3. Method-specific mappings stored for prediction phase
4. No separate subproblem selection methods needed

Training Data Structure
----------------------
all_trial_duals format: {(scenario_id, iteration): structured_dual_dict}
Where structured_dual_dict contains dual variables like:
- UFL: {'lambda': float, 'mu': {j: float}}  
- UHL: {'v': {k: float}, 'u': {l: float}}
- SSLP: {'lambda': {i: float}, 'mu': {j: float}}

Training Diagnostics (REG Method)
---------------------------------
When using the regression method, detailed per-iteration training diagnostics can be
exported. The module maintains a global list `TRAINING_CALL_LOGS` which collects
training metrics for analysis.

Legacy Integration (Deprecated)
-------------------------------
Legacy functions are maintained for backward compatibility but marked as deprecated:

- `train_feasible_predictor()`: Use `DualPredictor(method='REG').train()` instead
- `train_tree_predictor()`: Use `DualPredictor(method='TREE').train()` instead

Integration Steps (external):
 1. For REG method with detailed logging, pass a `metadata` dict containing:
          {
             'problem_class': str,
             'instance_filename': str,
             'num_subproblems': int,
             'num_first_stage_variables': int,
             'iteration': int
          }
 2. Access training logs via `from Dual_Prediction.prediction import TRAINING_CALL_LOGS`

Returns
-------
The `DualPredictor.train()` method returns training information including:
- Training status and method-specific parameters
- For REG: optimization status and slack variables
- For KNN/TREE: neighbor/mapping information
"""

from gurobipy import Model, GRB, quicksum
from typing import Optional, Union
import numpy as np
import logging
import time

from Dual_Prediction.parametric import (
    PARAMETRIC_PREDICTORS,
    RHS_RIDGE_PREDICTORS,
    ParametricDualPredictor,
    is_parametric_predictor,
    representative_counts,
    solve_parametric_pmedian_selection,
)
from collections import defaultdict

# Initialize logger first
from utils.exclusive_profiler import get_profiled_logger

logger = get_profiled_logger(__name__)

# ---------------------------------------------------------------------------
# Global container for per-call training diagnostics (populated only when
# prediction method REG is used AND metadata is supplied by the caller).
# ---------------------------------------------------------------------------
TRAINING_CALL_LOGS = []

# Nonparametric control that uses exactly the same p-median representative
# selection and count-weighted common anchor as the original RHS-Ridge method,
# but without the learned residual correction.
PMEDIAN_WEIGHTED_TRANSFER = "PMEDIAN_WEIGHTED_TRANSFER"
PMEDIAN_NEAREST_TRANSFER = "PMEDIAN_NEAREST_TRANSFER"

from utils.exclusive_profiler import get_profiled_logger

logger = get_profiled_logger(__name__)


def _weights_default_if_none(feature_vectors, solved_duals, sample_weights):
    """Heuristic sample weights when none are provided.

    Inputs:
    - feature_vectors: dict[int|tuple -> np.ndarray], scenario features for all scenarios.
    - solved_duals: dict[sid -> dict], training subset with true duals.
    - sample_weights: optional dict[sid -> float] or None.

    Outputs:
    - dict[sid -> float]: nonnegative weights over selected (training) scenarios.

    Use:
    - Weight each selected scenario by the number of unselected scenarios for which
      it is the nearest neighbor in feature space (at least 1 each).
    """
    if sample_weights is not None:
        return sample_weights
    # n_t = number of unselected scenarios that are closest to solved t (K=1)
    selected = list(solved_duals.keys())
    unselected = sorted(set(feature_vectors.keys()) - set(selected))
    if not selected:
        return {}
    weights = {t: 0 for t in selected}
    for u in unselected:
        phi_u = feature_vectors[u]
        # choose closest selected in Euclidean feature space
        t_star = min(selected, key=lambda t: np.linalg.norm(phi_u - feature_vectors[t]))
        weights[t_star] += 1
    # if someone gets 0, still give a unit weight so they count
    for t in selected:
        if weights[t] == 0:
            weights[t] = 1
    return weights


def _add_value_budget_constraints(model, problem_type, data,
                                  solved_duals, feature_vectors, w_dict,
                                  first_stage_val, sample_weights,
                                  name_prefix=""):
    """
    Build linear constraints for the absolute value discrepancy per training scenario.

    Adds, for each training scenario m in M':
        s_m >= |pred_value(m) - true_value(m)|
    and returns the weighted sum sum_{m in M'} n_m * s_m (as a linear expression).

    Inputs:
    - model: gurobipy.Model to which constraints/vars are added.
    - problem_type: str in {UFL, UHL, SSLP, SCFLP}.
    - data: problem data object with sets and parameters.
    - solved_duals: dict[sid -> true duals dict] used as training samples.
    - feature_vectors: dict[sid -> np.ndarray] scenario features.
    - w_dict: dict of Gurobi variables (weights) by dual component.
    - first_stage_val: dict-like of current first-stage solution values.
    - sample_weights: dict[sid -> float] weights n_m.
    - name_prefix: str prefix for created variable/constraint names.

    Output:
    - gurobipy.LinExpr: total weighted absolute discrepancy expression.
    """
    pt = problem_type.upper()
    num_feat = len(next(iter(feature_vectors.values())))
    total = 0.0
    s_vars = {}

    for sid in solved_duals:
        phi = feature_vectors[sid]
        wgt = float(sample_weights.get(sid, 1))
        s = model.addVar(lb=0.0, name=f"{name_prefix}s_{sid}")
        s_vars[sid] = s

        if pt == "UFL":
            w_lambda = w_dict["lambda"]; w_mu = w_dict["mu"]
            lam_pred = quicksum(w_lambda[f] * phi[f] for f in range(num_feat))
            mu_pred = {j: quicksum(w_mu[j, f] * phi[f] for f in range(num_feat)) for j in data.F}
            lam_true = solved_duals[sid]["lambda"]
            mu_true = solved_duals[sid]["mu"]
            pred_obj = lam_pred - quicksum(mu_pred[j] * first_stage_val[j] for j in data.F)
            true_obj = lam_true - sum(mu_true[j] * first_stage_val[j] for j in data.F)

        elif pt == "UHL":
            w_v = w_dict["v"]; w_u = w_dict["u"]
            # Extract OD pair key robustly from data.S
            idx = sid
            s_val = data.S[idx]
            if isinstance(s_val, tuple):
                i, j = s_val
            elif hasattr(s_val, "__len__") and len(s_val) >= 2:
                i, j = int(s_val[0]), int(s_val[1])
            else:
                i, j = s_val, 0
            v_pred = {k: quicksum(w_v[int(k) if isinstance(k, np.integer) else k, f] * phi[f]
                                   for f in range(num_feat)) for k in data.N}
            u_pred = {l: quicksum(w_u[int(l) if isinstance(l, np.integer) else l, f] * phi[f]
                                   for f in range(num_feat)) for l in data.N}
            v_true = solved_duals[sid]["v"]; u_true = solved_duals[sid]["u"]
            pred_obj = quicksum(v_pred[k] * float(first_stage_val[i, k]) for k in data.N) + \
                       quicksum(u_pred[l] * float(first_stage_val[j, l]) for l in data.N)
            true_obj = sum(float(v_true[k]) * float(first_stage_val[i, k]) for k in data.N) + \
                       sum(float(u_true[l]) * float(first_stage_val[j, l]) for l in data.N)

        elif pt == "SSLP":
            w_pi, w_gamma = w_dict["pi"], w_dict["gamma"]
            demand = data.demand[sid]
            pi_pred = {i: quicksum(w_pi[i, f] * phi[f] for f in range(num_feat))
                       for i in data.C}
            gamma_pred = {(i, j): quicksum(w_gamma[i, j, f] * phi[f] for f in range(num_feat))
                          for i in data.C for j in data.F}
            pi_true = solved_duals[sid]["pi"]; gamma_true = solved_duals[sid]["gamma"]
            pred_obj = quicksum(pi_pred[i] * demand[idx] for idx, i in enumerate(data.C)) - \
                       quicksum(gamma_pred[(i, j)] * demand[idx] * first_stage_val[j]
                                for idx, i in enumerate(data.C) for j in data.F)
            true_obj = sum(pi_true[i] * demand[idx] for idx, i in enumerate(data.C)) - \
                       sum(gamma_true[(i, j)] * demand[idx] * first_stage_val[j]
                           for idx, i in enumerate(data.C) for j in data.F)

        elif pt == "SCFLP":
            w_mu, w_nu = w_dict["mu"], w_dict["nu"]
            mu_pred = {i: quicksum(w_mu[int(i) if isinstance(i, np.integer) else i, f] * phi[f]
                                    for f in range(num_feat)) for i in data.I}
            nu_pred = {j: quicksum(w_nu[int(j) if isinstance(j, np.integer) else j, f] * phi[f]
                                    for f in range(num_feat)) for j in data.J}
            mu_true = solved_duals[sid]["mu"]; nu_true = solved_duals[sid]["nu"]
            demand = data.scenarios[sid]
            pred_obj = quicksum(mu_pred[i] * float(demand[int(i)]) for i in data.I) - \
                       quicksum(nu_pred[j] * float(data.capacity[int(j)]) * float(first_stage_val[int(j)])
                                for j in data.J)
            true_obj = sum(float(mu_true[int(i)]) * float(demand[int(i)]) for i in data.I) - \
                       sum(float(nu_true[int(j)]) * float(data.capacity[int(j)]) * float(first_stage_val[int(j)])
                           for j in data.J)

        else:
            raise NotImplementedError(f"Value budget not implemented for: {problem_type}")

        model.addConstr(s >=  pred_obj - true_obj, name=f"{name_prefix}s_pos_{sid}")
        model.addConstr(s >= -pred_obj + true_obj, name=f"{name_prefix}s_neg_{sid}")
        total += wgt * s

    return total


def _add_l1_dual_error_budget(model, problem_type, data,
                              solved_duals, feature_vectors, w_dict,
                              sample_weights, name_prefix=""):
    """
    Build component-wise absolute error constraints for dual predictions and return
    the weighted L1 sum over training scenarios.

    For each training scenario m and each dual component a, add:
        u_{m,a} >= |pred_{m,a} - true_{m,a}|
    and return sum_m n_m * sum_a u_{m,a}.

    Inputs:
    - model, problem_type, data, solved_duals, feature_vectors, w_dict, sample_weights, name_prefix

    Output:
    - gurobipy.LinExpr: total weighted L1 error expression.
    """
    pt = problem_type.upper()
    num_feat = len(next(iter(feature_vectors.values())))
    total = 0.0

    for sid in solved_duals:
        phi = feature_vectors[sid]
        wgt = float(sample_weights.get(sid, 1))

        if pt == "UFL":
            w_lambda = w_dict["lambda"]; w_mu = w_dict["mu"]
            lam_pred = quicksum(w_lambda[f] * phi[f] for f in range(num_feat))
            lam_true = solved_duals[sid]["lambda"]
            u_lam = model.addVar(lb=0.0, name=f"{name_prefix}u_lam_{sid}")
            model.addConstr(u_lam >=  lam_pred - lam_true, name=f"{name_prefix}u_lam_pos_{sid}")
            model.addConstr(u_lam >= -lam_pred + lam_true, name=f"{name_prefix}u_lam_neg_{sid}")
            total += wgt * u_lam
            for j in data.F:
                mu_pred = quicksum(w_mu[j, f] * phi[f] for f in range(num_feat))
                mu_true = solved_duals[sid]["mu"][j]
                u = model.addVar(lb=0.0, name=f"{name_prefix}u_mu_{sid}_{int(j)}")
                model.addConstr(u >=  mu_pred - mu_true, name=f"{name_prefix}u_mu_pos_{sid}_{int(j)}")
                model.addConstr(u >= -mu_pred + mu_true, name=f"{name_prefix}u_mu_neg_{sid}_{int(j)}")
                total += wgt * u

        elif pt == "UHL":
            w_v = w_dict["v"]; w_u = w_dict["u"]
            v_true = solved_duals[sid]["v"]; u_true = solved_duals[sid]["u"]
            for k in data.N:
                k_idx = int(k) if isinstance(k, np.integer) else k
                v_pred = quicksum(w_v[k_idx, f] * phi[f] for f in range(num_feat))
                uvar = model.addVar(lb=0.0, name=f"{name_prefix}u_v_{sid}_{k_idx}")
                model.addConstr(uvar >=  v_pred - v_true[k_idx], name=f"{name_prefix}u_v_pos_{sid}_{k_idx}")
                model.addConstr(uvar >= -v_pred + v_true[k_idx], name=f"{name_prefix}u_v_neg_{sid}_{k_idx}")
                total += wgt * uvar
            for l in data.N:
                l_idx = int(l) if isinstance(l, np.integer) else l
                u_pred = quicksum(w_u[l_idx, f] * phi[f] for f in range(num_feat))
                uvar = model.addVar(lb=0.0, name=f"{name_prefix}u_u_{sid}_{l_idx}")
                model.addConstr(uvar >=  u_pred - u_true[l_idx], name=f"{name_prefix}u_u_pos_{sid}_{l_idx}")
                model.addConstr(uvar >= -u_pred + u_true[l_idx], name=f"{name_prefix}u_u_neg_{sid}_{l_idx}")
                total += wgt * uvar

        elif pt == "SSLP":
            w_pi, w_gamma = w_dict["pi"], w_dict["gamma"]
            pi_true = solved_duals[sid]["pi"]; gamma_true = solved_duals[sid]["gamma"]
            for i in data.C:
                pi_pred = quicksum(w_pi[i, f] * phi[f] for f in range(num_feat))
                u = model.addVar(lb=0.0, name=f"{name_prefix}u_pi_{sid}_{int(i)}")
                model.addConstr(u >=  pi_pred - pi_true[i], name=f"{name_prefix}u_pi_pos_{sid}_{int(i)}")
                model.addConstr(u >= -pi_pred + pi_true[i], name=f"{name_prefix}u_pi_neg_{sid}_{int(i)}")
                total += wgt * u
            for i in data.C:
                for j in data.F:
                    gam_pred = quicksum(w_gamma[i, j, f] * phi[f] for f in range(num_feat))
                    u = model.addVar(lb=0.0, name=f"{name_prefix}u_gam_{sid}_{int(i)}_{int(j)}")
                    true = gamma_true[(i, j)]
                    model.addConstr(u >=  gam_pred - true, name=f"{name_prefix}u_gam_pos_{sid}_{int(i)}_{int(j)}")
                    model.addConstr(u >= -gam_pred + true, name=f"{name_prefix}u_gam_neg_{sid}_{int(i)}_{int(j)}")
                    total += wgt * u

        elif pt == "SCFLP":
            w_mu, w_nu = w_dict["mu"], w_dict["nu"]
            mu_true = solved_duals[sid]["mu"]; nu_true = solved_duals[sid]["nu"]
            for i in data.I:
                i_idx = int(i) if isinstance(i, np.integer) else i
                mu_pred = quicksum(w_mu[i_idx, f] * phi[f] for f in range(num_feat))
                u = model.addVar(lb=0.0, name=f"{name_prefix}u_mu_{sid}_{i_idx}")
                model.addConstr(u >=  mu_pred - mu_true[i_idx], name=f"{name_prefix}u_mu_pos_{sid}_{i_idx}")
                model.addConstr(u >= -mu_pred + mu_true[i_idx], name=f"{name_prefix}u_mu_neg_{sid}_{i_idx}")
                total += wgt * u
            for j in data.J:
                j_idx = int(j) if isinstance(j, np.integer) else j
                nu_pred = quicksum(w_nu[j_idx, f] * phi[f] for f in range(num_feat))
                u = model.addVar(lb=0.0, name=f"{name_prefix}u_nu_{sid}_{j_idx}")
                model.addConstr(u >=  nu_pred - nu_true[j_idx], name=f"{name_prefix}u_nu_pos_{sid}_{j_idx}")
                model.addConstr(u >= -nu_pred + nu_true[j_idx], name=f"{name_prefix}u_nu_neg_{sid}_{j_idx}")
                total += wgt * u

        else:
            raise NotImplementedError(f"L1 dual error budget not implemented for: {problem_type}")

    return total


def train_feasible_predictor(problem_type, data, solved_duals, feature_vectors,
                             first_stage_val=None,
                             L_v=None, L_d=None, p_v=1.0, p_d=1.0,
                             sample_weights=None,
                             metadata=None,
                             log_training=True):
    """Train a linear dual predictor model with feasibility constraints.

    Optimization model (LP):
        minimize    p_v * alpha_v + p_d * alpha_d
        subject to  sum_{m in M'} n_m |ValuePred(m) - ValueTrue(m)| <= L_v + alpha_v
                    sum_{m in M'} n_m ||PredDual(m) - TrueDual(m)||_1 <= L_d + alpha_d
                    PredDual(m) satisfies dual-feasibility for all m in M \\ M'

    Inputs:
    - problem_type: str, in {UFL, UHL, SSLP, SCFLP}.
    - data: problem data object containing sets (F, C, N, I, J, S, ...) and parameters.
    - solved_duals: dict[sid -> duals dict], training scenarios with exact duals.
    - feature_vectors: dict[sid -> np.ndarray], feature vectors for all scenarios.
    - first_stage_val: dict-like of first-stage solution values used to compute objective value.
    - L_v, L_d: float budgets for value discrepancy and L1 dual error, respectively.
    - p_v, p_d: objective weights for the slack variables alpha_v and alpha_d.
    - sample_weights: optional dict[sid -> float] weights; if None, a NN-based heuristic is used.

    Outputs:
    - dict of learned weights suitable for predict_duals() in callbacks:
        UFL:  {"lambda": np.ndarray[d], "mu": {j: np.ndarray[d] for j in F}}
        UHL:  {"v": {k: np.ndarray[d] for k in N}, "u": {l: np.ndarray[d] for l in N}}
        SSLP: {"pi": {i: np.ndarray[d] for i in C}, "gamma": {(i,j): np.ndarray[d] for i in C, j in F}}
        SCFLP:{"mu": {i: np.ndarray[d] for i in I}, "nu": {j: np.ndarray[d] for j in J}}

    Use:
    - Called from Problem/callbacks.py (Phase 2) to train regression weights over selected scenarios.
    - The returned weights are used to predict duals for unselected scenarios and generate cuts.
    """
    if not feature_vectors:
        logger.warning("[Training] Empty feature_vectors; skipping.")
        return None

    pt = problem_type.upper()
    num_feat = len(next(iter(feature_vectors.values())))
    model_name = f"{pt}_Train_LP"
    m = Model(model_name); m.Params.OutputFlag = 0

    # --- regression weights (NO sign restriction; feasibility is enforced on predictions) ---
    if pt == "UFL":
        # Keep lambda weights as a simple list for straightforward indexing and packing
        w_lambda = [m.addVar(lb=-GRB.INFINITY, name=f"w_lambda[{f}]") for f in range(num_feat)]
        w_mu     = m.addVars(list(data.F), range(num_feat), lb=-GRB.INFINITY, name="w_mu")
        w_dict   = {"lambda": w_lambda, "mu": w_mu}

    elif pt == "UHL":
        w_v = m.addVars(list(data.N), range(num_feat), lb=-GRB.INFINITY, name="w_v")
        w_u = m.addVars(list(data.N), range(num_feat), lb=-GRB.INFINITY, name="w_u")
        w_dict = {"v": w_v, "u": w_u}

    elif pt == "SSLP":
        w_pi    = m.addVars(list(data.C), range(num_feat), lb=-GRB.INFINITY, name="w_pi")
        w_gamma = m.addVars(list(data.C), list(data.F), range(num_feat), lb=-GRB.INFINITY, name="w_gamma")
        w_dict  = {"pi": w_pi, "gamma": w_gamma}

    elif pt == "SCFLP":
        w_mu = m.addVars(list(data.I), range(num_feat), lb=-GRB.INFINITY, name="w_mu")
        w_nu = m.addVars(list(data.J), range(num_feat), lb=-GRB.INFINITY, name="w_nu")
        w_dict = {"mu": w_mu, "nu": w_nu}

    else:
        raise NotImplementedError(f"Training not implemented for problem type: {problem_type}")

    # --- slacks and objective ---
    alpha_v = m.addVar(lb=0.0, name="alpha_v")
    alpha_d = m.addVar(lb=0.0, name="alpha_d")
    m.setObjective(p_v * alpha_v + p_d * alpha_d, GRB.MINIMIZE)

    # --- sets and weights ---
    solved_ids = list(solved_duals.keys())
    unsolved_ids = sorted(set(feature_vectors.keys()) - set(solved_ids))
    sample_weights = _weights_default_if_none(feature_vectors, solved_duals, sample_weights)

    # --- (1) Value discrepancy budget  Sum n_m |Delta_m| <= L_v + alpha_v ---
    if first_stage_val is None:
        raise ValueError("[Training] first_stage_val is required for the value budget (L_v).")
    if L_v is None:
        raise ValueError("[Training] L_v (value discrepancy budget) must be provided.")
    val_sum = _add_value_budget_constraints(
        m, pt, data, solved_duals, feature_vectors, w_dict, first_stage_val, sample_weights, name_prefix="val_"
    )
    m.addConstr(val_sum <= float(L_v) + alpha_v, name="value_budget")

    # --- (2) L1 dual error budget  Sum n_m ||Theta h_m - lambda_m||_1 <= L_d + alpha_d ---
    if L_d is None:
        raise ValueError("[Training] L_d (L1 dual error budget) must be provided.")
    l1_sum = _add_l1_dual_error_budget(
        m, pt, data, solved_duals, feature_vectors, w_dict, sample_weights, name_prefix="err_"
    )
    m.addConstr(l1_sum <= float(L_d) + alpha_d, name="dual_l1_budget")

    # --- (3) Dual-feasibility on predictions for UNSOLVED scenarios ---
    #      A_m^T (Theta h_m) >= -omega_m, plus sign constraints as dictated by each dual
    for sid in unsolved_ids:
        phi = feature_vectors[sid]

        if pt == "UFL":
            lam_pred = quicksum(w * phi_f for w, phi_f in zip(w_dict["lambda"], [phi[f] for f in range(num_feat)]))
            for j in data.F:
                mu_pred = quicksum(w_dict["mu"][j, f] * phi[f] for f in range(num_feat))
                # λ - μ_j ≤ c_ij * d_i
                i = sid
                m.addConstr(lam_pred - mu_pred <= data.c[i, j] * data.d[i], name=f"feas_ufl_cap_{sid}_{int(j)}")
                # μ_j ≥ 0
                m.addConstr(mu_pred >= 0.0, name=f"feas_ufl_nonneg_mu_{sid}_{int(j)}")

        elif pt == "UHL":
            # v_k + u_l ≤ w_ij(β d_ik + γ d_kl + β d_lj)
            s_val = data.S[sid]
            if isinstance(s_val, tuple):
                i, j = s_val
            elif hasattr(s_val, "__len__") and len(s_val) >= 2:
                i, j = int(s_val[0]), int(s_val[1])
            else:
                i, j = s_val, 0
            for k in data.N:
                for l in data.N:
                    v_pred = quicksum(w_dict["v"][int(k) if isinstance(k, np.integer) else k, f] * phi[f]
                                       for f in range(num_feat))
                    u_pred = quicksum(w_dict["u"][int(l) if isinstance(l, np.integer) else l, f] * phi[f]
                                       for f in range(num_feat))
                    rhs = data.w[i, j] * (data.beta * data.dist[i, k] +
                                          data.gamma * data.dist[k, l] +
                                          data.beta * data.dist[l, j])
                    m.addConstr(v_pred + u_pred <= rhs, name=f"feas_uhl_{sid}_{int(k)}_{int(l)}")

        elif pt == "SSLP":
            # π_i − γ_ij ≤ c_ij,   π_i ≤ P_i,   γ_ij ≥ 0
            for i in data.C:
                pi_pred = quicksum(w_dict["pi"][i, f] * phi[f] for f in range(num_feat))
                # penalty is indexed consistent with C; if it's a list, map by enumerate
                # Here we assume penalty is keyed by position; fall back to enumerate when used elsewhere.
                # If data.penalty is list-like, access by the position of i in data.C
                try:
                    pen = data.penalty[i]
                except Exception:
                    # fallback: find index
                    idx = list(data.C).index(i)
                    pen = data.penalty[idx]
                m.addConstr(pi_pred <= pen, name=f"feas_ssl_pi_cap_{sid}_{int(i)}")
                for j in data.F:
                    gam_pred = quicksum(w_dict["gamma"][i, j, f] * phi[f] for f in range(num_feat))
                    cij = data.serve_cost[(i, j)]
                    m.addConstr(pi_pred - gam_pred <= cij, name=f"feas_ssl_ij_{sid}_{int(i)}_{int(j)}")
                    m.addConstr(gam_pred >= 0.0, name=f"feas_ssl_gam_nonneg_{sid}_{int(i)}_{int(j)}")

        elif pt == "SCFLP":
            # μ_i − ν_j ≤ c_ij,   μ_i ≥ 0,   ν_j ≥ 0
            for i in data.I:
                i_idx = int(i) if isinstance(i, np.integer) else i
                mu_pred = quicksum(w_dict["mu"][i_idx, f] * phi[f] for f in range(num_feat))
                m.addConstr(mu_pred >= 0.0, name=f"feas_scflp_mu_nonneg_{sid}_{i_idx}")
                for j in data.J:
                    j_idx = int(j) if isinstance(j, np.integer) else j
                    nu_pred = quicksum(w_dict["nu"][j_idx, f] * phi[f] for f in range(num_feat))
                    if i == list(data.I)[0]:  # nonneg once per j to avoid duplicates
                        m.addConstr(nu_pred >= 0.0, name=f"feas_scflp_nu_nonneg_{sid}_{j_idx}")
                    c_ij = float(data.cost[i_idx, j_idx])
                    m.addConstr(mu_pred - nu_pred <= c_ij, name=f"feas_scflp_cap_{sid}_{i_idx}_{j_idx}")

        else:
            raise NotImplementedError(f"Feasibility not implemented for: {problem_type}")

    # --- solve ---
    m.optimize()
    status = m.status
    if status != GRB.OPTIMAL:
        logger.warning(f"[Training] No optimal solution for {model_name}. Status={status}")

    # Capture alpha slacks irrespective of status (they may be None if infeasible before var creation)
    try:
        alpha_v_val = alpha_v.X
    except Exception:
        alpha_v_val = None
    try:
        alpha_d_val = alpha_d.X
    except Exception:
        alpha_d_val = None

    # Log (only if requested and metadata present)
    if log_training and metadata and isinstance(metadata, dict):
        row = {
            'problem_class': metadata.get('problem_class'),
            'instance_filename': metadata.get('instance_filename'),
            'num_subproblems': metadata.get('num_subproblems'),
            'num_first_stage_variables': metadata.get('num_first_stage_variables'),
            'dual_difference_pred_loss': L_d,
            'obj_difference_pred_loss': L_v,
            'p_d': p_d,
            'p_v': p_v,
            'alpha_v': alpha_v_val,
            'alpha_d': alpha_d_val,
            'iteration': metadata.get('iteration'),
            'result': 'OPTIMAL' if status == GRB.OPTIMAL else str(status)
        }
        TRAINING_CALL_LOGS.append(row)

    if status != GRB.OPTIMAL:
        return None

    # --- pack weights like before ---
    if pt == "UFL":
        w_lambda_vec = np.array(m.getAttr('X', w_dict["lambda"]))
        w_mu_mat = {j: np.array([w_dict["mu"][j, f].X for f in range(num_feat)]) for j in data.F}
        result = {"lambda": w_lambda_vec, "mu": w_mu_mat}
        result['__alpha_v__'] = alpha_v_val
        result['__alpha_d__'] = alpha_d_val
        result['__status__'] = status
        return result

    elif pt == "UHL":
        w_v_mat = {int(k) if isinstance(k, np.integer) else k:
                   np.array([w_dict["v"][int(k) if isinstance(k, np.integer) else k, f].X
                             for f in range(num_feat)]) for k in data.N}
        w_u_mat = {int(l) if isinstance(l, np.integer) else l:
                   np.array([w_dict["u"][int(l) if isinstance(l, np.integer) else l, f].X
                             for f in range(num_feat)]) for l in data.N}
        result = {"v": w_v_mat, "u": w_u_mat}
        result['__alpha_v__'] = alpha_v_val
        result['__alpha_d__'] = alpha_d_val
        result['__status__'] = status
        return result

    elif pt == "SSLP":
        w_pi_mat = {i: np.array([w_dict["pi"][i, f].X for f in range(num_feat)])
                    for i in data.C}
        w_gamma_mat = {(i, j): np.array([w_dict["gamma"][i, j, f].X for f in range(num_feat)])
                       for i in data.C for j in data.F}
        result = {"pi": w_pi_mat, "gamma": w_gamma_mat}
        result['__alpha_v__'] = alpha_v_val
        result['__alpha_d__'] = alpha_d_val
        result['__status__'] = status
        return result

    elif pt == "SCFLP":
        w_mu_mat = {int(i) if isinstance(i, np.integer) else i:
                    np.array([w_dict["mu"][int(i) if isinstance(i, np.integer) else i, f].X
                              for f in range(num_feat)]) for i in data.I}
        w_nu_mat = {int(j) if isinstance(j, np.integer) else j:
                    np.array([w_dict["nu"][int(j) if isinstance(j, np.integer) else j, f].X
                              for f in range(num_feat)]) for j in data.J}
        result = {"mu": w_mu_mat, "nu": w_nu_mat}
        result['__alpha_v__'] = alpha_v_val
        result['__alpha_d__'] = alpha_d_val
        result['__status__'] = status
        return result
    
    else:
        raise ValueError(f"Unsupported problem type: {pt}")


def train_tree_predictor(problem_type, data, all_trial_duals, feature_vectors, 
                        selected_subproblems, max_depth=None, min_samples_leaf=5):
    """
    Train a tree-based predictor that creates fixed mappings from unselected to selected scenarios.
    
    This function implements the tree-based prediction method that works like KNN with fixed mappings.
    It builds a decision tree to optimally group scenarios, then creates static mappings from each 
    unselected scenario to a subset of selected scenarios with uniform weights.
    
    Mathematical Foundation:
    ----------------------
    **Tree-Based Grouping Objective**: 
    min_tree Σ_{L∈Leaves} Σ_{m'∈L∩U, k=1...K} ||λ_m'^(k) - λ̂_m'^(k)||_2^2
    
    Where:
    - L = leaf in the decision tree
    - U = set of unselected scenarios  
    - λ_m'^(k) = true dual solution for unselected scenario m' at iteration k
    - λ̂_m'^(k) = (1/|J_L|) * Σ_{j∈J_L} λ_j^(k) = uniform average over selected scenarios in leaf L
    - J_L = set of selected scenarios that fall in leaf L
    
    **Fixed Mapping Strategy**:
    1. Build decision tree using Phase 1 trial data to minimize prediction error
    2. For each unselected scenario, determine which leaf it falls into
    3. Create fixed mapping: unselected_scenario → {selected_scenarios_in_same_leaf}
    4. Use uniform weights: w_j = 1/|J_L| for all j ∈ J_L
    5. Apply these fixed mappings throughout Phase 2 (like KNN)
    
    **Key Characteristics**:
    - **Fixed mappings**: Once computed, mappings don't change across Phase 2 iterations
    - **Uniform weights**: All selected scenarios in same leaf get equal weight 1/|J_L|
    - **Variable group sizes**: Different unselected scenarios may map to different numbers of selected scenarios
    - **Consistent with KNN approach**: Static neighbor assignment with uniform averaging
    
    Args:
    ----
    problem_type (str): Problem type ('UFL', 'UHL', 'SSLP', 'SCFLP')
    data: Problem data object with sets and parameters
    all_trial_duals (dict): Dictionary mapping (sid, iteration) -> dual_dict for all scenarios
                           from Phase 1. Format: {(scenario_id, iteration): structured_dual_dict}
    feature_vectors (dict): Dictionary mapping scenario_id -> feature_vector for all scenarios
    selected_subproblems (set): Set of scenario IDs that will be solved exactly in Phase 2
    max_depth (int, optional): Maximum depth of the decision tree. Default: None (unlimited)
    min_samples_leaf (int): Minimum number of samples required to be at a leaf node. Default: 5
    
    Returns:
    -------
    dict: Dictionary containing fixed mappings from unselected to selected scenarios.
          
          **Structure** (similar to KNN):
          {
              '__prediction_mappings__': {
                  unselected_sid: {
                      'neighbors': [selected_sid1, selected_sid2, ...],  # Scenarios in same leaf
                      'weights': [w1, w2, ...],                         # Uniform weights = 1/|neighbors|
                  },
                  ...
              },
              '__tree_type__': 'TREE_FIXED_MAPPING',
              '__selected_scenarios__': set,
              '__problem_type__': str
          }
          
    Example Output:
    --------------
    ```
    {
        '__prediction_mappings__': {
            12: {'neighbors': [1, 2, 4], 'weights': [0.333, 0.333, 0.333]},
            13: {'neighbors': [1, 5, 7, 8, 9], 'weights': [0.2, 0.2, 0.2, 0.2, 0.2]},
            14: {'neighbors': [2, 3], 'weights': [0.5, 0.5]},
            ...
        },
        '__tree_type__': 'TREE_FIXED_MAPPING',
        '__selected_scenarios__': {1, 2, 3, 4, 5, 7, 8, 9},
        '__problem_type__': 'UFL'
    }
    ```
          
    Raises:
    ------
    ValueError: If insufficient training data provided
    ValueError: If unsupported problem type
    """
    
    if not all_trial_duals:
        raise ValueError("No training data provided in all_trial_duals")
    
    logger.info(f"Training tree-based fixed mapping predictor for {problem_type} with {len(all_trial_duals)} training samples")
    
    pt = problem_type.upper()
    
    # Separate selected and unselected scenarios
    all_scenarios = set(feature_vectors.keys())
    selected_scenarios = set(selected_subproblems)
    unselected_scenarios = all_scenarios - selected_scenarios
    
    if not unselected_scenarios:
        logger.warning("No unselected scenarios for tree training - all scenarios are selected")
        return {'__prediction_mappings__': {}, '__tree_type__': 'TREE_FIXED_MAPPING', 
                '__selected_scenarios__': selected_scenarios, '__problem_type__': pt}
    
    if not selected_scenarios:
        logger.warning("No selected scenarios for tree training - cannot perform uniform averaging")
        return {'__prediction_mappings__': {}, '__tree_type__': 'TREE_FIXED_MAPPING',
                '__selected_scenarios__': set(), '__problem_type__': pt}
    
    logger.info(f"Training tree with {len(selected_scenarios)} selected and {len(unselected_scenarios)} unselected scenarios")
    
    # Extract unique iterations from trial data
    iterations = set()
    for (sid, iteration) in all_trial_duals.keys():
        iterations.add(iteration)
    iterations = sorted(iterations)
    
    logger.info(f"Found {len(iterations)} trial iterations: {iterations}")
    
    # Build a single decision tree to determine scenario groupings
    # We'll use a representative dual variable (lambda for UFL, first component for others)
    logger.info("Building decision tree for scenario grouping...")
    
    # Choose representative dual variable based on problem type
    if pt == "UFL":
        repr_dual_var, repr_dual_index = 'lambda', None
    elif pt == "UHL":
        repr_dual_var, repr_dual_index = 'v', list(data.N)[0]
    elif pt == "SSLP":
        repr_dual_var, repr_dual_index = 'pi', list(data.C)[0]
    elif pt == "SCFLP":
        repr_dual_var, repr_dual_index = 'mu', list(data.I)[0]
    else:
        raise ValueError(f"Unsupported problem type for tree training: {problem_type}")
    
    # Build the grouping tree
    grouping_tree = _build_custom_tree(
        all_trial_duals, feature_vectors, selected_scenarios, unselected_scenarios,
        iterations, repr_dual_var, repr_dual_index, max_depth, min_samples_leaf
    )
    
    logger.info("Tree construction complete. Extracting fixed mappings...")
    
    # Extract fixed mappings from the tree
    prediction_mappings = {}
    
    for unselected_sid in unselected_scenarios:
        if unselected_sid not in feature_vectors:
            logger.warning(f"Feature vector not found for unselected scenario {unselected_sid}")
            continue
            
        feature_vector = feature_vectors[unselected_sid]
        
        # Traverse tree to find leaf
        leaf_selected_scenarios = _get_leaf_selected_scenarios(grouping_tree, feature_vector)
        
        if not leaf_selected_scenarios:
            # Fallback: use closest selected scenario in feature space
            logger.warning(f"No selected scenarios in leaf for unselected scenario {unselected_sid}, using nearest neighbor fallback")
            closest_selected = min(selected_scenarios, 
                                 key=lambda s: np.linalg.norm(feature_vector - feature_vectors[s])
                                 if s in feature_vectors else float('inf'))
            leaf_selected_scenarios = [closest_selected]
        
        # Create uniform weights
        n_neighbors = len(leaf_selected_scenarios)
        uniform_weight = 1.0 / n_neighbors
        weights = [uniform_weight] * n_neighbors
        
        prediction_mappings[unselected_sid] = {
            'neighbors': list(leaf_selected_scenarios),
            'weights': weights
        }
        
        # logger.debug(f"Unselected scenario {unselected_sid} → Selected scenarios {leaf_selected_scenarios} (uniform weights: {uniform_weight:.4f})")
    
    # Summary statistics
    neighbor_counts = [len(mapping['neighbors']) for mapping in prediction_mappings.values()]
    if neighbor_counts:
        avg_neighbors = np.mean(neighbor_counts)
        min_neighbors = min(neighbor_counts)
        max_neighbors = max(neighbor_counts)
        logger.info(f"Fixed mapping statistics: avg={avg_neighbors:.2f}, min={min_neighbors}, max={max_neighbors} neighbors per unselected scenario")
    
    # Return fixed mappings in KNN-compatible format
    result = {
        '__prediction_mappings__': prediction_mappings,
        '__tree_type__': 'TREE_FIXED_MAPPING',
        '__selected_scenarios__': selected_scenarios.copy(),
        '__problem_type__': pt
    }
    
    logger.info(f"Successfully created tree-based fixed mappings for {len(prediction_mappings)} unselected scenarios")
    
    return result


def _get_leaf_selected_scenarios(tree_node, feature_vector):
    """
    Traverse the decision tree to find selected scenarios in the leaf containing a given feature vector.
    
    This function performs tree traversal to determine which selected scenarios should be used
    for predicting dual variables for an unselected scenario with the given feature vector.
    Used during the fixed mapping creation phase.
    
    **Tree Traversal Algorithm**:
    1. **Start at root**: Begin traversal from provided tree node (typically root)
    2. **Internal node logic**: If feature[split_index] <= threshold, go left; otherwise go right
    3. **Leaf node termination**: Return selected scenarios stored in the leaf
    4. **Fallback handling**: Handle missing children gracefully
    
    **Fixed Mapping Usage**:
    This function is called once per unselected scenario during training to establish
    the fixed neighbor assignments that will be used throughout Phase 2.
    
    Args:
    ----
    tree_node (CustomTreeNode): Root of the decision tree or current node in traversal
    feature_vector (array-like): Feature representation of scenario requiring neighbor assignment
                                 Format: np.array or list of numerical feature values
        
    Returns:
    -------
    list: Selected scenario IDs that will serve as neighbors for prediction
          These scenarios fall in the same tree leaf as the input feature vector
          
    **Return Behavior**:
    - **Leaf reached**: Returns list(leaf.selected_scenarios)
    - **Missing children**: Returns selected_scenarios from current node (fallback)
    - **Empty leaf**: Returns empty list (will need special handling in caller)
    
    **Integration with Fixed Mappings**:
    ```python
    # Usage example during fixed mapping creation:
    for unselected_sid in unselected_scenarios:
        feature_vec = feature_vectors[unselected_sid]
        neighbors = _get_leaf_selected_scenarios(tree_root, feature_vec)
        weights = [1.0/len(neighbors)] * len(neighbors)  # Uniform weights
        mappings[unselected_sid] = {'neighbors': neighbors, 'weights': weights}
    ```
    
    Time Complexity: O(tree_depth) = O(log n) for balanced trees, O(n) worst case
    Space Complexity: O(tree_depth) due to recursion stack
    """
    if tree_node.is_leaf:
        return list(tree_node.selected_scenarios)
    else:
        # Internal node: traverse to appropriate child
        if tree_node.left is not None and feature_vector[tree_node.feature_index] <= tree_node.threshold:
            return _get_leaf_selected_scenarios(tree_node.left, feature_vector)
        elif tree_node.right is not None:
            return _get_leaf_selected_scenarios(tree_node.right, feature_vector)
        else:
            # Fallback: return selected scenarios from current node
            return list(tree_node.selected_scenarios)


class CustomTreeNode:
    """
    Custom decision tree node for ML-augmented Benders decomposition tree-based prediction.
    
    This class implements nodes for the optimization-based decision tree that directly optimizes
    the mathematical objective function specific to dual variable prediction in Benders decomposition.
    Unlike standard decision trees, this tree minimizes prediction error for dual variables using
    uniform averaging within leaves.
    
    Mathematical Basis:
    -----------------
    **Tree Construction Objective**:
    min_tree Σ_{L∈Leaves} Σ_{m'∈L∩U, k=1...K} ||λ_m'^(k) - λ̂_m'^(k)||_2^2
    
    Where:
    - L = leaf node in the decision tree
    - U = set of unselected scenarios (requiring prediction)
    - λ_m'^(k) = true dual solution for unselected scenario m' at trial iteration k
    - λ̂_m'^(k) = (1/|J_L|) * Σ_{j∈J_L} λ_j^(k) = uniform average over selected scenarios in leaf L
    - J_L = set of selected scenarios that fall into leaf L
    
    **Key Properties**:
    - **Binary splits**: Each internal node splits on feature[i] <= threshold
    - **Uniform averaging**: Leaf predictions use equal weights for all selected scenarios in leaf
    - **Fixed mappings**: Once built, tree provides static assignments (no recomputation in Phase 2)
    - **Dual-aware**: Splitting criterion directly optimizes dual variable prediction accuracy
    
    Node Structure:
    --------------
    **Internal Nodes**:
    - is_leaf = False
    - feature_index: Which feature dimension to split on (0 to n_features-1)
    - threshold: Split threshold (scenarios with feature[feature_index] <= threshold go left)
    - left/right: Child nodes for <= and > threshold respectively
    
    **Leaf Nodes**:
    - is_leaf = True  
    - selected_scenarios: Set of selected scenario IDs in this leaf
    - unselected_scenarios: Set of unselected scenario IDs in this leaf
    - Used for creating fixed mappings: unselected → selected scenarios in same leaf
    
    Usage in Fixed Mapping Pipeline:
    -------------------------------
    1. **Training Phase**: Build tree using Phase 1 dual solutions and features
    2. **Mapping Creation**: For each unselected scenario, find its leaf and extract selected scenarios
    3. **Weight Assignment**: Uniform weights = 1/|selected_scenarios_in_leaf|
    4. **Phase 2 Prediction**: Use fixed mappings for all subsequent iterations
    
    Integration with Prediction Methods:
    ----------------------------------
    - **Similar to KNN**: Provides fixed neighbor assignments with uniform weights
    - **Different from KNN**: Neighbors determined by tree partitioning, not feature distance
    - **Consistent interface**: Returns same fixed mapping format as KNN method
    """
    
    def __init__(self):
        self.is_leaf: bool = True
        self.feature_index: Optional[int] = None  # Index of the feature to split on
        self.threshold: Optional[float] = None    # Threshold value for the split
        self.left: Optional['CustomTreeNode'] = None   # Left child node
        self.right: Optional['CustomTreeNode'] = None  # Right child node
        self.scenarios: set = set()    # Set of scenario IDs in this node
        self.selected_scenarios: set = set()    # Selected scenarios in this node
        self.unselected_scenarios: set = set()  # Unselected scenarios in this node
        self.prediction_value: Optional[float] = None  # For leaf nodes: the uniform average prediction
        
    def predict(self, feature_vector, all_trial_duals, dual_var, dual_index, iterations, selected_scenarios_global):
        """
        Predict the dual value for a given feature vector.
        
        Args:
            feature_vector (np.array): Feature vector for the scenario
            all_trial_duals (dict): All trial dual solutions
            dual_var (str): Name of the dual variable ('lambda', 'mu', etc.)
            dual_index: Index for the dual variable (None for scalar, int/tuple for indexed)
            iterations (list): List of trial iterations
            selected_scenarios_global (set): Global set of selected scenarios
            
        Returns:
            float: Predicted dual value using uniform averaging
        """
        if self.is_leaf:
            # Leaf node: compute uniform average over selected scenarios in this leaf
            if not self.selected_scenarios:
                # Fallback: if no selected scenarios in this leaf, return 0
                logger.warning(f"No selected scenarios in leaf for dual variable {dual_var}")
                return 0.0
            
            # Compute uniform average over selected scenarios and all iterations
            total_sum = 0.0
            count = 0
            
            for sid in self.selected_scenarios:
                for iteration in iterations:
                    if (sid, iteration) in all_trial_duals:
                        dual_dict = all_trial_duals[(sid, iteration)]
                        if dual_index is None:
                            # Scalar dual variable (e.g., lambda)
                            total_sum += dual_dict[dual_var]
                        else:
                            # Indexed dual variable (e.g., mu[j], gamma[(i,j)])
                            total_sum += dual_dict[dual_var][dual_index]
                        count += 1
            
            if count == 0:
                logger.warning(f"No dual values found for prediction in leaf")
                return 0.0
                
            return total_sum / count
        else:
            # Internal node: traverse to appropriate child
            if self.left is not None and feature_vector[self.feature_index] <= self.threshold:
                return self.left.predict(feature_vector, all_trial_duals, dual_var, dual_index, iterations, selected_scenarios_global)
            elif self.right is not None:
                return self.right.predict(feature_vector, all_trial_duals, dual_var, dual_index, iterations, selected_scenarios_global)
            else:
                logger.error("Internal node missing children")
                return 0.0
    
    def get_leaf_assignment(self, feature_vector):
        """
        Get the leaf ID for a given feature vector.
        
        Returns:
            int: Unique identifier for the leaf node
        """
        if self.is_leaf:
            return id(self)  # Use object ID as unique leaf identifier
        else:
            if self.left is not None and feature_vector[self.feature_index] <= self.threshold:
                return self.left.get_leaf_assignment(feature_vector)
            elif self.right is not None:
                return self.right.get_leaf_assignment(feature_vector)
            else:
                logger.error("Internal node missing children")
                return id(self)  # Fallback to current node ID


def _build_custom_tree(all_trial_duals, feature_vectors, selected_scenarios, unselected_scenarios, 
                      iterations, dual_var, dual_index, max_depth, min_samples_leaf, current_depth=0):
    """
    Build a custom decision tree using optimization-based splitting criterion for dual prediction.
    
    This function implements the core tree construction algorithm that directly optimizes
    the mathematical objective specified for tree-based dual variable prediction in 
    ML-augmented Benders decomposition.
    
    **Tree Construction Algorithm**:
    1. **Objective-Based Splitting**: At each node, evaluate all possible feature splits
    2. **Error Evaluation**: For each split, compute prediction error using uniform averaging
    3. **Best Split Selection**: Choose split that minimally reduces total prediction error
    4. **Recursive Construction**: Apply same process to left and right child nodes
    5. **Stopping Criteria**: Stop when depth limit, minimum samples, or no improvement reached
    
    **Mathematical Splitting Criterion**:
    For a potential split at feature[i] <= threshold:
    
    Left Child Error = Σ_{m'∈L_left∩U, k} ||λ_m'^(k) - (1/|J_left|) * Σ_{j∈J_left} λ_j^(k)||_2^2
    Right Child Error = Σ_{m'∈L_right∩U, k} ||λ_m'^(k) - (1/|J_right|) * Σ_{j∈J_right} λ_j^(k)||_2^2
    
    Where:
    - L_left/L_right = scenarios going to left/right child
    - U = unselected scenarios (for error evaluation)
    - J_left/J_right = selected scenarios in left/right child (for averaging)
    - Best split minimizes: Left Child Error + Right Child Error
    
    **Stopping Criteria**:
    - Maximum depth reached (max_depth)
    - Insufficient samples for further splitting (< 2 * min_samples_leaf)
    - No unselected scenarios for error evaluation
    - No selected scenarios for averaging
    - No split found that improves prediction error
    
    Args:
    ----
    all_trial_duals (dict): Complete trial dual solutions from Phase 1
                           Format: {(scenario_id, iteration): structured_dual_dict}
    feature_vectors (dict): Feature representations for all scenarios 
                           Format: {scenario_id: np.array}
    selected_scenarios (set): Scenario IDs that will be solved exactly in Phase 2
                             Used for computing uniform averages in leaves
    unselected_scenarios (set): Scenario IDs requiring prediction in Phase 2  
                               Used for evaluating prediction error
    iterations (list): Trial iteration numbers from Phase 1 (e.g., [1, 2, 3, 4, 5])
    dual_var (str): Name of dual variable being optimized ('lambda', 'mu', 'v', 'u')
    dual_index: Index for dual variable access:
               - None for scalar variables (e.g., lambda)  
               - int for single-indexed variables (e.g., mu[j])
               - tuple for multi-indexed variables (e.g., gamma[(i,j)])
    max_depth (int): Maximum allowed tree depth (None = unlimited)
    min_samples_leaf (int): Minimum scenarios required in each leaf node
    current_depth (int): Current depth in recursive construction (starts at 0)
        
    Returns:
    -------
    CustomTreeNode: Root node of the constructed subtree
                   - If leaf: Contains selected/unselected scenarios for fixed mapping
                   - If internal: Contains feature_index, threshold, left/right children
                   
    **Tree Structure Output**:
    - **Leaf nodes**: is_leaf=True, contains selected_scenarios and unselected_scenarios
    - **Internal nodes**: is_leaf=False, contains feature_index, threshold, left/right children
    - **Fixed mappings**: Each unselected scenario maps to selected scenarios in same leaf
    - **Uniform weights**: All selected scenarios in leaf get equal weight 1/|selected_scenarios|
    
    **Example Leaf Structure**:
    ```
    leaf_node.selected_scenarios = {1, 2, 5}      # Selected scenarios in this leaf
    leaf_node.unselected_scenarios = {12, 13}     # Unselected scenarios in this leaf
    # Creates fixed mappings:
    # 12 → {1, 2, 5} with weights [1/3, 1/3, 1/3]  
    # 13 → {1, 2, 5} with weights [1/3, 1/3, 1/3]
    ```
    
    Performance Characteristics:
    ---------------------------
    - **Time Complexity**: O(n_scenarios * n_features * n_iterations * max_depth)
    - **Space Complexity**: O(max_depth + n_scenarios)
    - **Memory Usage**: Stores scenario sets at each node, feature splits
    - **Optimization**: Direct dual prediction error minimization (not classification accuracy)
    """
    
    # Create root node
    node = CustomTreeNode()
    all_scenarios_in_node = selected_scenarios | unselected_scenarios
    node.scenarios = all_scenarios_in_node.copy()
    node.selected_scenarios = selected_scenarios.copy()
    node.unselected_scenarios = unselected_scenarios.copy()
    
    # Stopping criteria
    if (max_depth is not None and current_depth >= max_depth) or \
       len(all_scenarios_in_node) < 2 * min_samples_leaf or \
       len(unselected_scenarios) == 0 or \
       len(selected_scenarios) == 0:
        # Make this a leaf node
        # logger.debug(f"Creating leaf node at depth {current_depth} with {len(selected_scenarios)} selected and {len(unselected_scenarios)} unselected scenarios")
        return node
    
    # Find the best split
    best_split = _find_best_split(all_trial_duals, feature_vectors, selected_scenarios, unselected_scenarios,
                                 iterations, dual_var, dual_index, min_samples_leaf)
    
    if best_split is None:
        # No valid split found, make this a leaf
        # logger.debug(f"No valid split found at depth {current_depth}, creating leaf")
        return node
    
    feature_idx, threshold, left_selected, left_unselected, right_selected, right_unselected = best_split
    
    # Create internal node
    node.is_leaf = False
    node.feature_index = feature_idx
    node.threshold = threshold
    
    # logger.debug(f"Creating split at depth {current_depth}: feature[{feature_idx}] <= {threshold:.4f}")
    # logger.debug(f"  Left: {len(left_selected)} selected, {len(left_unselected)} unselected")
    # logger.debug(f"  Right: {len(right_selected)} selected, {len(right_unselected)} unselected")
    
    # Recursively build left and right subtrees
    node.left = _build_custom_tree(all_trial_duals, feature_vectors, left_selected, left_unselected,
                                  iterations, dual_var, dual_index, max_depth, min_samples_leaf, current_depth + 1)
    
    node.right = _build_custom_tree(all_trial_duals, feature_vectors, right_selected, right_unselected, 
                                   iterations, dual_var, dual_index, max_depth, min_samples_leaf, current_depth + 1)
    
    return node


def _find_best_split(all_trial_duals, feature_vectors, selected_scenarios, unselected_scenarios,
                    iterations, dual_var, dual_index, min_samples_leaf):
    """
    Find the optimal binary split for the current tree node using the custom dual prediction objective.
    
    This function implements the core optimization logic that evaluates all possible feature splits
    to minimize prediction error for unselected scenarios using uniform averaging over selected scenarios.
    
    **Mathematical Objective**:
    For a split at feature[i] <= threshold, minimize:
    Total Error = Left Error + Right Error
    
    Where:
    Left Error = Σ_{m'∈L_left∩U, k=1...K} ||λ_m'^(k) - (1/|J_left|) * Σ_{j∈J_left} λ_j^(k)||_2^2
    Right Error = Σ_{m'∈L_right∩U, k=1...K} ||λ_m'^(k) - (1/|J_right|) * Σ_{j∈J_right} λ_j^(k)||_2^2
    
    **Terms**:
    - L_left/L_right = scenarios in left/right child after split
    - U = unselected scenarios (for error evaluation)  
    - J_left/J_right = selected scenarios in left/right child (for averaging)
    - λ_m'^(k) = true dual value for unselected scenario m' at iteration k
    - λ_j^(k) = true dual value for selected scenario j at iteration k
    
    **Split Evaluation Algorithm**:
    1. **Feature Iteration**: Try all features as potential split variables
    2. **Threshold Generation**: Use midpoints between consecutive feature values as thresholds
    3. **Partition Evaluation**: For each threshold, partition scenarios into left (<=) and right (>) children
    4. **Error Computation**: Calculate prediction error for unselected scenarios in each child
    5. **Best Selection**: Choose feature and threshold that minimizes total error
    
    **Feasibility Constraints**:
    - Each child must have at least min_samples_leaf scenarios
    - Each child must have at least one selected scenario (for averaging)
    - Each child should have at least one unselected scenario (for error evaluation)
    
    Args:
    ----
    all_trial_duals (dict): Complete Phase 1 dual solutions
                           Format: {(scenario_id, iteration): structured_dual_dict}
    feature_vectors (dict): Feature representations for all scenarios
                           Format: {scenario_id: np.array}
    selected_scenarios (set): Selected scenario IDs in current node (for uniform averaging)
    unselected_scenarios (set): Unselected scenario IDs in current node (for error evaluation)
    iterations (list): Phase 1 trial iteration numbers [1, 2, 3, 4, 5]
    dual_var (str): Name of dual variable being optimized ('lambda', 'mu', 'v', 'u')
    dual_index: Index for accessing dual variable values:
               - None: scalar variable (e.g., λ)
               - int: single index (e.g., μ[j])  
               - tuple: multiple indices (e.g., γ[(i,j)])
    min_samples_leaf (int): Minimum number of scenarios required in each child node
        
    Returns:
    -------
    tuple or None: Best split information if found, None otherwise
                  
    **Return Tuple Structure** (when split found):
    ```
    (feature_idx,        # int: Index of feature to split on (0 to n_features-1)
     threshold,          # float: Split threshold value  
     left_selected,      # set: Selected scenarios going to left child
     left_unselected,    # set: Unselected scenarios going to left child
     right_selected,     # set: Selected scenarios going to right child
     right_unselected)   # set: Unselected scenarios going to right child
    ```
    
    **Return None** when:
    - No unselected scenarios available for error evaluation
    - No feasible splits satisfy minimum sample constraints
    - All splits result in empty selected/unselected sets in children
    - No improvement over current node possible
    
    **Algorithm Complexity**:
    - **Time**: O(n_features * n_scenarios * log(n_scenarios) * n_iterations)
    - **Space**: O(n_scenarios) for temporary partitions
    - **Bottleneck**: Error computation for each threshold candidate
    
    **Implementation Details**:
    - **Threshold Selection**: Uses midpoints between sorted feature values to avoid bias
    - **Tie Breaking**: When errors are equal, prefers splits that better balance child sizes
    - **Numerical Stability**: Handles edge cases with zero scenarios or identical feature values
    - **Early Termination**: Skips further evaluation when minimum constraints violated
    
    Example:
    -------
    ```python
    # Evaluate split: feature[2] <= 0.5
    # Left child: scenarios with feature[2] <= 0.5  
    # Right child: scenarios with feature[2] > 0.5
    # Compute prediction error for unselected scenarios in each child
    # Return split with minimal total error
    ```
    """
    
    if len(unselected_scenarios) == 0:
        return None
        
    all_scenarios = selected_scenarios | unselected_scenarios
    scenario_list = list(all_scenarios)
    
    if not scenario_list or len(scenario_list) < 2 * min_samples_leaf:
        return None
    
    # Get feature matrix for scenarios in this node
    feature_matrix = np.array([feature_vectors[sid] for sid in scenario_list])
    n_features = feature_matrix.shape[1]
    
    best_error = float('inf')
    best_split = None
    
    # Try each feature
    for feature_idx in range(n_features):
        feature_values = feature_matrix[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        # Try thresholds between unique values
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2.0
            
            # Split scenarios based on threshold
            left_scenarios = set()
            right_scenarios = set()
            
            for idx, sid in enumerate(scenario_list):
                if feature_values[idx] <= threshold:
                    left_scenarios.add(sid)
                else:
                    right_scenarios.add(sid)
            
            # Separate selected and unselected in each child
            left_selected = left_scenarios & selected_scenarios
            left_unselected = left_scenarios & unselected_scenarios
            right_selected = right_scenarios & selected_scenarios  
            right_unselected = right_scenarios & unselected_scenarios
            
            # Check minimum leaf size constraint
            if len(left_scenarios) < min_samples_leaf or len(right_scenarios) < min_samples_leaf:
                continue
                
            # Ensure each child has at least one selected scenario for averaging
            if len(left_selected) == 0 or len(right_selected) == 0:
                continue
            
            # Compute prediction error for this split
            total_error = 0.0
            
            # Error for left child
            if len(left_unselected) > 0:
                left_error = _compute_prediction_error(all_trial_duals, left_selected, left_unselected,
                                                     iterations, dual_var, dual_index)
                total_error += left_error
            
            # Error for right child  
            if len(right_unselected) > 0:
                right_error = _compute_prediction_error(all_trial_duals, right_selected, right_unselected,
                                                      iterations, dual_var, dual_index)
                total_error += right_error
            
            # Update best split if this is better
            if total_error < best_error:
                best_error = total_error
                best_split = (feature_idx, threshold, left_selected, left_unselected, right_selected, right_unselected)
    
    return best_split


def _compute_prediction_error(all_trial_duals, selected_scenarios, unselected_scenarios, 
                            iterations, dual_var, dual_index):
    """
    Compute the prediction error for unselected scenarios using uniform averaging over selected scenarios.
    
    This implements the core error computation:
    Σ_{m'∈unselected, k∈iterations} ||λ_m'^(k) - λ̂_m'^(k)||_2^2
    
    Where λ̂_m'^(k) = (1/|selected|) * Σ_{j∈selected} λ_j^(k)
    
    Args:
        all_trial_duals (dict): All trial dual solutions
        selected_scenarios (set): Selected scenarios for averaging
        unselected_scenarios (set): Unselected scenarios for error evaluation
        iterations (list): Trial iterations
        dual_var (str): Dual variable name
        dual_index: Dual variable index
        
    Returns:
        float: Total squared prediction error
    """
    
    if not selected_scenarios or not unselected_scenarios:
        return 0.0
    
    total_error = 0.0
    
    for iteration in iterations:
        # Compute uniform average over selected scenarios for this iteration
        selected_values = []
        for sid in selected_scenarios:
            if (sid, iteration) in all_trial_duals:
                dual_dict = all_trial_duals[(sid, iteration)]
                if dual_index is None:
                    # Scalar dual variable
                    selected_values.append(dual_dict[dual_var])
                else:
                    # Indexed dual variable
                    selected_values.append(dual_dict[dual_var][dual_index])
        
        if not selected_values:
            continue
            
        uniform_average = np.mean(selected_values)
        
        # Compute squared error for each unselected scenario
        for sid in unselected_scenarios:
            if (sid, iteration) in all_trial_duals:
                dual_dict = all_trial_duals[(sid, iteration)]
                if dual_index is None:
                    true_value = dual_dict[dual_var]
                else:
                    true_value = dual_dict[dual_var][dual_index]
                
                error = (true_value - uniform_average) ** 2
                total_error += error
    
    return total_error


# =============================================================================
# UNIFIED DUAL PREDICTOR CLASS
# =============================================================================

class DualPredictor:
    """
    Unified dual prediction class that encapsulates all three prediction methods (REG, KNN, TREE) 
    for ML-augmented Benders decomposition.
    
    This class provides a consistent interface for training and predicting dual variables across 
    all supported prediction methods, centralizing all prediction-related functionality that was 
    previously scattered across callbacks.py and prediction.py.
    
    **Supported Methods**:
    - **REG**: Linear regression with dual feasibility constraints
    - **KNN**: K-nearest neighbors with fixed mappings and uniform averaging  
    - **TREE**: Decision tree with fixed mappings and uniform averaging
    - **DUAL_CLUSTERING**: K-means clustering with cluster representatives
    - **SIMPLE_AVG**: Simple averaging of all selected scenarios
    
    **Key Benefits**:
    - **Unified Interface**: Single class handles all prediction methods
    - **Centralized Logic**: All prediction code in one place (prediction.py)  
    - **Consistent API**: Same train() and predict() interface for all methods
    - **Clean Architecture**: Callbacks only handle problem-specific logic
    
    Usage Example:
    -------------
    ```python
    # Initialize predictor
    predictor = DualPredictor(
        method="TREE", 
        problem_type="UFL", 
        data=problem_data,
        max_depth=5,
        n_neighbors=3
    )
    
    # Train the predictor
    predictor.train(
        all_trial_duals=trial_data,
        feature_vectors=features,
        selected_subproblems=selected,
        first_stage_val=x_sol  # Only for REG
    )
    
    # Make predictions
    predictions = predictor.predict(unselected_subs, dual_cache)
    ```
    """
    
    def __init__(self, ml_method_type, predictor_type, problem_type, data, n_neighbors=5, max_depth=5, 
                 min_samples_leaf=2, L_d=1e5, L_v=1e5, p_d=1.0, p_v=1.0,
                 # Clustering specific parameters
                 clustering_random_state=42,
                 clustering_max_iter=300,
                 clustering_tol=1e-6,
                 clustering_init_method='k-means++',
                 # Random sampling specific parameters
                 random_sampling_random_state=42,
                 # Capacity scenarios configuration
                 use_stochastic_capacity=True,
                 # Parametric predictor configuration
                 parametric_config=None):
        """
        Initialize the unified dual predictor.
        
        Args:
            ml_method_type (str): ML method type from new architecture:
                'ML_Augmented_BD', 'ML_Augmented_BD_with_duals', 
                'Covariate_Similarity', 'Uncertainty_Similarity', 'Random_Sampling'
            predictor_type (str): Predictor type: 'KNN', 'TREE', 'CLUSTERING', or None for Random_Sampling
            problem_type (str): Problem type - "UFL", "UHL", "SSLP", "SCFLP"
            data: Problem data object with sets and parameters
            n_neighbors (int): Number of neighbors for KNN predictor
            max_depth (int): Maximum depth for TREE predictor
            min_samples_leaf (int): Minimum samples per leaf for TREE predictor
            L_d (float): L1 dual error budget for REG method (future)
            L_v (float): Value discrepancy budget for REG method (future)
            p_d (float): Dual slack objective weight for REG method (future)
            p_v (float): Value slack objective weight for REG method (future)
            clustering_random_state (int): Random seed for CLUSTERING predictor
            clustering_max_iter (int): Max iterations for CLUSTERING predictor
            clustering_tol (float): Convergence tolerance for CLUSTERING predictor
            clustering_init_method (str): Initialization method for CLUSTERING predictor
            random_sampling_random_state (int): Random seed for Random_Sampling
            use_stochastic_capacity (bool): Whether to use stochastic capacity scenarios
        """
        # Validate ML method type
        valid_ml_methods = {
            "ML_Augmented_BD", "ML_Augmented_BD_with_duals", 
            "Covariate_Similarity", "Uncertainty_Similarity", "Random_Sampling"
        }
        if ml_method_type not in valid_ml_methods:
            raise ValueError(f"ml_method_type must be one of {valid_ml_methods}, got {ml_method_type}")
        
        # Validate predictor type
        valid_predictors = {"KNN", "TREE", "CLUSTERING", PMEDIAN_WEIGHTED_TRANSFER, PMEDIAN_NEAREST_TRANSFER, None}.union(PARAMETRIC_PREDICTORS)
        if predictor_type not in valid_predictors:
            raise ValueError(f"predictor_type must be one of {valid_predictors}, got {predictor_type}")
        
        # Validate predictor type matches ML method type
        if ml_method_type == "Random_Sampling" and predictor_type is not None:
            raise ValueError("Random_Sampling should not have a predictor_type")
        elif ml_method_type != "Random_Sampling" and predictor_type is None:
            raise ValueError(f"{ml_method_type} requires a predictor_type")
        if is_parametric_predictor(predictor_type) and ml_method_type != "ML_Augmented_BD":
            raise ValueError(
                "Parametric predictors are variants inside ML_Augmented_BD and cannot "
                f"be used with {ml_method_type}."
            )
            
        self.ml_method_type = ml_method_type
        self.predictor_type = predictor_type
        self.problem_type = problem_type.upper()
        self.data = data
        
        # Predictor-specific parameters
        # KNN parameters
        self.n_neighbors = n_neighbors
        self.optimal_k = n_neighbors     # Store optimal k value (updated during training if find_optimal_k=True)
        
        # TREE parameters
        self.max_depth = max_depth  
        self.min_samples_leaf = min_samples_leaf
        
        # CLUSTERING parameters (unified across all clustering methods)
        self.clustering_random_state = clustering_random_state
        self.clustering_max_iter = clustering_max_iter
        self.clustering_tol = clustering_tol
        self.clustering_init_method = clustering_init_method
        
        # Random sampling parameters
        self.random_sampling_random_state = random_sampling_random_state
        
        # Future: REG method parameters
        self.L_d = L_d
        self.L_v = L_v
        self.p_d = p_d
        self.p_v = p_v
        
        # Capacity scenarios configuration
        self.use_stochastic_capacity = use_stochastic_capacity
        self.parametric_config = dict(parametric_config or {})
        self.pmedian_counts = {}
        self.parametric_predictor = None
        if is_parametric_predictor(self.predictor_type):
            self.parametric_predictor = ParametricDualPredictor(
                variant=self.predictor_type,
                problem_type=self.problem_type,
                data=self.data,
                parameters=self.parametric_config,
                use_stochastic_capacity=self.use_stochastic_capacity,
            )
        
        # Trained models storage (unified by predictor type)
        self.trained_weights = None      # For future REG method
        self.predictor_mappings = None   # Unified storage for all predictor mappings (KNN/TREE/CLUSTERING)
        self.feature_vectors = None      # Stored feature vectors for prediction
        self.selected_subproblems = None # Stored for neighbor computation
        
        # Unified training time tracking
        self.training_time = None        # Time spent on training/defining optimization problem
        
        self.is_trained = False
        
        logger.info(f"Initialized DualPredictor with ml_method_type={self.ml_method_type}, "
                   f"predictor_type={self.predictor_type}, problem_type={self.problem_type}")
    
    def train(self, all_trial_duals=None, feature_vectors=None, selected_subproblems=None, 
              first_stage_val=None, sample_weights=None, metadata=None, p=None, knn_params=None):
        """
        Train the predictor using the specified method.
        
        Args:
            all_trial_duals (dict): Trial dual solutions {(sid, iteration): dual_dict}
            feature_vectors (dict): Feature vectors {sid: np.array}  
            selected_subproblems (set): Selected scenario IDs (may be None for unified approach)
            first_stage_val (dict): First-stage solution (required for future REG)
            sample_weights (dict): Training sample weights (for future REG)
            metadata (dict): Training metadata (for REG logging)
            p (int): Number of scenarios to select (used for unified approach)
            knn_params (dict): KNN parameters (used for KNN predictor)
            
        Returns:
            dict: Training results (method-specific format)
        """
        if feature_vectors is None and self.ml_method_type != 'Random_Sampling':
            raise ValueError("feature_vectors required for all prediction methods except Random_Sampling")
            
        # Store feature vectors for prediction phase
        self.feature_vectors = feature_vectors
        
        # For unified approach, p determines selection size (not pre-selected scenarios)
        if selected_subproblems is not None:
            logger.info(f"Training {self.ml_method_type}/{self.predictor_type} predictor "
                       f"(passed {len(selected_subproblems)} selected scenarios, using p={p})")
        else:
            logger.info(f"Training {self.ml_method_type}/{self.predictor_type} predictor "
                       f"(unified approach with p={p})")
        
        try:
            # Start method-specific training time tracking (only measure actual training operations)
            training_start_time = time.time()
            
            # Route based on ML method type and predictor type
            if is_parametric_predictor(self.predictor_type):
                if p is None:
                    p = len(selected_subproblems) if selected_subproblems else 5
                parametric_params = {
                    'all_trial_duals': all_trial_duals,
                    'first_stage_by_iteration': (metadata or {}).get(
                        'first_stage_by_iteration', {}
                    ),
                }
                result = self._train_unified(feature_vectors, p, parametric_params)

            elif self.predictor_type in {PMEDIAN_WEIGHTED_TRANSFER, PMEDIAN_NEAREST_TRANSFER}:
                if p is None:
                    p = len(selected_subproblems) if selected_subproblems else 5
                result = self._train_unified(feature_vectors, p, self.parametric_config)

            elif self.predictor_type == 'KNN':
                # KNN uses unified approach
                if p is None:
                    p = len(selected_subproblems) if selected_subproblems else 5
                if knn_params is None:
                    knn_params = {'k': self.n_neighbors, 'find_optimal_k': False}
                result = self._train_unified(feature_vectors, p, knn_params)
                
            elif self.predictor_type == 'TREE':
                # TREE uses unified approach
                if p is None:
                    p = len(selected_subproblems) if selected_subproblems else 5
                result = self._train_unified(feature_vectors, p)
                
            elif self.predictor_type == 'CLUSTERING':
                # CLUSTERING uses unified approach
                if p is None:
                    p = len(selected_subproblems) if selected_subproblems else 5
                
                # Build method_params based on ML method type
                clustering_params = {
                    'random_state': self.clustering_random_state,
                    'init_method': self.clustering_init_method,
                    'max_iter': self.clustering_max_iter,
                    'tol': self.clustering_tol
                }
                
                # Add data source for ML_Augmented_BD_with_duals
                if self.ml_method_type == 'ML_Augmented_BD_with_duals':
                    if all_trial_duals is None:
                        raise ValueError("ML_Augmented_BD_with_duals requires all_trial_duals parameter")
                    clustering_params['all_trial_duals'] = all_trial_duals
                
                result = self._train_unified(feature_vectors, p, clustering_params)
                
            elif self.ml_method_type == 'Random_Sampling':
                # Random sampling - no training needed
                if p is None:
                    p = len(selected_subproblems) if selected_subproblems else 5
                random_params = {
                    'random_state': self.random_sampling_random_state
                }
                result = self._train_unified(feature_vectors, p, random_params)
                
            else:
                raise ValueError(f"Unknown predictor type: {self.predictor_type}")
            
            # Record unified training time (only actual training operations, not setup)
            self.training_time = time.time() - training_start_time
            
            logger.info(f"{self.ml_method_type}/{self.predictor_type} predictor training completed in {self.training_time:.2f} seconds")
            logger.info(f"Selected {len(self.selected_subproblems)} scenarios from training")
            
            return result
            
        except Exception as e:
            # Record training time even on failure
            self.training_time = time.time() - training_start_time
            logger.error(f"{self.ml_method_type}/{self.predictor_type} predictor training failed after {self.training_time:.2f} seconds: {e}")
            raise
    
    def predict(self, unselected_subs, dual_cache):
        """
        Predict dual variables for unselected subproblems.
        
        Args:
            unselected_subs (list/set): Unselected subproblem indices
            dual_cache (dict): Current dual solutions {selected_sid: dual_dict}
            
        Returns:
            dict: Predicted dual variables {unselected_sid: dual_dict}
        """
        if not self.is_trained:
            raise ValueError("Predictor must be trained before making predictions")
            
        logger.debug(f"Predicting duals for {len(unselected_subs)} unselected scenarios using "
                    f"{self.ml_method_type}/{self.predictor_type}")
        # All debug message commented out - 
        # logger.debug(f"Predicting duals for {len(unselected_subs)} unselected scenarios using "
                    # f"{self.ml_method_type}/{self.predictor_type}")
        
        # Route based on predictor type
        if is_parametric_predictor(self.predictor_type):
            return self.parametric_predictor.predict(unselected_subs)
        if self.predictor_type == PMEDIAN_WEIGHTED_TRANSFER:
            return self._predict_pmedian_weighted(unselected_subs, dual_cache)
        if self.predictor_type == PMEDIAN_NEAREST_TRANSFER:
            return self._predict_pmedian_nearest(unselected_subs, dual_cache)
        if self.predictor_type == 'KNN':
            return self._predict_knn(unselected_subs, dual_cache)
        elif self.predictor_type == 'TREE':
            return self._predict_tree(unselected_subs, dual_cache)
        elif self.predictor_type == 'CLUSTERING':
            return self._predict_clustering(unselected_subs, dual_cache)
        elif self.ml_method_type == 'Random_Sampling':
            return self._predict_bertsimas_random_sampling(unselected_subs, dual_cache)
        else:
            raise ValueError(f"Unknown predictor type: {self.predictor_type}")
    
    def _train_regression(self, all_trial_duals, feature_vectors, selected_subproblems,
                         first_stage_val, sample_weights, metadata):
        """REG training method - implementation removed, kept as stub."""
        logger.warning("REG method implementation removed - using placeholder")
        self.is_trained = False
        return None
    
    def _train_unified(self, feature_vectors, p, method_params=None):
        """
        Unified training: Performs both subproblem selection and mapping based on prediction method.
        
        Each prediction method includes its own optimal selection strategy.
        
        Args:
            feature_vectors (dict): All scenario feature vectors
            p (int): Number of scenarios to select
            method_params (dict): Method-specific parameters
            
        Returns:
            dict: Training result with selected scenarios and mappings
        """
        import time

        training_start_time = time.time()
        method_params = dict(method_params or {})

        # Parametric variants select representatives according to their configured
        # anchor mechanism.  RHS-Ridge variants fit their residual model once on
        # the complete warm-up panel.
        if is_parametric_predictor(self.predictor_type):
            selected_scenarios, method_mapping = self.parametric_predictor.select(
                feature_vectors, p
            )
            if self.predictor_type in RHS_RIDGE_PREDICTORS:
                self.parametric_predictor.fit_warmup_rhs_model(
                    method_params.get('all_trial_duals', {}),
                    method_params.get('first_stage_by_iteration', {}),
                )
            self.selected_subproblems = set(selected_scenarios)
            self.predictor_mappings = method_mapping
            self.feature_vectors = feature_vectors
            self.is_trained = True
            self.training_time = time.time() - training_start_time
            return {
                "__selected_scenarios__": selected_scenarios,
                "__method_mapping__": method_mapping,
                "__method__": f"ML_Augmented_BD_{self.predictor_type}",
            }

        if self.predictor_type in {PMEDIAN_WEIGHTED_TRANSFER, PMEDIAN_NEAREST_TRANSFER}:
            selection_params = dict(self.parametric_config)
            selection_params.update(method_params)
            selected_scenarios, method_mapping = solve_parametric_pmedian_selection(
                feature_vectors, p, selection_params
            )
            self.pmedian_counts = representative_counts(
                selected_scenarios, method_mapping
            )
            self.selected_subproblems = set(selected_scenarios)
            self.predictor_mappings = method_mapping
            self.feature_vectors = feature_vectors
            self.is_trained = True
            self.training_time = time.time() - training_start_time
            return {
                "__selected_scenarios__": selected_scenarios,
                "__method_mapping__": method_mapping,
                "__method__": f"ML_Augmented_BD_{self.predictor_type}",
            }

        from Dual_Prediction.training_and_selection import integrated_prediction_selection
        
        # Build default method_params based on predictor type if not provided
        if not method_params:
            if self.predictor_type == 'KNN':
                method_params = {'k': self.n_neighbors, 'find_optimal_k': False}
            elif self.predictor_type == 'TREE':
                method_params = {
                    'D_max': self.max_depth,
                    'n_min': self.min_samples_leaf,
                    'min_impurity_decrease': 0.0,
                    'quantile_grid_size': 10
                }
            elif self.predictor_type == 'CLUSTERING':
                method_params = {
                    'random_state': self.clustering_random_state,
                    'init_method': self.clustering_init_method,
                    'max_iter': self.clustering_max_iter,
                    'tol': self.clustering_tol
                }
            elif self.ml_method_type == 'Random_Sampling':
                method_params = {
                    'random_state': self.random_sampling_random_state
                }
            else:
                method_params = {}
            
        logger.info(f"Starting unified {self.ml_method_type}/{self.predictor_type} selection+mapping with p={p}")
        
        # Add method-specific data vectors to method_params based on ML method type
        if self.ml_method_type == 'ML_Augmented_BD_with_duals':
            # ML_Augmented_BD_with_duals uses all_trial_duals (should have been passed in method_params)
            # Ensure all_trial_duals is in method_params (it should have been added by train() method)
            if 'all_trial_duals' not in method_params:
                raise ValueError("ML_Augmented_BD_with_duals requires 'all_trial_duals' in method_params")
            logger.info(f"ML_Augmented_BD_with_duals using {len(method_params['all_trial_duals'])} dual vectors")
            
        elif self.ml_method_type == 'Uncertainty_Similarity':
            # Preferred source: caller-provided vectors (supports all problem classes).
            # Fallback source: derive from SCFLP data for backward compatibility.
            if method_params.get('uncertainty_vectors') is not None:
                logger.info(
                    "Uncertainty_Similarity using caller-provided uncertainty vectors (%d scenarios)",
                    len(method_params['uncertainty_vectors'])
                )
            elif self.problem_type == 'SCFLP' and self.data is not None:
                uncertainty_vectors = {}
                
                # Check if we should use combined demand-capacity uncertainty vectors
                use_combined_uncertainty = self.use_stochastic_capacity and \
                                         hasattr(self.data, 'capacity_scenarios') and \
                                         self.data.capacity_scenarios is not None
                
                for s in range(len(self.data.scenarios)):
                    demand_vector = self.data.scenarios[s]  # Demand vector for scenario s
                    
                    if use_combined_uncertainty and s < len(self.data.capacity_scenarios):
                        # Combine demand and capacity vectors for enhanced uncertainty representation
                        capacity_vector = self.data.capacity_scenarios[s]  # Capacity vector for scenario s
                        # Concatenate demand and capacity vectors
                        combined_vector = np.concatenate([demand_vector, capacity_vector])
                        uncertainty_vectors[s] = combined_vector
                    else:
                        # Use only demand vector (original behavior)
                        uncertainty_vectors[s] = demand_vector
                        if use_combined_uncertainty:
                            logger.warning(f"Capacity scenarios not available for scenario {s}, using demand-only uncertainty")
                
                if use_combined_uncertainty:
                    logger.info(f"Uncertainty_Similarity using combined demand-capacity uncertainty vectors")
                else:
                    logger.info(f"Uncertainty_Similarity using demand-only uncertainty vectors")
                    
                method_params['uncertainty_vectors'] = uncertainty_vectors
            else:
                raise ValueError(
                    "Uncertainty_Similarity requires 'uncertainty_vectors' in method_params "
                    "or SCFLP data with scenarios"
                )
                
        elif self.ml_method_type == 'Covariate_Similarity':
            # Preferred source: caller-provided vectors (supports all problem classes).
            # Fallback source: derive from SCFLP data for backward compatibility.
            if method_params.get('covariate_vectors') is not None:
                logger.info(
                    "Covariate_Similarity using caller-provided covariate vectors (%d scenarios)",
                    len(method_params['covariate_vectors'])
                )
            elif self.problem_type == 'SCFLP' and self.data is not None:
                if self.data.covariate_vectors is not None:
                    covariate_vectors = {}
                    for s in range(len(self.data.covariate_vectors)):
                        covariate_vectors[s] = self.data.covariate_vectors[s]  # Covariate vector for scenario s
                    method_params['covariate_vectors'] = covariate_vectors
                else:
                    raise ValueError("Covariate_Similarity requires SCFLP data with covariate_vectors")
            else:
                raise ValueError(
                    "Covariate_Similarity requires 'covariate_vectors' in method_params "
                    "or SCFLP data with covariate vectors"
                )
        
        # Perform integrated selection and mapping using new architecture
        selected_scenarios, method_mapping = integrated_prediction_selection(
            feature_vectors, self.ml_method_type, self.predictor_type, p, method_params
        )
        
        # Store results in unified storage
        if selected_scenarios is not None:
            self.selected_subproblems = set(selected_scenarios)
        else:
            self.selected_subproblems = set()
            
        # Store predictor mapping in unified storage (single variable for all predictors)
        self.predictor_mappings = method_mapping
        
        # Extract optimal k for KNN predictor if available
        if self.predictor_type == 'KNN':
            self.optimal_k = method_mapping.get('__optimal_k__', self.n_neighbors)
            
        self.feature_vectors = feature_vectors
        self.is_trained = True
        
        # Set training time
        self.training_time = time.time() - training_start_time
        
        # BUG FIX (Feb 22, 2026): Count only actual scenario mappings (exclude metadata keys starting with '__')
        num_mapped_scenarios = sum(1 for key in method_mapping.keys() if not str(key).startswith('__'))
        
        logger.info(f"Unified {self.ml_method_type}/{self.predictor_type} completed - "
                   f"selected {len(selected_scenarios) if selected_scenarios else 0} scenarios, "
                   f"mapped {num_mapped_scenarios} unselected scenarios")
        
        return {
            "__selected_scenarios__": selected_scenarios,
            "__method_mapping__": method_mapping, 
            "__method__": f"{self.ml_method_type}_{self.predictor_type}_UNIFIED"
        }

    def prepare_parametric_iteration(
        self, solved_duals, first_stage_values, predicted_ids, iteration=None
    ):
        """Fit/update a parametric variant and prepare certified predictions."""
        if self.parametric_predictor is None:
            raise ValueError("The configured predictor is not parametric.")
        return self.parametric_predictor.prepare_iteration(
            solved_duals=solved_duals,
            first_stage_values=first_stage_values,
            predicted_ids=predicted_ids,
            iteration=iteration,
        )

    def validate_parametric_dual(self, duals):
        """Strict final certification used immediately before cut generation."""
        if self.parametric_predictor is None:
            raise ValueError("The configured predictor is not parametric.")
        return self.parametric_predictor.validate_structured_dual(duals)

    def get_parametric_diagnostics(self):
        if self.parametric_predictor is None:
            return {}
        return self.parametric_predictor.diagnostics()

    def record_parametric_exact_fallback(self, count):
        if self.parametric_predictor is not None:
            self.parametric_predictor.record_exact_fallback(count)

    def record_parametric_selected_exact_solves(self, count):
        if self.parametric_predictor is not None:
            self.parametric_predictor.record_selected_exact_solves(count)

    def record_parametric_cut_evaluation(self, violation, added):
        if self.parametric_predictor is not None:
            self.parametric_predictor.record_cut_evaluation(violation, added)
    
    def _train_tree(self, all_trial_duals, feature_vectors, selected_subproblems):
        """TREE training method - implementation removed, kept as stub."""
        logger.warning("TREE method implementation removed - using placeholder")
        self.is_trained = False
        return None
    
    def _train_simple_avg(self, feature_vectors, selected_subproblems):
        """SIMPLE_AVG training method - implementation removed, kept as stub."""
        logger.warning("SIMPLE_AVG method implementation removed - using placeholder")
        self.is_trained = False
        return None
    
    def _predict_regression(self, unselected_subs):
        """REG prediction method - implementation removed, kept as stub."""
        logger.warning("REG prediction method not implemented")
        return {}
    
    def _predict_pmedian_weighted(self, unselected_subs, dual_cache):
        """Use the p-median count-weighted common feasible anchor directly.

        This is the nonparametric counterpart of the existing RHS-Ridge anchor:
        the selected set and assignment counts are identical, but no residual
        regression correction is added.
        """
        if not dual_cache:
            return {}

        ids = sorted(int(sid) for sid in dual_cache)
        raw_weights = np.asarray(
            [max(0.0, float(self.pmedian_counts.get(sid, 0.0))) for sid in ids],
            dtype=float,
        )
        if raw_weights.sum() <= 0.0:
            raw_weights[:] = 1.0
        raw_weights /= raw_weights.sum()

        duals = [dual_cache[sid] for sid in ids]
        common_anchor = self._weighted_average_structured_duals(duals, raw_weights)
        predictions = {}
        for sid in unselected_subs:
            predictions[int(sid)] = {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in common_anchor.items()
            }
        return predictions

    def _predict_pmedian_nearest(self, unselected_subs, dual_cache):
        """Copy the exact dual of each scenario's nearest selected p-median representative."""
        if self.predictor_mappings is None:
            raise ValueError("P-median nearest-representative mapping is unavailable.")
        predictions = {}
        for sid in unselected_subs:
            sid = int(sid)
            info = self.predictor_mappings.get(sid)
            if not isinstance(info, dict) or "representative" not in info:
                raise KeyError(f"No p-median representative for scenario {sid}.")
            representative = int(info["representative"])
            if representative not in dual_cache:
                raise KeyError(
                    f"Exact dual of p-median representative {representative} "
                    f"for scenario {sid} is unavailable."
                )
            dual = dual_cache[representative]
            predictions[sid] = {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in dual.items()
            }
        return predictions

    def _weighted_average_structured_duals(self, dual_list, weights):
        """Average structured dual dictionaries using nonnegative normalized weights."""
        if not dual_list:
            return {}
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(dual_list) != weights.size:
            raise ValueError("Dual list and weight vector have inconsistent lengths.")
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0.0:
            weights[:] = 1.0
        weights /= weights.sum()

        averaged = {}
        top_level_keys = set()
        for dual_dict in dual_list:
            if isinstance(dual_dict, dict):
                top_level_keys.update(dual_dict.keys())

        for key in top_level_keys:
            present = [
                (dual_dict[key], weights[i])
                for i, dual_dict in enumerate(dual_list)
                if isinstance(dual_dict, dict) and key in dual_dict
            ]
            if not present:
                continue
            first_value = present[0][0]
            if isinstance(first_value, dict):
                nested_keys = set()
                for value, _ in present:
                    if isinstance(value, dict):
                        nested_keys.update(value.keys())
                averaged[key] = {}
                for nested_key in nested_keys:
                    numerator = 0.0
                    denominator = 0.0
                    for value, weight in present:
                        if isinstance(value, dict) and nested_key in value:
                            numerator += weight * float(value[nested_key])
                            denominator += weight
                    if denominator > 0.0:
                        averaged[key][nested_key] = numerator / denominator
            else:
                numerator = sum(
                    weight * float(value)
                    for value, weight in present
                    if value is not None
                )
                denominator = sum(
                    weight for value, weight in present if value is not None
                )
                if denominator > 0.0:
                    averaged[key] = numerator / denominator
        return averaged

    def _predict_knn(self, unselected_subs, dual_cache):
        """Predict using KNN predictor with uniform averaging."""
        if self.predictor_mappings is None:
            raise ValueError("KNN predictor mappings not computed")
            
        return self._predict_duals_knn(unselected_subs, dual_cache, self.predictor_mappings,
                                      self.problem_type, self.data)
    
    def _predict_tree(self, unselected_subs, dual_cache):
        """
        Predict dual variables using TREE predictor with pre-computed mappings.
        
        For each unselected scenario, use the tree mapping to find its leaf anchor(s)
        and copy/average the anchor dual values.
        
        Args:
            unselected_subs (list/set): Unselected subproblem/scenario IDs
            dual_cache (dict): Current dual solutions {selected_sid: dual_dict}
            
        Returns:
            dict: Predicted dual variables {unselected_sid: dual_dict}
        """
        if self.predictor_mappings is None:
            logger.error("TREE predictor mappings not available - predictor may not be properly trained")
            return {}
        
        predictions = {}
        successful_predictions = 0
        
        for sid in unselected_subs:
            if sid in self.predictor_mappings:
                mapping_info = self.predictor_mappings[sid]
                anchor_ids = mapping_info.get('anchor')
                
                # Handle both single anchor and multiple anchors
                if anchor_ids:
                    # Ensure anchor_ids is always a list for uniform processing
                    if not isinstance(anchor_ids, list):
                        anchor_ids = [anchor_ids]
                    
                    # Find available anchors in dual cache
                    available_anchors = [aid for aid in anchor_ids if aid in dual_cache]
                    
                    if available_anchors:
                        if len(available_anchors) == 1:
                            # Single anchor - direct copy
                            anchor_id = available_anchors[0]
                            anchor_duals = dual_cache[anchor_id]
                            # Deep copy to avoid reference issues
                            if isinstance(anchor_duals, dict):
                                predictions[sid] = {
                                    var_name: (dict(var_value) if isinstance(var_value, dict) else var_value)
                                    for var_name, var_value in anchor_duals.items()
                                }
                            else:
                                predictions[sid] = anchor_duals
                            successful_predictions += 1
                        else:
                            # Multiple anchors - average their dual values
                            anchor_duals_list = [dual_cache[aid] for aid in available_anchors]
                            averaged_duals = self._average_structured_duals(anchor_duals_list)
                            predictions[sid] = averaged_duals
                            successful_predictions += 1
                            # logger.debug(f"Averaged {len(available_anchors)} anchors for scenario {sid}")
                    else:
                        logger.warning(f"None of anchors {anchor_ids} for scenario {sid} found in dual cache")
                else:
                    logger.warning(f"No anchor IDs found for scenario {sid} in tree mapping")
            else:
                logger.warning(f"No tree mapping found for unselected scenario {sid}")
        
        # logger.debug(f"TREE prediction: {successful_predictions}/{len(unselected_subs)} successful predictions")
        return predictions
    
    def _predict_clustering(self, unselected_subs, dual_cache):
        """
        Predict dual variables using CLUSTERING predictor with pre-computed cluster mappings.
        
        For each unselected scenario, use the cluster mapping to find its representative
        and copy the representative's dual values. This applies to all clustering-based methods
        regardless of the input data source (feature vectors, duals, covariates, uncertainty).
        
        Args:
            unselected_subs (list/set): Unselected subproblem/scenario IDs
            dual_cache (dict): Current dual solutions {selected_sid: dual_dict}
            
        Returns:
            dict: Predicted dual variables {unselected_sid: dual_dict}
        """
        if self.predictor_mappings is None:
            logger.error(f"{self.ml_method_type}/CLUSTERING mappings not available - predictor may not be properly trained")
            return {}
        
        # Validation: Check if all representatives are available in dual cache
        all_representatives = set()
        for mapping_info in self.predictor_mappings.values():
            if 'representative' in mapping_info:
                all_representatives.add(mapping_info['representative'])
        
        missing_representatives = all_representatives - set(dual_cache.keys())
        if missing_representatives:
            logger.error(f"{self.ml_method_type}/CLUSTERING: Missing representatives {missing_representatives} in dual cache. "
                        f"Available in cache: {set(dual_cache.keys())}. All required representatives: {all_representatives}")
        
        predictions = {}
        successful_predictions = 0
        
        for sid in unselected_subs:
            if sid in self.predictor_mappings:
                mapping_info = self.predictor_mappings[sid]
                representative_id = mapping_info.get('representative')
                cluster_id = mapping_info.get('cluster_id')
                
                # CRITICAL FIX (April 18, 2026): Check representative_id is not None, not just truthy
                # representative_id could be 0 (a valid scenario ID), which is falsy but valid
                if representative_id is not None and representative_id in dual_cache:
                    # Copy dual values from cluster representative
                    representative_duals = dual_cache[representative_id]
                    # Deep copy to avoid reference issues
                    if isinstance(representative_duals, dict):
                        predictions[sid] = {
                            var_name: (dict(var_value) if isinstance(var_value, dict) else var_value)
                            for var_name, var_value in representative_duals.items()
                        }
                    else:
                        predictions[sid] = representative_duals
                    successful_predictions += 1
                    logger.debug(f"Scenario {sid} -> Representative {representative_id} (Cluster {cluster_id})")
                else:
                    logger.warning(f"Representative {representative_id} (type={type(representative_id).__name__}) for scenario {sid} (cluster {cluster_id}) NOT in cache")
                    logger.debug(f"  representative_id={repr(representative_id)}, bool={bool(representative_id)}, in_cache={representative_id in dual_cache}")
                    logger.debug(f"  Available cache keys: {sorted(dual_cache.keys())}")
            else:
                logger.warning(f"No cluster mapping found for unselected scenario {sid}")
        
        # logger.debug(f"{self.ml_method_type}/CLUSTERING prediction: {successful_predictions}/{len(unselected_subs)} successful predictions")
        return predictions
    
    def _predict_random_sampling(self, unselected_subs, dual_cache):
        """
        Predict dual variables using Random_Sampling (dynamic random selection at iteration time).
        
        Returns empty dict since selection is done dynamically, not via fixed mapping.
        """
        # logger.debug(f"Random_Sampling: Dynamic selection at iteration time (no fixed mapping)")
        return {}
        
    
    def _predict_dual_clustering(self, unselected_subs, dual_cache):
        """
        Predict dual variables using DUAL_CLUSTERING method with pre-computed cluster mappings.
        
        For each unselected scenario, use the cluster mapping to find its representative
        and copy the representative's dual values.
        
        Args:
            unselected_subs (list/set): Unselected subproblem/scenario IDs
            dual_cache (dict): Current dual solutions {selected_sid: dual_dict}
            
        Returns:
            dict: Predicted dual variables {unselected_sid: dual_dict}
        """
        if not hasattr(self, 'clustering_mappings') or self.clustering_mappings is None:
            logger.error("DUAL_CLUSTERING mappings not available - predictor may not be properly trained")
            return {}
        
        # Validation: Check if all representatives are available in dual cache
        all_representatives = set()
        for mapping_info in self.clustering_mappings.values():
            if 'representative' in mapping_info:
                all_representatives.add(mapping_info['representative'])
        
        missing_representatives = all_representatives - set(dual_cache.keys())
        if missing_representatives:
            logger.error(f"DUAL_CLUSTERING: Missing representatives {missing_representatives} in dual cache. "
                        f"Available in cache: {set(dual_cache.keys())}. All required representatives: {all_representatives}")
        
        predictions = {}
        successful_predictions = 0
        
        for sid in unselected_subs:
            if sid in self.clustering_mappings:
                mapping_info = self.clustering_mappings[sid]
                representative_id = mapping_info.get('representative')
                cluster_id = mapping_info.get('cluster_id')
                
                if representative_id and representative_id in dual_cache:
                    # Copy dual values from cluster representative
                    representative_duals = dual_cache[representative_id]
                    # Deep copy to avoid reference issues
                    if isinstance(representative_duals, dict):
                        predictions[sid] = {
                            var_name: (dict(var_value) if isinstance(var_value, dict) else var_value)
                            for var_name, var_value in representative_duals.items()
                        }
                    else:
                        predictions[sid] = representative_duals
                    successful_predictions += 1
                    logger.debug(f"Scenario {sid} -> Representative {representative_id} (Cluster {cluster_id})")
                else:
                    logger.warning(f"Representative {representative_id} for scenario {sid} (cluster {cluster_id}) not found in dual cache")
                    logger.debug(f"Available cache keys: {list(dual_cache.keys())}")
            else:
                logger.warning(f"No cluster mapping found for unselected scenario {sid}")
        
        # logger.debug(f"DUAL_CLUSTERING prediction: {successful_predictions}/{len(unselected_subs)} successful predictions")
        return predictions
    
    def _predict_dual_clustering_with_dual(self, unselected_subs, dual_cache):
        """
        Predict dual variables using DUAL_CLUSTERING_WITH_DUAL method with pre-computed cluster mappings.
        
        For each unselected scenario, use the cluster mapping to find its representative
        and copy the representative's dual values. This method uses clustering based on actual
        dual solutions rather than feature vectors (same logic as DUAL_CLUSTERING).
        
        Args:
            unselected_subs (list/set): Unselected subproblem/scenario IDs
            dual_cache (dict): Current dual solutions {selected_sid: dual_dict}
            
        Returns:
            dict: Predicted dual variables {unselected_sid: dual_dict}
        """
        if not hasattr(self, 'dual_clustering_with_dual_mappings') or self.dual_clustering_with_dual_mappings is None:
            logger.error("DUAL_CLUSTERING_WITH_DUAL mappings not available - predictor may not be properly trained")
            return {}
        
        predictions = {}
        successful_predictions = 0
        
        for sid in unselected_subs:
            if sid in self.dual_clustering_with_dual_mappings:
                mapping_info = self.dual_clustering_with_dual_mappings[sid]
                representative_id = mapping_info.get('representative')
                cluster_id = mapping_info.get('cluster_id')
                
                if representative_id and representative_id in dual_cache:
                    # Copy dual values from cluster representative (same as DUAL_CLUSTERING)
                    representative_duals = dual_cache[representative_id]
                    # Deep copy to avoid reference issues
                    if isinstance(representative_duals, dict):
                        predictions[sid] = {
                            var_name: (dict(var_value) if isinstance(var_value, dict) else var_value)
                            for var_name, var_value in representative_duals.items()
                        }
                    else:
                        predictions[sid] = representative_duals
                    successful_predictions += 1
                    logger.debug(f"Scenario {sid} -> Representative {representative_id} (Cluster {cluster_id})")
                else:
                    logger.warning(f"Representative {representative_id} for scenario {sid} (cluster {cluster_id}) not found in dual cache")
            else:
                logger.warning(f"No dual clustering mapping found for unselected scenario {sid}")
        
        # logger.debug(f"DUAL_CLUSTERING_WITH_DUAL prediction: {successful_predictions}/{len(unselected_subs)} successful predictions")
        return predictions
    
    def _predict_uncertainty_clustering(self, unselected_subs, dual_cache):
        """
        Predict dual variables using UNCERTAINTY_CLUSTERING method with pre-computed cluster mappings.
        
        For each unselected scenario, use the uncertainty clustering mapping to find its 
        cluster representative and copy the representative's dual values.
        
        Args:
            unselected_subs (list/set): Unselected subproblem/scenario IDs
            dual_cache (dict): Current dual solutions {selected_sid: dual_dict}
            
        Returns:
            dict: Predicted dual variables {unselected_sid: dual_dict}
        """
        if not hasattr(self, 'uncertainty_clustering_mappings') or self.uncertainty_clustering_mappings is None:
            logger.error("UNCERTAINTY_CLUSTERING mappings not available - predictor may not be properly trained")
            return {}
        
        # Validation: Check if all representatives are available in dual cache
        all_representatives = set(self.uncertainty_clustering_mappings.values())
        missing_representatives = all_representatives - set(dual_cache.keys())
        if missing_representatives:
            logger.error(f"UNCERTAINTY_CLUSTERING: Missing representatives {missing_representatives} in dual cache. "
                        f"Available in cache: {set(dual_cache.keys())}. All required representatives: {all_representatives}")
        
        predictions = {}
        successful_predictions = 0
        
        for sid in unselected_subs:
            if sid in self.uncertainty_clustering_mappings:
                # Get representative scenario ID for this unselected scenario
                representative_id = self.uncertainty_clustering_mappings[sid]
                
                if representative_id in dual_cache:
                    # Copy dual values from cluster representative
                    representative_duals = dual_cache[representative_id]
                    # Deep copy to avoid reference issues
                    if isinstance(representative_duals, dict):
                        predictions[sid] = {
                            var_name: (dict(var_value) if isinstance(var_value, dict) else var_value)
                            for var_name, var_value in representative_duals.items()
                        }
                    else:
                        predictions[sid] = representative_duals
                    successful_predictions += 1
                    # logger.debug(f"UNCERTAINTY_CLUSTERING: Scenario {sid} -> Representative {representative_id}")
                else:
                    logger.warning(f"Representative {representative_id} for scenario {sid} not found in dual cache")
                    logger.debug(f"Available cache keys: {list(dual_cache.keys())}")
            else:
                logger.warning(f"No uncertainty clustering mapping found for unselected scenario {sid}")
        
        # logger.debug(f"UNCERTAINTY_CLUSTERING prediction: {successful_predictions}/{len(unselected_subs)} successful predictions")
        return predictions
    
    def _predict_covariate_clustering(self, unselected_subs, dual_cache):
        """
        Predict dual variables using COVARIATE_CLUSTERING method with pre-computed cluster mappings.
        
        For each unselected scenario, use the covariate clustering mapping to find its 
        cluster representative and copy the representative's dual values.
        
        Args:
            unselected_subs (list/set): Unselected subproblem/scenario IDs
            dual_cache (dict): Current dual solutions {selected_sid: dual_dict}
            
        Returns:
            dict: Predicted dual variables {unselected_sid: dual_dict}
        """
        if not hasattr(self, 'covariate_clustering_mappings') or self.covariate_clustering_mappings is None:
            logger.error("COVARIATE_CLUSTERING mappings not available - predictor may not be properly trained")
            return {}
        
        # Validation: Check if all representatives are available in dual cache
        all_representatives = set(self.covariate_clustering_mappings.values())
        missing_representatives = all_representatives - set(dual_cache.keys())
        if missing_representatives:
            logger.error(f"COVARIATE_CLUSTERING: Missing representatives {missing_representatives} in dual cache. "
                        f"Available in cache: {set(dual_cache.keys())}. All required representatives: {all_representatives}")
        
        predictions = {}
        successful_predictions = 0
        
        for sid in unselected_subs:
            if sid in self.covariate_clustering_mappings:
                # Get representative scenario ID for this unselected scenario
                representative_id = self.covariate_clustering_mappings[sid]
                
                if representative_id in dual_cache:
                    # Copy dual values from cluster representative
                    representative_duals = dual_cache[representative_id]
                    # Deep copy to avoid reference issues
                    if isinstance(representative_duals, dict):
                        predictions[sid] = {
                            var_name: (dict(var_value) if isinstance(var_value, dict) else var_value)
                            for var_name, var_value in representative_duals.items()
                        }
                    else:
                        predictions[sid] = representative_duals
                    successful_predictions += 1
                    # logger.debug(f"COVARIATE_CLUSTERING: Scenario {sid} -> Representative {representative_id}")
                else:
                    logger.warning(f"Representative {representative_id} for scenario {sid} not found in dual cache")
                    logger.debug(f"Available cache keys: {list(dual_cache.keys())}")
            else:
                logger.warning(f"No covariate clustering mapping found for unselected scenario {sid}")
        
        # logger.debug(f"COVARIATE_CLUSTERING prediction: {successful_predictions}/{len(unselected_subs)} successful predictions")
        return predictions
    
    def _predict_bertsimas_random_sampling(self, unselected_subs, dual_cache):
        """
        Predict dual variables using BERTSIMAS_RANDOM_SAMPLING method.
        
        For SCFLP, average all currently selected scenarios' dual values.
        For SCMND, retain the established baseline of copying the first
        selected scenario's dual values. The selected set can change each iteration.
        
        Args:
            unselected_subs (list/set): Unselected subproblem/scenario IDs
            dual_cache (dict): Current dual solutions {selected_sid: dual_dict}
            
        Returns:
            dict: Predicted dual variables {unselected_sid: dual_dict}
        """
        if not dual_cache:
            logger.warning("BERTSIMAS_RANDOM_SAMPLING: No dual solutions available for averaging")
            return {}
        
        predictions = {}
        selected_scenarios = list(dual_cache.keys())
        num_selected = len(selected_scenarios)
        
        # logger.debug(f"BERTSIMAS_RANDOM_SAMPLING: Computing average of {num_selected} selected scenarios")
        
        # Get the structure from the first dual solution to understand variable layout
        first_dual = dual_cache[selected_scenarios[0]]
        if self.problem_type != "SCFLP" and not getattr(
            self, "_random_sampling_first_dual_notice_logged", False
        ):
            if self.problem_type == "SCMND":
                logger.info(
                    "BERTSIMAS_RANDOM_SAMPLING SCMND baseline: copying the first "
                    "selected scenario's dual to unselected scenarios."
                )
            else:
                logger.warning(
                    "BERTSIMAS_RANDOM_SAMPLING: %s uses first-selected-dual fallback",
                    self.problem_type,
                )
            self._random_sampling_first_dual_notice_logged = True
        
        for sid in unselected_subs:
            # Initialize prediction dict with same structure as input duals
            if self.problem_type == "SCFLP":
                # Compute average of mu and nu variables across all selected scenarios
                avg_mu = {}
                avg_nu = {}
                
                # Get all customer and facility indices from first dual
                mu_keys = list(first_dual['mu'].keys())
                nu_keys = list(first_dual['nu'].keys())
                
                # Average mu values (customer dual variables)
                for i in mu_keys:
                    mu_sum = sum(dual_cache[sel_sid]['mu'][i] for sel_sid in selected_scenarios)
                    avg_mu[i] = mu_sum / num_selected
                
                # Average nu values (facility dual variables)  
                for j in nu_keys:
                    nu_sum = sum(dual_cache[sel_sid]['nu'][j] for sel_sid in selected_scenarios)
                    avg_nu[j] = nu_sum / num_selected
                
                predictions[sid] = {
                    'mu': avg_mu,
                    'nu': avg_nu
                }
            else:
                # Preserve the existing first-selected-dual baseline; no
                # per-scenario warning or numerical-method change.
                predictions[sid] = first_dual.copy()
        
        # logger.debug(f"BERTSIMAS_RANDOM_SAMPLING prediction: {len(predictions)}/{len(unselected_subs)} successful predictions")
        return predictions
    
    # Helper prediction methods (moved from callbacks.py)
    def _predict_duals_regression(self, subs, trained_weights, feature_vectors, problem_type, data):
        """REG prediction implementation (moved from callbacks.py predict_duals)."""
        pt = problem_type.upper()
        preds = {}
        
        if pt == "UFL":
            for sid in subs:
                feat = feature_vectors[sid]
                
                # Lambda prediction
                lam = float(np.dot(trained_weights['lambda'], feat))
                
                # Mu predictions
                mu_pred = {}
                for j in data.F:
                    mu_pred[j] = float(np.dot(trained_weights['mu'][j], feat))
                
                preds[sid] = {"lambda": lam, "mu": mu_pred}
                
        elif pt == "UHL":
            for sid in subs:
                feat = feature_vectors[sid]
                
                # V predictions
                v_pred = {}
                for k in data.N:
                    v_pred[k] = float(np.dot(trained_weights['v'][k], feat))
                
                # U predictions  
                u_pred = {}
                for l in data.N:
                    u_pred[l] = float(np.dot(trained_weights['u'][l], feat))
                
                preds[sid] = {"v": v_pred, "u": u_pred}
                
        elif pt == "SSLP":
            for sid in subs:
                feat = feature_vectors[sid]
                
                # Pi predictions
                pi_pred = {}
                for i in data.C:
                    pi_pred[i] = float(np.dot(trained_weights['pi'][i], feat))
                
                # Gamma predictions
                gamma_pred = {}
                for i in data.C:
                    for j in data.F:
                        gamma_pred[(i, j)] = float(np.dot(trained_weights['gamma'][(i, j)], feat))
                
                preds[sid] = {'pi': pi_pred, 'gamma': gamma_pred}
                
        elif pt == "SCFLP":
            for sid in subs:
                feat = feature_vectors[sid]
                
                # Mu predictions
                mu_pred = {}
                for i in data.I:
                    mu_pred[i] = float(np.dot(trained_weights['mu'][i], feat))
                
                # Nu predictions
                nu_pred = {}
                for j in data.J:
                    nu_pred[j] = float(np.dot(trained_weights['nu'][j], feat))
                
                preds[sid] = {'mu': mu_pred, 'nu': nu_pred}
        else:
            raise ValueError(f"Unsupported problem type for REG prediction: {problem_type}")
            
        return preds
    
    def _predict_duals_knn(self, subs, dual_cache, knn_neighbors, problem_type, data):
        """KNN prediction implementation (updated for integrated KNN format).""" 
        pt = problem_type.upper()
        preds = {}
        
        if pt == "UFL":
            for sid in subs:
                if sid not in knn_neighbors:
                    raise ValueError(f"No KNN neighbors found for scenario {sid}")
                
                # Handle integrated KNN format: {'neighbors': [...], 'distances': [...], 'k': int}
                neighbor_info = knn_neighbors[sid]
                if isinstance(neighbor_info, dict) and 'neighbors' in neighbor_info:
                    neighbors = neighbor_info['neighbors']
                else:
                    # Legacy format: direct list of neighbors
                    neighbors = neighbor_info
                
                valid_neighbors = [n for n in neighbors if n in dual_cache]
                
                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")
                
                k = len(valid_neighbors)
                
                # Lambda prediction (uniform averaging)
                lam = sum(dual_cache[n]['lambda'] for n in valid_neighbors) / k
                
                # Mu predictions (uniform averaging)
                mu_pred = {}
                for j in data.F:
                    mu_pred[j] = sum(dual_cache[n]['mu'][j] for n in valid_neighbors) / k
                
                preds[sid] = {"lambda": lam, "mu": mu_pred}
                
        elif pt == "UHL":
            for sid in subs:
                if sid not in knn_neighbors:
                    raise ValueError(f"No KNN neighbors found for scenario {sid}")
                
                # Handle integrated KNN format: {'neighbors': [...], 'distances': [...], 'k': int}
                neighbor_info = knn_neighbors[sid]
                if isinstance(neighbor_info, dict) and 'neighbors' in neighbor_info:
                    neighbors = neighbor_info['neighbors']
                else:
                    # Legacy format: direct list of neighbors
                    neighbors = neighbor_info
                
                valid_neighbors = [n for n in neighbors if n in dual_cache]
                
                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")
                
                k = len(valid_neighbors)
                
                # V predictions (uniform averaging)
                v_pred = {}
                for node in data.N:
                    v_pred[node] = sum(dual_cache[n]['v'][node] for n in valid_neighbors) / k
                
                # U predictions (uniform averaging)  
                u_pred = {}
                for node in data.N:
                    u_pred[node] = sum(dual_cache[n]['u'][node] for n in valid_neighbors) / k
                
                preds[sid] = {"v": v_pred, "u": u_pred}
                
        elif pt == "SSLP":
            for sid in subs:
                if sid not in knn_neighbors:
                    raise ValueError(f"No KNN neighbors found for scenario {sid}")
                
                # Handle integrated KNN format: {'neighbors': [...], 'distances': [...], 'k': int}
                neighbor_info = knn_neighbors[sid]
                if isinstance(neighbor_info, dict) and 'neighbors' in neighbor_info:
                    neighbors = neighbor_info['neighbors']
                else:
                    # Legacy format: direct list of neighbors
                    neighbors = neighbor_info
                
                valid_neighbors = [n for n in neighbors if n in dual_cache]
                
                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")
                
                k = len(valid_neighbors)
                
                # Pi predictions (uniform averaging)
                pi_pred = {}
                for i in data.C:
                    pi_pred[i] = sum(dual_cache[n]['pi'][i] for n in valid_neighbors) / k
                
                # Gamma predictions (uniform averaging)
                gamma_pred = {}
                for i in data.C:
                    for j in data.F:
                        gamma_pred[(i, j)] = sum(dual_cache[n]['gamma'][(i, j)] for n in valid_neighbors) / k
                
                preds[sid] = {'pi': pi_pred, 'gamma': gamma_pred}
                
        elif pt == "SCFLP":
            for sid in subs:
                if sid not in knn_neighbors:
                    raise ValueError(f"No KNN neighbors found for scenario {sid}")
                
                # Handle integrated KNN format: {'neighbors': [...], 'distances': [...], 'k': int}
                neighbor_info = knn_neighbors[sid]
                if isinstance(neighbor_info, dict) and 'neighbors' in neighbor_info:
                    neighbors = neighbor_info['neighbors']
                else:
                    # Legacy format: direct list of neighbors
                    neighbors = neighbor_info
                
                valid_neighbors = [n for n in neighbors if n in dual_cache]
                
                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")
                
                k = len(valid_neighbors)
                
                # Mu predictions (uniform averaging)
                mu_pred = {}
                for i in data.I:
                    mu_pred[i] = sum(dual_cache[n]['mu'][i] for n in valid_neighbors) / k
                
                # Nu predictions (uniform averaging)
                nu_pred = {}
                for j in data.J:
                    nu_pred[j] = sum(dual_cache[n]['nu'][j] for n in valid_neighbors) / k
                
                preds[sid] = {'mu': mu_pred, 'nu': nu_pred}
        elif pt in ["SUMND", "SCMND"]:
            for sid in subs:
                if sid not in knn_neighbors:
                    raise ValueError(f"No KNN neighbors found for scenario {sid}")

                # Handle integrated KNN format: {'neighbors': [...], 'distances': [...], 'k': int}
                neighbor_info = knn_neighbors[sid]
                if isinstance(neighbor_info, dict) and 'neighbors' in neighbor_info:
                    neighbors = neighbor_info['neighbors']
                else:
                    # Legacy format: direct list of neighbors
                    neighbors = neighbor_info

                valid_neighbors = [n for n in neighbors if n in dual_cache]

                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")

                k = len(valid_neighbors)

                # Pi predictions (uniform averaging over node-commodity duals)
                pi_pred = {}
                for commodity in data.K:
                    for node in data.N:
                        pi_pred[(node, commodity)] = (
                            sum(dual_cache[n]['pi'][(node, commodity)] for n in valid_neighbors) / k
                        )

                # Alpha predictions (uniform averaging over arc-capacity duals)
                alpha_pred = {}
                for idx in range(len(data.A)):
                    alpha_pred[idx] = sum(dual_cache[n]['alpha'][idx] for n in valid_neighbors) / k

                preds[sid] = {'pi': pi_pred, 'alpha': alpha_pred}
        else:
            raise ValueError(f"Unsupported problem type for KNN prediction: {problem_type}")
            
        return preds
    
    def _predict_duals_tree(self, subs, dual_cache, tree_mappings, problem_type, data):
        """TREE prediction implementation (moved from callbacks.py tree_predict)."""
        if tree_mappings.get('__tree_type__') != 'TREE_FIXED_MAPPING':
            raise ValueError("Expected TREE_FIXED_MAPPING models but found different type")
            
        prediction_mappings = tree_mappings.get('__prediction_mappings__', {})
        
        if not prediction_mappings:
            raise ValueError("Tree prediction mappings not available")
        
        pt = problem_type.upper()
        preds = {}
        
        if pt == "UFL":
            for sid in subs:
                if sid not in prediction_mappings:
                    raise ValueError(f"No tree mapping found for unselected scenario {sid}")
                    
                mapping = prediction_mappings[sid]
                neighbors = mapping['neighbors']
                
                # Validate neighbors are in dual_cache
                valid_neighbors = [n for n in neighbors if n in dual_cache]
                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")
                
                # Recompute uniform weights for available neighbors
                n_valid = len(valid_neighbors)
                uniform_weight = 1.0 / n_valid
                
                # Predict lambda using uniform averaging
                lam = sum(dual_cache[n]['lambda'] * uniform_weight for n in valid_neighbors)
                
                # Predict mu values using uniform averaging
                mu_pred = {}
                for j in data.F:
                    mu_pred[j] = sum(dual_cache[n]['mu'][j] * uniform_weight for n in valid_neighbors)
                
                preds[sid] = {"lambda": float(lam), "mu": mu_pred}

        elif pt == "UHL":
            for sid in subs:
                if sid not in prediction_mappings:
                    raise ValueError(f"No tree mapping found for unselected scenario {sid}")
                    
                mapping = prediction_mappings[sid]
                neighbors = mapping['neighbors']
                
                # Validate neighbors
                valid_neighbors = [n for n in neighbors if n in dual_cache]
                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")
                
                n_valid = len(valid_neighbors)
                uniform_weight = 1.0 / n_valid
                
                # Predict v values using uniform averaging
                v_pred = {}
                for k in data.N:
                    v_pred[k] = sum(dual_cache[n]['v'][k] * uniform_weight for n in valid_neighbors)
                
                # Predict u values using uniform averaging
                u_pred = {}
                for l in data.N:
                    u_pred[l] = sum(dual_cache[n]['u'][l] * uniform_weight for n in valid_neighbors)
                
                preds[sid] = {"v": v_pred, "u": u_pred}

        elif pt == "SSLP":
            for sid in subs:
                if sid not in prediction_mappings:
                    raise ValueError(f"No tree mapping found for unselected scenario {sid}")
                    
                mapping = prediction_mappings[sid]
                neighbors = mapping['neighbors']
                
                # Validate neighbors
                valid_neighbors = [n for n in neighbors if n in dual_cache]
                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")
                
                n_valid = len(valid_neighbors)
                uniform_weight = 1.0 / n_valid
                
                # Predict pi values using uniform averaging
                pi_pred = {}
                for i in data.C:
                    pi_pred[i] = sum(dual_cache[n]['pi'][i] * uniform_weight for n in valid_neighbors)
                
                # Predict gamma values using uniform averaging
                gamma_pred = {}
                for i in data.C:
                    for j in data.F:
                        gamma_pred[(i, j)] = sum(dual_cache[n]['gamma'][(i, j)] * uniform_weight for n in valid_neighbors)
                
                preds[sid] = {'pi': pi_pred, 'gamma': gamma_pred}

        elif pt == "SCFLP":
            for sid in subs:
                if sid not in prediction_mappings:
                    raise ValueError(f"No tree mapping found for unselected scenario {sid}")
                    
                mapping = prediction_mappings[sid]
                neighbors = mapping['neighbors']
                
                # Validate neighbors
                valid_neighbors = [n for n in neighbors if n in dual_cache]
                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")
                
                n_valid = len(valid_neighbors)
                uniform_weight = 1.0 / n_valid
                
                # Predict mu values using uniform averaging
                mu_pred = {}
                for i in data.I:
                    mu_pred[i] = sum(dual_cache[n]['mu'][i] * uniform_weight for n in valid_neighbors)
                
                # Predict nu values using uniform averaging
                nu_pred = {}
                for j in data.J:
                    nu_pred[j] = sum(dual_cache[n]['nu'][j] * uniform_weight for n in valid_neighbors)
                
                preds[sid] = {'mu': mu_pred, 'nu': nu_pred}

        elif pt in ["SUMND", "SCMND"]:
            for sid in subs:
                if sid not in prediction_mappings:
                    raise ValueError(f"No tree mapping found for unselected scenario {sid}")

                mapping = prediction_mappings[sid]
                neighbors = mapping['neighbors']

                # Validate neighbors
                valid_neighbors = [n for n in neighbors if n in dual_cache]
                if not valid_neighbors:
                    raise ValueError(f"No valid dual solutions found for neighbors of scenario {sid}")

                n_valid = len(valid_neighbors)
                uniform_weight = 1.0 / n_valid

                # Predict pi values using uniform averaging
                pi_pred = {}
                for commodity in data.K:
                    for node in data.N:
                        pi_pred[(node, commodity)] = (
                            sum(dual_cache[n]['pi'][(node, commodity)] * uniform_weight for n in valid_neighbors)
                        )

                # Predict alpha values using uniform averaging
                alpha_pred = {}
                for idx in range(len(data.A)):
                    alpha_pred[idx] = sum(dual_cache[n]['alpha'][idx] * uniform_weight for n in valid_neighbors)

                preds[sid] = {'pi': pi_pred, 'alpha': alpha_pred}

        else:
            raise ValueError(f"Unsupported problem type for tree prediction: {problem_type}")
            
        return preds
    
    def _predict_duals_simple_avg(self, subs, dual_cache, selected_scenarios, problem_type, data):
        """SIMPLE_AVG prediction implementation - average all selected scenarios."""
        pt = problem_type.upper()
        preds = {}
        
        # Validate that we have dual solutions for selected scenarios
        valid_selected = [n for n in selected_scenarios if n in dual_cache]
        if not valid_selected:
            raise ValueError("No valid dual solutions found for selected scenarios")
        
        n_selected = len(valid_selected)
        # logger.debug(f"SIMPLE_AVG: Using {n_selected} selected scenarios for averaging")
        
        if pt == "UFL":
            for sid in subs:
                # Lambda prediction (simple average of all selected scenarios)
                lam = sum(dual_cache[n]['lambda'] for n in valid_selected) / n_selected
                
                # Mu predictions (simple average of all selected scenarios)
                mu_pred = {}
                for j in data.F:
                    mu_pred[j] = sum(dual_cache[n]['mu'][j] for n in valid_selected) / n_selected
                
                preds[sid] = {"lambda": float(lam), "mu": mu_pred}
                
        elif pt == "UHL":
            for sid in subs:
                # V predictions (simple average of all selected scenarios)
                v_pred = {}
                for k in data.N:
                    v_pred[k] = sum(dual_cache[n]['v'][k] for n in valid_selected) / n_selected
                
                # U predictions (simple average of all selected scenarios)
                u_pred = {}
                for l in data.N:
                    u_pred[l] = sum(dual_cache[n]['u'][l] for n in valid_selected) / n_selected
                
                preds[sid] = {"v": v_pred, "u": u_pred}
                
        elif pt == "SSLP":
            for sid in subs:
                # Pi predictions (simple average of all selected scenarios)
                pi_pred = {}
                for i in data.C:
                    pi_pred[i] = sum(dual_cache[n]['pi'][i] for n in valid_selected) / n_selected
                
                # Gamma predictions (simple average of all selected scenarios)
                gamma_pred = {}
                for i in data.C:
                    for j in data.F:
                        gamma_pred[(i, j)] = sum(dual_cache[n]['gamma'][(i, j)] for n in valid_selected) / n_selected
                
                preds[sid] = {'pi': pi_pred, 'gamma': gamma_pred}
                
        elif pt == "SCFLP":
            for sid in subs:
                # Mu predictions (simple average of all selected scenarios)
                mu_pred = {}
                for i in data.I:
                    mu_pred[i] = sum(dual_cache[n]['mu'][i] for n in valid_selected) / n_selected
                
                # Nu predictions (simple average of all selected scenarios)
                nu_pred = {}
                for j in data.J:
                    nu_pred[j] = sum(dual_cache[n]['nu'][j] for n in valid_selected) / n_selected
                
                preds[sid] = {'mu': mu_pred, 'nu': nu_pred}
        elif pt in ["SUMND", "SCMND"]:
            for sid in subs:
                # Pi predictions (simple average of all selected scenarios)
                pi_pred = {}
                for commodity in data.K:
                    for node in data.N:
                        pi_pred[(node, commodity)] = (
                            sum(dual_cache[n]['pi'][(node, commodity)] for n in valid_selected) / n_selected
                        )

                # Alpha predictions (simple average of all selected scenarios)
                alpha_pred = {}
                for idx in range(len(data.A)):
                    alpha_pred[idx] = sum(dual_cache[n]['alpha'][idx] for n in valid_selected) / n_selected

                preds[sid] = {'pi': pi_pred, 'alpha': alpha_pred}
        else:
            raise ValueError(f"Unsupported problem type for SIMPLE_AVG prediction: {problem_type}")
            
        return preds
    
    def _average_structured_duals(self, dual_list):
        """
        Average multiple structured dual dictionaries component-wise.
        
        Args:
            dual_list (list): List of dual dictionaries to average
            
        Returns:
            dict: Averaged dual dictionary with same structure as input
        """
        if not dual_list:
            return {}
        
        if len(dual_list) == 1:
            # Single dual - return deep copy
            dual_dict = dual_list[0]
            if isinstance(dual_dict, dict):
                return {
                    var_name: (dict(var_value) if isinstance(var_value, dict) else var_value)
                    for var_name, var_value in dual_dict.items()
                }
            else:
                return dual_dict
        
        # Multiple duals - compute average for every top-level component.
        averaged_duals = {}
        top_level_keys = set()
        for dual_dict in dual_list:
            if isinstance(dual_dict, dict):
                top_level_keys.update(dual_dict.keys())

        for key in top_level_keys:
            sample_values = [dual_dict.get(key) for dual_dict in dual_list if isinstance(dual_dict, dict) and key in dual_dict]
            if not sample_values:
                continue

            # Dict-valued components (e.g., mu/nu or pi/alpha) are averaged component-wise.
            if isinstance(sample_values[0], dict):
                nested_keys = set()
                for value in sample_values:
                    if isinstance(value, dict):
                        nested_keys.update(value.keys())

                averaged_duals[key] = {}
                for nested_key in nested_keys:
                    nested_values = [value[nested_key] for value in sample_values if isinstance(value, dict) and nested_key in value]
                    if nested_values:
                        averaged_duals[key][nested_key] = sum(nested_values) / len(nested_values)
            else:
                # Scalar-valued components are averaged directly.
                numeric_values = [value for value in sample_values if value is not None]
                if numeric_values:
                    averaged_duals[key] = sum(numeric_values) / len(numeric_values)

        return averaged_duals
    
    def get_training_info(self):
        """Get information about the trained predictor."""
        info = {
            'ml_method_type': self.ml_method_type,
            'predictor_type': self.predictor_type,
            'problem_type': self.problem_type, 
            'is_trained': self.is_trained,
            'training_time': self.training_time  # Unified training time for all methods
        }
        
        # Add predictor-specific information based on predictor type
        if self.predictor_type == 'KNN' and self.predictor_mappings:
            info['n_neighbors'] = self.n_neighbors
            info['optimal_k'] = self.optimal_k
            info['n_unselected'] = len(self.predictor_mappings)
            
        elif self.predictor_type == 'TREE' and self.predictor_mappings:
            info['max_depth'] = self.max_depth
            info['min_samples_leaf'] = self.min_samples_leaf
            info['n_unselected'] = len(self.predictor_mappings)
                
        elif self.predictor_type == 'CLUSTERING' and self.predictor_mappings:
            info['clustering_random_state'] = self.clustering_random_state
            info['clustering_init_method'] = self.clustering_init_method
            info['n_unselected'] = len(self.predictor_mappings)
            info['n_selected'] = len(self.selected_subproblems) if self.selected_subproblems else 0
            
        elif self.predictor_type in {PMEDIAN_WEIGHTED_TRANSFER, PMEDIAN_NEAREST_TRANSFER}:
            info['n_selected'] = len(self.selected_subproblems) if self.selected_subproblems else 0
            info['pmedian_counts'] = dict(self.pmedian_counts)
        elif self.ml_method_type == 'Random_Sampling':
            info['n_selected'] = len(self.selected_subproblems) if self.selected_subproblems else 0
        elif is_parametric_predictor(self.predictor_type):
            info['n_selected'] = len(self.selected_subproblems) if self.selected_subproblems else 0
            info['parametric_diagnostics'] = self.get_parametric_diagnostics()
        
        return info
