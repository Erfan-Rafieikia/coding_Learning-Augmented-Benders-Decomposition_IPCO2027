# Dual Prediction Training and Selection (Code-Aligned)

This note documents what is currently implemented in code, based on:
- `src/Dual_Prediction/training_and_selection.py`
- `src/Dual_Prediction/prediction.py`
- `src/Problem/callbacks.py`

It intentionally reflects runtime behavior and active interfaces.

## 1. Runtime Architecture

### 1.1 One-time vs iterative roles

- `training_and_selection.py`: one-time Step 5 operation at Phase 1 -> Phase 2 transition.
  - selects `p` scenarios to solve exactly,
  - builds fixed mappings for unselected scenarios.
- `prediction.py` (`DualPredictor`): iterative Step 6 operation during Phase 2.
  - solves selected scenarios exactly (in callback),
  - predicts duals for unselected scenarios via the stored mappings.

### 1.2 Active unified entry points

- Selection/training backend:
  ```python
  integrated_prediction_selection(feature_vectors, ml_method_type, predictor_type, p, method_params=None)
  ```
- Predictor unified training:
  ```python
  DualPredictor._train_unified(feature_vectors, p, method_params=None)
  ```
- Predictor iterative inference:
  ```python
  DualPredictor.predict(unselected_subs, dual_cache)
  ```

In the callback, `p` is derived from `n_selected_subproblems` with fallback to all scenarios if missing/invalid.

## 2. Data Sources by ML Method Type

The backend first chooses input vectors according to `ml_method_type`:

- `ML_Augmented_BD`: uses `feature_vectors`.
- `ML_Augmented_BD_with_duals`: uses `method_params['all_trial_duals']`, then converts dual dictionaries to numeric vectors using `_build_stacked_dual_profiles`.
- `Uncertainty_Similarity`: uses `method_params['uncertainty_vectors']`.
- `Covariate_Similarity`: uses `method_params['covariate_vectors']`.
- `Random_Sampling`: returns immediately via `solve_bertsimas_random_sampling_selection` (no fixed selection mapping for prediction).

Then routing is by `predictor_type`:
- `KNN` -> `solve_knn_integrated_selection`
- `TREE` -> `solve_tree_integrated_selection`
- `CLUSTERING` -> `solve_dual_clustering_selection`

## 3. Implemented Selection + Mapping Mathematics

Notation:
- Scenario set: `S`
- Selected set: `T`, `|T| = p`
- Unselected set: `U = S \ T`
- Scenario vectors used by the selected method: `z_s`

### 3.1 KNN

Distance matrix (Euclidean):
$$
D_{ab} = \|z_a - z_b\|_2
$$

Objective evaluated by `_compute_knn_objective`:
$$
\min_{|T|=p} J_k(T), \quad
J_k(T)=\sum_{s\in S\setminus T} \frac{1}{|N_k(s;T)|}\sum_{t\in N_k(s;T)} D_{st}
$$
where `N_k(s;T)` is the set of nearest selected neighbors (or all selected if fewer than `k`).

If `find_optimal_k=True`, code enumerates
$$
k \in \{1,\dots,\min(p,26)\}
$$
and keeps the best pair `(T,k)`.

Mapping for each `u in U`:
```python
{
  'neighbors': [...],
  'distances': [...],
  'k': ...
}
```
plus metadata key `__optimal_k__`.

### 3.2 TREE (adaptive)

Dual dissimilarity matrix from averaged scenario dual vectors:
$$
\Delta_{ab} = \|\bar d_a - \bar d_b\|_2^2
$$

Node loss is medoid loss:
$$
L(A)=\min_{m\in A}\sum_{q\in A}\Delta_{mq}
$$

Split gain used in `_find_adaptive_split`:
$$
\text{gain}=L(A)-\left(\frac{|A_L|}{|A|}L(A_L)+\frac{|A_R|}{|A|}L(A_R)\right)
$$

Adaptive growth (`_build_tree_adaptive`) continues while leaf count `< p` and valid splits exist.
The first split can be forced; there is also emergency fallback to a 50/50 manual split if only one leaf remains in a case that should have at least two.

Selection rules in `solve_tree_integrated_selection`:
- if `num_leaves == p`: 1 medoid per leaf,
- if `num_leaves < p`: proportional allocation with minimum 1 per leaf and exact total `p`.

Allocation constraints:
$$
\sum_{\ell=1}^{L} x_\ell = p, \quad x_\ell \ge 1, \quad x_\ell \le |A_\ell|
$$

When `x_ell > 1`, extra points are added by max-min diversity:
$$
t^* = \arg\max_{t\in A_\ell\setminus B}\min_{b\in B}\Delta_{tb}
$$
where `B` is the already selected set in that leaf.

Mapping for each unselected scenario:
```python
{
  'anchor': [anchor_ids_in_same_leaf],
  'leaf_id': leaf_index
}
```
with diagnostics under `__tree_info__`.

Leaf assignment uses membership-based logic (`scenario_idx in leaf.scenario_ids`) in `_find_scenario_leaf`, with traversal only as fallback.

### 3.3 CLUSTERING

Implemented with `sklearn.cluster.KMeans`, `n_clusters=p`.

Objective:
$$
\min_{\{S_c,m_c\}_{c=1}^{p}} \sum_{c=1}^{p}\sum_{s\in S_c}\|z_s-m_c\|_2^2
$$

Cluster representative:
$$
r_c = \arg\min_{s\in S_c}\|z_s-m_c\|_2
$$

Mapping for unselected scenario `u`:
```python
{
  'representative': representative_id,
  'cluster_id': cluster_id
}
```

If a cluster is empty, it is skipped, so selected representative count can be `< p`.

### 3.4 Random Sampling

`solve_bertsimas_random_sampling_selection` returns:
- selected scenarios: `[]`
- mapping: metadata dictionary (`method`, `p`, `total_scenarios`, `random_state`)

Actual sampling is done dynamically in callback iterations, not by a fixed mapping built here.

## 4. Implemented Prediction Formulas (SCFLP)

Let exact duals in cache for selected scenarios be `(mu^t, nu^t)`.

### 4.1 KNN prediction

For scenario `u`, with usable neighbors `N_u` present in `dual_cache`:
$$
\hat\mu_i^u = \frac{1}{|N_u|}\sum_{t\in N_u}\mu_i^t,
\qquad
\hat\nu_j^u = \frac{1}{|N_u|}\sum_{t\in N_u}\nu_j^t
$$

### 4.2 TREE prediction

For scenario `u`, usable anchors `A_u` from its leaf:
- if `|A_u|=1`: copy anchor dual.
- if `|A_u|>1`: average via `_average_scflp_duals`:
$$
\hat\mu_i^u = \frac{1}{|A_u|}\sum_{t\in A_u}\mu_i^t,
\qquad
\hat\nu_j^u = \frac{1}{|A_u|}\sum_{t\in A_u}\nu_j^t
$$

### 4.3 CLUSTERING prediction

For scenario `u` with representative `r(u)`:
$$
\hat\mu_i^u = \mu_i^{r(u)},
\qquad
\hat\nu_j^u = \nu_j^{r(u)}
$$

## 5. Callback Objective Link (SCFLP)

Predicted duals are used in cut checks through:
$$
\hat Q_s(y) = \sum_{i\in I} \hat\mu_i^s d_i^s - \sum_{j\in J} \hat\nu_j^s \bar u_j^s y_j
$$
where `bar_u_j^s` follows callback capacity selection logic (deterministic or scenario capacity, depending on config).

## 6. Important Implementation Caveats

- Codebase contains legacy classes/functions and old comments; active Phase 2 path is the unified API above.
- In `_predict_clustering`, representative validation uses:
  ```python
  if representative_id and representative_id in dual_cache:
  ```
  so representative ID `0` is treated as false and skipped. This is current behavior.
- Multiple helper methods for removed/legacy methods remain as stubs; they are not the active path for KNN/TREE/CLUSTERING unified flow.

## 7. Minimal Active Flow Summary

1. Callback prepares `feature_vectors` and computes `p`.
2. Callback calls `DualPredictor._train_unified(feature_vectors, p, method_params)`.
3. Backend `integrated_prediction_selection(...)` returns `(selected_scenarios, mapping)`.
4. Callback solves selected scenarios exactly each iteration and passes `dual_cache` to `DualPredictor.predict(...)`.
5. Predictor reconstructs unselected duals using KNN/TREE/CLUSTERING mapping.
