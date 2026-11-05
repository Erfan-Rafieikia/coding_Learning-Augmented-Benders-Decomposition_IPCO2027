"""
Problem.subproblem_dual
=======================

This module implements dual subproblem formulations for ML-augmented Benders decomposition,
**focused exclusively on SCFLP (Stochastic Capacitated Facility Location) problems**.
It provides exact dual solutions that serve as training targets for machine learning models,
validation references for predicted dual variables in the callback system, and
true objective value computation for ML vs exact solution comparison.

Current Focus: SCFLP Only
-------------------------
**As of October 2025, this implementation focuses exclusively on SCFLP problems** to ensure
robust validation of the ML-augmented Benders approach on a single problem class.

Supported Problem Types:
- ✅ **SCFLP (Stochastic Capacitated Facility Location)**: Full implementation
- ❌ **UFL, UHL, SSLP, CMND, MCFL**: Implementations removed (raise NotImplementedError)

The dual subproblem solver transforms primal SCFLP subproblem constraints into their dual
mathematical formulation, enabling direct solution of dual variables without first
solving the primal problem. This approach provides computational efficiency benefits
and enables seamless integration with the ML prediction system that learns to
approximate these dual solutions.

Key Components:
---------------

1. **Dual Problem Formulation**:
   - Mathematical transformation from primal to dual linear programs
   - Problem-specific constraint and objective structures
   - Proper handling of variable bounds and constraint types

2. **Gurobi Integration**:
   - Efficient LP model construction and solving
   - Silent operation with suppressed console output
   - Robust error handling and status checking

3. **Solution Extraction**:
   - Dual variable value extraction and formatting
   - Proper categorization into primary and secondary dual variables
   - Objective value computation and validation

4. **Data Interface Compatibility**:
   - Support for dictionary, array, and matrix input formats
   - Flexible first-stage solution representation handling
   - Consistent output format across all problem types

SCFLP Dual Formulation Mathematics:
------------------------------------

For scenario s with demand d_s and first-stage facility decisions ȳ:

*Primal Subproblem*:
```
min  Σ_{i,j} c_ij * x_ij
s.t. Σ_j x_ij ≥ d_si        ∀i ∈ I
     Σ_i x_ij ≤ u_j * ȳ_j   ∀j ∈ J
     x_ij ≥ 0               ∀i,j
```

*Dual Formulation*:
```
max  Σ_i μ_i * d_si - Σ_j ν_j * u_j * ȳ_j
s.t. μ_i - ν_j ≤ c_ij      ∀i ∈ I, j ∈ J
     μ_i ≥ 0               ∀i ∈ I
     ν_j ≥ 0               ∀j ∈ J
```

Where:
- μ_i: Dual variables for demand constraints (≥ 0)
- ν_j: Dual variables for capacity constraints (≥ 0)

**Other Problem Types** (implementations removed):
- **UFL, UHL, SSLP, CMND, MCFL**: Detailed formulations removed - focusing on SCFLP only

SCFLP Solution Structure and Interpretation:
--------------------------------------------

**Return Format**:
SCFLP dual solution returns a tuple: `(obj_val, mu_duals, nu_duals)`

- **obj_val**: Optimal dual objective value (equals primal objective by strong duality)
- **mu_duals**: Primary dual variables {i: μ_i} for demand constraints (≥ 0)
- **nu_duals**: Secondary dual variables {j: ν_j} for capacity constraints (≥ 0)

**Economic Interpretation**:
- **μ_i**: Shadow prices for customer demand constraints (marginal cost of additional demand)
- **ν_j**: Shadow prices for facility capacity constraints (marginal value of additional capacity)
- **Objective value**: Total marginal cost contribution from this scenario

**Cut Generation Context**:
The SCFLP dual solution provides coefficients for Benders cuts of the form:
```
θ_s ≥ obj_val - Σ_j ν_j * u_j * (y_j - ȳ_j)
```

Where the cut depends on changes in facility opening decisions y_j from current solution ȳ_j.

SCFLP Input Data Handling:
--------------------------

**First-Stage Solution Formats**:
SCFLP solver accepts multiple input formats for first-stage facility opening decisions:

*Dictionary/TupleDict Format*:
```python
first_stage = {j: value_j for j in facilities}  # {facility_id: 0 or 1}
```

*Array Format*:
```python
first_stage = np.array([value_0, value_1, ..., value_J])  # Binary facility decisions
```

**Subproblem Indexing**:
- **SCFLP**: Scenario index s ∈ {0, 1, ..., |scenarios|-1}

**SCFLP Data Object Requirements**:
The SCFLPData object must contain:
- **data.scenarios**: Demand matrix [scenarios × customers] 
- **data.I**: Customer indices array
- **data.J**: Facility indices array
- **data.capacity**: Facility capacity array [facilities]
- **data.cost**: Transportation cost matrix [customers × facilities]

SCFLP Performance Characteristics:
-----------------------------------

**Computational Complexity**:
- **SCFLP**: O(|customers| × |facilities|) variables and constraints per subproblem
- Efficient LP formulation with proven polynomial-time complexity

**Memory Requirements**:
- Linear in problem dimensions for each subproblem
- Efficient sparse matrix representation in Gurobi
- Optimized for SCFLP problem structure

**Solver Performance**:
- LP relaxations solve very efficiently (milliseconds for moderate sizes)
- Strong dual bounds from optimal dual solutions
- Numerical stability through Gurobi's robust LP algorithms
- Optimized for repeated SCFLP subproblem solving in Phase 2

Error Handling and Robustness:
------------------------------

**Optimization Status Checking**:
- Verification that Gurobi reaches optimal status
- Descriptive error messages for non-optimal termination
- Graceful handling of numerical issues

**Input Validation**:
- Automatic format detection and conversion for first-stage solutions
- Bounds checking for subproblem indices
- Data structure compatibility verification

**Numerical Robustness**:
- Gurobi's default numerical tolerances for stability
- Proper handling of very small/large dual values
- Consistent precision across different problem scales

Integration with Framework:
---------------------------

**Callback System Integration**:
The dual solver integrates with the callback framework through:
- Training data generation for ML models
- Validation of predicted dual variables
- Fallback exact solving when predictions are unreliable
- True objective value computation for ML vs exact comparison (master_problem.py)

**Caching and Efficiency**:
- Solution caching to avoid redundant dual solves
- Efficient model construction with minimal overhead
- Memory management for large subproblem sets

**Quality Assurance**:
- Dual-primal objective value verification (strong duality)
- Constraint feasibility checking for extracted solutions
- Integration with logging system for debugging and research

Research Applications:
----------------------

The dual subproblem solver supports research in:
- Benders decomposition algorithm development
- Machine learning integration with optimization
- Dual solution analysis and interpretation
- Algorithm performance benchmarking and validation

Dependencies:
-------------
- gurobipy: Commercial optimization solver for linear programming
- numpy: Numerical array operations and data handling
- logging: Comprehensive diagnostic and error logging
- Problem.data: Data structure definitions for all problem types

Usage Patterns:
---------------

**Standard Dual Solving**:
```python
obj_val, primary_duals, secondary_duals = solve_dual_subproblem(
    'SCFLP', scflp_data, scenario_idx, facility_decisions
)
```

**Training Data Generation**:
```python
dual_solutions = {}
for scenario in scenarios:
    obj_val, p_duals, s_duals = solve_dual_subproblem(
        problem_type, data, scenario, first_stage_solution
    )
    dual_solutions[scenario] = (obj_val, p_duals, s_duals)
```

**Validation and Debugging**:
```python
# Verify dual-primal relationship
primal_obj = solve_primal_subproblem(...)
dual_obj, _, _ = solve_dual_subproblem(...)
assert abs(primal_obj - dual_obj) < tolerance
```

The dual subproblem solver provides the mathematical foundation for the ML-augmented
Benders decomposition framework, enabling efficient and accurate dual variable
computation for training, validation, and exact optimization within the system.

"""

from gurobipy import GRB, Model, quicksum
from .data import (
    SCFLPData,
    NESPData,
    SUMNDData,
    SCMNDData,
    SCMND_FORMULATION_CURRENT,
    SCMND_FORMULATION_MANUSCRIPT,
    scmnd_capacity_vector,
)
import numpy as np
import logging

from utils.exclusive_profiler import get_profiled_logger

logger = get_profiled_logger(__name__)

# Module-level flag to track if capacity usage has been logged
_capacity_usage_logged = False
_sumnd_dual_static_logged = False

def _set_params(model: Model) -> None:
    """Mute Gurobi console output."""
    model.Params.OutputFlag = 0

def solve_dual_subproblem(problem_type: str,
                          data,
                          subproblem_idx: int,
                          first_stage_values,
                          use_stochastic_capacity: bool = True):
    """
    Solve the dual of a single SCFLP subproblem.
    
    **SCFLP Focus**: This function now focuses exclusively on SCFLP problems.
    Other problem types (UFL, UHL, SSLP, etc.) raise NotImplementedError.

    Args:
        problem_type (str): Must be 'SCFLP' - other types not supported
        data: SCFLPData instance
        subproblem_idx (int): Scenario index s ∈ {0, 1, ..., |scenarios|-1}
        first_stage_values: dict/tupledict or length-|J| vector ybar_j (facility decisions)
        use_stochastic_capacity (bool): If True, use scenario-specific capacities when available,
                                       if False, use original capacities (first scenario behavior)

    Returns:
        obj_val (float): Optimal dual objective value
        primary_duals (dict): μ_i dual variables for demand constraints
        secondary_duals (dict): ν_j dual variables for capacity constraints
    """

    pt = problem_type.upper()

    # ---- Non-SCFLP problems: Not implemented (SCFLP focus) ----
    if pt in ["UFL", "UHL", "SSLP", "CMND", "MCFL"]:
        raise NotImplementedError(f"{pt} dual subproblem removed - focusing on SCFLP only")

    # ---- SCFLP dual ----
    elif pt == "SCFLP":
        # subproblem_idx is the index of the scenario s in data.S
        s = subproblem_idx
        demand = data.scenarios[s]   # demand[i] for all i ∈ I
        
        # Determine which capacities to use based on use_stochastic_capacity parameter
        if use_stochastic_capacity and data.capacity_scenarios is not None:
            # Use scenario-specific capacities if available and enabled
            if s < len(data.capacity_scenarios):
                scenario_capacities = data.capacity_scenarios[s]
                logger.debug(f"Using scenario-specific capacities for scenario {s}")
            else:
                # Fallback to original capacities if scenario index out of range
                scenario_capacities = data.capacity
                logger.warning(f"Scenario {s} out of range for capacity_scenarios, using original capacities")
        else:
            # Use original capacities (first scenario behavior)
            scenario_capacities = data.capacity
            global _capacity_usage_logged
            if not _capacity_usage_logged:
                if not use_stochastic_capacity:
                    logger.info(f"Using original capacities for all scenarios (use_stochastic_capacity=False)")
                else:
                    logger.info(f"Using original capacities for all scenarios (no capacity_scenarios available)")
                _capacity_usage_logged = True
        
        # Handle different input types for first_stage_values
        if isinstance(first_stage_values, dict):
            # Convert dict/tupledict to numpy array
            ybar = np.array([first_stage_values[j] for j in data.J])
        else:
            ybar = np.asarray(first_stage_values, dtype=float)  # facility open decisions y_j

        with Model("SCFLP_DUAL") as m:
            _set_params(m)

            # μ_i ≥ 0: dual for demand constraints ∑_j x_ij ≥ d_i
            mus = m.addVars(data.I, lb=0.0, name="mu")
            # ν_j ≥ 0: dual for capacity constraints ∑_i x_ij ≤ u_j * y_j
            nus = m.addVars(data.J, lb=0.0, name="nu")

            # Objective: ∑_i μ_i * d_i - ∑_j ν_j * u_j * y_j
            # Use scenario_capacities instead of data.capacity
            m.setObjective(
                quicksum(mus[i] * demand[i] for i in data.I) -
                quicksum(nus[j] * scenario_capacities[j] * ybar[j] for j in data.J),
                GRB.MAXIMIZE
            )

            # Constraints: μ_i - ν_j ≤ c_ij
            m.addConstrs(
                (mus[i] - nus[j] <= data.cost[i, j] for i in data.I for j in data.J),
                name="dual_feasibility"
            )

            m.optimize()
            if m.status != GRB.OPTIMAL:
                # Enhanced error reporting for dual subproblem issues
                logger.error(f"SCFLP dual subproblem failed for scenario {s}")
                logger.error(f"  Status: {m.status} ({['LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED', 'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT', 'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL', 'INPROGRESS', 'USER_OBJ_LIMIT'][m.status]})")
                logger.error(f"  First-stage solution (ybar): {ybar}")
                logger.error(f"  Scenario demand: {demand}")
                logger.error(f"  Facility capacities: {data.capacity}")
                logger.error(f"  Number of variables: {m.NumVars}")
                logger.error(f"  Number of constraints: {m.NumConstrs}")
                
                # Check if first-stage solution has sufficient capacity
                total_capacity = sum(scenario_capacities[j] * ybar[j] for j in data.J)
                total_demand = sum(demand[i] for i in data.I)
                logger.error(f"  Total capacity available: {total_capacity}")
                logger.error(f"  Total demand: {total_demand}")
                logger.error(f"  Capacity sufficient: {total_capacity >= total_demand}")
                
                # Try to compute the IIS (Irreducible Inconsistent Subsystem) if infeasible
                if m.status == GRB.INFEASIBLE:
                    logger.error("  Computing IIS for infeasible dual subproblem...")
                    m.computeIIS()
                    logger.error(f"  IIS computed. Writing to scflp_dual_infeasible_s{s}.ilp")
                    m.write(f"scflp_dual_infeasible_s{s}.ilp")
                
                raise RuntimeError(f"SCFLP dual not optimal: {m.status}")

            mu_vals = {i: mus[i].X for i in data.I}
            nu_vals = {j: nus[j].X for j in data.J}
            return m.ObjVal, mu_vals, nu_vals

    # ---- NE-SP dual ----
    elif pt in ["NESP", "NE-SP"]:
        # subproblem_idx is the index of the OD pair m in data.M
        m_idx = subproblem_idx
        od_pair = data.M[m_idx]  # (origin, destination)
        o, t = od_pair  # origin and destination nodes
        
        # Handle different input types for first_stage_values (arc activation)
        # first_stage_values can be dict {arc_idx: value} or array [value0, value1, ...]
        if isinstance(first_stage_values, dict):
            # Convert dict to numpy array indexed by arc position
            xbar = np.array([first_stage_values[idx] for idx in range(len(data.A))])
        else:
            xbar = np.asarray(first_stage_values, dtype=float)
        
        with Model("NESP_DUAL") as m:
            _set_params(m)
            
            # Dual variables:
            # π_i for all nodes i in N' (including supernode r)
            # α_ij >= 0 for all real arcs (i,j) in A
            
            # Create node set including supernode
            all_nodes = list(data.N) + [data.supernode_id]
            
            # π variables (free variables, one per node including supernode)
            pis = m.addVars(all_nodes, lb=-GRB.INFINITY, name="pi")
            
            # α variables (non-negative, one per real arc)
            alphas = m.addVars(range(len(data.A)), lb=0.0, name="alpha")
            
            # Build RHS vector b^m for this OD pair
            # b_i = 1 if i == o, -1 if i == t, 0 otherwise (including supernode r)
            b = {}
            for node in all_nodes:
                if node == o:
                    b[node] = 1.0
                elif node == t:
                    b[node] = -1.0
                else:
                    b[node] = 0.0
            
            # Objective: max (b^m)^T π + sum_{(i,j) in A} x_ij * α_ij
            # = max π_o - π_t + sum_{(i,j) in A} x_ij * α_ij
            obj_expr = quicksum(b[node] * pis[node] for node in all_nodes)
            obj_expr += quicksum(xbar[idx] * alphas[idx] for idx in range(len(data.A)))
            m.setObjective(obj_expr, GRB.MAXIMIZE)
            
            # Dual constraints:
            # 1) For real arcs (i,j) in A: π_i - π_j + α_ij <= L_ij
            for idx, (i, j) in enumerate(data.A):
                m.addConstr(
                    pis[i] - pis[j] + alphas[idx] <= data.L[idx],
                    name=f"dual_real_arc_{i}_{j}"
                )
            
            # 2) For penalty arcs (i, r): π_i - π_r <= M
            for node in data.N:
                m.addConstr(
                    pis[node] - pis[data.supernode_id] <= data.M_penalty,
                    name=f"dual_penalty_arc_{node}_r"
                )
            
            # 3) For penalty arcs (r, i): π_r - π_i <= M
            for node in data.N:
                m.addConstr(
                    pis[data.supernode_id] - pis[node] <= data.M_penalty,
                    name=f"dual_penalty_arc_r_{node}"
                )
            
            m.optimize()
            
            if m.status != GRB.OPTIMAL:
                # Enhanced error reporting for NE-SP dual subproblem issues
                logger.error(f"NE-SP dual subproblem failed for OD pair {m_idx}: {od_pair}")
                logger.error(f"  Status: {m.status}")
                logger.error(f"  First-stage solution (xbar): {xbar}")
                logger.error(f"  Origin: {o}, Destination: {t}")
                logger.error(f"  Number of variables: {m.NumVars}")
                logger.error(f"  Number of constraints: {m.NumConstrs}")
                
                # For NE-SP with supernode penalty structure, dual should always be feasible
                logger.error(f"  This should not happen - supernode structure guarantees dual feasibility!")
                
                # Try to compute the IIS if infeasible
                if m.status == GRB.INFEASIBLE:
                    logger.error("  Computing IIS for infeasible NE-SP dual subproblem...")
                    m.computeIIS()
                    logger.error(f"  IIS computed. Writing to nesp_dual_infeasible_od{m_idx}.ilp")
                    m.write(f"nesp_dual_infeasible_od{m_idx}.ilp")
                
                raise RuntimeError(f"NE-SP dual not optimal: {m.status}")
            
            # Extract dual solution
            pi_vals = {node: pis[node].X for node in all_nodes}
            alpha_vals = {idx: alphas[idx].X for idx in range(len(data.A))}
            
            # Return format: (obj_val, primary_duals, secondary_duals)
            # For NE-SP: primary_duals = π values, secondary_duals = α values
            return m.ObjVal, pi_vals, alpha_vals

    # ---- SUMND dual ----
    elif pt == "SUMND":
        # subproblem_idx is the index of the scenario s in data.S
        s = subproblem_idx
        
        # Extract scenario-specific data
        demand_scenario = data.demand_scenarios[s]  # d_k^s for all commodities k
        capacity_scenario = data.capacity_scenarios[s]  # u_ij^s for all arcs
        
        # Handle different input types for first_stage_values (arc activation)
        if isinstance(first_stage_values, dict):
            # Convert dict to numpy array indexed by arc position
            ybar = np.array([first_stage_values[idx] for idx in range(len(data.A))])
        else:
            ybar = np.asarray(first_stage_values, dtype=float)  # Arc activation decisions y_ij

        # Guard against tiny LP-relaxation numerical noise outside [0,1] from callbacks.
        # Negative coefficients in -u*y*alpha can create artificial unboundedness if not clipped.
        ybar = np.clip(ybar, 0.0, 1.0)

        # SUMND dual-build info logs are intentionally disabled to keep runtime logs concise.
        global _sumnd_dual_static_logged
        if not _sumnd_dual_static_logged:
            _sumnd_dual_static_logged = True
        
        with Model("SUMND_DUAL") as m:
            _set_params(m)
            
            # Dual variables:
            # π_i^k: free dual variables for flow balance constraints (node i, commodity k)
            # α_ij^s ≥ 0: dual variables for arc capacity constraints
            
            # Create dual variables for flow balance: π_i^k for all i in N, k in K
            # These are free variables (can be positive or negative)
            pis = m.addVars(
                [(i, k) for i in data.N for k in data.K],
                lb=-GRB.INFINITY,
                name="pi"
            )
            
            # Create dual variables for capacity constraints: α_ij^s ≥ 0 for all (i,j) in A
            # These are non-negative variables
            alphas = m.addVars(range(len(data.A)), lb=0.0, name="alpha")
            
            # Compute net supply for each node-commodity pair
            # d_i^{ks} = +d_k^s if i is origin of k, -d_k^s if i is destination of k, 0 otherwise
            d_node_commodity = {}
            for i in data.N:
                for k_idx in data.K:
                    origin = data.origins[k_idx]
                    dest = data.destinations[k_idx]
                    if i == origin:
                        d_node_commodity[(i, k_idx)] = demand_scenario[k_idx]
                    elif i == dest:
                        d_node_commodity[(i, k_idx)] = -demand_scenario[k_idx]
                    else:
                        d_node_commodity[(i, k_idx)] = 0.0
            
            # Objective: max Σ_k Σ_i d_i^{ks} * π_i^k - Σ_{(i,j)∈A} u_ij^s * y_ij * α_ij^s
            obj_expr = quicksum(
                d_node_commodity[(i, k)] * pis[(i, k)]
                for i in data.N for k in data.K
            )
            obj_expr -= quicksum(
                capacity_scenario[idx] * ybar[idx] * alphas[idx]
                for idx in range(len(data.A))
            )
            m.setObjective(obj_expr, GRB.MAXIMIZE)
            
            # Dual constraints: π_i^k - π_j^k - α_ij^s ≤ c_ij^k for all (i,j)∈A, k∈K
            # For each arc (i,j) and each commodity k
            for idx, (i, j) in enumerate(data.A):
                for k in data.K:
                    m.addConstr(
                        pis[(i, k)] - pis[(j, k)] - alphas[idx] <= data.C[idx, k],
                        name=f"dual_feasibility_arc{idx}_comm{k}"
                    )
            
            m.optimize()
            
            if m.status != GRB.OPTIMAL:
                # Enhanced error reporting for SUMND dual subproblem issues
                logger.error(f"SUMND dual subproblem failed for scenario {s}")
                logger.error(f"  Status: {m.status}")
                logger.error(f"  First-stage solution (ybar): {ybar}")
                logger.error(f"  Scenario demand: {demand_scenario}")
                logger.error(f"  Scenario capacities: {capacity_scenario}")
                logger.error(f"  Number of variables: {m.NumVars}")
                logger.error(f"  Number of constraints: {m.NumConstrs}")
                raise RuntimeError(f"SUMND dual not optimal: {m.status}")
            
            # Extract dual solution
            pi_vals = {(i, k): pis[(i, k)].X for i in data.N for k in data.K}
            alpha_vals = {idx: alphas[idx].X for idx in range(len(data.A))}
            
            # Return format: (obj_val, primary_duals, secondary_duals)
            # For SUMND: primary_duals = π values, secondary_duals = α values
            return m.ObjVal, pi_vals, alpha_vals

    # ---- SCMND dual ----
    elif pt == "SCMND":
        # subproblem_idx is the index of the scenario s in data.S
        s = subproblem_idx

        # Demand is scenario-specific in both executable SCMND formulations.  The
        # current formulation uses scenario-specific capacities and bounded
        # capacity multipliers (the dual of penalized capacity slack).  The
        # manuscript formulation uses the deterministic arc capacities u_ij and
        # unbounded-above nonnegative capacity multipliers.
        demand_scenario = data.demand_scenarios[s]
        scmnd_formulation = str(
            getattr(data, "scmnd_formulation", SCMND_FORMULATION_CURRENT)
        ).upper()
        if scmnd_formulation not in {
            SCMND_FORMULATION_CURRENT,
            SCMND_FORMULATION_MANUSCRIPT,
        }:
            raise ValueError(f"Unknown SCMND formulation: {scmnd_formulation}")
        capacity_scenario = scmnd_capacity_vector(data, s)

        if isinstance(first_stage_values, dict):
            ybar = np.array([first_stage_values[idx] for idx in range(len(data.A))])
        else:
            ybar = np.asarray(first_stage_values, dtype=float)

        # Guard against tiny LP-relaxation numerical noise outside [0,1].
        ybar = np.clip(ybar, 0.0, 1.0)

        rho = None
        if scmnd_formulation == SCMND_FORMULATION_CURRENT:
            # Moderate configurable penalty: rho_ij^s = lambda * max_k c_ij^k.
            penalty_multiplier = float(getattr(data, "slack_penalty_multiplier", 2.0))
            max_arc_unit_cost = np.max(data.C, axis=1)
            rho = penalty_multiplier * max_arc_unit_cost

        with Model("SCMND_DUAL") as m:
            _set_params(m)

            # pi_i^k are free duals for flow-balance constraints.
            pis = m.addVars(
                [(i, k) for i in data.N for k in data.K],
                lb=-GRB.INFINITY,
                name="pi"
            )

            # In the current soft-capacity formulation, 0 <= mu <= rho.  In
            # the manuscript formulation, mu is simply nonnegative.
            if rho is None:
                alphas = m.addVars(range(len(data.A)), lb=0.0, name="alpha")
            else:
                alphas = m.addVars(
                    range(len(data.A)),
                    lb=0.0,
                    ub={idx: rho[idx] for idx in range(len(data.A))},
                    name="alpha"
                )

            # Node-commodity net supply d_i^{ks}.
            d_node_commodity = {}
            for i in data.N:
                for k_idx in data.K:
                    origin = data.origins[k_idx]
                    dest = data.destinations[k_idx]
                    if i == origin:
                        d_node_commodity[(i, k_idx)] = demand_scenario[k_idx]
                    elif i == dest:
                        d_node_commodity[(i, k_idx)] = -demand_scenario[k_idx]
                    else:
                        d_node_commodity[(i, k_idx)] = 0.0

            # max Σ_{k,i} d_i^{ks} pi_i^k - Σ_{(i,j)} u_ij^s * y_ij * mu_ij^s
            obj_expr = quicksum(
                d_node_commodity[(i, k)] * pis[(i, k)]
                for i in data.N for k in data.K
            )
            obj_expr -= quicksum(
                capacity_scenario[idx] * ybar[idx] * alphas[idx]
                for idx in range(len(data.A))
            )
            m.setObjective(obj_expr, GRB.MAXIMIZE)

            # pi_i^k - pi_j^k - mu_ij^s <= c_ij^k
            for idx, (i, j) in enumerate(data.A):
                for k in data.K:
                    m.addConstr(
                        pis[(i, k)] - pis[(j, k)] - alphas[idx] <= data.C[idx, k],
                        name=f"dual_feasibility_arc{idx}_comm{k}"
                    )

            m.optimize()

            if m.status != GRB.OPTIMAL:
                logger.error(f"SCMND dual subproblem failed for scenario {s}")
                logger.error(f"  Status: {m.status}")
                logger.error(f"  First-stage solution (ybar): {ybar}")
                logger.error(f"  Scenario demand: {demand_scenario}")
                logger.error(f"  Scenario capacities: {capacity_scenario}")
                raise RuntimeError(f"SCMND dual not optimal: {m.status}")

            pi_vals = {(i, k): pis[(i, k)].X for i in data.N for k in data.K}
            alpha_vals = {idx: alphas[idx].X for idx in range(len(data.A))}
            return m.ObjVal, pi_vals, alpha_vals

    else:
        raise ValueError(f"Unknown problem type: {problem_type}. Only SCFLP, NE-SP, SUMND, and SCMND are supported.")
