"""
Stochastic Uncapacitated Multicommodity Network Design (SUMND) Data Generator

This module generates stochastic instances from deterministic MMCF benchmark files
(families C, C+, and R) using various generation methods, with primary support for
the Merve-style with covariates approach.

The generator creates stochastic demand and optionally stochastic capacity scenarios,
computes scenario covariates via SVD, and outputs unified JSON files compatible with
the ML-augmented Benders Decomposition framework.

Author: Research Project - Benders Integration with ML
Date: March 2026
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Arc:
    """
    Represents a directed arc in the network.
    
    Field order matches the input file format and must be preserved
    for correct scenario matrix alignment.
    """
    tail: int              # i: tail node
    head: int              # j: head node
    unit_cost: float       # c_ij: variable transportation cost
    capacity: float        # u_ij: deterministic base capacity
    fixed_cost: float      # f_ij: fixed activation cost
    extra1: float          # extra field 1 (stored but not essential)
    extra2: float          # extra field 2 (stored but not essential)
    index: int             # Position in the arc list (0-indexed)


@dataclass
class Commodity:
    """
    Represents a commodity (OD pair with demand).
    
    Field order matches the input file format and must be preserved
    for correct scenario matrix alignment.
    """
    origin: int            # o_k: origin node
    destination: int       # t_k: destination node
    demand: float          # d_k: deterministic base demand
    index: int             # Position in the commodity list (0-indexed)


@dataclass
class DeterministicInstance:
    """
    Complete representation of a deterministic MMCF instance.
    
    Arc and commodity orders must be preserved exactly as they appear
    in the input file to ensure correct scenario matrix indexing.
    """
    n_nodes: int
    n_arcs: int
    n_commodities: int
    arcs: List[Arc]        # Ordered list of arcs (preserve file order)
    commodities: List[Commodity]  # Ordered list of commodities (preserve file order)
    source_file: Optional[str] = None


@dataclass
class StochasticScenarios:
    """
    Container for generated stochastic scenarios.
    
    Matrix dimensions:
    - demand_scenarios: (n_scenarios, n_commodities)
    - capacity_scenarios: (n_scenarios, n_arcs)
    - covariates: (n_scenarios, n_covariates) if generated
    """
    demand_scenarios: np.ndarray      # Shape: (S, K)
    capacity_scenarios: np.ndarray    # Shape: (S, A)
    probabilities: np.ndarray         # Shape: (S,)
    covariates: Optional[np.ndarray] = None  # Shape: (S, n_cov)
    
    # Generation metadata
    sigma_demand: Optional[np.ndarray] = None     # Shape: (K,)
    sigma_capacity: Optional[np.ndarray] = None   # Shape: (A,)


# ============================================================================
# DETERMINISTIC FILE PARSER
# ============================================================================

def parse_deterministic_instance(file_path: Union[str, Path]) -> DeterministicInstance:
    """
    Parse a deterministic MMCF instance file (C, C+, or R family format).
    
    File structure:
    - Line 1: n m K (nodes, arcs, commodities)
    - Lines 2 to m+1: arc data (i j c u f extra1 extra2)
    - Lines m+2 to m+K+1: commodity data (o t d)
    
    CRITICAL: Preserves exact arc and commodity order from the file.
    This order determines scenario matrix indexing.
    
    Args:
        file_path: Path to the deterministic instance file
        
    Returns:
        DeterministicInstance object with all data
        
    Raises:
        ValueError: If file format is invalid
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Instance file not found: {file_path}")
    
    logger.info(f"Parsing deterministic instance: {file_path.name}")
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Skip comment lines (if any)
    lines = [line for line in lines if not line.startswith('#') and not line.startswith('MULTIGEN')]
    
    if len(lines) < 1:
        raise ValueError(f"File {file_path} is empty or has no valid data")
    
    # Parse header
    header = lines[0].split()
    if len(header) < 3:
        raise ValueError(f"Invalid header format in {file_path}: expected 'n m K'")
    
    n_nodes = int(header[0])
    n_arcs = int(header[1])
    n_commodities = int(header[2])
    
    logger.debug(f"Header: nodes={n_nodes}, arcs={n_arcs}, commodities={n_commodities}")
    
    # Verify sufficient lines
    expected_lines = 1 + n_arcs + n_commodities
    if len(lines) < expected_lines:
        raise ValueError(
            f"Insufficient data in {file_path}: "
            f"expected {expected_lines} lines, found {len(lines)}"
        )
    
    # Parse arcs (preserve exact order)
    arcs = []
    for arc_idx in range(n_arcs):
        line_idx = 1 + arc_idx
        parts = lines[line_idx].split()
        
        if len(parts) < 7:
            raise ValueError(f"Invalid arc format at line {line_idx + 1}: {lines[line_idx]}")
        
        arc = Arc(
            tail=int(parts[0]),
            head=int(parts[1]),
            unit_cost=float(parts[2]),
            capacity=float(parts[3]),
            fixed_cost=float(parts[4]),
            extra1=float(parts[5]),
            extra2=float(parts[6]),
            index=arc_idx
        )
        arcs.append(arc)
    
    logger.debug(f"Parsed {len(arcs)} arcs")
    
    # Parse commodities (preserve exact order)
    commodities = []
    for comm_idx in range(n_commodities):
        line_idx = 1 + n_arcs + comm_idx
        parts = lines[line_idx].split()
        
        if len(parts) < 3:
            raise ValueError(
                f"Invalid commodity format at line {line_idx + 1}: {lines[line_idx]}"
            )
        
        commodity = Commodity(
            origin=int(parts[0]),
            destination=int(parts[1]),
            demand=float(parts[2]),
            index=comm_idx
        )
        commodities.append(commodity)
    
    logger.debug(f"Parsed {len(commodities)} commodities")
    
    instance = DeterministicInstance(
        n_nodes=n_nodes,
        n_arcs=n_arcs,
        n_commodities=n_commodities,
        arcs=arcs,
        commodities=commodities,
        source_file=str(file_path)
    )
    
    logger.info(
        f"Successfully parsed {file_path.name}: "
        f"{n_nodes} nodes, {n_arcs} arcs, {n_commodities} commodities"
    )
    
    return instance


# ============================================================================
# COVARIATE GENERATION (SVD-BASED)
# ============================================================================

def generate_covariates_svd(
    scenario_matrix: np.ndarray,
    n_covariates: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate scenario covariates using Singular Value Decomposition (SVD).
    
    This implements the Merve-style covariate generation approach:
    1. Center the scenario matrix columns
    2. Compute SVD: X_centered = U Σ V^T
    3. Use first n_covariates left singular vectors as covariates
    
    Args:
        scenario_matrix: Shape (n_scenarios, n_features)
            For Demand_Only: features are commodities
            For Both: features are [commodities | arcs]
        n_covariates: Number of covariate dimensions to extract
        
    Returns:
        covariates: Shape (n_scenarios, n_covariates)
        info: Dictionary with SVD statistics
    """
    n_scenarios, n_features = scenario_matrix.shape
    
    # Ensure we don't request more covariates than possible
    max_covariates = min(n_scenarios, n_features)
    if n_covariates > max_covariates:
        logger.warning(
            f"Requested {n_covariates} covariates but only {max_covariates} available. "
            f"Using {max_covariates} instead."
        )
        n_covariates = max_covariates
    
    # Center the columns (each feature has zero mean across scenarios)
    col_means = np.mean(scenario_matrix, axis=0)
    centered_matrix = scenario_matrix - col_means
    
    # Compute SVD: X_centered = U Σ V^T
    # U: left singular vectors (n_scenarios, min(n_scenarios, n_features))
    # Σ: singular values
    # V^T: right singular vectors
    U, singular_values, Vt = np.linalg.svd(centered_matrix, full_matrices=False)
    
    # Extract first n_covariates left singular vectors
    # These are the principal components in scenario space
    covariates = U[:, :n_covariates]
    
    # Compute explained variance ratios
    explained_variance = singular_values ** 2
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    info = {
        'n_covariates': n_covariates,
        'n_scenarios': n_scenarios,
        'n_features': n_features,
        'singular_values': singular_values[:n_covariates].tolist(),
        'explained_variance_ratio': explained_variance_ratio[:n_covariates].tolist(),
        'cumulative_variance_ratio': cumulative_variance[:n_covariates].tolist(),
        'total_variance_explained': float(cumulative_variance[n_covariates - 1]) if n_covariates > 0 else 0.0
    }
    
    logger.debug(
        f"Generated {n_covariates} covariates explaining "
        f"{info['total_variance_explained']:.2%} of variance"
    )
    
    return covariates, info


# ============================================================================
# MERVE WITH COVARIATE METHOD (FULLY IMPLEMENTED)
# ============================================================================

def generate_merve_with_covariate(
    det_instance: DeterministicInstance,
    n_scenarios: int,
    scenario_probabilities: Optional[np.ndarray],
    stochastic_combination_type: str,
    sigma_L: float,
    sigma_U: float,
    generate_covariates: bool,
    n_covariates: int,
    include_deterministic_first: bool,
    seed: Optional[int],
    **kwargs
) -> StochasticScenarios:
    """
    Generate stochastic scenarios using the Merve-style with covariates method.
    
    This method generates scenarios with scenario-independent standard deviations
    drawn from a uniform distribution, then samples from normal distributions.
    
    Generation Process:
    
    For Demand:
    1. Draw σ_d[k] ~ Uniform(σ_L * μ_d[k], σ_U * μ_d[k]) for each commodity k
    2. For each scenario s, draw d[s,k] ~ Normal(μ_d[k], σ_d[k]²)
    3. Clip negative values to zero
    
    For Capacity (if Both_Demand_and_Capacity):
    1. Draw σ_u[a] ~ Uniform(σ_L * μ_u[a], σ_U * μ_u[a]) for each arc a
    2. For each scenario s, draw u[s,a] ~ Normal(μ_u[a], σ_u[a]²)
    3. Clip negative values to zero
    
    If include_deterministic_first=True, scenario 0 equals deterministic base data.
    
    Args:
        det_instance: Deterministic base instance
        n_scenarios: Number of scenarios to generate
        scenario_probabilities: Scenario probabilities (or None for uniform)
        stochastic_combination_type: "Demand_Only" or "Both_Demand_and_Capacity"
        sigma_L: Lower bound for coefficient of variation
        sigma_U: Upper bound for coefficient of variation
        generate_covariates: Whether to generate covariates via SVD
        n_covariates: Number of covariate dimensions
        include_deterministic_first: If True, scenario 0 = deterministic data
        seed: Random seed for reproducibility
        **kwargs: Additional method-specific parameters (ignored)
        
    Returns:
        StochasticScenarios object with demand, capacity, probabilities, and covariates
    """
    logger.info(f"Generating scenarios using Merve_with_Covariate method")
    logger.info(f"  Scenarios: {n_scenarios}")
    logger.info(f"  Type: {stochastic_combination_type}")
    logger.info(f"  σ range: [{sigma_L}, {sigma_U}]")
    logger.info(f"  Include deterministic first: {include_deterministic_first}")
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        logger.debug(f"Random seed set to {seed}")
    
    # Extract deterministic base data
    mu_d = np.array([c.demand for c in det_instance.commodities])  # Shape: (K,)
    mu_u = np.array([a.capacity for a in det_instance.arcs])       # Shape: (A,)
    
    n_commodities = det_instance.n_commodities
    n_arcs = det_instance.n_arcs
    
    # Validate parameters
    if sigma_L < 0 or sigma_U < sigma_L:
        raise ValueError(f"Invalid sigma range: [{sigma_L}, {sigma_U}]")
    
    # Initialize scenario probabilities
    if scenario_probabilities is None:
        probabilities = np.ones(n_scenarios) / n_scenarios
        logger.debug(f"Using uniform probabilities: {1/n_scenarios:.6f}")
    else:
        if len(scenario_probabilities) != n_scenarios:
            raise ValueError(
                f"Probability array length {len(scenario_probabilities)} "
                f"doesn't match n_scenarios {n_scenarios}"
            )
        probabilities = np.array(scenario_probabilities)
        if not np.isclose(probabilities.sum(), 1.0):
            logger.warning(f"Probabilities sum to {probabilities.sum():.6f}, normalizing")
            probabilities = probabilities / probabilities.sum()
    
    # ========================================================================
    # DEMAND GENERATION (always generated)
    # ========================================================================
    
    logger.debug("Generating stochastic demand scenarios...")
    
    # Draw scenario-independent standard deviations for each commodity
    sigma_d = np.random.uniform(
        low=sigma_L * mu_d,
        high=sigma_U * mu_d,
        size=n_commodities
    )
    
    # Generate demand scenarios
    demand_scenarios = np.zeros((n_scenarios, n_commodities))
    
    for s in range(n_scenarios):
        raw_demand = np.random.normal(loc=mu_d, scale=sigma_d)
        demand_scenarios[s, :] = np.maximum(0.0, raw_demand)  # Clip negatives
    
    # If include_deterministic_first, override scenario 0
    if include_deterministic_first:
        demand_scenarios[0, :] = mu_d
        logger.debug("Scenario 0 set to deterministic demand")
    
    # Count clipped values
    n_demand_clipped = np.sum(demand_scenarios == 0.0) - np.sum(mu_d == 0.0) * n_scenarios
    if n_demand_clipped > 0:
        logger.debug(f"Clipped {n_demand_clipped} negative demand values to zero")
    
    # ========================================================================
    # CAPACITY GENERATION (depends on stochastic_combination_type)
    # ========================================================================
    
    if stochastic_combination_type == "Both_Demand_and_Capacity":
        logger.debug("Generating stochastic capacity scenarios...")
        
        # Draw scenario-independent standard deviations for each arc
        sigma_u = np.random.uniform(
            low=sigma_L * mu_u,
            high=sigma_U * mu_u,
            size=n_arcs
        )
        
        # Generate capacity scenarios
        capacity_scenarios = np.zeros((n_scenarios, n_arcs))
        
        for s in range(n_scenarios):
            raw_capacity = np.random.normal(loc=mu_u, scale=sigma_u)
            capacity_scenarios[s, :] = np.maximum(0.0, raw_capacity)  # Clip negatives
        
        # If include_deterministic_first, override scenario 0
        if include_deterministic_first:
            capacity_scenarios[0, :] = mu_u
            logger.debug("Scenario 0 set to deterministic capacity")
        
        # Count clipped values
        n_capacity_clipped = np.sum(capacity_scenarios == 0.0) - np.sum(mu_u == 0.0) * n_scenarios
        if n_capacity_clipped > 0:
            logger.debug(f"Clipped {n_capacity_clipped} negative capacity values to zero")
    
    elif stochastic_combination_type == "Demand_Only":
        logger.debug("Replicating deterministic capacities across scenarios")
        
        # Replicate deterministic capacities for all scenarios
        capacity_scenarios = np.tile(mu_u, (n_scenarios, 1))
        sigma_u = None
    
    else:
        raise ValueError(
            f"Invalid stochastic_combination_type: {stochastic_combination_type}. "
            f"Must be 'Demand_Only' or 'Both_Demand_and_Capacity'"
        )
    
    # ========================================================================
    # COVARIATE GENERATION (via SVD)
    # ========================================================================
    
    covariates = None
    covariate_info = None
    
    if generate_covariates:
        logger.debug("Generating covariates via SVD...")
        
        if stochastic_combination_type == "Demand_Only":
            # Use demand matrix D: (n_scenarios, n_commodities)
            scenario_matrix = demand_scenarios
            logger.debug(f"Using demand matrix for covariates: {scenario_matrix.shape}")
        
        elif stochastic_combination_type == "Both_Demand_and_Capacity":
            # Use combined matrix [D | U]: (n_scenarios, n_commodities + n_arcs)
            scenario_matrix = np.hstack([demand_scenarios, capacity_scenarios])
            logger.debug(
                f"Using combined [demand | capacity] matrix for covariates: "
                f"{scenario_matrix.shape}"
            )
        
        covariates, covariate_info = generate_covariates_svd(
            scenario_matrix=scenario_matrix,
            n_covariates=n_covariates
        )
        
        logger.info(
            f"Generated {covariate_info['n_covariates']} covariates "
            f"(variance explained: {covariate_info['total_variance_explained']:.2%})"
        )
    
    # ========================================================================
    # RETURN STOCHASTIC SCENARIOS OBJECT
    # ========================================================================
    
    scenarios = StochasticScenarios(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios,
        probabilities=probabilities,
        covariates=covariates,
        sigma_demand=sigma_d,
        sigma_capacity=sigma_u
    )
    
    logger.info(
        f"Successfully generated {n_scenarios} scenarios "
        f"({n_commodities} commodities, {n_arcs} arcs)"
    )
    
    return scenarios


# ============================================================================
# METHOD STUBS (TO BE IMPLEMENTED)
# ============================================================================

def generate_method_b(
    det_instance: DeterministicInstance,
    n_scenarios: int,
    scenario_probabilities: Optional[np.ndarray],
    stochastic_combination_type: str,
    seed: Optional[int],
    **kwargs
) -> StochasticScenarios:
    """
    Generate stochastic scenarios using Method B.
    
    TODO: Implement Method B generation logic.
    
    This is a placeholder function. When implementing:
    - Define the specific generation algorithm for Method B
    - Document the mathematical approach
    - Ensure compatibility with the unified output schema
    - Support both Demand_Only and Both_Demand_and_Capacity types
    
    Args:
        det_instance: Deterministic base instance
        n_scenarios: Number of scenarios to generate
        scenario_probabilities: Scenario probabilities (or None for uniform)
        stochastic_combination_type: "Demand_Only" or "Both_Demand_and_Capacity"
        seed: Random seed for reproducibility
        **kwargs: Method-specific parameters
        
    Returns:
        StochasticScenarios object
        
    Raises:
        NotImplementedError: Method not yet implemented
    """
    raise NotImplementedError(
        "Method B generation is not yet implemented. "
        "Please use 'Merve_with_Covariate' or implement this method."
    )


def generate_method_c(
    det_instance: DeterministicInstance,
    n_scenarios: int,
    scenario_probabilities: Optional[np.ndarray],
    stochastic_combination_type: str,
    seed: Optional[int],
    **kwargs
) -> StochasticScenarios:
    """
    Generate stochastic scenarios using Method C.
    
    TODO: Implement Method C generation logic.
    
    This is a placeholder function. When implementing:
    - Define the specific generation algorithm for Method C
    - Document the mathematical approach
    - Ensure compatibility with the unified output schema
    - Support both Demand_Only and Both_Demand_and_Capacity types
    
    Args:
        det_instance: Deterministic base instance
        n_scenarios: Number of scenarios to generate
        scenario_probabilities: Scenario probabilities (or None for uniform)
        stochastic_combination_type: "Demand_Only" or "Both_Demand_and_Capacity"
        seed: Random seed for reproducibility
        **kwargs: Method-specific parameters
        
    Returns:
        StochasticScenarios object
        
    Raises:
        NotImplementedError: Method not yet implemented
    """
    raise NotImplementedError(
        "Method C generation is not yet implemented. "
        "Please use 'Merve_with_Covariate' or implement this method."
    )


# ============================================================================
# STATISTICS CALCULATION
# ============================================================================

def calculate_statistics(
    det_instance: DeterministicInstance,
    stoch_scenarios: StochasticScenarios,
    stochastic_combination_type: str
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for generated stochastic scenarios.
    
    Computes summary statistics including means, standard deviations,
    coefficients of variation, min/max values, and zero counts.
    
    Args:
        det_instance: Deterministic base instance
        stoch_scenarios: Generated stochastic scenarios
        stochastic_combination_type: Generation type
        
    Returns:
        Dictionary with comprehensive statistics
    """
    stats = {
        'generation_timestamp': datetime.now().isoformat(),
        'n_scenarios': len(stoch_scenarios.probabilities),
        'n_commodities': det_instance.n_commodities,
        'n_arcs': det_instance.n_arcs,
    }
    
    # Demand statistics
    demand_stats = {
        'mean_per_commodity': np.mean(stoch_scenarios.demand_scenarios, axis=0).tolist(),
        'std_per_commodity': np.std(stoch_scenarios.demand_scenarios, axis=0).tolist(),
        'min_per_commodity': np.min(stoch_scenarios.demand_scenarios, axis=0).tolist(),
        'max_per_commodity': np.max(stoch_scenarios.demand_scenarios, axis=0).tolist(),
        'overall_mean': float(np.mean(stoch_scenarios.demand_scenarios)),
        'overall_std': float(np.std(stoch_scenarios.demand_scenarios)),
        'n_zeros': int(np.sum(stoch_scenarios.demand_scenarios == 0.0)),
    }
    
    # Add CV if sigma_demand available
    if stoch_scenarios.sigma_demand is not None:
        mu_d = np.array([c.demand for c in det_instance.commodities])
        cv_demand = stoch_scenarios.sigma_demand / (mu_d + 1e-10)
        demand_stats['cv_per_commodity'] = cv_demand.tolist()
        demand_stats['cv_mean'] = float(np.mean(cv_demand))
    
    stats['demand'] = demand_stats
    
    # Capacity statistics
    capacity_stats = {
        'mean_per_arc': np.mean(stoch_scenarios.capacity_scenarios, axis=0).tolist(),
        'std_per_arc': np.std(stoch_scenarios.capacity_scenarios, axis=0).tolist(),
        'min_per_arc': np.min(stoch_scenarios.capacity_scenarios, axis=0).tolist(),
        'max_per_arc': np.max(stoch_scenarios.capacity_scenarios, axis=0).tolist(),
        'overall_mean': float(np.mean(stoch_scenarios.capacity_scenarios)),
        'overall_std': float(np.std(stoch_scenarios.capacity_scenarios)),
        'n_zeros': int(np.sum(stoch_scenarios.capacity_scenarios == 0.0)),
        'is_stochastic': stochastic_combination_type == "Both_Demand_and_Capacity",
    }
    
    # Add CV if sigma_capacity available
    if stoch_scenarios.sigma_capacity is not None:
        mu_u = np.array([a.capacity for a in det_instance.arcs])
        cv_capacity = stoch_scenarios.sigma_capacity / (mu_u + 1e-10)
        capacity_stats['cv_per_arc'] = cv_capacity.tolist()
        capacity_stats['cv_mean'] = float(np.mean(cv_capacity))
    
    stats['capacity'] = capacity_stats
    
    # Covariate statistics (if generated)
    if stoch_scenarios.covariates is not None:
        cov_stats = {
            'n_covariates': stoch_scenarios.covariates.shape[1],
            'mean_per_covariate': np.mean(stoch_scenarios.covariates, axis=0).tolist(),
            'std_per_covariate': np.std(stoch_scenarios.covariates, axis=0).tolist(),
        }
        stats['covariates'] = cov_stats
    
    logger.debug(f"Calculated statistics for {stats['n_scenarios']} scenarios")
    
    return stats


# ============================================================================
# JSON OUTPUT GENERATION
# ============================================================================

def create_output_json(
    det_instance: DeterministicInstance,
    stoch_scenarios: StochasticScenarios,
    generation_metadata: Dict[str, Any],
    statistics: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create unified JSON output structure for stochastic instance.
    
    This schema is designed to be:
    - Self-describing: all metadata included
    - Unified: same structure for all generation methods and types
    - Reader-friendly: easy to parse and validate
    - Extensible: can add new fields without breaking compatibility
    
    Args:
        det_instance: Deterministic base instance
        stoch_scenarios: Generated stochastic scenarios
        generation_metadata: Metadata about generation process
        statistics: Optional statistics dictionary
        
    Returns:
        Dictionary ready for JSON serialization
    """
    output = {
        'format': {
            'name': 'SUMND_Stochastic_Instance',
            'version': '1.0',
            'description': 'Stochastic Uncapacitated Multicommodity Network Design instance',
            'schema_date': '2026-03-11'
        },
        
        'metadata': {
            **generation_metadata,
            'generation_timestamp': datetime.now().isoformat(),
        },
        
        'deterministic': {
            'n_nodes': det_instance.n_nodes,
            'n_arcs': det_instance.n_arcs,
            'n_commodities': det_instance.n_commodities,
            
            # Store arcs in exact file order (preserve indexing)
            'arcs': [
                {
                    'index': a.index,
                    'tail': a.tail,
                    'head': a.head,
                    'unit_cost': a.unit_cost,
                    'capacity': a.capacity,
                    'fixed_cost': a.fixed_cost,
                    'extra1': a.extra1,
                    'extra2': a.extra2,
                }
                for a in det_instance.arcs
            ],
            
            # Store commodities in exact file order (preserve indexing)
            'commodities': [
                {
                    'index': c.index,
                    'origin': c.origin,
                    'destination': c.destination,
                    'demand': c.demand,
                }
                for c in det_instance.commodities
            ],
        },
        
        'scenarios': {
            'n_scenarios': len(stoch_scenarios.probabilities),
            'probabilities': stoch_scenarios.probabilities.tolist(),
            
            # Demand scenarios: shape (n_scenarios, n_commodities)
            # demand_scenarios[s][k] corresponds to commodity k in scenario s
            'demand_scenarios': stoch_scenarios.demand_scenarios.tolist(),
            
            # Capacity scenarios: shape (n_scenarios, n_arcs)
            # capacity_scenarios[s][a] corresponds to arc a in scenario s
            # Even in Demand_Only mode, this contains replicated deterministic capacities
            'capacity_scenarios': stoch_scenarios.capacity_scenarios.tolist(),
            
            # Covariates (if generated): shape (n_scenarios, n_covariates)
            'covariates': (
                stoch_scenarios.covariates.tolist()
                if stoch_scenarios.covariates is not None
                else None
            ),
        },
    }
    
    # Add statistics if provided
    if statistics is not None:
        output['statistics'] = statistics
    
    logger.debug("Created unified JSON output structure")
    
    return output


def write_json_output(
    output_data: Dict[str, Any],
    output_path: Union[str, Path],
    overwrite: bool = False,
    indent: int = 2
) -> None:
    """
    Write JSON output to file with validation.
    
    Args:
        output_data: Dictionary to serialize
        output_path: Output file path
        overwrite: Whether to overwrite existing files
        indent: JSON indentation level (2 for readable, None for compact)
        
    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    output_path = Path(output_path)
    
    # Check for existing file
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            f"Use overwrite=True to replace."
        )
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=indent)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Wrote output to {output_path} ({file_size_mb:.2f} MB)")


# ============================================================================
# MAIN GENERATOR FUNCTION
# ============================================================================

def generate_stochastic_instances(
    input_deterministic_instances: Union[str, Path, List[Union[str, Path]]],
    data_generation_method: str = "Merve_with_Covariate",
    stochastic_combination_type: str = "Demand_Only",
    compute_statistics: bool = True,
    output_dir: Union[str, Path] = "stochastic_out",
    output_format: str = "json",
    n_scenarios: int = 50,
    scenario_probabilities: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    sigma_L: float = 0.05,
    sigma_U: float = 0.30,
    generate_covariates: bool = True,
    n_covariates: int = 5,
    include_deterministic_first: bool = True,
    overwrite: bool = False,
    output_suffix: str = ".stoch.json",
    **method_params
) -> List[Path]:
    """
    Generate stochastic instances from deterministic MMCF files.
    
    This is the main entry point for stochastic instance generation.
    It processes one or more deterministic files, applies the specified
    generation method, and outputs unified JSON files.
    
    Args:
        input_deterministic_instances: Single path or list of paths to deterministic files
        data_generation_method: Generation method
            - "Merve_with_Covariate": Fully implemented Merve-style method
            - "Method_B": Placeholder (not yet implemented)
            - "Method_C": Placeholder (not yet implemented)
        stochastic_combination_type: Type of stochasticity
            - "Demand_Only": Only demand is stochastic, capacities deterministic
            - "Both_Demand_and_Capacity": Both demand and capacity are stochastic
        compute_statistics: Whether to compute and include statistics
        output_dir: Directory for output files
        output_format: Output format (currently only "json" supported)
        n_scenarios: Number of scenarios to generate
        scenario_probabilities: Scenario probabilities (None for uniform)
        seed: Random seed for reproducibility
        sigma_L: Lower bound for coefficient of variation (Merve method)
        sigma_U: Upper bound for coefficient of variation (Merve method)
        generate_covariates: Whether to generate covariates via SVD
        n_covariates: Number of covariate dimensions
        include_deterministic_first: If True, scenario 0 = deterministic data
        overwrite: Whether to overwrite existing output files
        output_suffix: Suffix for output files (e.g., ".stoch.json")
        **method_params: Additional method-specific parameters
        
    Returns:
        List of output file paths
        
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If input files don't exist
        NotImplementedError: If unsupported method is specified
    """
    # ========================================================================
    # VALIDATE INPUTS
    # ========================================================================
    
    # Validate method
    valid_methods = ["Merve_with_Covariate", "Method_B", "Method_C"]
    if data_generation_method not in valid_methods:
        raise ValueError(
            f"Invalid data_generation_method: {data_generation_method}. "
            f"Must be one of {valid_methods}"
        )
    
    # Validate stochastic type
    valid_types = ["Demand_Only", "Both_Demand_and_Capacity"]
    if stochastic_combination_type not in valid_types:
        raise ValueError(
            f"Invalid stochastic_combination_type: {stochastic_combination_type}. "
            f"Must be one of {valid_types}"
        )
    
    # Validate output format
    if output_format != "json":
        raise ValueError(f"Only 'json' output format is currently supported")
    
    # Convert input to list
    if isinstance(input_deterministic_instances, (str, Path)):
        input_files = [Path(input_deterministic_instances)]
    else:
        input_files = [Path(f) for f in input_deterministic_instances]
    
    # Validate input files exist
    for input_file in input_files:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("STOCHASTIC INSTANCE GENERATION - SUMND")
    logger.info("=" * 70)
    logger.info(f"Method: {data_generation_method}")
    logger.info(f"Type: {stochastic_combination_type}")
    logger.info(f"Input files: {len(input_files)}")
    logger.info(f"Scenarios: {n_scenarios}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
    # ========================================================================
    # SELECT GENERATION METHOD
    # ========================================================================
    
    if data_generation_method == "Merve_with_Covariate":
        generator_func = generate_merve_with_covariate
    elif data_generation_method == "Method_B":
        generator_func = generate_method_b
    elif data_generation_method == "Method_C":
        generator_func = generate_method_c
    else:
        raise ValueError(f"Unsupported method: {data_generation_method}")
    
    # ========================================================================
    # PROCESS EACH INPUT FILE
    # ========================================================================
    
    output_paths = []
    
    for input_idx, input_file in enumerate(input_files, 1):
        logger.info(f"\n[{input_idx}/{len(input_files)}] Processing: {input_file.name}")
        
        try:
            # Parse deterministic instance
            det_instance = parse_deterministic_instance(input_file)
            
            # Prepare generation metadata
            generation_metadata = {
                'generation_method': data_generation_method,
                'stochastic_combination_type': stochastic_combination_type,
                'n_scenarios': n_scenarios,
                'seed': seed,
                'sigma_L': sigma_L,
                'sigma_U': sigma_U,
                'generate_covariates': generate_covariates,
                'n_covariates': n_covariates if generate_covariates else None,
                'include_deterministic_first': include_deterministic_first,
                'source_deterministic_file': str(input_file),
                'source_file_name': input_file.name,
                'output_directory': str(output_dir),
            }
            
            # Generate stochastic scenarios
            stoch_scenarios = generator_func(
                det_instance=det_instance,
                n_scenarios=n_scenarios,
                scenario_probabilities=scenario_probabilities,
                stochastic_combination_type=stochastic_combination_type,
                sigma_L=sigma_L,
                sigma_U=sigma_U,
                generate_covariates=generate_covariates,
                n_covariates=n_covariates,
                include_deterministic_first=include_deterministic_first,
                seed=seed,
                **method_params
            )
            
            # Calculate statistics if requested
            statistics = None
            if compute_statistics:
                statistics = calculate_statistics(
                    det_instance=det_instance,
                    stoch_scenarios=stoch_scenarios,
                    stochastic_combination_type=stochastic_combination_type
                )
            
            # Create output JSON
            output_data = create_output_json(
                det_instance=det_instance,
                stoch_scenarios=stoch_scenarios,
                generation_metadata=generation_metadata,
                statistics=statistics
            )
            
            # Determine output filename
            output_filename = input_file.stem + output_suffix
            output_path = output_dir / output_filename
            
            # Write output
            write_json_output(
                output_data=output_data,
                output_path=output_path,
                overwrite=overwrite,
                indent=2
            )
            
            output_paths.append(output_path)
            logger.info(f"✓ Successfully generated: {output_path.name}")
        
        except Exception as e:
            logger.error(f"✗ Failed to process {input_file.name}: {e}")
            raise
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Successfully generated {len(output_paths)} stochastic instances")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
    return output_paths


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """
    Command-line interface for stochastic instance generation.
    
    All paths can be either absolute or relative to PROJECT_ROOT:
    - Relative paths are automatically resolved from PROJECT_ROOT
    - Absolute paths are used as-is
    
    Example usage (relative paths):
        python SUMND_data_generation.py --input_dir DATA/NE-SP/R \\
            --output_dir DATA/SUMND/R_stochastic \\
            --method Merve_with_Covariate \\
            --type Demand_Only \\
            --scenarios 50 \\
            --seed 42
    
    Example usage (absolute paths):
        python SUMND_data_generation.py --input_dir /full/path/to/DATA/NE-SP/R \\
            --output_dir /full/path/to/DATA/SUMND/R_stochastic \\
            --method Merve_with_Covariate
    """
    import argparse
    
    # ========== DYNAMIC PROJECT ROOT DETECTION ==========
    # Detect PROJECT_ROOT: this script is in src/utils/, so PROJECT_ROOT is 2 levels up
    SCRIPT_DIR = Path(__file__).parent.parent.parent.resolve()  # Go up from src/utils -> src -> PROJECT_ROOT
    PROJECT_ROOT = str(SCRIPT_DIR)
    
    parser = argparse.ArgumentParser(
        description='Generate stochastic SUMND instances from deterministic MMCF files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing deterministic instance files (relative to PROJECT_ROOT or absolute)')
    parser.add_argument('--input_files', nargs='+', type=str,
                        help='Specific input files to process')
    parser.add_argument('--output_dir', type=str, default='stochastic_out',
                        help='Output directory for generated instances (relative to PROJECT_ROOT or absolute)')
    
    # Generation parameters
    parser.add_argument('--method', type=str, default='Merve_with_Covariate',
                        choices=['Merve_with_Covariate', 'Method_B', 'Method_C'],
                        help='Data generation method')
    parser.add_argument('--type', type=str, default='Demand_Only',
                        choices=['Demand_Only', 'Both_Demand_and_Capacity'],
                        help='Stochastic combination type')
    parser.add_argument('--scenarios', type=int, default=50,
                        help='Number of scenarios to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Method parameters
    parser.add_argument('--sigma_L', type=float, default=0.05,
                        help='Lower bound for coefficient of variation')
    parser.add_argument('--sigma_U', type=float, default=0.30,
                        help='Upper bound for coefficient of variation')
    parser.add_argument('--n_covariates', type=int, default=5,
                        help='Number of covariate dimensions')
    
    # Flags
    parser.add_argument('--no_covariates', action='store_true',
                        help='Disable covariate generation')
    parser.add_argument('--no_deterministic_first', action='store_true',
                        help='Do not include deterministic scenario as scenario 0')
    parser.add_argument('--no_statistics', action='store_true',
                        help='Do not calculate statistics')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')
    
    # Output format
    parser.add_argument('--suffix', type=str, default='.stoch.json',
                        help='Suffix for output files')
    
    args = parser.parse_args()
    
    # ========== PATH RESOLUTION WITH PROJECT_ROOT ==========
    # Convert relative paths to absolute paths using PROJECT_ROOT
    def resolve_path(path_str):
        """Resolve path: if relative, resolve relative to PROJECT_ROOT; if absolute, use as-is"""
        p = Path(path_str)
        if p.is_absolute():
            return p
        else:
            return Path(PROJECT_ROOT) / p
    
    # Determine input files
    if args.input_files:
        input_files = [resolve_path(f) for f in args.input_files]
    elif args.input_dir:
        input_dir = resolve_path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return
        input_files = sorted(input_dir.glob('*.dow'))
        if not input_files:
            logger.error(f"No .dow files found in {input_dir}")
            return
    else:
        logger.error("Must specify either --input_dir or --input_files")
        return
    
    # Resolve output directory
    output_dir = resolve_path(args.output_dir)
    
    # Generate stochastic instances
    try:
        output_paths = generate_stochastic_instances(
            input_deterministic_instances=input_files,
            data_generation_method=args.method,
            stochastic_combination_type=args.type,
            compute_statistics=not args.no_statistics,
            output_dir=str(output_dir),
            n_scenarios=args.scenarios,
            seed=args.seed,
            sigma_L=args.sigma_L,
            sigma_U=args.sigma_U,
            generate_covariates=not args.no_covariates,
            n_covariates=args.n_covariates,
            include_deterministic_first=not args.no_deterministic_first,
            overwrite=args.overwrite,
            output_suffix=args.suffix,
        )
        
        logger.info(f"\n✓ Successfully generated {len(output_paths)} instances")
        
    except Exception as e:
        logger.error(f"\n✗ Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
