import numpy as np
import pandas as pd
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import List, Optional
import sys

# Add the src directory to the Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# UHL-specific parameters removed - SCFLP focus only
# from config import p_hub, beta, gamma  # Removed - UHL parameters not needed for SCFLP focus
import pickle

# Logger setup
logger = logging.getLogger(__name__)

# Attempt to import pysmps parsers
try:
    from pysmps.smps_loader import load_mps, load_smps, load_2stage_problem
    logger.info("Successfully imported pysmps modules")
except ImportError as e:
    logger.error(f"Failed to import from pysmps: {e}")
    load_mps = None
    load_smps = None
    load_2stage_problem = None


# ============================
# SCFLP Data and Reader
# ============================
@dataclass
class SCFLPData:
    """Stochastic Capacitated Facility Location Problem data structure"""
    I: np.ndarray  # Customer index list
    J: np.ndarray  # Facility index list
    S: np.ndarray  # Scenario index list
    demands: np.ndarray  # Customer demands
    scenarios: np.ndarray  # Demand Scenarios
    capacities: np.ndarray  # Facility capacities
    fixed_costs: np.ndarray  # Facility opening costs
    shipment_costs: np.ndarray  # Transportation costs
    max_demand_sum_over_scenario: float  # maximum of demand sums under scenarios
    scenario_generation_variance: float  # Legacy variance parameter (for backward compatibility)
    scenario_generation_method: str  # Legacy method parameter (for backward compatibility)
    # New metrics from version2 data generation (CV-driven)
    distribution: str = ""  # Distribution method used for generation (normal, uniform, gamma, lognormal, merve_normal)
    target_cv: float = 0.0  # Target coefficient of variation (input parameter)
    empirical_cv_mean: float = 0.0  # Achieved empirical CV mean across customers
    r_overall: float = 0.0  # Vector-level normalized range dispersion metric
    # Merve_Normal specific parameters
    sigma_lower: float = 0.0  # Lower bound for sigma_i in Merve_Normal method
    sigma_upper: float = 0.0  # Upper bound for sigma_i in Merve_Normal method
    # ARMA covariate fields (optional, for methods that generate feature vectors)
    covariate_vectors: Optional[np.ndarray] = None  # Covariate vectors X^(s) for each scenario s
    num_covariates: int = 0  # Number of covariates/features
    # ARMA validation metrics (optional, for methods with train/validation split)
    validation_r_squared: Optional[float] = None  # R² from regression validation
    validation_mse: Optional[float] = None  # MSE from regression validation  
    validation_mape: Optional[float] = None  # MAPE from regression validation
    # Capacity scenarios (optional, for methods with varying capacity per scenario)
    capacity_scenarios: Optional[np.ndarray] = None  # Capacity scenarios for each scenario [scenarios × facilities]

    @property
    def cost(self):
        """Alias for shipment_costs to maintain compatibility with dual subproblem code"""
        return self.shipment_costs
    
    @property
    def capacity(self):
        """Alias for capacities to maintain compatibility with dual subproblem code"""
        return self.capacities

def read_scflp_data(file_path: str) -> SCFLPData:
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse header information for original format compatibility
    header_text = "".join(lines[:10])
    var_match = re.search(r"variance\s*=\s*([\d.]+)", header_text)
    method_match = re.search(r"method\s*=\s*([a-zA-Z]+)", header_text)
    scenario_variance = float(var_match.group(1)) if var_match else 0.0
    scenario_method = method_match.group(1) if method_match else "unknown"
    
    # Parse new version2 metrics from header or diagnostics section
    target_cv_match = re.search(r"target_cv\s*=\s*([\d.]+|null)", header_text)
    distribution_match = re.search(r"distribution\s*=\s*([a-zA-Z0-9_(),]+)", header_text)
    target_cv = float(target_cv_match.group(1)) if target_cv_match and target_cv_match.group(1) != "null" else 0.0
    distribution = distribution_match.group(1) if distribution_match else "unknown"
    
    # Parse diagnostics section for empirical metrics
    full_text = "".join(lines)
    empirical_cv_mean_match = re.search(r"empirical_cv_mean\s*=\s*([\d.]+)", full_text)
    r_overall_match = re.search(r"r_overall\s*=\s*([\d.]+)", full_text)
    empirical_cv_mean = float(empirical_cv_mean_match.group(1)) if empirical_cv_mean_match else 0.0
    r_overall = float(r_overall_match.group(1)) if r_overall_match else 0.0
    
    # Parse Merve_Normal specific parameters
    sigma_lower_match = re.search(r"sigma_lower\s*=\s*([\d.]+|null)", full_text)
    sigma_upper_match = re.search(r"sigma_upper\s*=\s*([\d.]+|null)", full_text)
    sigma_lower = float(sigma_lower_match.group(1)) if sigma_lower_match and sigma_lower_match.group(1) != "null" else 0.0
    sigma_upper = float(sigma_upper_match.group(1)) if sigma_upper_match and sigma_upper_match.group(1) != "null" else 0.0
    
    # Parse ARMA-specific parameter from header
    num_covariates_match = re.search(r"num_covariates\s*=\s*(\d+)", header_text)
    num_covariates_header = int(num_covariates_match.group(1)) if num_covariates_match else 0

    content_lines = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    m, n, s = map(int, content_lines[0].split())

    capacities = np.array(list(map(float, content_lines[1].split())))
    fixed_costs = np.array(list(map(float, content_lines[2].split())))
    demands = np.array(list(map(float, content_lines[3].split())))

    shipment_costs = np.array([
        list(map(float, row.split()))
        for row in content_lines[4:4 + n]
    ])

    scenarios = []
    i = 4 + n
    while len(scenarios) < s and i < len(content_lines):
        line = content_lines[i]
        if not line.startswith("# Scenario") and not line.startswith("# Feature"):
            try:
                values = list(map(float, line.split()))
                if len(values) == n:
                    scenarios.append(values)
            except ValueError:
                pass
        i += 1
    scenarios = np.array(scenarios)

    # Parse capacity scenarios section (optional - only present in methods with varying capacity per scenario)
    capacity_scenarios = None
    full_text = "".join(lines)
    if "# Capacity scenarios" in full_text:
        # Parse capacity scenarios from the raw lines
        cap_scenarios = []
        in_capacity_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("# Capacity scenarios"):
                in_capacity_section = True
                continue
            elif in_capacity_section:
                if line.startswith("# Diagnostics") or line.startswith("# Feature"):
                    break  # End of capacity scenarios section
                elif line.startswith("# Capacity scenario for scenario") or line.startswith("# Number of"):
                    continue  # Skip comment lines within capacity section
                elif line and not line.startswith("#"):
                    # This should be a capacity scenario data line
                    try:
                        values = list(map(float, line.split()))
                        if len(values) == m:  # Should match number of facilities
                            cap_scenarios.append(values)
                    except ValueError:
                        continue  # Skip lines that can't be parsed as floats
        
        if cap_scenarios:
            capacity_scenarios = np.array(cap_scenarios)
            logger.info(f"Parsed {len(capacity_scenarios)} capacity scenarios from file")

    # Parse ARMA covariate vectors section (optional - only present in ARMA method)
    covariate_vectors = None
    num_covariates = num_covariates_header  # Use header value as starting point
    
    # Look for feature/covariate vectors section in the file (support both naming conventions)
    if "# Feature vectors" in full_text or "# Covariate vectors" in full_text:
        # Parse covariate vectors from the raw lines (not content_lines which filters out comments)
        features = []
        in_feature_section = False
        
        for line in lines:
            line = line.strip()
            # Support both "# Feature vectors" and "# Covariate vectors" headers
            if line.startswith("# Feature vectors") or line.startswith("# Covariate vectors"):
                in_feature_section = True
                continue
            elif in_feature_section:
                if line.startswith("# Diagnostics"):
                    break  # End of feature section
                # Skip comment lines within feature section (including covariates header and scenario labels)
                elif line.startswith("# Covariate vector for scenario") or line.startswith("# Feature vector for scenario") or line.startswith("# Number of covariates"):
                    continue  # Skip comment lines within feature section
                elif line and not line.startswith("#"):
                    # This should be a covariate vector data line
                    try:
                        values = list(map(float, line.split()))
                        if len(values) > 0:
                            features.append(values)
                    except ValueError:
                        continue  # Skip lines that can't be parsed as floats
        
        if features:
            covariate_vectors = np.array(features)
            # Update num_covariates from actual data if header was missing/wrong
            if len(covariate_vectors.shape) > 1:
                num_covariates = covariate_vectors.shape[1]

    # Parse ARMA validation metrics (optional - only present in ARMA method)
    validation_r_squared = None
    validation_mse = None  
    validation_mape = None
    
    # Look for validation metrics in diagnostics (ARMA format uses different names)
    r2_match = re.search(r"arma_validation_r2_mean\s*=\s*([-+]?[\d.]+)", full_text)
    mse_match = re.search(r"arma_validation_mse_mean\s*=\s*([\d.]+)", full_text)
    mape_match = re.search(r"arma_validation_mape_mean\s*=\s*([\d.]+)%?", full_text)
    
    if r2_match:
        validation_r_squared = float(r2_match.group(1))
    if mse_match:
        validation_mse = float(mse_match.group(1))
    if mape_match:
        validation_mape = float(mape_match.group(1))

    # Compute the maximum total demand across all scenarios
    # This is max_{s ∈ S} Σ_{i ∈ I} d_{i,s} as per the mathematical formulation
    max_demand = 0.0
    if len(scenarios) > 0:
        scenario_totals = np.sum(scenarios, axis=1)  # Sum demand across customers for each scenario
        max_demand = float(np.max(scenario_totals))   # Maximum across all scenarios
    else:
        # Fallback: if no scenarios loaded, use the base demand
        max_demand = float(np.sum(demands))

    I = np.arange(n)
    J = np.arange(m)
    S = np.arange(s)

    return SCFLPData(
        I=I,
        J=J,
        S=S,
        demands=demands,
        scenarios=scenarios,
        capacities=capacities,
        capacity_scenarios=capacity_scenarios,
        fixed_costs=fixed_costs,
        shipment_costs=shipment_costs,
        max_demand_sum_over_scenario=max_demand,
        scenario_generation_variance=scenario_variance,
        scenario_generation_method=scenario_method,
        distribution=distribution,
        target_cv=target_cv,
        empirical_cv_mean=empirical_cv_mean,
        r_overall=r_overall,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        covariate_vectors=covariate_vectors,
        num_covariates=num_covariates,
        validation_r_squared=validation_r_squared,
        validation_mse=validation_mse,
        validation_mape=validation_mape
    )


# ============================
# NE-SP Data and Reader
# ============================
@dataclass
class NESPData:
    """Network Expansion with OD Shortest-Path Recourse data structure"""
    N: np.ndarray  # Node index list (real nodes)
    A: List[tuple]  # Arc list [(i, j), ...] (real arcs only)
    A_extended: List[tuple]  # Extended arc list (real + penalty arcs)
    M: List[tuple]  # OD pairs [(o, t), ...]
    n_nodes: int  # Number of real nodes
    n_arcs: int  # Number of real arcs
    n_od_pairs: int  # Number of OD pairs
    supernode_id: int  # ID for the supernode (r)
    
    # Arc attributes (indexed by position in A)
    F: np.ndarray  # Construction/activation costs F_ij
    L: np.ndarray  # Travel lengths/distances L_ij
    u: np.ndarray  # Capacities u_ij (optional, not used in basic NE-SP)
    
    # OD pair attributes
    weights: np.ndarray  # Weights omega_m for each OD pair
    origins: np.ndarray  # Origin nodes for each OD pair
    destinations: np.ndarray  # Destination nodes for each OD pair
    demands: np.ndarray  # Demand values d_k (optional, for weight calculation)
    
    # Penalty parameters
    L_max: float  # Maximum travel length among real arcs
    M_penalty: float  # Penalty cost for artificial arcs (20 * (n-1) * L_max)
    
    # Arc mapping dictionaries for efficient lookup
    arc_to_index: dict = field(default_factory=dict)  # {(i,j): index in A}
    
    def __post_init__(self):
        """Build arc_to_index mapping after initialization"""
        self.arc_to_index = {arc: idx for idx, arc in enumerate(self.A)}


def read_nesp_data(file_path: str, weight_mode: str = "uniform") -> NESPData:
    """
    Parse a NE-SP instance file with CommaLAB MMCF format (C, C+, R families).
    
    Parameters:
    -----------
    file_path : str
        Path to the instance file (e.g., r01.1.dow)
    weight_mode : str
        How to set OD pair weights:
        - "uniform": omega_m = 1 for all OD pairs
        - "demand": omega_m = d_k (commodity demand)
    
    Returns:
    --------
    NESPData object with all problem data and penalty structure
    """
    # logger.info(f"Reading NE-SP instance from {file_path}")
    
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    # Remove empty lines, comments, and header lines (e.g., "MULTIGEN.DAT:")
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip header lines that contain colons or other text markers
        if ":" in line or not line.split()[0].lstrip('-').replace('.','',1).isdigit():
            continue
        processed_lines.append(line)
    
    lines = processed_lines
    
    # Parse header: n_nodes m_arcs K_commodities
    n_nodes, m_arcs, K = map(int, lines[0].split())
    logger.info(f"Instance size: {n_nodes} nodes, {m_arcs} arcs, {K} OD pairs")
    
    # Parse arc block (next m_arcs lines)
    # Format: i j c_ij u_ij f_ij extra1 extra2
    # For NE-SP: F_ij = f_ij (construction cost), L_ij = c_ij (travel length)
    A = []
    F = []
    L = []
    u = []
    
    for idx in range(1, m_arcs + 1):
        parts = lines[idx].split()
        i, j = int(parts[0]), int(parts[1])
        c_ij = float(parts[2])  # Travel length
        u_ij = float(parts[3])  # Capacity (not used in basic NE-SP)
        f_ij = float(parts[4])  # Construction cost
        
        A.append((i, j))
        L.append(c_ij)
        u.append(u_ij)
        F.append(f_ij)
    
    # Convert to numpy arrays
    F = np.array(F)
    L = np.array(L)
    u = np.array(u)
    
    # Compute L_max and penalty parameter M
    L_max = np.max(L)
    M_penalty = 20.0 * (n_nodes - 1) * L_max
    logger.info(f"L_max = {L_max:.2f}, Penalty M = {M_penalty:.2f}")
    
    # Parse commodity block (next K lines)
    # Format: o_k t_k d_k
    M_od_pairs = []
    origins = []
    destinations = []
    demands = []
    weights = []
    
    for idx in range(m_arcs + 1, m_arcs + 1 + K):
        parts = lines[idx].split()
        o_k = int(parts[0])
        t_k = int(parts[1])
        d_k = float(parts[2])
        
        M_od_pairs.append((o_k, t_k))
        origins.append(o_k)
        destinations.append(t_k)
        demands.append(d_k)
        
        # Set weight based on mode
        if weight_mode == "uniform":
            weights.append(1.0)
        elif weight_mode == "demand":
            weights.append(d_k)
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")
    
    # Convert to numpy arrays
    origins = np.array(origins)
    destinations = np.array(destinations)
    demands = np.array(demands)
    weights = np.array(weights)
    
    # Create supernode (ID = max real node + 1)
    supernode_id = max(max(i for i, j in A), max(j for i, j in A)) + 1
    logger.info(f"Supernode ID: {supernode_id}")
    
    # Create extended arc list with penalty arcs
    # Penalty arcs: (i, r) and (r, i) for all real nodes i
    A_extended = list(A)  # Start with real arcs
    real_nodes = set()
    for i, j in A:
        real_nodes.add(i)
        real_nodes.add(j)
    
    for node in sorted(real_nodes):
        A_extended.append((node, supernode_id))  # i -> r
        A_extended.append((supernode_id, node))  # r -> i
    
    logger.info(f"Extended arc set: {len(A_extended)} arcs ({len(A)} real + {len(A_extended) - len(A)} penalty)")
    
    # Create node index array
    N = np.array(sorted(real_nodes))
    
    return NESPData(
        N=N,
        A=A,
        A_extended=A_extended,
        M=M_od_pairs,
        n_nodes=n_nodes,
        n_arcs=m_arcs,
        n_od_pairs=K,
        supernode_id=supernode_id,
        F=F,
        L=L,
        u=u,
        weights=weights,
        origins=origins,
        destinations=destinations,
        demands=demands,
        L_max=L_max,
        M_penalty=M_penalty
    )


# ============================
# SUMND Data and Reader
# ============================
SCMND_FORMULATION_CURRENT = "CURRENT_SOFT_CAPACITY"
SCMND_FORMULATION_MANUSCRIPT = "MANUSCRIPT_ENVELOPE"


def normalize_scmnd_formulation(value: str) -> str:
    aliases = {
        "CURRENT": SCMND_FORMULATION_CURRENT,
        "CURRENT_SOFT_CAPACITY": SCMND_FORMULATION_CURRENT,
        "SOFT_CAPACITY": SCMND_FORMULATION_CURRENT,
        "MANUSCRIPT": SCMND_FORMULATION_MANUSCRIPT,
        "MANUSCRIPT_ENVELOPE": SCMND_FORMULATION_MANUSCRIPT,
        "ENVELOPE": SCMND_FORMULATION_MANUSCRIPT,
    }
    normalized = str(value or SCMND_FORMULATION_CURRENT).upper()
    if normalized not in aliases:
        raise ValueError(
            "scmnd_formulation must be CURRENT_SOFT_CAPACITY or MANUSCRIPT_ENVELOPE"
        )
    return aliases[normalized]


def scmnd_uses_manuscript_formulation(data) -> bool:
    return normalize_scmnd_formulation(
        getattr(data, "scmnd_formulation", SCMND_FORMULATION_CURRENT)
    ) == SCMND_FORMULATION_MANUSCRIPT


def scmnd_capacity_vector(data, scenario_id: int) -> np.ndarray:
    """Capacity coefficients shared by the SCMND dual objective and cut."""
    if scmnd_uses_manuscript_formulation(data):
        return np.asarray(data.u_base, dtype=float)
    return np.asarray(data.capacity_scenarios[int(scenario_id)], dtype=float)


def scmnd_envelope_demands(data) -> np.ndarray:
    """Commoditywise maximum sampled demand used in model:scmnd-master."""
    return np.max(np.asarray(data.demand_scenarios, dtype=float), axis=0)


@dataclass
class SUMNDData:
    """Stochastic Uncapacitated Multicommodity Network Design data structure"""
    N: np.ndarray  # Node index list
    A: List[tuple]  # Arc list [(i, j), ...]
    K: np.ndarray  # Commodity index list
    S: np.ndarray  # Scenario index list
    n_nodes: int  # Number of nodes
    n_arcs: int  # Number of arcs
    n_commodities: int  # Number of commodities
    n_scenarios: int  # Number of scenarios
    
    # Arc attributes (indexed by position in A)
    F: np.ndarray  # Fixed activation costs f_ij
    C: np.ndarray  # Unit transportation costs c_ij^k (shape: n_arcs x n_commodities)
    u_base: np.ndarray  # Base arc capacities u_ij (from deterministic file)
    
    # Commodity attributes (deterministic)
    origins: np.ndarray  # Origin nodes for each commodity
    destinations: np.ndarray  # Destination nodes for each commodity
    demands_base: np.ndarray  # Base demands d_k (from deterministic file)
    
    # Stochastic parameters
    probabilities: np.ndarray  # Scenario probabilities p_s (shape: n_scenarios)
    demand_scenarios: np.ndarray  # Stochastic demands d_k^s (shape: n_scenarios x n_commodities)
    capacity_scenarios: np.ndarray  # Stochastic capacities u_ij^s (shape: n_scenarios x n_arcs)

    # Optional covariate vectors for ML methods
    covariate_vectors: Optional[np.ndarray] = None  # Shape: (n_scenarios, n_covariates)
    num_covariates: int = 0  # Number of covariates
    
    # Metadata
    generation_method: str = "unknown"
    generation_params: dict = field(default_factory=dict)  # Generation parameters for reproducibility
    
    # Arc mapping for efficient lookup
    arc_to_index: dict = field(default_factory=dict)  # {(i,j): index in A}
    
    def __post_init__(self):
        """Build arc_to_index mapping after initialization"""
        self.arc_to_index = {arc: idx for idx, arc in enumerate(self.A)}


@dataclass
class SCMNDData:
    """Stochastic Capacitated Multicommodity Network Design data structure"""
    N: np.ndarray  # Node index list
    A: List[tuple]  # Arc list [(i, j), ...]
    K: np.ndarray  # Commodity index list
    S: np.ndarray  # Scenario index list
    n_nodes: int  # Number of nodes
    n_arcs: int  # Number of arcs
    n_commodities: int  # Number of commodities
    n_scenarios: int  # Number of scenarios

    # Arc attributes (indexed by position in A)
    F: np.ndarray  # Fixed activation costs f_ij
    C: np.ndarray  # Unit transportation costs c_ij^k (shape: n_arcs x n_commodities)
    u_base: np.ndarray  # Base arc capacities u_ij (from deterministic file)

    # Commodity attributes (deterministic)
    origins: np.ndarray  # Origin nodes for each commodity
    destinations: np.ndarray  # Destination nodes for each commodity
    demands_base: np.ndarray  # Base demands d_k (from deterministic file)

    # Stochastic parameters
    probabilities: np.ndarray  # Scenario probabilities p_s (shape: n_scenarios)
    demand_scenarios: np.ndarray  # Stochastic demands d_k^s (shape: n_scenarios x n_commodities)
    capacity_scenarios: np.ndarray  # Stochastic capacities u_ij^s (shape: n_scenarios x n_arcs)

    # Slack-penalty parameter for SCMND recourse feasibility
    slack_penalty_multiplier: float = 2.0

    # Selectable executable formulation.  The current formulation is preserved;
    # the manuscript formulation uses fixed capacities and envelope-flow master constraints.
    scmnd_formulation: str = SCMND_FORMULATION_CURRENT

    # Optional covariate vectors for ML methods
    covariate_vectors: Optional[np.ndarray] = None  # Shape: (n_scenarios, n_covariates)
    num_covariates: int = 0  # Number of covariates

    # Metadata
    generation_method: str = "unknown"
    generation_params: dict = field(default_factory=dict)  # Generation parameters for reproducibility

    # Arc mapping for efficient lookup
    arc_to_index: dict = field(default_factory=dict)  # {(i,j): index in A}

    def __post_init__(self):
        """Build arc_to_index mapping after initialization"""
        self.scmnd_formulation = normalize_scmnd_formulation(self.scmnd_formulation)
        self.arc_to_index = {arc: idx for idx, arc in enumerate(self.A)}


def _read_stochastic_multicommodity_data(
    file_path: str,
    enforce_uncapacitated: bool,
    slack_penalty_multiplier: float = 2.0,
):
    """Shared JSON reader for SUMND/SCMND; optionally enforces uncapacitated capacities."""
    import json

    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract metadata
    metadata = data.get("metadata", {})
    generation_method = metadata.get("generation_method", "unknown")

    # Store generation parameters for reproducibility
    generation_params = {
        "seed": metadata.get("seed"),
        "sigma_L": metadata.get("sigma_L"),
        "sigma_U": metadata.get("sigma_U"),
        "stochastic_combination_type": metadata.get("stochastic_combination_type"),
        "source_file": metadata.get("source_file_name")
    }

    # Extract deterministic data
    det = data["deterministic"]
    n_nodes = det["n_nodes"]
    n_arcs = det["n_arcs"]
    n_commodities = det["n_commodities"]

    # Parse arcs
    A = []
    F = []
    u_base = []
    c_dict = {}

    for arc_data in det["arcs"]:
        i = arc_data["tail"]
        j = arc_data["head"]
        arc_idx = arc_data["index"]

        A.append((i, j))
        F.append(arc_data["fixed_cost"])
        u_base.append(arc_data["capacity"])
        c_dict[arc_idx] = arc_data["unit_cost"]

    F = np.array(F)
    u_base = np.array(u_base)

    # Build unit cost matrix C[arc_idx, commodity_idx]
    C = np.zeros((n_arcs, n_commodities))
    for arc_idx in range(n_arcs):
        C[arc_idx, :] = c_dict[arc_idx]

    # Parse commodities
    origins = []
    destinations = []
    demands_base = []

    for comm_data in det["commodities"]:
        origins.append(comm_data["origin"])
        destinations.append(comm_data["destination"])
        demands_base.append(comm_data["demand"])

    origins = np.array(origins)
    destinations = np.array(destinations)
    demands_base = np.array(demands_base)

    # Extract stochastic scenarios
    scenarios = data["scenarios"]
    n_scenarios_actual = scenarios["n_scenarios"]
    probabilities = np.array(scenarios["probabilities"])
    demand_scenarios = np.array(scenarios["demand_scenarios"])
    capacity_scenarios = np.array(scenarios["capacity_scenarios"])

    # Extract covariates if present
    covariate_vectors = None
    num_covariates = 0
    if "covariates" in scenarios and scenarios["covariates"] is not None:
        covariate_vectors = np.array(scenarios["covariates"])
        num_covariates = covariate_vectors.shape[1] if len(covariate_vectors.shape) > 1 else 0
        logger.info(f"Loaded {num_covariates} covariates for {n_scenarios_actual} scenarios")

    # SUMND enforces uncapacitated behavior by overriding capacities.
    if enforce_uncapacitated:
        logger.info("Overriding arc capacities to ensure uncapacitated formulation...")
        for s in range(n_scenarios_actual):
            total_demand_s = np.sum(demand_scenarios[s, :])
            capacity_scenarios[s, :] = total_demand_s
        logger.info("Set uniform capacity per scenario based on total commodity demand")

    # Create index arrays
    N = np.arange(1, n_nodes + 1)  # Nodes are 1-indexed
    K = np.arange(n_commodities)
    S = np.arange(n_scenarios_actual)

    return {
        "N": N,
        "A": A,
        "K": K,
        "S": S,
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "n_commodities": n_commodities,
        "n_scenarios": n_scenarios_actual,
        "F": F,
        "C": C,
        "u_base": u_base,
        "origins": origins,
        "destinations": destinations,
        "demands_base": demands_base,
        "probabilities": probabilities,
        "demand_scenarios": demand_scenarios,
        "capacity_scenarios": capacity_scenarios,
        "covariate_vectors": covariate_vectors,
        "num_covariates": num_covariates,
        "generation_method": generation_method,
        "generation_params": generation_params,
        "slack_penalty_multiplier": float(slack_penalty_multiplier),
    }


def read_sumnd_data(file_path: str) -> SUMNDData:
    """
    Parse a SUMND (Stochastic Uncapacitated Multicommodity Network Design) instance file.
    
    Parameters:
    -----------
    file_path : str
        Path to the stochastic JSON file (e.g., r01.1.stoch.json)
    
    Returns:
    --------
    SUMNDData object with all problem data including stochastic scenarios
    """
    # logger.info(f"Reading SUMND instance from {file_path}")
    parsed = _read_stochastic_multicommodity_data(
        file_path=file_path,
        enforce_uncapacitated=True,
        slack_penalty_multiplier=2.0,
    )

    A = parsed["A"]
    F = parsed["F"]
    u_base = parsed["u_base"]
    C = parsed["C"]
    origins = parsed["origins"]
    destinations = parsed["destinations"]
    demand_scenarios = parsed["demand_scenarios"]
    n_nodes = parsed["n_nodes"]
    n_arcs = parsed["n_arcs"]
    n_commodities = parsed["n_commodities"]
    n_scenarios_actual = parsed["n_scenarios"]

    # Detailed SUMND parameter dump for visual validation in logs.
    logger.info("============================")
    logger.info("fixed cost value for arc:")
    for arc_idx, (i, j) in enumerate(A):
        logger.info("(%s, %s): %s", i, j, float(F[arc_idx]))

    logger.info("--------------------")
    logger.info("capcity value for arc:")
    for arc_idx, (i, j) in enumerate(A):
        logger.info("(%s, %s): %s", i, j, float(u_base[arc_idx]))

    logger.info("-----------------------")
    logger.info("unit cost value for arc:")
    for arc_idx, (i, j) in enumerate(A):
        logger.info("(%s, %s): %s", i, j, float(C[arc_idx, 0]))

    logger.info("----------------------")
    logger.info("demand for arcs")
    for k_idx in range(n_commodities):
        o_k = int(origins[k_idx])
        t_k = int(destinations[k_idx])
        scenario_demands_k = [float(demand_scenarios[s_idx, k_idx]) for s_idx in range(n_scenarios_actual)]
        logger.info(
            "origin=%s destination=%s commodity=%s scenario_demands=%s",
            o_k,
            t_k,
            k_idx,
            scenario_demands_k,
        )
    logger.info("============================")
    
    logger.info(f"Instance size: {n_nodes} nodes, {n_arcs} arcs, {n_commodities} commodities, {n_scenarios_actual} scenarios")
    
    return SUMNDData(**{k: v for k, v in parsed.items() if k != "slack_penalty_multiplier"})


def read_scmnd_data(
    file_path: str,
    slack_penalty_multiplier: float = 2.0,
    scmnd_formulation: str = SCMND_FORMULATION_CURRENT,
) -> SCMNDData:
    """
    Parse an SCMND instance from the same JSON schema as SUMND.

    Difference from SUMND:
    - SCMND preserves scenario capacities from input (no override).
    - Stores slack_penalty_multiplier for bounded dual variables in recourse.
    """
    # logger.info(f"Reading SCMND instance from {file_path}")
    parsed = _read_stochastic_multicommodity_data(
        file_path=file_path,
        enforce_uncapacitated=False,
        slack_penalty_multiplier=slack_penalty_multiplier,
    )

    parsed["scmnd_formulation"] = normalize_scmnd_formulation(scmnd_formulation)
    return SCMNDData(**parsed)
