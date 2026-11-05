"""
Phase 0 Generator for Learning-Augmented Benders Decomposition

This module implements Phase 0 of the extended LABD framework, which generates
uniformly sampled feasible first-stage solutions to enrich the training data
before running the classic Benders decomposition (Phase 1).

Phase 0 Purpose:
- Generate m_trials random but feasible first-stage solutions
- Solve all subproblems exactly for each solution to collect optimal duals
- Add cuts from these duals to strengthen the master problem before Phase 1
- Combine duals from Phase 0 and Phase 1 for feature generation in Phase 2

The Pure Random but Filtered Method:
- Randomly sample binary values for first-stage variables (facility open/close)
- Check feasibility (e.g., total capacity >= worst-case demand in SCFLP)
- If infeasible, iteratively repair by opening random facilities
- Ensure no duplicate solutions are generated

Integration with LABD Framework:
- m_trials = 0: Skip Phase 0, use current LABD (Phase 1 → Phase 2)
- n_trials = 0: Skip Phase 1, rely only on Phase 0 duals for ML
- Both > 0: Full hybrid (Phase 0 + Phase 1 → Phase 2)
"""

import numpy as np
import logging
from typing import List, Set, Tuple
import sys
import os

# Add the src directory to the Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Problem.data import SCFLPData, NESPData

logger = logging.getLogger(__name__)


def generate_phase0_solutions(problem_data, m_trials: int, problem_class: str) -> List[np.ndarray]:
    """
    Generate m_trials feasible first-stage solutions using Pure Random but Filtered method.
    
    These solutions serve as Phase 0 warm-up for the Learning-Augmented Benders Decomposition:
    1. Random first-stage solutions are generated and repaired to ensure feasibility
    2. All subproblems are solved exactly for each solution to collect optimal duals
    3. Cuts generated from these duals are added to the master problem before Phase 1
    4. Duals from Phase 0 and Phase 1 are combined for feature generation in Phase 2
    
    Args:
        problem_data: Problem instance data (SCFLPData or NESPData)
        m_trials (int): Number of Phase 0 solutions to generate
        problem_class (str): Problem type ('SCFLP' or 'NESP')
        
    Returns:
        List[np.ndarray]: List of m_trials feasible first-stage solutions
        
    Raises:
        ValueError: If problem_class is not supported or m_trials <= 0
        RuntimeError: If unable to generate feasible solution after max attempts
    """
    if m_trials <= 0:
        logger.warning("m_trials <= 0, returning empty list for Phase 0")
        return []
        
    logger.info(f"Generating {m_trials} Phase 0 warm-up solutions for {problem_class}")
    
    if problem_class.upper() == 'SCFLP':
        return _generate_scflp_solutions(problem_data, m_trials)
    elif problem_class.upper() in ['NESP', 'NE-SP']:
        return _generate_nesp_solutions(problem_data, m_trials)
    elif problem_class.upper() in ['UFL', 'UHL', 'SSLP', 'CMND', 'MCFL']:
        raise NotImplementedError(f"Phase 0 generator for {problem_class} removed - focusing on SCFLP and NE-SP only")
    else:
        raise ValueError(f"Unsupported problem class: {problem_class}")


def _generate_scflp_solutions(data: SCFLPData, m_trials: int) -> List[np.ndarray]:
    """
    Generate feasible SCFLP first-stage solutions using Pure Random but Filtered method.
    
    SCFLP Feasibility: Total opened facility capacity >= worst-case demand across all scenarios
    
    Args:
        data (SCFLPData): SCFLP problem instance data
        m_trials (int): Number of solutions to generate
        
    Returns:
        List[np.ndarray]: List of feasible binary facility opening solutions
    """
    solutions = []
    solution_set: Set[Tuple] = set()  # Track duplicates
    max_attempts_per_solution = 1000
    
    # Calculate worst-case demand across all scenarios  
    # Use pre-computed max_demand_sum_over_scenario from SCFLPData
    worst_case_demand = data.max_demand_sum_over_scenario
    logger.debug(f"SCFLP worst-case demand: {worst_case_demand}")
    logger.debug(f"Total available capacity: {np.sum(data.capacities)}")
    
    for trial in range(m_trials):
        attempts = 0
        while attempts < max_attempts_per_solution:
            # Generate random binary solution
            y = np.random.choice([0, 1], size=len(data.J))
            
            # Check if solution provides enough capacity
            total_capacity = np.sum(data.capacities * y)
            
            if total_capacity >= worst_case_demand:
                # Solution is feasible, check for duplicates
                y_tuple = tuple(y)
                if y_tuple not in solution_set:
                    solutions.append(y.copy())
                    solution_set.add(y_tuple)
                    logger.debug(f"Generated feasible SCFLP solution {trial+1}/{m_trials} "
                               f"with capacity {total_capacity:.2f} >= {worst_case_demand:.2f}")
                    break
            else:
                # Repair solution by opening random facilities
                y = _repair_scflp_solution(y, data.capacities, worst_case_demand)
                y_tuple = tuple(y)
                if y_tuple not in solution_set:
                    solutions.append(y.copy())
                    solution_set.add(y_tuple)
                    total_capacity = np.sum(data.capacities * y)
                    logger.debug(f"Repaired and generated feasible SCFLP solution {trial+1}/{m_trials} "
                               f"with capacity {total_capacity:.2f} >= {worst_case_demand:.2f}")
                    break
            
            attempts += 1
            
        if attempts >= max_attempts_per_solution:
            raise RuntimeError(f"Failed to generate feasible SCFLP solution {trial+1} after {max_attempts_per_solution} attempts")
    
    logger.info(f"Successfully generated {len(solutions)} unique feasible SCFLP Phase 0 solutions")
    return solutions


def _repair_scflp_solution(y: np.ndarray, capacities: np.ndarray, required_capacity: float) -> np.ndarray:
    """
    Repair an infeasible SCFLP solution by iteratively opening random facilities.
    
    Args:
        y (np.ndarray): Binary facility opening solution to repair
        capacities (np.ndarray): Facility capacities
        required_capacity (float): Minimum total capacity required
        
    Returns:
        np.ndarray: Repaired feasible solution
    """
    y_repaired = y.copy()
    current_capacity = np.sum(capacities * y_repaired)
    
    # Get indices of closed facilities that can be opened
    closed_facilities = np.where(y_repaired == 0)[0]
    
    while current_capacity < required_capacity and len(closed_facilities) > 0:
        # Randomly select a closed facility to open
        facility_to_open = np.random.choice(closed_facilities)
        y_repaired[facility_to_open] = 1
        current_capacity += capacities[facility_to_open]
        
        # Update closed facilities list
        closed_facilities = np.where(y_repaired == 0)[0]
    
    if current_capacity < required_capacity:
        logger.warning(f"Could not repair solution to meet capacity requirement: "
                      f"{current_capacity} < {required_capacity}")
    
    return y_repaired


def _generate_nesp_solutions(data: NESPData, m_trials: int) -> List[np.ndarray]:
    """
    Generate feasible NE-SP first-stage solutions using Pure Random sampling.
    
    For NE-SP with supernode penalty structure:
    - ALL first-stage solutions are feasible (complete recourse via supernode)
    - No explicit feasibility repair needed
    - Only need to ensure diverse solutions (no duplicates)
    
    Args:
        data (NESPData): NE-SP problem instance data
        m_trials (int): Number of solutions to generate
        
    Returns:
        List[np.ndarray]: List of binary arc activation solutions
    """
    solutions = []
    solution_set: Set[Tuple] = set()  # Track duplicates
    max_attempts_per_solution = 1000
    
    n_arcs = len(data.A)
    logger.debug(f"NE-SP has {n_arcs} candidate arcs for activation")
    
    for trial in range(m_trials):
        attempts = 0
        while attempts < max_attempts_per_solution:
            # Generate random binary solution (arc activation)
            # Use probability p=0.3 to activate arcs (avoid trivial all-zero or all-one solutions)
            x = np.random.choice([0, 1], size=n_arcs, p=[0.7, 0.3])
            
            # Check for duplicates
            x_tuple = tuple(x)
            if x_tuple not in solution_set:
                solutions.append(x.copy())
                solution_set.add(x_tuple)
                activated_arcs = np.sum(x)
                total_cost = np.sum(data.F * x)
                logger.debug(f"Generated NE-SP solution {trial+1}/{m_trials}: "
                           f"{activated_arcs}/{n_arcs} arcs activated, cost={total_cost:.2f}")
                break
            
            attempts += 1
            
        if attempts >= max_attempts_per_solution:
            # If we can't find unique solution, just force a new one by flipping a random bit
            logger.warning(f"Difficulty finding unique solution for trial {trial+1}, using modified solution")
            base_solution = solutions[-1] if solutions else np.zeros(n_arcs, dtype=int)
            x = base_solution.copy()
            # Flip a random arc to ensure uniqueness
            flip_idx = np.random.randint(0, n_arcs)
            x[flip_idx] = 1 - x[flip_idx]
            x_tuple = tuple(x)
            if x_tuple not in solution_set:
                solutions.append(x.copy())
                solution_set.add(x_tuple)
            else:
                # If still duplicate, just accept it (extremely rare)
                solutions.append(x.copy())
                logger.warning(f"Accepting potentially duplicate solution for trial {trial+1}")
    
    logger.info(f"Successfully generated {len(solutions)} NE-SP Phase 0 solutions "
               f"({len(solution_set)} unique)")
    return solutions


def _generate_ufl_solutions(data, m_trials: int) -> List[np.ndarray]:
    """UFL Phase 0 generator - implementation removed, SCFLP focus only."""
    raise NotImplementedError("UFL Phase 0 generator removed - focusing on SCFLP only")


def _repair_ufl_solution(y: np.ndarray, capacities: np.ndarray, required_capacity: float) -> np.ndarray:
    """UFL repair function - implementation removed, SCFLP focus only."""
    raise NotImplementedError("UFL repair function removed - focusing on SCFLP only")


def _generate_uhl_solutions(data, m_trials: int) -> List[np.ndarray]:
    """UHL Phase 0 generator - implementation removed, SCFLP focus only."""
    raise NotImplementedError("UHL Phase 0 generator removed - focusing on SCFLP only")


def _repair_uhl_solution(y: np.ndarray, min_hubs: int) -> np.ndarray:
    """UHL repair function - implementation removed, SCFLP focus only."""
    raise NotImplementedError("UHL repair function removed - focusing on SCFLP only")


if __name__ == "__main__":
    """
    Test the Phase 0 generator with mock SCFLP data.
    
    This demonstrates how Phase 0 warm-up solutions are generated before
    the classic Benders decomposition (Phase 1) begins. The generated
    solutions will later be used to:
    1. Solve all subproblems exactly to collect optimal duals
    2. Generate cuts from these duals and add them to the master problem
    3. Combine these duals with Phase 1 duals for feature generation in Phase 2
    """
    # Set up logging for the test
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("Testing Phase 0 Generator with Mock SCFLP Data")
    print("=" * 50)
    
    # Create mock SCFLP data
    num_facilities = 8
    num_customers = 5
    num_scenarios = 3
    
    # Mock data arrays  
    J = np.arange(num_facilities)  # Facility indices
    I = np.arange(num_customers)   # Customer indices
    S = np.arange(num_scenarios)   # Scenario indices
    
    # Facility capacities (some facilities have higher capacity)
    capacities = np.array([100, 150, 80, 120, 90, 110, 130, 95])
    
    # Facility fixed costs
    fixed_costs = np.array([50, 60, 45, 55, 48, 52, 58, 46])
    
    # Customer demands per scenario (scenarios x customers)
    scenarios = np.array([
        [20, 15, 25, 18, 22],  # Scenario 0: total = 100
        [25, 18, 30, 20, 28],  # Scenario 1: total = 121 (highest demand)
        [18, 12, 20, 15, 19]   # Scenario 2: total = 84
    ])
    
    # Transportation costs (facilities x customers)
    shipment_costs = np.random.uniform(5, 15, size=(num_facilities, num_customers))
    
    # Calculate max demand across scenarios
    max_demand_sum = float(np.max(np.sum(scenarios, axis=1)))  # 121
    
    # Create SCFLPData object
    from dataclasses import dataclass
    
    @dataclass
    class MockSCFLPData:
        J: np.ndarray               # Facility indices
        I: np.ndarray               # Customer indices
        S: np.ndarray               # Scenario indices
        capacities: np.ndarray      # Facility capacities
        fixed_costs: np.ndarray     # Facility fixed costs
        scenarios: np.ndarray       # Demand scenarios
        shipment_costs: np.ndarray  # Transportation costs
        max_demand_sum_over_scenario: float  # Pre-computed max demand
    
    mock_data = MockSCFLPData(
        J=J, I=I, S=S, 
        capacities=capacities, 
        fixed_costs=fixed_costs, 
        scenarios=scenarios,
        shipment_costs=shipment_costs,
        max_demand_sum_over_scenario=max_demand_sum
    )
    
    print(f"Mock SCFLP Instance:")
    print(f"  Facilities: {num_facilities}")
    print(f"  Customers: {num_customers}")
    print(f"  Scenarios: {num_scenarios}")
    print(f"  Total available capacity: {np.sum(capacities)}")
    print(f"  Worst-case demand: {max_demand_sum}")
    print()
    
    # Generate Phase 0 solutions
    m_trials = 5
    print(f"Generating {m_trials} Phase 0 warm-up solutions...")
    print()
    
    try:
        solutions = generate_phase0_solutions(mock_data, m_trials, 'SCFLP')
        
        print("Generated Feasible Phase 0 Solutions:")
        print("-" * 40)
        for i, solution in enumerate(solutions):
            total_capacity = np.sum(capacities * solution)
            num_open = np.sum(solution)
            open_facilities = np.where(solution == 1)[0]
            
            print(f"Solution {i+1}: {solution}")
            print(f"  Open facilities: {list(open_facilities)} ({num_open}/{num_facilities})")
            print(f"  Total capacity: {total_capacity}")
            print(f"  Feasible: {total_capacity >= max_demand_sum}")
            print()
            
        print("Phase 0 Integration Notes:")
        print("- These solutions will be used to solve all subproblems exactly")
        print("- Optimal duals from subproblems will be collected and stored")
        print("- Cuts generated from these duals will strengthen the master problem")
        print("- Phase 1 (classic Benders) will begin with this strengthened master")
        print("- Duals from both Phase 0 and Phase 1 will be combined for ML in Phase 2")
        
    except Exception as e:
        print(f"Error generating Phase 0 solutions: {e}")
        import traceback
        traceback.print_exc()