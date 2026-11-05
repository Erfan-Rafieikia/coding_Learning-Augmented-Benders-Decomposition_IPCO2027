"""
Read and parse stochastic Capacitated Facility Location Problem (SCFLP) data from text files.
This module provides functionality to read multiple SCFLP instances from text files,
parse their contents, and return structured data for further processing.
"""


import os
import numpy as np
from dataclasses import dataclass
import re

@dataclass
class Data:
    I: np.ndarray  # Customer index list
    J: np.ndarray  # Facility index list
    S: np.ndarray  # Scenario index list
    demands: np.ndarray  # Customer demands
    scenarios: np.ndarray  # Demand Scenarios
    capacities: np.ndarray  # Facility capacities
    fixed_costs: np.ndarray  # Facility opening costs
    shipment_costs: np.ndarray  # Transportation costs
    max_demand_sum_over_scenario: float  # maximum of demand sums under scenarios
    scenario_generation_variance: float
    scenario_generation_method: str

def parse_txt_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract generation parameters from header
    header_text = "".join(lines[:10])  # just the top part
    var_match = re.search(r"variance\s*=\s*([\d.]+)", header_text)
    method_match = re.search(r"method\s*=\s*([a-zA-Z]+)", header_text)
    scenario_variance = float(var_match.group(1)) if var_match else None
    scenario_method = method_match.group(1) if method_match else None

    # Skip comment lines and empty lines
    content_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    # Read structure
    m, n, s = map(int, content_lines[0].split())  # facilities, customers, scenarios

    capacities = np.array(list(map(float, content_lines[1].split())))
    fixed_costs = np.array(list(map(float, content_lines[2].split())))
    demands = np.array(list(map(float, content_lines[3].split())))

    # Read shipment costs
    shipment_cost_lines = content_lines[4:4+n]
    shipment_costs = np.array([list(map(float, row.split())) for row in shipment_cost_lines])

    # Read scenarios - skip comment lines that start with "# Scenario"
    scenario_start_idx = 4 + n
    scenarios = []
    i = scenario_start_idx
    while len(scenarios) < s and i < len(content_lines):
        line = content_lines[i].strip()
        # Skip lines that are comments about scenarios
        if not line.startswith("# Scenario") and line:
            try:
                scenario_data = list(map(float, line.split()))
                if len(scenario_data) == n:  # Valid scenario with correct number of customers
                    scenarios.append(scenario_data)
            except ValueError:
                pass  # Skip lines that can't be parsed as numbers
        i += 1
    
    scenarios = np.array(scenarios)

    # Look for max demand line - it should be at the end and might be a comment
    max_demand = None
    for line in reversed(lines):
        if "max" in line.lower() and "demand" in line.lower():
            numbers = re.findall(r"[\d.]+", line)
            if numbers:
                max_demand = float(numbers[0])
                break

    I = np.arange(n)
    J = np.arange(m)
    S = np.arange(s)

    return Data(
        I=I,
        J=J,
        S=S,
        demands=demands,
        scenarios=scenarios,
        capacities=capacities,
        fixed_costs=fixed_costs,
        shipment_costs=shipment_costs,
        max_demand_sum_over_scenario=max_demand if max_demand is not None else 0.0,
        scenario_generation_variance=scenario_variance if scenario_variance is not None else 0.0,
        scenario_generation_method=scenario_method if scenario_method is not None else "unknown"
    )

def load_multiple_txt_instances(file_list, directory_path):
    """
    Reads multiple .txt stochastic CFLP instance files and returns a list of Data objects.

    Args:
        file_list (list[str]): Filenames (with or without `.txt`).
        directory_path (str): Path to the directory containing the files.

    Returns:
        list[Data]: List of parsed Data objects for use in optimization.
    """
    data_instances = []

    for fname in file_list:
        if not fname.endswith(".txt"):
            fname += ".txt"
        file_path = os.path.join(directory_path, fname)
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {fname}")
            continue
        try:
            data = parse_txt_file(file_path)
            data_instances.append(data)
            print(f"✅ Loaded: {fname}")
        except Exception as e:
            print(f"❌ Failed to parse {fname}: {e}")

    return data_instances


files = ["p1_stochastic", "p2_stochastic", "p5_stochastic"]
directory = "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP/scenarios_s100_v0.3_mnormal"

instances = load_multiple_txt_instances(files, directory)

# Example access:
if instances:
    print(f"First instance shipment_costs shape: {instances[0].shipment_costs.shape}")
    print(f"First instance scenario_generation_variance: {instances[0].scenario_generation_variance}")
    print(f"First instance scenario_generation_method: {instances[0].scenario_generation_method}")
    print(f"First instance scenarios shape: {instances[0].scenarios.shape}")
    print(f"First instance number of facilities: {len(instances[0].J)}")
    print(f"First instance number of customers: {len(instances[0].I)}")
    print(f"First instance number of scenarios: {len(instances[0].S)}")
    print(f"Max demand sum over scenarios: {instances[0].max_demand_sum_over_scenario}")
else:
    print("No instances loaded!")
