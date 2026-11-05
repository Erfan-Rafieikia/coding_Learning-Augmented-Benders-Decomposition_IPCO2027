"""
Generate stochastic instances from selected deterministic CFLP files.

This script reads a specified list of `.txt` instance files from an input directory,
generates stochastic demand scenarios (normal or uniform), and saves each
as a `.pkl` or `.txt` file in an output subdirectory automatically named based on
the scenario generation parameters.

Author: Your Name
"""

import os
import numpy as np
import pickle
from dataclasses import dataclass

@dataclass
class Data:
    I: np.ndarray
    J: np.ndarray
    S: np.ndarray
    demands: np.ndarray
    scenarios: np.ndarray
    capacities: np.ndarray
    fixed_costs: np.ndarray
    shipment_costs: np.ndarray
    max_demand_sum_over_scenario: float

def word_reader(file_path):
    with open(file_path, "r") as file:
        for line in file:
            for word in line.split():
                yield word

def generate_scenarios(demands, capacities, num_scenarios=10, variance_factor=0.2, method="normal", seed=42):
    """
    Generate demand scenarios ensuring feasibility.
    
    Args:
        demands: Original demand vector
        capacities: Facility capacity vector
        num_scenarios: Number of scenarios to generate
        variance_factor: Variance factor for demand generation
        method: Distribution method ("normal" or "uniform")
        seed: Random seed for reproducibility
    
    Returns:
        Array of feasible demand scenarios
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_capacity = np.sum(capacities)
    
    if method == "normal":
        scenarios = np.array([
            [max(0, np.random.normal(mu, variance_factor * mu)) for mu in demands]
            for _ in range(num_scenarios)
        ])
    elif method == "uniform":
        scenarios = np.array([
            [max(0, np.random.uniform(mu * (1 - variance_factor), mu * (1 + variance_factor))) for mu in demands]
            for _ in range(num_scenarios)
        ])
    else:
        raise ValueError("Unsupported method. Use 'normal' or 'uniform'.")
    
    # Ensure feasibility: scale down scenarios if total demand exceeds capacity
    max_scenario_demand = np.max(np.sum(scenarios, axis=1))
    
    if max_scenario_demand > total_capacity:
        # Calculate exact scaling factor needed: (capacity / max_demand) * safety_factor
        safety_factor = 0.95  # 5% buffer to ensure strict feasibility
        scaling_factor = (total_capacity / max_scenario_demand) * safety_factor
        
        # Apply scaling to all scenarios
        scenarios = scenarios * scaling_factor
        
        # Verify the result
        new_max_demand = np.max(np.sum(scenarios, axis=1))
        
        print(f"⚠️  Scenarios scaled to ensure feasibility:")
        print(f"   Original max demand: {max_scenario_demand:.2f}")
        print(f"   Total capacity: {total_capacity}")
        print(f"   Scaling factor: {scaling_factor:.4f} (= {total_capacity}/{max_scenario_demand:.2f} × {safety_factor})")
        print(f"   New max demand: {new_max_demand:.2f}")
        print(f"   ✅ Feasibility achieved: {new_max_demand:.2f} ≤ {total_capacity}")
    
    return scenarios

def read_dataset(file_path, num_scenarios=10, variance_factor=0.2, method="normal", seed=42):
    # Holmberg/OR-Library layout: (capacity, fixed cost) PAIRS per facility, and
    # c_ij is the cost of a customer's whole demand -> unit cost c_ij / d_j.
    # (Fixed 2026-11-01; earlier code mixed the pairs and used c_ij as a unit cost.)
    try:
        from holmberg_format import read_holmberg_instance
    except ImportError:
        from utils.holmberg_format import read_holmberg_instance
    capacities, fixed_costs, demands, shipment_costs = read_holmberg_instance(file_path)
    num_facilities, num_customers = len(capacities), len(demands)
    capacities = capacities.astype(int)
    fixed_costs = [int(x) if float(x).is_integer() else float(x) for x in fixed_costs]

    I = np.arange(num_customers)
    J = np.arange(num_facilities)
    S = np.arange(num_scenarios)

    scenarios = generate_scenarios(demands, capacities, num_scenarios, variance_factor, method, seed)
    max_demand_sum = max(np.sum(scenarios[s]) for s in S)

    return Data(
        I=I, J=J, S=S, demands=demands, scenarios=scenarios,
        capacities=capacities, fixed_costs=np.array(fixed_costs),
        shipment_costs=shipment_costs, max_demand_sum_over_scenario=max_demand_sum
    )

def process_files(input_dir, file_list, num_scenarios=10, variance=0.2, method="normal", output_format="txt", seed=42):
    """
    Process specific files from `file_list`, generate stochastic scenarios,
    and save them to a named subfolder inside `input_dir`.

    Args:
        input_dir (str): Directory where the original .txt files are located.
        file_list (list): List of file names (e.g., ["p1", "p2"]) without extensions.
        num_scenarios (int): Number of demand scenarios to generate.
        variance (float): Variance factor for demand generation.
        method (str): Distribution method ("normal" or "uniform").
        output_format (str): "txt" or "pkl"
        seed (int): Random seed for reproducible results.
    """
    folder_name = f"scenarios_s{num_scenarios}_v{variance}_m{method}"
    output_dir = os.path.join(input_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    for fname in file_list:
        full_path = os.path.join(input_dir, fname)
        if not os.path.exists(full_path):
            print(f"⚠️ File not found: {fname}")
            continue

        print(f"🔄 Processing: {fname}")
        try:
            data = read_dataset(full_path, num_scenarios, variance, method, seed)
            
            # Print feasibility information
            total_capacity = np.sum(data.capacities)
            print(f"   📊 Total capacity: {total_capacity}")
            print(f"   📊 Max scenario demand: {data.max_demand_sum_over_scenario:.2f}")
            print(f"   ✅ Feasibility ratio: {data.max_demand_sum_over_scenario/total_capacity:.4f}")
            
            if data.max_demand_sum_over_scenario <= total_capacity:
                print(f"   ✅ Instance is FEASIBLE")
            else:
                print(f"   ❌ Instance would be INFEASIBLE (this should not happen now)")

            if output_format == "txt":
                out_path = os.path.join(output_dir, f"{fname}_stochastic.txt")

                with open(out_path, "w") as f:
                    f.write(f"# Stochastic CFLP Instance generated from {fname}\n")
                    f.write(f"# Parameters:\n")
                    f.write(f"#   scenarios = {num_scenarios}\n")
                    f.write(f"#   variance  = {variance}\n")
                    f.write(f"#   method    = {method}\n")
                    f.write("\n")

                    f.write(f"{len(data.J)} {len(data.I)} {len(data.S)}\n\n")

                    f.write("# Facility capacities\n")
                    f.write(" ".join(map(str, data.capacities)) + "\n\n")

                    f.write("# Fixed costs\n")
                    f.write(" ".join(map(str, data.fixed_costs)) + "\n\n")

                    f.write("# Original demands\n")
                    f.write(" ".join(map(str, data.demands)) + "\n\n")

                    f.write("# Shipment costs (customers x facilities)\n")
                    for row in data.shipment_costs:
                        f.write(" ".join(map(str, row)) + "\n")
                    f.write("\n")

                    f.write("# Demand scenarios\n")
                    for idx, scenario in enumerate(data.scenarios):
                        f.write(f"# Scenario {idx + 1}\n")
                        f.write(" ".join(map(str, scenario)) + "\n")
                    f.write("\n")

                    f.write(f"# Max demand sum over scenarios: {data.max_demand_sum_over_scenario}\n")

            elif output_format == "pkl":
                out_path = os.path.join(output_dir, f"{fname}.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(data, f)

            else:
                raise ValueError("Invalid output_format. Use 'txt' or 'pkl'.")

        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

    print(f"\n✅ All selected files processed and saved to: {output_dir}")

# ======================
# Run the function here
# ======================
if __name__ == "__main__":
    selected_files = [
        "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
        "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20",
        "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29", "p30",
        "p31", "p32", "p33", "p34", "p35", "p36", "p37", "p38", "p39", "p40",
        "p41", "p42", "p43", "p44", "p45", "p46", "p47", "p48", "p49", "p50",
        "p51", "p52", "p53", "p54", "p55", "p56", "p57", "p58", "p59", "p60",
        "p61", "p62", "p63", "p64", "p65", "p66", "p67", "p68", "p69", "p70",
        "p71"
        # Add more filenames without `.txt` extension
    ]

    print("🚀 Generating full SCFLP stochastic dataset...")
    print("📋 Settings: scenarios=[100,400,800], variance=[0.1,0.3,0.5,1,2], methods=[normal,uniform]")
    
    # The setting to implement : num_scenarios=100,400,800 , variance=0.1, 0.3, 0.5, 1,2 , method=normal, uniform
    for num_scenarios in [200]:
        for variance in [2]:
            for method in ["normal", "uniform"]:
                print(f"\n📋 Processing: {num_scenarios} scenarios, variance={variance}, method={method}")
                process_files(
                    input_dir="/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SCFLP",
                    file_list=selected_files,
                    num_scenarios=num_scenarios,
                    variance=variance,
                    method=method,
                    output_format="txt",  # or "pkl"
                    seed=42  # Set a seed for reproducible results
                )