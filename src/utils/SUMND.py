# pyright: reportMissingImports=false

"""Standalone Gurobi solver for the project SUMND instance.

This script solves the extensive-form stochastic uncapacitated multicommodity
network design model directly in Gurobi, without Benders decomposition and
without the artificial recourse arcs used only to force complete recourse in the
decomposition-based implementation.

Mathematical formulation
========================

Sets and indices:
	N: nodes
	A: directed real arcs
	K: commodities
	S: scenarios

Decision variables:
	y_a in {0,1}          for each a in A
	x_{a,k}^s >= 0        for each a in A, k in K, s in S

Objective:

	min  sum_{a in A} f_a y_a
	   + sum_{s in S} p_s (sum_{k in K} sum_{a in A} c_a^k x_{a,k}^s)

Constraints:

	sum_{j in N^+(i)} x_{ij,k}^s - sum_{j in N^-(i)} x_{ji,k}^s = d_{i,k}^s
		for all i in N, k in K, s in S

	sum_{k in K} x_{ij,k}^s <= u_{ij}^s y_{ij}
		for all (i,j) in A, s in S

	y_{ij} in {0,1}
		for all (i,j) in A

	x_{ij,k}^s >= 0
		for all (i,j) in A, k in K, s in S

Here d_{i,k}^s is the node-commodity supply vector induced by scenario demand:
	+d_k^s at the origin of commodity k,
	-d_k^s at the destination of commodity k,
	0 otherwise.

Project-consistent data semantics
=================================

This standalone script intentionally reuses the SUMND reader from
src/Problem/data.py so that the parameters match the current project exactly:

1. Arc activation costs f_a are read from the deterministic arc data.
2. Arc flow costs c_a^k are built exactly as in the project: each arc's
   unit_cost is replicated across all commodities.
3. Scenario probabilities p_s and demands d_k^s are read directly from the
   stochastic JSON file.
4. Scenario capacities u_{ij}^s follow the same override used by the Benders
   project: for each scenario s, every real arc gets capacity equal to the
   total scenario demand sum_k d_k^s. That makes the formulation effectively
   uncapacitated while preserving the current codebase semantics.

What is deliberately removed
============================

This script does not use:
	- theta variables
	- scenario dual subproblems
	- Benders cuts
	- artificial OD recourse arcs
	- penalty M

Prompt text
===========

The PROMPT_TEXT constant below restates the modeling request in a reusable form.
Run with --print-prompt if you want to print it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from gurobipy import GRB, GurobiError, Model, quicksum


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from config import DATA_ROOT, problem_classes_to_instances
from Problem.data import SUMNDData, read_sumnd_data


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


PROMPT_TEXT = """Write a standalone Gurobi Python script for the SUMND instance used in this project, without Benders decomposition and without artificial recourse arcs. The script must solve the full extensive-form primal model directly. Use exactly the same parameter-reading semantics as the current project implementation for SUMND: read the same activation costs, the same arc unit costs, the same scenario probabilities, and the same stochastic demands; also preserve the project rule that overwrites every scenario arc capacity with the total demand of that scenario so the model behaves as uncapacitated. Do not include theta variables, dual subproblems, optimality cuts, or the artificial-arc penalty M. Use the mathematical model

min sum_{(i,j) in A} f_ij y_ij + sum_{s in S} p_s (sum_{k in K} sum_{(i,j) in A} c_ij^k x_ij^{ks})

subject to

sum_{j in N^+(i)} x_ij^{ks} - sum_{j in N^-(i)} x_ji^{ks} = d_i^{ks},    for all i in N, k in K, s in S,
sum_{k in K} x_ij^{ks} <= u_ij^s y_ij,                                    for all (i,j) in A, s in S,
y_ij in {0,1},                                                            for all (i,j) in A,
x_ij^{ks} >= 0,                                                           for all (i,j) in A, k in K, s in S.

Here d_i^{ks} equals +d_k^s at the origin node of commodity k, -d_k^s at the destination node, and 0 otherwise. The script should default to the instance currently selected in config.py for problem class SUMND and should report the final objective value, fixed-cost component, expected flow-cost component, and the set of opened arcs."""


# ─── Instance list ─────────────────────────────────────────────────────────────
# Add more absolute file paths here to solve additional instances in one run.
INSTANCES = [
    "/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/DATA/SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r01.1.stoch.json",
]
# ───────────────────────────────────────────────────────────────────────────────


def get_default_instance_path() -> Path:
	sumnd_instances = problem_classes_to_instances.get("SUMND", [])
	if not sumnd_instances:
		raise ValueError("config.py does not define a default SUMND instance")
	return Path(DATA_ROOT) / sumnd_instances[0]


def resolve_instance_path(instance_arg: str | None) -> Path:
	if instance_arg is None:
		return get_default_instance_path()

	candidate = Path(instance_arg)
	if candidate.is_absolute() and candidate.exists():
		return candidate

	project_relative = PROJECT_ROOT / candidate
	if project_relative.exists():
		return project_relative

	data_relative = Path(DATA_ROOT) / candidate
	if data_relative.exists():
		return data_relative

	if candidate.is_absolute():
		return candidate
	return data_relative


def build_node_arc_lists(data: SUMNDData) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
	outgoing: dict[int, list[int]] = {int(node): [] for node in data.N}
	incoming: dict[int, list[int]] = {int(node): [] for node in data.N}
	for arc_idx, (tail, head) in enumerate(data.A):
		outgoing[int(tail)].append(arc_idx)
		incoming[int(head)].append(arc_idx)
	return outgoing, incoming


def node_commodity_rhs(data: SUMNDData, scenario: int, node: int, commodity: int) -> float:
	demand = float(data.demand_scenarios[scenario, commodity])
	if node == int(data.origins[commodity]):
		return demand
	if node == int(data.destinations[commodity]):
		return -demand
	return 0.0


def build_model(data: SUMNDData) -> tuple[Model, Any, Any]:
	model = Model("SUMND_Extensive_Form")

	scenario_ids = [int(s) for s in data.S]
	arc_ids = list(range(len(data.A)))
	commodity_ids = [int(k) for k in data.K]
	outgoing, incoming = build_node_arc_lists(data)

	y = model.addVars(arc_ids, vtype=GRB.BINARY, name="y")
	x = model.addVars(scenario_ids, arc_ids, commodity_ids, lb=0.0, name="x")

	fixed_cost = quicksum(float(data.F[arc]) * y[arc] for arc in arc_ids)
	expected_flow_cost = quicksum(
		float(data.probabilities[s]) * float(data.C[arc, commodity]) * x[s, arc, commodity]
		for s in scenario_ids
		for arc in arc_ids
		for commodity in commodity_ids
	)
	model.setObjective(fixed_cost + expected_flow_cost, GRB.MINIMIZE)

	for s in scenario_ids:
		for commodity in commodity_ids:
			for node in [int(n) for n in data.N]:
				rhs = node_commodity_rhs(data, s, node, commodity)
				model.addConstr(
					quicksum(x[s, arc, commodity] for arc in outgoing[node])
					- quicksum(x[s, arc, commodity] for arc in incoming[node])
					== rhs,
					name=f"flow_s{s}_k{commodity}_n{node}",
				)

	for s in scenario_ids:
		for arc in arc_ids:
			model.addConstr(
				quicksum(x[s, arc, commodity] for commodity in commodity_ids)
				<= float(data.capacity_scenarios[s, arc]) * y[arc],
				name=f"capacity_s{s}_a{arc}",
			)

	return model, y, x


def summarize_solution(data: SUMNDData, y: Any, x: Any) -> dict[str, Any]:
	scenario_ids = [int(s) for s in data.S]
	arc_ids = list(range(len(data.A)))
	commodity_ids = [int(k) for k in data.K]

	opened_arcs = [
		{
			"arc_index": arc,
			"tail": int(data.A[arc][0]),
			"head": int(data.A[arc][1]),
			"fixed_cost": float(data.F[arc]),
		}
		for arc in arc_ids
		if y[arc].X > 0.5
	]

	fixed_cost_value = sum(float(data.F[arc]) * y[arc].X for arc in arc_ids)
	expected_flow_cost_value = sum(
		float(data.probabilities[s]) * float(data.C[arc, commodity]) * x[s, arc, commodity].X
		for s in scenario_ids
		for arc in arc_ids
		for commodity in commodity_ids
	)

	return {
		"objective_value": fixed_cost_value + expected_flow_cost_value,
		"fixed_cost_component": fixed_cost_value,
		"expected_flow_cost_component": expected_flow_cost_value,
		"num_open_arcs": len(opened_arcs),
		"opened_arcs": opened_arcs,
	}


def solve_sumnd_extensive_form(
	instance_path: Path,
	time_limit: float | None,
	mip_gap: float | None,
	threads: int | None,
	write_lp: str | None,
) -> dict[str, Any]:
	logger.info("Reading SUMND instance from %s", instance_path)
	data = read_sumnd_data(str(instance_path))

	logger.info(
		"Building extensive-form model: |N|=%d, |A|=%d, |K|=%d, |S|=%d",
		data.n_nodes,
		data.n_arcs,
		data.n_commodities,
		data.n_scenarios,
	)
	model, y, x = build_model(data)
	model.update()

	# Index lists needed when extracting solution values below
	scenario_ids = [int(s) for s in data.S]
	arc_ids = list(range(len(data.A)))
	commodity_ids = [int(k) for k in data.K]

	model.Params.OutputFlag = 1
	if time_limit is not None:
		model.Params.TimeLimit = time_limit
	if mip_gap is not None:
		model.Params.MIPGap = mip_gap
	if threads is not None:
		model.Params.Threads = threads

	if write_lp:
		model.write(write_lp)

	start = time.time()
	try:
		model.optimize()
	except GurobiError as exc:
		runtime = time.time() - start
		return {
			"instance_path": str(instance_path),
			"solver_status": None,
			"runtime_seconds": runtime,
			"model_statistics": {
				"num_variables": int(model.NumVars),
				"num_binary_variables": int(model.NumBinVars),
				"num_constraints": int(model.NumConstrs),
			},
			"error_type": "GurobiError",
			"error_message": str(exc),
			"license_limited": "size-limited license" in str(exc).lower(),
		}
	runtime = time.time() - start

	result: dict[str, Any] = {
		"instance_path": str(instance_path),
		"solver_status": int(model.Status),
		"runtime_seconds": runtime,
		"model_statistics": {
			"num_variables": int(model.NumVars),
			"num_binary_variables": int(model.NumBinVars),
			"num_constraints": int(model.NumConstrs),
		},
		"best_bound": float(model.ObjBound) if model.SolCount > 0 or model.Status != GRB.LOADED else None,
		"mip_gap": float(model.MIPGap) if model.SolCount > 0 else None,
	}

	if model.SolCount > 0:
		result.update(summarize_solution(data, y, x))
		# Raw y values: list of binary values in arc-index order
		result["y_values"] = [int(round(y[arc].X)) for arc in arc_ids]
		# Raw x values: only non-zero entries stored as "s,arc,k" -> value
		result["x_values"] = {
			f"{s},{arc},{commodity}": round(x[s, arc, commodity].X, 6)
			for s in scenario_ids
			for arc in arc_ids
			for commodity in commodity_ids
			if x[s, arc, commodity].X > 1e-9
		}

	return result


def export_to_excel(results: list[dict[str, Any]], output_path: Path) -> None:
	"""Write one row per solved instance to an Excel file.

	Columns:
	    instance_path          – path to the JSON input file
	    objective_value        – optimal (or best feasible) objective value
	    fixed_cost_component   – sum of f_a * y_a
	    expected_flow_cost     – sum_s p_s * sum_{a,k} c_a^k * x_{a,k}^s
	    solver_status          – Gurobi status code (None on license error)
	    runtime_seconds        – wall-clock solve time
	    mip_gap                – final MIP gap (None if no solution)
	    y_values               – list of binary values [y_0, y_1, ..., y_{|A|-1}]
	    x_values               – dict of non-zero flow values {"s,arc,k": value}
	"""
	try:
		import pandas as pd  # type: ignore[import]
	except ImportError:
		from openpyxl import Workbook  # type: ignore[import]
		wb = Workbook()
		ws = wb.worksheets[0]
		ws.title = "Results"
		headers = [
			"instance_path", "objective_value", "fixed_cost_component",
			"expected_flow_cost", "solver_status", "runtime_seconds",
			"mip_gap", "y_values", "x_values",
		]
		ws.append(headers)
		for r in results:
			ws.append([
				r.get("instance_path"),
				r.get("objective_value"),
				r.get("fixed_cost_component"),
				r.get("expected_flow_cost_component"),
				r.get("solver_status"),
				r.get("runtime_seconds"),
				r.get("mip_gap"),
				json.dumps(r.get("y_values")),
				json.dumps(r.get("x_values")),
			])
		wb.save(str(output_path))
		logger.info("Results written to %s", output_path)
		return

	rows = []
	for r in results:
		rows.append({
			"instance_path":        r.get("instance_path"),
			"objective_value":      r.get("objective_value"),
			"fixed_cost_component": r.get("fixed_cost_component"),
			"expected_flow_cost":   r.get("expected_flow_cost_component"),
			"solver_status":        r.get("solver_status"),
			"runtime_seconds":      r.get("runtime_seconds"),
			"mip_gap":              r.get("mip_gap"),
			# Store as compact JSON strings so each value fits in one Excel cell
			"y_values":             json.dumps(r.get("y_values")),
			"x_values":             json.dumps(r.get("x_values")),
		})
	df = pd.DataFrame(rows)
	df.to_excel(str(output_path), index=False)
	logger.info("Results written to %s", output_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Solve the SUMND extensive form directly with Gurobi")
	parser.add_argument(
		"--instance",
		type=str,
		default=None,
		help="Absolute path, project-relative path, or DATA_ROOT-relative path to a SUMND JSON instance",
	)
	parser.add_argument("--time-limit", type=float, default=None, help="Optional Gurobi time limit in seconds")
	parser.add_argument("--mip-gap", type=float, default=None, help="Optional relative MIP gap")
	parser.add_argument("--threads", type=int, default=None, help="Optional Gurobi thread count")
	parser.add_argument("--write-lp", type=str, default=None, help="Optional path to write the LP/MPS model file")
	parser.add_argument("--write-json", type=str, default=None, help="Optional path to write a JSON summary")
	parser.add_argument("--print-prompt", action="store_true", help="Print the reusable prompt text and exit")
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	if args.print_prompt:
		print(PROMPT_TEXT)
		return 0

	# ── Build the list of instances to solve ────────────────────────────────────
	# If --instance was passed on the command line, solve only that one file;
	# otherwise iterate over the INSTANCES list defined at the top of this file.
	if args.instance is not None:
		instances_to_solve = [resolve_instance_path(args.instance)]
	else:
		instances_to_solve = [Path(p) for p in INSTANCES]

	# ── Solve each instance ──────────────────────────────────────────────────────
	all_results: list[dict[str, Any]] = []
	for instance_path in instances_to_solve:
		logger.info("=" * 60)
		logger.info("Solving instance: %s", instance_path)
		result = solve_sumnd_extensive_form(
			instance_path=instance_path,
			time_limit=args.time_limit,
			mip_gap=args.mip_gap,
			threads=args.threads,
			write_lp=args.write_lp,
		)
		all_results.append(result)
		print(json.dumps(
			{k: v for k, v in result.items() if k not in ("y_values", "x_values")},
			indent=2,
		))

	# ── Optional per-run JSON dump ───────────────────────────────────────────────
	if args.write_json:
		output_path = Path(args.write_json)
		output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
		logger.info("Wrote JSON summary to %s", output_path)

	# ── Excel export ─────────────────────────────────────────────────────────────
	excel_path = PROJECT_ROOT / "SUMND_results.xlsx"
	export_to_excel(all_results, excel_path)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
