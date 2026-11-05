# pyright: reportMissingImports=false

"""Standalone multi-cut Benders decomposition solver for the SUMND instance.

This variant obtains feasibility-cut rays directly from Gurobi's UnbdRay
attribute on the dual subproblem variables, instead of solving a separate
normalized feasibility-ray LP.

The master model and Excel export are intentionally kept as close as possible
to SUMND_benders_multicut.py so that the two scripts can be run and compared.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gurobipy import GRB, GurobiError, LinExpr, Model, quicksum


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DATA_ROOT, problem_classes_to_instances
from Problem.data import SUMNDData, read_sumnd_data


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


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


def node_commodity_rhs(data: SUMNDData, scenario: int, node: int, commodity: int) -> float:
    demand = float(data.demand_scenarios[scenario, commodity])
    if node == int(data.origins[commodity]):
        return demand
    if node == int(data.destinations[commodity]):
        return -demand
    return 0.0


@dataclass
class CutData:
    constant_term: float
    y_coefficients: dict[int, float]
    scenario: int
    violation: float


class ScenarioDual:
    """Per-scenario dual subproblem used for both optimality and feasibility separation."""

    def __init__(self, data: SUMNDData, scenario: int, nodes: list[int], arcs: list[int], commodities: list[int]) -> None:
        self.data = data
        self.scenario = scenario
        self.nodes = nodes
        self.arcs = arcs
        self.commodities = commodities

        self.capacity = {a: float(data.capacity_scenarios[scenario, a]) for a in arcs}
        self.rhs = {
            (node, commodity): node_commodity_rhs(data, scenario, node, commodity)
            for node in nodes
            for commodity in commodities
        }

        model = Model(f"SP_dual_s{scenario}")
        model.Params.OutputFlag = 0
        model.Params.DualReductions = 0
        model.Params.InfUnbdInfo = 1
        model.Params.Method = 1  # simplex-based solve to make ray retrieval more reliable
        model.ModelSense = GRB.MAXIMIZE

        alpha = model.addVars(nodes, commodities, lb=-GRB.INFINITY, name="alpha")
        mu = model.addVars(arcs, lb=0.0, name="mu")

        for a in arcs:
            tail = int(data.A[a][0])
            head = int(data.A[a][1])
            for commodity in commodities:
                model.addConstr(
                    alpha[tail, commodity] - alpha[head, commodity] - mu[a] <= float(data.C[a, commodity]),
                    name=f"dual_arc_a{a}_k{commodity}",
                )

        model.update()

        self.model = model
        self.alpha = alpha
        self.mu = mu

    def _set_objective(self, y_values: dict[int, float]) -> None:
        obj = LinExpr()
        for (node, commodity), rhs_value in self.rhs.items():
            if abs(rhs_value) > 1e-12:
                obj += rhs_value * self.alpha[node, commodity]
        for a in self.arcs:
            coeff = self.capacity[a] * float(y_values[a])
            if abs(coeff) > 1e-12:
                obj += (-coeff) * self.mu[a]
        self.model.setObjective(obj, GRB.MAXIMIZE)

    def _extract_optimality_cut(self) -> CutData:
        constant_term = sum(
            rhs_value * float(self.alpha[node, commodity].X)
            for (node, commodity), rhs_value in self.rhs.items()
            if abs(rhs_value) > 1e-12
        )
        y_coefficients = {
            a: self.capacity[a] * float(self.mu[a].X)
            for a in self.arcs
            if abs(float(self.mu[a].X)) > 1e-12
        }
        return CutData(
            constant_term=float(constant_term),
            y_coefficients=y_coefficients,
            scenario=self.scenario,
            violation=0.0,
        )

    def _extract_unbounded_ray_cut(self, y_values: dict[int, float], tolerance: float) -> CutData:
        alpha_ray = {
            (node, commodity): float(self.alpha[node, commodity].UnbdRay)
            for node in self.nodes
            for commodity in self.commodities
        }
        mu_ray = {a: float(self.mu[a].UnbdRay) for a in self.arcs}

        # Optional scaling normalization for numerical stability.
        l1_norm = sum(abs(v) for v in alpha_ray.values()) + sum(abs(v) for v in mu_ray.values())
        if l1_norm <= 1e-12:
            raise RuntimeError(
                f"Scenario {self.scenario}: Gurobi returned an essentially zero unbounded ray."
            )

        constant_term = sum(
            rhs_value * alpha_ray[node, commodity]
            for (node, commodity), rhs_value in self.rhs.items()
            if abs(rhs_value) > 1e-12
        )
        y_coefficients = {
            a: self.capacity[a] * mu_ray[a]
            for a in self.arcs
            if abs(mu_ray[a]) > 1e-12
        }

        constant_term /= l1_norm
        y_coefficients = {a: coeff / l1_norm for a, coeff in y_coefficients.items()}

        violation = constant_term - sum(y_coefficients.get(a, 0.0) * float(y_values[a]) for a in self.arcs)
        if violation <= tolerance:
            raise RuntimeError(
                f"Scenario {self.scenario}: extracted unbounded ray does not yield a violated feasibility cut "
                f"(violation={violation:.6e})."
            )

        return CutData(
            constant_term=float(constant_term),
            y_coefficients=y_coefficients,
            scenario=self.scenario,
            violation=float(violation),
        )

    def solve(
        self,
        y_values: dict[int, float],
        tolerance: float,
    ) -> tuple[str, float | None, CutData | None]:
        self._set_objective(y_values)
        self.model.optimize()

        status = int(self.model.Status)

        # Rare safeguard: if solver returns INF_OR_UNBD, reset and re-optimize with current parameters.
        if status == GRB.INF_OR_UNBD:
            self.model.reset()
            self._set_objective(y_values)
            self.model.optimize()
            status = int(self.model.Status)

        if status == GRB.OPTIMAL:
            return "optimality", float(self.model.ObjVal), self._extract_optimality_cut()

        if status == GRB.UNBOUNDED:
            return "feasibility", None, self._extract_unbounded_ray_cut(y_values, tolerance)

        raise RuntimeError(f"Scenario {self.scenario}: unexpected dual status {status}.")


class ScenarioOracle:
    def __init__(self, data: SUMNDData, scenario: int, nodes: list[int], arcs: list[int], commodities: list[int]) -> None:
        self.scenario = scenario
        self.dual = ScenarioDual(data, scenario, nodes, arcs, commodities)

    def evaluate(
        self,
        y_values: dict[int, float],
        theta_value: float,
        cut_tol: float,
    ) -> tuple[str, CutData | None, float | None]:
        cut_type, recourse_value, cut_data = self.dual.solve(y_values=y_values, tolerance=cut_tol)

        if cut_type == "optimality":
            assert recourse_value is not None
            assert cut_data is not None
            violation = recourse_value - theta_value
            if violation > cut_tol:
                cut_data.violation = float(violation)
                return "optimality", cut_data, float(recourse_value)
            return "none", None, float(recourse_value)

        if cut_type == "feasibility":
            assert cut_data is not None
            return "feasibility", cut_data, None

        raise RuntimeError(f"Scenario {self.scenario}: unexpected cut type {cut_type}.")


class BendersCallbackContext:
    def __init__(
        self,
        data: SUMNDData,
        y_vars: Any,
        theta_vars: Any,
        scenario_oracles: dict[int, ScenarioOracle],
        arc_ids: list[int],
        scenario_ids: list[int],
        cut_tolerance: float,
    ) -> None:
        self.data = data
        self.y_vars = y_vars
        self.theta_vars = theta_vars
        self.scenario_oracles = scenario_oracles
        self.arc_ids = arc_ids
        self.scenario_ids = scenario_ids
        self.cut_tolerance = cut_tolerance

        self.callback_count = 0
        self.mipsol_count = 0
        self.optimality_cuts = 0
        self.feasibility_cuts = 0

    def callback(self, model: Model, where: int) -> None:
        self.callback_count += 1

        if where != GRB.Callback.MIPSOL:
            return

        self.mipsol_count += 1
        y_values = {a: float(model.cbGetSolution(self.y_vars[a])) for a in self.arc_ids}
        theta_values = {s: float(model.cbGetSolution(self.theta_vars[s])) for s in self.scenario_ids}

        for s in self.scenario_ids:
            cut_type, cut_data, _ = self.scenario_oracles[s].evaluate(
                y_values=y_values,
                theta_value=theta_values[s],
                cut_tol=self.cut_tolerance,
            )
            if cut_type == "none" or cut_data is None:
                continue

            if cut_type == "optimality":
                model.cbLazy(
                    self.theta_vars[s]
                    >= cut_data.constant_term
                    - quicksum(cut_data.y_coefficients.get(a, 0.0) * self.y_vars[a] for a in self.arc_ids)
                )
                self.optimality_cuts += 1
            elif cut_type == "feasibility":
                model.cbLazy(
                    quicksum(cut_data.y_coefficients.get(a, 0.0) * self.y_vars[a] for a in self.arc_ids)
                    >= cut_data.constant_term
                )
                self.feasibility_cuts += 1
            else:
                raise RuntimeError(f"Unexpected cut type: {cut_type}")


def build_master_problem(data: SUMNDData) -> tuple[Model, Any, Any, list[int], list[int], list[int]]:
    model = Model("SUMND_Benders_Master_DirectRay")
    scenario_ids = [int(s) for s in data.S]
    arc_ids = list(range(len(data.A)))
    node_ids = [int(n) for n in data.N]
    commodity_ids = [int(k) for k in data.K]

    y = model.addVars(arc_ids, vtype=GRB.BINARY, name="y")
    theta = model.addVars(scenario_ids, lb=0.0, name="theta")

    fixed_cost = quicksum(float(data.F[a]) * y[a] for a in arc_ids)
    expected_recourse = quicksum(float(data.probabilities[s]) * theta[s] for s in scenario_ids)
    model.setObjective(fixed_cost + expected_recourse, GRB.MINIMIZE)
    model.Params.LazyConstraints = 1

    return model, y, theta, node_ids, arc_ids, commodity_ids


def summarize_first_stage_solution(data: SUMNDData, y_values: dict[int, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arc in range(len(data.A)):
        rows.append(
            {
                "arc_index": arc,
                "tail": int(data.A[arc][0]),
                "head": int(data.A[arc][1]),
                "fixed_cost": float(data.F[arc]),
                "y_value": int(round(y_values[arc])),
            }
        )
    return rows


def compute_expected_recourse_cost(
    scenario_oracles: dict[int, ScenarioOracle],
    scenario_ids: list[int],
    probabilities: Any,
    y_values: dict[int, float],
) -> float:
    total = 0.0
    for s in scenario_ids:
        cut_type, recourse_value, _ = scenario_oracles[s].dual.solve(y_values=y_values, tolerance=1e-9)
        if cut_type != "optimality" or recourse_value is None:
            raise RuntimeError(f"Could not recover final recourse value for scenario {s}.")
        total += float(probabilities[s]) * float(recourse_value)
    return total


def export_to_excel(results: list[dict[str, Any]], output_path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"

    headers = [
        "instance_path",
        "solver_status",
        "objective_value",
        "fixed_cost_component",
        "expected_flow_cost_component",
        "runtime_seconds",
        "best_bound",
        "mip_gap",
        "num_feasibility_cuts",
        "num_optimality_cuts",
        "num_nodes_investigated",
        "num_callbacks",
        "num_mipsol_callbacks",
        "num_open_arcs",
        "opened_arcs_json",
        "y_values_json",
    ]
    ws.append(headers)

    header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for result in results:
        ws.append(
            [
                result.get("instance_path"),
                result.get("solver_status"),
                result.get("objective_value"),
                result.get("fixed_cost_component"),
                result.get("expected_flow_cost_component"),
                result.get("runtime_seconds"),
                result.get("best_bound"),
                result.get("mip_gap"),
                result.get("num_feasibility_cuts"),
                result.get("num_optimality_cuts"),
                result.get("num_nodes_investigated"),
                result.get("num_callbacks"),
                result.get("num_mipsol_callbacks"),
                result.get("num_open_arcs"),
                json.dumps(result.get("opened_arcs", [])),
                json.dumps(result.get("y_values", [])),
            ]
        )

    for column_cells in ws.columns:
        max_len = max(len(str(c.value)) if c.value is not None else 0 for c in column_cells)
        ws.column_dimensions[get_column_letter(column_cells[0].column)].width = min(max(max_len + 2, 14), 60)

    for row_idx, result in enumerate(results, start=2):
        sheet_name = f"y_solution_{row_idx - 1}"
        y_ws = wb.create_sheet(title=sheet_name[:31])
        y_headers = ["arc_index", "tail", "head", "fixed_cost", "y_value"]
        y_ws.append(y_headers)
        for col_idx, header in enumerate(y_headers, start=1):
            cell = y_ws.cell(row=1, column=col_idx, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for row in result.get("first_stage_solution_rows", []):
            y_ws.append([row[h] for h in y_headers])
        for column_cells in y_ws.columns:
            max_len = max(len(str(c.value)) if c.value is not None else 0 for c in column_cells)
            y_ws.column_dimensions[get_column_letter(column_cells[0].column)].width = min(max(max_len + 2, 12), 24)

    wb.save(str(output_path))
    logger.info("Results written to %s", output_path)


def solve_sumnd_benders_multicut_directray(
    instance_path: Path,
    time_limit: float | None,
    mip_gap: float | None,
    threads: int | None,
    write_lp: str | None,
    cut_tolerance: float,
) -> dict[str, Any]:
    logger.info("Reading SUMND instance from %s", instance_path)
    data = read_sumnd_data(str(instance_path))

    logger.info(
        "Building Benders master/subproblem data: |N|=%d, |A|=%d, |K|=%d, |S|=%d",
        data.n_nodes,
        data.n_arcs,
        data.n_commodities,
        data.n_scenarios,
    )

    model, y, theta, node_ids, arc_ids, commodity_ids = build_master_problem(data)
    scenario_ids = [int(s) for s in data.S]
    scenario_oracles = {
        s: ScenarioOracle(data, s, node_ids, arc_ids, commodity_ids)
        for s in scenario_ids
    }

    callback_ctx = BendersCallbackContext(
        data=data,
        y_vars=y,
        theta_vars=theta,
        scenario_oracles=scenario_oracles,
        arc_ids=arc_ids,
        scenario_ids=scenario_ids,
        cut_tolerance=cut_tolerance,
    )

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
        model.optimize(callback_ctx.callback)
    except GurobiError as exc:
        runtime = time.time() - start
        return {
            "instance_path": str(instance_path),
            "solver_status": None,
            "runtime_seconds": runtime,
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
        "best_bound": float(model.ObjBound) if model.Status != GRB.LOADED else None,
        "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None,
        "num_feasibility_cuts": callback_ctx.feasibility_cuts,
        "num_optimality_cuts": callback_ctx.optimality_cuts,
        "num_callbacks": callback_ctx.callback_count,
        "num_mipsol_callbacks": callback_ctx.mipsol_count,
        "num_nodes_investigated": int(model.NodeCount) if model.NodeCount is not None else None,
    }

    if model.SolCount > 0:
        y_values = {a: float(y[a].X) for a in arc_ids}
        theta_values = {s: float(theta[s].X) for s in scenario_ids}
        fixed_cost_value = sum(float(data.F[a]) * y_values[a] for a in arc_ids)
        expected_flow_cost_value = compute_expected_recourse_cost(
            scenario_oracles=scenario_oracles,
            scenario_ids=scenario_ids,
            probabilities=data.probabilities,
            y_values=y_values,
        )

        first_stage_solution_rows = summarize_first_stage_solution(data, y_values)
        opened_arcs = [row for row in first_stage_solution_rows if row["y_value"] == 1]

        result.update(
            {
                "objective_value": fixed_cost_value + expected_flow_cost_value,
                "fixed_cost_component": fixed_cost_value,
                "expected_flow_cost_component": expected_flow_cost_value,
                "num_open_arcs": len(opened_arcs),
                "opened_arcs": opened_arcs,
                "first_stage_solution_rows": first_stage_solution_rows,
                "y_values": [int(round(y_values[a])) for a in arc_ids],
                "theta_values": {str(s): theta_values[s] for s in scenario_ids},
            }
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the SUMND model with multi-cut Benders decomposition in Gurobi using direct UnbdRay extraction"
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Absolute path, project-relative path, or DATA_ROOT-relative path to a SUMND JSON instance",
    )
    parser.add_argument("--time-limit", type=float, default=None, help="Optional Gurobi time limit in seconds")
    parser.add_argument("--mip-gap", type=float, default=None, help="Optional relative MIP gap")
    parser.add_argument("--threads", type=int, default=None, help="Optional Gurobi thread count")
    parser.add_argument("--write-lp", type=str, default=None, help="Optional path to write the master LP/MPS model file")
    parser.add_argument("--write-json", type=str, default=None, help="Optional path to write a JSON summary")
    parser.add_argument(
        "--excel-out",
        type=str,
        default=None,
        help="Optional output path for the Excel workbook. Defaults to PROJECT_ROOT / 'SUMND_benders_directray_results.xlsx'",
    )
    parser.add_argument(
        "--cut-tol",
        type=float,
        default=1e-6,
        help="Tolerance for deciding whether a feasibility or optimality cut is violated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.instance is not None:
        instances_to_solve = [resolve_instance_path(args.instance)]
    else:
        instances_to_solve = [Path(p) for p in INSTANCES]

    all_results: list[dict[str, Any]] = []
    for instance_path in instances_to_solve:
        logger.info("=" * 80)
        logger.info("Solving instance with multi-cut Benders (direct UnbdRay): %s", instance_path)
        result = solve_sumnd_benders_multicut_directray(
            instance_path=instance_path,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            threads=args.threads,
            write_lp=args.write_lp,
            cut_tolerance=args.cut_tol,
        )
        all_results.append(result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("first_stage_solution_rows", "opened_arcs", "y_values", "theta_values")},
                indent=2,
            )
        )

    if args.write_json:
        output_path = Path(args.write_json)
        output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        logger.info("Wrote JSON summary to %s", output_path)

    excel_path = Path(args.excel_out) if args.excel_out else (PROJECT_ROOT / "SUMND_benders_directray_results.xlsx")
    export_to_excel(all_results, excel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
