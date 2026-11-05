#!/usr/bin/env python3
"""Build LABD input files for the external literature benchmarks.

This script is ADDITIVE ONLY.  It never edits src/, the original DATA/
directory, or any frozen campaign.  It writes a separate data root whose
files use exactly the formats already read by the frozen solver
(src/Problem/data.py at 89331ce):

SCFLP  OR-Library CAP instances (Beasley 1988; copies distributed with the
       SCIP SCFLP example).  CAP allocation costs are for a customer's whole
       demand, so they are divided by the customer's demand to get unit costs
       (the same conversion as SCIP's reader).  Stochastic scenarios are then
       generated with the UNCHANGED project generator
       src/utils/SCFLP_data_generation_version4.process_files(
           method="merve_with_covariate_with_random_demand_capacity",
           sigma 0.1-0.3, 6 covariates, seed 42)
       i.e. the same method used for the current Holmberg data.

SCFLP  Holmberg p-instances (corrected reading, src/utils/holmberg_format.py):
       same generator and settings, written to SCFLP_HOLMBERG/.

SCMND  Crainic R networks (.dow, copies from the Bertsimas et al. IJOC 2025
       repository) with the PUBLISHED Rahmaniani et al. demand scenarios
       (1,000 per network class and correlation level).  The first S
       scenarios are used (the convention of the Bertsimas et al. code),
       weights 1/S, deterministic capacities replicated, and 5 SVD covariates
       computed with the UNCHANGED src/utils/SUMND_data_generation helpers.

Usage:
  python external_benchmarks/build_external_datasets.py --out-root DIR
         [--scflp cap41 cap44 ...] [--scmnd r05.1 r05.2 ...]
         [--scenarios 100 300 500] [--correlation 0]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "utils"))

import SCFLP_data_generation_version4 as scflp_gen  # noqa: E402  (unchanged project code)
import SUMND_data_generation as sumnd_gen  # noqa: E402  (unchanged project code)
from holmberg_format import read_holmberg_instance  # noqa: E402

SOURCES = HERE / "sources"
CAP_DIR = SOURCES / "cap_orlib_via_scip"
DOW_DIR = SOURCES / "r_networks_crainic"
RAH_DIR = SOURCES / "r_scenarios_rahmaniani"
HOLMBERG_DIR = REPO / "DATA" / "SCFLP"  # Holmberg p1..p71, tracked in git

DEFAULT_SCFLP = ["cap41", "cap44", "cap61", "cap64", "cap71"]  # in SCIP data AND Ramirez-Pico et al. (cap41-44, 61-64, 71-74)
DEFAULT_HOLMBERG: list = []  # corrected Holmberg data is regenerated in DATA/ on Nibi (scripts/regenerate_corrected_data.py)
DEFAULT_SCMND = ["r04.1", "r04.3", "r05.1", "r05.3", "r06.1", "r06.3"]  # classes/variants used by Rahmaniani et al. and Ramirez-Pico et al.
SCFLP_SIGMA = (0.1, 0.3)
SCFLP_COVARIATES = 6
SCMND_COVARIATES = 5
SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- SCFLP
def read_cap(path: Path):
    """OR-Library CAP: m n; m x (capacity fixed_cost); n x (demand, m costs)."""
    words = path.read_text().split()
    m, n = int(words[0]), int(words[1])
    pos = 2
    capacity, fixed = [], []
    for _ in range(m):
        capacity.append(float(words[pos]))
        fixed.append(float(words[pos + 1]))
        pos += 2
    demand, total_cost = [], []
    for _ in range(n):
        demand.append(float(words[pos]))
        pos += 1
        total_cost.append([float(x) for x in words[pos:pos + m]])
        pos += m
    if pos != len(words):
        raise ValueError(f"{path.name}: {len(words) - pos} unread tokens")
    demand = np.array(demand)
    total_cost = np.array(total_cost)  # customers x facilities, whole-demand costs
    unit_cost = total_cost / demand[:, None]
    return np.array(capacity), np.array(fixed), demand, unit_cost, total_cost


def write_holmberg_layout(path: Path, capacity, fixed, demand, total_cost):
    """Write the Holmberg/OR-Library layout read by holmberg_format.read_holmberg_instance:
    m n / m x (capacity fixed) / n demands / m rows of n whole-demand costs."""
    m, n = len(capacity), len(demand)
    if np.any(capacity != np.round(capacity)):
        raise ValueError("generator writes capacities as int; non-integer capacity found")
    lines = [f"{m} {n}"]
    lines += [f"{int(capacity[i])} {repr(float(fixed[i]))}" for i in range(m)]
    lines.append(" ".join(repr(float(d)) for d in demand))
    for i in range(m):
        lines.append(" ".join(repr(float(total_cost[j, i])) for j in range(n)))
    path.write_text("\n".join(lines) + "\n")


def stage_cap(name: str, staging: Path):
    cap, fx, dem, unit, total = read_cap(CAP_DIR / f"{name}.cap")
    write_holmberg_layout(staging / name, cap, fx, dem, total)
    return cap, fx, dem, unit


def stage_holmberg(name: str, staging: Path):
    src = HOLMBERG_DIR / name
    shutil.copy2(src, staging / name)
    return read_holmberg_instance(str(src))


def build_scflp(names, scenario_counts, out_root: Path, log, family: str):
    """family 'CAP' -> SCFLP_CAP/..., 'HOLMBERG' -> SCFLP_HOLMBERG/... (corrected reading)."""
    stage = stage_cap if family == "CAP" else stage_holmberg
    staging = Path(tempfile.mkdtemp(prefix=f"{family.lower()}_stage_"))
    produced = []
    try:
        for name in names:
            cap, fx, dem, unit = stage(name, staging)
            # Round trip: the generator's read_dataset must recover exactly these arrays.
            d = scflp_gen.read_dataset(str(staging / name), num_scenarios=2, method="normal")
            assert np.array_equal(d.capacities, cap) and np.array_equal(d.fixed_costs, fx)
            assert np.array_equal(d.demands, dem) and np.array_equal(d.shipment_costs, unit)
        for s in scenario_counts:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                scflp_gen.process_files(
                    input_dir=str(staging), file_list=list(names), num_scenarios=s,
                    method="merve_with_covariate_with_random_demand_capacity",
                    output_format="txt", seed=SEED,
                    sigma_lower=SCFLP_SIGMA[0], sigma_upper=SCFLP_SIGMA[1],
                    num_covariates=SCFLP_COVARIATES,
                )
            log.write(buf.getvalue())
            folder = (f"scenarios_s{s}_sigma{SCFLP_SIGMA[0]}_{SCFLP_SIGMA[1]}"
                      f"_cov{SCFLP_COVARIATES}_merve_with_covariates_and_capacities")
            dest = out_root / f"SCFLP_{family}" / folder
            dest.mkdir(parents=True, exist_ok=True)
            for name in names:
                src = staging / folder / f"{name}_stochastic.txt"
                if not src.is_file():
                    raise RuntimeError(f"generator did not write {src}")
                shutil.copy2(src, dest / src.name)
                produced.append(dest / src.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return produced



SCIP_SEED = 1             # SCIP SCFLP reader default (reading/scflpreader/randomseed)
SCIP_SD = (0.1, 0.2)      # SCIP reader_scflp.c: stdDev = U(0.1*mean, 0.2*mean) (its doc says 0.3; code is 0.2)


def scip_scenarios(demand: np.ndarray, n: int) -> np.ndarray:
    """SCIP SCFLP example method (examples/SCFLP/src/reader_scflp.c): for each customer, draw
    sd ~ U(0.1 d, 0.2 d) once, then n Normal(d, sd) demands rounded to the nearest integer.
    Same loop order (customer-major); numpy's generator replaces SCIP's, so the method is the
    same but the numbers are not SCIP's.  Negative draws (practically impossible at sd <= 0.2 d)
    are clipped to 0."""
    rng = np.random.default_rng(SCIP_SEED)
    out = np.zeros((n, len(demand)))
    for j, mean in enumerate(demand):
        sd = rng.uniform(SCIP_SD[0] * mean, SCIP_SD[1] * mean)
        out[:, j] = np.floor(rng.normal(mean, sd, size=n) + 0.5)
    return np.maximum(out, 0.0)


def write_scflp_file(path: Path, name, cap, fx, dem, unit, scen, cov):
    """Same text format as SCFLP_data_generation_version4 (read by src/Problem/data.read_scflp_data)."""
    m, n, S = len(cap), len(dem), len(scen)
    fmt = lambda x: str(int(x)) if float(x).is_integer() else str(x)
    L = [f"# Stochastic CFLP Instance generated from {name} (OR-Library CAP)", "# Parameters",
         f"#   scenarios = {S}", "#   target_cv = null",
         f"#   sigma_lower = {SCIP_SD[0]}", f"#   sigma_upper = {SCIP_SD[1]}",
         f"#   num_covariates = {cov.shape[1]}", f"#   seed = {SCIP_SEED}",
         "#   distribution = scip_scflp_normal_rounded", "", f"{m} {n} {S}", "",
         "# Facility capacities", " ".join(str(int(c)) for c in cap), "",
         "# Fixed costs", " ".join(fmt(x) for x in fx), "",
         "# Original demands (means)", " ".join(map(str, dem)), "",
         "# Shipment costs (customers x facilities)"]
    L += [" ".join(map(str, row)) for row in unit]
    L += ["", "# Demand scenarios"]
    for i, row in enumerate(scen):
        L += [f"# Scenario {i + 1}", " ".join(map(str, row))]
    L += ["", "# Covariate vectors (PCA-generated covariates)", f"# Number of covariates: {cov.shape[1]}"]
    for i, row in enumerate(cov):
        L += [f"# Covariate vector for scenario {i + 1}", " ".join(map(str, row))]
    L += ["", "# Diagnostics (single-file dispersion summaries)",
          "#   source = SCIP examples/SCFLP method; covariates = top singular vectors of centred demand matrix"]
    path.write_text("\n".join(L) + "\n")


def build_scflp_scip(names, scenario_counts, out_root: Path):
    produced = []
    for name in names:
        cap, fx, dem, unit, _ = read_cap(CAP_DIR / f"{name}.cap")
        for s in scenario_counts:
            scen = scip_scenarios(dem, s)
            cov, _ = sumnd_gen.generate_covariates_svd(scen, SCFLP_COVARIATES)
            dest = out_root / "SCFLP_CAP" / f"scenarios_s{s}_scip_normal_sd0.1_0.2_cov{SCFLP_COVARIATES}"
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"{name}_stochastic.txt"
            write_scflp_file(path, name, cap, fx, dem, unit, scen, cov)
            produced.append(path)
    return produced


# ---------------------------------------------------------------- SCMND
def build_scmnd(names, scenario_counts, correlation, out_root: Path):
    produced = []
    for name in names:
        cls = name.split(".")[0]
        det = sumnd_gen.parse_deterministic_instance(DOW_DIR / f"{name}.dow")
        scen_file = RAH_DIR / f"{cls}-{correlation}-1000"
        raw = np.loadtxt(scen_file, skiprows=1)
        if raw.shape[1] - 1 != det.n_commodities:
            raise ValueError(f"{scen_file.name}: {raw.shape[1]-1} demands vs {det.n_commodities} commodities")
        for s in scenario_counts:
            if s > raw.shape[0]:
                print(f"SKIP {name} S={s}: only {raw.shape[0]} published scenarios")
                continue
            demand = raw[:s, 1:].astype(float)
            if np.any(demand < 0):
                raise ValueError(f"{scen_file.name}: negative demand")
            capacity = np.tile([a.capacity for a in det.arcs], (s, 1)).astype(float)
            probs = np.full(s, 1.0 / s)
            cov, cov_info = sumnd_gen.generate_covariates_svd(demand, SCMND_COVARIATES)
            stoch = sumnd_gen.StochasticScenarios(
                demand_scenarios=demand, capacity_scenarios=capacity,
                probabilities=probs, covariates=cov)
            meta = {
                "generation_method": "Rahmaniani_published_scenarios",
                "stochastic_combination_type": "Demand_Only",
                "n_scenarios": s,
                "seed": None, "sigma_L": None, "sigma_U": None,
                "n_covariates": SCMND_COVARIATES,
                "include_deterministic_first": False,
                "source_file_name": f"{name}.dow",
                "scenario_source_file": scen_file.name,
                "scenario_source_rows": f"first {s} of 1000",
                "rahmaniani_correlation_label": correlation,
                "scenario_source": "https://github.com/Ragheb2464/R-Instances",
                "network_source": "https://github.com/INFORMSJoC/2023.0074 (benchmarks/R)",
                "covariate_info": cov_info,
            }
            stats = sumnd_gen.calculate_statistics(det, stoch, "Demand_Only") \
                if hasattr(sumnd_gen, "calculate_statistics") else None
            output = sumnd_gen.create_output_json(det, stoch, meta, stats)
            # Fixed timestamp so rebuilding gives byte-identical files.
            output["metadata"]["generation_timestamp"] = "2026-11-01T00:00:00"
            if isinstance(output.get("statistics"), dict) and "generation_timestamp" in output["statistics"]:
                output["statistics"]["generation_timestamp"] = "2026-11-01T00:00:00"
            path = (out_root / "SUMND" / f"R_rahmaniani_corr{correlation}"
                    / f"S{s}_published_cov{SCMND_COVARIATES}" / f"{name}.stoch.json")
            sumnd_gen.write_json_output(output, path, overwrite=True)
            produced.append(path)
    return produced


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--scflp", nargs="*", default=DEFAULT_SCFLP)
    ap.add_argument("--holmberg", nargs="*", default=DEFAULT_HOLMBERG)
    ap.add_argument("--scmnd", nargs="*", default=DEFAULT_SCMND)
    ap.add_argument("--scenarios", nargs="*", type=int, default=[100, 300, 500, 700, 900, 1000, 1100])
    ap.add_argument("--correlation", default="0", choices=["0", "0.2", "0.4", "0.6", "0.8"])
    args = ap.parse_args()
    out = args.out_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scflp_generator_stdout.log").open("w") as log:
        files = build_scflp_scip(args.scflp, args.scenarios, out)
        if args.holmberg:
            files += build_scflp(args.holmberg, args.scenarios, out, log, "HOLMBERG")
    files += build_scmnd(args.scmnd, args.scenarios, args.correlation, out)
    provenance = {
        "builder": "external_benchmarks/build_external_datasets.py",
        "python": platform.python_version(), "numpy": np.__version__,
        "scflp_instances": args.scflp, "holmberg_instances_corrected": args.holmberg, "scmnd_instances": args.scmnd,
        "scenarios": args.scenarios, "rahmaniani_correlation": args.correlation,
        "scflp_method": "SCIP SCFLP example: Normal(d, U(0.1d,0.2d)) rounded, seed 1; 6 demand-SVD covariates",
        "files": {str(p.relative_to(out)): sha256(p) for p in sorted(files)},
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(files)} instance files under {out}")


if __name__ == "__main__":
    main()
