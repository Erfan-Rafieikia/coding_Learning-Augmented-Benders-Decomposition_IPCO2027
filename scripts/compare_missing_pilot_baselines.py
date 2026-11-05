#!/usr/bin/env python3
"""Compute honest two-pilot times/objectives, using one Standard BD denominator.

New Random Sampling, Tree and Clustering runs share the recent P-median
resource/coding framework. All rows are labeled by cohort and exact problem.
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULTS = Path.home() / "scratch/Benders_Integration_with_ML/RESULTS"
RUN_BD = RESULTS / "MASTER_RUNS/pmedian_standard_bd_reference_20261019_125011"
RUN_PM = RESULTS / "MASTER_RUNS/parametric_objective_closure_20261019_021805"
RUN_GUARD = RESULTS / "MASTER_RUNS/parametric_guard_zero_fastpath_20261018_185736"

def read(root, task, problem):
    paths = list((RESULTS / f"{problem}_class" / "RESULTS" / root.name).glob(f"task_{task}_*/solution_log.xlsx"))
    if len(paths) != 1:
        raise ValueError(f"Expected one workbook for {root.name}/{problem}/{task}, got {paths}")
    df = pd.read_excel(paths[0])
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {paths[0]}")
    return df.iloc[0].to_dict()

def mapped(root):
    result = defaultdict(dict)
    manifest = root / "task_pairs.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest}")
    with manifest.open(newline="") as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            target = int(rec["pair_parametric_task_id"])
            method = rec.get("method", "BD")
            problem = rec["problem_class"]
            row = read(root, int(rec["new_task_id"]), problem)
            if method in result[target]:
                raise ValueError(f"Duplicate {target} / {method}")
            result[target][method] = row
    return result

def metric(row, field):
    try:
        z = float(row[field])
    except (KeyError, ValueError, TypeError):
        raise ValueError(f"Missing numeric {field}")
    if not math.isfinite(z):
        raise ValueError(f"Invalid {field}: {z}")
    return z

def verify(ref, other, target, name):
    for field in ("problem_class", "instance_filename", "num_subproblems"):
        if str(ref.get(field)) != str(other.get(field)):
            raise ValueError(f"{target}/{name}: mismatched {field}")
    if str(ref.get("problem_class")) == "SCMND" and str(ref.get("scmnd_formulation")) != str(other.get("scmnd_formulation")):
        raise ValueError(f"{target}/{name}: mismatched SCMND formulation")
    if str(other.get("stopping_reason", "")).upper() != "OPTIMAL":
        raise ValueError(f"{target}/{name}: solver stopping_reason={other.get('stopping_reason')!r}")
    if name != "Standard BD":
        if float(other.get("selected_subproblems_ratio", -1)) != 0.05:
            raise ValueError(f"{target}/{name}: mismatched selected ratio")

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/compare_missing_pilot_baselines.py NEW_BASELINE_ROOT")
    run = Path(sys.argv[1]).expanduser()
    bd, pm, guards, base = mapped(RUN_BD), mapped(RUN_PM), mapped(RUN_GUARD), mapped(run)
    required_arms = {
        "Standard BD": (bd, {"BD"}),
        "New baseline": (base, {"RANDOM", "TREE", "CLUSTERING"}),
        "Objective-aware pilot": (pm, {"NONPARAM", "STRUCT"}),
        "Guard pilot": (guards, {"OFF", "GUARD", "FAST"}),
    }
    for label, (group, expected) in required_arms.items():
        for target in (46, 199):
            actual = set(group.get(target, {}))
            missing = expected - actual
            if missing:
                raise ValueError(
                    f"{label}, target {target}: missing arms {sorted(missing)}; "
                    f"found {sorted(actual)}. Verify the run manifest and source root."
                )
    report = []
    stats = defaultdict(lambda: {"tau":[], "delta":[]})
    for target in (46,199):
        problem = "SCFLP" if target==46 else "SCMND"
        ref = bd[target]["BD"]
        if str(ref.get("method_category")).lower() != "exact":
            raise ValueError(f"{target}: reference not Exact")
        std_time, std_obj = metric(ref,"solution_time_sec"), metric(ref,"objective_value")
        methods = {
            "Standard BD": ref,
            "Random Sampling": base[target]["RANDOM"],
            "Nonparametric Tree": base[target]["TREE"],
            "Nonparametric Clustering": base[target]["CLUSTERING"],
            "Nonparametric P-median nearest": pm[target]["NONPARAM"],
            "Parametric P-median, original guard OFF": guards[target]["OFF"],
            "Parametric P-median, safeguard": guards[target]["GUARD"],
            "Parametric P-median, gamma-zero fastpath": guards[target]["FAST"],
            "Parametric P-median, objective-aware": pm[target]["STRUCT"],
        }
        print(f"\n{problem}: Standard BD runtime={std_time:.2f}s; objective={std_obj:.6f}")
        for name,row in methods.items():
            verify(ref,row,target,name)
            runtime = metric(row,"solution_time_sec")
            obj = metric(row,"objective_value" if name == "Standard BD" else "original_Prob_Obj_Value")
            tau = runtime/std_time
            delta = 100*(obj-std_obj)/abs(std_obj)
            stats[name]["tau"].append(tau)
            stats[name]["delta"].append(delta)
            report.append({"pilot":target,"problem":problem,"method":name,"method_time_sec":runtime,"bd_time_sec":std_time,"tau":tau,"objective_value":obj,"bd_objective":std_obj,"delta_pct":delta})
            print(f"  {name:46s} {runtime:8.2f}s  tau={tau:7.4f}  delta={delta:+8.4f}%")
    print("\nTWO-PILOT MEDIANS (do not mix with historical 92-case cohort):")
    print(f"{'Method':46s} {'Median tau':>10s} {'Median delta':>13s}")
    for name,d in stats.items():
        print(f"{name:46s} {statistics.median(d['tau']):10.4f} {statistics.median(d['delta']):+12.4f}%")
    out=run/"pmedian_two_pilot_comparison.csv"
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    print(f"\nSaved {out}")
    print("Warning: results were run in different batches, and wall-clock runtime may vary by node/time.")
if __name__=="__main__":
    main()
