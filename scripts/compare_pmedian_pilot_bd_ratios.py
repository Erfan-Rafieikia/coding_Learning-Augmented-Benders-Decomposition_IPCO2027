#!/usr/bin/env python3
"""Report target-matched Standard BD time ratios for the P-median repair pilots.

The historical 92-case median ratios CANNOT be compared as a single homogeneous
cohort with the two-instance targeted repair pilot reported here.
"""
import csv
import math
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

RESULTS = Path.home() / "scratch/Benders_Integration_with_ML/RESULTS"
ROOTS = {
    "latest": RESULTS / "MASTER_RUNS/parametric_objective_closure_20261019_021805",
    "structure": RESULTS / "MASTER_RUNS/parametric_structure_repair_recovery_20261019_011024",
    "fastpath": RESULTS / "MASTER_RUNS/parametric_guard_zero_fastpath_20261018_185736",
}
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

def column_index(address):
    result = 0
    for letter in address:
        if not letter.isalpha():
            break
        result = 26 * result + ord(letter.upper()) - ord("A") + 1
    return result - 1

def read_single_row_xlsx(path):
    # This intentionally reads values only, leaves every original XLSX untouched,
    # and avoids importing large 92-case experiment workbooks into memory.
    with zipfile.ZipFile(path) as archive:
        texts = []
        if "xl/sharedStrings.xml" in archive.namelist():
            with archive.open("xl/sharedStrings.xml") as stream:
                for _, element in ET.iterparse(stream, events=("end",)):
                    if element.tag == NS + "si":
                        texts.append("".join((t.text or "") for t in element.iter(NS + "t")))
                        element.clear()
        rows = []
        with archive.open("xl/worksheets/sheet1.xml") as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != NS + "row":
                    continue
                cells = {}
                for cell in element.findall(NS + "c"):
                    index = column_index(cell.attrib["r"])
                    v = cell.find(NS + "v")
                    if v is not None:
                        value = v.text or ""
                        if cell.attrib.get("t") == "s":
                            value = texts[int(value)]
                    else:
                        inline = cell.find(NS + "is")
                        if inline is None:
                            continue
                        value = "".join((t.text or "") for t in inline.iter(NS + "t"))
                    cells[index] = value
                if cells:
                    rows.append(cells)
                element.clear()
                if len(rows) > 2:
                    raise ValueError(f"Unexpected multiple result rows: {path}")
        if len(rows) != 2:
            raise ValueError(f"Expected one header and one result: {path}: {len(rows)}")
        return {name: rows[1].get(i) for i, name in rows[0].items() if name}

def num(row, key):
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required field {key}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}: {value}")
    return value

def mapped_rows(root, wanted_methods=None):
    root = Path(root)
    pairs_path = root / "task_pairs.tsv"
    if not pairs_path.is_file():
        raise ValueError(f"No task_pairs.tsv: {root}")
    result = defaultdict(dict)
    with pairs_path.open(newline="") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            target = int(item["pair_parametric_task_id"])
            method = item.get("method", "BD")
            if wanted_methods is not None and method not in wanted_methods:
                continue
            problem = item["problem_class"]
            idx = int(item["new_task_id"])
            base = RESULTS / f"{problem}_class" / "RESULTS" / root.name
            paths = list(base.glob(f"task_{idx}_*/solution_log.xlsx"))
            if len(paths) != 1:
                raise ValueError(f"Expected one result for {root.name}/{target}/{method}, got {paths}")
            if method in result[target]:
                raise ValueError(f"Duplicate {root.name}/{target}/{method}")
            result[target][method] = read_single_row_xlsx(paths[0])
    return result

def same_instance(ref, row, target, label):
    for field in ("instance_filename", "problem_class", "num_subproblems"):
        if str(row.get(field)) != str(ref.get(field)):
            raise ValueError(f"{target}/{label}: mismatch in {field}: {row.get(field)} vs {ref.get(field)}")
    if str(row.get("problem_class")) == "SCMND":
        if str(row.get("scmnd_formulation")) != str(ref.get("scmnd_formulation")):
            raise ValueError(f"{target}/{label}: SCMND formulation mismatch")
    for field in ("use_stochastic_capacity",):
        a, b = str(row.get(field)).lower(), str(ref.get(field)).lower()
        if a not in ("", "none") and b not in ("", "none") and a != b:
            raise ValueError(f"{target}/{label}: mismatch in {field}")

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/compare_pmedian_pilot_bd_ratios.py BD_REFERENCE_ROOT")
    bd_root = Path(sys.argv[1]).expanduser()
    bd = mapped_rows(bd_root)
    latest = mapped_rows(ROOTS["latest"])
    structure = mapped_rows(ROOTS["structure"], {"STRUCT"})
    fast = mapped_rows(ROOTS["fastpath"], {"OFF", "GUARD", "FAST"})
    wanted = {
        "NONPARAM": ("P-median nearest, nonparametric", latest),
        "OFF": ("P-median nearest, unguarded", fast),
        "GUARD": ("P-median nearest, standard guard", fast),
        "FAST": ("P-median nearest, guard fastpath", fast),
        "LINE": ("P-median nearest, guarded line repair", latest),
        "STRUCT": ("P-median nearest, structure-aware repair", structure),
        "OBJECTIVE": ("P-median nearest, objective-aware repair", latest),
    }
    targets = sorted(bd)
    if targets != [46, 199]:
        raise ValueError(f"Expected pilot targets [46,199], got {targets}")
    detail = []
    ratios = defaultdict(list)
    gaps = defaultdict(list)
    for target in targets:
        if set(bd[target]) != {"BD"}:
            raise ValueError(f"Expected one BD reference for {target}: {list(bd[target])}")
        ref = bd[target]["BD"]
        if str(ref.get("method_category")).lower() != "exact":
            raise ValueError(f"{target}: reference is not Standard BD")
        bd_time, bd_obj = num(ref, "solution_time_sec"), num(ref, "objective_value")
        if bd_time <= 0 or abs(bd_obj) < 1e-8:
            raise ValueError(f"{target}: invalid Standard BD baseline")
        rows = {
            "NONPARAM": latest[target]["NONPARAM"],
            "OFF": fast[target]["OFF"],
            "GUARD": fast[target]["GUARD"],
            "FAST": fast[target]["FAST"],
            "LINE": latest[target]["LINE"],
            "STRUCT": structure[target]["STRUCT"],
            "OBJECTIVE": latest[target]["STRUCT"],
        }
        print(f"\nTARGET {target} {ref.get('problem_class')}: BD = {bd_time:.2f} s; objective = {bd_obj:.4f}")
        for key, row in rows.items():
            label, _ = wanted[key]
            same_instance(ref, row, target, label)
            if key != "NONPARAM" and "P-median" in label:
                pass
            obj = num(row, "original_Prob_Obj_Value")
            tau = num(row, "solution_time_sec") / bd_time
            delta = 100.0 * (obj - bd_obj) / abs(bd_obj)
            ratios[key].append(tau)
            gaps[key].append(delta)
            detail.append({
                "target": target,
                "problem": ref["problem_class"],
                "method": label,
                "standard_bd_sec": bd_time,
                "method_sec": num(row, "solution_time_sec"),
                "time_ratio": tau,
                "standard_bd_obj": bd_obj,
                "reevaluated_method_obj": obj,
                "objective_difference_pct": delta,
                "run_source": ("fastpath" if key in ("OFF", "GUARD", "FAST") else
                               "structure_recovery" if key == "STRUCT" else "latest_objective_pilot"),
            })
            print(f"  {label}: tau={tau:.4f}; obj_diff={delta:+.4f}%")
    print("\nTWO-PILOT MEDIANS (not the historical 92-case benchmark):")
    print(f"  {'Method':44s} {'Median tau':>11s} {'Median obj diff':>16s}")
    for key, (label, _) in wanted.items():
        print(f"  {label:44s} {statistics.median(ratios[key]):11.4f} {statistics.median(gaps[key]):+15.4f}%")
    output = bd_root / "pmedian_pilot_standard_bd_ratios.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    print(f"Saved: {output}")
    print("Caution: the two-pilot medians and historical 92-case medians are different cohorts.")
    print("Caution: methods from distinct pilot batches may have ordinary solver runtime variability.")

if __name__ == "__main__":
    main()
