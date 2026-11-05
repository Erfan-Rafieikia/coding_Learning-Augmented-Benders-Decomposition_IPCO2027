#!/usr/bin/env python3
"""Prepare an isolated, NO-SUBMISSION convergence/timing pilot matrix.

Select the same p4/S100 SCFLP and r01.1/S100 SCMND inputs for exact BD,
random sampling, nonparametric Tree/Cluster and RHS-Ridge P-median prediction.
Copy the original 1221-run overrides, replacing only SCMND formulation with
MANUSCRIPT_ENVELOPE in the NEW copies; preserve the originals.
The separate Slurm submitter sets the 1%/0.1% Gurobi gap via environment.
Never alter source configurations, original results or source data.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys

try:
    from .verify_convergence_snapshot import snapshot_source, write_hashes
except ImportError:
    from verify_convergence_snapshot import snapshot_source, write_hashes

METHODS = ("Standard_BD", "Random_Sampling", "TREE", "CLUSTERING",
           "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR")
CLASSES = ("SCFLP", "SCMND")
GAPS = (("1pct", "0.01"), ("0p1pct", "0.001"))


def parse_assignments(path):
    contents = path.read_text(encoding="utf-8")
    try:
        parsed = ast.parse(contents, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"Invalid original override {path}: {exc}") from exc
    out = {}
    for statement in parsed.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError, SyntaxError):
                        continue
    return out


def only_value(conf, key):
    x = conf.get(key)
    return x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else None


def classify(conf):
    if only_value(conf, "method_categories_grid") == "Exact":
        return "Standard_BD"
    if only_value(conf, "ml_method_types_grid") == "Random_Sampling":
        return "Random_Sampling"
    v = only_value(conf, "predictor_ML_Augmented_BD_grid")
    if v in METHODS:
        return v
    return None


def is_pilot_instance(problem, instance):
    normalized = instance.replace("\\", "/").lower()
    if problem == "SCFLP":
        return "scenarios_s100_" in normalized and normalized.endswith("/p4_stochastic.txt")
    return "s100_" in normalized and normalized.endswith("/r01.1.stoch.json")


def matching(conf, problem, method):
    paths = (conf.get("problem_classes_to_instances") or {}).get(problem)
    if not isinstance(paths, list) or len(paths) != 1:
        return None
    instance = paths[0]
    if not is_pilot_instance(problem, instance) or classify(conf) != method:
        return None
    if problem == "SCMND":
        # Historical source overrides may be soft; new copies are always hard.
        if only_value(conf, "scmnd_formulation_grid") not in {"CURRENT_SOFT_CAPACITY", "MANUSCRIPT_ENVELOPE"}:
            return None
    if method != "Standard_BD":
        ratio = only_value(conf, "selected_subproblems_ratio_grid")
        if ratio is None or abs(float(ratio) - 0.05) > 1e-10:
            return None
    return instance


def hard_capacity_override(contents, problem):
    """Migrate only a copied SCMND override; never modify the source config."""
    if problem != "SCMND":
        return contents
    parsed = ast.parse(contents)
    assignments = [node for node in parsed.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "scmnd_formulation_grid" for t in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Expected exactly one explicit SCMND formulation assignment")
    node = assignments[0]
    value = ast.literal_eval(node.value)
    if value not in (["CURRENT_SOFT_CAPACITY"], ["MANUSCRIPT_ENVELOPE"]):
        raise ValueError("Unexpected source SCMND formulation")
    lines = contents.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['scmnd_formulation_grid = ["MANUSCRIPT_ENVELOPE"]\n']
    return ''.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-root", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    args = ap.parse_args()
    master = args.master_root.expanduser().resolve()
    data = args.data_root.expanduser().resolve()
    results = args.results_root.expanduser().resolve()
    # Full-campaign preparation uses manifest.tsv plus configs/task_NNNN.py,
    # not the task_list.txt format used by older experiment launchers.
    manifest = master / "manifest.tsv"
    if not manifest.is_file() or not (master / "config.py").is_file():
        ap.error(f"Original manifest.tsv and config.py absent: {master}")
    if not data.is_dir():
        ap.error(f"Original source data directory absent: {data}")
    with manifest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        expected_columns = {"task_id", "problem", "instance_path", "method",
                            "repair", "selected_ratio"}
        if not expected_columns.issubset(reader.fieldnames or []):
            ap.error(f"Unexpected original manifest columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 1221 or sorted(int(r["task_id"]) for r in rows) != list(range(1, 1222)):
        ap.error(f"Expected 1221 original experiments, got {len(rows)}")
    method_lookup = {
        "Standard_BD": "STANDARD_BD",
        "Random_Sampling": "RANDOM",
        "TREE": "TREE",
        "CLUSTERING": "CLUSTERING",
        "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR":
            "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR",
    }
    matched = {}
    for row in rows:
        problem, instance, source_method = (
            row["problem"], row["instance_path"], row["method"]
        )
        if problem not in CLASSES or not is_pilot_instance(problem, instance):
            continue
        for method, manifest_method in method_lookup.items():
            if source_method != manifest_method:
                continue
            if method != "Standard_BD":
                if abs(float(row["selected_ratio"]) - 0.05) > 1e-10:
                    continue
            if method.startswith("PARAMETRIC_") and row["repair"] != "LOCAL":
                continue
            source_id = int(row["task_id"])
            if not 1 <= source_id <= 1221:
                ap.error(f"Invalid original task ID: {source_id}")
            rel = f"configs/task_{source_id:04d}.py"
            file = master / rel
            if not file.is_file():
                ap.error(f"Missing original override: {file}")
            conf = parse_assignments(file)
            if matching(conf, problem, method) != instance:
                ap.error(f"Manifest/config disagreement on {source_id}: {rel}")
            key = (problem, method)
            if key in matched:
                ap.error(f"Ambiguous original pilot selection {key}: {source_id}")
            if not (data / instance).is_file():
                ap.error(f"Source dataset missing: {data / instance}")
            matched[key] = (source_id, rel, instance)
    missing = [(p, m) for p in CLASSES for m in METHODS if (p, m) not in matched]
    if missing:
        ap.error("Missing matched original pilot configurations: "
                 + ", ".join(f"{p}:{m}" for p, m in missing))
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = results / "MASTER_RUNS" / f"labd_convergence_pilot_{now}"
    if root.exists():
        ap.error(f"Pilot output root already exists: {root}")
    (root / "configs").mkdir(parents=True)
    (root / "SLURM_LOGS").mkdir()
    commit = snapshot_source(Path(__file__).resolve().parents[1], root)
    shutil.copyfile(master / "config.py", root / "config.py")
    with (root / "task_list.txt").open("w", encoding="utf-8") as tasks, (
        root / "pilot_manifest.tsv"
    ).open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf, delimiter="\t")
        writer.writerow(("task_id", "gap", "problem", "method", "source_task_id", "instance", "scmnd_formulation"))
        idx = 0
        # One globally throttled array contains both independent gap targets.
        for gap_label, gap in GAPS:
            for problem in CLASSES:
                for method in METHODS:
                    source_id, rel, instance = matched[(problem, method)]
                    idx += 1
                    dest = f"configs/task_{idx:04d}.py"
                    (root / dest).write_text(hard_capacity_override((master / rel).read_text(), problem), encoding="utf-8")
                    tasks.write(f"{dest}|{problem}\n")
                    writer.writerow((idx, gap, problem, method, source_id, instance, "MANUSCRIPT_ENVELOPE" if problem == "SCMND" else ""))
    (root / "original_master.txt").write_text(str(master) + "\n")
    (root / "source_data_root.txt").write_text(str(data) + "\n")
    (root / "gap_ranges.txt").write_text("1-10: 0.01\n11-20: 0.001\n")
    write_hashes(root, data, [item[2] for item in matched.values()])
    print(f"Frozen source commit: {commit}")
    print(f"Prepared {idx} pilot tasks (NO JOBS SUBMITTED).")
    print("Per gap: 5 methods x 2 problems = 10 tasks. SCMND: MANUSCRIPT_ENVELOPE only.")
    print(f"Pilot root: {root}")
    print("Task manifest:")
    print((root / "pilot_manifest.tsv").read_text()[:5000])
    print(f"Next: bash scripts/submit_convergence_pilots.sh {root}")

if __name__ == "__main__":
    main()
