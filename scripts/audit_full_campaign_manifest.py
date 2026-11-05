#!/usr/bin/env python3
"""Read-only audit of the original 1,221-config LABD full campaign.

This tool does not edit or prepare configs, data, results, or submit any jobs.
It reports source SCMND formulation; any required hard-capacity migration
must be done later to separately frozen COPIES, not the original overrides.

Example:
  python scripts/audit_full_campaign_manifest.py \
    --master-root /scratch/.../labd_full_16cpu8g_CmLHQNO4 \
    --data-root ~/Erfan_projects/Benders_Integration_with_ML/DATA
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

from prepare_convergence_pilots import only_value, parse_assignments

EXPECTED_METHODS = frozenset({
    "STANDARD_BD", "RANDOM", "TREE", "CLUSTERING",
    "PMEDIAN_WEIGHTED_TRANSFER", "PMEDIAN_NEAREST_TRANSFER",
    "PARAMETRIC_RHS_RIDGE_REPAIR",
    "PARAMETRIC_RHS_RIDGE_TREE_ANCHOR",
    "PARAMETRIC_RHS_RIDGE_CLUSTER_ANCHOR",
    "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR",
})


def actual_method(conf):
    if only_value(conf, "method_categories_grid") == "Exact":
        return "STANDARD_BD"
    if only_value(conf, "ml_method_types_grid") == "Random_Sampling":
        return "RANDOM"
    return only_value(conf, "predictor_ML_Augmented_BD_grid")
REQUIRED_COLUMNS = {
    "task_id", "problem", "instance_label", "instance_path",
    "method", "repair", "selected_ratio",
}


def dataset_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(master, data):
    errors = []
    warnings = []
    manifest = master / "manifest.tsv"
    if not manifest.is_file() or not (master / "config.py").is_file():
        raise ValueError("Original manifest.tsv and config.py must exist")
    if not data.is_dir():
        raise ValueError("Original DATA directory does not exist")
    with manifest.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected manifest columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 1221:
        errors.append(f"Expected 1221 manifest records; found {len(rows)}")
    try:
        ids = [int(r["task_id"]) for r in rows]
    except (ValueError, KeyError) as exc:
        raise ValueError("Malformed task_id field") from exc
    if sorted(ids) != list(range(1, 1222)):
        errors.append("Task IDs are not exactly 1..1221")

    methods, problems, ratios, repairs, formations = (
        Counter(), Counter(), Counter(), Counter(), Counter()
    )
    per_instance = Counter()
    datasets = {}
    for row in rows:
        task_id = int(row["task_id"])
        problem = row["problem"]
        method = row["method"]
        instance = row["instance_path"]
        label = row["instance_label"]
        key = (problem, label, instance)
        methods[method] += 1
        problems[problem] += 1
        per_instance[key] += 1
        repairs[(method, row["repair"])] += 1
        ratios[(method, row["selected_ratio"])] += 1
        if method not in EXPECTED_METHODS:
            errors.append(f"Task {task_id}: unexpected method {method}")
        if problem not in {"SCFLP", "SCMND"}:
            errors.append(f"Task {task_id}: unexpected problem {problem}")
        rel_config = Path(f"configs/task_{task_id:04d}.py")
        file = master / rel_config
        if not file.is_file():
            errors.append(f"Task {task_id}: missing original override {rel_config}")
            continue
        try:
            conf = parse_assignments(file)
        except ValueError as exc:
            errors.append(f"Task {task_id}: invalid config: {exc}")
            continue
        selected_instances = (conf.get("problem_classes_to_instances") or {}).get(problem)
        if selected_instances != [instance]:
            errors.append(f"Task {task_id}: override instance differs from manifest")
        if method in EXPECTED_METHODS and actual_method(conf) != method:
            errors.append(
                f"Task {task_id}: override method {actual_method(conf)!r} "
                f"differs from manifest {method!r}"
            )
        ratio_config = only_value(conf, "selected_subproblems_ratio_grid")
        if method != "STANDARD_BD":
            try:
                if ratio_config is None or abs(float(ratio_config) - float(row["selected_ratio"])) > 1e-10:
                    errors.append(f"Task {task_id}: selected ratio mismatch")
            except ValueError:
                errors.append(f"Task {task_id}: selected ratio not numeric")
        if method.startswith("PARAMETRIC_"):
            repair = only_value(conf, "parametric_repair_mode_grid")
            if repair != row["repair"]:
                errors.append(f"Task {task_id}: parametric repair mismatch {repair} != {row['repair']}")
        if problem == "SCMND":
            formulation = only_value(conf, "scmnd_formulation_grid")
            formations[str(formulation)] += 1
            if formulation not in {"CURRENT_SOFT_CAPACITY", "MANUSCRIPT_ENVELOPE"}:
                errors.append(f"Task {task_id}: unexpected SCMND formulation {formulation}")
        if Path(instance).is_absolute() or ".." in Path(instance).parts:
            errors.append(f"Task {task_id}: unsafe instance path {instance}")
        else:
            datasets[instance] = data / instance

    for (problem, label, instance), count in per_instance.items():
        if count != 37:
            errors.append(f"{problem}/{label} ({instance}): {count} config rows, expected 37")
    if len(per_instance) != 33:
        errors.append(f"Found {len(per_instance)} unique instance configurations, expected 33")
    expected_problem_counts = {"SCFLP": 555, "SCMND": 666}
    if dict(problems) != expected_problem_counts:
        errors.append(f"Problem counts differ from {expected_problem_counts}: {dict(problems)}")
    fingerprints = {}
    for rel, path in sorted(datasets.items()):
        if not path.is_file():
            errors.append(f"Missing dataset: {rel}")
        else:
            fingerprints[rel] = {
                "size_bytes": path.stat().st_size,
                "sha256": dataset_sha256(path),
            }

    soft = formations.get("CURRENT_SOFT_CAPACITY", 0)
    if soft:
        warnings.append(
            f"{soft} original SCMND overrides use legacy soft-capacity mode; "
            "the full launch must freeze hard MANUSCRIPT_ENVELOPE COPIES"
        )
    if any(method.startswith("PARAMETRIC_") and repair != "LOCAL"
           for (method, repair) in repairs):
        warnings.append(
            "Original manifest has non-LOCAL parametric repair variants; "
            "do not change repair policy or launch until this is reconciled"
        )
    return {
        "status": "FAILED" if errors else "PASSED_ORIGINAL_MANIFEST",
        "original_master": str(master),
        "source_data_root": str(data),
        "total_tasks": len(rows),
        "unique_instances": len(per_instance),
        "unique_datasets_hashed": len(fingerprints),
        "problem_counts": dict(sorted(problems.items())),
        "method_counts": dict(sorted(methods.items())),
        "repair_counts": [
            {"method": method, "repair": repair, "count": count}
            for (method, repair), count in sorted(repairs.items())
        ],
        "selected_ratio_counts": [
            {"method": method, "selected_ratio": ratio, "count": count}
            for (method, ratio), count in sorted(ratios.items())
        ],
        "scmnd_source_formulation_counts": dict(sorted(formations.items())),
        "dataset_sha256": fingerprints,
        "warnings": warnings,
        "errors": errors,
        "launch_approved": False,
        "note": "Read-only original manifest validation; does not validate a full frozen launch or submit jobs.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path,
                        help="Optional output report path (must not be inside original master or DATA)")
    args = parser.parse_args()
    master, data = args.master_root.expanduser().resolve(), args.data_root.expanduser().resolve()
    report = audit(master, data)
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.report_json:
        target = args.report_json.expanduser().resolve()
        if target.is_relative_to(master) or target.is_relative_to(data):
            parser.error("Report must be stored outside original master/data roots")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
        print("Report saved:", target)
    # Compact summary for terminal pastes; full SHA256 values in optional report.
    summary = {k: v for k, v in report.items() if k != "dataset_sha256"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Data SHA256 values checked:", len(report["dataset_sha256"]))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
