#!/usr/bin/env python3
"""Prepare a frozen, isolated full LABD campaign; never submit from this script.

Preserves each original method, selected ratio and parametric repair mode exactly.
The ONLY model change in copied overrides is to make all SCMND runs use the
manuscript MANUSCRIPT_ENVELOPE formulation (as in the validated hard pilot).
Use a separate invocation/root for each gap target.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

from audit_full_campaign_manifest import audit
from prepare_convergence_pilots import hard_capacity_override, parse_assignments, only_value
from verify_convergence_snapshot import snapshot_source


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(master, data, results, gap, repository):
    report = audit(master, data)
    if report["errors"]:
        raise ValueError(
            "Source full manifest audit failed: " +
            "; ".join(report["errors"][:12])
        )
    if report["total_tasks"] != 1221 or report["unique_instances"] != 33:
        raise ValueError("Full grid dimensions are not 1221 tasks / 33 instances")
    # Preserve the original 1221 variants and their configured LOCAL or
    # STRUCTURE_AWARE repair. Never silently convert one into another.
    label = "0p1pct" if gap == "0.001" else "1pct"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = results / "MASTER_RUNS" / f"labd_convergence_full_{label}_{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    (root / "SLURM_LOGS").mkdir()
    commit = snapshot_source(repository, root)
    shutil.copy2(master / "config.py", root / "config.py")
    with (master / "manifest.tsv").open(encoding="utf-8", newline="") as f:
        original = list(csv.DictReader(f, delimiter="\t"))
    if len(original) != 1221:
        raise ValueError("Unexpected source grid size")
    entries = []
    with (root / "task_list.txt").open("w", encoding="utf-8") as task_list:
        for row in original:
            task_id = int(row["task_id"])
            problem = row["problem"]
            rel = f"configs/task_{task_id:04d}.py"
            src = master / rel
            original_text = src.read_text(encoding="utf-8")
            updated = hard_capacity_override(original_text, problem)
            (root / rel).write_text(updated, encoding="utf-8")
            parsed = parse_assignments(root / rel)
            if problem == "SCMND" and only_value(
                parsed, "scmnd_formulation_grid"
            ) != "MANUSCRIPT_ENVELOPE":
                raise ValueError(f"Task {task_id}: hard SCMND override failed")
            if problem == "SCFLP" and updated != original_text:
                raise ValueError(f"Task {task_id}: SCFLP unexpectedly changed")
            if row["method"].startswith("PARAMETRIC_") and only_value(
                parsed, "parametric_repair_mode_grid"
            ) != row["repair"]:
                raise ValueError(f"Task {task_id}: original parametric repair changed")
            task_list.write(f"{rel}|{problem}\n")
            entries.append({
                **row, "gap": gap,
                "scmnd_formulation": "MANUSCRIPT_ENVELOPE" if problem == "SCMND" else "",
            })
    columns = list(original[0]) + ["gap", "scmnd_formulation"]
    with (root / "full_manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(entries)
    (root / "source_data_root.txt").write_text(str(data) + "\n", encoding="utf-8")
    (root / "original_master.txt").write_text(str(master) + "\n", encoding="utf-8")
    (root / "target_gap.txt").write_text(gap + "\n", encoding="utf-8")
    (root / "source_audit_summary.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "dataset_sha256"},
                   indent=2) + "\n", encoding="utf-8"
    )
    # One full integrity capture, checked before submission. Per-task check
    # rehashes just the current override/dataset and critical launch files.
    paths = [p for p in (root / "source").rglob("*") if p.is_file()]
    paths.extend((root / "configs").glob("task_*.py"))
    paths.extend(
        root / x for x in ("config.py", "task_list.txt", "full_manifest.tsv",
                          "source_commit.txt", "source_data_root.txt",
                          "original_master.txt", "target_gap.txt",
                          "source_audit_summary.json")
    )
    manifest = {
        "schema": "labd_full_snapshot_v1",
        "source_commit": commit,
        "gap": gap, "tasks": 1221,
        "files": {
            str(p.relative_to(root)): digest(p)
            for p in sorted(paths)
        },
        "data": {
            rel: digest(data / rel)
            for rel in sorted(report["dataset_sha256"])
        },
    }
    (root / "full_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PREPARED (NO JOBS SUBMITTED)")
    print("Full campaign root:", root)
    print("FULL_ROOT=" + str(root))
    print("Frozen source commit:", commit)
    print("MIPGap target:", gap)
    print("Tasks:", len(entries))
    print("SCMND hard formulation:", sum(x["problem"] == "SCMND" for x in entries))
    print("Parametric repair modes (original, unchanged):", report["repair_counts"])
    print("Dataset hashes:", len(manifest["data"]))
    print("NEXT: bash scripts/submit_convergence_full.sh", root, "--dry-run")
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--gap", choices=("0.001", "0.01"), default="0.001")
    args = p.parse_args()
    prepare(args.master_root.expanduser().resolve(),
            args.data_root.expanduser().resolve(),
            args.results_root.expanduser().resolve(),
            args.gap, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    sys.exit(main())
