#!/usr/bin/env python3
"""Integrity gate for a separately frozen 1,221-task LABD full sweep."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root, task_id=None):
    source = json.loads((root / "full_sha256.json").read_text())
    if source["schema"] != "labd_full_snapshot_v1" or source["tasks"] != 1221:
        raise ValueError("Wrong full campaign snapshot schema/task count")
    if source["source_commit"] != (root / "source_commit.txt").read_text().strip():
        raise ValueError("Frozen source commit mismatch")
    if source["gap"] != (root / "target_gap.txt").read_text().strip():
        raise ValueError("Frozen target-gap mismatch")
    if source["gap"] not in {"0.001", "0.01"}:
        raise ValueError("Unexpected gap")
    data = Path((root / "source_data_root.txt").read_text().strip())
    if not data.is_dir():
        raise ValueError("Frozen DATA root missing")
    with (root / "full_manifest.tsv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 1221 or sorted(int(r["task_id"]) for r in rows) != list(range(1, 1222)):
        raise ValueError("Full manifest must contain IDs 1..1221 exactly")
    if any(r["gap"] != source["gap"] for r in rows):
        raise ValueError("Manifest gap mismatch")
    task_lines = (root / "task_list.txt").read_text().splitlines()
    if len(task_lines) != 1221:
        raise ValueError("Wrong number of executable task-list lines")
    for row in rows:
        i = int(row["task_id"])
        expected = f"configs/task_{i:04d}.py|{row['problem']}"
        if task_lines[i-1] != expected:
            raise ValueError(f"Task {i} executable-list mismatch")
        if row["problem"] == "SCMND" and row["scmnd_formulation"] != "MANUSCRIPT_ENVELOPE":
            raise ValueError(f"SCMND task {i} does not have hard model metadata")
    if task_id is None:
        checks = source["files"]
        dchecks = source["data"]
    else:
        if not 1 <= task_id <= 1221:
            raise ValueError("Task ID must be between 1 and 1221")
        row = rows[task_id-1]
        files = {
            "source_commit.txt", "source_data_root.txt", "target_gap.txt",
            "full_manifest.tsv", "task_list.txt", "config.py",
            f"configs/task_{task_id:04d}.py",
            "source/my_job.pmedian_nearest.alliance.sbatch",
            "source/scripts/run_convergence_full_array.sbatch",
            "source/scripts/verify_convergence_full_snapshot.py",
            "source/src/main.py",
            "source/src/Problem/master_problem.py",
            "source/src/utils/parametric_iteration_sidecar.py",
            "source/src/Dual_Prediction/prediction.py",
        }
        checks = {k: source["files"][k] for k in sorted(files)}
        dchecks = {row["instance_path"]: source["data"][row["instance_path"]]}
    for rel, expected in checks.items():
        path = root / rel
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen full-campaign file changed: {rel}")
    for rel, expected in dchecks.items():
        path = data / rel
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen dataset changed: {rel}")
    print(
        "Verified full snapshot", source["source_commit"], "gap", source["gap"],
        "task", task_id if task_id is not None else "ALL",
        "files", len(checks), "datasets", len(dchecks),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("full_root", type=Path)
    p.add_argument("--task-id", type=int)
    args = p.parse_args()
    verify(args.full_root.expanduser().resolve(), args.task_id)


if __name__ == "__main__":
    sys.exit(main())
