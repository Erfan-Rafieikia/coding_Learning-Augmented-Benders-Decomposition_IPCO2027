#!/usr/bin/env python3
"""Prepare a frozen 1,221-task LABD campaign from an existing frozen root, changing ONLY data paths.

Apples-to-apples by construction:
  * source/            copied byte-for-byte from the template root (frozen at 89331ce);
  * config.py          copied byte-for-byte;
  * configs/task_*.py  copied with EXACTLY ONE edit: the quoted instance path in
                       problem_classes_to_instances (all other assignments are checked
                       to be identical to the template);
  * task_list.txt      identical task order (1..1221), so the frozen verifier, array
                       wrapper and batched launcher work unchanged.
Template instance k is replaced by instance k of the chosen family; template scenario
size s is replaced by SIZE_MAP[s] (default identity).  Never submits jobs.

Families (paths are relative to --data-root):
  SCFLP  HOLMBERG  SCFLP/scenarios_s{s}_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/{name}_stochastic.txt
         CAP       SCFLP_CAP/scenarios_s{s}_scip_normal_sd0.1_0.2_cov6/{name}_stochastic.txt  (SCIP method)
  SCMND  GENERATED SUMND/R_stochastic/S{s}_sigma0.10-0.30_cov5/{name}.stoch.json
         RAHMANIANI SUMND/R_rahmaniani_corr{corr}/S{s}_published_cov5/{name}.stoch.json

Examples (Nibi):
  # corrected Holmberg + current SCMND data, 100/300/500 and 700/900/1100
  python scripts/prepare_external_benchmark_campaign.py --template-root $T --data-root $DATA \
      --results-root $RES --gap 0.001 --scflp-family HOLMBERG --scmnd-family GENERATED
  python scripts/prepare_external_benchmark_campaign.py ... --size-map 100:700 300:900 500:1100
  # external literature benchmarks
  python scripts/prepare_external_benchmark_campaign.py ... --data-root ~/Erfan_projects/LABD_EXTERNAL_DATA \
      --scflp-family CAP --scmnd-family RAHMANIANI   # second campaign: --size-map 100:700 300:900 500:1000
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

EXPECTED_SOURCE_COMMIT = "89331ce56f688f969f3f9b3317d777fedf5b5e4a"
FOLDER = "scenarios_s{s}_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities"
PATHS = {
    ("SCFLP", "HOLMBERG"): "SCFLP/" + FOLDER + "/{name}_stochastic.txt",
    ("SCFLP", "CAP"): "SCFLP_CAP/scenarios_s{s}_scip_normal_sd0.1_0.2_cov6/{name}_stochastic.txt",
    ("SCMND", "GENERATED"): "SUMND/R_stochastic/S{s}_sigma0.10-0.30_cov5/{name}.stoch.json",
    ("SCMND", "RAHMANIANI"): "SUMND/R_rahmaniani_corr{corr}/S{s}_published_cov5/{name}.stoch.json",
}
DEFAULT_NAMES = {
    ("SCFLP", "HOLMBERG"): None,  # same instances as the template (p4 p5 p10 p20 p50)
    ("SCFLP", "CAP"): ["cap41", "cap44", "cap61", "cap64", "cap71"],
    ("SCMND", "GENERATED"): None,  # same instances as the template
    ("SCMND", "RAHMANIANI"): ["r04.1", "r04.3", "r05.1", "r05.3", "r06.1", "r06.3"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assignments(text: str) -> dict:
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError):
                        out[t.id] = ("<non-literal>", ast.dump(node.value))
    return out


def scenario_count(problem: str, path: str) -> int:
    m = re.search(r"scenarios_s(\d+)_" if problem == "SCFLP" else r"(?:^|/)S(\d+)_", path)
    if not m:
        raise ValueError(f"Cannot read scenario count from {path}")
    return int(m.group(1))


def instance_name(problem: str, path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    return base.replace("_stochastic.txt", "") if problem == "SCFLP" else base.replace(".stoch.json", "")


def verify_template_files(template: Path) -> dict:
    """Check the template's frozen source/config files (not its data, which may have been regenerated)."""
    snap = json.loads((template / "full_sha256.json").read_text())
    if snap.get("schema") != "labd_full_snapshot_v1" or snap.get("tasks") != 1221:
        raise ValueError("Template is not a 1,221-task frozen full root")
    for rel, expected in snap["files"].items():
        if sha256(template / rel) != expected:
            raise ValueError(f"Template file changed since freezing: {rel}")
    return snap


def prepare(template, data, results, gap, scflp_family, scmnd_family, names, size_map, corr,
            allow_other_commit=False):
    snap = verify_template_files(template)
    commit = (template / "source_commit.txt").read_text().strip()
    if commit != EXPECTED_SOURCE_COMMIT and not allow_other_commit:
        raise ValueError(f"Template source commit {commit} != {EXPECTED_SOURCE_COMMIT}")
    if snap["source_commit"] != commit:
        raise ValueError("Template snapshot/source commit mismatch")
    with (template / "full_manifest.tsv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    columns = list(rows[0].keys())
    family = {"SCFLP": scflp_family, "SCMND": scmnd_family}

    order = {"SCFLP": [], "SCMND": []}
    for r in rows:
        n = instance_name(r["problem"], r["instance_path"])
        if n not in order[r["problem"]]:
            order[r["problem"]].append(n)
    mapping = {}
    for p in order:
        target = names.get(p) or DEFAULT_NAMES[(p, family[p])] or order[p]
        if len(target) != len(order[p]):
            raise ValueError(f"{p}: template has {len(order[p])} instances, got {len(target)}")
        mapping[p] = dict(zip(order[p], target))

    label = "0p1pct" if gap == "0.001" else "1pct"
    root = results / "MASTER_RUNS" / f"labd_convergence_full_{label}_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _populate(root, template, data, gap, rows, columns, family, mapping, size_map, corr,
                         commit, scflp_family, scmnd_family)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)  # never leave a half-built campaign root
        raise


def _populate(root, template, data, gap, rows, columns, family, mapping, size_map, corr,
              commit, scflp_family, scmnd_family):
    (root / "configs").mkdir()
    (root / "SLURM_LOGS").mkdir()
    shutil.copytree(template / "source", root / "source")
    for name in ("source_commit.txt", "config.py", "task_list.txt"):
        shutil.copy2(template / name, root / name)

    new_rows, used = [], {}
    for r in rows:
        tid, problem, old = int(r["task_id"]), r["problem"], r["instance_path"]
        s_old = scenario_count(problem, old)
        s = size_map.get(s_old, s_old)
        name = mapping[problem][instance_name(problem, old)]
        new = PATHS[(problem, family[problem])].format(s=s, name=name, corr=corr)
        if not (data / new).is_file():
            raise FileNotFoundError(f"Task {tid}: missing dataset {data / new}")
        rel = f"configs/task_{tid:04d}.py"
        text = (template / rel).read_text(encoding="utf-8")
        if new == old:
            new_text = text
        else:
            hits = [q + old + q for q in ('"', "'") if (q + old + q) in text]
            if len(hits) != 1 or text.count(hits[0]) != 1:
                raise ValueError(f"Task {tid}: expected exactly one quoted occurrence of {old}")
            new_text = text.replace(hits[0], hits[0][0] + new + hits[0][0])
        before, after = assignments(text), assignments(new_text)
        if after.get("problem_classes_to_instances", {}).get(problem) != [new]:
            raise ValueError(f"Task {tid}: instance substitution failed")
        before.pop("problem_classes_to_instances"); after.pop("problem_classes_to_instances")
        if before != after:
            raise ValueError(f"Task {tid}: a setting other than the instance path changed")
        (root / rel).write_text(new_text, encoding="utf-8")
        used[new] = sha256(data / new)
        new_rows.append({**r, "instance_label": f"{name}_S{s}", "instance_path": new, "gap": gap})

    with (root / "full_manifest.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        w.writeheader()
        w.writerows(new_rows)
    (root / "source_data_root.txt").write_text(str(data) + "\n", encoding="utf-8")
    (root / "original_master.txt").write_text((template / "original_master.txt").read_text(), encoding="utf-8")
    (root / "target_gap.txt").write_text(gap + "\n", encoding="utf-8")
    summary = {
        "campaign_kind": "DATA_VARIANT_OF_FROZEN_CAMPAIGN",
        "template_root": str(template), "template_source_commit": commit,
        "scflp_family": scflp_family, "scmnd_family": scmnd_family,
        "instance_mapping": mapping, "size_map": {str(k): v for k, v in size_map.items()},
        "rahmaniani_correlation": corr if scmnd_family == "RAHMANIANI" else None,
        "data_root": str(data), "tasks": len(new_rows),
        "only_change_per_task": "quoted instance path in problem_classes_to_instances",
    }
    for fname in ("source_audit_summary.json", "CAMPAIGN_DATA_VARIANT.json"):
        (root / fname).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    paths = [p for p in (root / "source").rglob("*") if p.is_file()]
    paths += list((root / "configs").glob("task_*.py"))
    paths += [root / x for x in ("config.py", "task_list.txt", "full_manifest.tsv", "source_commit.txt",
                                 "source_data_root.txt", "original_master.txt", "target_gap.txt",
                                 "source_audit_summary.json")]
    snapshot = {"schema": "labd_full_snapshot_v1", "source_commit": commit, "gap": gap,
                "tasks": len(new_rows),
                "files": {str(p.relative_to(root)): sha256(p) for p in sorted(paths)},
                "data": dict(sorted(used.items()))}
    (root / "full_sha256.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    # The new root must pass the same frozen verifier that every array task runs.
    sys.path.insert(0, str(root / "source" / "scripts"))
    import verify_convergence_full_snapshot as v  # noqa: E402
    v.verify(root)
    print("PREPARED (NO JOBS SUBMITTED)")
    print("FULL_ROOT=" + str(root))
    print(f"commit {commit} | gap {gap} | tasks {len(new_rows)} | SCFLP {scflp_family} | SCMND {scmnd_family}")
    print("sizes:", sorted({scenario_count(r['problem'], r['instance_path']) for r in new_rows}))
    print("mapping:", json.dumps(mapping))
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--gap", choices=("0.001", "0.01"), required=True)
    ap.add_argument("--scflp-family", choices=("HOLMBERG", "CAP"), required=True)
    ap.add_argument("--scmnd-family", choices=("GENERATED", "RAHMANIANI"), required=True)
    ap.add_argument("--scflp-names", nargs=5)
    ap.add_argument("--scmnd-names", nargs=6)
    ap.add_argument("--size-map", nargs="*", default=[], help="e.g. 100:700 300:900 500:1100")
    ap.add_argument("--correlation", default="0", choices=["0", "0.2", "0.4", "0.6", "0.8"])
    ap.add_argument("--allow-other-commit", action="store_true")
    a = ap.parse_args()
    size_map = {int(x.split(":")[0]): int(x.split(":")[1]) for x in a.size_map}
    prepare(a.template_root.expanduser().resolve(), a.data_root.expanduser().resolve(),
            a.results_root.expanduser().resolve(), a.gap, a.scflp_family, a.scmnd_family,
            {"SCFLP": a.scflp_names, "SCMND": a.scmnd_names}, size_map, a.correlation,
            a.allow_other_commit)


if __name__ == "__main__":
    sys.exit(main())
