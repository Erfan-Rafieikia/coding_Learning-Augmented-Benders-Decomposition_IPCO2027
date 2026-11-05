#!/usr/bin/env python3
"""Prepare an IPCO 2027 run root from the journal's campaign-1 root. Never submits.

For each chosen SCFLP instance at 500 scenarios and each exact share, this copies the
campaign-1 task config of PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR with LOCAL repair
and changes exactly ONE thing: the predictor name in predictor_ML_Augmented_BD_grid
becomes the IPCO predictor. Everything else (data path, gap, time limit, guard,
repair mode, ridge settings, warm-up, seeds) is checked to be identical. config.py,
the data root and the gap target are copied from the template root unchanged.

The code that runs is a snapshot (git archive) of this repository's HEAD, stored in
ROOT/source, with its commit in ROOT/source_commit.txt.

Example (Nibi):
  python scripts/ipco/prepare_ipco_root.py \
      --template-root /scratch/rafieiki/Benders_Integration_with_ML/RESULTS/MASTER_RUNS/labd_convergence_full_0p1pct_20261102_141136 \
      --results-root /scratch/rafieiki/LABD_IPCO2027/RESULTS --label smoke \
      --instances p4 p10 --ratios 0.05 0.15 0.25
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys

TEMPLATE_METHOD = "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"
IPCO_METHODS = ("PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN", "PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN")
PREDICTOR_KEY = "predictor_ML_Augmented_BD_grid"


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


def ratio_matches(value: str, wanted: float) -> bool:
    try:
        return abs(float(value) - wanted) < 1e-9
    except ValueError:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template-root", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--label", default="smoke", help="smoke or full (part of the root name)")
    p.add_argument("--method", default=IPCO_METHODS[0], choices=IPCO_METHODS)
    p.add_argument("--instances", nargs="+", default=["p4"])
    p.add_argument("--scenarios", type=int, default=500)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.05, 0.15, 0.25])
    p.add_argument("--repair", default="LOCAL")
    args = p.parse_args()

    template = args.template_root.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    with (template / "full_manifest.tsv").open(encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f, delimiter="\t"))

    chosen = []
    for inst in args.instances:
        label = f"{inst}_S{args.scenarios}"
        for ratio in args.ratios:
            rows = [r for r in manifest
                    if r.get("problem") == "SCFLP" and r.get("instance_label") == label
                    and r.get("method") == TEMPLATE_METHOD and r.get("repair", "") == args.repair
                    and ratio_matches(r.get("selected_ratio", ""), ratio)]
            if len(rows) != 1:
                raise SystemExit(f"Expected one template task for {label} ratio {ratio}, found {len(rows)}")
            chosen.append(rows[0])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "pmedian" if args.method.endswith("PMEDIAN_MEAN") else "random"
    root = args.results_root.expanduser().resolve() / "MASTER_RUNS" / f"ipco_{args.label}_{tag}_{stamp}"
    (root / "configs").mkdir(parents=True, exist_ok=False)
    (root / "SLURM_LOGS").mkdir()

    # Code snapshot of this repository's HEAD (must be clean).
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit("The IPCO code repository has uncommitted changes; commit or stash them first.")
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    (root / "source").mkdir()
    archive = subprocess.run(["git", "-C", str(repo), "archive", "HEAD"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(root / "source")], input=archive, check=True)
    (root / "source_commit.txt").write_text(commit + "\n")

    shutil.copy2(template / "config.py", root / "config.py")
    for name in ("source_data_root.txt", "target_gap.txt"):
        shutil.copy2(template / name, root / name)

    out_rows = []
    with (root / "task_list.txt").open("w", encoding="utf-8") as task_list:
        for new_id, row in enumerate(chosen, start=1):
            src = template / f"configs/task_{int(row['task_id']):04d}.py"
            text = src.read_text(encoding="utf-8")
            old = assignments(text)
            if old.get(PREDICTOR_KEY) != [TEMPLATE_METHOD]:
                raise SystemExit(f"{src}: unexpected {PREDICTOR_KEY} = {old.get(PREDICTOR_KEY)}")
            if text.count(f"'{TEMPLATE_METHOD}'") + text.count(f'"{TEMPLATE_METHOD}"') != 1:
                raise SystemExit(f"{src}: the template predictor name must appear exactly once")
            new_text = text.replace(f"'{TEMPLATE_METHOD}'", f"'{args.method}'").replace(
                f'"{TEMPLATE_METHOD}"', f'"{args.method}"')
            new = assignments(new_text)
            changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            if changed != [PREDICTOR_KEY]:
                raise SystemExit(f"{src}: unexpected changes {changed}")
            shown = {k: v for k, v in new.items() if re.search(r"guard|repair|ratio|gap|time_limit", k, re.I)}
            print(f"  {src.name} -> {args.method}: {shown}")
            rel = f"configs/task_{new_id:04d}.py"
            (root / rel).write_text(new_text, encoding="utf-8")
            task_list.write(f"{rel}|SCFLP\n")
            out_rows.append({"task_id": new_id, "template_task_id": row["task_id"],
                             "instance_label": row["instance_label"], "method": args.method,
                             "selected_ratio": row["selected_ratio"], "repair": row.get("repair", ""),
                             "template_config": str(src)})

    with (root / "manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    (root / "template_root.txt").write_text(str(template) + "\n")

    print("PREPARED (NO JOBS SUBMITTED)")
    print("IPCO_ROOT=" + str(root))
    print("Code commit:", commit)
    print("Gap:", (root / "target_gap.txt").read_text().strip(), "| Data:", (root / "source_data_root.txt").read_text().strip())
    for r in out_rows:
        print(f"  task {r['task_id']}: {r['instance_label']} ratio {r['selected_ratio']} (from campaign-1 task {r['template_task_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
