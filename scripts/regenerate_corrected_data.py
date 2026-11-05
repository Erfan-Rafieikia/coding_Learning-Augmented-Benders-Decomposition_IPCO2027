#!/usr/bin/env python3
"""Regenerate the stochastic SCFLP data with the corrected Holmberg reader, add missing SCMND
sizes, then install.  Resumable and parallel.  Run on Nibi with the bd_ml environment.

Steps (each can be run separately; nothing in DATA/ changes until --install):
  --check              validate DATA/SCFLP/p1..p71 and DATA/NE-SP/R/*.dow; list what would change
  --generate-scflp S   generate p1..p71 for size(s) S into the persistent stage folder
                       DATA/_regen_stage_20261101/<folder>; files already present and complete are
                       kept, so a killed job can be resubmitted.  --jobs N runs N files in parallel
                       (each file uses its own seed-42 generator, so results do not depend on N).
  --generate-scmnd S   generate missing SCMND size(s) with the unchanged batch generator (writes a new
                       DATA/SUMND/R_stochastic/S{S}_sigma0.10-0.30_cov5 folder; existing ones untouched)
  --install            only if every requested size has all 71 complete files: move EVERY existing
                       DATA/SCFLP/scenarios_* folder to DATA/SCFLP/_INVALID_swapped_capacity_fixedcost_pre20261101/
                       and move the staged folders into DATA/SCFLP; write DATA/REGENERATION_20261101.json.
Generator settings are those of the experiments: merve_with_covariate_with_random_demand_capacity,
sigma 0.1-0.3, 6 covariates, seed 42.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "utils"))
sys.path.insert(0, str(REPO / "src"))

FOLDER = "scenarios_s{s}_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities"
INVALID = "_INVALID_swapped_capacity_fixedcost_pre20261101"
STAGE = "_regen_stage_20261101"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def p_names(scflp_dir: Path):
    return sorted((p.name for p in scflp_dir.iterdir()
                   if p.is_file() and p.name[0] == "p" and p.name[1:].isdigit()), key=lambda x: int(x[1:]))


def complete(path: Path, s: int) -> bool:
    """A staged file is complete if the solver reader loads it with s scenarios and covariates."""
    if not path.is_file():
        return False
    try:
        from Problem.data import read_scflp_data
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = read_scflp_data(str(path))
        return d.scenarios.shape[0] == s and d.covariate_vectors is not None \
            and d.covariate_vectors.shape[0] == s and "# Diagnostics" in path.read_text()
    except Exception:
        return False


def _one(stage: str, name: str, s: int) -> str:
    """Generate ONE p-file at size s inside the stage folder with the unchanged generator."""
    import SCFLP_data_generation_version4 as gen
    work = Path(stage) / f"_work_{name}_{s}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shutil.copy2(Path(stage) / "_inputs" / name, work / name)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen.process_files(input_dir=str(work), file_list=[name], num_scenarios=s,
                          method="merve_with_covariate_with_random_demand_capacity",
                          output_format="txt", seed=42, sigma_lower=0.1, sigma_upper=0.3,
                          num_covariates=6)
    src = work / FOLDER.format(s=s) / f"{name}_stochastic.txt"
    dest = Path(stage) / FOLDER.format(s=s) / f"{name}_stochastic.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dest)  # atomic: a killed job never leaves a half-written final file
    shutil.rmtree(work, ignore_errors=True)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--generate-scflp", nargs="*", type=int, default=[])
    ap.add_argument("--generate-scmnd", nargs="*", type=int, default=[])
    ap.add_argument("--install", nargs="*", type=int, default=None,
                    help="sizes that must be complete before installing (default 100..1100)")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    data = a.data_root.expanduser().resolve()
    scflp_dir, r_dir, sumnd_dir = data / "SCFLP", data / "NE-SP" / "R", data / "SUMND" / "R_stochastic"
    stage = data / STAGE
    names = p_names(scflp_dir)

    if a.check:
        from holmberg_format import read_holmberg_instance
        import SUMND_data_generation as g
        for n in names:
            read_holmberg_instance(str(scflp_dir / n))
        for f in sorted(r_dir.glob("*.dow")):
            d = g.parse_deterministic_instance(f)
            assert d.n_arcs == len(d.arcs) and d.n_commodities == len(d.commodities)
        print(f"OK: {len(names)} p-files and {len(list(r_dir.glob('*.dow')))} .dow files parse")
        for s in (100, 300, 500, 700, 900, 1100):
            done = sum(complete(stage / FOLDER.format(s=s) / f"{n}_stochastic.txt", s) for n in names)
            print(f"  staged S={s}: {done}/{len(names)} complete")
        print("  existing SCFLP stochastic folders:", [p.name for p in sorted(scflp_dir.glob('scenarios_*'))])

    if a.generate_scflp:
        (stage / "_inputs").mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy2(scflp_dir / n, stage / "_inputs" / n)
        for s in a.generate_scflp:
            todo = [n for n in names if not complete(stage / FOLDER.format(s=s) / f"{n}_stochastic.txt", s)]
            print(f"S={s}: {len(names) - len(todo)} already complete, generating {len(todo)} with {a.jobs} workers",
                  flush=True)
            with ProcessPoolExecutor(max_workers=a.jobs) as pool:
                futs = {pool.submit(_one, str(stage), n, s): n for n in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    fut.result()
                    print(f"  S={s} {i}/{len(todo)} {futs[fut]}", flush=True)
            bad = [n for n in names if not complete(stage / FOLDER.format(s=s) / f"{n}_stochastic.txt", s)]
            if bad:
                raise RuntimeError(f"S={s}: incomplete after generation: {bad}")
            print(f"S={s}: all {len(names)} complete", flush=True)

    for s in a.generate_scmnd:
        import batch_generate_SUMND as bg
        if (sumnd_dir / f"S{s}_sigma0.10-0.30_cov5").is_dir():
            print(f"SCMND S={s} already exists; not regenerated")
            continue
        print(f"generating SCMND S={s} ...", flush=True)
        if not bg.run_generation(input_dir=str(r_dir), base_output_dir=str(sumnd_dir), n_scenarios=s):
            raise RuntimeError(f"SCMND generation failed for S={s}")
        print(f"SCMND S={s}: {len(list((sumnd_dir / f'S{s}_sigma0.10-0.30_cov5').glob('*.stoch.json')))} files")

    if a.install is not None:
        sizes = a.install or [100, 300, 500, 700, 900, 1100]
        missing = {s: [n for n in names if not complete(stage / FOLDER.format(s=s) / f"{n}_stochastic.txt", s)]
                   for s in sizes}
        missing = {s: v for s, v in missing.items() if v}
        if missing:
            raise SystemExit(f"NOT INSTALLED: incomplete sizes {({s: len(v) for s, v in missing.items()})}")
        invalid = scflp_dir / INVALID
        invalid.mkdir(exist_ok=True)
        for p in sorted(scflp_dir.glob("scenarios_*")):
            shutil.move(str(p), str(invalid / p.name))
        record = {"date": datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
                  "repo_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                         text=True).strip(),
                  "scflp_sizes": sizes, "files": {}}
        for s in sizes:
            dest = scflp_dir / FOLDER.format(s=s)
            shutil.move(str(stage / FOLDER.format(s=s)), str(dest))
            for f in sorted(dest.glob("*_stochastic.txt")):
                record["files"][str(f.relative_to(data))] = sha256(f)
        for f in sorted(sumnd_dir.glob("S1100_sigma0.10-0.30_cov5/*.stoch.json")):
            record["files"][str(f.relative_to(data))] = sha256(f)
        (data / "REGENERATION_20261101.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"INSTALLED {len(sizes)} SCFLP folders; old folders in SCFLP/{INVALID}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
