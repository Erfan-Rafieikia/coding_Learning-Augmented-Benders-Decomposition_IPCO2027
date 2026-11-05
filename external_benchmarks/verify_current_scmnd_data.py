#!/usr/bin/env python3
"""Read-only: confirm each SCMND JSON used by the campaign matches its source .dow network.

For every *.stoch.json under DATA/SUMND/R_stochastic it re-reads the .dow file named in
its metadata, checks arc tail/head/unit cost/capacity/fixed cost and commodity
origin/destination/demand field by field, compares the JSON as the frozen solver reads
it, and also reports whether the .dow is byte-identical to the copy in
external_benchmarks/sources/r_networks_crainic (from the Bertsimas et al. repository).

  python external_benchmarks/verify_current_scmnd_data.py ~/Erfan_projects/Benders_Integration_with_ML/DATA
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from Problem.data import read_scmnd_data  # noqa: E402

REF = REPO / "external_benchmarks" / "sources" / "r_networks_crainic"


def dow(path: Path):
    lines = [l.split() for l in path.read_text().splitlines()
             if l.strip() and not l.strip().startswith(("#", "MULTIGEN"))]
    n, m, k = map(int, lines[0][:3])
    arcs = [tuple(float(x) for x in l[:5]) for l in lines[1:1 + m]]   # tail head unit_cost capacity fixed
    coms = [tuple(float(x) for x in l[:3]) for l in lines[1 + m:1 + m + k]]  # origin dest demand
    return n, arcs, coms


def main():
    data = Path(sys.argv[1]).expanduser().resolve()
    bad = 0
    files = sorted((data / "SUMND" / "R_stochastic").glob("*/*.stoch.json"))
    for f in files:
        j = json.loads(f.read_text())
        src_name = Path(j["metadata"].get("source_file_name") or f.name.replace(".stoch.json", ".dow")).name
        cands = list(data.rglob(src_name))
        if not cands:
            print(f"?? {f.relative_to(data)}: source {src_name} not found under DATA"); bad += 1; continue
        n, arcs, coms = dow(cands[0])
        det = j["deterministic"]
        ja = [(a["tail"], a["head"], a["unit_cost"], a["capacity"], a["fixed_cost"]) for a in det["arcs"]]
        jc = [(c["origin"], c["destination"], c["demand"]) for c in det["commodities"]]
        ok = det["n_nodes"] == n and ja == arcs and jc == coms
        d = read_scmnd_data(str(f), scmnd_formulation="MANUSCRIPT_ENVELOPE")
        ok &= np.array_equal(d.u_base, [a[3] for a in arcs]) and np.array_equal(d.F, [a[4] for a in arcs])
        ok &= np.array_equal(d.C[:, 0], [a[2] for a in arcs])
        ref = REF / src_name
        same = (ref.is_file() and hashlib.sha256(ref.read_bytes()).hexdigest()
                == hashlib.sha256(cands[0].read_bytes()).hexdigest())
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {f.relative_to(data)}  source={cands[0].relative_to(data)}  "
              f"arcs={len(arcs)} commodities={len(coms)}  dow_identical_to_published_copy={same}")
    print("checked", len(files), "files; problems:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
