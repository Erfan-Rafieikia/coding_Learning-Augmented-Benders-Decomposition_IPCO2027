#!/usr/bin/env python3
"""Read-only checks of built external-benchmark files with the UNCHANGED readers.

For every file:
  * it loads with src/Problem/data.py (read_scflp_data / read_scmnd_data with
    MANUSCRIPT_ENVELOPE), i.e. exactly as the frozen solver will read it;
  * relatively complete recourse (RCR) for the manuscript formulations:
      SCFLP  sum_j u_j >= max_m sum_i d_im   (constraint model:scflp-saa:recourse
             is then satisfiable with all facilities open)
      SCMND  the commodity-wise envelope max_m d_qm is routable with every arc
             open (constraints model:scmnd-saa:envelope-flow/-capacity);
             computed as the max concurrent-flow factor lambda* >= 1.
Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from Problem.data import read_scflp_data, read_scmnd_data, scmnd_envelope_demands  # noqa: E402


def envelope_lambda(data) -> float:
    arcs = list(data.A)
    n, K = len(arcs), len(data.origins)
    nodes = sorted({i for a in arcs for i in a})
    vol = scmnd_envelope_demands(data)
    nv = n * K + 1
    r, c, v, row = [], [], [], 0
    for q in range(K):
        o, d = int(data.origins[q]), int(data.destinations[q])
        for i in nodes:
            for a, (s, t) in enumerate(arcs):
                if s == i:
                    r.append(row); c.append(q * n + a); v.append(1.0)
                if t == i:
                    r.append(row); c.append(q * n + a); v.append(-1.0)
            rhs = vol[q] if i == o else (-vol[q] if i == d else 0.0)
            r.append(row); c.append(nv - 1); v.append(-rhs)
            row += 1
    a_eq = sp.csr_matrix((v, (r, c)), shape=(row, nv))
    a_ub = sp.csr_matrix(([1.0] * (n * K), ([a for a in range(n) for _ in range(K)],
                                            [q * n + a for a in range(n) for q in range(K)])),
                         shape=(n, nv))
    obj = np.zeros(nv); obj[-1] = -1.0
    res = linprog(obj, A_ub=a_ub, b_ub=np.asarray(data.u_base, float), A_eq=a_eq,
                  b_eq=np.zeros(row), bounds=[(0, None)] * nv, method="highs")
    return -res.fun if res.status == 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_root", type=Path)
    args = ap.parse_args()
    root = args.data_root.resolve()
    failures = 0
    for f in sorted(root.glob("SCFLP_*/*/*_stochastic.txt")):
        d = read_scflp_data(str(f))
        ratio = float(np.sum(d.capacities)) / d.max_demand_sum_over_scenario
        ok = ratio >= 1.0 and d.scenarios.shape == (len(d.S), len(d.I)) and d.covariate_vectors is not None
        failures += not ok
        print(f"{'OK ' if ok else 'FAIL'} SCFLP {f.relative_to(root)}  |J|={len(d.J)} |I|={len(d.I)} "
              f"S={len(d.S)} cov={d.num_covariates} total_cap/max_total_demand={ratio:.3f}")
    for f in sorted(root.glob("SUMND/*/*/*.stoch.json")):
        d = read_scmnd_data(str(f), scmnd_formulation="MANUSCRIPT_ENVELOPE")
        lam = envelope_lambda(d)
        ok = lam >= 1.0 and np.allclose(d.probabilities.sum(), 1.0) and d.covariate_vectors is not None
        failures += not ok
        print(f"{'OK ' if ok else 'FAIL'} SCMND {f.relative_to(root)}  |A|={d.n_arcs} |K|={d.n_commodities} "
              f"S={d.n_scenarios} cov={d.num_covariates} envelope_lambda*={lam:.2f}")
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
