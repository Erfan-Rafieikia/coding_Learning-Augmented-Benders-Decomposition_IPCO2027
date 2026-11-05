#!/usr/bin/env python3
"""Read-only: every SCFLP stochastic file must carry exactly the deterministic Holmberg data.

For each DATA/SCFLP/scenarios_*/p*_stochastic.txt, as read by the solver reader
(src/Problem/data.py), compares capacities, fixed costs, mean demands and unit costs with
src/utils/holmberg_format.read_holmberg_instance(DATA/SCFLP/pX), and checks relatively
complete recourse (sum of capacities >= largest scenario total demand).

  python scripts/verify_corrected_scflp_data.py ~/Erfan_projects/Benders_Integration_with_ML/DATA
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "src" / "utils"))
from Problem.data import read_scflp_data  # noqa: E402
from holmberg_format import read_holmberg_instance  # noqa: E402


def main():
    data = Path(sys.argv[1]).expanduser().resolve() / "SCFLP"
    det = {}
    bad, per_folder = [], Counter()
    for f in sorted(data.glob("scenarios_*/p*_stochastic.txt")):
        name = f.name.replace("_stochastic.txt", "")
        if name not in det:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                det[name] = read_holmberg_instance(str(data / name))
        cap, fx, dem, cost = det[name]
        d = read_scflp_data(str(f))
        ok = (np.array_equal(d.capacities, cap) and np.array_equal(d.fixed_costs, fx)
              and np.allclose(d.demands, dem) and np.allclose(d.shipment_costs, cost, rtol=1e-12)
              and d.capacities.sum() >= d.max_demand_sum_over_scenario)
        per_folder[f.parent.name] += 1
        if not ok:
            bad.append(str(f.relative_to(data)))
    for k, v in sorted(per_folder.items()):
        print(f"{v:4d} files  {k}")
    print(f"checked {sum(per_folder.values())} files; problems: {len(bad)}")
    for b in bad[:20]:
        print("BAD", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
