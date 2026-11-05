"""Reader for deterministic capacitated facility location files in the
Holmberg / OR-Library layout used by DATA/SCFLP/p1..p71.

Format (Holmberg, users.mai.liu.se/kajho48/problemdata/cloc):
    m n
    for each facility i = 1..m:  S_i f_i        (capacity, fixed cost)
    for each customer j = 1..n:  D_j            (demand)
    matrix c_ij, one row per facility (m rows of n values)
c_ij is the cost of assigning ALL of customer j's demand to facility i
(single-sourcing data).  The LABD SCFLP recourse ships y_ij units, so the unit
cost is c_ij / D_j, the multiple-sourcing conversion of Lu et al. (2014) and
of SCIP's SCFLP reader for OR-Library CAP data.

Before 2026-11-01 every SCFLP_data_generation*.py read the first m numbers
after the header as capacities and the next m as fixed costs, which mixes the
(capacity, fixed cost) pairs, and used c_ij directly as a unit cost.
"""
from __future__ import annotations

import numpy as np


def read_holmberg_instance(file_path: str, costs_are_per_customer_demand: bool = True):
    """Return (capacities, fixed_costs, demands, unit_costs[customers x facilities])."""
    with open(file_path, "r") as f:
        words = f.read().split()
    m, n = int(words[0]), int(words[1])
    expected = 2 + 2 * m + n + m * n
    if len(words) < expected:
        raise ValueError(f"{file_path}: {len(words)} numbers, expected {expected} for m={m}, n={n}")
    if len(words) > expected:
        # e.g. Holmberg p26 as distributed: the instance, a 1995 e-mail header, then an
        # identical second copy.  Use the first complete instance and say so.
        import warnings
        warnings.warn(f"{file_path}: {len(words) - expected} trailing tokens after the instance "
                      f"(starting {' '.join(words[expected:expected + 3])!r}); using the first {expected}")
        words = words[:expected]
    pairs = np.array(words[2:2 + 2 * m], dtype=float).reshape(m, 2)
    capacities, fixed_costs = pairs[:, 0].copy(), pairs[:, 1].copy()
    pos = 2 + 2 * m
    demands = np.array(words[pos:pos + n], dtype=float)
    pos += n
    costs = np.array(words[pos:pos + m * n], dtype=float).reshape(m, n).T  # customers x facilities
    if costs_are_per_customer_demand:
        if np.any(demands <= 0):
            raise ValueError(f"{file_path}: non-positive demand; cannot form unit costs")
        costs = costs / demands[:, None]
    return capacities, fixed_costs, demands, costs
