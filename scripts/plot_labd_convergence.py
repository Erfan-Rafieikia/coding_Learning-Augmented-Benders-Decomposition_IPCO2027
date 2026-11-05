#!/usr/bin/env python3
"""Plot Gurobi master bounds or master gaps, as observed during optimization.

Example:
 python scripts/plot_labd_convergence.py --x time --output bounds.pdf traces/*.csv
 python scripts/plot_labd_convergence.py --x iteration --gap --output gap.png traces/*.csv

For inexact methods these are surrogate/master objectives, not verified original
upper bounds. Older CSVs use their recorded master_incumbent_proxy column only.
"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import math

def rows_from(path, x_axis, y_name):
    xfield = "elapsed_seconds" if x_axis == "time" else "iteration"
    points = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("event") == "verified_candidate":
                continue  # Historical v1 reconstructed events are not solver observations.
            if (x_axis == "iteration" and row.get("trace_schema_version") == "gurobi_master_v3_events"
                    and row.get("event") not in {"solver_iteration_end", "solver_final"}):
                continue  # One completed-round observation, plus terminal state.
            try:
                xval = float(row[xfield])
                lower = float(row["lower_bound"]) if row.get("lower_bound") else None
                raw_upper = row.get("master_upper_bound", row.get("master_incumbent_proxy"))
                upper = float(raw_upper) if raw_upper else None
                if y_name == "master_relative_gap":
                    if lower is None or upper is None:
                        continue
                    yval = abs(upper-lower)/abs(upper) if upper else (0.0 if lower == 0 else math.inf)
                else:
                    yval = lower if y_name == "lower_bound" else upper
                if yval is None or not math.isfinite(yval) or abs(yval) >= 1e100:
                    continue
            except (ValueError, TypeError, KeyError):
                continue
            points.append((xval, yval))
    return points

def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", type=Path, nargs="+")
    p.add_argument("--x", choices=("time", "iteration"), default="time")
    p.add_argument("--gap", action="store_true",
                   help="plot Gurobi master relative gap instead of bounds")
    p.add_argument("--output", type=Path, default=Path("labd_convergence.pdf"))
    p.add_argument("--title", default="LABD convergence")
    args = p.parse_args()
    fig, ax = plt.subplots(figsize=(8, 5))
    for trace in args.traces:
        label = trace.stem
        keys = ("master_relative_gap",) if args.gap else ("lower_bound", "master_upper_bound")
        for key in keys:
            points = rows_from(trace, args.x, key)
            if not points:
                continue
            xs, ys = zip(*points)
            name = label if args.gap else label + (" master LB" if key == "lower_bound" else " master UB")
            ax.step(xs, ys, where="post", label=name)
    ax.set_xlabel("Elapsed optimization time (seconds)" if args.x == "time" else "Callback iteration")
    ax.set_ylabel("Gurobi master relative gap" if args.gap else "Gurobi master objective value")
    ax.set_title(args.title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)

if __name__ == "__main__":
    main()
