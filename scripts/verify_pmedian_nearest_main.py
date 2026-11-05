"""Basic output-completeness screen for the 198 main-paper P-median runs.

Includes the four validated pilots and 194 other main-paper tasks. The appendix
CURRENT_SOFT_CAPACITY cases (100-153, 253-306) are deliberately excluded.

Usage:
    python scripts/verify_pmedian_nearest_main.py "$MASTER"

Produces main_verified.txt only if every main-paper configuration has one
correctly labeled, usable Excel outcome. This is an output screen, not a
scientific guarantee of objective accuracy or solver optimality.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


METHODS = (
    "PMEDIAN_NEAREST_TRANSFER",
    "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR",
)
MAIN_GROUPS = (
    (range(1, 46), "SCFLP", METHODS[0], None),
    (range(46, 91), "SCFLP", METHODS[1], None),
    (range(91, 145), "SCMND", METHODS[0], "MANUSCRIPT_ENVELOPE"),
    (range(199, 253), "SCMND", METHODS[1], "MANUSCRIPT_ENVELOPE"),
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("master_root", nargs="?", default=None)
    args = p.parse_args()
    results_root = (
        Path.home() / "scratch" / "Benders_Integration_with_ML" / "RESULTS"
    )
    pointer = results_root / "MASTER_RUNS" / "latest_pmedian_nearest_master.txt"
    master = (
        Path(args.master_root) if args.master_root else Path(pointer.read_text().strip())
    ).expanduser().resolve()
    if not master.name.startswith("pmedian_nearest_"):
        raise SystemExit(f"Unexpected master path: {master}")
    ts = master.name.removeprefix("pmedian_nearest_")
    run_id = f"pmedian_nearest_transfer_{ts}"

    checked = 0
    errors = []
    drifts = []
    for indices, problem, method, formulation in MAIN_GROUPS:
        class_root = results_root / f"{problem}_class" / "RESULTS" / run_id
        for task in indices:
            logs = list(class_root.glob(f"task_{task}_*/solution_log.xlsx"))
            if len(logs) != 1:
                errors.append(f"task {task}: expected one solution_log.xlsx, found {len(logs)}")
                continue
            try:
                df = pd.read_excel(logs[0])
                if len(df) != 1:
                    raise ValueError(f"expected one row, found {len(df)}")
                row = df.iloc[0]
                actual = str(row.get("predictor_type", ""))
                if actual != method:
                    raise ValueError(f"predictor_type {actual!r} instead of {method!r}")
                if formulation is not None:
                    actual_form = str(row.get("scmnd_formulation", ""))
                    if actual_form != formulation:
                        raise ValueError(f"formulation {actual_form!r} != {formulation!r}")
                stopping = str(row.get("stopping_reason", ""))
                if stopping not in ("OPTIMAL", "TIME_LIMIT"):
                    raise ValueError(f"unusable stopping reason {stopping!r}")
                for key in ("error_type", "error_message"):
                    value = row.get(key)
                    if pd.notna(value) and str(value).strip() not in ("", "nan", "None"):
                        raise ValueError(f"{key}: {value}")
                z = float(row.get("original_Prob_Obj_Value"))
                if not math.isfinite(z):
                    raise ValueError("missing/nonfinite exact reevaluated objective")
                master_z = pd.to_numeric(
                    pd.Series([row.get("objective_value")]), errors="coerce"
                ).iloc[0]
                if math.isfinite(float(master_z)) and abs(z) > 1e-12:
                    drift = abs(float(master_z) - z) / abs(z)
                    if drift > 0.05:
                        drifts.append((task, drift, str(logs[0])))
                checked += 1
            except Exception as exc:
                errors.append(f"task {task}: {exc}")

    print(f"Main-paper usable Excel results: {checked}/198")
    print(f"Main-paper objective drifts over 5%: {len(drifts)}")
    for task, drift, path in drifts[:20]:
        print(f"REVIEW task={task}: relative master-vs-exact drift={drift:.1%} {path}")

    if errors:
        for error in errors[:40]:
            print("FAIL", error)
        raise SystemExit(
            f"Main-paper output screen failed with {len(errors)} invalid/missing tasks; "
            "appendix submission remains blocked."
        )

    marker = master / "main_verified.txt"
    marker.write_text(
        "All 198 main-paper runs passed basic result presence, method, formulation,\n"
        "stopping-status, and finite re-evaluated-objective screening.\n"
        f"Master-to-true-objective drifts over 5 percent: {len(drifts)} "
        "(listed above; investigate before scientific reporting).\n",
        encoding="utf-8",
    )
    print(f"Basic screening passed; wrote {marker}")
    print("Appendix may be submitted as a separate stage after reviewing main results.")


if __name__ == "__main__":
    main()
