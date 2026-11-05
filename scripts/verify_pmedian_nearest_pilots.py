"""Validate four main-paper nearest-P-median pilot Excel outcomes on Nibi.

Run:
  ~/micromamba/envs/bd_ml/bin/python scripts/verify_pmedian_nearest_pilots.py
Or pass the prepared master directory explicitly. A verification marker is written
only after every required pilot has a valid, correctly identified Excel result.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import pandas as pd


PILOTS = {
    1: ("SCFLP", "PMEDIAN_NEAREST_TRANSFER", None),
    46: ("SCFLP", "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR", None),
    91: ("SCMND", "PMEDIAN_NEAREST_TRANSFER", "MANUSCRIPT_ENVELOPE"),
    199: ("SCMND", "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR", "MANUSCRIPT_ENVELOPE"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "master_root", nargs="?", default=None,
        help="Prepared pmedian_nearest_YYYYMMDD_HHMMSS directory",
    )
    args = parser.parse_args()
    results_root = Path.home() / "scratch/Benders_Integration_with_ML/RESULTS"
    pointer = results_root / "MASTER_RUNS/latest_pmedian_nearest_master.txt"
    if args.master_root:
        master = Path(args.master_root).expanduser().resolve()
    else:
        master = Path(pointer.read_text().strip()).expanduser().resolve()

    if not master.name.startswith("pmedian_nearest_"):
        raise SystemExit(f"Refusing unexpected experiment master directory: {master}")
    timestamp = master.name.removeprefix("pmedian_nearest_")
    run_id = f"pmedian_nearest_transfer_{timestamp}"
    errors = []
    for idx, (problem, expected_method, formulation) in PILOTS.items():
        task_root = results_root / f"{problem}_class" / "RESULTS" / run_id
        files = sorted(task_root.glob(f"task_{idx}_*/solution_log.xlsx"))
        if len(files) != 1:
            errors.append(
                f"task {idx}: expected exactly one Excel file, found {len(files)} in {task_root}"
            )
            continue
        try:
            df = pd.read_excel(files[0])
            if len(df) != 1:
                raise ValueError(f"expected one experiment row, got {len(df)}")
            row = df.iloc[0]
            name = str(row.get("predictor_type", ""))
            if name != expected_method:
                raise ValueError(f"method label {name!r}; expected {expected_method!r}")
            if formulation is not None:
                got = str(row.get("scmnd_formulation", ""))
                if got != formulation:
                    raise ValueError(f"SCMND formulation {got!r}; expected {formulation!r}")
            reason = str(row.get("stopping_reason", ""))
            if reason not in {"OPTIMAL", "TIME_LIMIT"}:
                raise ValueError(f"unusable stopping_reason {reason!r}")
            for name in ("error_type", "error_message"):
                val = row.get(name, None)
                if pd.notna(val) and str(val).strip() not in {"", "None", "nan"}:
                    raise ValueError(f"{name}: {val}")
            obj = pd.to_numeric(pd.Series([row.get("original_Prob_Obj_Value")]), errors="coerce").iloc[0]
            if not math.isfinite(float(obj)):
                raise ValueError("missing or non-finite exact re-evaluated objective")
            print(
                f"PASS task={idx}, problem={problem}, method={expected_method}, "
                f"formulation={formulation or '-'}, status={reason}, exact_obj={obj:.6g}"
            )
        except Exception as exc:
            errors.append(f"task {idx}, file={files[0]}: {exc}")

    if errors:
        for error in errors:
            print("FAIL", error)
        raise SystemExit(
            "Pilot Excel validation failed; remaining-array submission is blocked."
        )

    marker = master / "pilot_verified.txt"
    marker.write_text(
        "All four main-paper pilot Excel files passed method/formulation/status/objective validation.\n",
        encoding="utf-8",
    )
    print(f"Four main-paper pilots verified. Wrote {marker}")
    print("Safe next command: bash submit_pmedian_nearest.sh main")


if __name__ == "__main__":
    main()
