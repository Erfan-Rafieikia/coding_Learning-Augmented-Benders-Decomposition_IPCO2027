#!/usr/bin/env python3
"""Read-only summary of an IPCO root: Slurm state, stopping reason, time, objective and
the parametric diagnostics that show how often raw predictions were infeasible and how
often the objective guard accepted them.

Usage: python scripts/ipco/summarize_ipco_root.py IPCO_ROOT [--csv OUT.csv]
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys

import pandas as pd

KEEP = re.compile(r"(raw_infeasible|raw_predictions|repaired_predictions|full_reversions|"
                  r"certification_failures|objective_guard_(checks|accepted|rejected|skipped)|"
                  r"gamma_(mean|median|min|max)|repair_gamma|direct_|prediction_opportunities|training_failures)", re.I)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--csv")
    a = p.parse_args()
    root = os.path.realpath(a.root)
    stamp = root.split("/MASTER_RUNS/")[1]
    results = os.path.join(os.path.dirname(os.path.dirname(root)), "SCFLP_class", "RESULTS", f"labd_ipco_{stamp}")
    man = pd.read_csv(os.path.join(root, "manifest.tsv"), sep="\t", dtype=str)
    states = {}
    jf = os.path.join(root, "job_id.txt")
    if os.path.exists(jf):
        job = open(jf).read().strip()
        out = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "-o", "JobID,State,Elapsed"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            jid, st, el = line.split("|")
            m = re.fullmatch(rf"{job}_(\d+)", jid)
            if m:
                states[m.group(1)] = (st.split()[0], el)
    rows = []
    for _, t in man.iterrows():
        rec = {"task": t.task_id, "instance": t.instance_label, "method": str(t.method).replace("PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN", "IPCO_PMEDIAN").replace("PARAMETRIC_DIRECT_RIDGE_RANDOM_MEAN", "IPCO_RANDOM"), "ratio": t.selected_ratio,
               "slurm": states.get(t.task_id, ("UNKNOWN", ""))[0],
               "elapsed": states.get(t.task_id, ("", ""))[1]}
        dirs = sorted(glob.glob(os.path.join(results, f"task_{t.task_id}_*")), key=os.path.getmtime)
        x = os.path.join(dirs[-1], "solution_log.xlsx") if dirs else ""
        d = pd.read_excel(x) if x and os.path.exists(x) else None
        if d is not None and len(d) == 0:
            rec["stop"] = "EMPTY_LOG (task running or failed)"
        elif d is not None:
            r = d.iloc[0]
            rec.update(stop=r.get("stopping_reason"), time_s=r.get("solution_time_sec"),
                       obj=r.get("original_Prob_Obj_Value"))
            for col in r.index:
                if KEEP.search(str(col)):
                    rec[col] = r[col]
        else:
            rec["stop"] = "NO_RESULT"
        rows.append(rec)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(df.to_string(index=False))
    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        df.to_csv(a.csv, index=False)
        print("written", a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
