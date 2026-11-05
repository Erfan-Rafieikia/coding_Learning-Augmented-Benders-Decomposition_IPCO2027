#!/usr/bin/env python3
"""Compare 2G and 8G per CPU on the same Standard BD pilot data.
Validate solver trajectories and output timing/discrepancy report.
"""
import csv
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

BASE=Path.home()/"scratch/Benders_Integration_with_ML/RESULTS"
REF=BASE/"MASTER_RUNS/pmedian_standard_bd_reference_20261019_125011"

def load(run_id,task,problem):
    paths=list((BASE/f"{problem}_class"/"RESULTS"/run_id).glob(f"task_{task}_*/solution_log.xlsx"))
    if len(paths)!=1: raise ValueError(f"Expected one result {run_id} {task} {problem}; found {paths}")
    rows=pd.read_excel(paths[0])
    if len(rows)!=1: raise ValueError(f"Wrong result row count: {paths[0]}")
    return rows.iloc[0].to_dict()

def csvtab(path):
    with path.open(newline="") as f: return list(csv.DictReader(f,delimiter="\t"))

def close(a,b):
    try: return math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-5)
    except (TypeError,ValueError): return str(a)==str(b)

def audit(ref,row,item):
    for key in ("instance_filename","problem_class","scmnd_formulation","exact_callback_type",
                "use_stochastic_capacity","objective_value","ObjBound_LB","iteration",
                "bnb_nodes","nodes_explored","stopping_reason"):
        if key=="scmnd_formulation" and item["problem"]!="SCMND": continue
        if not close(ref.get(key),row.get(key)):
            raise ValueError(f"Mismatch {item['memory_per_cpu']} {item['problem']} repeat={item['replication']} {key}: {ref.get(key)!r} vs {row.get(key)!r}")
    if str(row.get("method_category")).lower()!="exact": raise ValueError("Not Standard BD")

def main():
    if len(sys.argv)!=2: raise SystemExit("Usage: python scripts/compare_standard_bd_memory_audit.py MASTER_ROOT")
    root=Path(sys.argv[1]).expanduser()
    rows=csvtab(root/"manifest.tsv")
    jobs=csvtab(root/"jobs.tsv")
    ref={46:load(REF.name,1,"SCFLP"),199:load(REF.name,2,"SCMND")}
    output=[]
    groups=defaultdict(list)
    for item in rows:
        memory=item["memory_per_cpu"]
        row=load(item["run_id"],int(item["task"]),item["problem"])
        old=ref[int(item["pilot"])]
        audit(old,row,item)
        duration=float(row["solution_time_sec"])
        previous=float(old["solution_time_sec"])
        entry={**item,"runtime_sec":duration,"previous_8G_reference_sec":previous,
               "runtime_over_previous_8G":duration/previous}
        output.append(entry)
        groups[(item["problem"],memory)].append(duration)
        print(f"{item['problem']:<5} rep={item['replication']} {memory}: {duration:8.2f} s; ratio to previous reference={duration/previous:.3f}")
    print("\nMedian Gurobi runtimes:")
    for problem in ("SCFLP","SCMND"):
        for mem in ("2G","8G"):
            values=groups[(problem,mem)]
            print(f"  {problem}, {mem}: n={len(values)}, median={statistics.median(values):.2f}, min={min(values):.2f}, max={max(values):.2f}")
        ratio=statistics.median(groups[(problem,"2G")])/statistics.median(groups[(problem,"8G")])
        print(f"  {problem}, 2G / 8G median runtime = {ratio:.4f}")
    path=root/"standard_bd_memory_audit_comparison.csv"
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print("\nActual Slurm allocations and MaxRSS (including step records):")
    for job in jobs:
        print(f"\n{job['memory_per_cpu']} job {job['job_id']}:")
        try:
            result=subprocess.run(["sacct","-j",job["job_id"],"-n","-P",
              "--format=JobID,State,Elapsed,ReqMem,AllocTRES,NodeList,MaxRSS"],
              capture_output=True,text=True,check=True)
            print(result.stdout.strip())
        except (OSError,subprocess.CalledProcessError) as e:
            print(f"Cannot read sacct: {e}")
    print(f"\nSaved: {path}")
    print("Check Slurm job logs for CPU models and library versions.")
    print("Matching solver counts and objectives does not imply identical runtime.")
if __name__=="__main__": main()
