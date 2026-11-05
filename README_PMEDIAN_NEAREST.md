# Scenario-specific P-median experiments — main manuscript first

## Methods and preservation of old experiments

Branch: `agent/pmedian-scenario-specific-transfer`
(created from `agent/weighting-cross-combinations`).

- `PMEDIAN_NEAREST_TRANSFER` — nonparametric, uses scenario-specific
  nearest selected P-median representative to transfer an exact dual.
- `PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR` — uses that same
  scenario-specific P-median representative as the feasible anchor for
  warm-up-trained RHS-Ridge and its existing feasibility repair.

The prior nonparametric `PMEDIAN_WEIGHTED_TRANSFER` and parametric
`PARAMETRIC_RHS_RIDGE_REPAIR` methods remain unchanged. The Excel
`predictor_type` and `prediction_method` columns keep these exact
different method identifiers.

The *existing* `submit.sh` launches old methods. Do NOT use it for
new P-median experiments. Use only `submit_pmedian_nearest.sh`.

## Scope: 198 main-paper runs first, 108 appendix runs later

| Problem/formulation | Per new method | Both methods | Priority |
|---|---:|---:|---|
| SCFLP | 45 | 90 | Main |
| SCMND `MANUSCRIPT_ENVELOPE` | 54 | 108 | Main |
| SCMND `CURRENT_SOFT_CAPACITY` | 54 | 108 | Appendix, later |
| Total | 153 | 306 | |

Each task runs one method × instance × subproblem share × formulation:
- SCFLP: 5 base instances, 100/300/500 scenarios, 5%/15%/25% exact.
- SCMND: 6 base instances, 100/300/500 scenarios, 5%/15%/25% exact,
  separately for each formulation.
- Warm-up = 5, seed 42, `ML_at_both`, `features_only`.
- 3-hour Gurobi solve limit, 1% master MIP gap; 16-hour Slurm allocation
  accommodates exact objective re-evaluation and diagnostics.
- Job uses previously approved personal WLS license from `~/gurobi.lic`
  and 12 concurrent tasks by default (within approved 20 sessions).
- Result files are segregated in
  `~/scratch/Benders_Integration_with_ML/RESULTS/` under run ID
  `pmedian_nearest_transfer_<TIMESTAMP>`.

The 306 task configurations are prepared together for reproducibility,
but **no appendix task runs until the separate appendix stage is submitted**.

## Connect and test

```bash
ssh rafieiki@nibi.alliancecan.ca
cd ~/Erfan_projects/Benders_Integration_with_ML
git status --short
git fetch origin
git switch agent/pmedian-scenario-specific-transfer
git pull --ff-only

PY=~/micromamba/envs/bd_ml/bin/python
bash -n submit_pmedian_nearest.sh
bash -n my_job.pmedian_nearest.alliance.sbatch
"$PY" -m pytest -q tests/test_pmedian_nearest_transfer.py tests/test_pmedian_nearest_experiment_matrix.py
```

If Git reports local changes preventing branch switching, inspect them;
do not force-reset.

## Stage A: Prepare all 306 configurations (SUBMITS NOTHING)

Always run this again after updating the GitHub branch. Previously prepared
manifests pin their previous Git commit; they cannot be reused after source
changes. The old prepared directory is retained for audit and is not submitted.

```bash
bash submit_pmedian_nearest.sh prepare
MASTER="$(cat ~/scratch/Benders_Integration_with_ML/RESULTS/MASTER_RUNS/latest_pmedian_nearest_master.txt)"
wc -l "$MASTER/task_list.txt"  # exactly 306
sed -n '1p;46p;91p;145p;199p;253p' "$MASTER/task_list.txt"
```

Task indexes are:
- 1–45: nonparametric SCFLP
- 46–90: parametric SCFLP
- 91–144: nonparametric main SCMND (MANUSCRIPT_ENVELOPE)
- 145–198: nonparametric appendix SCMND (CURRENT_SOFT_CAPACITY)
- 199–252: parametric main SCMND (MANUSCRIPT_ENVELOPE)
- 253–306: parametric appendix SCMND (CURRENT_SOFT_CAPACITY)

## Stage B: Four main-paper pilot runs (4/198)

```bash
bash submit_pmedian_nearest.sh pilot "$MASTER"
PILOT_JOB="$(cat "$MASTER/job_id_pilot.txt")"
squeue -j "$PILOT_JOB" -r
sacct -j "$PILOT_JOB" -X -n -P --format=JobID,State,Elapsed,ExitCode
```

Pilot indexes: `1,46,91,199` (one SCFLP and one main-SCMND
case for each method). No appendix experiments are included.

After the pilots have finished, verify the Excel outputs:

```bash
"$PY" scripts/verify_pmedian_nearest_pilots.py "$MASTER"
```

This creates `pilot_verified.txt` only if all four have a matching
method label, SCMND formulation, usable stopping status, and finite
exactly re-evaluated objective. This is a basic output check, not
a guarantee of optimality.

## Stage C: Remaining main-paper cases (194/198)

Only after the four pilots pass:

```bash
bash submit_pmedian_nearest.sh main "$MASTER"
MAIN_JOB="$(cat "$MASTER/job_id_main.txt")"
squeue -j "$MAIN_JOB" -r
sacct -j "$MAIN_JOB" -X -n -P --format=JobID,State,Elapsed,ExitCode
```

The main array is `2-45,47-90,92-144,200-252`, explicitly excluding
the four pilots and all appendix tasks.

After main-paper jobs finish:

```bash
"$PY" scripts/verify_pmedian_nearest_main.py "$MASTER"
```

This checks 198 main-paper Excel files (including the four pilots)
for configuration identity and completeness and reports master-vs-exact
objective drifts above 5%. It writes `main_verified.txt` when all 198
are present and pass the basic screen. Investigate any objective drift
separately before scientific reporting.

## Stage D: Appendix SCMND only (108), after main-paper results

```bash
bash submit_pmedian_nearest.sh appendix "$MASTER"
APPENDIX_JOB="$(cat "$MASTER/job_id_appendix.txt")"
squeue -j "$APPENDIX_JOB" -r
sacct -j "$APPENDIX_JOB" -X -n -P --format=JobID,State,Elapsed,ExitCode
```

The appendix array is `145-198,253-306` and exclusively runs
`CURRENT_SOFT_CAPACITY`. You may defer this until after examining
the main-paper comparisons.

The submission script rejects a stage if a `job_id_<stage>.txt` already
exists, preventing accidental duplicate array submissions.

## Status of earlier recovery experiments

```bash
squeue -j 23780596 -r
sacct -j 23780596 -X -n -P --format=JobID,State,Elapsed,ExitCode |
awk -F'|' '$1 ~ /^23780596_[0-9]+$/ {n[$2]++}
END {for (s in n) print s, n[s]}'
```

Prior production arrays include `23273725`, `23642684`, and
`23716753`. A Slurm COMPLETED status does not, by itself, validate
the experiment's Excel row or objective.

**Operational note:** No remote Nibi tests or jobs have been executed
as part of preparing this branch; run the preflight and pilot stages
before initiating production.
