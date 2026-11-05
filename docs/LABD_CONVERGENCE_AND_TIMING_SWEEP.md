# LABD convergence, final reporting and timing policy

Updated 2026-10-28 (America/Cambridge_Bay). Active specification on
`agent/labd-convergence-diagnostics-gap-sweep`. Author decisions D019–D022
supersede D011 (historical exact UB reconstruction), D012 (inclusive solution
time) and D014 (unresolved SCMND model choice).

## Evaluate only the final returned decision

Use Gurobi's final first-stage decision regardless of its stopping reason.
Do not search historical candidates for a different decision.

| Item | Standard Benders | Every inexact method, including Random Sampling |
| --- | --- | --- |
| True objective (`original_Prob_Obj_Value`) | Reuse Gurobi ObjVal | Exactly solve every scenario for the returned decision |
| Additional final-decision evaluations | 0 | 1, if an incumbent exists |
| Additional final scenario solves | 0 | Number of scenarios |
| Additional reporting time | Exactly 0 | Measure separately; exclude from solution time |
| Historical candidate snapshots/evaluations | Removed | Removed |

For 100 scenarios, one final evaluation means 100 scenario solves. It uses the
same formulation, capacities and scenario weights as the run. Both exact
Benders callback modes skip this postsolve work. No incumbent means an
unavailable true objective and zero postsolve evaluations/time, with status
`no_incumbent`. Failed exact evaluation records `failed`, an unavailable true
objective and actual attempt count/time; the current audit rejects it. Never
substitute the surrogate objective when exact reporting fails.

## Gurobi bound history and stopping

CSV schema `gurobi_master_v3_events` records **both** observed bound-pair
changes and every handled Benders iteration's start/end, plus a terminal
record. The five-second throttle is removed. A change in Gurobi's observed
applied-cut count also creates a record, even if the bounds stay unchanged.

Each row retains `elapsed_seconds`, `iteration`, `node_count`, `event`,
`lower_bound`, `master_upper_bound`, `master_relative_gap`,
`lb_ub_inconsistent`, and `trace_schema_version`. New fields are:

| Field | Meaning |
| --- | --- |
| `callback_context` | MIP, MIPSOL, MIPNODE, or TERMINAL |
| `bound_changed` | Whether the available bound pair differs from the preceding row |
| `gurobi_mip_cut_count` | Gurobi MIP_CUTCNT at this MIP callback; blank in other contexts |
| `gurobi_cut_count_changed` | Whether that count differs from its preceding MIP observation |
| `submitted_exact_cuts`, `submitted_ml_cuts` | Cumulative submissions by actual dual origin |
| `submitted_lazy_cuts`, `submitted_user_cuts` | The same submissions partitioned by cbLazy/cbCut |

Events are `solver_initial`, `solver_bound_change`, `solver_cut_change`,
`solver_iteration_start`, `solver_iteration_end`, and `solver_final`.
Both boundary rows are saved even for a round with unchanged bounds and zero
cuts. Subtract start from end submission counters to obtain that round's
exact/ML and lazy/user cut counts; do not add the two partitions together.

Here `iteration` counts handled Benders callback rounds (integer candidates
and eligible LP nodes), not all Gurobi internal iterations. **Bounds can
change between consecutive Benders rounds.** Query the appropriate OBJBND and
OBJBST codes in MIP, MIPSOL and MIPNODE contexts. MIPSOL_OBJ is a candidate,
not necessarily an accepted incumbent, and is never used for the UB. End
records contain bounds visible before Gurobi processes submitted cuts and
reoptimizes; they do not claim that this work has already happened. The
terminal record uses final model ObjBound/ObjVal.

Preserve raw observed values and timestamps without smoothing, clamping or
backdating. Unavailable bound sentinels are blank. Recording changes has no
five-second delay, but can only capture updates visible through callbacks,
not every internal update. No first-stage snapshots or additional exact
subproblem evaluations are needed. Observation/parsing work remains in the
existing `diagnostics_logging_file_io` timer inside algorithm time.

For inexact methods these are surrogate/master bounds, not verified original
upper bounds. Final exact evaluation only populates workbook diagnostics; it
never replaces the history. `verified_final_relative_gap` uses the final
returned decision's true objective and terminal LB, not the best historical
candidate. Interpretation as an original gap still requires valid cuts.

Stopping remains Gurobi MIPGap `0.001` (0.1%) or `0.01` (1%), or TimeLimit
`10800` seconds. No verified-gap stopping rule is added.

```bash
python scripts/plot_labd_convergence.py --x time --output convergence.pdf path/to/traces/*.csv
# Optional: --x iteration or --gap
```

The plotter labels master objectives/gaps. For historical CSVs it uses the
recorded `master_incumbent_proxy` and solver events, never retrospective UBs.
For v3, time plots use all observations; iteration plots use completed-round
(`solver_iteration_end`) observations and the final state. A final update can
share the last iteration number because Gurobi finishes after the callback.

## Solution time, reporting time, and all 33 components

| Workbook field | Meaning |
| --- | --- |
| `solution_time_sec` | Algorithm elapsed time, excluding final reporting evaluation |
| `final_exact_evaluation_time_sec` | Separate reporting-only evaluation cost |
| `final_objective_validation_time_sec` | Compatibility alias of that same reporting interval; do not add twice |
| `end_to_end_wall_time_sec` | Inclusive solution plus final reporting interval, before exports |
| `experiment_solve_wall_time_sec` | Compatibility alias of the inclusive interval |
| `gurobi_solver_runtime_sec` | Raw Gurobi Runtime; convergence time axis |
| `master_optimize_wall_time_sec` | Monotonic wall interval around optimize() |
| `final_exact_evaluation_count` | 0 for standard, 1 for inexact with an incumbent |
| `final_exact_subproblem_solves` | Actual final scenario solve attempts |
| `final_exact_evaluation_status` | reused_gurobi_objective, evaluated, no_incumbent, or failed |

Solution time includes loading, construction, preprocessing, feature generation,
selection/training, prediction, repair, cuts, and all exact scenario solves
needed during optimization. Workbook/CSV/plot exports remain outside both
elapsed intervals.

```text
solution_time_sec + final_exact_evaluation_time_sec = end_to_end_wall_time_sec
sum(end_to_end_component_breakdown_json.values()) = end_to_end_wall_time_sec
solution_component_sum_sec = solution_time_sec
```

The same 33 category names, operation counts and phase tags remain available
in `src/utils/exclusive_profiler.py`. Final evaluation is one component of the
inclusive total. Retired `all_candidate_expost_reevaluation`, its count,
`candidate_reevaluation_time_sec`, and `num_solutions_reevaluated` are zero.
Historical objective-drift statistics are blank because they are not measured.

Versions: `exclusive_33_v2_final_only` (timing),
`algorithm_excludes_final_reporting_v2` (solution-time metric), and
`gurobi_master_v3_events` (trace). Residual buckets remain explicit. Never add phase
spans to their contained components, hide negative residuals, or rescale
measurements. JSON/Excel times are compared with floating-point tolerance.

Retain original `solution_log.xlsx` workbooks on Nibi, all sheets and full
precision objective values. Use distinct output roots. Never pool old inclusive
v1 solution times with v2 algorithm times in speed ratios. The internal legacy
`Solution.solution_time` attribute remains Gurobi Runtime; main.py computes the
exported `solution_time_sec` metric.

### Exact versus ML cut construction

New workbooks additionally export `cut_construction_timing_version =
exact_ml_construction_v1` and these fields:

| Field | Meaning |
| --- | --- |
| `exact_cut_construction_time_sec` | Construct expressions from exactly solved scenario duals |
| `ml_cut_construction_time_sec` | Construct expressions from predicted or transferred duals |
| `exact_cut_construction_count` | Timed exact-origin construction calls |
| `ml_cut_construction_count` | Timed ML/transfer-origin construction calls |

Origin follows the dual used for the cut, not the overall method name. Exact
includes Phase 0/1, selected scenarios, exact fallbacks and exact switches.
ML includes Random transfers, parametric/nonparametric predictions and KNN
fallbacks; a repaired predicted dual remains in the ML group.

These times measure expression construction for cuts submitted by the callback.
They exclude solving subproblems, evaluating the dual objective for the
violation test, prediction, repair, and the nested `cbLazy`/`cbCut` insertion
interval. Those retain their existing timers. The two construction times are
**subsets already included in `cuts_evaluation_generation`**; do not add them
again to the 33-category total. Their sum can be smaller than that parent
category because dual-objective evaluation is also in the parent.

The existing timing and solution-time versions remain unchanged. The auditor
checks these details when present (nonnegative values, construction time no
larger than its parent, and construction counts matching insertion calls).
Earlier workbooks without the fields remain auditable; their split is unknown,
not zero, and cannot be recovered from the pooled time alone.

### What the cut counts mean

The callback checks violation, submits a cut through `cbLazy` at integer
candidates or `cbCut` at LP relaxations, and increments its origin counter.
These are **submission counts**. Repeating the same inequality twice counts
twice; they are not deduplicated counts of Gurobi's final model rows.

For lazy constraints, the callback rejects a violating candidate; the callback
must still check subsequent candidates even against previously submitted lazy
constraints. For LP user cuts, Gurobi can filter cuts or keep them in its pool
without immediately using them in the LP. Violation alone does not imply that
a user cut is selected for the active LP. This distinction is not evidence that
any particular cut was lost in our pilot. See Gurobi's
[callback documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.cbLazy)
and [staff explanation of user-cut filtering](https://support.gurobi.com/hc/en-us/community/posts/36303960406417-User-Cuts-Callback-Efficiency-and-Internal-Cut-Generation).

For analysis, sum exact origins (`trial`, `selected`, `fallback`, `exact_switch`)
over integer/LP counters, plus `phase0_cuts_added`; ML is `cuts_mip_ml +
cuts_rel_ml`. Do not also add the derived `unselected_total` columns. These are
run totals. The v3 trace additionally supplies per-round counts through the
cumulative start/end records described above.

### Gurobi's applied-cut counts (additional, different measurements)

Workbook schema `gurobi_cut_count_schema_version = gurobi_cuts_v1` exports:

| Workbook field | Meaning |
| --- | --- |
| `gurobi_mip_cut_count_last_observed` | Last observed MIP_CUTCNT, not a terminal model attribute |
| `gurobi_mip_cut_count_peak_observed` | Maximum of observed MIP_CUTCNT values |
| `gurobi_mip_cut_count_last_observed_time_sec` | Solver time of the last query, even if its value was unchanged |
| `gurobi_mip_cut_count_observation_count` | Number of successful MIP_CUTCNT queries |
| `gurobi_last_lp_cut_summary_status` | reported, not_reported, or incomplete |
| `gurobi_last_lp_cut_counts_json` | All cut-type counts in the final log's Cutting planes section |
| `gurobi_last_lp_total_cut_count` | Sum of that section, including Gurobi-generated types |
| `gurobi_last_lp_user_cut_count` | User entry in that complete section |
| `gurobi_last_lp_lazy_constraint_count` | Lazy constraints entry in that complete section |

MIP_CUTCNT is Gurobi's aggregate **current count of applied cutting planes**,
queried only at MIP callbacks. It can decrease as well as increase; it is not
a cumulative accepted-cut count. Last/peak values are observations, and a
missing MIP observation leaves them blank. The streaming MESSAGE callback
captures the final log summary without retaining the entire log. That section
counts cuts present in the **last LP relaxation**, not all cuts ever used.
A missing or incomplete section leaves its numeric fields blank, not zero;
an absent type within a complete reported section is zero.

These APIs do not expose retained cuts by our exact/ML origins or a lifetime
unique acceptance count. Solver-generated cuts are included in the aggregate,
and pool/active-LP handling can change the count. Therefore do not divide a
solver count by our submissions and call it an acceptance rate. A difference
between them does not mean that a violated lazy constraint was ignored.

See Gurobi's [callback codes](https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/callbacks.html)
and [MIP log summary definition](https://docs.gurobi.com/projects/optimizer/en/current/concepts/logging/mip.html).
The audit checks complete paired rounds, monotone submission counters,
terminal submission reconciliation, valid callback contexts, last/peak solver
count reconciliation, and explicit missing-summary handling. It permits
solver cut counts to decrease.

## Hard-capacity SCMND only

The active study uses `MANUSCRIPT_ENVELOPE`: fixed base arc capacities, hard
scenario capacity constraints, and first-stage envelope flows for maximum
sampled commodity demands. It does not buy excess capacity at a penalty.

The default grid contains only this mode. main.py rejects soft-capacity SCMND
overrides. Pilot preparation may read historical soft overrides, but changes
only the formulation in NEW copies, records the hard formulation in the new
manifest, and freezes/hashes those inputs. Original configs, code snapshots,
datasets and workbooks remain intact. Legacy soft-capacity internals remain
for historical code comparisons and are excluded from the active campaign.

## Evidence, validation, and next steps

Historical pilot 25069401 at `e7e4d4c2ad8127382789eda2e53f7cfbe6d833d1`:
20/20 completed; strict v1 audit zero failures and 12 gap-warned runs, per the
author-supplied transcript (EV025). Its SCMND model was soft capacity and its
solution time included final and historical evaluations. It is historical
instrumentation evidence, not hard-model results or v2 timing validation.

See [FINAL_REPORTING_VALIDATION.md](FINAL_REPORTING_VALIDATION.md) for local
checks. Synthetic cases do not establish Nibi benchmark performance or memory
requirements. Before the full campaign, validate this revision on Nibi, check
the changed hard-SCMND/reporting paths on representative data, inspect the
actual frozen 1,221-row grid/data hashes, and finish the full launcher. An
identical rerun of the old soft pilot is not the goal. The current pilot
launcher remains limited to 20 tasks.

Subsequent author-supplied evidence: pilot 25185958 at `7105ffa` completed all
20 tasks with master status OPTIMAL; the current-policy audit reported zero
failures and eight warned runs (EV030). That pilot predates the separate cut
construction fields and v3 event/cut trace. Preserve its frozen source and results. The cut-timing and event/cut-trace
follow-ups are validated locally; it has not been benchmarked on Nibi.

The full plan remains 1,221 runs at 0.1% first, then 1,221 at 1%, max 18
concurrent, 16 CPUs and 128 GiB per task. No benchmark jobs were submitted by
this implementation/documentation update.

From a fresh checkout, run validation without submitting jobs:

```bash
export LABD_PYTHON="$HOME/micromamba/envs/bd_ml/bin/python"
bash scripts/validate_convergence_revision.sh
```

Later, `scripts/prepare_convergence_pilots.py` accepts the original master,
results and data roots, creates new hard-capacity SCMND copies, and submits
nothing. The submitter defaults to dry-run, verifies source/config/data hashes,
and uses one `1-20%18` array with explicit Gurobi thread allocation.

The Slurm task's workbook gate imports the current timing, solution-time
metric and trace version constants from the frozen source. Use the launcher
follow-up revision: the initial final-reporting commit `3e8a588` still had a
v1-only launcher check that would incorrectly fail successful v2 exports.

After a new pilot completes:

```bash
"$LABD_PYTHON" scripts/audit_convergence_pilots.py \
  --pilot-root "$PILOT_ROOT" --job-id "$JOB_ID" --require-current --detail
```

`--require-current` enforces v2 timing, final-only evaluation counts, v3 Gurobi
event/cut traces and hard SCMND. Without that flag, the audit also accepts
legacy `gurobi_master_v2` sampled traces without inventing their missing fields. `--require-33` also understands historical v1 workbooks
without reinterpreting their saved times. Do not use `--require-current` for
the historical pilot.

The Random baseline remains unchanged: SCFLP averages selected duals; SCMND
copies the first selected dual. Manuscript prose alignment remains open.
