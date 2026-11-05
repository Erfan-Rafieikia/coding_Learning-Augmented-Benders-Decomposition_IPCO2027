# Final-only reporting change — local validation

Recorded 2026-10-28 (America/Edmonton), based on the audited source revision
`e7e4d4c2ad8127382789eda2e53f7cfbe6d833d1`. This record describes the tested
changes in this commit; it is not evidence of a new Nibi benchmark run.

`scripts/validate_convergence_revision.sh` passed:

- Python compilation and shell syntax checks.
- 88 focused tests, plus 10 parameterized subtests.
- 52 tiny Gurobi cases: all 13 campaign method/repair variants, both problem
  classes, both gap targets. SCMND uses MANUSCRIPT_ENVELOPE.
- Two additional hard-capacity checks: the feasible one-arc model matches
  the known objective 15.5; insufficient base capacity is infeasible even
  when historical scenario capacities would have permitted routing.
- Six complete main.py workbook/CSV exports: standard Benders, Random Sampling
  and RHS-Ridge P-median nearest for each problem class, checked by the
  current-policy strict audit. This verifies standard's zero evaluation cost,
  one decision/eight scenario solves for each inexact fixture, retained
  full-precision objective fields, Gurobi-only traces, and timing partition.

The solver suite reports three tests with 58 subtests; it performs 60 small
master solves in total (52 matrix cases + 2 hard-capacity checks + 6 exports).
These counts are not 60 production experiments.

Environment: Python 3.12.14; Gurobi 13.0.3 restricted local license; NumPy 2.3.5;
gensim 4.4.0; pytest 9.2.0. This is not the Nibi environment.

The expanded workbook check initially exposed strict equality between JSON
timings and Excel floating-point values. The audit now uses a 1e-9-second
absolute / 1e-12 relative tolerance for that comparison; the final full suite
passed after correction. No measured durations are rescaled.

No Slurm jobs were submitted, cancelled or modified. Existing pilot results,
frozen server sources and original workbooks were not changed. The code must
still be fetched and validated on Nibi, followed by representative checks of
the changed hard-capacity/reporting paths before a full campaign. The actual
frozen 1,221-row grid and full launcher remain to be reviewed.

## Slurm workbook gate follow-up (2026-10-28)

Command review found a remaining v1-only check in the Slurm task wrapper.
It would reject a successfully exported v2 workbook. The gate now imports the
current timing, solution-time metric and trace version constants from the
frozen source. A regression executes the actual embedded workbook check: the
current row passes; stale/missing versions and caught-code-error rows fail.
It reproduced the old rejection before the fix. The full validation script
then passed again: 89 focused tests with 18 subtests, followed by all 60 tiny
master solves (three solver tests with 58 subtests). Shell syntax passed.
This fixes the launcher check; Nibi benchmark validation is still pending.

## Exact/ML cut-construction follow-up (2026-10-28)

Based on upstream `7105ffa0253d60075c1298441756d55928be359c`, the follow-up
adds construction-only timings and counts by actual dual origin. The details
are subsets of the existing cut-evaluation/generation category; the 33-category
schema and final-reporting time policy remain unchanged. The bound sampler is
also unchanged. The policy document distinguishes implemented behavior from
the recommendation to record bound changes and iteration boundaries.

The full validation script passed locally: compilation, shell syntax,
**93 focused tests with 28 subtests**, then **three solver tests with 58
subtests (60 tiny master solves)**. Deterministic tests verify exclusion of
nested insertion time, absence of double counting, exception cleanup, exact
versus KNN fallback attribution, and zero construction for nonviolated cuts.
All 52 method/problem/gap matrix cases check construction counts against the
existing exact/ML origin counters and insertion calls. Six workbook/CSV
exports check the new fields and the read-only auditor's detail validation.

Environment: Python 3.12.14, Gurobi 12.0.2 restricted local license, NumPy
1.26.4, SciPy 1.13.1, gensim 4.3.3 and pytest 9.2.0. The initial reconstructed
local checkout lacked the unchanged snapshot-verification helper, causing
test collection errors; restoring that upstream file resolved them. The
final run passed. The existing pysmps matrix deprecation warning remains.

Pilot 25185958 at `7105ffa` is completed benchmark evidence (EV030), but
predates these new fields. No Nibi benchmark was launched or changed here;
new fields are available only in future runs of this revision.

## Bound events and solver cut counts (2026-10-28)

Based on upstream `32eade5c97cf2969a94b75e90c91c32456ba766f`, D022 implements
both callback-observed bound changes and every handled Benders round's start
and end, including zero-cut rounds. The new trace version is
`gurobi_master_v3_events`. Bound queries use the valid codes in each callback
context and never use a MIPSOL candidate objective as an accepted incumbent.

The same CSV records cumulative exact/ML and lazy/user submissions. Workbook
schema `gurobi_cuts_v1` separately records last/peak observed MIP_CUTCNT and
its observation count/time, plus the final log's last-LP cut summary by type.
Missing log summaries remain unavailable, and applied counts may decrease.
No retained exact/ML identity or lifetime acceptance ratio is claimed.
The 33 additive timing categories, construction-time subsets, stopping rules,
final-only exact evaluation policy, and algorithm behavior are preserved.

The full validation script passed locally: compilation, shell syntax,
**102 focused tests with 46 subtests**, then **three solver tests with 58
subtests (60 tiny master solves)**. Tests cover rapid bound/count changes,
callback-code validity, candidate-versus-incumbent semantics, skipped LP
rounds, zero-cut boundaries, per-round submission differences, time/iteration
plot selection, truncated/missing log summaries, decreasing applied counts,
audit tamper detection, and continued support for older v2 traces. All 52
matrix cases check paired boundaries and terminal counts; six complete
workbook/CSV exports pass the current audit and contain parsed log summaries.
The Slurm gate accepts the v3 trace and rejects stale sampled-v2 exports.

Environment: Python 3.12.14, Gurobi 12.0.2 restricted local license, NumPy
1.26.4, SciPy 1.13.1, gensim 4.3.3 and pytest 9.2.0. Early development tests
exposed an incorrect fixture assumption that OutputFlag=0 suppresses MESSAGE
callbacks and a missing mock callback-count dictionary. Both fixtures were
corrected before the final passing run. A restored session also required
repairing the local virtual environment's missing Python launcher; source
code was unaffected. The pysmps matrix deprecation warning remains.

No Nibi jobs were submitted or modified. Pilot 25185958 at `7105ffa` predates
these additions; new measurements require future runs of the new revision.
Validate on Nibi before the full campaign. Its actual frozen 1,221-row grid,
data hashes and full launcher still require review.

## Decisions carried forward

- D019 supersedes D014: hard-capacity MANUSCRIPT_ENVELOPE only for SCMND.
- D020 supersedes D012: final exact reporting time is separate and excluded
  from solution_time_sec; standard Benders reuses its returned objective.
- D021 supersedes D011: use only Gurobi master UB/LB history; remove saved
  candidate storage and historical exact reevaluation.
- D022 extends D021: retain both observed bound changes and Benders round
  boundaries; distinguish submissions from Gurobi applied/last-LP cut counts.
- D001/D002/D004/D018 remain: Gurobi stopping at 0.1% or 1% / time limit,
  full 0.1% campaign first, global max 18 and 128 GiB/task, and original
  downloadable Excel workbooks retained on Nibi.

Historical D016 restricted the earlier documentation-only task. The author's
current request explicitly authorizes these code/documentation changes. It
does not request benchmark submission. Historical events remain in the shared
append-only research log; EV027 records this implementation and its limits.
