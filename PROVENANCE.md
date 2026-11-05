# Provenance

This repository holds the code for the **IPCO 2027** version of Learning-Augmented Benders Decomposition (LABD).

It was created on 2026-11-05 as an exact snapshot (no git history) of:

- Source repo: `Erfan-Rafieikia/Benders_Integration_with_ML`
- Branch: `agent/external-benchmarks-20261101`
- Commit: `6b2d452164523e80a195b53054b32fdf2de1a2b7` (includes the anchor-tolerance fix `d33e906`)

The first commit here is the unmodified copy. All IPCO-specific changes come in later commits, so `git diff <first commit>` shows exactly what differs from the journal (OR) code.

**Do not merge this repo back into `Benders_Integration_with_ML`, and do not run the journal campaigns from here.** The journal campaigns stay frozen on their own commits in the original repo.

Planned IPCO scope (to be implemented on top of this copy):
- Problem class: SCFLP only, |M| = 500 scenarios.
- Baselines: Standard BD and RND (as in the original code).
- Method: parametric dual prediction that predicts the duals directly (no residual/intercept), with a closed-form segment repair toward a feasible reference point (the mean of the exact duals in M_e) and the objective guard.
- Exact-set selection: p-median (agreed with Borzou) and random (variant under consideration).
