# IPCO 2027 results, 2026-11-08 (100, 300, 500 scenarios)

Source: `ipco_results_20261108.tgz` from `~/scratch/LABD_IPCO2027/RESULTS`.

| Scenarios | Root | Job |
|---|---|---|
| 100 | ipco_full100_pmedian_v0_20261107_211627 (BD, RND, LABD) | 26455979 |
| 300 | ipco_full300_pmedian_v0_20261107_211634 (BD, RND, LABD) | 26455980 |
| 500 | ipco_full_pmedian_20261105_193605 (BD, RND) and ipco_cmp_pmedian_v0_20261106_171133 (LABD) | 26255335, 26344540 |

LABD = PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_V0. Metrics as in results/ipco_20261107/README.md.
Rebuild: `python scripts/ipco/collect_ipco_results.py <dir>` then `python scripts/ipco/ipco_metrics_all.py`.
