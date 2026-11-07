# IPCO 2027 results, 2026-11-07

Source: `ipco_results_20261107.tgz` packed on Nibi from `~/scratch/LABD_IPCO2027/RESULTS`.

- Standard BD and RND: root `ipco_full_pmedian_20261105_193605` (job 26255335).
- LABD in the paper (p-median, repair starting at the mean, no clipping): root `ipco_cmp_pmedian_v0_20261106_171133` (job 26344540, method `PARAMETRIC_DIRECT_RIDGE_PMEDIAN_MEAN_V0`).

Delta = 100 (V - V_BD) / |V_BD| with V = `original_Prob_Obj_Value`; tau = `solution_time_sec` / BD `solution_time_sec`; cert = 100 x `verified_final_relative_gap` = 100 (V - LB) / V.

Collect with `python scripts/ipco/collect_ipco_results.py <extracted archive dir>`.
