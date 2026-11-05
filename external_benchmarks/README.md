# External literature benchmarks for LABD

This folder adds a **second, independent test set** built from standard literature instances. It is **additive only**:

- Nothing in `src/` changes.
- The original `DATA/` folder and the running campaign (frozen at `89331ce`) are untouched.
- An external-benchmark campaign reuses the frozen source tree byte-for-byte. Each task config differs from the running campaign only in its instance path.

## Datasets and the rule used to choose them

If a paper publishes stochastic data or a generation method for a benchmark, that data or method is used, together with instances that paper used. Otherwise the project generator is applied.

| Problem | Instances | Stochastic data | Source and papers |
|---|---|---|---|
| SCFLP | OR-Library CAP `cap41`, `cap44`, `cap61`, `cap64`, `cap71` (16 facilities × 50 customers). All are in SCIP's data and in the set used by Ramírez-Pico et al. (cap41–44, 61–64, 71–74). | **SCIP SCFLP method** (Maher, SCIP `examples/SCFLP/src/reader_scflp.c`): per customer sd ~ U(0.1d, 0.2d) (the code; its documentation says 0.3); Normal(d, sd) rounded to an integer; seed 1, with numpy's generator in place of SCIP's. Unit cost = CAP cost ÷ demand, as in SCIP's reader. Covariates: 6 leading singular vectors of the centred demand matrix. | CAP via github.com/scipopt/scip `examples/SCFLP/data` (Apache-2.0). SCIP's model matches the manuscript SCFLP, including the sufficient-capacity constraint for relatively complete recourse. |
| SCMND | Crainic R `r04.1`, `r04.3`, `r05.1`, `r05.3`, `r06.1`, `r06.3`. These classes and the odd cost/capacity variants (.1 .3 .5 .7 .9) are the ones used by Rahmaniani et al. (2018) and Ramírez-Pico et al. | **Published Rahmaniani scenarios**, correlation label 0, the first S of 1,000 (the Bertsimas et al. 2025 convention), weights 1/S, 5 demand-SVD covariates. | github.com/Ragheb2464/R-Instances; networks from github.com/INFORMSJoC/2023.0074 (MIT) |

Sizes:
- CAP: 100, 300, 500, 700, 900, 1,000 and 1,100.
- Rahmaniani: up to 1,000, the number of published scenarios.

The external campaigns use 100/300/500 and 700/900/1,000.

## Files

- `sources/`: raw downloaded files plus `SHA256SUMS` and licences.
- `data/`: built LABD input files, in the formats read by `src/Problem/data.py`. `data/PROVENANCE.json` lists their SHA-256 hashes. Rebuilds are byte-identical.
- `build_external_datasets.py`: rebuilds `data/` from `sources/`. Options choose other instances, scenario counts or correlation levels.
- `check_external_datasets.py`: loads every file with the frozen readers and checks relatively complete recourse.
- `../scripts/prepare_external_benchmark_campaign.py`: creates a frozen 1,221-task campaign root from an existing frozen root. It never submits.

## Running on Nibi

```bash
cd ~/Erfan_projects/labd_external_benchmarks            # worktree of this branch
# data/ is not in git (the repository ignores folders named data), so build it on Nibi.
# The build is byte-identical to the one made when this branch was written.
~/micromamba/envs/bd_ml/bin/python external_benchmarks/build_external_datasets.py --out-root ~/Erfan_projects/LABD_EXTERNAL_DATA
~/micromamba/envs/bd_ml/bin/python external_benchmarks/check_external_datasets.py ~/Erfan_projects/LABD_EXTERNAL_DATA
python scripts/prepare_external_benchmark_campaign.py \
  --template-root /scratch/rafieiki/Benders_Integration_with_ML/RESULTS/MASTER_RUNS/labd_convergence_full_0p1pct_20261102_141136 \
  --data-root ~/Erfan_projects/LABD_EXTERNAL_DATA \
  --results-root /scratch/rafieiki/Benders_Integration_with_ML/RESULTS_EXTERNAL --gap 0.001
bash ~/labd_launcher/submit_convergence_full_batched_squeuefix.sh <FULL_ROOT> --dry-run
```

The batched launcher allows only one LABD full array in Slurm at a time, whichever campaign it belongs to. External batches therefore run between or after the original campaign's batches.

## Holmberg reading bug (found and fixed 2026-11-01)

Every `SCFLP_data_generation*.py` used to read the first *m* numbers after the header as capacities and the next *m* as fixed costs. Holmberg files store each facility as a *(capacity, fixed cost)* pair. The scripts also used the whole-demand assignment cost c_ij as a unit cost.

This was confirmed on Nibi: the running campaign's `p4_stochastic.txt` has capacities `258 700 128 700 296 700 225 700 310 700` and fixed costs `373 700 329 700 179 700 114 700 321 700`. The true values are capacities `258 128 296 225 310 373 329 179 114 321` and fixed costs of 700 throughout. Demands were read correctly.

**The fix.** All four generators now call `src/utils/holmberg_format.read_holmberg_instance`. It reads the pairs and divides c_ij by d_j, the multiple-sourcing convention of Lu et al. (2014) and of SCIP's CAP reader.

**Not changed.**
- The solver reader (`src/Problem/data.py`).
- The running campaign: it executes its own frozen copy of 89331ce.
- The original `DATA/`.

**Corrected files.** The corrected Holmberg data is regenerated in place on Nibi with `scripts/regenerate_corrected_data.py`, submitted via `regenerate_corrected_data.sbatch`. It covers p1–p71 at 100–1,100 scenarios with the same generator, settings and seed. The old stochastic folders are moved to `DATA/SCFLP/_INVALID_swapped_capacity_fixedcost_pre20261101/`. The deterministic p-files are the published inputs and are not changed. Holmberg's p26 file contains the instance twice, with a 1995 e-mail header in between; the first copy is used.

**Campaigns.** Six scenario sizes give two 1,221-task campaigns per gap, built with `prepare_external_benchmark_campaign.py`:
- `--scflp-family HOLMBERG --scmnd-family GENERATED` (100/300/500)
- the same with `--size-map 100:700 300:900 500:1100`

The R-network reader (`.dow`: tail, head, unit cost, capacity, fixed cost) matches the Kothari–Luedtke and Bertsimas et al. code. `verify_current_scmnd_data.py` checks the current SCMND JSON files against their `.dow` networks.
