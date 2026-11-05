# Lossless parametric iteration history and quiet Random baseline logs

This change affects **result persistence and logging only**. It does **not** change dual prediction, feasibility repair (LOCAL/GLOBAL/STRUCTURE_AWARE), the objective guard, selected scenarios, solver settings, or either Random baseline policy.

## Workbook and sidecar

The `parametric_iteration_records` column now stores a compact JSON reference with `schema`, `file`, `sha256`, `bytes`, and `records` fields.
The full original JSON array is written as a UTF-8 sidecar in `parametric_iteration_records/`, relative to `solution_log.xlsx`, to avoid Excel's 32,767-character cell limit.
Sidecar filenames include a workbook row number and digest prefix. Temporary-file plus atomic rename is used, with SHA-256 verification.

Use `utils.parametric_iteration_sidecar.read_parametric_iteration_records` to read full histories and validate hashes. It also accepts legacy non-truncated inline JSON lists.

**Archive the whole experiment output directory, not only the Excel workbook.** Sidecar files must be retained with their workbook. Missing sidecars should be treated as an integrity issue.

## Random baseline behavior

SCFLP still averages selected exact duals. SCMND still copies the first selected exact dual. The SCMND policy is logged once per predictor, not as a repetitive per-scenario warning.

## Validate before full campaign

```bash
PYTHONPATH=src python -m unittest tests.test_parametric_iteration_sidecar tests.test_random_sampling_quiet_logging -v
```

Additionally run the existing full focused and tiny-solver integration test suites on Nibi.
Original pilot 25292536 and frozen commit 8de5556 remain untouched. These fixes have not yet received Nibi integration validation.
