"""Prevent incorrect Nibi task-index assumptions in nearest-P-median arrays.

This test prepares the complete 306-config task manifest inside pytest's
temporary directory (no Slurm submission) and independently verifies that
pilot, main, and appendix Slurm index sets cover the intended experiments.

Run on the intended branch:
  python -m pytest -q tests/test_pmedian_nearest_experiment_matrix.py
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


METHOD_NP = "PMEDIAN_NEAREST_TRANSFER"
METHOD_P = "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR"
GROUPS = (
    (range(1, 46), "SCFLP", METHOD_NP, None),
    (range(46, 91), "SCFLP", METHOD_P, None),
    (range(91, 145), "SCMND", METHOD_NP, "MANUSCRIPT_ENVELOPE"),
    (range(145, 199), "SCMND", METHOD_NP, "CURRENT_SOFT_CAPACITY"),
    (range(199, 253), "SCMND", METHOD_P, "MANUSCRIPT_ENVELOPE"),
    (range(253, 307), "SCMND", METHOD_P, "CURRENT_SOFT_CAPACITY"),
)


def _array_indices(script: str, stage: str) -> set[int]:
    match = re.search(
        rf"(?m)^    {re.escape(stage)}\)\n(.*?)^      ;;",
        script,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"Slurm stage {stage!r} is missing"
    spec = re.search(r'ARRAY_SPEC="([^"]+)"', match.group(1))
    assert spec is not None, f"Slurm stage {stage!r} has no task array"
    value = spec.group(1).split("%", 1)[0]
    indices: set[int] = set()
    for segment in value.split(","):
        if "-" in segment:
            lo, hi = (int(x) for x in segment.split("-"))
            indices.update(range(lo, hi + 1))
        else:
            indices.add(int(segment))
    return indices


def test_task_manifest_and_array_stages(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    results = tmp_path / "RESULTS"
    env = dict(os.environ, RESULTS_ROOT_OVERRIDE=str(results))
    completed = subprocess.run(
        ["bash", "submit_pmedian_nearest.sh", "prepare"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr + "\n" + completed.stdout
    assert "Validated all 306 task configs" in completed.stdout

    pointer = results / "MASTER_RUNS" / "latest_pmedian_nearest_master.txt"
    master = Path(pointer.read_text().strip())
    manifest = (master / "task_list.txt").read_text().splitlines()
    assert len(manifest) == 306

    expected_main: set[int] = set()
    expected_appendix: set[int] = set()
    for indices, problem, method, formulation in GROUPS:
        for index in indices:
            path = master / "configs" / f"task_{index:04d}.py"
            content = path.read_text()
            assert f'predictor_ML_Augmented_BD_grid = ["{method}"]' in content
            assert f'    "{problem}": [' in content
            assert manifest[index - 1].endswith("|" + problem)
            if formulation is not None:
                assert f'scmnd_formulation_grid = ["{formulation}"]' in content
                if formulation == "CURRENT_SOFT_CAPACITY":
                    expected_appendix.add(index)
                else:
                    expected_main.add(index)
            else:
                assert 'scmnd_formulation_grid = [' not in content
                expected_main.add(index)

    assert len(expected_main) == 198
    assert len(expected_appendix) == 108

    submission_script = (root / "submit_pmedian_nearest.sh").read_text()
    pilot = _array_indices(submission_script, "pilot")
    main = _array_indices(submission_script, "main")
    appendix = _array_indices(submission_script, "appendix")

    assert pilot == {1, 46, 91, 199}
    assert len(main) == 194
    assert len(appendix) == 108
    assert pilot.isdisjoint(main)
    assert (pilot | main) == expected_main
    assert appendix == expected_appendix
    assert (pilot | main).isdisjoint(appendix)
    assert (pilot | main | appendix) == set(range(1, 307))
