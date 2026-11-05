"""Deterministic checks for the focused 72-run implementation smoke matrix."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config  # noqa: E402


PREDICTORS = [
    "TREE",
    "CLUSTERING",
    "PARAMETRIC_MANUSCRIPT",
    "PARAMETRIC_PENALIZED_ERRORS",
    "PARAMETRIC_RIDGE_REPAIR",
    "PARAMETRIC_RHS_RIDGE_REPAIR",
]


class SmokeExperimentConfigurationTests(unittest.TestCase):
    def test_local_main_matrix(self):
        self.assertEqual(set(config.problem_classes_to_instances), {"SCFLP", "SCMND"})
        self.assertEqual(len(config.problem_classes_to_instances["SCFLP"]), 3)
        self.assertEqual(len(config.problem_classes_to_instances["SCMND"]), 3)
        self.assertEqual(config.method_categories_grid, ["Exact", "ML"])
        self.assertEqual(
            config.ml_method_types_grid,
            ["ML_Augmented_BD", "Random_Sampling"],
        )
        self.assertEqual(config.predictor_ML_Augmented_BD_grid, PREDICTORS)
        self.assertEqual(config.feature_transition_rule, "MANUSCRIPT")
        self.assertEqual(config.use_covariates_grid, ["features_only"])
        self.assertEqual(config.selected_subproblems_ratio_grid, [0.05])
        self.assertEqual(config.feature_dim_grid, [3])
        self.assertEqual(
            config.scmnd_formulation_grid,
            ["CURRENT_SOFT_CAPACITY", "MANUSCRIPT_ENVELOPE"],
        )

        experiment_groups = (
            len(config.problem_classes_to_instances["SCFLP"])
            + len(config.problem_classes_to_instances["SCMND"])
            * len(config.scmnd_formulation_grid)
        )
        algorithms = 1 + len(PREDICTORS) + 1  # Exact + LABD predictors + Random
        self.assertEqual(experiment_groups * algorithms, 72)

    def test_submit_dry_run_generates_one_task_per_algorithm(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            environment = os.environ.copy()
            environment["RESULTS_ROOT_OVERRIDE"] = temporary_root
            environment["SUBMIT_DRY_RUN"] = "1"
            completed = subprocess.run(
                ["bash", "submit.sh"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Total tasks to submit: 72", completed.stdout)

            task_lists = list(Path(temporary_root).glob("MASTER_RUNS/*/task_list.txt"))
            self.assertEqual(len(task_lists), 1)
            lines = task_lists[0].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 72)
            self.assertEqual(sum("|SCFLP" in line for line in lines), 24)
            self.assertEqual(sum("|SCMND" in line for line in lines), 48)
            self.assertEqual(sum("_Exact.py" in line for line in lines), 9)
            self.assertEqual(sum("ML_Random_Sampling" in line for line in lines), 9)
            self.assertEqual(
                sum("ML_ML_Augmented_BD_" in line for line in lines),
                54,
            )
            self.assertEqual(
                sum("CURRENT_SOFT_CAPACITY" in line for line in lines),
                24,
            )
            self.assertEqual(
                sum("MANUSCRIPT_ENVELOPE" in line for line in lines),
                24,
            )

            for predictor in PREDICTORS:
                self.assertEqual(sum(predictor in line for line in lines), 9)

            override_directory = task_lists[0].parent
            for task_line in lines:
                override_basename, problem_class = task_line.split("|", maxsplit=1)
                override_path = override_directory / override_basename
                ast.parse(override_path.read_text(encoding="utf-8"))
                override = runpy.run_path(str(override_path))

                if problem_class == "SCMND":
                    self.assertEqual(len(override["scmnd_formulation_grid"]), 1)
                else:
                    self.assertNotIn("scmnd_formulation_grid", override)

                if override["ml_method_types_grid"] == ["ML_Augmented_BD"]:
                    self.assertEqual(
                        len(override["predictor_ML_Augmented_BD_grid"]),
                        1,
                    )


if __name__ == "__main__":
    unittest.main()
