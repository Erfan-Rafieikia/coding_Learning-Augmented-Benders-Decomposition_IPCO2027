"""Check structure-repair comparison logging and isolated launcher contract."""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.config_utils import validate_method_config


class StructureRepairExportTests(unittest.TestCase):
    def test_every_structure_diagnostic_is_in_fixed_excel_schema(self):
        main = (ROOT / "src/main.py").read_text()
        assigns = [
            x for x in ast.parse(main).body if isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "DESIRED_SOLVING_LOG_COLUMNS"
                    for t in x.targets)
        ]
        self.assertEqual(len(assigns), 1)
        cols = ast.literal_eval(assigns[0].value)
        for key in (
            "parametric_repair_mode_actual",
            "parametric_structure_repair_attempts",
            "parametric_structure_repair_feasible_candidates",
            "parametric_structure_repair_anchor_fallbacks",
            "parametric_structure_repair_nonanchor_candidates",
            "parametric_structure_repair_positive_objectives",
            "parametric_structure_repair_positive_gain_total",
            "parametric_structure_repair_selected_kinds",
            "parametric_objective_guard_skipped_equal_anchor",
        ):
            self.assertEqual(cols.count(key), 1, key)
            self.assertIn(f'"{key}": ', main)

    def test_full_method_validation_accepts_structural_repair(self):
        # Regression: the predictor constructor previously accepted
        # STRUCTURE_AWARE but validate_method_config rejected the run
        # before any Benders callback could execute.
        config = {
            "method_category": "ML",
            "ml_method_type": "ML_Augmented_BD",
            "ml_callback_type": "ML_at_integer_only",
            "predictor_type": "PARAMETRIC_RHS_RIDGE_PMEDIAN_NEAREST_ANCHOR",
            "n_trials": 5,
            "parametric_pmedian_solver": "MIP",
        }
        for mode in ("LOCAL", "GLOBAL", "STRUCTURE_AWARE", "structure_aware"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    validate_method_config({**config, "parametric_repair_mode": mode}),
                    (True, ""),
                )
        for mode in ("PROJECT_QP", "STRUCTURAL", ""):
            with self.subTest(invalid=mode):
                accepted, message = validate_method_config(
                    {**config, "parametric_repair_mode": mode}
                )
                self.assertFalse(accepted)
                self.assertIn("parametric_repair_mode", message)

    def test_original_default_unchanged(self):
        config = (ROOT / "src/config_base.py").read_text()
        self.assertIn('parametric_repair_mode_grid = ["LOCAL"]', config)
        self.assertIn("parametric_objective_guard = False", config)

    def test_objective_closure_pilot_is_isolated(self):
        script = (ROOT / "submit_parametric_objective_closure.sh").read_text()
        for marker in (
            'BRANCH="agent/parametric-objective-aware-closure"',
            "NONPARAM LINE STRUCT",
            'run="parametric_objective_closure_$timestamp"',
            'latest_parametric_objective_closure.txt',
            'UNIQUE_ID="parametric_objective_closure"',
        ):
            self.assertIn(marker, script)

    def test_failed_arms_only_recovery_contract(self):
        script = (ROOT / "submit_parametric_structure_repair_recovery.sh").read_text()
        for marker in (
            "for old in 3 6",
            "merge-base --is-ancestor",
            "diff --name-only",
            "src/utils/config_utils.py",
            'if (( new == 1 ))',
            '--array="1-2%',
            "comparison_original_root.txt",
        ):
            self.assertIn(marker, script)
        comparison = (ROOT / "scripts/compare_parametric_structure_repair.py").read_text()
        self.assertIn('if arm == "STRUCT" and recovery is not None:', comparison)
        self.assertIn('output_path = (recovery or root)', comparison)

    def test_experiment_matches_three_methods(self):
        script = (ROOT / "submit_parametric_structure_repair.sh").read_text()
        for text in (
            "NONPARAM LINE STRUCT",
            'parametric_repair_mode_grid = ["$mode"]',
            '[[ "$arm" == STRUCT ]] && mode=STRUCTURE_AWARE',
            'setattr(_fast_main, "parametric_repair_mode_grid", parametric_repair_mode_grid)',
            "LABD_DATA_ROOT",
            "my_job.pmedian_nearest.alliance.sbatch",
            "MAX_CONCURRENT",
        ):
            self.assertIn(text, script)


if __name__ == "__main__":
    unittest.main()
