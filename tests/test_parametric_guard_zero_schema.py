"""Static smoke checks for gamma-zero fastpath experiment reporting."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FastpathSchemaTests(unittest.TestCase):
    def test_exported_columns_and_values_exist(self):
        source = (ROOT / "src/main.py").read_text()
        tree = ast.parse(source)
        match = next(
            n for n in tree.body
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DESIRED_SOLVING_LOG_COLUMNS"
                for t in n.targets
            )
        )
        columns = ast.literal_eval(match.value)
        for key in (
            "parametric_objective_guard_gamma_zero_fastpath",
            "parametric_objective_guard_skipped_zero",
            "parametric_objective_guard_evaluated",
            "parametric_objective_guard_time_sec",
        ):
            self.assertEqual(columns.count(key), 1)
            self.assertIn(f'"{key}": parametric_diagnostics.get(', source)
        self.assertIn(
            "'parametric_guard_gamma_zero_fastpath': parametric_guard_gamma_zero_fastpath",
            source,
        )

    def test_matched_launcher_contains_all_four_arms(self):
        shell = (ROOT / "submit_parametric_guard_zero_fastpath.sh").read_text()
        self.assertIn("NONPARAM OFF GUARD FAST", shell)
        self.assertIn("parametric_diagnostic_probe = False", shell)
        self.assertIn("parametric_guard_gamma_zero_fastpath = $fast", shell)
        self.assertIn("LABD_DATA_ROOT", shell)
        self.assertIn("32 --mem-per-cpu=8G", shell)
        self.assertIn("sbatch", shell)

    def test_fastpath_default_is_off(self):
        source = (ROOT / "src/config_base.py").read_text()
        self.assertIn("parametric_guard_gamma_zero_fastpath = False", source)


if __name__ == "__main__":
    unittest.main()
