"""Guard diagnostic reporting contract and launch consistency tests."""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticExportTests(unittest.TestCase):
    def test_diagnostics_have_matching_excel_columns_and_values(self):
        source = (ROOT / "src/main.py").read_text()
        tree = ast.parse(source)
        assignments = [
            node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "DESIRED_SOLVING_LOG_COLUMNS"
                    for t in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        exported = ast.literal_eval(assignments[0].value)
        names = (
            "parametric_diagnostic_probe_enabled",
            "parametric_diag_gamma_zero", "parametric_diag_gamma_positive_count",
            "parametric_diag_gamma_positive_mean", "parametric_diag_gamma_max",
            "parametric_diag_gamma_gt_1e12", "parametric_diag_gamma_gt_1e8",
            "parametric_diag_gamma_gt_1e6", "parametric_diag_gamma_gt_1e3",
            "parametric_diag_gamma_gt_01",
            "parametric_diag_probe_checks", "parametric_diag_probe_improved",
            "parametric_diag_probe_negative", "parametric_diag_probe_tied",
            "parametric_diag_probe_positive_total",
            "parametric_diag_probe_negative_total",
            "parametric_diag_probe_max_positive",
            "parametric_diag_probe_anchor_check_time_sec",
            "parametric_diag_probe_objective_time_sec",
            "parametric_diag_probe_total_time_sec",
        )
        for name in names:
            self.assertEqual(exported.count(name), 1, name)
            self.assertIn(f'"{name}": parametric_diagnostics.get(', source)
        self.assertIn("'parametric_diagnostic_probe': parametric_diagnostic_probe", source)

    def test_all_three_method_arms_are_in_launcher(self):
        script = (ROOT / "submit_parametric_guard_diagnostics.sh").read_text()
        self.assertIn("NONPARAM OFF ON", script)
        self.assertIn("parametric_diagnostic_probe = True", script)
        self.assertIn("DIAGNOSTIC_TASKS", script)
        self.assertIn("LABD_DATA_ROOT", script)
        self.assertIn("32 --mem-per-cpu=8G", script)


if __name__ == "__main__":
    unittest.main()
