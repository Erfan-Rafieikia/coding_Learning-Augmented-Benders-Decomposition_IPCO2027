"""Check objective-guard diagnostics survive both original and corrected Excel schemas."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_parametric_guard_pilots.py"
SPEC = importlib.util.spec_from_file_location("labd_guard_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class GuardReportingTests(unittest.TestCase):
    def legacy_row(self):
        return pd.Series({
            "parametric_config": json.dumps({
                "parametric_repair_mode": "LOCAL",
                "parametric_objective_guard": True,
            }),
            "parametric_iteration_records": json.dumps([
                {"iteration": 1, "objective_guard_enabled": True,
                 "objective_guard_checked": 3,
                 "objective_guard_accepted": 2, "objective_guard_rejected": 1},
                {"iteration": 2, "objective_guard_enabled": True,
                 "objective_guard_checked": 4,
                 "objective_guard_accepted": 1, "objective_guard_rejected": 3},
            ]),
            "parametric_prediction_opportunities": 7,
        })

    def test_legacy_excel_json_recovers_actual_counts(self):
        result = comparison.guard_metrics(self.legacy_row())
        self.assertEqual(result["guard_checks"], 7)
        self.assertEqual(result["guard_accepted"], 3)
        self.assertEqual(result["guard_rejected"], 4)
        self.assertIsNone(result["guard_rejected_negative"])
        self.assertIsNone(result["guard_total_improvement"])
        self.assertEqual(result["guard_diagnostic_source"], "excel_iteration_records")

    def test_new_export_prefers_dedicated_columns(self):
        row = self.legacy_row()
        row["parametric_objective_guard_checks"] = 7
        row["parametric_objective_guard_accepted"] = 3
        row["parametric_objective_guard_rejected"] = 4
        row["parametric_objective_guard_rejected_negative"] = 2
        row["parametric_objective_guard_total_improvement"] = 5.25
        result = comparison.guard_metrics(row)
        self.assertEqual(result["guard_diagnostic_source"], "excel_dedicated_columns")
        self.assertEqual(result["guard_rejected_negative"], 2)
        self.assertAlmostEqual(result["guard_total_improvement"], 5.25)

    def test_disabled_or_missing_configuration_is_rejected(self):
        row = self.legacy_row()
        row["parametric_config"] = json.dumps({
            "parametric_objective_guard": False
        })
        with self.assertRaisesRegex(ValueError, "not confirmed"):
            comparison.guard_metrics(row)

    def test_inconsistent_iteration_counts_are_rejected(self):
        row = self.legacy_row()
        row["parametric_iteration_records"] = json.dumps([
            {"objective_guard_enabled": True, "objective_guard_checked": 3,
             "objective_guard_accepted": 1, "objective_guard_rejected": 1}
        ])
        with self.assertRaisesRegex(ValueError, "Inconsistent guard counts"):
            comparison.guard_metrics(row)

    def test_truncated_iteration_json_does_not_invent_counts(self):
        row = self.legacy_row()
        row["parametric_iteration_records"] = (
            '[{"objective_guard_enabled": true, "objective_guard_checked": '
            + "1" * 32710
        )
        result = comparison.guard_metrics(row)
        self.assertIsNone(result["guard_checks"])
        self.assertEqual(result["guard_diagnostic_source"], "excel_iteration_records_truncated")

    def test_fixed_excel_schema_has_all_six_guard_columns(self):
        module = ast.parse((ROOT / "src" / "main.py").read_text())
        columns = None
        for node in module.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DESIRED_SOLVING_LOG_COLUMNS"
                for t in node.targets
            ):
                columns = ast.literal_eval(node.value)
                break
        self.assertIsNotNone(columns)
        for suffix in (
            "enabled", "checks", "accepted", "rejected",
            "rejected_negative", "total_improvement"
        ):
            self.assertIn("parametric_objective_guard_" + suffix, columns)


if __name__ == "__main__":
    unittest.main()
