"""Regression: Excel cannot truncate full parametric iteration JSON.

PYTHONPATH=src python -m unittest tests.test_parametric_iteration_sidecar -v
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook, load_workbook

from utils.parametric_iteration_sidecar import (
    SIDECAR_SCHEMA,
    externalize_parametric_iteration_records,
    read_parametric_iteration_records,
)


class ParametricIterationSidecarTests(unittest.TestCase):
    def test_large_json_survives_excel_and_round_trips(self):
        with tempfile.TemporaryDirectory() as root:
            workbook_path = Path(root) / "solution_log.xlsx"
            history = [{"iteration": i, "notes": "x" * 100} for i in range(3000)]
            raw = json.dumps(history)
            self.assertGreater(len(raw), 32767)
            original = {"parametric_iteration_records": raw, "method": "PARAMETRIC"}
            external = externalize_parametric_iteration_records(original, workbook_path, 1)
            self.assertEqual(raw, original["parametric_iteration_records"])
            pointer = external["parametric_iteration_records"]
            self.assertLess(len(pointer), 32767)
            reference = json.loads(pointer)
            self.assertEqual(SIDECAR_SCHEMA, reference["schema"])
            self.assertEqual(len(history), reference["records"])
            self.assertEqual(hashlib.sha256(raw.encode("utf-8")).hexdigest(), reference["sha256"])

            wb = Workbook()
            wb.active["A1"] = "parametric_iteration_records"
            wb.active["A2"] = pointer
            wb.save(workbook_path)
            saved = load_workbook(workbook_path, read_only=True, data_only=True)
            cell = saved.active["A2"].value
            saved.close()
            self.assertEqual(history, read_parametric_iteration_records(cell, workbook_path))

            # Idempotent reuse must not silently overwrite the existing sidecar.
            second = externalize_parametric_iteration_records(original, workbook_path, 1)
            self.assertEqual(pointer, second["parametric_iteration_records"])

    def test_separate_rows_and_empty_records(self):
        with tempfile.TemporaryDirectory() as root:
            workbook_path = Path(root) / "solution_log.xlsx"
            first = externalize_parametric_iteration_records(
                {"parametric_iteration_records": "[{\"iteration\":1}]"}, workbook_path, 1
            )
            second = externalize_parametric_iteration_records(
                {"parametric_iteration_records": "[{\"iteration\":1}]"}, workbook_path, 2
            )
            self.assertNotEqual(first["parametric_iteration_records"], second["parametric_iteration_records"])
            self.assertEqual(
                {"parametric_iteration_records": ""},
                externalize_parametric_iteration_records(
                    {"parametric_iteration_records": ""}, workbook_path, 3
                ),
            )
            self.assertEqual(
                [{"iteration": 1}],
                read_parametric_iteration_records(first["parametric_iteration_records"], workbook_path),
            )

    def test_tampered_sidecar_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "solution_log.xlsx"
            entry = externalize_parametric_iteration_records(
                {"parametric_iteration_records": "[1,2,3]"}, path, 1
            )
            pointer = entry["parametric_iteration_records"]
            file = Path(root) / json.loads(pointer)["file"]
            file.write_text("[4,5,6]", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_parametric_iteration_records(pointer, path)

    def test_legacy_inline_json(self):
        self.assertEqual(
            [{"iteration": 1}],
            read_parametric_iteration_records('[{"iteration":1}]', "any/solution_log.xlsx"),
        )

    def test_invalid_input_and_unsafe_path(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "solution_log.xlsx"
            with self.assertRaises(ValueError):
                externalize_parametric_iteration_records(
                    {"parametric_iteration_records": "{not-json"}, path, 1
                )
            malicious = json.dumps({
                "schema": SIDECAR_SCHEMA, "file": "../outside.json",
                "sha256": "0" * 64, "bytes": 2, "records": 0
            })
            with self.assertRaises(ValueError):
                read_parametric_iteration_records(malicious, path)


if __name__ == "__main__":
    unittest.main()
