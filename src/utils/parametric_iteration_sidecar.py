"""Lossless per-run parametric iteration diagnostics stored beside Excel output.

An Excel cell cannot hold more than 32,767 characters. Keep the full JSON
payload in a separately hashed sidecar and a compact reference in the workbook.
This function is called after optimization, not inside a Benders callback.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


SIDECAR_SCHEMA = "parametric_iteration_records_sidecar_v1"


def externalize_parametric_iteration_records(log_entry, workbook_path, row_number):
    """Return a copied log entry with full iteration JSON saved outside Excel.

    Non-parametric/empty records are unchanged. Filenames include the workbook
    row number and content digest so separate results cannot silently collide.
    Saving is atomic; existing matching files are reused and mismatches fail.
    """
    entry = dict(log_entry)
    raw = entry.get("parametric_iteration_records")
    if not isinstance(raw, str) or not raw:
        return entry
    if int(row_number) != row_number or row_number < 1:
        raise ValueError("row_number must be a positive integer")

    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("parametric_iteration_records must serialize a JSON list")

    data = raw.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    directory = Path(workbook_path).parent / "parametric_iteration_records"
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"row_{int(row_number):06d}_{digest[:16]}.json"
    target = directory / filename

    if not target.exists():
        descriptor, temporary_name = tempfile.mkstemp(prefix=".iter_", suffix=".tmp", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Parametric iteration sidecar checksum mismatch: {target}")

    entry["parametric_iteration_records"] = json.dumps(
        {
            "schema": SIDECAR_SCHEMA,
            "file": f"parametric_iteration_records/{filename}",
            "sha256": digest,
            "bytes": len(data),
            "records": len(parsed),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return entry


def read_parametric_iteration_records(workbook_cell, workbook_path):
    """Recover either a legacy inline JSON list or a new hashed sidecar."""
    payload = json.loads(workbook_cell)
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict) or payload.get("schema") != SIDECAR_SCHEMA:
        raise ValueError("Unrecognized parametric iteration record reference")
    root = Path(workbook_path).parent.resolve()
    relative = Path(payload["file"])
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Missing or unsafe parametric iteration record sidecar")
    data = target.read_bytes()
    if len(data) != payload["bytes"] or hashlib.sha256(data).hexdigest() != payload["sha256"]:
        raise ValueError("Parametric iteration sidecar size/checksum mismatch")
    result = json.loads(data)
    if not isinstance(result, list) or len(result) != payload["records"]:
        raise ValueError("Parametric iteration sidecar structure/count mismatch")
    return result
