"""
evaluator.py
============
Evaluacion opcional contra ground truth manual.

El archivo esperado es JSON:
{
  "Cedula_0001": {
    "vivienda.agua_entubada.si": false,
    "vivienda.numero_cuartos": "2"
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_checkbox_fields(predictions: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    total = 0
    correct = 0
    misses = []

    for doc_id, expected_fields in truth.items():
        pred_doc = predictions.get(doc_id, {}).get("fields", {})
        for field_id, expected in expected_fields.items():
            pred_field = pred_doc.get(field_id)
            if not pred_field or pred_field.get("type") != "checkbox":
                continue
            total += 1
            got = bool(pred_field.get("value"))
            exp = bool(expected)
            if got == exp:
                correct += 1
            else:
                misses.append({"doc_id": doc_id, "field": field_id, "expected": exp, "got": got})

    return {
        "checkbox_total": total,
        "checkbox_correct": correct,
        "checkbox_accuracy": round(correct / total, 4) if total else None,
        "checkbox_errors": misses,
    }
