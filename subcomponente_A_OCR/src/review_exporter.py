"""
review_exporter.py
==================
Exporta una vista compacta para validacion humana.

Incluye solo campos que el usuario necesita comparar rapido contra el PDF:
checkboxes verdaderos y valores numericos ya predichos.
"""

from __future__ import annotations

from typing import Any


def build_review_export(predictions: dict[str, Any]) -> dict[str, Any]:
    """Construye salida compacta por cedula."""
    export: dict[str, Any] = {}
    for doc_id, doc in predictions.items():
        active: dict[str, Any] = {}
        numeric: dict[str, Any] = {}
        review: dict[str, Any] = {}
        for field_id, field in doc.get("fields", {}).items():
            kind = field.get("type")
            if kind == "checkbox" and bool(field.get("value")):
                active[field_id] = {
                    "value": True,
                    "confidence": field.get("confidence"),
                }
            elif kind == "number":
                numeric[field_id] = {
                    "value": field.get("value"),
                    "confidence": field.get("confidence"),
                    "needs_review": bool(field.get("needs_review", False)),
                }
            elif field.get("needs_review"):
                review[field_id] = {
                    "value": field.get("value"),
                    "roi": field.get("roi"),
                    "type": kind,
                }

        export[doc_id] = {
            "campos_activos": active,
            "valores_numericos": numeric,
            "pendientes_revision": review,
        }
    return export
