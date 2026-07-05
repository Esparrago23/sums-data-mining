"""
checkbox_trainer.py
===================
Entrenamiento local del clasificador de checkboxes.

Usa las features visuales producidas por `checkbox_model.py` y el ground truth
manual. No usa modelos OCR preentrenados; entrena un clasificador pequeño sobre
los ejemplos etiquetados de estas cedulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAIN_DOCS = [f"Cédula_{i:04d}" for i in range(1, 9)]
TEST_DOCS = ["Cédula_0009", "Cédula_0010"]

EXCLUSIVE_GROUPS = [
    "vivienda.material_techo",
    "vivienda.material_paredes",
    "vivienda.material_piso",
    "vivienda.agua_entubada",
    "vivienda.energia_electrica",
    "vivienda.cocina_ubicacion",
    "vivienda.cocina_lena",
    "vivienda.excretas",
    "vivienda.perros_gatos_dentro",
    "vivienda.mascotas_vacunas",
    "vivienda.mascotas_esterilizadas",
    "vivienda.red_alcantarillado",
    "vivienda.fosa_septica",
]


@dataclass
class TrainedCheckboxModel:
    pipe: Pipeline
    train_docs: list[str]
    test_docs: list[str]
    n_train: int


def _group_for(field_id: str) -> str:
    parts = field_id.split(".")
    return ".".join(parts[:2]) if len(parts) >= 3 else field_id


def _feature_row(field_id: str, pred_field: dict[str, Any]) -> dict[str, Any]:
    f = pred_field.get("features", {})
    return {
        "field": field_id,
        "group": _group_for(field_id),
        "score": float(f.get("score", 0.0)),
        "ink_ratio": float(f.get("ink_ratio", 0.0)),
        "component_count": float(f.get("component_count", 0.0)),
        "diagonal_ratio": float(f.get("diagonal_ratio", 0.0)),
    }


def _labeled_rows(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    doc_ids: list[str],
) -> tuple[list[dict[str, Any]], list[int], list[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    y: list[int] = []
    keys: list[tuple[str, str]] = []
    for doc_id in doc_ids:
        expected_fields = truth.get(doc_id, {})
        pred_fields = predictions.get(doc_id, {}).get("fields", {})
        for field_id, expected in expected_fields.items():
            pred_field = pred_fields.get(field_id)
            if not pred_field or pred_field.get("type") != "checkbox":
                continue
            rows.append(_feature_row(field_id, pred_field))
            y.append(1 if bool(expected) else 0)
            keys.append((doc_id, field_id))
    return rows, y, keys


def train_checkbox_model(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    train_docs: list[str] | None = None,
    test_docs: list[str] | None = None,
) -> TrainedCheckboxModel | None:
    """Entrena LogisticRegression con docs de entrenamiento."""
    train_docs = train_docs or TRAIN_DOCS
    test_docs = test_docs or TEST_DOCS
    rows, y, _ = _labeled_rows(predictions, truth, train_docs)
    if len(set(y)) < 2:
        return None
    pipe = Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=False)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(rows, y)
    return TrainedCheckboxModel(pipe=pipe, train_docs=train_docs, test_docs=test_docs, n_train=len(y))


def apply_trained_checkbox_model(predictions: dict[str, Any], model: TrainedCheckboxModel) -> None:
    """Actualiza in-place los campos checkbox con probabilidad aprendida."""
    for doc in predictions.values():
        for field_id, pred_field in doc.get("fields", {}).items():
            if pred_field.get("type") != "checkbox":
                continue
            row = _feature_row(field_id, pred_field)
            prob = float(model.pipe.predict_proba([row])[0, 1])
            pred_field["raw_rule_value"] = bool(pred_field.get("value"))
            pred_field["raw_rule_score"] = pred_field.get("features", {}).get("score")
            pred_field["value"] = prob >= 0.5
            pred_field["confidence"] = round(max(prob, 1.0 - prob), 4)
            pred_field["features"]["trained_probability"] = round(prob, 4)


def apply_exclusive_groups(predictions: dict[str, Any]) -> None:
    """Fuerza una sola opcion marcada dentro de grupos mutuamente excluyentes."""
    for doc in predictions.values():
        fields = doc.get("fields", {})
        for group in EXCLUSIVE_GROUPS:
            candidates = [
                (fid, f)
                for fid, f in fields.items()
                if fid.startswith(group + ".") and f.get("type") == "checkbox"
            ]
            if len(candidates) < 2:
                continue
            def score(item: tuple[str, dict[str, Any]]) -> float:
                f = item[1]
                return float(f.get("features", {}).get("trained_probability", f.get("features", {}).get("score", 0.0)))

            winner_id, _ = max(candidates, key=score)
            for fid, field in candidates:
                field["value_before_group_normalization"] = bool(field.get("value"))
                field["value"] = fid == winner_id
                field["group_normalized"] = True


def evaluate_docs(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    doc_ids: list[str],
) -> dict[str, Any]:
    total = 0
    correct = 0
    errors = []
    for doc_id in doc_ids:
        pred_fields = predictions.get(doc_id, {}).get("fields", {})
        for field_id, expected in truth.get(doc_id, {}).items():
            pred_field = pred_fields.get(field_id)
            if not pred_field or pred_field.get("type") != "checkbox":
                continue
            total += 1
            got = bool(pred_field.get("value"))
            exp = bool(expected)
            if got == exp:
                correct += 1
            else:
                errors.append({"doc_id": doc_id, "field": field_id, "expected": exp, "got": got})
    return {
        "docs": doc_ids,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "errors": errors,
    }


def evaluate_labeled_split(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    train_docs: list[str],
    test_docs: list[str],
) -> dict[str, Any]:
    return {
        "train": evaluate_docs(predictions, truth, train_docs),
        "test": evaluate_docs(predictions, truth, test_docs),
    }
