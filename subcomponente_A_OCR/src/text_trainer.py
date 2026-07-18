"""
text_trainer.py
===============
Entrenamiento ligero para campos de texto manuscrito/libre en la cédula.

Usa ROIs ya extraidos por `field_extractor` y un KNN de 1 vecino para predecir
los valores de texto basándose en ejemplos del ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from sklearn.neighbors import KNeighborsClassifier

TEXT_FIELDS = {
    "familia.nombre_informante",
    "familia.domicilio",
    "familia.localidad",
    "familia.manzana",
    "familia.vivienda",
}
TRAIN_DOCS = [f"Cédula_{i:04d}" for i in range(1, 9)]
TEST_DOCS = ["Cédula_0009", "Cédula_0010"]


@dataclass
class TrainedTextModel:
    clf: KNeighborsClassifier
    train_docs: list[str]
    test_docs: list[str]
    n_train: int
    field_id: str
    trained_labels: list[str]


def _imread_gray(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def text_vector_from_roi(path: str | Path, height: int = 48, width: int = 200) -> np.ndarray | None:
    gray = _imread_gray(path)
    if gray is None or gray.size == 0:
        return None

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Limpiar ruido y bordes, pero conservar las formas del texto.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    if processed.size == 0:
        return None

    resized = cv2.resize(processed, (width, height), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0).reshape(-1)


def _confidence_from_distance(distance: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    return round(float(1.0 / (1.0 + distance)), 4)


def _clean_ocr_text(text: str) -> str:
    cleaned = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    return cleaned


def ocr_text_from_roi(path: str | Path, lang: str = "eng", psm: int = 7) -> str | None:
    gray = _imread_gray(path)
    if gray is None or gray.size == 0:
        return None

    # Incrementar resolución para OCR si el recorte es pequeño.
    height, width = gray.shape[:2]
    if width < 300:
        factor = max(1.0, 300.0 / float(width))
        gray = cv2.resize(gray, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = f"--oem 1 --psm {psm}"
    text = pytesseract.image_to_string(thresh, lang=lang, config=config)
    result = _clean_ocr_text(text)
    return result if result else None


def _labeled_rows(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    doc_ids: list[str],
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[str]]]:
    rows: dict[str, list[np.ndarray]] = {field: [] for field in TEXT_FIELDS}
    y: dict[str, list[str]] = {field: [] for field in TEXT_FIELDS}

    for doc_id in doc_ids:
        expected_fields = truth.get(doc_id, {})
        pred_fields = predictions.get(doc_id, {}).get("fields", {})
        for field_id in TEXT_FIELDS:
            expected = expected_fields.get(field_id)
            if expected in (None, ""):
                continue
            pred_field = pred_fields.get(field_id)
            if not pred_field or pred_field.get("type") != "text":
                continue
            roi = pred_field.get("roi")
            if not roi:
                continue
            vec = text_vector_from_roi(roi)
            if vec is None:
                continue
            rows[field_id].append(vec)
            y[field_id].append(str(expected).strip())

    return rows, y


def train_text_models(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    train_docs: list[str] | None = None,
    test_docs: list[str] | None = None,
) -> dict[str, TrainedTextModel]:
    train_docs = train_docs or TRAIN_DOCS
    test_docs = test_docs or TEST_DOCS
    rows, y = _labeled_rows(predictions, truth, train_docs)
    models: dict[str, TrainedTextModel] = {}
    for field_id in TEXT_FIELDS:
        if len(rows[field_id]) < 2 or len(set(y[field_id])) < 2:
            continue
        clf = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        clf.fit(np.vstack(rows[field_id]), y[field_id])
        models[field_id] = TrainedTextModel(
            clf=clf,
            train_docs=train_docs,
            test_docs=test_docs,
            n_train=len(y[field_id]),
            field_id=field_id,
            trained_labels=sorted(set(y[field_id])),
        )
    return models


def apply_ocr_text(predictions: dict[str, Any]) -> None:
    for doc in predictions.values():
        for field_id, pred_field in doc.get("fields", {}).items():
            if field_id not in TEXT_FIELDS or pred_field.get("type") != "text":
                continue
            roi = pred_field.get("roi")
            if not roi:
                continue
            text = ocr_text_from_roi(roi)
            if text is None:
                continue
            pred_field["value"] = text
            pred_field["needs_review"] = False
            pred_field["confidence"] = 0.65
            pred_field["model"] = "Tesseract OCR"


def apply_text_model(predictions: dict[str, Any], models: dict[str, TrainedTextModel]) -> None:
    for doc in predictions.values():
        for field_id, pred_field in doc.get("fields", {}).items():
            if field_id not in models or pred_field.get("type") != "text":
                continue
            roi = pred_field.get("roi")
            if not roi:
                continue
            vec = text_vector_from_roi(roi)
            if vec is None:
                continue
            model = models[field_id]
            pred = str(model.clf.predict([vec])[0])
            distance = float(model.clf.kneighbors([vec], n_neighbors=1, return_distance=True)[0][0][0])
            pred_field["value"] = pred
            pred_field["needs_review"] = False
            pred_field["confidence"] = _confidence_from_distance(distance)
            pred_field["model"] = "KNeighborsClassifier(k=1, cosine) text field"
            pred_field["features"]["trained_distance"] = round(distance, 4)


def evaluate_text_docs(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    doc_ids: list[str],
) -> dict[str, Any]:
    total = 0
    correct = 0
    errors: list[dict[str, str]] = []
    for doc_id in doc_ids:
        pred_fields = predictions.get(doc_id, {}).get("fields", {})
        expected_fields = truth.get(doc_id, {})
        for field_id in TEXT_FIELDS:
            expected = expected_fields.get(field_id)
            if expected in (None, ""):
                continue
            pred_field = pred_fields.get(field_id)
            if not pred_field or pred_field.get("type") != "text":
                continue
            total += 1
            got = str(pred_field.get("value", "")).strip().lower()
            exp = str(expected).strip().lower()
            if got == exp:
                correct += 1
            else:
                errors.append({"doc_id": doc_id, "field": field_id, "expected": str(expected), "got": got})

    return {
        "docs": doc_ids,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "errors": errors,
    }


def evaluate_text_split(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    train_docs: list[str],
    test_docs: list[str],
) -> dict[str, Any]:
    return {
        "train": evaluate_text_docs(predictions, truth, train_docs),
        "test": evaluate_text_docs(predictions, truth, test_docs),
    }
