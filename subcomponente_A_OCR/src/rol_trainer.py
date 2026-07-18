"""
rol_trainer.py
==============
Clasificador local para el campo `familia.rol_familiar`.

El campo se trata como una palabra completa de catalogo cerrado. No usa OCR
preentrenado: aprende desde los ROIs etiquetados manualmente en ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


TRAIN_DOCS = [f"Cédula_{i:04d}" for i in range(1, 9)]
TEST_DOCS = ["Cédula_0009", "Cédula_0010"]
ROL_FIELD = "familia.rol_familiar"
ROL_CATALOG = ["hijo", "hija", "abuela", "yerno", "madre", "padre"]


@dataclass
class TrainedRolModel:
    clf: KNeighborsClassifier
    train_docs: list[str]
    test_docs: list[str]
    n_train: int
    trained_labels: list[str]
    untrained_labels: list[str]


def _imread_gray(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "hijo(a)": "hijo",
        "abuelo": "abuela",
        "abuelo(a)": "abuela",
    }
    return aliases.get(text, text)


def rol_vector_from_roi(path: str | Path, height: int = 48, width: int = 160) -> np.ndarray | None:
    """ROI gris -> vector binario normalizado de la palabra manuscrita."""
    gray = _imread_gray(path)
    if gray is None or gray.size == 0:
        return None

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = binary.shape

    # El ROI contiene instrucciones impresas arriba; la respuesta manuscrita
    # queda debajo de esa franja en las cedulas actuales.
    y1 = int(h * 0.46)
    y2 = int(h * 0.92)
    x1 = int(w * 0.02)
    x2 = int(w * 0.95)
    work = binary[y1:y2, x1:x2]
    if work.size == 0:
        work = binary

    wh, ww = work.shape
    if wh >= 8 and ww >= 8:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, ww // 2), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, wh // 2)))
        horizontal = cv2.morphologyEx(work, cv2.MORPH_OPEN, h_kernel)
        vertical = cv2.morphologyEx(work, cv2.MORPH_OPEN, v_kernel)
        work = cv2.bitwise_and(work, cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)))

    work = cv2.morphologyEx(
        work,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    keep = np.zeros_like(work)
    min_area = max(3, int(work.size * 0.0008))
    for label in range(1, n_labels):
        x, y, cw, ch, area = stats[label]
        if area < min_area:
            continue
        if ch < work.shape[0] * 0.08:
            continue
        if cw > work.shape[1] * 0.85 and ch < work.shape[0] * 0.18:
            continue
        keep[labels == label] = 255

    ys, xs = np.where(keep > 0)
    if len(xs) == 0:
        return np.zeros(height * width, dtype=np.float32)

    crop = keep[
        max(0, ys.min() - 3) : min(keep.shape[0], ys.max() + 4),
        max(0, xs.min() - 3) : min(keep.shape[1], xs.max() + 4),
    ]
    ch, cw = crop.shape
    scale = min((height - 8) / float(ch or 1), (width - 8) / float(cw or 1))
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = 4
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return (canvas.astype(np.float32) / 255.0).reshape(-1)


def _labeled_rows(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    doc_ids: list[str],
) -> tuple[list[np.ndarray], list[str], list[tuple[str, str]]]:
    rows: list[np.ndarray] = []
    y: list[str] = []
    keys: list[tuple[str, str]] = []
    for doc_id in doc_ids:
        expected = _normalize_label(truth.get(doc_id, {}).get(ROL_FIELD))
        if expected not in ROL_CATALOG:
            continue
        pred_field = predictions.get(doc_id, {}).get("fields", {}).get(ROL_FIELD)
        if not pred_field or pred_field.get("type") != "text":
            continue
        roi = pred_field.get("roi")
        if not roi:
            continue
        vec = rol_vector_from_roi(roi)
        if vec is None:
            continue
        rows.append(vec)
        y.append(expected)
        keys.append((doc_id, ROL_FIELD))
    return rows, y, keys


def train_rol_model(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    train_docs: list[str] | None = None,
    test_docs: list[str] | None = None,
) -> TrainedRolModel | None:
    train_docs = train_docs or TRAIN_DOCS
    test_docs = test_docs or TEST_DOCS
    rows, y, _ = _labeled_rows(predictions, truth, train_docs)
    if len(rows) < 2 or len(set(y)) < 2:
        return None
    clf = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    clf.fit(np.vstack(rows), y)
    trained_labels = sorted(set(y))
    untrained_labels = [label for label in ROL_CATALOG if label not in trained_labels]
    return TrainedRolModel(
        clf=clf,
        train_docs=train_docs,
        test_docs=test_docs,
        n_train=len(y),
        trained_labels=trained_labels,
        untrained_labels=untrained_labels,
    )


def _confidence_from_distance(distance: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    return round(float(1.0 / (1.0 + distance)), 4)


def apply_rol_model(predictions: dict[str, Any], model: TrainedRolModel) -> None:
    for doc in predictions.values():
        pred_field = doc.get("fields", {}).get(ROL_FIELD)
        if not pred_field or pred_field.get("type") != "text":
            continue
        roi = pred_field.get("roi")
        if not roi:
            continue
        vec = rol_vector_from_roi(roi)
        if vec is None:
            continue
        pred = str(model.clf.predict([vec])[0])
        distance = float(model.clf.kneighbors([vec], n_neighbors=1, return_distance=True)[0][0][0])
        pred_field["value"] = pred
        pred_field["needs_review"] = False
        pred_field["confidence"] = _confidence_from_distance(distance)
        pred_field["model"] = "KNeighborsClassifier(k=1, cosine) palabra completa"
        pred_field["catalog"] = ROL_CATALOG


def evaluate_rol_docs(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    doc_ids: list[str],
) -> dict[str, Any]:
    total = 0
    correct = 0
    errors = []
    for doc_id in doc_ids:
        expected = _normalize_label(truth.get(doc_id, {}).get(ROL_FIELD))
        if expected not in ROL_CATALOG:
            continue
        pred_field = predictions.get(doc_id, {}).get("fields", {}).get(ROL_FIELD)
        if not pred_field or pred_field.get("type") != "text":
            continue
        total += 1
        got = _normalize_label(pred_field.get("value"))
        if got == expected:
            correct += 1
        else:
            errors.append({"doc_id": doc_id, "field": ROL_FIELD, "expected": expected, "got": got})
    return {
        "docs": doc_ids,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "errors": errors,
    }


def evaluate_rol_split(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    train_docs: list[str],
    test_docs: list[str],
) -> dict[str, Any]:
    return {
        "train": evaluate_rol_docs(predictions, truth, train_docs),
        "test": evaluate_rol_docs(predictions, truth, test_docs),
    }
