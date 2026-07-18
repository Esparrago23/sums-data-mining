"""
checkbox_model.py
=================
Clasificador propio para celdas marcadas/no marcadas.

No usa modelos OCR preentrenados. Extrae features visuales simples de cada ROI y
aplica una decision interpretable. Si despues se agrega ground truth, las mismas
features pueden calibrarse con un umbral aprendido localmente.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CheckboxPrediction:
    marked: bool
    score: float
    ink_ratio: float
    component_count: int
    diagonal_ratio: float


def answer_zone(binary_roi: np.ndarray) -> np.ndarray:
    """Reduce una celda de formulario a la zona donde aparece la respuesta.

    Las cajas del field_map suelen cubrir la celda completa para tolerar
    desplazamientos del escaneo. La parte superior contiene texto impreso
    ("Si", "No", "Concreto", etc.), que no debe contar como marca manuscrita.
    """
    if binary_roi.size == 0:
        return binary_roi

    h, w = binary_roi.shape
    # En esta cedula, las etiquetas impresas estan usualmente arriba y la marca
    # manuscrita queda en el centro-bajo de la celda.
    y1 = int(h * 0.38)
    y2 = int(h * 0.95)
    x1 = int(w * 0.08)
    x2 = int(w * 0.92)
    zone = binary_roi[y1:y2, x1:x2].copy()
    if zone.size == 0:
        zone = binary_roi.copy()

    # Quitar lineas largas de tabla para que bordes de celda no parezcan tinta.
    zh, zw = zone.shape
    if zh >= 8 and zw >= 8:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, zw // 2), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, zh // 2)))
        horizontal = cv2.morphologyEx(zone, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(zone, cv2.MORPH_OPEN, vertical_kernel)
        grid = cv2.bitwise_or(horizontal, vertical)
        zone = cv2.bitwise_and(zone, cv2.bitwise_not(grid))

    return zone


def checkbox_features(binary_roi: np.ndarray) -> dict[str, float]:
    """Calcula features de una celda binaria con tinta=255."""
    if binary_roi.size == 0:
        return {"ink_ratio": 0.0, "component_count": 0.0, "diagonal_ratio": 0.0}

    roi = answer_zone(binary_roi)
    h, w = roi.shape
    margin_y = max(1, int(h * 0.08))
    margin_x = max(1, int(w * 0.08))
    inner = roi[margin_y : h - margin_y, margin_x : w - margin_x]
    if inner.size == 0:
        inner = roi

    ink_ratio = float(np.count_nonzero(inner)) / float(inner.size)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)
    component_count = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max(3, inner.size * 0.002):
            component_count += 1

    edges = cv2.Canny(inner, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(5, min(inner.shape) // 5),
        minLineLength=max(5, min(inner.shape) // 3),
        maxLineGap=3,
    )
    diagonal = 0
    total = 0
    if lines is not None:
        lines = np.asarray(lines).reshape(-1, 4)
        for x1, y1, x2, y2 in lines:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            angle = min(angle, 180 - angle)
            total += 1
            if 20 <= angle <= 70:
                diagonal += 1
    diagonal_ratio = float(diagonal) / float(total or 1)

    return {
        "ink_ratio": ink_ratio,
        "component_count": float(component_count),
        "diagonal_ratio": diagonal_ratio,
    }


def predict_checkbox(binary_roi: np.ndarray) -> CheckboxPrediction:
    """Predice si una celda contiene marca manuscrita tipo X/palomita."""
    f = checkbox_features(binary_roi)
    score = (
        2.8 * f["ink_ratio"]
        + 0.35 * min(f["component_count"], 3.0)
        + 0.45 * f["diagonal_ratio"]
    )
    marked = score >= 0.42 and f["ink_ratio"] >= 0.015
    return CheckboxPrediction(
        marked=bool(marked),
        score=float(round(score, 4)),
        ink_ratio=float(round(f["ink_ratio"], 4)),
        component_count=int(f["component_count"]),
        diagonal_ratio=float(round(f["diagonal_ratio"], 4)),
    )
