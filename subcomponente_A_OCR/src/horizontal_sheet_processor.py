from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import pytesseract


CATALOGS = {
    "escolaridad": [
        "sin escolaridad",
        "primaria",
        "primaria truncada",
        "secundaria",
        "secundaria truncada",
        "preparatoria",
        "preparatoria truncada",
        "bachillerato",
        "bachillerato truncado",
        "licenciatura",
        "licenciatura truncada",
    ],
    "seguridad_social": ["si", "no"],
    "frecuencia_servicios": ["anual", "mensual"],
}


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("\n", " ").replace("\r", " ").split())


def match_catalog_option(text: str, catalog: Iterable[str]) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    candidates = []
    for option in catalog:
        option_norm = _normalize_text(option)
        if option_norm == normalized:
            return option
        if normalized.startswith(option_norm):
            candidates.append(option)
        if normalized.endswith(option_norm):
            candidates.append(option)
        if option_norm in normalized:
            candidates.append(option)
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=len)
    return candidates[0]


def detect_orientation(image: np.ndarray, box: tuple[int, int, int, int]) -> int:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return 0
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return 0
    upper = crop[: max(1, h // 2), :]
    lower = crop[max(1, h // 2) :, :]
    upper_score = float(np.count_nonzero(upper))
    lower_score = float(np.count_nonzero(lower))
    return 0 if upper_score >= lower_score else 180


def classify_horizontal_page(image: np.ndarray, box: tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return "desconocida"
    aspect = float(w) / max(1.0, float(h))
    if aspect >= 1.35:
        return "horizontal"
    return "desconocida"


def extract_catalog_value_from_roi(roi: np.ndarray, field: str) -> dict[str, object]:
    if field not in CATALOGS:
        return {"value": None, "matched": False, "raw_text": ""}
    if roi.size == 0:
        return {"value": None, "matched": False, "raw_text": ""}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    if gray.size == 0:
        return {"value": None, "matched": False, "raw_text": ""}

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary, lang="eng", config="--oem 1 --psm 6")
    cleaned = _normalize_text(text)
    matched = match_catalog_option(cleaned, CATALOGS[field])
    return {"value": matched, "matched": matched is not None, "raw_text": cleaned}
