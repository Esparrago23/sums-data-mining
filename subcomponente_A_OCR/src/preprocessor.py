"""
preprocessor.py
===============
Subcomponente A - normalizacion visual de cedulas escaneadas.

Objetivo: convertir paginas escaneadas con margen, ruido e inclinacion leve en
imagenes binarias y recortes alineados por la caja principal del formulario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PageImage:
    source: Path
    gray: np.ndarray
    binary: np.ndarray
    form_box: tuple[int, int, int, int]
    page_kind: str


def load_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")
    return img


def binarize(gray: np.ndarray) -> np.ndarray:
    """Binariza con Otsu. Devuelve tinta=255, fondo=0."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return inv


def estimate_form_box(binary: np.ndarray) -> tuple[int, int, int, int]:
    """Caja aproximada del formulario ignorando bordes negros de escaneo."""
    h, w = binary.shape
    work = binary.copy()
    # Los escaneos Canon traen con frecuencia franjas negras pegadas al borde.
    # Si no se eliminan, el detector cree que toda la hoja es el formulario.
    mx = max(4, int(w * 0.04))
    my = max(4, int(h * 0.04))
    work[:my, :] = 0
    work[h - my :, :] = 0
    work[:, :mx] = 0
    work[:, w - mx :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    ink = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 0.01 * w * h:
            continue
        if bw < 0.25 * w or bh < 0.20 * h:
            continue
        boxes.append((x, y, bw, bh, area))

    if not boxes:
        ys, xs = np.where(work > 0)
        if len(xs) == 0:
            return (0, 0, w, h)
        return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))

    x, y, bw, bh, _ = max(boxes, key=lambda b: b[4])
    pad = int(min(w, h) * 0.01)
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)
    return (x, y, bw, bh)


def classify_page(page_num: int, box: tuple[int, int, int, int]) -> str:
    """Clasificacion simple por posicion en el paquete."""
    if page_num == 1:
        return "datos_vivienda"
    if page_num == 2:
        return "vacunacion_a"
    if page_num == 3:
        return "familia_a"
    if page_num == 4:
        return "familia_b"
    return "desconocida"


def normalize_page(image_path: str | Path, page_num: int) -> PageImage:
    """Carga, binariza y calcula caja del formulario."""
    source = Path(image_path)
    gray = load_gray(source)
    binary = binarize(gray)
    box = estimate_form_box(binary)
    return PageImage(
        source=source,
        gray=gray,
        binary=binary,
        form_box=box,
        page_kind=classify_page(page_num, box),
    )


def crop_relative(img: np.ndarray, box: tuple[int, int, int, int], rel: list[float]) -> np.ndarray:
    """Recorta una region relativa a la caja del formulario.

    `rel` = [x1, y1, x2, y2] en coordenadas normalizadas 0..1.
    """
    x, y, w, h = box
    x1 = int(x + rel[0] * w)
    y1 = int(y + rel[1] * h)
    x2 = int(x + rel[2] * w)
    y2 = int(y + rel[3] * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    return img[y1:y2, x1:x2]
