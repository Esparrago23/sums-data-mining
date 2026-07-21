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
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
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


def _line_density(binary: np.ndarray, min_len_frac: float) -> tuple[float, float]:
    """Densidad de lineas largas horizontales/verticales en una imagen binaria."""
    h, w = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(w * min_len_frac)), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, int(h * min_len_frac))))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    area = float(binary.size or 1)
    return float(np.count_nonzero(horizontal)) / area, float(np.count_nonzero(vertical)) / area


def classify_page(binary: np.ndarray, box: tuple[int, int, int, int]) -> str:
    """Clasifica la pagina por estructura visual, no por orden en el PDF.

    El objetivo inicial es robustecer la extraccion de la pagina de vivienda:
    si las hojas se escanean/desordenan, seguimos buscando la pagina que contiene
    identificacion familiar + caracteristicas de vivienda + esquema de vacunacion
    al pie.
    """
    x, y, w, h = box
    page_h, page_w = binary.shape
    crop = binary[y : y + h, x : x + w]
    if crop.size == 0:
        return "desconocida"

    form_height_ratio = h / float(page_h or 1)
    form_width_ratio = w / float(page_w or 1)
    aspect = h / float(w or 1)
    horizontal_density, vertical_density = _line_density(crop, 0.35)
    top = crop[: max(1, int(crop.shape[0] * 0.32)), :]
    mid = crop[int(crop.shape[0] * 0.32) : int(crop.shape[0] * 0.76), :]
    bottom = crop[int(crop.shape[0] * 0.76) :, :]
    top_ink = np.count_nonzero(top) / float(top.size or 1)
    mid_ink = np.count_nonzero(mid) / float(mid.size or 1)
    bottom_ink = np.count_nonzero(bottom) / float(bottom.size or 1)

    # Paginas de vacunacion sueltas: tabla baja/centrada y caja mas corta.
    if form_height_ratio < 0.72 and aspect < 1.35:
        return "vacunacion"

    # Hojas horizontales de composicion familiar o servicios de salud.
    if aspect >= 1.35 and form_height_ratio < 0.82:
        return "horizontal"

    # Paginas de composicion familiar escaneadas de lado: caja mas alta/estrecha.
    # En las muestras reales la pagina de vivienda ronda aspect~1.25, mientras
    # las tablas laterales de familia suelen rondar 1.35-1.45.
    if aspect > 1.32 and form_height_ratio > 0.75:
        return "familia"
    if aspect > 1.25 and vertical_density > horizontal_density * 1.8:
        return "familia"

    # Pagina de datos/vivienda: caja amplia, alta, con secciones cargadas arriba
    # y a media pagina; ademas suele traer una tabla de vacunacion al pie.
    vivienda_score = 0
    vivienda_score += 1 if form_height_ratio >= 0.72 else 0
    vivienda_score += 1 if form_width_ratio >= 0.70 else 0
    vivienda_score += 1 if top_ink > 0.045 else 0
    vivienda_score += 1 if mid_ink > 0.035 else 0
    vivienda_score += 1 if bottom_ink > 0.018 else 0
    if vivienda_score >= 4:
        return "datos_vivienda"

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
        page_kind=classify_page(binary, box),
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
