# -*- coding: utf-8 -*-
"""
doctr_engine.py
===============
Subcomponente A — motor de texto ALTERNATIVO con docTR (Mindee): un modelo de
OCR general (detección + reconocimiento entrenado en documentos reales), a
diferencia del pipeline específico de este proyecto (Tesseract + preprocesado
OpenCV a medida, ver text_trainer.py). Sugerido por un profesor como una de 3
alternativas a evaluar (docTR / PaddleOCR / DeepSeek-OCR).

Por qué docTR y no PaddleOCR/DeepSeek-OCR:
  - Reutiliza PyTorch, que YA es dependencia del proyecto (lo instaló
    sentence-transformers para el buscador semántico de C) — no agrega un
    SEGUNDO framework de deep learning (PaddleOCR necesita paddlepaddle aparte,
    un runtime completo distinto de torch).
  - Licencia Apache 2.0, arquitecturas "mobile" livianas (db_mobilenet_v3_large
    + crnn_mobilenet_v3_small) mucho más chicas que DeepSeek-OCR (un VLM
    multimodal de varios GB pensado para GPU) — factible en un disco con
    margen ajustado (~18GB libres en esta máquina).

ALCANCE INTENCIONALMENTE LIMITADO: docTR es un OCR de texto general, no
entiende "campo de formulario" ni detecta checkboxes — por eso NO reemplaza
checkbox_model.py (el corazón del pipeline: 37 de los 45 campos de la cédula).
Se usa ÚNICAMENTE como motor alternativo para los campos de TEXTO LIBRE
(TEXT_FIELDS, ver text_trainer.py), donde sí compite de tú a tú con Tesseract:
ambos reciben la MISMA imagen ROI recortada por plantilla (field_extractor.py)
y devuelven texto — head-to-head justo.

`apply_doctr_text(predictions)` tiene la MISMA firma/contrato que
`text_trainer.apply_ocr_text(predictions)`: se puede llamar sobre una copia de
`predictions` para comparar cabeza a cabeza contra Tesseract (ver run_all.py,
sección de comparación de motores de texto).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from text_trainer import TEXT_FIELDS, _clean_ocr_text, _imread_gray

# ---------------------------------------------------------------------------
# Import opcional + carga perezosa del predictor (una sola vez por proceso):
# si `python-doctr` no está instalado, el módulo se importa igual (no crashea
# el resto del pipeline OCR); solo falla al intentar USAR este motor.
# ---------------------------------------------------------------------------
try:
    from doctr.models import ocr_predictor
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depende de la máquina
    ocr_predictor = None
    _IMPORT_ERROR = exc

DET_ARCH = "db_mobilenet_v3_large"
RECO_ARCH = "crnn_mobilenet_v3_small"
NOMBRE_MODELO = f"docTR ({DET_ARCH} + {RECO_ARCH})"

_predictor = None


def _get_predictor():
    """Instancia (o reutiliza) el predictor docTR. Descarga los pesos
    pre-entrenados la primera vez (cache local de torch.hub después)."""
    global _predictor
    if ocr_predictor is None:
        raise RuntimeError(
            "doctr_engine no disponible: falta instalar 'python-doctr' "
            '(pip install "python-doctr[torch]"). '
            f"Error original de import: {_IMPORT_ERROR!r}"
        )
    if _predictor is None:
        _predictor = ocr_predictor(
            det_arch=DET_ARCH,
            reco_arch=RECO_ARCH,
            pretrained=True,
        )
    return _predictor


def ocr_text_from_roi_doctr(path: str | Path) -> str | None:
    """Ejecuta docTR sobre el ROI (mismo recorte que usa Tesseract vía
    text_trainer.ocr_text_from_roi) y devuelve el texto reconocido
    concatenado en orden de lectura, o None si no detectó nada."""
    gray = _imread_gray(path)
    if gray is None or gray.size == 0:
        return None

    # docTR espera imágenes RGB (H, W, 3) uint8; el ROI viene en escala de grises.
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    predictor = _get_predictor()
    result = predictor([rgb])

    palabras: list[str] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    palabras.append(word.value)

    texto = _clean_ocr_text(" ".join(palabras))
    return texto or None


def apply_doctr_text(predictions: dict[str, Any]) -> None:
    """Igual que text_trainer.apply_ocr_text, pero usando docTR en vez de
    Tesseract. Mismo contrato: muta `predictions` in-place, mismos campos
    (TEXT_FIELDS), mismo formato de salida (`value`/`confidence`/`model`) —
    así se puede evaluar con evaluate_text_docs exactamente igual que la ruta
    de Tesseract."""
    for doc in predictions.values():
        for field_id, pred_field in doc.get("fields", {}).items():
            if field_id not in TEXT_FIELDS or pred_field.get("type") != "text":
                continue
            roi = pred_field.get("roi")
            if not roi:
                continue
            texto = ocr_text_from_roi_doctr(roi)
            if texto is None:
                continue
            pred_field["value"] = texto
            pred_field["needs_review"] = False
            pred_field["confidence"] = 0.65
            pred_field["model"] = NOMBRE_MODELO
