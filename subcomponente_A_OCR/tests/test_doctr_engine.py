# -*- coding: utf-8 -*-
"""Pruebas de doctr_engine.py: degradación con gracia + contrato de apply_doctr_text."""
import os
import sys

import numpy as np
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

doctr = pytest.importorskip("doctr", reason="python-doctr no instalado en este entorno")

import cv2  # noqa: E402

import doctr_engine  # noqa: E402
from text_trainer import TEXT_FIELDS  # noqa: E402


def _crear_roi_con_texto(tmp_path, texto="Juan Perez"):
    img = np.full((60, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    ruta = tmp_path / "roi_test.png"
    cv2.imwrite(str(ruta), img)
    return str(ruta)


def test_ocr_text_from_roi_doctr_reconoce_algo(tmp_path):
    ruta = _crear_roi_con_texto(tmp_path)
    texto = doctr_engine.ocr_text_from_roi_doctr(ruta)
    assert texto is not None
    assert len(texto.strip()) > 0


def test_ocr_text_from_roi_doctr_roi_inexistente_lanza_filenotfound(tmp_path):
    # Mismo comportamiento que text_trainer.ocr_text_from_roi: ambos reusan
    # _imread_gray (np.fromfile), que lanza FileNotFoundError para una ruta
    # que no existe -- no lo atrapan ni devuelven None. Comportamiento
    # preexistente y consistente entre los dos motores, no algo nuevo de
    # doctr_engine.
    ruta_falsa = str(tmp_path / "no_existe.png")
    with pytest.raises(FileNotFoundError):
        doctr_engine.ocr_text_from_roi_doctr(ruta_falsa)


def test_apply_doctr_text_respeta_el_contrato_de_apply_ocr_text(tmp_path):
    """Mismo contrato que text_trainer.apply_ocr_text: muta in-place, mismos
    campos (TEXT_FIELDS), mismas llaves de salida (value/needs_review/
    confidence/model)."""
    field_id = next(iter(TEXT_FIELDS))
    roi_path = _crear_roi_con_texto(tmp_path, "Domicilio Prueba")

    predictions = {
        "doc_test": {
            "fields": {
                field_id: {"type": "text", "value": None, "needs_review": True, "roi": roi_path},
                "otro.campo": {"type": "checkbox", "value": True},  # no debe tocarse
            }
        }
    }

    doctr_engine.apply_doctr_text(predictions)

    campo = predictions["doc_test"]["fields"][field_id]
    assert campo["value"] is not None
    assert campo["needs_review"] is False
    assert 0.0 <= campo["confidence"] <= 1.0
    assert campo["model"] == doctr_engine.NOMBRE_MODELO

    # El campo de otro tipo (checkbox) no debe modificarse.
    assert predictions["doc_test"]["fields"]["otro.campo"] == {"type": "checkbox", "value": True}


def test_get_predictor_es_singleton():
    p1 = doctr_engine._get_predictor()
    p2 = doctr_engine._get_predictor()
    assert p1 is p2
