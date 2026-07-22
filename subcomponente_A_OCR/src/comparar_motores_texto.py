# -*- coding: utf-8 -*-
"""
comparar_motores_texto.py
==========================
Subcomponente A — comparación cabeza a cabeza de motores de OCR para los
campos de TEXTO LIBRE de la cédula (TEXT_FIELDS): Tesseract (el que ya usa el
pipeline, ver text_trainer.py) vs. docTR (motor general sugerido por un
profesor como alternativa, ver doctr_engine.py).

LIMITACIÓN DESCUBIERTA (no introducida por este script, ya existía): el
ground truth manual (`data/ground_truth/campos_esperados.json`) NO tiene
etiquetas para los 5 campos de texto libre (nombre_informante, domicilio,
localidad, manzana, familia.vivienda) — solo cubre checkboxes, números y
`familia.rol_familiar`. Por eso `text_trainer.evaluate_text_docs` siempre ha
devuelto `accuracy=None` para estos campos en este proyecto, aun antes de
agregar docTR (se puede confirmar corriendo run_all.py: `text_split_metrics`
sale `null` en report.json). Etiquetar esos 5 campos a mano en los 10
documentos con ground truth es trabajo futuro razonable, pero no se hizo
aquí para no introducir transcripciones propias disfrazadas de "verdad".

Por eso esta comparación NO usa accuracy contra ground truth (no se puede
calcular honestamente hoy); en su lugar mide, sobre los MISMOS ROIs
(recortados por plantilla) de los 40 documentos disponibles:
  - Tasa de campos con salida no vacía (¿el motor "se atreve" a leer algo?).
  - Longitud promedio del texto reconocido.
  - Acuerdo palabra-a-palabra (Jaccard) entre lo que dice cada motor para el
    MISMO ROI — un acuerdo alto sugiere que ambos leen lo mismo; uno bajo
    indica que difieren y hay que revisar manualmente cuál acierta.
  - Latencia por campo.
  - Un puñado de ejemplos lado a lado para inspección cualitativa manual
    (la forma honesta de comparar calidad sin ground truth: leerlo uno mismo).

Requisito previo: correr `python src/run_all.py` al menos una vez para generar
los ROIs en data/processed/rois/.

Uso:  python src/comparar_motores_texto.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
ROOT_DIR = os.path.dirname(THIS_DIR)

from text_trainer import TEXT_FIELDS, ocr_text_from_roi  # noqa: E402


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _recolectar_rois(predictions: dict) -> list[tuple[str, str, str]]:
    """[(doc_id, field_id, ruta_roi), ...] para todos los TEXT_FIELDS que
    tengan un ROI real en disco (across los 40 documentos procesados)."""
    items = []
    for doc_id, doc in predictions.items():
        for field_id, pred_field in doc.get("fields", {}).items():
            if field_id not in TEXT_FIELDS or pred_field.get("type") != "text":
                continue
            roi = pred_field.get("roi")
            if roi and Path(roi).exists():
                items.append((doc_id, field_id, roi))
    return items


def _evaluar_motor(nombre: str, ocr_fn, rois: list[tuple[str, str, str]]) -> dict:
    salidas: dict[tuple[str, str], str] = {}
    tiempo_total = 0.0
    for doc_id, field_id, roi in rois:
        t0 = time.time()
        texto = ocr_fn(roi) or ""
        tiempo_total += time.time() - t0
        salidas[(doc_id, field_id)] = texto

    no_vacios = sum(1 for t in salidas.values() if t.strip())
    longitudes = [len(t) for t in salidas.values() if t.strip()]

    return {
        "motor": nombre,
        "n_rois": len(rois),
        "no_vacios": no_vacios,
        "tasa_no_vacio": round(no_vacios / len(rois), 4) if rois else None,
        "longitud_promedio": round(sum(longitudes) / len(longitudes), 1) if longitudes else None,
        "ms_por_campo": round(tiempo_total / len(rois) * 1000, 1) if rois else None,
        "salidas": salidas,
    }


def main() -> int:
    processed_dir = Path(ROOT_DIR) / "data" / "processed"
    predictions = json.loads((processed_dir / "predictions.json").read_text(encoding="utf-8"))
    rois = _recolectar_rois(predictions)

    print("=" * 78)
    print(f"Comparación de motores OCR para TEXT_FIELDS — {len(rois)} ROIs "
          f"({len(TEXT_FIELDS)} campos x hasta {len(predictions)} documentos)")
    print("=" * 78)

    res_tess = _evaluar_motor("Tesseract (actual)", ocr_text_from_roi, rois)

    doctr_disponible = True
    try:
        from doctr_engine import NOMBRE_MODELO, ocr_text_from_roi_doctr
        res_doctr = _evaluar_motor(NOMBRE_MODELO, ocr_text_from_roi_doctr, rois)
    except RuntimeError as exc:
        print(f"\n[AVISO] docTR no disponible en este entorno: {exc}")
        res_doctr = None
        doctr_disponible = False

    for res in (res_tess, res_doctr):
        if res is None:
            continue
        print(f"\n--- {res['motor']} ---")
        print(f"  ROIs procesados      : {res['n_rois']}")
        print(f"  Con texto (no vacío) : {res['no_vacios']} ({res['tasa_no_vacio']:.1%})")
        print(f"  Longitud promedio    : {res['longitud_promedio']} caracteres")
        print(f"  ms/campo             : {res['ms_por_campo']}")

    acuerdo = None
    if doctr_disponible:
        jaccards = [
            _jaccard(res_tess["salidas"][k], res_doctr["salidas"][k])
            for k in res_tess["salidas"]
        ]
        acuerdo = round(sum(jaccards) / len(jaccards), 4) if jaccards else None
        print(f"\nAcuerdo palabra-a-palabra (Jaccard) Tesseract vs docTR: {acuerdo}")
        print("(1.0 = leen exactamente las mismas palabras; 0.0 = no comparten ninguna)")

        print("\n--- Ejemplos lado a lado (primeros 8 ROIs con contenido) ---")
        mostrados = 0
        for (doc_id, field_id), t_tess in res_tess["salidas"].items():
            t_doctr = res_doctr["salidas"].get((doc_id, field_id), "")
            if not t_tess.strip() and not t_doctr.strip():
                continue
            print(f"  [{doc_id} / {field_id}]")
            print(f"    Tesseract: {t_tess!r}")
            print(f"    docTR    : {t_doctr!r}")
            mostrados += 1
            if mostrados >= 8:
                break

    salida = {
        "nota_limitacion": (
            "El ground truth no cubre estos 5 campos de texto libre; esta "
            "comparación mide acuerdo/latencia/tasa-no-vacío, no accuracy."
        ),
        "tesseract": {k: v for k, v in res_tess.items() if k != "salidas"},
        "doctr": ({k: v for k, v in res_doctr.items() if k != "salidas"} if res_doctr else None),
        "acuerdo_jaccard_promedio": acuerdo,
    }
    out_path = processed_dir / "comparacion_motores_texto.json"
    out_path.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Comparación guardada -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
