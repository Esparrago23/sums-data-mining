"""
field_extractor.py
==================
Extraccion por plantilla para la Cedula de Microdiagnostico Familiar.

El pipeline no intenta leer toda la pagina. Recorta campos esperados y aplica el
modelo adecuado por tipo: checkbox, numero o texto. Para texto manuscrito el MVP
guarda ROI/metadata y marca el campo como `needs_review`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from checkbox_model import answer_zone, predict_checkbox
from horizontal_sheet_processor import extract_catalog_value_from_roi
from preprocessor import PageImage, crop_relative


def load_field_map(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ink_summary(binary_roi: np.ndarray) -> dict[str, float]:
    if binary_roi.size == 0:
        return {"ink_ratio": 0.0}
    return {"ink_ratio": round(float(np.count_nonzero(binary_roi)) / float(binary_roi.size), 4)}


def _imwrite_unicode(path: Path, imagen: np.ndarray) -> None:
    """Escribe un PNG en `path` sin pasar por cv2.imwrite(str(path), ...).

    cv2.imwrite falla EN SILENCIO en Windows para rutas con caracteres fuera
    del codepage ANSI activo del proceso (ej. "Cédula", con acento) -- no
    levanta excepción, simplemente no escribe el archivo, así que el bug
    pasaba desapercibido hasta que algo más adelante intentaba LEER ese ROI
    inexistente. Se evita codificando el PNG en memoria (cv2.imencode, que sí
    funciona bien porque no toca el sistema de archivos) y escribiendo los
    bytes con pathlib (usa las APIs de Windows de ancho completo / UTF-16,
    sin ese límite) -- mismo truco que ya usa `_imread_gray` en
    text_trainer.py/number_trainer.py para LEER ROIs con nombres acentuados.
    """
    ok, buffer = cv2.imencode(".png", imagen)
    if not ok:
        raise IOError(f"cv2.imencode no pudo codificar la imagen para {path}")
    path.write_bytes(buffer.tobytes())


def _save_roi(gray_roi: np.ndarray, out_dir: Path, field_id: str) -> str:
    _ensure_dir(out_dir)
    safe = field_id.replace(".", "__").replace("[", "_").replace("]", "")
    out = out_dir / f"{safe}.png"
    _imwrite_unicode(out, gray_roi)
    return str(out)


def _save_binary_roi(binary_roi: np.ndarray, out_dir: Path, field_id: str) -> str:
    _ensure_dir(out_dir)
    safe = field_id.replace(".", "__").replace("[", "_").replace("]", "")
    out = out_dir / f"{safe}.png"
    # Invertimos para inspeccion humana: tinta negra sobre fondo blanco.
    _imwrite_unicode(out, 255 - binary_roi)
    return str(out)


def extract_page(
    page: PageImage,
    fields: list[dict[str, Any]],
    roi_dir: Path,
    *,
    page_source: str | None = None,
) -> dict[str, Any]:
    """Extrae todos los campos aplicables a una pagina normalizada.

    Args:
        page:        Imagen normalizada de la pagina.
        fields:      Lista de campos de la plantilla para esta pagina.
        roi_dir:     Directorio donde se guardan los recortes ROI.
        page_source: Path absoluto al PNG de la pagina completa. Se embebe
                     en los campos de texto para que ``apply_paddle_text``
                     pueda cargar la imagen sin buscar en el meta de paginas.
    """
    out: dict[str, Any] = {}
    for field in fields:
        field_id = field["id"]
        kind = field["type"]
        rel = field["bbox"]
        gray_roi = crop_relative(page.gray, page.form_box, rel)
        bin_roi = crop_relative(page.binary, page.form_box, rel)

        if kind == "checkbox":
            pred = predict_checkbox(bin_roi)
            zone_path = _save_binary_roi(
                answer_zone(bin_roi),
                roi_dir / "_checkbox_answer_zones",
                field_id,
            )
            out[field_id] = {
                "type": kind,
                "value": pred.marked,
                "confidence": min(1.0, round(abs(pred.score - 0.42) + 0.5, 3)),
                "answer_zone_roi": zone_path,
                "features": {
                    "score": pred.score,
                    "ink_ratio": pred.ink_ratio,
                    "component_count": pred.component_count,
                    "diagonal_ratio": pred.diagonal_ratio,
                },
            }
        elif kind == "catalog":
            roi_path = _save_roi(gray_roi, roi_dir, field_id)
            catalog_field = field.get("catalog", field_id.split(".")[-1])
            catalog_result = extract_catalog_value_from_roi(gray_roi, catalog_field)
            out[field_id] = {
                "type": kind,
                "value": catalog_result.get("value"),
                "confidence": 0.8 if catalog_result.get("matched") else 0.2,
                "needs_review": not catalog_result.get("matched", False),
                "roi": roi_path,
                "catalog": catalog_field,
                "raw_text": catalog_result.get("raw_text"),
                "features": _ink_summary(bin_roi),
            }
        elif kind in {"number", "date", "text"}:
            roi_path = _save_roi(gray_roi, roi_dir, field_id)
            field_entry: dict[str, Any] = {
                "type": kind,
                "value": None,
                "confidence": 0.0,
                "needs_review": True,
                "roi": roi_path,
                "features": _ink_summary(bin_roi),
            }
            # Embeber referencia a la imagen completa y form_box para que
            # apply_paddle_text pueda cargar la pagina sin iterar metadatos.
            if kind == "text" and page_source is not None:
                field_entry["_page_source"] = page_source
                field_entry["_form_box"] = list(page.form_box)
            out[field_id] = field_entry
        else:
            out[field_id] = {
                "type": kind,
                "value": None,
                "confidence": 0.0,
                "needs_review": True,
                "features": _ink_summary(bin_roi),
            }
    return out


def extract_document(
    doc_id: str,
    pages: list[PageImage],
    field_map: dict[str, Any],
    out_root: str | Path,
) -> dict[str, Any]:
    """Extrae campos de un documento completo."""
    out_root = Path(out_root)
    roi_dir = out_root / "rois" / doc_id
    document: dict[str, Any] = {"doc_id": doc_id, "pages": {}, "fields": {}}
    datos_candidates = [
        (i, page)
        for i, page in enumerate(pages, start=1)
        if page.page_kind == "datos_vivienda"
    ]
    datos_page_index = None
    if datos_candidates:
        def score(item: tuple[int, PageImage]) -> float:
            _, page = item
            _, _, w, h = page.form_box
            aspect = h / float(w or 1)
            # La pagina de vivienda real es la candidata cuya caja se parece mas
            # a la plantilla canónica; evita que paginas laterales sobrescriban.
            return -abs(aspect - 1.25)

        datos_page_index = max(datos_candidates, key=score)[0]

    for page_index, page in enumerate(pages, start=1):
        fields = field_map.get("page_kinds", {}).get(page.page_kind)
        if fields is None and page.page_kind == "datos_vivienda":
            fields = field_map.get("pages", {}).get("1", []) if page_index == datos_page_index else []
        if fields is None:
            fields = field_map.get("pages", {}).get(str(page_index), [])
        page_result = extract_page(
            page,
            fields,
            roi_dir / f"page_{page_index}",
            page_source=str(page.source),
        )
        document["pages"][str(page_index)] = {
            "source": str(page.source),
            "kind": page.page_kind,
            "form_box": list(page.form_box),
            "n_fields": len(page_result),
        }
        document["fields"].update(page_result)

    return document
