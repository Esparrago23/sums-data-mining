"""
fuzzy_mapper.py
===============
Subcomponente A — Mapeo de bloques OCR a campos de plantilla.

Estrategia de asociación en dos pasos:
1. **Intersección de coordenadas**: filtra sólo los bloques OCR cuyo bounding
   box se solapa con la región absoluta del campo en la imagen.
2. **Fuzzy matching**: si el campo tiene catálogo de valores válidos, aplica
   ``rapidfuzz`` para corregir variaciones ortográficas del texto extraído.

No depende de ningún módulo fuera de ``subcomponente_A_OCR/``.

Dependencias:
    rapidfuzz>=3.6.1   (sólo subcomponente_A_OCR/requirements.txt)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from paddle_extractor import OcrBlock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos públicos
# ---------------------------------------------------------------------------


@dataclass
class FieldTemplate:
    """Descriptor de un campo de la plantilla de documento.

    Attributes:
        field_id:    Identificador único del campo (ej. ``"familia.domicilio"``).
        bbox_rel:    Coordenadas relativas al ``form_box``: (x1, y1, x2, y2)
                     en rango [0, 1].
        catalog:     Lista de valores válidos. Si es ``None`` el texto se devuelve
                     tal cual.
        field_type:  ``"text"``, ``"catalog"``, ``"number"``, ``"checkbox"``.
    """

    field_id: str
    bbox_rel: tuple[float, float, float, float]
    catalog: list[str] | None = None
    field_type: str = "text"


@dataclass
class MappedField:
    """Resultado del mapeo para un campo concreto.

    Attributes:
        field_id:       Identificador del campo.
        raw_text:       Texto concatenado de todos los bloques solapantes.
        matched_value:  Valor tras fuzzy matching (o ``raw_text`` si sin catálogo).
        confidence:     Score de confianza en [0.0, 1.0].
        fuzzy_score:    Score de rapidfuzz (0–100). ``None`` si no hubo matching.
        source_blocks:  Bloques OCR que contribuyeron al campo.
        needs_review:   ``True`` cuando la confianza cae por debajo del umbral.
    """

    field_id: str
    raw_text: str
    matched_value: str | None
    confidence: float
    fuzzy_score: int | None = None
    source_blocks: list[OcrBlock] = field(default_factory=list)
    needs_review: bool = False


# ---------------------------------------------------------------------------
# Helpers de geometría
# ---------------------------------------------------------------------------


def _rel_to_abs(
    bbox_rel: tuple[float, float, float, float],
    form_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Convierte coordenadas relativas al form_box en píxeles absolutos.

    Args:
        bbox_rel:  (x1_rel, y1_rel, x2_rel, y2_rel) en [0, 1].
        form_box:  (x, y, w, h) del formulario en la imagen completa.

    Returns:
        (ax1, ay1, ax2, ay2) en píxeles absolutos.
    """
    fx, fy, fw, fh = form_box
    x1 = int(fx + bbox_rel[0] * fw)
    y1 = int(fy + bbox_rel[1] * fh)
    x2 = int(fx + bbox_rel[2] * fw)
    y2 = int(fy + bbox_rel[3] * fh)
    return (x1, y1, x2, y2)


def _intersection_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    """Área de intersección entre dos rectángulos.

    Rectángulo formato: (x1, y1, x2, y2).
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _block_rect(block: OcrBlock) -> tuple[int, int, int, int]:
    """Convierte OcrBlock.bbox (x, y, w, h) a (x1, y1, x2, y2)."""
    x, y, w, h = block.bbox
    return (x, y, x + w, y + h)


def _field_rect(
    template: FieldTemplate,
    form_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Devuelve el rectángulo absoluto del campo."""
    return _rel_to_abs(template.bbox_rel, form_box)


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Normaliza texto: minúsculas y colapsa espacios."""
    return " ".join(text.lower().split())


def _fuzzy_match(
    text: str,
    catalog: list[str],
    threshold: int = 70,
) -> tuple[str | None, int]:
    """Devuelve la mejor coincidencia del catálogo y su score.

    Usa ``rapidfuzz.process.extractOne`` con la métrica WRatio (robusta a
    transposiciones, omisiones y sustituciones frecuentes en OCR).

    Args:
        text:      Texto a corregir.
        catalog:   Valores válidos del catálogo.
        threshold: Score mínimo aceptable (0–100).

    Returns:
        (valor_matched | None, score)
    """
    if not text or not catalog:
        return None, 0

    try:
        from rapidfuzz import process as rf_process, fuzz  # type: ignore[import]
    except ImportError:
        logger.warning("[fuzzy_mapper] rapidfuzz no instalado; comparación exacta.")
        norm = _normalize(text)
        for option in catalog:
            if _normalize(option) == norm:
                return option, 100
        return None, 0

    normalized_catalog = {_normalize(opt): opt for opt in catalog}
    result = rf_process.extractOne(
        _normalize(text),
        normalized_catalog.keys(),
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
    )
    if result is None:
        return None, 0
    matched_norm, score, _ = result
    return normalized_catalog[matched_norm], int(score)


# ---------------------------------------------------------------------------
# Mapeo principal
# ---------------------------------------------------------------------------


def map_blocks_to_fields(
    blocks: list[OcrBlock],
    templates: Sequence[FieldTemplate],
    form_box: tuple[int, int, int, int],
    fuzzy_threshold: int = 70,
    min_overlap_px: int = 4,
) -> dict[str, MappedField]:
    """Asocia bloques OCR a campos de la plantilla por coordenadas + fuzzy.

    Algoritmo:
    1. Calcula el rectángulo absoluto de cada campo usando ``form_box``.
    2. Filtra los bloques cuyo bbox intersecta al menos ``min_overlap_px²``
       con el campo.
    3. Concatena el texto de los bloques candidatos (ordenados por posición).
    4. Si el campo tiene catálogo, aplica fuzzy matching para corregir OCR.
    5. Agrega el campo con ``needs_review=True`` si la confianza final es baja.

    Args:
        blocks:         Salida de ``PaddleOCREngine.scan_page_blocks``.
        templates:      Campos de la plantilla (``FieldTemplate``).
        form_box:       (x, y, w, h) del formulario en la imagen.
        fuzzy_threshold: Umbral de aceptación para rapidfuzz (0–100).
        min_overlap_px: Área mínima de intersección en píxeles cuadrados.

    Returns:
        Diccionario ``{field_id: MappedField}``.
    """
    results: dict[str, MappedField] = {}

    for tmpl in templates:
        # Sólo procesamos campos de tipo texto/catálogo con este mapper.
        if tmpl.field_type not in {"text", "catalog"}:
            continue

        field_rect = _field_rect(tmpl, form_box)
        candidates: list[OcrBlock] = []

        for block in blocks:
            block_rect = _block_rect(block)
            overlap = _intersection_area(field_rect, block_rect)
            if overlap >= min_overlap_px:
                candidates.append(block)

        if not candidates:
            results[tmpl.field_id] = MappedField(
                field_id=tmpl.field_id,
                raw_text="",
                matched_value=None,
                confidence=0.0,
                source_blocks=[],
                needs_review=True,
            )
            logger.debug("[fuzzy_mapper] Campo '%s': sin bloques solapantes.", tmpl.field_id)
            continue

        # Ordenar candidatos por posición (arriba→abajo, izquierda→derecha).
        candidates.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        raw_text = " ".join(b.text for b in candidates)
        avg_confidence = float(
            sum(b.confidence for b in candidates) / len(candidates)
        )

        matched_value: str | None = None
        fuzzy_score: int | None = None

        if tmpl.catalog:
            matched_value, fuzzy_score = _fuzzy_match(
                raw_text, tmpl.catalog, threshold=fuzzy_threshold
            )
            # La confianza final combina el score de OCR con el de fuzzy.
            if matched_value is not None:
                final_confidence = float(
                    0.5 * avg_confidence + 0.5 * (fuzzy_score / 100.0)
                )
            else:
                final_confidence = avg_confidence * 0.4  # sin match = confianza baja
        else:
            # Sin catálogo: devolver texto crudo con la confianza del OCR.
            matched_value = raw_text if raw_text.strip() else None
            final_confidence = avg_confidence

        needs_review = final_confidence < 0.50

        results[tmpl.field_id] = MappedField(
            field_id=tmpl.field_id,
            raw_text=raw_text,
            matched_value=matched_value,
            confidence=round(final_confidence, 4),
            fuzzy_score=fuzzy_score,
            source_blocks=candidates,
            needs_review=needs_review,
        )
        logger.debug(
            "[fuzzy_mapper] Campo '%s': raw='%s' matched='%s' conf=%.3f",
            tmpl.field_id,
            raw_text[:60],
            matched_value,
            final_confidence,
        )

    return results


# ---------------------------------------------------------------------------
# Utilidades de conversión desde field_map_sums.json
# ---------------------------------------------------------------------------


def templates_from_field_map(
    field_map: dict,
    page_key: str = "1",
    only_types: tuple[str, ...] = ("text", "catalog"),
) -> list[FieldTemplate]:
    """Crea ``FieldTemplate`` desde el formato de ``field_map_sums.json``.

    Sólo extrae los campos de los tipos especificados. Los checkboxes y
    números se siguen gestionando por los módulos originales del pipeline.

    Args:
        field_map:   Resultado de ``field_extractor.load_field_map``.
        page_key:    Clave de página en ``field_map["pages"]`` (por defecto ``"1"``).
        only_types:  Tipos de campo que se incluyen.

    Returns:
        Lista de ``FieldTemplate`` lista para ``map_blocks_to_fields``.
    """
    from horizontal_sheet_processor import CATALOGS  # tipo seguro, mismo subcomponente

    templates: list[FieldTemplate] = []

    page_fields: list[dict] = field_map.get("pages", {}).get(page_key, [])
    for fdef in page_fields:
        ftype = fdef.get("type", "")
        if ftype not in only_types:
            continue
        catalog_key = fdef.get("catalog")
        catalog = CATALOGS.get(catalog_key) if catalog_key else None
        templates.append(
            FieldTemplate(
                field_id=fdef["id"],
                bbox_rel=tuple(fdef["bbox"]),  # type: ignore[arg-type]
                catalog=catalog,
                field_type=ftype,
            )
        )

    return templates
