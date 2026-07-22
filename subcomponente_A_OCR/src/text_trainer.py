"""
text_trainer.py
===============
Entrenamiento ligero para campos de texto manuscrito/libre en la cédula.

Motor primario  : PaddleOCR + fuzzy mapping por coordenadas (apply_paddle_text).
Motor fallback  : Tesseract con recorte individual por bbox_rel (apply_ocr_text).
Motor heredado  : KNN sobre vectores de imagen (apply_text_model).

La seleccion de motor ocurre en run_all.py; este modulo solo expone las
funciones que cada motor necesita.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from sklearn.neighbors import KNeighborsClassifier

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Importacion condicional de modulos PaddleOCR (mismo subcomponente A)
# ---------------------------------------------------------------------------
try:
    from paddle_extractor import BaseOCREngine, get_ocr_engine  # noqa: F401
    from fuzzy_mapper import (
        FieldTemplate,
        MappedField,
        map_blocks_to_fields,
        templates_from_field_map,
    )  # noqa: F401
    from secondary_validator import (
        CONFIDENCE_THRESHOLD,
        validate_low_confidence_fields,
    )  # noqa: F401
    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False

TEXT_FIELDS = {
    "familia.nombre_informante",
    "familia.rol_familiar",
    "familia.domicilio",
    "familia.localidad",
    "familia.manzana",
    "familia.vivienda",
}
TRAIN_DOCS = [f"Cédula_{i:04d}" for i in range(1, 9)]
TEST_DOCS = ["Cédula_0009", "Cédula_0010"]


@dataclass
class TrainedTextModel:
    clf: KNeighborsClassifier
    train_docs: list[str]
    test_docs: list[str]
    n_train: int
    field_id: str
    trained_labels: list[str]


def _imread_gray(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def text_vector_from_roi(path: str | Path, height: int = 48, width: int = 200) -> np.ndarray | None:
    gray = _imread_gray(path)
    if gray is None or gray.size == 0:
        return None

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    if processed.size == 0:
        return None

    resized = cv2.resize(processed, (width, height), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0).reshape(-1)


def _confidence_from_distance(distance: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    return round(float(1.0 / (1.0 + distance)), 4)


def _clean_ocr_text(text: str) -> str:
    cleaned = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    for ch in ("|", "=", "_", "…"):
        cleaned = cleaned.replace(ch, "")
    for ch in ("-", "~"):
        cleaned = cleaned.replace(ch, " ")
    return " ".join(cleaned.split())


def _score_ocr_candidate(text: str, confidences: list[int] | tuple[int, ...] | None = None) -> float:
    if not text:
        return 0.0
    score = 0.0
    words = [w for w in text.split() if len(w) > 1]
    score += min(20.0, len(words) * 3.0)
    score += sum(min(8.0, len(w)) for w in words) / max(1, len(words))
    if confidences:
        score += sum(float(c) for c in confidences[:3]) / max(1, len(confidences[:3])) / 20.0
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(1, len(text))
    score *= 0.9 + 0.2 * alpha_ratio
    return round(score, 4)


def _preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    if gray is None or gray.size == 0:
        return gray

    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def ocr_text_from_roi(path_or_array: str | Path | np.ndarray, lang: str = "eng", psm: int = 6) -> str | None:
    """Ejecuta Tesseract sobre un path de imagen o un array numpy ya cargado.

    Args:
        path_or_array: Ruta al archivo de imagen O array numpy (grayscale uint8).
        lang:          Idioma de Tesseract (``'spa'`` para español).
        psm:           Page segmentation mode. 7 = single line, 6 = block.

    Returns:
        Texto limpio o ``None`` si la confianza es muy baja.
    """
    if isinstance(path_or_array, np.ndarray):
        gray = path_or_array
    else:
        gray = _imread_gray(path_or_array)

    if gray is None or gray.size == 0:
        return None

    # Escalar si el recorte es demasiado pequenio para Tesseract.
    height, width = gray.shape[:2]
    if width < 300:
        factor = max(1.0, 300.0 / float(width))
        gray = cv2.resize(gray, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_CUBIC)

    variants = [
        ("adaptive", _preprocess_for_ocr(gray)),
        ("otsu", cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("equalized", cv2.equalizeHist(gray)),
    ]

    best_text = None
    best_score = -1.0
    for _, image in variants:
        config = f"--oem 1 --psm {psm}"
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        cleaned = _clean_ocr_text(text)
        if not cleaned:
            continue
        score = _score_ocr_candidate(cleaned)
        if score > best_score:
            best_score = score
            best_text = cleaned

    return best_text if best_score > 2.5 else None


def _labeled_rows(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    doc_ids: list[str],
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[str]]]:
    rows: dict[str, list[np.ndarray]] = {field: [] for field in TEXT_FIELDS}
    y: dict[str, list[str]] = {field: [] for field in TEXT_FIELDS}

    for doc_id in doc_ids:
        expected_fields = truth.get(doc_id, {})
        pred_fields = predictions.get(doc_id, {}).get("fields", {})
        for field_id in TEXT_FIELDS:
            expected = expected_fields.get(field_id)
            if expected in (None, ""):
                continue
            pred_field = pred_fields.get(field_id)
            if not pred_field or pred_field.get("type") != "text":
                continue
            roi = pred_field.get("roi")
            if not roi:
                continue
            vec = text_vector_from_roi(roi)
            if vec is None:
                continue
            rows[field_id].append(vec)
            y[field_id].append(str(expected).strip())

    return rows, y


def train_text_models(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    train_docs: list[str] | None = None,
    test_docs: list[str] | None = None,
) -> dict[str, TrainedTextModel]:
    train_docs = train_docs or TRAIN_DOCS
    test_docs = test_docs or TEST_DOCS
    rows, y = _labeled_rows(predictions, truth, train_docs)
    models: dict[str, TrainedTextModel] = {}
    for field_id in TEXT_FIELDS:
        if len(rows[field_id]) < 2 or len(set(y[field_id])) < 2:
            continue
        clf = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        clf.fit(np.vstack(rows[field_id]), y[field_id])
        models[field_id] = TrainedTextModel(
            clf=clf,
            train_docs=train_docs,
            test_docs=test_docs,
            n_train=len(y[field_id]),
            field_id=field_id,
            trained_labels=sorted(set(y[field_id])),
        )
    return models


# ---------------------------------------------------------------------------
# Motor fallback: Tesseract con recorte individual por bbox_rel
# ---------------------------------------------------------------------------

def apply_ocr_text(
    predictions: dict[str, Any],
    field_map: dict[str, Any] | None = None,
    page_key: str = "1",
) -> None:
    """Fallback de texto: Tesseract con recorte individual por bbox_rel.

    Para cada campo ``type=text``, carga la imagen completa de la pagina
    (embebida como ``_page_source``/``_form_box`` por ``field_extractor``) y
    recorta exactamente la region del campo usando ``bbox_rel`` del
    ``field_map``. Cada campo recibe su propio crop — no se lee la misma franja
    para todos.

    Si ``field_map`` no se pasa, cae al ROI pre-guardado por ``field_extractor``
    (comportamiento original, menos preciso pero siempre disponible).

    Args:
        predictions: Diccionario de predicciones del pipeline.
        field_map:   Resultado de ``load_field_map``. Opcional pero recomendado.
        page_key:    Clave de pagina en ``field_map["pages"]``.
    """
    # Lookup: field_id -> bbox_rel desde el field_map.
    bbox_lookup: dict[str, tuple[float, float, float, float]] = {}
    if field_map is not None:
        for fdef in field_map.get("pages", {}).get(page_key, []):
            if fdef.get("type") == "text":
                bbox_lookup[fdef["id"]] = tuple(fdef["bbox"])  # type: ignore[assignment]

    # Cache de imagenes: evitar re-leer el mismo PNG por cada campo.
    _img_cache: dict[str, np.ndarray | None] = {}

    for doc_id, doc in predictions.items():
        for field_id, pred_field in doc.get("fields", {}).items():
            if pred_field.get("type") != "text":
                continue

            page_source: str | None = pred_field.get("_page_source")
            form_box_raw: list | None = pred_field.get("_form_box")
            bbox_rel = bbox_lookup.get(field_id)

            # --- Recorte individual por bbox_rel (camino principal) ---
            if page_source and form_box_raw and bbox_rel:
                if page_source not in _img_cache:
                    try:
                        data = np.fromfile(page_source, dtype=np.uint8)
                        _img_cache[page_source] = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "[text_trainer] Doc '%s': no se pudo cargar '%s': %s",
                            doc_id, page_source, exc,
                        )
                        _img_cache[page_source] = None

                gray_full = _img_cache[page_source]
                if gray_full is not None:
                    fx, fy, fw, fh = (
                        int(form_box_raw[0]), int(form_box_raw[1]),
                        int(form_box_raw[2]), int(form_box_raw[3]),
                    )
                    x1r, y1r, x2r, y2r = bbox_rel
                    # Convertir coordenadas relativas al form_box a pixeles absolutos.
                    ax1 = max(0, fx + int(x1r * fw))
                    ay1 = max(0, fy + int(y1r * fh))
                    ax2 = min(gray_full.shape[1], fx + int(x2r * fw))
                    ay2 = min(gray_full.shape[0], fy + int(y2r * fh))
                    crop = gray_full[ay1:ay2, ax1:ax2]

                    if crop.size > 0:
                        # Agrandar si el crop es demasiado angosto para Tesseract.
                        h_crop, w_crop = crop.shape[:2]
                        if w_crop < 150:
                            scale = max(2.0, 150.0 / float(w_crop))
                            crop = cv2.resize(
                                crop,
                                (int(w_crop * scale), int(h_crop * scale)),
                                interpolation=cv2.INTER_CUBIC,
                            )
                        # PSM 7 = una sola linea de texto (optimo para campos de cédula).
                        text = ocr_text_from_roi(crop, lang="spa", psm=7)
                        if text:
                            pred_field["value"] = text[:300]
                            pred_field["needs_review"] = True
                            pred_field["confidence"] = 0.50
                            pred_field["model"] = "Tesseract/bbox_crop"
                            log.debug(
                                "[text_trainer] Doc '%s' campo '%s': Tesseract='%s'",
                                doc_id, field_id, text[:60],
                            )
                            continue

            # --- Fallback final: ROI pre-guardado por field_extractor ---
            roi = pred_field.get("roi")
            if not roi:
                continue
            text = ocr_text_from_roi(roi)
            if text is None:
                continue
            pred_field["value"] = text
            pred_field["needs_review"] = True
            pred_field["confidence"] = 0.40
            pred_field["model"] = "Tesseract/roi"


# ---------------------------------------------------------------------------
# Motor principal: PaddleOCR + fuzzy mapping
# ---------------------------------------------------------------------------

def apply_paddle_text(
    predictions: dict[str, Any],
    engine: "BaseOCREngine | None" = None,
    field_map: dict[str, Any] | None = None,
    page_key: str = "1",
    low_conf_threshold: float | None = None,
    use_deepseek: bool = False,
) -> None:
    """Extrae campos de texto usando PaddleOCR + fuzzy mapping por coordenadas.

    Para cada documento del diccionario ``predictions``, escanea la imagen de
    la pagina correspondiente *una sola vez* y mapea los bloques de texto a los
    campos de la plantilla mediante interseccion de coordenadas + rapidfuzz.

    Los campos de tipo ``checkbox`` y ``number`` **no se tocan**: este modulo
    solo escribe sobre campos con ``type in {'text', 'catalog'}``.

    Args:
        predictions:        Diccionario de predicciones del pipeline.
        engine:             Motor OCR a usar. Si ``None``, se crea con
                            ``get_ocr_engine()`` (PaddleOCR o NullOCREngine).
        field_map:          Mapa de campos (salida de ``load_field_map``). Si
                            ``None``, los templates se omiten.
        page_key:           Clave de pagina en ``field_map["pages"]``.
        low_conf_threshold: Umbral de confianza bajo el que se activa la
                            validacion secundaria.
        use_deepseek:       Activar llamada a DeepSeek en el validador secundario
                            (stub hasta que se implemente).

    Note:
        Si ``_PADDLE_AVAILABLE`` es ``False`` (PaddlePaddle no instalado), la
        funcion registra un warning y retorna sin modificar ``predictions``.
        El pipeline debe caer entonces a ``apply_ocr_text`` (Tesseract).
    """
    if not _PADDLE_AVAILABLE:
        log.warning(
            "[text_trainer] apply_paddle_text: modulos PaddleOCR no disponibles. "
            "Usar apply_ocr_text (Tesseract) como fallback."
        )
        return

    threshold = low_conf_threshold if low_conf_threshold is not None else CONFIDENCE_THRESHOLD

    # Instanciar motor si no se paso uno explicito.
    _engine = engine if engine is not None else get_ocr_engine()
    if not _engine.is_available:
        log.warning(
            "[text_trainer] PaddleOCR no disponible (NullOCREngine). "
            "Retornando sin modificar predicciones; el pipeline debe caer a Tesseract."
        )
        return

    # Construir templates de coordenadas si hay field_map.
    templates = []
    if field_map is not None:
        templates = templates_from_field_map(
            field_map, page_key=page_key, only_types=("text", "catalog")
        )

    for doc_id, doc in predictions.items():
        # ------------------------------------------------------------------
        # Obtener imagen + form_box.
        # Camino 1: _page_source/_form_box embebidos en cada campo de texto
        #           (disponible si field_extractor fue parcheado).
        # Camino 2: metadatos de paginas doc["pages"] (compatibilidad legacy).
        # ------------------------------------------------------------------
        page_source: str | None = None
        form_box: tuple[int, int, int, int] | None = None

        for pred_field in doc.get("fields", {}).values():
            if pred_field.get("type") == "text":
                src = pred_field.get("_page_source")
                fb = pred_field.get("_form_box")
                if src and fb:
                    page_source = src
                    form_box = tuple(int(v) for v in fb)  # type: ignore[assignment]
                    break

        if page_source is None or form_box is None:
            for page_meta in doc.get("pages", {}).values():
                src = page_meta.get("source")
                fb = page_meta.get("form_box")
                if src and fb:
                    page_source = src
                    form_box = tuple(int(v) for v in fb)  # type: ignore[assignment]
                    break

        if page_source is None or form_box is None:
            log.debug("[text_trainer] Doc '%s': sin metadata de pagina, saltando.", doc_id)
            continue

        # Cargar imagen en escala de grises.
        try:
            data = np.fromfile(page_source, dtype=np.uint8)
            gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(page_source)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[text_trainer] Doc '%s': no se pudo cargar imagen '%s': %s",
                doc_id, page_source, exc,
            )
            continue

        # ---- Escaneo completo de la pagina (una sola llamada por pagina) ----
        blocks = _engine.scan_page_blocks(gray)
        log.debug(
            "[text_trainer] Doc '%s': PaddleOCR detecto %d bloques.",
            doc_id, len(blocks),
        )

        # ---- Mapeo por coordenadas + fuzzy ----
        if templates and blocks:
            mapped = map_blocks_to_fields(
                blocks=blocks,
                templates=templates,
                form_box=form_box,
            )
        else:
            mapped = {}

        # ---- Validacion secundaria para campos de baja confianza ----
        low_conf_validated = {}
        if mapped:
            low_conf_validated = validate_low_confidence_fields(
                mapped,
                confidence_threshold=threshold,
                use_deepseek=use_deepseek,
            )

        # ---- Escribir resultados en predictions ----
        for field_id, pred_field in doc.get("fields", {}).items():
            if pred_field.get("type") not in {"text", "catalog"}:
                continue

            # Priorizar resultado del mapper por coordenadas.
            mf = mapped.get(field_id)
            if mf is not None and mf.matched_value:
                vr = low_conf_validated.get(field_id)
                if vr is not None and vr.confidence > (mf.confidence or 0):
                    pred_field["value"] = vr.value
                    pred_field["confidence"] = vr.confidence
                    pred_field["model"] = f"PaddleOCR+{vr.source}"
                    pred_field["needs_review"] = vr.confidence < threshold
                else:
                    pred_field["value"] = mf.matched_value
                    pred_field["confidence"] = mf.confidence
                    pred_field["model"] = "PaddleOCR+fuzzy"
                    pred_field["needs_review"] = mf.needs_review
                    if mf.fuzzy_score is not None:
                        pred_field.setdefault("features", {})["fuzzy_score"] = mf.fuzzy_score
                continue

            # Fallback a texto crudo si no hubo match de coordenadas.
            if mf is not None and mf.raw_text:
                pred_field["value"] = mf.raw_text
                pred_field["confidence"] = mf.confidence
                pred_field["model"] = "PaddleOCR+raw"
                pred_field["needs_review"] = True
                continue

            # Ultimo recurso: concatenar primeros bloques de la pagina.
            if field_id in TEXT_FIELDS and blocks:
                all_text = " ".join(b.text for b in blocks[:10])
                if all_text.strip():
                    pred_field["value"] = all_text[:200]
                    pred_field["confidence"] = 0.30
                    pred_field["model"] = "PaddleOCR+fullpage_fallback"
                    pred_field["needs_review"] = True


def apply_text_model(predictions: dict[str, Any], models: dict[str, TrainedTextModel]) -> None:
    for doc in predictions.values():
        for field_id, pred_field in doc.get("fields", {}).items():
            if field_id not in models or pred_field.get("type") != "text":
                continue
            roi = pred_field.get("roi")
            if not roi:
                continue
            vec = text_vector_from_roi(roi)
            if vec is None:
                continue
            model = models[field_id]
            pred = str(model.clf.predict([vec])[0])
            distance = float(model.clf.kneighbors([vec], n_neighbors=1, return_distance=True)[0][0][0])
            pred_field["value"] = pred
            pred_field["needs_review"] = False
            pred_field["confidence"] = _confidence_from_distance(distance)
            pred_field["model"] = "KNeighborsClassifier(k=1, cosine) text field"
            pred_field["features"]["trained_distance"] = round(distance, 4)


def evaluate_text_docs(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    doc_ids: list[str],
) -> dict[str, Any]:
    total = 0
    correct = 0
    errors: list[dict[str, str]] = []
    for doc_id in doc_ids:
        pred_fields = predictions.get(doc_id, {}).get("fields", {})
        expected_fields = truth.get(doc_id, {})
        for field_id in TEXT_FIELDS:
            expected = expected_fields.get(field_id)
            if expected in (None, ""):
                continue
            pred_field = pred_fields.get(field_id)
            if not pred_field or pred_field.get("type") != "text":
                continue
            total += 1
            got = str(pred_field.get("value", "")).strip().lower()
            exp = str(expected).strip().lower()
            if got == exp:
                correct += 1
            else:
                errors.append({"doc_id": doc_id, "field": field_id, "expected": str(expected), "got": got})

    return {
        "docs": doc_ids,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "errors": errors,
    }


def evaluate_text_split(
    predictions: dict[str, Any],
    truth: dict[str, Any],
    train_docs: list[str],
    test_docs: list[str],
) -> dict[str, Any]:
    return {
        "train": evaluate_text_docs(predictions, truth, train_docs),
        "test": evaluate_text_docs(predictions, truth, test_docs),
    }
