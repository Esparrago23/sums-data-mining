"""
run_all.py
==========
Subcomponente A - pipeline OCR estructurado para cedulas PDF escaneadas.

Uso:
    python3 src/run_all.py

Salida:
    data/processed/rendered_pages/   paginas PNG
    data/processed/rois/             recortes de campos manuscritos
    data/processed/predictions.json  campos extraidos
    data/processed/report.json       resumen y metricas opcionales
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
os.chdir(ROOT_DIR)

from evaluator import evaluate_checkbox_fields, load_ground_truth  # noqa: E402
from field_extractor import extract_document, load_field_map  # noqa: E402
from pdf_renderer import render_all  # noqa: E402
from preprocessor import normalize_page  # noqa: E402
from checkbox_trainer import (  # noqa: E402
    TEST_DOCS,
    TRAIN_DOCS,
    apply_exclusive_groups,
    apply_trained_checkbox_model,
    evaluate_labeled_split,
    train_checkbox_model,
)
from number_trainer import (  # noqa: E402
    TEST_DOCS as NUMBER_TEST_DOCS,
    TRAIN_DOCS as NUMBER_TRAIN_DOCS,
    apply_number_model,
    evaluate_number_split,
    train_number_model,
)
from review_exporter import build_review_export  # noqa: E402
from rol_trainer import (  # noqa: E402
    ROL_CATALOG,
    TEST_DOCS as ROL_TEST_DOCS,
    TRAIN_DOCS as ROL_TRAIN_DOCS,
    apply_rol_model,
    evaluate_rol_split,
    train_rol_model,
)
from text_trainer import (  # noqa: E402
    TEXT_FIELDS,
    TEST_DOCS as TEXT_TEST_DOCS,
    TRAIN_DOCS as TEXT_TRAIN_DOCS,
    apply_text_model,
    apply_ocr_text,
    evaluate_text_split,
    train_text_models,
)

# Importación condicional del nuevo motor PaddleOCR.
try:
    from text_trainer import apply_paddle_text  # noqa: E402
    from paddle_extractor import get_ocr_engine  # noqa: E402
    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False


def _page_num(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    return int(match.group(1)) if match else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR estructurado de cedulas SUMS")
    parser.add_argument("--raw-dir", default="data/raw_pdfs")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--field-map", default="config/field_map_sums.json")
    parser.add_argument("--ground-truth", default="data/ground_truth/campos_esperados.json")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    rendered_dir = processed_dir / "rendered_pages"
    processed_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No hay PDFs en {raw_dir}")

    print(f"[A] PDFs encontrados: {len(pdfs)}")
    rendered = render_all(raw_dir, rendered_dir, dpi=args.dpi)
    field_map = load_field_map(args.field_map)

    predictions: dict[str, dict] = {}
    report_docs = []
    for doc_id, page_paths in rendered.items():
        page_paths = sorted(page_paths, key=_page_num)
        pages = [normalize_page(p, _page_num(p)) for p in page_paths]
        pred = extract_document(doc_id, pages, field_map, processed_dir)
        predictions[doc_id] = pred
        report_docs.append({
            "doc_id": doc_id,
            "pages": len(pages),
            "fields": len(pred["fields"]),
            "needs_review": sum(1 for f in pred["fields"].values() if f.get("needs_review")),
        })
        print(f"[A] {doc_id}: {len(pages)} paginas, {len(pred['fields'])} campos")

    pred_path = processed_dir / "predictions.json"
    pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    truth = load_ground_truth(args.ground_truth)
    trained_info = None
    split_metrics = None
    number_info = None
    number_split_metrics = None
    rol_info = None
    rol_split_metrics = None
    trained_model = train_checkbox_model(predictions, truth, TRAIN_DOCS, TEST_DOCS)
    if trained_model is not None:
        apply_trained_checkbox_model(predictions, trained_model)
        apply_exclusive_groups(predictions)
        trained_info = {
            "type": "LogisticRegression + DictVectorizer",
            "train_docs": trained_model.train_docs,
            "test_docs": trained_model.test_docs,
            "n_train_examples": trained_model.n_train,
            "exclusive_groups": True,
        }
        split_metrics = evaluate_labeled_split(
            predictions, truth, trained_model.train_docs, trained_model.test_docs
        )

    number_model = train_number_model(predictions, truth, NUMBER_TRAIN_DOCS, NUMBER_TEST_DOCS)
    if number_model is not None:
        apply_number_model(predictions, number_model)
        number_info = {
            "type": "KNeighborsClassifier(k=1) + loop heuristic for habitantes=9",
            "train_docs": number_model.train_docs,
            "test_docs": number_model.test_docs,
            "n_train_examples": number_model.n_train,
            "fields": ["vivienda.numero_cuartos", "vivienda.numero_habitantes"],
        }
        number_split_metrics = evaluate_number_split(
            predictions, truth, number_model.train_docs, number_model.test_docs
        )

    rol_model = train_rol_model(predictions, truth, ROL_TRAIN_DOCS, ROL_TEST_DOCS)
    if rol_model is not None:
        apply_rol_model(predictions, rol_model)
        rol_info = {
            "type": "KNeighborsClassifier(k=1, cosine) palabra completa",
            "train_docs": rol_model.train_docs,
            "test_docs": rol_model.test_docs,
            "n_train_examples": rol_model.n_train,
            "field": "familia.rol_familiar",
            "catalog": ROL_CATALOG,
            "trained_labels": rol_model.trained_labels,
            "untrained_labels": rol_model.untrained_labels,
        }
        rol_split_metrics = evaluate_rol_split(
            predictions, truth, rol_model.train_docs, rol_model.test_docs
        )

    import logging as _logging_run
    _log_run = _logging_run.getLogger("sums.ocr.run_all")

    from text_trainer import apply_ocr_text, train_text_models  # noqa: E402

    # -----------------------------------------------------------------------
    # Selección dinámica del motor de texto:
    # 1º PaddleOCR + fuzzy mapping (si está instalado y operativo)
    # 2º Tesseract OCR (fallback existente, siempre disponible)
    # Los checkboxes y números NO se ven afectados por este bloque.
    # -----------------------------------------------------------------------
    active_text_engine = "Tesseract OCR"
    paddle_engine = None

    if _PADDLE_AVAILABLE:
        try:
            paddle_engine = get_ocr_engine()
            if paddle_engine.is_available:
                _log_run.info("[A] Motor de texto: PaddleOCR + fuzzy mapping")
                apply_paddle_text(
                    predictions,
                    engine=paddle_engine,
                    field_map=field_map,
                    page_key="1",
                )
                active_text_engine = "PaddleOCR+fuzzy"
            else:
                _log_run.warning("[A] PaddleOCR no disponible, usando Tesseract OCR.")
                apply_ocr_text(predictions, field_map=field_map, page_key="1")
        except Exception as _paddle_err:  # noqa: BLE001
            _log_run.error("[A] Error en PaddleOCR: %s, usando Tesseract OCR.", _paddle_err)
            apply_ocr_text(predictions, field_map=field_map, page_key="1")
    else:
        _log_run.info("[A] Motor de texto: Tesseract OCR (PaddlePaddle no instalado)")
        apply_ocr_text(predictions, field_map=field_map, page_key="1")
    text_models = train_text_models(predictions, truth, TEXT_TRAIN_DOCS, TEXT_TEST_DOCS)
    text_info = {
        "type": active_text_engine,
        "fields": sorted(TEXT_FIELDS),
        "predicted_fields": sum(
            1 for doc in predictions.values() for f in doc.get("fields", {}).values()
            if f.get("type") == "text" and f.get("value") is not None
        ),
        "paddle_available": _PADDLE_AVAILABLE and (paddle_engine.is_available if paddle_engine else False),
    }
    text_split_metrics = None
    if text_models:
        apply_text_model(predictions, text_models)
        text_info["type"] = "Hybrid OCR + KNeighborsClassifier(k=1, cosine) text field"
        text_info["train_docs"] = TEXT_TRAIN_DOCS
        text_info["test_docs"] = TEXT_TEST_DOCS
        text_info["n_train_examples"] = sum(model.n_train for model in text_models.values())
        text_info["trained_fields"] = sorted(text_models.keys())
        text_split_metrics = evaluate_text_split(
            predictions, truth, TEXT_TRAIN_DOCS, TEXT_TEST_DOCS
        )

    pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    review_export = build_review_export(predictions)
    review_path = processed_dir / "review_output.json"
    review_path.write_text(json.dumps(review_export, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = evaluate_checkbox_fields(predictions, truth)
    report = {
        "n_pdfs": len(pdfs),
        "dpi": args.dpi,
        "documents": report_docs,
        "metrics": metrics,
        "trained_checkbox_model": trained_info,
        "split_metrics": split_metrics,
        "trained_number_model": number_info,
        "number_split_metrics": number_split_metrics,
        "trained_rol_model": rol_info,
        "rol_split_metrics": rol_split_metrics,
        "trained_text_model": text_info,
        "text_split_metrics": text_split_metrics,
        "outputs": {
            "predictions": str(pred_path),
            "review_output": str(review_path),
            "rendered_pages": str(rendered_dir),
            "rois": str(processed_dir / "rois"),
        },
    }
    report_path = processed_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Predicciones -> {pred_path}")
    print(f"[OK] Validación    -> {review_path}")
    print(f"[OK] Reporte      -> {report_path}")
    if metrics["checkbox_total"] == 0:
        print("[INFO] Sin ground truth: no se calcularon metricas de accuracy.")
    else:
        print(f"[OK] Checkbox accuracy: {metrics['checkbox_accuracy']}")
    if split_metrics:
        print(
            "[OK] Train/Test checkbox accuracy: "
            f"{split_metrics['train']['accuracy']} / {split_metrics['test']['accuracy']}"
        )
    if number_split_metrics:
        print(
            "[OK] Train/Test number accuracy: "
            f"{number_split_metrics['train']['accuracy']} / {number_split_metrics['test']['accuracy']}"
        )
    if rol_split_metrics:
        print(
            "[OK] Train/Test rol accuracy: "
            f"{rol_split_metrics['train']['accuracy']} / {rol_split_metrics['test']['accuracy']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
