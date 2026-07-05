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

    pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = evaluate_checkbox_fields(predictions, truth)
    report = {
        "n_pdfs": len(pdfs),
        "dpi": args.dpi,
        "documents": report_docs,
        "metrics": metrics,
        "trained_checkbox_model": trained_info,
        "split_metrics": split_metrics,
        "outputs": {
            "predictions": str(pred_path),
            "rendered_pages": str(rendered_dir),
            "rois": str(processed_dir / "rois"),
        },
    }
    report_path = processed_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Predicciones -> {pred_path}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
