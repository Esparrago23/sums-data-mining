"""
debug_overlay.py
================
Genera imagenes con las cajas del field_map dibujadas sobre una pagina renderizada.

Sirve para calibrar coordenadas del OCR estructurado.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from preprocessor import normalize_page  # noqa: E402


def _page_num(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    return int(match.group(1)) if match else 1


def _imwrite_unicode(path: Path, imagen: np.ndarray) -> None:
    # cv2.imwrite falla en silencio en Windows para rutas con acentos (ANSI codepage).
    ok, buffer = cv2.imencode(".png", imagen)
    if not ok:
        raise IOError(f"cv2.imencode no pudo codificar la imagen para {path}")
    path.write_bytes(buffer.tobytes())


def draw_overlay(image_path: str | Path, field_map_path: str | Path, out_path: str | Path) -> Path:
    image_path = Path(image_path)
    field_map = json.loads(Path(field_map_path).read_text(encoding="utf-8"))
    page_num = _page_num(image_path)
    page = normalize_page(image_path, page_num)

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    x, y, w, h = page.form_box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 180, 255), 3)

    fields = field_map.get("page_kinds", {}).get(page.page_kind)
    if fields is None and page.page_kind == "datos_vivienda":
        fields = field_map.get("pages", {}).get("1", [])
    if fields is None:
        fields = field_map.get("pages", {}).get(str(page_num), [])
    for field in fields:
        rx1, ry1, rx2, ry2 = field["bbox"]
        x1 = int(x + rx1 * w)
        y1 = int(y + ry1 * h)
        x2 = int(x + rx2 * w)
        y2 = int(y + ry2 * h)
        color = (0, 200, 0) if field["type"] == "checkbox" else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            field["id"].split(".")[-1][:12],
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite_unicode(out_path, img)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Dibuja overlay del field_map sobre una pagina")
    parser.add_argument("image")
    parser.add_argument("--field-map", default="config/field_map_sums.json")
    parser.add_argument("--out", default="data/processed/debug_overlay.png")
    args = parser.parse_args()
    out = draw_overlay(args.image, args.field_map, args.out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
