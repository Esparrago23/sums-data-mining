"""
pdf_renderer.py
===============
Subcomponente A - renderizado local de PDFs escaneados.

Usa `pdftoppm` (Poppler) para convertir cada pagina del PDF a PNG. No usa OCR ni
modelos externos; solo prepara imagenes para el pipeline propio.
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

import fitz  # PyMuPDF


def _render_pdf_pymupdf(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    """Fallback puro Python cuando Poppler/pdftoppm no esta disponible."""
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    out_paths: list[Path] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out = out_dir / f"{pdf_path.stem}-{idx}.png"
            pix.save(out)
            out_paths.append(out)
    return out_paths


def render_pdf(pdf_path: str | Path, out_dir: str | Path, dpi: int = 180) -> list[Path]:
    """Renderiza un PDF a PNG y devuelve las rutas de paginas generadas.

    Los nombres siguen el patron `<stem>-1.png`, `<stem>-2.png`, etc.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = out_dir / pdf_path.stem
    pdftoppm = shutil.which("pdftoppm") or shutil.which("pdftoppm.cmd")
    if pdftoppm:
        cmd = [
            pdftoppm,
            "-r",
            str(dpi),
            "-png",
            str(pdf_path),
            str(prefix),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return sorted(out_dir.glob(f"{pdf_path.stem}-*.png"))
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    return _render_pdf_pymupdf(pdf_path, out_dir, dpi)


def render_all(raw_dir: str | Path, out_dir: str | Path, dpi: int = 180) -> dict[str, list[Path]]:
    """Renderiza todos los PDFs de `raw_dir`."""
    raw_dir = Path(raw_dir)
    result: dict[str, list[Path]] = {}
    for pdf in sorted(raw_dir.glob("*.pdf")):
        result[pdf.stem] = render_pdf(pdf, out_dir, dpi=dpi)
    return result
