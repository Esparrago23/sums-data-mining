"""
pdf_renderer.py
===============
Subcomponente A - renderizado local de PDFs escaneados.

Usa `pdftoppm` (Poppler) para convertir cada pagina del PDF a PNG. No usa OCR ni
modelos externos; solo prepara imagenes para el pipeline propio.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def render_pdf(pdf_path: str | Path, out_dir: str | Path, dpi: int = 180) -> list[Path]:
    """Renderiza un PDF a PNG y devuelve las rutas de paginas generadas.

    Los nombres siguen el patron `<stem>-1.png`, `<stem>-2.png`, etc.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = out_dir / pdf_path.stem
    cmd = [
        "pdftoppm",
        "-r",
        str(dpi),
        "-png",
        str(pdf_path),
        str(prefix),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sorted(out_dir.glob(f"{pdf_path.stem}-*.png"))


def render_all(raw_dir: str | Path, out_dir: str | Path, dpi: int = 180) -> dict[str, list[Path]]:
    """Renderiza todos los PDFs de `raw_dir`."""
    raw_dir = Path(raw_dir)
    result: dict[str, list[Path]] = {}
    for pdf in sorted(raw_dir.glob("*.pdf")):
        result[pdf.stem] = render_pdf(pdf, out_dir, dpi=dpi)
    return result
