"""
doctr_fallback.py
=================
Subcomponente A — Interfaz abstracta y stub para DocTR como motor alternativo.

DocTR (Document Text Recognition de Mindee) es una alternativa ligera a
PaddleOCR, especialmente útil en entornos con restricciones de memoria o donde
PaddlePaddle no puede instalarse.

Estado actual: **STUB — no ejecuta extracción real.**
Para activar DocTR, implementar el cuerpo de ``DocTRFallback.extract``:

    pip install python-doctr[torch]   # o [tensorflow]

y reemplazar el ``raise NotImplementedError`` con la lógica real.

No depende de ningún módulo fuera de ``subcomponente_A_OCR/``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from paddle_extractor import BaseOCREngine, OcrBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interfaz de fallback (heredada de BaseOCREngine para compatibilidad total)
# ---------------------------------------------------------------------------


class BaseOCRFallback(BaseOCREngine, ABC):
    """Extensión de ``BaseOCREngine`` para motores de fallback.

    Agrega metadatos de prioridad y un método de diagnóstico.
    """

    #: Nombre del motor para logging y reporte.
    engine_name: str = "AbstractFallback"

    #: Prioridad de uso (mayor = se prefiere antes). PaddleOCR = 100.
    priority: int = 50

    @abstractmethod
    def extract(self, gray: np.ndarray) -> list[OcrBlock]:
        """Extrae bloques de texto de la imagen.

        Semánticamente idéntico a ``scan_page_blocks``; se aliasa abajo.
        """

    def scan_page_blocks(self, gray: np.ndarray) -> list[OcrBlock]:
        """Delega a ``extract`` para cumplir el contrato de ``BaseOCREngine``."""
        return self.extract(gray)

    def describe(self) -> dict:
        """Devuelve metadatos del motor para logging."""
        return {
            "engine": self.engine_name,
            "available": self.is_available,
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# Stub de DocTR
# ---------------------------------------------------------------------------


class DocTRFallback(BaseOCRFallback):
    """Stub de DocTR — preparado para integración futura.

    Cuando se desee activar DocTR:
    1. Instalar: ``pip install python-doctr[torch]``
    2. Reemplazar el cuerpo de ``extract`` con la llamada real a la API de DocTR.
    3. Cambiar ``_implemented = False`` a ``True``.

    Ejemplo de implementación real::

        from doctr.models import ocr_predictor
        from doctr.io import DocumentFile

        def extract(self, gray: np.ndarray) -> list[OcrBlock]:
            model = ocr_predictor(pretrained=True)
            doc = DocumentFile.from_images([gray])
            result = model(doc)
            blocks = []
            for page in result.pages:
                h, w = page.dimensions
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            (x1r, y1r), (x2r, y2r) = word.geometry
                            x, y = int(x1r * w), int(y1r * h)
                            bw = int((x2r - x1r) * w)
                            bh = int((y2r - y1r) * h)
                            blocks.append(OcrBlock(
                                text=word.value,
                                bbox=(x, y, bw, bh),
                                confidence=word.confidence,
                                engine="DocTR",
                            ))
            return blocks
    """

    engine_name: str = "DocTR"
    priority: int = 60  # Mayor prioridad que NullOCREngine, menor que PaddleOCR
    _implemented: bool = False  # Cambiar a True cuando se complete la integración

    @property
    def is_available(self) -> bool:
        """Devuelve ``False`` mientras sea un stub."""
        return self._implemented

    def extract(self, gray: np.ndarray) -> list[OcrBlock]:  # noqa: ARG002
        """Lanza NotImplementedError hasta que se implemente el cuerpo real.

        Raises:
            NotImplementedError: Siempre, hasta que ``_implemented = True``.
        """
        raise NotImplementedError(
            "[doctr_fallback] DocTRFallback no está implementado todavía.\n"
            "Para activar DocTR:\n"
            "  1. pip install python-doctr[torch]\n"
            "  2. Implementar el cuerpo de DocTRFallback.extract()\n"
            "  3. Establecer DocTRFallback._implemented = True\n"
            "Ver docstring de la clase para el ejemplo completo."
        )


# ---------------------------------------------------------------------------
# Registro de fallbacks disponibles (para selección dinámica en el pipeline)
# ---------------------------------------------------------------------------

#: Lista ordenada por prioridad descendente de motores de fallback.
#: Extender esta lista cuando se añadan nuevos motores.
REGISTERED_FALLBACKS: list[type[BaseOCRFallback]] = [
    DocTRFallback,
]


def get_best_fallback() -> BaseOCRFallback | None:
    """Devuelve el mejor fallback disponible, o ``None`` si no hay ninguno.

    Itera ``REGISTERED_FALLBACKS`` en orden de prioridad y devuelve la primera
    instancia cuyo ``is_available`` sea ``True``.
    """
    for cls in sorted(REGISTERED_FALLBACKS, key=lambda c: c.priority, reverse=True):
        try:
            instance = cls()
            if instance.is_available:
                logger.info("[doctr_fallback] Fallback activo: %s", cls.engine_name)
                return instance
        except Exception as exc:  # noqa: BLE001
            logger.warning("[doctr_fallback] Error instanciando %s: %s", cls.__name__, exc)
    logger.debug("[doctr_fallback] Ningún fallback disponible.")
    return None
