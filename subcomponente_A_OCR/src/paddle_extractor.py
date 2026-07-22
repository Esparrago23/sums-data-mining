"""
paddle_extractor.py
===================
Subcomponente A — Motor principal de extracción de texto con PaddleOCR.

Lee una página completa una sola vez y devuelve todos los bloques de texto con
sus coordenadas absolutas (bounding box en píxeles) y score de confianza.

Diseñado para correr en CPU (Linux ARM64/x86_64) dentro de un contenedor Docker.
Si PaddlePaddle no está instalado el módulo degrada automáticamente a
``NullOCREngine``, que devuelve listas vacías sin interrumpir el pipeline.

Dependencias (sólo en subcomponente_A_OCR/requirements.txt):
    paddlepaddle>=2.6.2
    paddleocr>=2.7.3
"""

from __future__ import annotations

# --- FIX CRÍTICO PARA WINDOWS: Apagar aceleración problemática ---
import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
# -----------------------------------------------------------------

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos públicos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OcrBlock:
    """Bloque de texto detectado por el motor OCR.

    Attributes:
        text:       Texto extraído (ya limpio de saltos de línea redundantes).
        bbox:       Bounding box absoluta en píxeles: (x, y, w, h).
        confidence: Score de confianza en [0.0, 1.0].
        engine:     Nombre del motor que produjo este bloque.
    """

    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    engine: str = "PaddleOCR"


# ---------------------------------------------------------------------------
# Interfaz común de motor OCR
# ---------------------------------------------------------------------------


class BaseOCREngine(ABC):
    """Contrato mínimo que deben cumplir todos los motores de este pipeline."""

    @abstractmethod
    def scan_page_blocks(self, gray: np.ndarray) -> list[OcrBlock]:
        """Escanea una imagen en escala de grises y devuelve todos los bloques.

        El motor lee la página **completa** una sola vez. El recorte por campo
        se realiza posteriormente en ``fuzzy_mapper``.

        Args:
            gray: Imagen en escala de grises (dtype uint8, shape HxW).

        Returns:
            Lista de ``OcrBlock`` ordenada de arriba-izquierda a abajo-derecha.
        """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True si el motor está operativo."""


# ---------------------------------------------------------------------------
# Implementación nula (fallback cuando PaddlePaddle no está instalado)
# ---------------------------------------------------------------------------


class NullOCREngine(BaseOCREngine):
    """Motor de sustitución que no extrae nada pero no lanza excepciones."""

    def scan_page_blocks(self, gray: np.ndarray) -> list[OcrBlock]:  # noqa: ARG002
        logger.warning(
            "[paddle_extractor] NullOCREngine activo: "
            "PaddlePaddle no está instalado. Devolviendo lista vacía."
        )
        return []

    @property
    def is_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Motor principal: PaddleOCR
# ---------------------------------------------------------------------------


class PaddleOCREngine(BaseOCREngine):
    """Singleton lazy-init que envuelve PaddleOCR.

    El modelo se carga una única vez en memoria la primera vez que se llama a
    ``scan_page_blocks``. Las llamadas subsiguientes reutilizan el mismo objeto.

    Args:
        lang:        Idioma del modelo. ``'es'`` para español (PP-OCRv4 multi).
        use_angle_cls: Detecta texto rotado. Útil para cédulas mal escaneadas.
        use_gpu:     ``False`` en servidores ARM sin GPU (Raspberry Pi 4).
        det_db_thresh: Umbral de binarización del mapa de segmentación.
                       Valores bajos detectan más texto pero más ruido.
        rec_batch_num: Tamaño de lote para reconocimiento. Reducir en RAM baja.
    """

    _instance: PaddleOCREngine | None = None
    _model: Any = None  # paddleocr.PaddleOCR instance

    def __new__(cls, **kwargs: Any) -> PaddleOCREngine:
        # Singleton: un único engine por proceso.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        lang: str = "es",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        det_db_thresh: float = 0.3,
        rec_batch_num: int = 6,
    ) -> None:
        # Evitar re-inicialización si el modelo ya fue cargado.
        if self._model is not None:
            return

        try:
            from paddleocr import PaddleOCR  # type: ignore[import]

            logger.info("[paddle_extractor] Cargando modelo PaddleOCR (lang=%s)...", lang)
            self._model = PaddleOCR(
                lang=lang,
                use_angle_cls=use_angle_cls,
                det_db_thresh=det_db_thresh,
                rec_batch_num=rec_batch_num,
                enable_mkldnn=False,
            )
            logger.info("[paddle_extractor] Modelo PaddleOCR cargado correctamente.")
        except ImportError: 
        


        # try:
        #    from paddleocr import PaddleOCR  # type: ignore[import]

        #    logger.info("[paddle_extractor] Cargando modelo PaddleOCR (lang=%s)…", lang)
        #    self._model = PaddleOCR(
        #        lang=lang,
        #        use_angle_cls=use_angle_cls,
                # use_gpu=use_gpu,
        #        det_db_thresh=det_db_thresh,
        #        rec_batch_num=rec_batch_num,
                # Silenciar el log verboso de PaddlePaddle en producción.
                # show_log=False,
        #    )
        #    logger.info("[paddle_extractor] Modelo PaddleOCR cargado correctamente.")
        #except ImportError:
            logger.error(
                "[paddle_extractor] paddleocr/paddlepaddle no instalado. "
                "Usa NullOCREngine como alternativa."
            )
            self._model = None
        except Exception as exc:  # noqa: BLE001
            logger.error("[paddle_extractor] Error al inicializar PaddleOCR: %s", exc)
            self._model = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def scan_page_blocks(self, gray: np.ndarray) -> list[OcrBlock]:
        """Extrae todos los bloques de texto de la página completa.

        PaddleOCR requiere una imagen BGR o en escala de grises. Si recibe
        escala de grises la convierte internamente a BGR para compatibilidad.

        Args:
            gray: Imagen en escala de grises (uint8, HxW).

        Returns:
            Lista de ``OcrBlock`` con texto, bbox absoluta y confianza.
            Devuelve lista vacía si el motor no está disponible o falla.
        """
        if not self.is_available:
            logger.warning("[paddle_extractor] Motor no disponible, devolviendo [].")
            return []

        if gray is None or gray.size == 0:
            return []

        # PaddleOCR acepta numpy arrays BGR. Convertimos escala de grises.
        if len(gray.shape) == 2:
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            bgr = gray  # Ya es BGR

        try:
            # result = self._model.ocr(bgr, cls=True)
            result = self._model.ocr(bgr)
        except Exception as exc:  # noqa: BLE001
            logger.error("[paddle_extractor] Error en ocr(): %s", exc)
            return []

        return self._parse_result(result)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    @staticmethod
    def _poly_to_bbox(poly: list[list[float]]) -> tuple[int, int, int, int]:
        """Convierte el polígono de PaddleOCR (4 puntos) a (x, y, w, h)."""
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs)) - x
        h = int(max(ys)) - y
        return (x, y, max(1, w), max(1, h))

    @staticmethod
    def _clean_text(raw: str) -> str:
        """Limpia saltos de línea y espacios redundantes."""
        return " ".join(raw.replace("\n", " ").replace("\r", " ").split()).strip()

    def _parse_result(self, raw_result: Any) -> list[OcrBlock]:
        """Convierte la salida de PaddleOCR al tipo interno ``OcrBlock``.

        Formato de PaddleOCR:
            result = [ [  # una lista por página (sólo procesamos una)
                [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, score) ],
                ...
            ] ]
        """
        blocks: list[OcrBlock] = []

        if not raw_result:
            return blocks

        # PaddleOCR devuelve una lista de páginas; tomamos la primera.
        page_data = raw_result[0] if isinstance(raw_result[0], list) else raw_result

        if page_data is None:
            return blocks

        for line in page_data:
            try:
                poly, (text, score) = line
                cleaned = self._clean_text(text)
                if not cleaned:
                    continue
                bbox = self._poly_to_bbox(poly)
                blocks.append(
                    OcrBlock(
                        text=cleaned,
                        bbox=bbox,
                        confidence=float(round(score, 4)),
                        engine="PaddleOCR",
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                logger.debug("[paddle_extractor] Línea ignorada (%s): %s", exc, line)
                continue

        # Ordenar de arriba-izquierda a abajo-derecha para lecturas más predecibles.
        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        return blocks


# ---------------------------------------------------------------------------
# Función de conveniencia: obtener el mejor motor disponible
# ---------------------------------------------------------------------------


def get_ocr_engine(**kwargs: Any) -> BaseOCREngine:
    """Devuelve el mejor motor OCR disponible en el entorno actual.

    Intenta instanciar ``PaddleOCREngine``; si PaddlePaddle no está instalado
    o el modelo falla al cargar, devuelve ``NullOCREngine``.

    Usage::

        engine = get_ocr_engine()
        blocks = engine.scan_page_blocks(gray_image)
    """
    try:
        engine = PaddleOCREngine(**kwargs)
        if engine.is_available:
            return engine
        logger.warning("[paddle_extractor] PaddleOCREngine no disponible → NullOCREngine.")
        return NullOCREngine()
    except Exception as exc:  # noqa: BLE001
        logger.error("[paddle_extractor] Fallo al crear PaddleOCREngine: %s → NullOCREngine.", exc)
        return NullOCREngine()
