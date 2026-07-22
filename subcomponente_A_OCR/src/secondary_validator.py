"""
secondary_validator.py
======================
Subcomponente A — Validación secundaria para campos de baja confianza.

Cuando PaddleOCR devuelve un score inferior a ``CONFIDENCE_THRESHOLD`` en un
campo clave, el sistema invoca este módulo. El flujo es:

    1. Heurísticas locales (limpieza, normalización, búsqueda en catálogo).
    2. [Futuro] Llamada a la API de DeepSeek si las heurísticas no resuelven.

Estado del stub DeepSeek: **NOT IMPLEMENTED** — ver ``_call_deepseek_api``.

No depende de ningún módulo fuera de ``subcomponente_A_OCR/``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

#: Confianza mínima aceptada. Por debajo → validación secundaria.
CONFIDENCE_THRESHOLD: float = 0.70

#: Número máximo de caracteres a enviar a la API externa en un prompt.
_MAX_PROMPT_CHARS: int = 500


# ---------------------------------------------------------------------------
# Tipos públicos
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Resultado de la validación secundaria de un campo.

    Attributes:
        value:      Valor final aceptado para el campo (puede ser ``None``).
        confidence: Confianza estimada tras la validación (0.0–1.0).
        source:     Origen del valor: ``"heuristic"``, ``"deepseek"``,
                    ``"catalog_fallback"``, ``"unresolved"``.
        raw_input:  Texto original que se validó.
        notes:      Texto libre para debugging / auditoría.
    """

    value: str | None
    confidence: float
    source: str
    raw_input: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Limpieza y normalización de texto
# ---------------------------------------------------------------------------


def _remove_accents(text: str) -> str:
    """Elimina acentos para comparaciones tolerantes."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_ocr_text(text: str) -> str:
    """Normalización agresiva típica de errores OCR en cédulas.

    - Minúsculas, strip, colapso de espacios.
    - Elimina caracteres de ruido de escaneo: ``|``, ``=``, ``_``, ``~``.
    - Elimina dígitos aislados (frecuente en bordes de tabla mal recortados).
    - Unifica caracteres OCR confundibles: ``0→o``, ``1→l`` en contexto de texto.
    """
    text = text.lower().strip()
    text = re.sub(r"[|=_~…]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _text_to_number(text: str) -> int | None:
    """Intenta extraer un número entero del texto OCR."""
    # PaddleOCR confunde 0 con O, 1 con l/I, 5 con S en campos numéricos.
    substitutions = {"o": "0", "O": "0", "l": "1", "I": "1", "S": "5", "s": "5"}
    cleaned = "".join(substitutions.get(c, c) for c in text)
    match = re.search(r"\d+", cleaned)
    if match:
        return int(match.group())
    return None


# ---------------------------------------------------------------------------
# Heurísticas por tipo de campo
# ---------------------------------------------------------------------------


def _heuristic_text(raw: str, field_id: str) -> ValidationResult:
    """Heurística genérica para campos de texto libre.

    Normaliza el texto, filtra líneas vacías y retorna si hay contenido útil.
    """
    cleaned = _normalize_ocr_text(raw)
    if not cleaned or len(cleaned) < 2:
        return ValidationResult(
            value=None,
            confidence=0.0,
            source="heuristic",
            raw_input=raw,
            notes=f"Texto vacío o demasiado corto para '{field_id}'.",
        )
    return ValidationResult(
        value=cleaned,
        confidence=0.55,
        source="heuristic",
        raw_input=raw,
        notes="Texto normalizado por heurística local.",
    )


def _heuristic_catalog(
    raw: str,
    field_id: str,
    catalog: list[str],
    fuzzy_threshold: int = 60,
) -> ValidationResult:
    """Heurística para campos con catálogo: fuzzy matching agresivo.

    Usa un umbral más bajo que el mapper principal para intentar rescatar
    textos con mayor distorsión.
    """
    normalized_raw = _remove_accents(_normalize_ocr_text(raw))
    if not normalized_raw:
        return ValidationResult(
            value=None,
            confidence=0.0,
            source="heuristic",
            raw_input=raw,
            notes=f"Texto vacío para catálogo de '{field_id}'.",
        )

    # Búsqueda exacta por substrings primero (más rápida).
    for option in catalog:
        opt_norm = _remove_accents(_normalize_ocr_text(option))
        if opt_norm in normalized_raw or normalized_raw in opt_norm:
            return ValidationResult(
                value=option,
                confidence=0.65,
                source="catalog_fallback",
                raw_input=raw,
                notes=f"Coincidencia por substring con '{option}'.",
            )

    # Fuzzy matching de rescate con rapidfuzz.
    try:
        from rapidfuzz import process as rf_process, fuzz  # type: ignore[import]

        norm_catalog = {_remove_accents(_normalize_ocr_text(o)): o for o in catalog}
        result = rf_process.extractOne(
            normalized_raw,
            norm_catalog.keys(),
            scorer=fuzz.partial_ratio,
            score_cutoff=fuzzy_threshold,
        )
        if result is not None:
            matched_norm, score, _ = result
            matched = norm_catalog[matched_norm]
            return ValidationResult(
                value=matched,
                confidence=round(0.4 + 0.3 * (score / 100.0), 4),
                source="catalog_fallback",
                raw_input=raw,
                notes=f"Fuzzy rescue (partial_ratio={score}) → '{matched}'.",
            )
    except ImportError:
        pass

    return ValidationResult(
        value=None,
        confidence=0.0,
        source="heuristic",
        raw_input=raw,
        notes=f"Sin coincidencia en catálogo para '{field_id}'.",
    )


def _heuristic_number(raw: str, field_id: str) -> ValidationResult:
    """Heurística para campos numéricos: extrae el primer entero válido."""
    number = _text_to_number(raw)
    if number is not None:
        return ValidationResult(
            value=str(number),
            confidence=0.60,
            source="heuristic",
            raw_input=raw,
            notes=f"Número extraído por heurística: {number}",
        )
    return ValidationResult(
        value=None,
        confidence=0.0,
        source="heuristic",
        raw_input=raw,
        notes=f"No se encontró número en texto OCR para '{field_id}'.",
    )


# ---------------------------------------------------------------------------
# Stub API DeepSeek
# ---------------------------------------------------------------------------


def _call_deepseek_api(prompt: str) -> str:  # noqa: ARG001
    """[STUB] Llama a la API de DeepSeek para corrección de texto OCR.

    Cuando se implemente:
    1. Añadir ``DEEPSEEK_API_KEY`` al entorno del contenedor Docker.
    2. Instalar ``httpx`` (ya en requirements raíz, disponible en la API).
    3. Reemplazar el ``raise NotImplementedError`` con la llamada real.

    Ejemplo de implementación::

        import os, httpx
        API_URL = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        r = httpx.post(API_URL, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    Args:
        prompt: Texto del prompt a enviar (máx. ``_MAX_PROMPT_CHARS`` chars).

    Raises:
        NotImplementedError: Siempre, hasta que se implemente el cuerpo real.
    """
    raise NotImplementedError(
        "[secondary_validator] DeepSeek API no implementada todavía.\n"
        "Ver docstring de _call_deepseek_api() para instrucciones de integración."
    )


def _build_deepseek_prompt(field_id: str, raw_text: str, context: dict[str, Any]) -> str:
    """Construye un prompt acotado para corrección de texto OCR."""
    ctx_str = "; ".join(f"{k}={v}" for k, v in list(context.items())[:3])
    prompt = (
        f"Eres un asistente de corrección de OCR para formularios médicos en español.\n"
        f"Campo: '{field_id}'\n"
        f"Texto OCR crudo: '{raw_text}'\n"
        f"Contexto del formulario: {ctx_str}\n"
        f"Devuelve SÓLO el texto corregido, sin explicaciones."
    )
    return prompt[:_MAX_PROMPT_CHARS]


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------


def validate_field(
    field_id: str,
    raw_text: str,
    context: dict[str, Any] | None = None,
    field_type: str = "text",
    catalog: list[str] | None = None,
    use_deepseek: bool = False,
) -> ValidationResult:
    """Valida y corrige el texto de un campo de baja confianza.

    Flujo:
    1. Aplica heurísticas locales según ``field_type``.
    2. Si el resultado sigue siendo de baja confianza Y ``use_deepseek=True``,
       intenta llamar a la API de DeepSeek (stub por ahora).

    Args:
        field_id:    Identificador del campo (para logging y contexto).
        raw_text:    Texto bruto extraído por OCR.
        context:     Diccionario de contexto adicional (otros campos ya extraídos).
        field_type:  Tipo del campo: ``"text"``, ``"catalog"``, ``"number"``.
        catalog:     Lista de valores válidos (requerido si ``field_type="catalog"``).
        use_deepseek: Si ``True``, intenta llamar a DeepSeek tras las heurísticas.

    Returns:
        ``ValidationResult`` con el valor corregido y la fuente.
    """
    context = context or {}
    result: ValidationResult

    # --- Paso 1: heurísticas locales ---
    if field_type == "number":
        result = _heuristic_number(raw_text, field_id)
    elif field_type == "catalog" and catalog:
        result = _heuristic_catalog(raw_text, field_id, catalog)
    else:
        result = _heuristic_text(raw_text, field_id)

    logger.debug(
        "[secondary_validator] Campo '%s': heurística → value='%s' conf=%.3f source='%s'",
        field_id,
        result.value,
        result.confidence,
        result.source,
    )

    # --- Paso 2: DeepSeek (stub) ---
    if use_deepseek and result.confidence < CONFIDENCE_THRESHOLD:
        prompt = _build_deepseek_prompt(field_id, raw_text, context)
        try:
            corrected = _call_deepseek_api(prompt)
            result = ValidationResult(
                value=corrected,
                confidence=0.80,
                source="deepseek",
                raw_input=raw_text,
                notes="Corregido por DeepSeek API.",
            )
            logger.info(
                "[secondary_validator] Campo '%s' corregido por DeepSeek: '%s'",
                field_id,
                corrected,
            )
        except NotImplementedError:
            logger.debug(
                "[secondary_validator] DeepSeek no implementado; usando resultado heurístico."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[secondary_validator] Error en DeepSeek para '%s': %s", field_id, exc
            )

    if result.value is None:
        result = ValidationResult(
            value=None,
            confidence=0.0,
            source="unresolved",
            raw_input=raw_text,
            notes=f"Campo '{field_id}' sin resolución tras todas las estrategias.",
        )

    return result


# ---------------------------------------------------------------------------
# Conveniencia: validar batch de MappedFields con baja confianza
# ---------------------------------------------------------------------------


def validate_low_confidence_fields(
    mapped_fields: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    use_deepseek: bool = False,
) -> dict[str, ValidationResult]:
    """Aplica ``validate_field`` a todos los campos por debajo del umbral.

    Args:
        mapped_fields:        Salida de ``fuzzy_mapper.map_blocks_to_fields``.
        confidence_threshold: Umbral de confianza mínima.
        use_deepseek:         Activar llamadas a DeepSeek (stub por defecto).

    Returns:
        Diccionario ``{field_id: ValidationResult}`` sólo para campos validados.
    """
    from fuzzy_mapper import MappedField  # importación local para evitar ciclos

    results: dict[str, ValidationResult] = {}
    for field_id, mf in mapped_fields.items():
        if not isinstance(mf, MappedField):
            continue
        if mf.confidence >= confidence_threshold and not mf.needs_review:
            continue
        logger.info(
            "[secondary_validator] Validando campo bajo confianza: '%s' (conf=%.3f)",
            field_id,
            mf.confidence,
        )
        vr = validate_field(
            field_id=field_id,
            raw_text=mf.raw_text,
            field_type=mf.field_id.split(".")[-1] if "." in mf.field_id else "text",
            use_deepseek=use_deepseek,
        )
        results[field_id] = vr
    return results
