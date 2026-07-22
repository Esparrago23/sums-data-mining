# -*- coding: utf-8 -*-
"""
buscador_estructurado.py
==========================
Subcomponente C — Búsqueda ESTRUCTURADA sobre datos de la cédula (no texto libre).

POR QUÉ EXISTE ESTE MÓDULO (y no solo el buscador de notas): el campo
`observaciones` de una cédula real es, en el propio generador sintético
(`synthetic_generator.py`), un placeholder ("Cédula sintética #123...") -- NO
una narrativa clínica rica. El corpus de 150 notas que indexan
tfidf_engine/bm25_engine/embeddings_engine se INVENTÓ aparte para el ejercicio
de IR (Lab 1-3) y no tiene conexión con datos reales de una cédula. Por eso
consultas como "sarampión sin vacunar" o "familias con mascotas" NUNCA
podrían resolverse bien con similitud de texto (no hay texto que decir eso) --
son FILTROS sobre datos que SÍ existen siempre en cada cédula (edad, vacunas
aplicadas, animales en la vivienda, atención de embarazo, nutrición por
integrante). Este módulo responde ese tipo de consulta con un filtro real,
no con una búsqueda de similitud.

DISEÑO: un enrutador de intención determinístico (por palabras clave contra
catálogos YA conocidos y finitos -- VACUNAS, CAT_ANIMAL, geografía -- más
detección simple de negación), no un modelo entrenado. Es deliberado: con
vocabulario cerrado y conocido, el matching por catálogo es MÁS confiable que
esperar que un embedding "adivine" la intención, y además es 100% explicable
(se puede decir exactamente por qué una cédula coincidió).

LIMITACIÓN HONESTA sobre "dirección cercana": no hay coordenadas geográficas
en los datos (ni sintéticos ni en BD_MAPPING.md), así que "cercanía" se
aproxima por MISMA COLONIA/CALLE, no por distancia real en metros.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from catalogos_sums import (
    CAT_ANIMAL, CAT_ENFERMEDAD_CRONICA, VACUNAS,
    COLONIAS_SUCHIAPA, COLONIAS_TUXTLA, CALLES_SUCHIAPA, CALLES_TUXTLA,
)
from grupos_vulnerables import calcular_banderas

NEGACIONES = {"sin", "no", "falta", "faltan", "carece", "carecen"}
UMBRAL_NUTRICION_BAJA = 3  # mismo umbral que compute_risk (synthetic_generator.py)

# Palabras cortas/conectores que NUNCA cuentan como palabra significativa de un
# catálogo, aunque cumplan la longitud mínima -- evita falsos positivos como
# que cualquier consulta con "del" dispare "VPH (Virus DEL Papiloma Humano)",
# o que "otra" (parte del catálogo "Otra" vacuna, un cajón de sastre) dispare
# con cualquier frase que la contenga.
CONECTORES_EXCLUIDOS = {
    "del", "los", "las", "con", "por", "una", "uno", "que", "para", "otra",
    "otro", "sus", "tos",
}


def _quitar_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _normalizar(s: str) -> str:
    return _quitar_acentos(s.lower()).strip()


def _palabras_significativas(catalogo_str: str) -> set[str]:
    """Extrae palabras de una entrada de catálogo (permite acrónimos cortos
    como "BCG"/"VPH", pero descarta conectores) para matchear "sarampion"
    contra "SR (Sarampión,rubeola)" sin necesitar el texto exacto."""
    limpio = _normalizar(re.sub(r"[()]", " ", catalogo_str))
    return {
        w for w in re.findall(r"[a-z]+", limpio)
        if len(w) >= 3 and w not in CONECTORES_EXCLUIDOS
    }


def _hay_negacion(query_norm: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", query_norm))
    return bool(tokens & NEGACIONES)


@dataclass
class ResultadoFiltro:
    disponible: bool
    categoria: str | None = None
    detalle: str | None = None
    negado: bool = False


def _detectar_vacuna(query_norm: str) -> str | None:
    for vacuna in VACUNAS:
        if _palabras_significativas(vacuna) & set(re.findall(r"[a-z]+", query_norm)):
            return vacuna
    return None


def _detectar_enfermedad(query_norm: str) -> str | None:
    for enf in CAT_ENFERMEDAD_CRONICA:
        if _palabras_significativas(enf) & set(re.findall(r"[a-z]+", query_norm)):
            return enf
    return None


def _detectar_geografia(query_norm: str) -> str | None:
    for lugar in COLONIAS_SUCHIAPA + COLONIAS_TUXTLA + CALLES_SUCHIAPA + CALLES_TUXTLA:
        if _normalizar(lugar) in query_norm:
            return lugar
    return None


PALABRAS_NUTRICION = {"nutricion", "proteina", "fruta", "verdura", "cereal", "alimentacion", "alimentar"}
PALABRAS_MASCOTA = {"mascota", "animal", "perro", "gato", "corral", "ave", "bovino", "porcino", "ganado"}
PALABRAS_EMBARAZO = {"embaraz", "gestante", "prenatal"}
PALABRAS_BEBE = {"bebe", "recien", "lactante", "neonat"}
PALABRAS_ADULTO_MAYOR = {"anciano", "adulto mayor", "tercera edad", "abuelo", "abuela"}


def interpretar_consulta(q: str) -> ResultadoFiltro:
    """Decide a qué categoría de filtro corresponde la consulta en lenguaje
    natural. Devuelve disponible=False si no matchea ninguna categoría
    conocida (mejor ser honesto que forzar un filtro que no corresponde)."""
    qn = _normalizar(q)
    negado = _hay_negacion(qn)

    vacuna = _detectar_vacuna(qn)
    if vacuna:
        return ResultadoFiltro(True, "vacuna", vacuna, negado)

    enfermedad = _detectar_enfermedad(qn)
    if enfermedad:
        return ResultadoFiltro(True, "enfermedad_cronica", enfermedad, negado)

    if any(p in qn for p in PALABRAS_EMBARAZO):
        return ResultadoFiltro(True, "embarazo", None, negado)
    if any(p in qn for p in PALABRAS_BEBE):
        return ResultadoFiltro(True, "menor_1_anio", None, negado)
    if any(p in qn for p in PALABRAS_ADULTO_MAYOR):
        return ResultadoFiltro(True, "adulto_mayor", None, negado)
    if any(p in qn for p in PALABRAS_NUTRICION):
        return ResultadoFiltro(True, "nutricion", None, negado)
    if any(p in qn for p in PALABRAS_MASCOTA):
        return ResultadoFiltro(True, "mascotas", None, negado)

    lugar = _detectar_geografia(qn)
    if lugar:
        return ResultadoFiltro(True, "direccion", lugar, negado)

    return ResultadoFiltro(False)


def _tiene_vacuna(payload: dict, nombre_vacuna: str) -> bool:
    vacunas = payload.get("vacunacion", {}).get("vacunas", [])
    return any(v.get("vacuna") == nombre_vacuna for v in vacunas)


def _tiene_enfermedad(payload: dict, enfermedad: str) -> bool:
    return any(enfermedad in i.get("enfermedades_cronicas", []) for i in payload.get("integrantes", []))


def _nutricion_baja(payload: dict) -> bool:
    integrantes = payload.get("integrantes", [])
    if not integrantes:
        return False
    prot = sum(i.get("dias_proteina", 7) for i in integrantes) / len(integrantes)
    fv = sum(i.get("dias_frutas_verduras", 7) for i in integrantes) / len(integrantes)
    return prot < UMBRAL_NUTRICION_BAJA or fv < UMBRAL_NUTRICION_BAJA


def _tiene_mascotas(payload: dict) -> bool:
    viv = payload.get("vivienda", {})
    return bool(viv.get("perros_gatos_dentro")) or bool(viv.get("otros_animales"))


def _coincide_geografia(payload: dict, lugar: str) -> bool:
    fam = payload.get("familia", {})
    lugar_norm = _normalizar(lugar)
    return lugar_norm in _normalizar(fam.get("colonia", "")) or lugar_norm in _normalizar(fam.get("calle", ""))


def _evaluar_familia(payload: dict, filtro: ResultadoFiltro) -> tuple[bool, str]:
    """Aplica el filtro interpretado a UNA familia. Devuelve (coincide, detalle_legible)."""
    categoria = filtro.categoria

    if categoria == "vacuna":
        tiene = _tiene_vacuna(payload, filtro.detalle)
        coincide = (not tiene) if filtro.negado else tiene
        detalle = f"{'sin' if filtro.negado else 'con'} registro de {filtro.detalle}"
        return coincide, detalle

    if categoria == "enfermedad_cronica":
        tiene = _tiene_enfermedad(payload, filtro.detalle)
        coincide = (not tiene) if filtro.negado else tiene
        return coincide, f"{'sin' if filtro.negado else 'con'} {filtro.detalle} registrada"

    if categoria == "nutricion":
        baja = _nutricion_baja(payload)
        coincide = (not baja) if filtro.negado else baja
        return coincide, "nutrición baja (proteína/frutas-verduras < 3 días/sem.)" if baja else "nutrición dentro de rango"

    if categoria == "mascotas":
        tiene = _tiene_mascotas(payload)
        coincide = (not tiene) if filtro.negado else tiene
        return coincide, f"{'sin' if filtro.negado else 'con'} mascotas/animales en la vivienda"

    if categoria in ("embarazo", "menor_1_anio", "adulto_mayor"):
        banderas = calcular_banderas(
            payload.get("integrantes", []), payload.get("vacunacion", {}).get("vacunas", [])
        )
        clave = {"embarazo": "tiene_embarazada", "menor_1_anio": "tiene_menor_1_anio",
                 "adulto_mayor": "tiene_adulto_mayor_solo"}[categoria]
        tiene = banderas[clave]
        coincide = (not tiene) if filtro.negado else tiene
        return coincide, clave.replace("tiene_", "").replace("_", " ")

    if categoria == "direccion":
        coincide = _coincide_geografia(payload, filtro.detalle)
        return coincide, f"en/cerca de {filtro.detalle} (misma colonia/calle -- no es distancia real en metros)"

    return False, ""


def buscar_estructurado(q: str, familias: list[dict], k: int = 20) -> dict:
    """Interpreta `q` y filtra la lista de familias (payloads de
    families_full.json). Devuelve {disponible, categoria, resultados: [...]}."""
    filtro = interpretar_consulta(q)
    if not filtro.disponible:
        return {
            "disponible": False,
            "categoria": None,
            "mensaje": (
                "La consulta no coincidió con ninguna categoría soportada. "
                "Prueba con: nombre de vacuna, enfermedad crónica, 'embarazo', "
                "'bebé'/'menor de 1 año', 'adulto mayor', 'nutrición', "
                "'mascotas', o una colonia/calle conocida."
            ),
            "resultados": [],
        }

    resultados = []
    for idx, payload in enumerate(familias):
        coincide, detalle = _evaluar_familia(payload, filtro)
        if coincide:
            fam = payload.get("familia", {})
            resultados.append({
                "familia_id": idx,
                "nombre_informante": fam.get("informante_nombre"),
                "domicilio": f"{fam.get('calle', '')} #{fam.get('numero_exterior', '')}, Col. {fam.get('colonia', '')}",
                "colonia": fam.get("colonia"),
                "localidad": fam.get("localidad"),
                "coincidencia": detalle,
            })

    return {
        "disponible": True,
        "categoria": filtro.categoria,
        "negado": filtro.negado,
        "total_coincidencias": len(resultados),
        "resultados": resultados[:k],
    }
