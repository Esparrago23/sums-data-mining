# -*- coding: utf-8 -*-
"""
corpus_familias.py - Subcomponente C (Motor de Busqueda SUMS)

Cierra el hueco que encontro la auditoria: el corpus de 150 notas que indexan
tfidf_engine/bm25_engine/embeddings_engine (ver corpus_builder.py) es 100%
sintetico e inventado por plantillas, TOTALMENTE DESCONECTADO de las familias
reales/sinteticas de `subcomponente_B_ETL_Risk/data/families_full.json`. Ese
corpus sigue siendo util como banco de pruebas con qrels para demostrar la
tecnica (ver /buscar y /buscar/metricas), pero nunca puede responder "busco
'enfermedad rara' y me regresan las cedulas que aplican" -- no hay forma de
mapear una nota inventada de vuelta a una familia real.

Este modulo construye el corpus a partir del campo `observaciones` REAL de
cada familia (enriquecido en `synthetic_generator.generar_observaciones`),
con el mismo esquema {id, titulo, texto} que ya consumen
`preprocess.preprocesar` (BM25/TF-IDF) y `embeddings_engine.MotorSemantico`
-- no hace falta tocar ninguno de los dos.

CONVENCION DE ID: el "familia_id" ya usado en `buscador_estructurado.py:245`
es simplemente el indice de `enumerate(familias)` (no hay un id explicito en
el dict de families_full.json). Se usa la MISMA convencion aqui para que un
resultado de este corpus (id="37") se pueda mapear 1:1 de vuelta a
`familias[37]` sin ambiguedad.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("sums.mineria.corpus_familias")


def construir_corpus_desde_familias(familias: list[dict]) -> list[dict]:
    """Construye el corpus {id, titulo, texto} a partir de `observaciones` real
    de cada familia (payloads de families_full.json).

    OMITE (no incluye en el corpus) las familias cuyo `observaciones` esta
    vacio, sea None, o sea solo espacios en blanco tras `.strip()` -- no tiene
    sentido indexar un documento sin texto. Se registra (log) cuantas se
    omitieron para poder reportarlo; el llamador tambien puede derivarlo como
    `len(familias) - len(corpus)`.

    Devuelve la lista de {"id": str(idx), "titulo": ..., "texto": ...}, con
    id = indice de enumerate(familias) -- MISMA convencion de "familia_id" que
    ya usa `buscador_estructurado.buscar_estructurado` (buscador_estructurado.py:245),
    para que un resultado de este corpus (id="37") se pueda mapear 1:1 de
    vuelta a `familias[37]` sin ambiguedad."""
    corpus = []
    n_omitidas = 0

    for idx, fam in enumerate(familias):
        observaciones = (fam.get('observaciones') or '').strip()
        if not observaciones:
            n_omitidas += 1
            continue

        datos_familia = fam.get('familia', {})
        titulo = (
            f"{datos_familia.get('calle', '')} #{datos_familia.get('numero_exterior', '')}, "
            f"Col. {datos_familia.get('colonia', '')}"
        )
        corpus.append({
            'id': str(idx),
            'titulo': titulo,
            'texto': observaciones,
        })

    if n_omitidas:
        logger.info(
            "construir_corpus_desde_familias: %d/%d familias omitidas por "
            "observaciones vacias/None/solo-espacios.", n_omitidas, len(familias),
        )

    return corpus
