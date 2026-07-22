# -*- coding: utf-8 -*-
"""
embeddings_engine.py - Subcomponente C (Motor de Busqueda SUMS)

Tercer motor de busqueda: BUSQUEDA SEMANTICA con embeddings Sentence-BERT
(mejora 3.1 de la auditoria: los motores lexicos -TF-IDF y BM25, "desde cero"
en tfidf_engine.py / bm25_engine.py- solo matchean TOKENS COMPARTIDOS. No
encuentran sinonimos reales del dominio de salud: una consulta como
"azucar alta" nunca recupera una nota que dice "diabetes", ni "presion"
recupera una nota que dice "hipertension", porque no comparten ni una sola
raiz lexica. Un modelo de embeddings entrenado para similitud semantica SI
captura esa relacion porque compara SIGNIFICADO, no tokens.

A diferencia de MotorTFIDF/MotorBM25, este motor SI usa una libreria externa
(sentence-transformers) para la logica de embeddings: no tiene sentido
reimplementar un transformer "desde cero" para este entregable, y la
consigna del Lab 3 solo pedia TF-IDF/BM25 "desde cero".

Modelo: "hiiamsid/sentence_similarity_spanish_es" (Sentence-BERT en espanol,
~420MB, ya validado por el equipo). Se descarga una sola vez (cache de
huggingface) la primera vez que se instancia MotorSemantico en una maquina.

-----------------------------------------------------------------------------
Por que este motor NO usa preprocess.preprocesar() (lematizacion)
-----------------------------------------------------------------------------
Decision INTENCIONAL, no un descuido:

  1. Un modelo Sentence-BERT esta entrenado para producir embeddings de
     ORACIONES/FRASES COMPLETAS en lenguaje natural (aprendio la sintaxis y
     la semantica de esas oraciones durante el preentrenamiento). Lematizar
     destruye esa estructura (conjugaciones, genero, numero, orden) y le
     entrega al modelo una bolsa de lemas sueltos -una entrada fuera de la
     distribucion con la que fue entrenado-, lo que en la practica DEGRADA
     la calidad del embedding en lugar de mejorarla. TF-IDF/BM25 SI necesitan
     lematizar porque su unidad de match es el TOKEN exacto (lematizar reduce
     la dispersion lexica); un transformer ya generaliza morfologia y
     sinonimia dentro de su propio espacio vectorial, por lo que ese paso es
     innecesario y contraproducente aqui.
  2. Evita heredar cualquier bug del pipeline de lematizacion (spaCy +
     stopwords + manejo de negaciones) en un motor que no lo necesita para
     funcionar: menos superficie de acoplamiento entre motores independientes.

Por eso este motor indexa el texto CRUDO del corpus (campo "texto"), pasado
como mucho por preprocess.normalizar(texto, quitar_acentos=False) -minusculas,
NFC, sin URLs/HTML, espacios colapsados- que es normalizacion ligera, NO
lematizacion, y no cambia el significado de la oracion.
"""
from __future__ import annotations

import shutil

# ---------------------------------------------------------------------------
# Import opcional: sentence-transformers (y torch, que arrastra) puede no
# estar instalado, o la maquina puede no tener internet para bajar el modelo
# la primera vez. Este modulo se importa SIEMPRE al arrancar la API
# (api_mineria.py hace `from embeddings_engine import MotorSemantico` a nivel
# de modulo), asi que el import de la libreria NUNCA debe reventar aqui: si
# falla, se captura y MotorSemantico.__init__ levanta RuntimeError al
# intentar instanciarse (api_mineria.py ya envuelve eso en try/except).
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depende de la maquina
    SentenceTransformer = None
    _IMPORT_ERROR = exc

from preprocess import normalizar

MODELO = 'hiiamsid/sentence_similarity_spanish_es'

# Umbral minimo de espacio libre en disco tras cargar/descargar el modelo
# (~420MB el modelo + margen para cache de huggingface y torch).
MIN_LIBRE_GB = 1.5


def _gb_libres(ruta='C:/'):
    """GB libres en la unidad de `ruta` (redondeado)."""
    return shutil.disk_usage(ruta).free / (1024 ** 3)


class MotorSemantico:
    """Indexa un corpus CRUDO (sin tokenizar/lematizar) con embeddings
    Sentence-BERT y permite buscar por similitud coseno.

    corpus_crudo: lista de {id, titulo, texto} (ver docstring del modulo:
    aqui NO se usa preprocess.preprocesar/lematizacion)."""

    def __init__(self, corpus_crudo):
        if SentenceTransformer is None:
            raise RuntimeError(
                "MotorSemantico no disponible: falta instalar 'sentence-transformers' "
                "(pip install sentence-transformers). Error original de import: "
                f"{_IMPORT_ERROR!r}"
            )

        # --- verificacion de espacio en disco ANTES de cargar/descargar ----
        libres_antes = _gb_libres()
        if libres_antes < MIN_LIBRE_GB:
            raise RuntimeError(
                f"MotorSemantico no disponible: solo quedan {libres_antes:.2f} GB "
                f"libres en disco (minimo requerido {MIN_LIBRE_GB} GB). Libera "
                "espacio antes de cargar el modelo Sentence-BERT (~420MB + cache)."
            )

        try:
            self.model = SentenceTransformer(MODELO)
        except Exception as exc:  # descarga/carga fallida (sin internet, HF caido, etc.)
            raise RuntimeError(
                f"MotorSemantico no disponible: fallo al cargar/descargar el modelo "
                f"'{MODELO}' (sentence-transformers). Verifique conexion a internet "
                f"o la cache local de huggingface. Error original: {exc!r}"
            ) from exc

        # --- verificacion de espacio en disco DESPUES de cargar/descargar --
        libres_despues = _gb_libres()
        if libres_despues < MIN_LIBRE_GB:
            raise RuntimeError(
                f"MotorSemantico no disponible: tras cargar el modelo solo quedan "
                f"{libres_despues:.2f} GB libres en disco (minimo requerido "
                f"{MIN_LIBRE_GB} GB). Libere espacio antes de usar este motor."
            )

        self.corpus = corpus_crudo
        self.ids = [d['id'] for d in corpus_crudo]
        self.titulos = [d['titulo'] for d in corpus_crudo]

        if not corpus_crudo:
            # Corpus vacio: matriz de embeddings vacia, sin llamar a encode()
            # con una lista vacia (algunas versiones de sentence-transformers
            # no manejan bien ese caso) y sin crashear en buscar_semantico.
            import numpy as np
            self.embeddings = np.zeros((0, 0), dtype='float32')
            return

        textos = [normalizar(d.get('texto', ''), quitar_acentos=False)
                  for d in corpus_crudo]
        self.embeddings = self.model.encode(
            textos,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def buscar_semantico(self, consulta: str, k: int = 5) -> list[tuple[float, str, str]]:
        """Top-k documentos por similitud coseno de embeddings.

        Devuelve [(score, id, titulo), ...] ordenado descendente, misma forma
        que MotorBM25.buscar_bm25 / MotorTFIDF.buscar_tfidf."""
        if not self.corpus:
            return []

        q_texto = normalizar(consulta, quitar_acentos=False)
        q_emb = self.model.encode(
            [q_texto],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        # Embeddings y consulta ya normalizados (norma 1) -> producto punto
        # equivale a similitud coseno.
        scores = self.embeddings @ q_emb

        rank = [(float(scores[i]), self.ids[i], self.titulos[i])
                for i in range(len(self.corpus))]
        rank.sort(key=lambda x: x[0], reverse=True)
        return rank[:k]
