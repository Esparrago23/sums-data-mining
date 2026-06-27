# -*- coding: utf-8 -*-
"""
tfidf_engine.py - Subcomponente C (Motor de Busqueda SUMS)

Motor TF-IDF + similitud coseno DESDE CERO (Lab 2). Sin scikit-learn ni
librerias de IR para la logica: solo math y Counter.

Funciones puras (reproducen la clave del Lab 2):
  tf(doc)                 -> {termino: frecuencia normalizada}
  idf(corpus)             -> {termino: log(N/df)}
  tfidf(doc, idf_)        -> {termino: tf * idf}
  coseno(v1, v2)          -> similitud coseno (maneja norma cero)

Para indexar y buscar se usa la clase MotorTFIDF, que encapsula el corpus, el
IDF y el INDICE, y expone:
  vectorizar_consulta(texto)  -> vector tf-idf de la consulta (MISMO IDF)
  buscar_tfidf(consulta, k)   -> top-k [(score, id, titulo)]

La consulta se preprocesa con EXACTAMENTE el mismo pipeline (preprocess.preprocesar)
y el MISMO IDF que el corpus. El IDF NO se recalcula incluyendo la consulta.
"""
import math
import operator
from collections import Counter

from preprocess import preprocesar


# ---------------------------------------------------------------------------
# Funciones puras desde cero (Lab 2)
# ---------------------------------------------------------------------------
def tf(doc):
    """Frecuencia de termino normalizada por longitud del documento."""
    n = len(doc)
    if n == 0:
        return {}
    return {t: f / n for t, f in Counter(doc).items()}


def idf(corpus):
    """IDF clasico: log(N / df). corpus = lista de listas de tokens."""
    N = len(corpus)
    df = Counter(t for d in corpus for t in set(d))
    return {t: math.log(N / df[t]) for t in df}


def tfidf(doc, idf_):
    """Vector tf-idf de un documento dado el idf del corpus."""
    return {t: w * idf_.get(t, 0.0) for t, w in tf(doc).items()}


def coseno(v1, v2):
    """Similitud coseno entre dos vectores dispersos (dicts).

    Maneja el caso de NORMA CERO: si cualquiera de las normas es 0 devuelve 0.0
    (sin division por cero)."""
    comunes = set(v1) & set(v2)
    num = sum(v1[t] * v2[t] for t in comunes)
    n1 = math.sqrt(sum(w * w for w in v1.values()))
    n2 = math.sqrt(sum(w * w for w in v2.values()))
    return 0.0 if n1 == 0 or n2 == 0 else num / (n1 * n2)


# ---------------------------------------------------------------------------
# Motor: encapsula corpus, IDF e INDICE (evita estado global fragil).
# ---------------------------------------------------------------------------
class MotorTFIDF:
    """Indexa un corpus procesado y permite buscar por TF-IDF + coseno.

    corpus_procesado: lista de {id, titulo, tokens}."""

    def __init__(self, corpus_procesado):
        self.corpus = corpus_procesado
        self.documentos = [d['tokens'] for d in corpus_procesado]
        self.IDF = idf(self.documentos)               # IDF del corpus (sin consulta)
        self.INDICE = [tfidf(doc, self.IDF) for doc in self.documentos]

    def vectorizar_consulta(self, texto):
        """Vectoriza la consulta con el MISMO preprocesamiento e IDF del corpus."""
        return {t: w * self.IDF.get(t, 0.0)
                for t, w in tf(preprocesar(texto)).items()}

    def buscar_tfidf(self, consulta, k=5):
        """Top-k documentos por similitud coseno. Devuelve [(score, id, titulo)]."""
        q = self.vectorizar_consulta(consulta)
        rank = [(coseno(q, self.INDICE[i]),
                 self.corpus[i]['id'], self.corpus[i]['titulo'])
                for i in range(len(self.corpus))]
        rank.sort(key=lambda x: x[0], reverse=True)
        return rank[:k]

    def top_terminos(self, idx, k=5):
        """Auxiliar didactico: terminos de mayor peso en un documento."""
        return sorted(self.INDICE[idx].items(),
                      key=operator.itemgetter(1), reverse=True)[:k]
