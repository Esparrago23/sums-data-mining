# -*- coding: utf-8 -*-
"""
bm25_engine.py - Subcomponente C (Motor de Busqueda SUMS)

Motor BM25 DESDE CERO (Lab 3, Parte A). Sin librerias de IR para la logica.

Formula de la clase:
  score(d,q) = sum_{t in q} IDF(t) * ( f*(k1+1) ) / ( f + k1*(1 - b + b*|d|/avgdl) )

con IDF de BM25 suavizado (variante que nunca es negativa):
  IDF(t) = ln( 1 + (N - df + 0.5) / (df + 0.5) )

Parametros:
  k1 controla la saturacion de la frecuencia de termino.
  b  controla la penalizacion por longitud de documento (b=0 la ignora).

La consulta se preprocesa con el MISMO pipeline (preprocess.preprocesar).
"""
import math
from collections import Counter

from preprocess import preprocesar


def avgdl(documentos):
    """Longitud media de documento sobre TODO el corpus."""
    n = len(documentos)
    return sum(len(d) for d in documentos) / n if n else 0.0


def idf_bm25(corpus):
    """IDF de BM25 suavizado (nunca negativo). corpus = listas de tokens."""
    N = len(corpus)
    df = Counter(t for d in corpus for t in set(d))
    return {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in df}


def bm25(doc, q_tokens, idf_bm25_, avgdl_, k1=1.5, b=0.75):
    """Score BM25 de un documento (lista de tokens) frente a la consulta
    (lista de tokens). Acumula termino por termino segun la formula."""
    score = 0.0
    freqs = Counter(doc)
    dl = len(doc)
    for t in q_tokens:
        f = freqs.get(t, 0)
        if f == 0:
            continue
        idf_t = idf_bm25_.get(t, 0.0)
        score += idf_t * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl_))
    return score


class MotorBM25:
    """Indexa un corpus procesado y permite buscar por BM25.

    corpus_procesado: lista de {id, titulo, tokens}."""

    def __init__(self, corpus_procesado):
        self.corpus = corpus_procesado
        self.documentos = [d['tokens'] for d in corpus_procesado]
        self.avgdl = avgdl(self.documentos)
        self.IDF_BM25 = idf_bm25(self.documentos)

    def buscar_bm25(self, consulta, k=5, k1=1.5, b=0.75):
        """Top-k documentos por score BM25. Devuelve [(score, id, titulo)]."""
        q = preprocesar(consulta)
        rank = [(bm25(self.documentos[i], q, self.IDF_BM25, self.avgdl, k1, b),
                 self.corpus[i]['id'], self.corpus[i]['titulo'])
                for i in range(len(self.corpus))]
        rank.sort(key=lambda x: x[0], reverse=True)
        return rank[:k]
