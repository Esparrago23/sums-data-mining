# -*- coding: utf-8 -*-
"""
ir_metrics.py - Subcomponente C (Motor de Busqueda SUMS)

Las 5 metricas de Recuperacion de Informacion DESDE CERO (Lab 3, Parte B):
  precision_at_k, recall_at_k, mrr, average_precision, ndcg_at_k

Convenciones:
  - ranking: lista de tuplas (score, id, titulo) ya ordenada de mayor a menor.
  - qrels:   dict {consulta: {doc_id: relevancia_graduada}} con relevancia
             graduada 0-3 (3 = muy relevante, ausente = 0).
  - qid:     la clave de la consulta dentro de qrels (es el propio texto).

Todas devuelven valores en [0, 1].
"""
import math


def _rel(qrels, qid, doc):
    """Relevancia graduada de un documento para una consulta (0 si ausente)."""
    return qrels[qid].get(doc, 0)


def _relevantes(qrels, qid):
    """Conjunto de documentos con relevancia > 0 para la consulta."""
    return {d for d, g in qrels[qid].items() if g > 0}


def precision_at_k(ranking, qrels, qid, k=5):
    """Fraccion de los top-k que son relevantes."""
    rel = _relevantes(qrels, qid)
    top = [d for _, d, _ in ranking[:k]]
    return sum(1 for d in top if d in rel) / k


def recall_at_k(ranking, qrels, qid, k=5):
    """Fraccion de los relevantes recuperados en los top-k."""
    rel = _relevantes(qrels, qid)
    if not rel:
        return 0.0
    top = [d for _, d, _ in ranking[:k]]
    return sum(1 for d in top if d in rel) / len(rel)


def mrr(ranking, qrels, qid):
    """Reciproco del rango del primer documento relevante."""
    rel = _relevantes(qrels, qid)
    for i, (_, d, _) in enumerate(ranking, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranking, qrels, qid):
    """Promedio de la precision en cada acierto (base de MAP)."""
    rel = _relevantes(qrels, qid)
    if not rel:
        return 0.0
    hits, suma = 0, 0.0
    for i, (_, d, _) in enumerate(ranking, start=1):
        if d in rel:
            hits += 1
            suma += hits / i
    return suma / len(rel)


def ndcg_at_k(ranking, qrels, qid, k=5):
    """nDCG@k con relevancia graduada: DCG / IDCG."""
    dcg = sum(_rel(qrels, qid, d) / math.log2(i + 1)
              for i, (_, d, _) in enumerate(ranking[:k], start=1))
    ideal = sorted(qrels[qid].values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
