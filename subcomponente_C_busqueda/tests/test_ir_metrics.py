# -*- coding: utf-8 -*-
"""
test_ir_metrics.py - Subcomponente C (Motor de Busqueda SUMS)

Suite de pytest para ir_metrics.py (funciones puras de evaluacion IR):
precision_at_k, recall_at_k, mrr, average_precision, ndcg_at_k.

Convenciones (ver ir_metrics.py):
  - ranking: lista de tuplas (score, id, titulo) YA ORDENADA de mayor a menor.
  - qrels:   dict {consulta: {doc_id: relevancia_graduada 0-3}}.

Cubre los casos borde identificados por el equipo: qrels vacio, ranking mas
corto que k, todos irrelevantes, empates de score (orden dado) y ranking
ideal (nDCG=1.0). Todas las funciones devuelven valores en [0, 1].
"""
from ir_metrics import (
    average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _assert_en_rango(valor):
    assert 0.0 <= valor <= 1.0


# ---------------------------------------------------------------------------
# Caso borde 1: qrels con el set de relevantes VACIO para la consulta.
# ---------------------------------------------------------------------------
def test_recall_at_k_sin_relevantes_devuelve_cero_sin_dividir_por_cero():
    ranking = [(0.9, "d1", "t1"), (0.5, "d2", "t2")]
    qrels = {"q1": {}}  # ningun documento relevante para q1

    resultado = recall_at_k(ranking, qrels, "q1", k=5)

    assert resultado == 0.0
    _assert_en_rango(resultado)


def test_average_precision_sin_relevantes_devuelve_cero():
    ranking = [(0.9, "d1", "t1"), (0.5, "d2", "t2")]
    qrels = {"q1": {}}

    resultado = average_precision(ranking, qrels, "q1")

    assert resultado == 0.0
    _assert_en_rango(resultado)


def test_ndcg_at_k_sin_relevantes_devuelve_cero_rama_idcg_cero():
    ranking = [(0.9, "d1", "t1"), (0.5, "d2", "t2")]
    qrels = {"q1": {}}  # ideal = [] -> idcg = 0

    resultado = ndcg_at_k(ranking, qrels, "q1", k=5)

    assert resultado == 0.0
    _assert_en_rango(resultado)


# ---------------------------------------------------------------------------
# Caso borde 2: ranking MAS CORTO que k. precision_at_k divide por el k FIJO,
# no por len(ranking). Documentamos este comportamiento explicitamente.
# ---------------------------------------------------------------------------
def test_precision_at_k_ranking_mas_corto_que_k_divide_por_k_fijo():
    # Solo 2 documentos en el ranking, pero k=5.
    ranking = [(0.9, "d1", "t1"), (0.5, "d2", "t2")]
    qrels = {"q1": {"d1": 3, "d2": 0}}  # solo d1 es relevante

    resultado = precision_at_k(ranking, qrels, "q1", k=5)

    # 1 acierto (d1) / k=5 FIJO == 0.2 (NO 1/len(ranking)=0.5,
    # ni 1/2 elementos disponibles: la funcion no re-escala por len(ranking)).
    esperado = 1 / 5
    assert resultado == esperado
    assert resultado != 1 / len(ranking)  # documenta que NO usa len(ranking)
    _assert_en_rango(resultado)


# ---------------------------------------------------------------------------
# Caso borde 3: TODOS los documentos del ranking son irrelevantes (relevancia
# 0 explicita o ausentes de qrels), pero SI existe un relevante fuera del
# ranking (para que idcg > 0 y se ejercite la rama dcg=0/idcg>0 de nDCG).
# ---------------------------------------------------------------------------
def test_todos_irrelevantes_en_el_ranking_mrr_ap_ndcg_dan_cero():
    ranking = [(0.9, "d1", "t1"), (0.5, "d2", "t2"), (0.1, "d3", "t3")]
    # d1 relevancia 0 explicita, d2/d3 ausentes (=0 implicito).
    # "otro" es relevante pero nunca aparece en el ranking.
    qrels = {"q1": {"d1": 0, "otro": 3}}

    r_mrr = mrr(ranking, qrels, "q1")
    r_ap = average_precision(ranking, qrels, "q1")
    r_ndcg = ndcg_at_k(ranking, qrels, "q1", k=5)

    assert r_mrr == 0.0
    assert r_ap == 0.0
    assert r_ndcg == 0.0
    for r in (r_mrr, r_ap, r_ndcg):
        _assert_en_rango(r)


# ---------------------------------------------------------------------------
# Caso borde 4: EMPATES de score. mrr debe respetar el ORDEN dado en la
# lista (primer relevante segun el orden recibido), sin reordenar por su
# cuenta (p. ej. por id).
# ---------------------------------------------------------------------------
def test_mrr_con_empates_de_score_respeta_el_orden_recibido():
    # Mismo score para los 3 documentos: si la funcion reordenara (p. ej.
    # alfabeticamente por id), "a_doc" quedaria primero y mrr seria 1.0.
    # El orden RECIBIDO pone "z_doc" primero (irrelevante) y "a_doc" segundo
    # (relevante) -> mrr debe ser 1/2, no 1.0.
    ranking = [(0.5, "z_doc", "t1"), (0.5, "a_doc", "t2"), (0.5, "m_doc", "t3")]
    qrels = {"q1": {"a_doc": 2}}

    resultado = mrr(ranking, qrels, "q1")

    assert resultado == 1 / 2
    _assert_en_rango(resultado)


# ---------------------------------------------------------------------------
# Caso borde 5: ranking PERFECTO (coincide exactamente con el orden ideal
# por relevancia graduada) -> ndcg_at_k debe dar 1.0.
# ---------------------------------------------------------------------------
def test_ndcg_at_k_ranking_perfecto_da_uno():
    qrels = {"q1": {"d1": 3, "d2": 2, "d3": 1}}
    # Orden del ranking EXACTAMENTE igual al orden ideal por relevancia.
    ranking = [(0.9, "d1", "t1"), (0.6, "d2", "t2"), (0.3, "d3", "t3")]

    resultado = ndcg_at_k(ranking, qrels, "q1", k=3)

    assert resultado == 1.0
    _assert_en_rango(resultado)
