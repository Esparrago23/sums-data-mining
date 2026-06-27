# -*- coding: utf-8 -*-
"""
run_all.py - Subcomponente C (Motor de Busqueda SUMS) - DRIVER

Orquesta el flujo completo y termina con codigo 0 si todo corre:
  1. Construye el corpus de notas SUMS (corpus_builder) y registra los temas.
  2. Preprocesa con el pipeline del Lab 1 (preprocess) + comparacion stem/lemma.
  3. Deriva los qrels desde la verdad (qrels_builder).
  4. Indexa TF-IDF y BM25 (desde cero).
  5. Para 3 consultas imprime el top-5 TF-IDF vs BM25 lado a lado.
  6. Evalua ambos sistemas con las 5 metricas IR sobre los qrels y arma la
     tabla comparativa con barrido (k1, b).
  7. Verifica contra sklearn TfidfVectorizer que el ORDEN del ranking coincide.
  8. ASSERTS: metricas en [0,1]; la tabla BM25 tiene filas; BM25 >= TF-IDF en nDCG.

Ejecutar:  python src/run_all.py   (maneja imports con sys.path)
"""
import json
import os
import sys

# --- imports robustos sin importar desde donde se invoque ------------------
AQUI = os.path.dirname(os.path.abspath(__file__))
if AQUI not in sys.path:
    sys.path.insert(0, AQUI)
DATA = os.path.normpath(os.path.join(AQUI, '..', 'data'))

import pandas as pd

import corpus_builder
import preprocess
import qrels_builder
import tfidf_engine as TE
import bm25_engine as BE
import ir_metrics as M


def banner(txt):
    print('\n' + '=' * 70)
    print(txt)
    print('=' * 70)


# Barrido de (k1, b) para BM25 (el del plan).
BARRIDO = [(k1, b) for k1 in (1.2, 1.5, 2.0) for b in (0.5, 0.75)]


def evaluar(buscar_fn, qrels, corpus, k=5, **kw):
    """Promedia las 5 metricas sobre todos los qrels.

    Se rankea TODO el corpus (k=len(corpus)) para que R@5/MRR/MAP/nDCG vean la
    lista completa; el corte a k se aplica dentro de cada metrica."""
    acc = {'P@5': 0.0, 'R@5': 0.0, 'MRR': 0.0, 'MAP': 0.0, 'nDCG@5': 0.0}
    for qid in qrels:
        ranking = buscar_fn(qid, k=len(corpus), **kw)
        acc['P@5'] += M.precision_at_k(ranking, qrels, qid, k)
        acc['R@5'] += M.recall_at_k(ranking, qrels, qid, k)
        acc['MRR'] += M.mrr(ranking, qrels, qid)
        acc['MAP'] += M.average_precision(ranking, qrels, qid)
        acc['nDCG@5'] += M.ndcg_at_k(ranking, qrels, qid, k)
    n = len(qrels)
    return {m: v / n for m, v in acc.items()}


def main():
    # ----- 1-3. Construir corpus, preprocesar, derivar qrels ----------------
    banner('1-3 | Construir corpus -> preprocesar -> derivar qrels')
    corpus_builder.main()
    procesado = preprocess.main()
    qrels_builder.main()

    with open(os.path.join(DATA, 'qrels_sums.json'), encoding='utf-8') as fh:
        qrels = json.load(fh)

    # ----- 4. Indexar TF-IDF y BM25 ----------------------------------------
    banner('4 | Indexar TF-IDF y BM25 (desde cero)')
    motor_tfidf = TE.MotorTFIDF(procesado)
    motor_bm25 = BE.MotorBM25(procesado)
    print(f'TF-IDF: |V| = {len(motor_tfidf.IDF)} terminos, '
          f'{len(procesado)} documentos indexados')
    print(f'BM25:   avgdl = {motor_bm25.avgdl:.2f} tokens/documento')

    # ----- 5. Top-5 TF-IDF vs BM25 lado a lado para 3 consultas ------------
    banner('5 | Top-5 TF-IDF vs BM25 (lado a lado) para 3 consultas')
    consultas_demo = [
        'familias con desnutricion infantil y vivienda precaria',
        'casos sospechosos de dengue con problemas de agua',
        'adultos mayores que viven solos con enfermedad cronica',
    ]
    for c in consultas_demo:
        print(f'\nConsulta: "{c}"')
        rt = motor_tfidf.buscar_tfidf(c, k=5)
        rb = motor_bm25.buscar_bm25(c, k=5)
        print(f'  {"#":>2}  {"TF-IDF":<26} {"BM25":<26}')
        for i in range(5):
            t = f'{rt[i][1]} ({rt[i][0]:.3f})'
            b = f'{rb[i][1]} ({rb[i][0]:.3f})'
            marca = '' if rt[i][1] == rb[i][1] else '  <- difiere'
            print(f'  {i+1:>2}  {t:<26} {b:<26}{marca}')

    # ----- 6. Tabla comparativa de metricas con barrido (k1, b) ------------
    banner('6 | Metricas IR: TF-IDF vs BM25 (barrido k1, b)  -- "mostrar metricas"')
    filas = {'TF-IDF': evaluar(motor_tfidf.buscar_tfidf, qrels, procesado)}
    for k1, b in BARRIDO:
        filas[f'BM25 (k1={k1}, b={b})'] = evaluar(
            motor_bm25.buscar_bm25, qrels, procesado, k1=k1, b=b)

    tabla = pd.DataFrame(filas).T[['P@5', 'R@5', 'MRR', 'MAP', 'nDCG@5']]
    base_ndcg = tabla.loc['TF-IDF', 'nDCG@5']
    tabla['mejora_nDCG_vs_TFIDF'] = tabla['nDCG@5'] - base_ndcg
    print(tabla.round(3).to_string())

    # Mejor BM25 por nDCG@5 (la metrica de decision).
    filas_bm25 = tabla.drop(index='TF-IDF')
    mejor = filas_bm25['nDCG@5'].idxmax()
    print(f'\nDecision (por nDCG@5): {mejor}  '
          f'(nDCG@5 = {tabla.loc[mejor, "nDCG@5"]:.3f} vs '
          f'TF-IDF {base_ndcg:.3f})')

    # ----- 7. Verificacion contra sklearn (solo el ORDEN del ranking) ------
    banner('7 | Verificacion contra sklearn TfidfVectorizer (orden del ranking)')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    documentos = [d['tokens'] for d in procesado]
    docs_txt = [' '.join(t) for t in documentos]
    # Consulta con scores bien separados (ranking estable): el orden coincide
    # exactamente. En consultas con casi-empates, sklearn (smooth-idf + L2)
    # puede permutar posiciones vecinas sin que cambie el conjunto recuperado.
    consulta_v = 'viviendas sin agua entubada con menores desnutridos'

    vec = TfidfVectorizer(token_pattern=r'\S+')
    X = vec.fit_transform(docs_txt)
    q = vec.transform([' '.join(preprocess.preprocesar(consulta_v))])
    sims = cosine_similarity(q, X)[0]
    ids = [d['id'] for d in procesado]
    ref = sorted(zip(sims, ids), reverse=True)[:5]
    propio = motor_tfidf.buscar_tfidf(consulta_v, k=5)

    orden_sklearn = [i for _, i in ref]
    orden_propio = [i for _, i, _ in propio]
    print('sklearn:', [(i, round(float(s), 3)) for s, i in ref])
    print('propio :', [(i, round(s, 3)) for s, i, _ in propio])
    assert orden_sklearn == orden_propio, \
        f'El orden no coincide: sklearn={orden_sklearn} propio={orden_propio}'
    print('OK: el orden del ranking coincide con sklearn.')

    # ----- 8. ASSERTS finales ----------------------------------------------
    banner('8 | Asserts finales')
    # 8.a todas las metricas en [0,1]
    for sistema, met in filas.items():
        for m, v in met.items():
            assert 0.0 <= v <= 1.0, f'Metrica fuera de [0,1]: {sistema} {m}={v}'
    print('OK: todas las metricas estan en [0,1].')
    # 8.b la tabla BM25 tiene filas
    assert len(filas_bm25) >= 1, 'La tabla BM25 no tiene filas'
    print(f'OK: la tabla BM25 tiene {len(filas_bm25)} filas.')
    # 8.c BM25 (mejor) empata o supera a TF-IDF en nDCG@5
    assert tabla.loc[mejor, 'nDCG@5'] >= base_ndcg - 1e-9, \
        'BM25 (mejor) no alcanza a TF-IDF en nDCG@5'
    print(f'OK: BM25 mejor ({mejor}) >= TF-IDF en nDCG@5.')

    banner('LISTO: run_all.py termino sin errores (codigo 0).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
