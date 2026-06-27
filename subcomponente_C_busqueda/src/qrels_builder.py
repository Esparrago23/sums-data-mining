# -*- coding: utf-8 -*-
"""
qrels_builder.py - Subcomponente C (Motor de Busqueda SUMS)

Deriva data/qrels_sums.json desde la VERDAD del corpus (data/corpus_themes.json).
No se etiquetan documentos "a ojo": cada consulta se ancla a un tema PRINCIPAL y
(en consultas compuestas, las realistas en salud) a un CO-TEMA. Los documentos
relevantes son EXACTAMENTE los que el generador marco con esos temas, y la
relevancia se gradua por CENTRALIDAD del tema y por la presencia del co-tema:

  - 3 (muy relevante): el tema principal es el DOMINANTE de la nota (posicion 0)
                       Y ademas aparece el co-tema -> calce perfecto de la consulta.
  - 2 (relevante):     el tema principal es el dominante (sin co-tema), o bien
                       aparece el principal (no dominante) junto con el co-tema.
  - 1 (marginal):      el tema principal aparece pero no es dominante y no hay co-tema.
  - 0:                 el tema principal no aparece.

Anclar a co-temas hace que el conjunto relevante sea acotado y GRADUADO de
verdad (no "todo lo que toca el tema"), de modo que el ORDEN importa y las
metricas (nDCG@5, MAP) discriminan entre TF-IDF y BM25.

Cada consulta queda garantizada con >=2 documentos de grado >0 presentes en el
corpus. 8 consultas en lenguaje natural de salud (>=5 que pide la rubrica).
"""
import json
import os

# Cada consulta -> (tema_principal, co_tema).
# El co_tema codifica el matiz de la consulta (p. ej. desnutricion EN VIVIENDA
# precaria, dengue CON problema de agua, etc.), tal como lo redactaria un
# promotor de salud que busca casos compuestos.
CONSULTAS = {
    'familias con desnutricion infantil y vivienda precaria':
        ('nutricion', 'vivienda'),
    'casos sospechosos de dengue con problemas de agua':
        ('vector', 'agua'),
    'viviendas sin agua entubada con menores desnutridos':
        ('agua', 'nutricion'),
    'adultos mayores que viven solos con enfermedad cronica':
        ('adulto_mayor', 'cronica'),
    'embarazadas sin control prenatal y esquema de vacunacion incompleto':
        ('embarazo', 'vacunacion'),
    'diabetes o hipertension sin control con consumo de alcohol':
        ('cronica', 'toxico'),
    'esquema de vacunacion incompleto con riesgo de dengue':
        ('vacunacion', 'vector'),
    'consumo de alcohol o tabaquismo en adulto mayor':
        ('toxico', 'adulto_mayor'),
}


def _grado(temas_doc, principal, co):
    """Gradua la relevancia de una nota para una consulta (principal, co)."""
    if principal not in temas_doc:
        return 0
    dominante = temas_doc[0] == principal
    tiene_co = co in temas_doc
    if dominante and tiene_co:
        return 3            # calce perfecto: tema central + matiz de la consulta
    if dominante or tiene_co:
        return 2            # central solo, o presente + matiz
    return 1                # presente pero secundario y sin matiz


def derivar(themes):
    """Construye el dict de qrels {consulta: {doc_id: grado}} desde themes."""
    qrels = {}
    for consulta, (principal, co) in CONSULTAS.items():
        juicios = {}
        for doc_id, temas_doc in themes.items():
            g = _grado(temas_doc, principal, co)
            if g > 0:
                juicios[doc_id] = g
        qrels[consulta] = juicios
    return qrels


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(aqui, '..', 'data')
    ruta_themes = os.path.join(data_dir, 'corpus_themes.json')
    ruta_qrels = os.path.join(data_dir, 'qrels_sums.json')

    with open(ruta_themes, encoding='utf-8') as fh:
        themes = json.load(fh)

    qrels = derivar(themes)

    # Verificacion dura: cada consulta con >=2 relevantes presentes y al menos
    # un grado 3 (asi la relevancia esta realmente graduada).
    for consulta, juicios in qrels.items():
        relevantes = [d for d, g in juicios.items() if g > 0]
        assert len(relevantes) >= 2, \
            f'La consulta "{consulta}" tiene <2 relevantes'
        assert any(g == 3 for g in juicios.values()), \
            f'La consulta "{consulta}" no tiene ningun grado 3'
        for d in juicios:
            assert d in themes, f'qrels refiere id inexistente: {d}'

    with open(ruta_qrels, 'w', encoding='utf-8') as fh:
        json.dump(qrels, fh, ensure_ascii=False, indent=2)

    print(f'{len(qrels)} consultas -> {ruta_qrels}')
    for consulta, juicios in qrels.items():
        n3 = sum(1 for g in juicios.values() if g == 3)
        n2 = sum(1 for g in juicios.values() if g == 2)
        n1 = sum(1 for g in juicios.values() if g == 1)
        print(f'  rel={len(juicios):2} (g3={n3:2} g2={n2:2} g1={n1:2})  {consulta}')
    return qrels


if __name__ == '__main__':
    main()
