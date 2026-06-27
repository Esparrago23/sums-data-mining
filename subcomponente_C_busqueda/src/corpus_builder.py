# -*- coding: utf-8 -*-
"""
corpus_builder.py - Subcomponente C (Motor de Busqueda SUMS)

Genera ~150 notas de observacion / visita domiciliaria sinteticas del dominio
SUMS (Suchiapa), con formato {id, titulo, texto}. Es DETERMINISTICO
(random.Random(42)) para que el corpus, los qrels y las metricas sean
reproducibles.

Ademas del corpus crudo, emite data/corpus_themes.json (id -> [temas]) que
registra, EN ORDEN DE DOMINANCIA, los temas que generaron cada nota. Ese
registro es la "verdad" (ground truth) desde la cual se derivan los qrels:
el primer tema es el dominante (mas central) y se gradua mas alto.

El banco de frases parte del plan (seccion Subcomponente C) pero se AMPLIA con
varias frases por tema para que el vocabulario sea rico y los rankings de
TF-IDF y BM25 difieran (BM25 normaliza por longitud de documento; TF-IDF no).
"""
import json
import os
import random

# ---------------------------------------------------------------------------
# Banco de frases por dimension de riesgo (lenguaje real de visita domiciliaria).
# Cada tema tiene varias variantes con vocabulario distinto para enriquecer el
# corpus y separar los rankings de TF-IDF vs BM25.
# ---------------------------------------------------------------------------
FRASES = {
    'vivienda': [
        "vivienda con piso de tierra y techo de lamina",
        "casa de adobe en mal estado, paredes con humedad y hacinamiento",
        "hogar de un solo cuarto donde duermen seis personas, hacinamiento severo",
        "vivienda sin drenaje, letrina en mal estado en el patio trasero",
        "techo de carton y lamina con goteras durante la temporada de lluvias",
    ],
    'agua': [
        "sin agua entubada, acarrean agua del pozo comunitario",
        "servicio de agua intermitente; almacenan en tambos sin tapa",
        "no cuentan con agua potable, hierven el agua antes de beberla",
        "viviendas sin agua entubada que dependen de pipas una vez por semana",
        "almacenamiento de agua en cubetas destapadas, riesgo de contaminacion",
    ],
    'cronica': [
        "adulto mayor con diabetes sin control glucemico",
        "paciente con hipertension arterial y sobrepeso, sin tratamiento",
        "diabetico que abandono el tratamiento por falta de medicamento",
        "hipertension no controlada, no acude a sus citas de seguimiento",
        "paciente cronico con diabetes e hipertension, requiere control mensual",
    ],
    'nutricion': [
        "menor de 3 anos con bajo peso para la edad",
        "dieta baja en proteina; consumo de frutas y verduras menor a tres dias por semana",
        "nino con desnutricion cronica y talla baja para su edad",
        "lactante con bajo peso y signos de desnutricion infantil",
        "familia con inseguridad alimentaria, dos menores con peso bajo",
    ],
    'vacunacion': [
        "esquema de vacunacion incompleto en dos menores",
        "falta refuerzo de tetanos en mujer gestante",
        "cartilla de vacunacion atrasada, faltan dosis de sarampion",
        "menor sin vacuna contra influenza ni refuerzo pendiente",
        "esquema de inmunizacion incompleto, se agenda visita de seguimiento",
    ],
    'embarazo': [
        "embarazada sin control prenatal en el primer trimestre",
        "gestante de 19 anos, primer embarazo, atencion en el hogar",
        "mujer embarazada sin consultas prenatales ni acido folico",
        "gestante con embarazo de alto riesgo, requiere referencia a hospital",
        "embarazada en tercer trimestre sin ultrasonido ni control medico",
    ],
    'vector': [
        "criaderos de mosco en el patio; riesgo de dengue",
        "caso sospechoso de dengue con fiebre alta en un menor",
        "abundancia de larvas de mosquito en cacharros, foco de dengue",
        "paciente con fiebre, dolor de cuerpo y sospecha de dengue",
        "patio con cacharros y llantas que acumulan agua, criadero de mosco",
    ],
    'toxico': [
        "jefe de familia con consumo de alcohol frecuente",
        "tabaquismo activo en dos integrantes del nucleo familiar",
        "consumo nocivo de alcohol que afecta la dinamica familiar",
        "fumador cronico con tos persistente, riesgo respiratorio",
        "alcoholismo en el hogar asociado a violencia intrafamiliar",
    ],
    'adulto_mayor': [
        "adulto mayor que vive solo, con movilidad reducida",
        "persona de 78 anos con artritis, requiere visita domiciliaria periodica",
        "anciano que vive solo sin redes de apoyo cercanas",
        "adulto mayor postrado en cama que depende de cuidador",
        "persona mayor con caidas frecuentes y riesgo de fractura",
    ],
}

# Colonias y localidades de Suchiapa para contextualizar la nota.
COLONIAS = ['Centro', 'San Jose', 'Pacu', 'San Jacinto', 'San Antonio',
            'San Roque', 'Las Palmas', 'Independencia', 'El Calvario']


def construir_nota(i, rng):
    """Construye una nota {id, titulo, texto} y devuelve tambien sus temas en
    orden de dominancia (el primero es el tema central de la nota)."""
    temas = rng.sample(list(FRASES), k=rng.randint(2, 3))
    # Cada tema aporta una frase (vocabulario distinto por variante).
    cuerpo = '. '.join(rng.choice(FRASES[t]) for t in temas)
    colonia = rng.choice(COLONIAS)
    texto = f"Visita domiciliaria en colonia {colonia}, Suchiapa. {cuerpo}."
    nota = {
        'id': f'n{i:03d}',
        'titulo': f'Nota de visita {i:03d} - {temas[0]}',
        'texto': texto,
    }
    return nota, temas


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(aqui, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    rng = random.Random(42)  # determinismo
    corpus = []
    themes = {}
    for i in range(1, 151):  # ~150 notas
        nota, temas = construir_nota(i, rng)
        corpus.append(nota)
        themes[nota['id']] = temas  # orden = dominancia

    ruta_corpus = os.path.join(data_dir, 'corpus_crudo_sums.json')
    ruta_themes = os.path.join(data_dir, 'corpus_themes.json')
    with open(ruta_corpus, 'w', encoding='utf-8') as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)
    with open(ruta_themes, 'w', encoding='utf-8') as fh:
        json.dump(themes, fh, ensure_ascii=False, indent=2)

    print(f'{len(corpus)} notas generadas -> {ruta_corpus}')
    print(f'temas registrados -> {ruta_themes}')
    print('Ejemplo:', corpus[0]['texto'])
    return corpus, themes


if __name__ == '__main__':
    main()
