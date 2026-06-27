# -*- coding: utf-8 -*-
"""
preprocess.py - Subcomponente C (Motor de Busqueda SUMS)

Pipeline de preprocesamiento IDENTICO al del Lab 1 (clave del profesor):
  - normalizar: minusculas, NFC, quitar URLs/HTML, colapsar espacios.
  - lematizar con spaCy (es_core_news_sm), NO stemming, para el pipeline final.
  - conservar negaciones/intensificadores al filtrar stopwords (critico en
    texto de salud: "sin control prenatal", "no controlado").
  - quitar acentos SOBRE EL LEMA (sube el recall; el usuario rara vez teclea
    acentos), y filtrar tokens de len<=2.

EXACTAMENTE el mismo pipeline (preprocesar) se usa para el corpus y para la
consulta -> es la unica forma de que TF-IDF/BM25 comparen lo mismo.

Incluye tambien la comparacion stemming-vs-lematizacion que pide el Lab 1
(tamanos de vocabulario + ejemplos divergentes) en comparar_stem_vs_lemma().
"""
import json
import os
import re
import unicodedata

import spacy
from nltk.stem import SnowballStemmer

# ---------------------------------------------------------------------------
# Modelo y recursos (se cargan una sola vez al importar el modulo).
# ---------------------------------------------------------------------------
nlp = spacy.load('es_core_news_sm')

RE_URL = re.compile(r'https?://\S+')
RE_HTML = re.compile(r'</?[a-zA-Z][^>]*>')

# Negaciones e intensificadores que SE CONSERVAN aunque spaCy los marque stop:
# invierten o modulan el sentido (clave en salud).
CONSERVAR = {'no', 'nunca', 'sin', 'ni', 'muy', 'poco', 'nada', 'tampoco'}
MIS_STOPWORDS = set(nlp.Defaults.stop_words) - CONSERVAR

_stemmer = SnowballStemmer('spanish')


def _qd(s):
    """Quita diacriticos (acentos) de una cadena."""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalizar(texto, quitar_acentos=False):
    """texto crudo -> string normalizado (aun sin tokenizar).

    Para lematizar se llama con quitar_acentos=False (spaCy espera los acentos
    para analizar bien la morfologia); los acentos se quitan despues sobre el
    lema, alineando con la decision de retrieval.
    """
    t = unicodedata.normalize('NFC', texto.lower())
    t = RE_URL.sub(' ', t)
    t = RE_HTML.sub(' ', t)
    if quitar_acentos:
        t = _qd(t)
    return re.sub(r'\s+', ' ', t).strip()


def preprocesar(texto):
    """Pipeline DEFINITIVO (Lab 1): lematizacion + acentos quitados sobre el
    lema + negaciones conservadas + filtro de tokens cortos.

    Devuelve la lista de tokens (lemas) lista para indexar."""
    doc = nlp(normalizar(texto, quitar_acentos=False))
    out = []
    for tok in doc:
        if tok.is_punct or tok.is_space or tok.like_num or tok.is_stop:
            continue
        lema = _qd(tok.lemma_.lower())
        if lema in MIS_STOPWORDS or len(lema) <= 2:
            continue
        out.append(lema)
    return out


# ---------------------------------------------------------------------------
# Comparacion stemming vs lematizacion (Lab 1, seccion 4).
# ---------------------------------------------------------------------------
def tokens_stemming(texto):
    """Variante con stemming (Snowball) para la comparacion del Lab 1."""
    base = normalizar(texto, quitar_acentos=True)
    toks = re.findall(r'[a-z0-9]+', base)
    return [_stemmer.stem(w) for w in toks
            if w not in MIS_STOPWORDS and len(w) > 2]


def comparar_stem_vs_lemma(textos, muestra=None):
    """Compara tamanos de vocabulario y devuelve ejemplos divergentes.

    Devuelve un dict con |V_stem|, |V_lemma| y una lista de filas
    (palabra, stem, lemma) para inspeccionar el sobre-colapso del stemming."""
    V_stem = {t for tx in textos for t in tokens_stemming(tx)}
    V_lemma = {t for tx in textos for t in preprocesar(tx)}
    if muestra is None:
        muestra = ['embarazada', 'desnutricion', 'criaderos', 'vacunacion',
                   'hipertension', 'viviendas', 'controlado', 'almacenan',
                   'acarrean', 'gestante']
    filas = []
    for w in muestra:
        stem = _stemmer.stem(w)
        lemma = _qd(nlp(w)[0].lemma_.lower())
        filas.append((w, stem, lemma))
    return {'V_stem': len(V_stem), 'V_lemma': len(V_lemma), 'ejemplos': filas}


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(aqui, '..', 'data')
    ruta_crudo = os.path.join(data_dir, 'corpus_crudo_sums.json')
    ruta_proc = os.path.join(data_dir, 'corpus_procesado_sums.json')

    with open(ruta_crudo, encoding='utf-8') as fh:
        corpus = json.load(fh)

    procesado = [{'id': f['id'], 'titulo': f['titulo'],
                  'tokens': preprocesar(f['texto'])} for f in corpus]

    with open(ruta_proc, 'w', encoding='utf-8') as fh:
        json.dump(procesado, fh, ensure_ascii=False, indent=2)

    print(f'{len(procesado)} documentos procesados -> {ruta_proc}')
    print('Ejemplo', procesado[0]['id'], '->', procesado[0]['tokens'][:10])

    # Comparacion stemming vs lematizacion (Lab 1).
    textos = [f['texto'] for f in corpus]
    cmp = comparar_stem_vs_lemma(textos)
    print(f"\n|V_stemming| = {cmp['V_stem']}   |V_lemma| = {cmp['V_lemma']}")
    print('palabra           stemming        lemma')
    for w, s, l in cmp['ejemplos']:
        print(f'{w:18}{s:16}{l}')
    return procesado


if __name__ == '__main__':
    main()
