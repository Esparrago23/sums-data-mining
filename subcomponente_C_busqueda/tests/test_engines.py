# -*- coding: utf-8 -*-
"""
test_engines.py - Subcomponente C (Motor de Busqueda SUMS)

Suite de pytest para MotorBM25 (bm25_engine.py) y MotorTFIDF (tfidf_engine.py)
usando un corpus SINTETICO ya pre-tokenizado (no requiere spaCy en tiempo de
ejecucion para construir los motores).

NOTA DE ENTORNO: tanto bm25_engine.py como tfidf_engine.py hacen
`from preprocess import preprocesar` a nivel de modulo, y preprocess.py carga
el modelo de spaCy `es_core_news_sm` al importarse (linea `nlp =
spacy.load(...)`). Esto significa que, aunque construir MotorBM25/MotorTFIDF
con tokens ya pre-tokenizados NO necesita spaCy en tiempo de ejecucion, el
mero `import bm25_engine` / `import tfidf_engine` SI requiere que el modelo
este instalado. Si no lo esta, se salta todo el archivo en vez de romper la
coleccion de pytest.
"""
import pytest

# pytest.importorskip("spacy") por si el PAQUETE spacy no esta instalado.
# Esto NO cubre el caso (mas comun aqui) de que spacy SI este instalado pero
# falte el modelo `es_core_news_sm`: ese caso lo cubre el try/except de abajo,
# que envuelve el import real de los motores (preprocess.py hace
# `spacy.load('es_core_news_sm')` a nivel de modulo y lanza OSError si el
# modelo no esta descargado).
pytest.importorskip("spacy")

try:
    from bm25_engine import MotorBM25
    from tfidf_engine import MotorTFIDF
    from preprocess import preprocesar
except Exception as exc:  # pragma: no cover - depende del entorno
    pytest.skip(
        f"bm25_engine/tfidf_engine/preprocess no importables en este entorno "
        f"(probablemente falta el modelo de spaCy es_core_news_sm): {exc}",
        allow_module_level=True,
    )


# Corpus sintetico pre-tokenizado: no pasa por preprocesar(), asi que la
# construccion de los motores no depende del pipeline de spaCy.
CORPUS_SINTETICO = [
    {"id": "n1", "titulo": "Dengue en la comunidad",
     "tokens": ["dengue", "mosquito", "fiebre", "sintoma"]},
    {"id": "n2", "titulo": "Agua y saneamiento",
     "tokens": ["agua", "saneamiento", "letrina", "comunidad"]},
    {"id": "n3", "titulo": "Vacunacion infantil",
     "tokens": ["vacuna", "infantil", "salud", "comunidad"]},
    {"id": "n4", "titulo": "Nutricion materna",
     "tokens": ["nutricion", "materna", "embarazo", "salud"]},
]


# ---------------------------------------------------------------------------
# Regresion: bm25.N NO existia (AttributeError en GET /corpus/estadisticas de
# integracion/api_mineria.py, que lee bm25.N). Ahora MotorBM25 debe exponerlo.
# ---------------------------------------------------------------------------
def test_motor_bm25_expone_atributo_N():
    motor = MotorBM25(CORPUS_SINTETICO)

    assert hasattr(motor, "N")
    assert motor.N == len(CORPUS_SINTETICO)


def _assert_ranking_valido(ranking, k):
    assert len(ranking) <= k
    for item in ranking:
        assert isinstance(item, tuple)
        assert len(item) == 3
        score, doc_id, titulo = item
        assert isinstance(score, float)
        assert isinstance(doc_id, str)
        assert isinstance(titulo, str)
    scores = [s for s, _, _ in ranking]
    assert scores == sorted(scores, reverse=True)


def test_buscar_bm25_ordena_descendente_y_respeta_k():
    motor = MotorBM25(CORPUS_SINTETICO)
    k = 2

    ranking = motor.buscar_bm25("dengue mosquito", k=k)

    _assert_ranking_valido(ranking, k)


def test_buscar_tfidf_ordena_descendente_y_respeta_k():
    motor = MotorTFIDF(CORPUS_SINTETICO)
    k = 2

    ranking = motor.buscar_tfidf("dengue mosquito", k=k)

    _assert_ranking_valido(ranking, k)


# ---------------------------------------------------------------------------
# Tests para preprocess.preprocesar() - dependen de un fix EN PARALELO (otra
# tarea) para dos bugs. Si ese fix aun no ha aterrizado en este entorno, estos
# tests FALLARAN (no se saltan): es la senal esperada de que preprocess.py
# todavia no tiene el arreglo. No se modifica preprocess.py desde aqui.
# ---------------------------------------------------------------------------
def test_excepciones_lema_corrige_dengue():
    # spaCy (es_core_news_sm) lematiza mal "dengue" -> "denguir" en este
    # dominio; EXCEPCIONES_LEMA debe corregirlo a "dengue".
    tokens = preprocesar("casos sospechosos de dengue")

    assert "dengue" in tokens
    assert "denguir" not in tokens


def test_excepciones_lema_corrige_atencion():
    # spaCy lematiza mal "atencion" -> "atencionir"; debe corregirse a
    # "atencion".
    tokens = preprocesar("atencion en el hogar")

    assert "atencion" in tokens
    assert "atencionir" not in tokens


def test_preprocesar_no_genera_tokens_con_espacio_embebido():
    # Bug del pseudo-token "beber el": ningun token del resultado debe
    # contener un espacio embebido.
    tokens = preprocesar("es importante beberla siempre despues de hervirla")

    assert all(" " not in t for t in tokens)
