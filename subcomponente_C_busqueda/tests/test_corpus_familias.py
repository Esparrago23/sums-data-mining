# -*- coding: utf-8 -*-
"""Pruebas de corpus_familias.py: builder del corpus de busqueda sobre
observaciones REALES de familia (no el corpus-benchmark de corpus_builder.py)."""
from corpus_familias import construir_corpus_desde_familias


def _familia(observaciones, calle="Calle 5", numero_exterior="10", colonia="Centro"):
    return {
        "observaciones": observaciones,
        "familia": {
            "informante_nombre": "Juan Perez",
            "calle": calle,
            "numero_exterior": numero_exterior,
            "colonia": colonia,
            "localidad": "Suchiapa",
        },
    }


def test_omite_familias_con_observaciones_none():
    familias = [_familia(None), _familia("Nota real de visita.")]
    corpus = construir_corpus_desde_familias(familias)
    assert len(corpus) == 1
    assert corpus[0]["texto"] == "Nota real de visita."


def test_omite_familias_con_observaciones_vacias():
    familias = [_familia(""), _familia("Nota real de visita.")]
    corpus = construir_corpus_desde_familias(familias)
    assert len(corpus) == 1


def test_omite_familias_con_observaciones_solo_espacios():
    familias = [_familia("    \n\t  "), _familia("Nota real de visita.")]
    corpus = construir_corpus_desde_familias(familias)
    assert len(corpus) == 1


def test_id_sigue_la_convencion_de_indice_de_enumerate():
    # Misma convencion que buscador_estructurado.buscar_estructurado (el
    # "familia_id" es el indice dentro de la lista completa, incluyendo las
    # familias omitidas -- el id NO se recompacta).
    familias = [_familia(None), _familia("Segunda familia, con nota."), _familia(None), _familia("Cuarta familia.")]
    corpus = construir_corpus_desde_familias(familias)
    ids = [d["id"] for d in corpus]
    assert ids == ["1", "3"]


def test_esquema_de_salida_es_id_titulo_texto():
    familias = [_familia("Visita domiciliaria en colonia Centro, Suchiapa.", calle="Av. Reforma", numero_exterior="42", colonia="Centro")]
    corpus = construir_corpus_desde_familias(familias)
    assert len(corpus) == 1
    doc = corpus[0]
    assert set(doc.keys()) == {"id", "titulo", "texto"}
    assert doc["id"] == "0"
    assert "Av. Reforma" in doc["titulo"]
    assert "42" in doc["titulo"]
    assert "Centro" in doc["titulo"]
    assert doc["texto"] == "Visita domiciliaria en colonia Centro, Suchiapa."


def test_corpus_vacio_si_todas_las_familias_no_tienen_observaciones():
    familias = [_familia(None), _familia(""), _familia("   ")]
    corpus = construir_corpus_desde_familias(familias)
    assert corpus == []


def test_lista_vacia_devuelve_corpus_vacio():
    assert construir_corpus_desde_familias([]) == []
