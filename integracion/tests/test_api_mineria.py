# -*- coding: utf-8 -*-
"""
test_api_mineria.py — primeras pruebas de integración de la API de minería
(FastAPI TestClient), sobre `integracion/api_mineria.py`.

Cubre: autenticación por API key (`X-API-Key`), validación de entrada en
`/buscar` (límite de longitud M4) y en `/riesgo/predecir(-lote)`, y las
regresiones de bugs ya corregidos (p. ej. `/corpus/estadisticas` que antes
crasheaba con 500/AttributeError).

Requiere que `conftest.py` (mismo directorio) haya fijado MINERIA_API_KEY
ANTES de importar api_mineria; los fixtures `client` y `api_key` vienen de ahí.
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# /salud — único endpoint público (sin X-API-Key)
# ─────────────────────────────────────────────────────────────────────────────
def test_salud_sin_header_es_publico_y_200(client):
    resp = client.get("/salud")
    assert resp.status_code == 200
    data = resp.json()
    assert "componentes" in data


# ─────────────────────────────────────────────────────────────────────────────
# /buscar — autenticación (401) y validación de `q` (400)
# ─────────────────────────────────────────────────────────────────────────────
def test_buscar_sin_api_key_devuelve_401(client):
    resp = client.get("/buscar", params={"q": "vivienda", "motor": "bm25"})
    assert resp.status_code == 401


def test_buscar_con_api_key_incorrecta_devuelve_401(client):
    resp = client.get(
        "/buscar",
        params={"q": "vivienda", "motor": "bm25"},
        headers={"X-API-Key": "esta-clave-es-incorrecta"},
    )
    assert resp.status_code == 401


def test_buscar_con_query_vacia_devuelve_400(client, api_key):
    resp = client.get(
        "/buscar",
        params={"q": "", "motor": "bm25"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 400


def test_buscar_con_query_demasiado_larga_devuelve_400(client, api_key):
    # Mejora M4: MAX_LONGITUD_CONSULTA = 500. Se genera una consulta de 501
    # caracteres (uno por encima del límite) para probar el borde.
    query_larga = "a" * 501
    resp = client.get(
        "/buscar",
        params={"q": query_larga, "motor": "bm25"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 400


def test_buscar_motor_bm25_con_api_key_correcta_devuelve_200(client, api_key):
    resp = client.get(
        "/buscar",
        params={"q": "familias con desnutricion infantil", "motor": "bm25"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["resultados"], list)


def test_buscar_motor_tfidf_con_api_key_correcta_devuelve_200(client, api_key):
    resp = client.get(
        "/buscar",
        params={"q": "familias con desnutricion infantil", "motor": "tfidf"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["resultados"], list)


def test_buscar_motor_semantico_200_o_503_si_no_disponible(client, api_key):
    # El motor semántico (sentence-transformers + modelo Sentence-BERT en
    # español) es OPCIONAL: si sentence-transformers o el modelo no pudieron
    # instalarse/descargarse en este entorno, api_mineria degrada con gracia
    # (ESTADO["semantico"] = None) y /buscar responde 503 en vez de crashear.
    # Por eso este test acepta ambos códigos: 200 (motor cargó) o 503 (no
    # disponible en este entorno), y NO trata 503 como fallo de la suite.
    resp = client.get(
        "/buscar",
        params={"q": "familias con desnutricion infantil", "motor": "semantico"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert isinstance(resp.json()["resultados"], list)


# ─────────────────────────────────────────────────────────────────────────────
# /corpus/estadisticas — regresión: antes crasheaba con 500/AttributeError
# ─────────────────────────────────────────────────────────────────────────────
def test_corpus_estadisticas_no_crashea_y_tiene_documentos(client, api_key):
    resp = client.get("/corpus/estadisticas", headers={"X-API-Key": api_key})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_documentos"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# /riesgo/predecir — todos los campos tienen default, body {} debe bastar
# ─────────────────────────────────────────────────────────────────────────────
def test_riesgo_predecir_con_body_vacio_devuelve_200_y_nivel_valido(client, api_key):
    resp = client.post(
        "/riesgo/predecir", json={}, headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["nivel_riesgo"] in {"ALTO", "MEDIO", "BAJO"}
    assert 0 <= data["probabilidad_alto"] <= 1


def test_riesgo_predecir_con_valor_de_catalogo_invalido_devuelve_422(client, api_key):
    resp = client.post(
        "/riesgo/predecir",
        json={"material_techo": "valor_que_no_existe_en_el_catalogo"},
        headers={"X-API-Key": api_key},
    )
    # Pydantic Literal (validado contra CAT_MATERIAL_TECHO_PAREDES) debe
    # rechazar cualquier valor fuera del catálogo oficial con 422.
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /riesgo/predecir-lote — límites de tamaño del lote (vacío y > 500)
# ─────────────────────────────────────────────────────────────────────────────
def test_riesgo_predecir_lote_vacio_devuelve_400(client, api_key):
    resp = client.post(
        "/riesgo/predecir-lote",
        json={"familias": []},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 400


def test_riesgo_predecir_lote_excede_maximo_devuelve_400(client, api_key):
    familias_501 = [{} for _ in range(501)]
    resp = client.post(
        "/riesgo/predecir-lote",
        json={"familias": familias_501},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# /riesgo/predecir — banderas de grupo vulnerable / riesgo zoonótico
# ─────────────────────────────────────────────────────────────────────────────
def test_riesgo_predecir_con_bandera_zoonotica_marca_prioridad_urgente(client, api_key):
    resp = client.post(
        "/riesgo/predecir",
        json={"tiene_mascota_sin_vacunar": True},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["prioridad_visita"] == "URGENTE"
    assert "zoon" in data["motivo_prioridad"]


def test_riesgo_predecir_sin_banderas_ni_riesgo_alto_es_regular(client, api_key):
    resp = client.post(
        "/riesgo/predecir",
        json={"ingreso_nivel": 5, "count_enfermedades_cronicas": 0, "agua_entubada": True},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    # No podemos forzar BAJO/MEDIO con certeza (depende del modelo), pero si
    # el nivel no salió ALTO y no se mandó ninguna bandera, debe ser REGULAR.
    if data["nivel_riesgo"] != "ALTO":
        assert data["prioridad_visita"] == "REGULAR"


# ─────────────────────────────────────────────────────────────────────────────
# /riesgo/zonas — riesgo de cúmulo geográfico (mejora nueva)
# ─────────────────────────────────────────────────────────────────────────────
def test_riesgo_zonas_devuelve_lista_con_columnas_esperadas(client, api_key):
    resp = client.get("/riesgo/zonas", headers={"X-API-Key": api_key})
    assert resp.status_code == 200
    zonas = resp.json()
    assert isinstance(zonas, list)
    assert len(zonas) > 0
    primera = zonas[0]
    for campo in ("zona", "total_familias", "pct_alto_o_bandera", "nivel_alerta_zona"):
        assert campo in primera
    assert primera["nivel_alerta_zona"] in {"ALTO", "MEDIO", "BAJO"}


def test_riesgo_zonas_ordenado_de_mayor_a_menor_concentracion(client, api_key):
    resp = client.get("/riesgo/zonas", headers={"X-API-Key": api_key})
    zonas = resp.json()
    porcentajes = [z["pct_alto_o_bandera"] for z in zonas]
    assert porcentajes == sorted(porcentajes, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# /buscar/estructurado — filtro sobre datos de la cédula (no similitud de texto)
# ─────────────────────────────────────────────────────────────────────────────
def test_buscar_estructurado_categoria_conocida_devuelve_disponible(client, api_key):
    resp = client.get(
        "/buscar/estructurado", params={"q": "mascotas"}, headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["disponible"] is True
    assert data["categoria"] == "mascotas"


def test_buscar_estructurado_categoria_desconocida_devuelve_no_disponible(client, api_key):
    resp = client.get(
        "/buscar/estructurado",
        params={"q": "xyzabc123 no es nada reconocible"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["disponible"] is False
    assert "mensaje" in data


def test_buscar_estructurado_sin_api_key_devuelve_401(client):
    resp = client.get("/buscar/estructurado", params={"q": "mascotas"})
    assert resp.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# /buscar/familias — motor de búsqueda sobre observaciones REALES de familia
# (no el corpus-benchmark de 150 notas de /buscar). Ver corpus_familias.py.
# ─────────────────────────────────────────────────────────────────────────────
def test_buscar_familias_sin_api_key_devuelve_401(client):
    resp = client.get("/buscar/familias", params={"q": "embarazada", "motor": "bm25"})
    assert resp.status_code == 401


def test_buscar_familias_con_api_key_incorrecta_devuelve_401(client):
    resp = client.get(
        "/buscar/familias",
        params={"q": "embarazada", "motor": "bm25"},
        headers={"X-API-Key": "esta-clave-es-incorrecta"},
    )
    assert resp.status_code == 401


def test_buscar_familias_con_query_vacia_devuelve_400(client, api_key):
    resp = client.get(
        "/buscar/familias",
        params={"q": "", "motor": "bm25"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 400


def test_buscar_familias_con_query_demasiado_larga_devuelve_400(client, api_key):
    query_larga = "a" * 501
    resp = client.get(
        "/buscar/familias",
        params={"q": query_larga, "motor": "bm25"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 400


def test_buscar_familias_motor_invalido_devuelve_422(client, api_key):
    # `motor` es un Literal["bm25", "tfidf", "semantico"]; FastAPI/Pydantic
    # debe rechazar cualquier otro valor con 422 (igual que /buscar).
    resp = client.get(
        "/buscar/familias",
        params={"q": "embarazada", "motor": "no_existe"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 422


def test_buscar_familias_bm25_devuelve_estructura_esperada_y_familias_indexadas(client, api_key):
    resp = client.get(
        "/buscar/familias",
        params={"q": "embarazada vivienda mascota", "motor": "bm25", "k": 5},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["consulta"] == "embarazada vivienda mascota"
    assert data["motor"] == "bm25"
    assert isinstance(data["resultados"], list)
    # Tras la Parte 1 (observaciones sintéticas enriquecidas), la gran mayoría
    # de las 4000 familias deben quedar indexadas (todas tienen al menos la
    # oración base "Visita domiciliaria en colonia...").
    assert data["familias_indexadas"] > 0
    if data["resultados"]:
        primero = data["resultados"][0]
        for campo in (
            "familia_id", "nombre_informante", "domicilio", "colonia",
            "localidad", "texto_observacion", "score",
        ):
            assert campo in primero
        assert isinstance(primero["familia_id"], int)


def test_buscar_familias_tfidf_devuelve_200(client, api_key):
    resp = client.get(
        "/buscar/familias",
        params={"q": "embarazada vivienda mascota", "motor": "tfidf", "k": 5},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["motor"] == "tfidf"
    assert isinstance(data["resultados"], list)


def test_buscar_familias_semantico_200_o_503_si_no_disponible(client, api_key):
    resp = client.get(
        "/buscar/familias",
        params={"q": "embarazada vivienda mascota", "motor": "semantico", "k": 5},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert isinstance(resp.json()["resultados"], list)
