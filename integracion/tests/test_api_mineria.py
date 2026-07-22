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
