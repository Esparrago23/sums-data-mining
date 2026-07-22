# -*- coding: utf-8 -*-
"""Pruebas de buscador_estructurado.py: enrutador de intención + filtros reales."""
from buscador_estructurado import buscar_estructurado, interpretar_consulta


def _familia(nombre="Juan Perez", colonia="Centro", calle="Calle 5", integrantes=None,
             vacunas=None, mascotas=False, otros_animales=None):
    return {
        "familia": {"informante_nombre": nombre, "colonia": colonia, "calle": calle, "numero_exterior": "10", "localidad": "Suchiapa"},
        "vivienda": {"perros_gatos_dentro": mascotas, "otros_animales": otros_animales or []},
        "integrantes": integrantes or [{"nombre": nombre, "edad": 35, "enfermedades_cronicas": [], "dias_proteina": 6, "dias_frutas_verduras": 6}],
        "vacunacion": {"se_aplico_vacuna": bool(vacunas), "vacunas": vacunas or []},
    }


def test_interpretar_detecta_vacuna_sin_texto_exacto_del_catalogo():
    r = interpretar_consulta("sarampion")
    assert r.disponible is True
    assert r.categoria == "vacuna"
    assert "Sarampión" in r.detalle


def test_interpretar_detecta_negacion():
    r = interpretar_consulta("sin vacuna de sarampion")
    assert r.negado is True
    r2 = interpretar_consulta("vacuna de sarampion")
    assert r2.negado is False


def test_interpretar_no_matchea_nada_devuelve_no_disponible():
    r = interpretar_consulta("xyzabc123 no es nada conocido")
    assert r.disponible is False


def test_buscar_vacuna_con_regresa_solo_los_que_la_tienen():
    familias = [
        _familia("Ana", vacunas=[{"paciente": "Ana", "vacuna": "BCG", "dosis": "Única"}]),
        _familia("Beto", vacunas=[]),
    ]
    res = buscar_estructurado("BCG", familias)
    assert res["disponible"] is True
    assert res["total_coincidencias"] == 1
    assert res["resultados"][0]["nombre_informante"] == "Ana"


def test_buscar_vacuna_sin_regresa_los_que_no_la_tienen():
    familias = [
        _familia("Ana", vacunas=[{"paciente": "Ana", "vacuna": "BCG", "dosis": "Única"}]),
        _familia("Beto", vacunas=[]),
    ]
    res = buscar_estructurado("sin BCG", familias)
    assert res["total_coincidencias"] == 1
    assert res["resultados"][0]["nombre_informante"] == "Beto"


def test_buscar_mascotas():
    familias = [_familia("Con mascota", mascotas=True), _familia("Sin mascota", mascotas=False)]
    res = buscar_estructurado("mascotas", familias)
    assert res["categoria"] == "mascotas"
    assert res["total_coincidencias"] == 1
    assert res["resultados"][0]["nombre_informante"] == "Con mascota"


def test_buscar_nutricion_baja():
    familias = [
        _familia("Nutricion baja", integrantes=[{"nombre": "x", "edad": 30, "dias_proteina": 1, "dias_frutas_verduras": 1}]),
        _familia("Nutricion normal", integrantes=[{"nombre": "y", "edad": 30, "dias_proteina": 6, "dias_frutas_verduras": 6}]),
    ]
    res = buscar_estructurado("mala nutricion", familias)
    assert res["total_coincidencias"] == 1
    assert res["resultados"][0]["nombre_informante"] == "Nutricion baja"


def test_buscar_embarazo():
    familias = [
        _familia("Con embarazada", integrantes=[{"nombre": "e", "edad": 25, "atencion_embarazo": "Sector Público"}]),
        _familia("Sin embarazada"),
    ]
    res = buscar_estructurado("embarazo", familias)
    assert res["total_coincidencias"] == 1
    assert res["resultados"][0]["nombre_informante"] == "Con embarazada"


def test_buscar_direccion_por_colonia():
    familias = [_familia("En Centro", colonia="Centro"), _familia("En otra colonia", colonia="La Lomita")]
    res = buscar_estructurado("Centro", familias)
    assert res["categoria"] == "direccion"
    assert res["total_coincidencias"] == 1
    assert res["resultados"][0]["nombre_informante"] == "En Centro"


def test_buscar_sin_coincidencia_devuelve_disponible_false_con_mensaje():
    res = buscar_estructurado("blablabla desconocido", [_familia()])
    assert res["disponible"] is False
    assert res["resultados"] == []
    assert "mensaje" in res


def test_buscar_respeta_k():
    familias = [_familia(f"Persona{i}", mascotas=True) for i in range(30)]
    res = buscar_estructurado("mascotas", familias, k=5)
    assert len(res["resultados"]) == 5
    assert res["total_coincidencias"] == 30  # el total real no se trunca, solo la página
