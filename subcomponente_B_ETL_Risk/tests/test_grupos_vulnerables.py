# -*- coding: utf-8 -*-
"""Pruebas de grupos_vulnerables.py: banderas de prioridad por composición familiar."""
from grupos_vulnerables import calcular_banderas, motivo_prioridad


def test_familia_sin_banderas():
    integrantes = [{"nombre": "Juan", "edad": 35}, {"nombre": "Ana", "edad": 33}]
    b = calcular_banderas(integrantes, [])
    assert not b["requiere_atencion_prioritaria"]
    assert not any([b["tiene_embarazada"], b["tiene_menor_1_anio"], b["tiene_menor_5_sin_vacunas"], b["tiene_adulto_mayor_solo"]])


def test_detecta_embarazada():
    integrantes = [{"nombre": "Juan", "edad": 35}, {"nombre": "Ana", "edad": 28, "atencion_embarazo": "Sector Público"}]
    b = calcular_banderas(integrantes, [])
    assert b["tiene_embarazada"] is True
    assert b["requiere_atencion_prioritaria"] is True


def test_detecta_bebe_menor_de_un_anio():
    integrantes = [{"nombre": "Juan", "edad": 35}, {"nombre": "Bebe", "edad": 0}]
    b = calcular_banderas(integrantes, [])
    assert b["tiene_menor_1_anio"] is True
    assert b["requiere_atencion_prioritaria"] is True


def test_menor_de_cinco_sin_vacunas_detectado():
    integrantes = [{"nombre": "Juan", "edad": 35}, {"nombre": "Niño", "edad": 3}]
    b = calcular_banderas(integrantes, vacunas_aplicadas=[])
    assert b["tiene_menor_5_sin_vacunas"] is True


def test_menor_de_cinco_con_vacunas_no_se_marca():
    integrantes = [{"nombre": "Juan", "edad": 35}, {"nombre": "Niño", "edad": 3}]
    vacunas = [{"paciente": "Niño", "vacuna": "BCG", "dosis": "Única"}]
    b = calcular_banderas(integrantes, vacunas_aplicadas=vacunas)
    assert b["tiene_menor_5_sin_vacunas"] is False
    assert b["requiere_atencion_prioritaria"] is False


def test_adulto_mayor_solo_cuando_todo_el_hogar_es_60_mas():
    integrantes = [{"nombre": "Abuela", "edad": 78}]
    b = calcular_banderas(integrantes, [])
    assert b["tiene_adulto_mayor_solo"] is True


def test_adulto_mayor_no_se_marca_solo_si_hay_mas_jovenes_en_el_hogar():
    integrantes = [{"nombre": "Abuela", "edad": 78}, {"nombre": "Nieto", "edad": 20}]
    b = calcular_banderas(integrantes, [])
    assert b["tiene_adulto_mayor_solo"] is False


def test_hogar_vacio_no_marca_adulto_mayor_solo():
    b = calcular_banderas([], [])
    assert b["tiene_adulto_mayor_solo"] is False
    assert b["requiere_atencion_prioritaria"] is False


def test_motivo_prioridad_combina_ml_y_banderas():
    b = calcular_banderas([{"nombre": "Ana", "edad": 28, "atencion_embarazo": "Hogar"}], [])
    motivo = motivo_prioridad(b, nivel_riesgo_ml="MEDIO")
    assert "embarazada" in motivo
    assert "riesgo familiar ALTO" not in motivo  # nivel_riesgo_ml no era ALTO


def test_motivo_prioridad_sin_banderas_ni_ml_alto():
    b = calcular_banderas([{"nombre": "Juan", "edad": 40}], [])
    assert motivo_prioridad(b, nivel_riesgo_ml="BAJO") == "sin motivo de prioridad"


# ── Riesgo zoonótico (mascotas sin vacunar) ──────────────────────────────────

def test_detecta_mascota_sin_vacunar():
    integrantes = [{"nombre": "Juan", "edad": 35}]
    vivienda = {"perros_gatos_dentro": True, "mascotas_vacunas_corrientes": False}
    b = calcular_banderas(integrantes, [], vivienda=vivienda)
    assert b["tiene_mascota_sin_vacunar"] is True
    assert b["requiere_atencion_prioritaria"] is True


def test_mascota_vacunada_no_se_marca():
    integrantes = [{"nombre": "Juan", "edad": 35}]
    vivienda = {"perros_gatos_dentro": True, "mascotas_vacunas_corrientes": True}
    b = calcular_banderas(integrantes, [], vivienda=vivienda)
    assert b["tiene_mascota_sin_vacunar"] is False


def test_sin_mascotas_no_se_marca_aunque_falte_vacunacion_corriente():
    integrantes = [{"nombre": "Juan", "edad": 35}]
    vivienda = {"perros_gatos_dentro": False, "mascotas_vacunas_corrientes": False}
    b = calcular_banderas(integrantes, [], vivienda=vivienda)
    assert b["tiene_mascota_sin_vacunar"] is False


def test_sin_vivienda_no_marca_bandera_zoonotica():
    """Caso de /riesgo/predecir plano: si no se provee `vivienda`, la bandera
    zoonótica queda en False (el caller la manda directo como campo plano)."""
    b = calcular_banderas([{"nombre": "Juan", "edad": 35}], [])
    assert b["tiene_mascota_sin_vacunar"] is False


def test_motivo_prioridad_incluye_zoonotico():
    b = calcular_banderas(
        [{"nombre": "Juan", "edad": 35}], [],
        vivienda={"perros_gatos_dentro": True, "mascotas_vacunas_corrientes": False},
    )
    motivo = motivo_prioridad(b, nivel_riesgo_ml="BAJO")
    assert "zoonótico" in motivo
