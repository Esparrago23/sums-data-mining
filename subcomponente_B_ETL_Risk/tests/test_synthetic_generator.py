# -*- coding: utf-8 -*-
"""Pruebas de synthetic_generator.generar_observaciones (Parte 1 de la tarea de
enriquecimiento de observaciones sintéticas).

Cubre: determinismo (mismo idx + mismos argumentos -> mismo resultado),
límite de longitud (<=300, columna real VARCHAR(300)), y que la bandera
'tiene_embarazada' hace aparecer la mención de embarazo con alta probabilidad
(prueba probabilística razonable dado que la probabilidad configurada es ~0.85).
"""
from synthetic_generator import LIMITE_OBSERVACIONES, generar_observaciones

BANDERAS_SIN_NADA = {
    "tiene_embarazada": False,
    "tiene_menor_1_anio": False,
    "tiene_menor_5_sin_vacunas": False,
    "tiene_adulto_mayor_solo": False,
    "tiene_mascota_sin_vacunar": False,
    "requiere_atencion_prioritaria": False,
}


def _banderas(**overrides):
    banderas = dict(BANDERAS_SIN_NADA)
    banderas.update(overrides)
    return banderas


def test_es_determinístico_mismo_idx_mismos_argumentos():
    r1 = generar_observaciones(123, _banderas(), "Centro", "Suchiapa", "Concreto o cemento", "Concreto o cemento")
    r2 = generar_observaciones(123, _banderas(), "Centro", "Suchiapa", "Concreto o cemento", "Concreto o cemento")
    assert r1 == r2


def test_longitud_no_excede_el_limite_varchar_300():
    for idx in range(200):
        texto = generar_observaciones(
            idx,
            _banderas(tiene_embarazada=True, tiene_menor_5_sin_vacunas=True, tiene_mascota_sin_vacunar=True),
            "Independencia", "Tuxtla Gutiérrez", "Lámina", "Madera",
        )
        assert len(texto) <= LIMITE_OBSERVACIONES


def test_siempre_incluye_la_oracion_base_de_visita_domiciliaria():
    texto = generar_observaciones(1, _banderas(), "Pacú", "Suchiapa", "Concreto o cemento", "Concreto o cemento")
    assert "Visita domiciliaria en colonia Pacú, Suchiapa." in texto


def test_no_corta_a_la_mitad_de_una_palabra():
    for idx in range(50):
        texto = generar_observaciones(
            idx,
            _banderas(tiene_embarazada=True, tiene_menor_5_sin_vacunas=True,
                      tiene_mascota_sin_vacunar=True, tiene_adulto_mayor_solo=True),
            "San Jacinto", "Suchiapa", "Lámina", "Madera",
        )
        # El texto siempre debe terminar en un signo de puntuación de cierre
        # de oración (nunca a mitad de palabra) tras el truncado.
        assert texto.endswith(".")


def test_bandera_embarazada_hace_aparecer_embarazo_con_alta_probabilidad():
    # Probabilidad configurada ~0.85 cuando la bandera está activa: de 30 idx
    # distintos, se espera que al menos 20 mencionen embarazo.
    banderas = _banderas(tiene_embarazada=True)
    apariciones = 0
    for idx in range(30):
        texto = generar_observaciones(idx, banderas, "Centro", "Suchiapa", "Concreto o cemento", "Concreto o cemento")
        if "embarazad" in texto.lower() or "gestaci" in texto.lower() or "prenatal" in texto.lower():
            apariciones += 1
    assert apariciones >= 20


def test_sin_bandera_embarazada_rara_vez_menciona_embarazo():
    # Probabilidad de "ruido" ~0.06-0.10 cuando la bandera NO está activa: de
    # 30 idx distintos, se espera que muy pocos mencionen embarazo.
    banderas = _banderas(tiene_embarazada=False)
    apariciones = 0
    for idx in range(30):
        texto = generar_observaciones(idx, banderas, "Centro", "Suchiapa", "Concreto o cemento", "Concreto o cemento")
        if "embarazad" in texto.lower() or "gestaci" in texto.lower() or "prenatal" in texto.lower():
            apariciones += 1
    assert apariciones <= 10


def test_diferentes_idx_producen_texto_variado():
    # No deben ser todos idénticos (el RNG independiente por idx sí varía).
    textos = {
        generar_observaciones(idx, _banderas(tiene_embarazada=True), "Centro", "Suchiapa", "Lámina", "Madera")
        for idx in range(20)
    }
    assert len(textos) > 1
