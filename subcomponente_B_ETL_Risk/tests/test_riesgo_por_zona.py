# -*- coding: utf-8 -*-
"""Pruebas de risk_report.resumen_por_zona: riesgo de cúmulo geográfico."""
import pandas as pd

from risk_report import resumen_por_zona


def _poblacion_dos_zonas():
    """Zona A: 10 familias, 8 ALTO (concentración fuerte).
    Zona B: 10 familias, 1 ALTO (concentración baja)."""
    filas = []
    for i in range(10):
        filas.append({
            "colonia": "Zona A",
            "riesgo_predicho": "ALTO" if i < 8 else "BAJO",
            "requiere_atencion_prioritaria": i < 8,
        })
    for i in range(10):
        filas.append({
            "colonia": "Zona B",
            "riesgo_predicho": "ALTO" if i < 1 else "BAJO",
            "requiere_atencion_prioritaria": i < 1,
        })
    return pd.DataFrame(filas)


def test_zona_con_mas_concentracion_de_riesgo_queda_primero():
    poblacion = _poblacion_dos_zonas()
    resumen = resumen_por_zona(poblacion, columna_zona="colonia")
    assert resumen.iloc[0]["zona"] == "Zona A"
    assert resumen.iloc[0]["pct_alto_o_bandera"] > resumen.iloc[1]["pct_alto_o_bandera"]


def test_columnas_y_totales_correctos():
    poblacion = _poblacion_dos_zonas()
    resumen = resumen_por_zona(poblacion, columna_zona="colonia")
    fila_a = resumen[resumen["zona"] == "Zona A"].iloc[0]
    assert fila_a["total_familias"] == 10
    assert fila_a["n_alto"] == 8
    assert fila_a["pct_alto"] == 0.8


def test_nivel_alerta_zona_es_relativo_no_fijo():
    """Con solo 2 zonas, la de mayor concentración debe salir ALTO y la otra
    no debe salir también ALTO (el corte por percentil debe discriminar)."""
    poblacion = _poblacion_dos_zonas()
    resumen = resumen_por_zona(poblacion, columna_zona="colonia")
    niveles = dict(zip(resumen["zona"], resumen["nivel_alerta_zona"]))
    assert niveles["Zona A"] != niveles["Zona B"]


def test_columna_zona_faltante_lanza_keyerror():
    poblacion = _poblacion_dos_zonas().drop(columns=["colonia"])
    try:
        resumen_por_zona(poblacion, columna_zona="colonia")
        assert False, "debia lanzar KeyError"
    except KeyError:
        pass


def test_sin_columna_de_banderas_no_rompe():
    """Compatibilidad hacia atrás: si el df no trae banderas, sigue
    funcionando usando solo riesgo_predicho."""
    poblacion = _poblacion_dos_zonas().drop(columns=["requiere_atencion_prioritaria"])
    resumen = resumen_por_zona(poblacion, columna_zona="colonia")
    assert len(resumen) == 2
    assert resumen.iloc[0]["zona"] == "Zona A"
