# -*- coding: utf-8 -*-
"""Pruebas de monitoreo.py (PSI): sin drift, con drift fuerte, e interpretación."""
import numpy as np
import pandas as pd
import pytest

from monitoreo import calcular_psi, interpretar_psi, psi_por_feature


def test_psi_identica_es_practicamente_cero():
    rng = np.random.RandomState(0)
    serie = pd.Series(rng.normal(loc=10, scale=2, size=1000))
    psi = calcular_psi(serie, serie, n_bins=10)
    assert psi == pytest.approx(0.0, abs=1e-9)
    assert interpretar_psi(psi) == "sin cambio significativo"


def test_psi_distribucion_desplazada_es_alto():
    rng = np.random.RandomState(0)
    esperado = pd.Series(rng.normal(loc=10, scale=2, size=1000))
    # Desplaza toda la distribución varias desviaciones estándar: debe disparar
    # el umbral "significativo" (>= 0.25), no solo "moderado".
    real = esperado + 15
    psi = calcular_psi(esperado, real, n_bins=10)
    assert psi >= 0.25
    assert interpretar_psi(psi) == "cambio significativo, reentrenar el modelo"


def test_psi_desplazamiento_leve_es_moderado_o_menor():
    rng = np.random.RandomState(1)
    esperado = pd.Series(rng.normal(loc=10, scale=2, size=2000))
    # Un desplazamiento pequeño (una fracción de la desviación estándar) no
    # debería disparar el umbral más alto.
    real = esperado + 0.3
    psi = calcular_psi(esperado, real, n_bins=10)
    assert psi < 0.25


def test_calcular_psi_rechaza_series_vacias():
    with pytest.raises(ValueError):
        calcular_psi(pd.Series([], dtype=float), pd.Series([1.0]))


def test_psi_por_feature_ordena_de_mayor_a_menor():
    rng = np.random.RandomState(2)
    df_esperado = pd.DataFrame({
        "estable": rng.normal(0, 1, 500),
        "con_drift": rng.normal(0, 1, 500),
    })
    df_real = pd.DataFrame({
        "estable": rng.normal(0, 1, 500),
        "con_drift": rng.normal(20, 1, 500),  # muy desplazada
    })
    tabla = psi_por_feature(df_esperado, df_real, ["estable", "con_drift"])
    assert list(tabla["feature"])[0] == "con_drift"
    assert tabla.iloc[0]["psi"] > tabla.iloc[1]["psi"]
    assert tabla.iloc[0]["interpretacion"] == "cambio significativo, reentrenar el modelo"
