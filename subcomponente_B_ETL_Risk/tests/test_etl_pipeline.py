"""
test_etl_pipeline.py
=====================
Pruebas unitarias de etl_pipeline.py: `clean_and_transform` (coerción de
booleanos, imputación numérica/categórica, recálculo de hacinamiento) y
`build_xy` (validación de columnas y forma del X/y resultante).
"""

import numpy as np
import pandas as pd
import pytest

from etl_pipeline import (
    CLASES_ORDEN,
    FEATURES,
    TARGET,
    build_xy,
    clean_and_transform,
)


# ─────────────────────────────────────────────────────────────────────────────
# clean_and_transform — coerción de booleanos
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "valores_entrada, esperado",
    [
        (["True", "False", "True", "False"], [True, False, True, False]),
        (["1", "0", "1", "0"], [True, False, True, False]),
        (["VERDADERO", "FALSO", "VERDADERO", "FALSO"], [True, False, True, False]),
        ([True, False, True, False], [True, False, True, False]),  # bool nativo
    ],
)
def test_clean_and_transform_coerce_booleanos(valores_entrada, esperado):
    df = pd.DataFrame({"agua_entubada": valores_entrada})
    resultado = clean_and_transform(df)

    assert resultado["agua_entubada"].dtype == bool
    assert resultado["agua_entubada"].tolist() == esperado


def test_clean_and_transform_coerce_booleanos_representaciones_mixtas():
    """Una sola columna con una mezcla de representaciones (string, bool nativo)."""
    df = pd.DataFrame(
        {"vacunacion_completa": ["True", "0", "VERDADERO", False, "1", "FALSO"]}
    )
    resultado = clean_and_transform(df)

    assert resultado["vacunacion_completa"].dtype == bool
    assert resultado["vacunacion_completa"].tolist() == [
        True, False, True, False, True, False,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# clean_and_transform — imputación numérica (mediana)
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_and_transform_imputa_numericas_con_mediana():
    df = pd.DataFrame({"numero_cuartos": [1, 2, 3, np.nan, 5]})
    resultado = clean_and_transform(df)

    assert not resultado["numero_cuartos"].isna().any()
    mediana_esperada = pd.Series([1, 2, 3, 5]).median()  # 2.5
    assert resultado["numero_cuartos"].iloc[3] == mediana_esperada


def test_clean_and_transform_no_altera_numericas_sin_nulos():
    df = pd.DataFrame({"numero_habitantes": [1, 2, 3, 4]})
    resultado = clean_and_transform(df)
    assert resultado["numero_habitantes"].tolist() == [1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────────────────────
# clean_and_transform — imputación categórica (moda)
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_and_transform_imputa_categoricas_con_moda():
    df = pd.DataFrame(
        {"material_techo": ["Madera", "Madera", "Concreto o cemento", np.nan]}
    )
    resultado = clean_and_transform(df)

    assert not resultado["material_techo"].isna().any()
    assert resultado["material_techo"].iloc[3] == "Madera"


# ─────────────────────────────────────────────────────────────────────────────
# clean_and_transform — recálculo de personas_por_cuarto sin división por cero
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_and_transform_personas_por_cuarto_sin_division_por_cero():
    df = pd.DataFrame(
        {
            "numero_habitantes": [4, 5, 6],
            "numero_cuartos": [0, 2, 3],
        }
    )
    resultado = clean_and_transform(df)

    assert not resultado["personas_por_cuarto"].isna().any()
    assert not np.isinf(resultado["personas_por_cuarto"]).any()
    # numero_cuartos=0 se trata como 1 (evita división por cero).
    assert resultado["personas_por_cuarto"].iloc[0] == pytest.approx(4.0)
    assert resultado["personas_por_cuarto"].iloc[1] == pytest.approx(2.5)
    assert resultado["personas_por_cuarto"].iloc[2] == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# build_xy
# ─────────────────────────────────────────────────────────────────────────────

def test_build_xy_lanza_keyerror_si_falta_una_columna_de_features(df_sintetico_pequeno):
    df_incompleto = df_sintetico_pequeno.drop(columns=["numero_cuartos"])
    with pytest.raises(KeyError):
        build_xy(df_incompleto)


def test_build_xy_lanza_keyerror_si_falta_el_target(df_sintetico_pequeno):
    df_incompleto = df_sintetico_pequeno.drop(columns=[TARGET])
    with pytest.raises(KeyError):
        build_xy(df_incompleto)


def test_build_xy_con_df_valido_devuelve_columnas_exactas_en_orden(df_sintetico_pequeno):
    X, y = build_xy(df_sintetico_pequeno)

    assert list(X.columns) == FEATURES
    assert len(X) == len(df_sintetico_pequeno)
    assert isinstance(y, pd.Series)
    assert all(isinstance(v, str) for v in y)
    assert set(y.unique()).issubset(set(CLASES_ORDEN))
