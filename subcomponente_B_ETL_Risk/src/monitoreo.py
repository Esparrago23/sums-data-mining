# -*- coding: utf-8 -*-
"""
monitoreo.py
============
Subcomponente B — Monitoreo de deriva (drift) del modelo de riesgo.

Con datos SINTÉTICOS de un solo snapshot no hay deriva real que medir (el label
`nivel_riesgo` es una función determinística de los features, ver etl_pipeline.py:
"ANTI-LEAKAGE"). Este módulo existe para cuando el modelo se conecte a datos
reales de IMSS-BIENESTAR y haya que decidir CUÁNDO reentrenar: no en un
calendario arbitrario, sino cuando la distribución de los datos que llegan hoy
se aleja demasiado de la distribución con la que se entrenó el modelo (data
drift / covariate shift).

PSI (Population Stability Index) es la métrica estándar de la industria para
esto — viene de scoring de riesgo crediticio, el mismo tipo de problema que
este subcomponente resuelve (clasificar personas/familias en niveles de riesgo).

Regla de interpretación (estándar de la industria, no inventada aquí):
  PSI < 0.10            -> sin cambio significativo, no actuar.
  0.10 <= PSI < 0.25     -> cambio moderado, vigilar de cerca.
  PSI >= 0.25            -> cambio significativo, reentrenar el modelo.

Uso previsto en producción: cada vez que se acumule un lote nuevo de familias
capturadas (ej. mensual), comparar sus features numéricas contra el dataset con
el que se entrenó el modelo activo. Si alguna feature clave supera PSI >= 0.25,
disparar reentrenamiento (además del disparo por antigüedad del cache que ya
implementa `integracion/api_mineria.py::_cargar_o_entrenar_modelo`, basado en
si cambió el CSV fuente).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_UMBRAL_MODERADO = 0.10
PSI_UMBRAL_SIGNIFICATIVO = 0.25


def _bins_por_cuantiles(esperado: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Bordes de bin por cuantiles de la distribución ESPERADA (de referencia/entrenamiento).

    Usar cuantiles de `esperado` (no de `real`) es la convención estándar de
    PSI: los bins se definen sobre la distribución base, y luego se mide cómo
    se redistribuye `real` sobre esos MISMOS bins.
    """
    cuantiles = np.linspace(0, 1, n_bins + 1)
    bordes = np.unique(esperado.quantile(cuantiles).to_numpy())
    if len(bordes) < 3:
        # Distribución casi constante (muy poca varianza): evita crashear con
        # menos de 2 bins válidos, usando min/max como único corte.
        bordes = np.array([esperado.min(), esperado.max()])
    bordes = bordes.astype(float)
    bordes[0] = -np.inf
    bordes[-1] = np.inf
    return bordes


def calcular_psi(esperado: pd.Series, real: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index entre dos distribuciones de una feature NUMÉRICA.

    `esperado`: distribución de referencia (ej. datos de entrenamiento).
    `real`: distribución nueva a comparar (ej. un lote reciente de producción).

    PSI = sum( (pct_real - pct_esperado) * ln(pct_real / pct_esperado) )
    sobre cada bin (cuantiles de `esperado`). Los bins con 0 observaciones se
    suavizan con un epsilon para evitar log(0) / división por cero — sin este
    suavizado, un solo bin vacío haría que PSI fuera infinito o NaN.
    """
    if len(esperado) == 0 or len(real) == 0:
        raise ValueError("calcular_psi requiere series no vacías.")

    bordes = _bins_por_cuantiles(esperado, n_bins)
    eps = 1e-4

    conteo_esperado = pd.cut(esperado, bordes).value_counts(sort=False)
    conteo_real = pd.cut(real, bordes).value_counts(sort=False)

    pct_esperado = (conteo_esperado / len(esperado)).clip(lower=eps)
    pct_real = (conteo_real.reindex(pct_esperado.index, fill_value=0) / len(real)).clip(lower=eps)

    return float(np.sum((pct_real - pct_esperado) * np.log(pct_real / pct_esperado)))


def interpretar_psi(psi: float) -> str:
    """Traduce un valor de PSI a la recomendación operativa estándar."""
    if psi < PSI_UMBRAL_MODERADO:
        return "sin cambio significativo"
    if psi < PSI_UMBRAL_SIGNIFICATIVO:
        return "cambio moderado, vigilar de cerca"
    return "cambio significativo, reentrenar el modelo"


def psi_por_feature(
    df_esperado: pd.DataFrame, df_real: pd.DataFrame, features: list[str], n_bins: int = 10
) -> pd.DataFrame:
    """PSI de cada feature NUMÉRICA en `features`, para ver cuáles cambiaron
    más entre el dataset de entrenamiento (`df_esperado`) y un lote nuevo
    (`df_real`). Devuelve un DataFrame ordenado de mayor a menor PSI (las
    features más urgentes de vigilar aparecen primero)."""
    filas = []
    for col in features:
        psi = calcular_psi(df_esperado[col], df_real[col], n_bins=n_bins)
        filas.append({"feature": col, "psi": round(psi, 4), "interpretacion": interpretar_psi(psi)})
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Smoke test manual: compara el dataset sintético contra sí mismo (PSI ~ 0)
    # y contra una versión desplazada artificialmente (PSI alto), para ilustrar
    # la métrica sin necesitar datos reales de producción todavía.
    from etl_pipeline import FEATURES_NUMERICAS, load_dataset

    df, X, y = load_dataset()
    print("=== PSI: dataset contra sí mismo (debe ser ~0, sin drift) ===")
    print(psi_por_feature(df, df, FEATURES_NUMERICAS))

    df_desplazado = df.copy()
    df_desplazado["numero_habitantes"] = df_desplazado["numero_habitantes"] + 5
    df_desplazado["ingreso_nivel"] = 0  # simula que TODAS las familias nuevas caen en el nivel más bajo
    print("\n=== PSI: dataset vs. una versión desplazada artificialmente ===")
    print(psi_por_feature(df, df_desplazado, FEATURES_NUMERICAS))
