"""
risk_report.py
==============
Subcomponente B — Fase B-6: genera la LISTA PRIORIZADA DE VISITAS preventivas.

Usa el modelo ganador (entrenado en model_trainer) para predecir el nivel de
riesgo sobre TODO el dataset, calcula la probabilidad de la clase ALTO y produce
una lista de las familias clasificadas como ALTO, ordenadas de mayor a menor
probabilidad (las visitas más urgentes primero).

Salida: `data/processed/lista_visitas_prioritarias.csv`.

NOTA de privacidad (hallazgo de auditoría B-2): esta lista reincorpora
nombre/domicilio en claro (columnas de COLUMNAS_IDENTIFICACION) que sí se
excluyen de los FEATURES del modelo (ver etl_pipeline.COLUMNAS_EXCLUIDAS,
anti-leakage). Es intencional para que quien haga la visita sepa a quién
buscar. La mitigación exigida es que ningún endpoint que exponga esta lista
(GET /riesgo/lista en api_mineria.py) esté disponible sin autenticación —
ya se resolvió en B-3 (X-API-Key obligatoria + CORS restringido). Con datos
reales de producción, considerar además cifrar en reposo o reemplazar estas
columnas por un id_familia que solo se resuelva a datos de contacto en un
endpoint separado y auditado.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from etl_pipeline import COLUMNAS_IDENTIFICACION, FEATURES
from model_trainer import resolver_label_encoder

PROCESSED_DIR = os.path.join("data", "processed")
CLASE_ALTO = "ALTO"


def _predecir(pipe: Pipeline, X: pd.DataFrame, le: LabelEncoder | None):
    """
    Predice etiquetas y probabilidad de la clase ALTO.
    Maneja tanto modelos string-nativos (RF/DT) como XGBoost (enteros + LabelEncoder).
    """
    clases = list(pipe.named_steps["clf"].classes_)
    es_xgb = le is not None and all(isinstance(c, (int, np.integer)) for c in clases)

    pred_raw = pipe.predict(X)
    proba = pipe.predict_proba(X)

    if es_xgb:
        # XGB devuelve enteros: mapear a strings y localizar el índice de ALTO.
        clases_str = list(le.inverse_transform(np.array(clases)))
        pred = le.inverse_transform(pred_raw)
    else:
        clases_str = [str(c) for c in clases]
        pred = pred_raw

    if CLASE_ALTO in clases_str:
        idx_alto = clases_str.index(CLASE_ALTO)
        prob_alto = proba[:, idx_alto]
    else:
        prob_alto = np.zeros(len(X))

    return np.asarray(pred).astype(str), prob_alto


def generar_lista_visitas(
    df: pd.DataFrame,
    pipe: Pipeline,
    le: LabelEncoder | None = None,
    processed_dir: str = PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Predice sobre todo `df`, arma la lista priorizada de familias ALTO y la guarda.

    Devuelve el DataFrame de la lista priorizada (índice 'prioridad', 1..N).
    """
    os.makedirs(processed_dir, exist_ok=True)

    X = df[FEATURES].copy()
    pred, prob_alto = _predecir(pipe, X, le)

    out = df.copy()
    out["riesgo_predicho"] = pred
    out["probabilidad_alto"] = np.round(prob_alto, 4)

    # Columnas de identificación disponibles (las que existan en el df).
    id_cols = [c for c in COLUMNAS_IDENTIFICACION if c in out.columns]
    extra_cols = [c for c in ["score_total"] if c in out.columns]
    cols_reporte = (
        id_cols + ["riesgo_predicho", "probabilidad_alto"] + extra_cols
    )

    lista = (
        out[out["riesgo_predicho"] == CLASE_ALTO]
        .sort_values("probabilidad_alto", ascending=False)
        [cols_reporte]
        .reset_index(drop=True)
    )
    lista.index = lista.index + 1
    lista.index.name = "prioridad"

    out_path = os.path.join(processed_dir, "lista_visitas_prioritarias.csv")
    lista.to_csv(out_path, encoding="utf-8")

    return lista


def resumen_predicciones(df: pd.DataFrame, pipe: Pipeline, le: LabelEncoder | None = None) -> dict:
    """Resumen de la distribución de riesgo predicho sobre todo el dataset."""
    X = df[FEATURES].copy()
    pred, _ = _predecir(pipe, X, le)
    s = pd.Series(pred)
    return {
        "n_total": int(len(s)),
        "distribucion_predicha": s.value_counts().to_dict(),
        "n_alto": int((s == CLASE_ALTO).sum()),
    }


if __name__ == "__main__":
    # Smoke test: entrena el ganador y genera la lista.
    from etl_pipeline import load_dataset
    from model_trainer import train_and_evaluate

    df, X, y = load_dataset()
    res = train_and_evaluate(X, y)
    winner = res["winner"]
    pipe = res["fitted"][winner]
    le = resolver_label_encoder(winner, res["label_encoder"])

    lista = generar_lista_visitas(df, pipe, le)
    print(f"Modelo ganador: {winner}")
    print(f"Familias en lista de visitas ALTO: {len(lista)}")
    print(lista.head(10))
