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

from etl_pipeline import COLUMNAS_BANDERAS, COLUMNAS_IDENTIFICACION, FEATURES
from grupos_vulnerables import motivo_prioridad
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


def _predecir_y_anotar(df: pd.DataFrame, pipe: Pipeline, le: LabelEncoder | None) -> tuple[pd.DataFrame, bool]:
    """Predice sobre TODO `df` (sin filtrar) y anota riesgo_predicho,
    probabilidad_alto y motivo_prioridad. Compartido por `generar_lista_visitas`
    (filtra a la lista priorizada) y `evaluar_riesgo_poblacional` (mantiene la
    población completa, para el resumen por zona)."""
    X = df[FEATURES].copy()
    pred, prob_alto = _predecir(pipe, X, le)

    out = df.copy()
    out["riesgo_predicho"] = pred
    out["probabilidad_alto"] = np.round(prob_alto, 4)

    banderas_cols = [c for c in COLUMNAS_BANDERAS if c in out.columns]
    tiene_banderas = bool(banderas_cols) and "requiere_atencion_prioritaria" in banderas_cols
    if tiene_banderas:
        out["motivo_prioridad"] = out.apply(
            lambda fila: motivo_prioridad(
                {c: fila[c] for c in banderas_cols}, nivel_riesgo_ml=fila["riesgo_predicho"]
            ),
            axis=1,
        )
    return out, tiene_banderas


def generar_lista_visitas(
    df: pd.DataFrame,
    pipe: Pipeline,
    le: LabelEncoder | None = None,
    processed_dir: str = PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Predice sobre todo `df`, arma la lista priorizada de visitas y la guarda.

    La lista incluye una familia si el modelo la clasifica ALTO **O** si tiene
    alguna bandera de grupo vulnerable activa (embarazada, menor de 1 año,
    menor de 5 sin vacunas, adulto mayor solo, mascota sin vacunar -- ver
    grupos_vulnerables.py). Esto es intencional: una familia con buena
    vivienda/ingreso puede salir BAJO/MEDIO en el score agregado y aun así
    tener una necesidad puntual urgente que el score no ve (el score se
    calcula sobre conteos/promedios de la familia, no sobre la composición
    individual). Se ordena primero por bandera (hecho observado, no
    probabilístico) y dentro de cada grupo por probabilidad de ALTO, para que
    estos casos no queden enterrados debajo de familias ALTO sin ninguna
    bandera.

    Devuelve el DataFrame de la lista priorizada (índice 'prioridad', 1..N).
    """
    os.makedirs(processed_dir, exist_ok=True)

    out, tiene_banderas = _predecir_y_anotar(df, pipe, le)

    if tiene_banderas:
        incluir = (out["riesgo_predicho"] == CLASE_ALTO) | out["requiere_atencion_prioritaria"]
    else:
        # Compatibilidad hacia atrás: si el df no trae las banderas (CSV viejo
        # sin regenerar), se comporta igual que antes de este cambio.
        incluir = out["riesgo_predicho"] == CLASE_ALTO

    # Columnas de identificación disponibles (las que existan en el df).
    id_cols = [c for c in COLUMNAS_IDENTIFICACION if c in out.columns]
    extra_cols = [c for c in ["score_total", "requiere_atencion_prioritaria"] if c in out.columns]
    motivo_col = ["motivo_prioridad"] if tiene_banderas else []
    cols_reporte = id_cols + ["riesgo_predicho", "probabilidad_alto"] + motivo_col + extra_cols

    seleccion = out[incluir].copy()
    if tiene_banderas:
        orden = seleccion.sort_values(
            ["requiere_atencion_prioritaria", "probabilidad_alto"], ascending=[False, False]
        )
    else:
        orden = seleccion.sort_values("probabilidad_alto", ascending=False)

    lista = orden[cols_reporte].reset_index(drop=True)
    lista.index = lista.index + 1
    lista.index.name = "prioridad"

    out_path = os.path.join(processed_dir, "lista_visitas_prioritarias.csv")
    lista.to_csv(out_path, encoding="utf-8")

    return lista


def evaluar_riesgo_poblacional(df: pd.DataFrame, pipe: Pipeline, le: LabelEncoder | None = None) -> pd.DataFrame:
    """Predice sobre TODA la población (no solo la lista priorizada) y
    devuelve el DataFrame completo anotado -- insumo de `resumen_por_zona`.
    """
    out, _ = _predecir_y_anotar(df, pipe, le)
    return out


def resumen_por_zona(
    out_poblacional: pd.DataFrame, columna_zona: str = "colonia", processed_dir: str = PROCESSED_DIR
) -> pd.DataFrame:
    """MEJORA — riesgo de CÚMULO GEOGRÁFICO (no de contagio persona-a-persona):
    agrupa la población completa por `columna_zona` (colonia por defecto) y
    mide qué tan concentrado está el riesgo ahí, en vez de solo mirar familia
    por familia.

    Motivación: si una familia tiene un padecimiento transmisible, sus VECINOS
    podrían estar en riesgo -- pero hoy no existe en el modelo de datos un
    campo estructurado de "enfermedad transmisible activa" (las enfermedades
    de catalogos_sums.CAT_ENFERMEDAD_CRONICA son crónicas no transmisibles;
    lo más cercano es el TEMA de una nota de texto libre en el subcomponente
    C, que ya sabemos que casi nunca se llena -- ver buscador_estructurado.py).
    Por eso esto es un PROXY honesto con los datos que sí existen: zonas con
    concentración anormal de familias ALTO/con-bandera son candidatas a
    visitarse como zona, no familia por familia, mientras no exista un campo
    real de enfermedad transmisible para modelar contagio directo.

    Devuelve un DataFrame ordenado de mayor a menor `pct_alto_o_bandera`, con
    columnas: zona, total_familias, n_alto, pct_alto, n_con_bandera,
    pct_con_bandera, pct_alto_o_bandera, nivel_alerta_zona.
    """
    if columna_zona not in out_poblacional.columns:
        raise KeyError(f"'{columna_zona}' no está en el DataFrame poblacional.")

    tiene_bandera_col = "requiere_atencion_prioritaria" in out_poblacional.columns
    df = out_poblacional.copy()
    df["_es_alto"] = df["riesgo_predicho"] == CLASE_ALTO
    if tiene_bandera_col:
        df["_alto_o_bandera"] = df["_es_alto"] | df["requiere_atencion_prioritaria"]
    else:
        df["_alto_o_bandera"] = df["_es_alto"]

    filas = []
    for zona, grupo in df.groupby(columna_zona):
        total = len(grupo)
        n_alto = int(grupo["_es_alto"].sum())
        n_bandera = int(grupo["requiere_atencion_prioritaria"].sum()) if tiene_bandera_col else 0
        n_alto_o_bandera = int(grupo["_alto_o_bandera"].sum())
        pct_alto_o_bandera = n_alto_o_bandera / total if total else 0.0

        filas.append({
            "zona": zona,
            "total_familias": total,
            "n_alto": n_alto,
            "pct_alto": round(n_alto / total, 4) if total else 0.0,
            "n_con_bandera": n_bandera,
            "pct_con_bandera": round(n_bandera / total, 4) if total else 0.0,
            "pct_alto_o_bandera": round(pct_alto_o_bandera, 4),
        })

    resumen = (
        pd.DataFrame(filas)
        .sort_values("pct_alto_o_bandera", ascending=False)
        .reset_index(drop=True)
    )

    # Nivel de alerta por RANGO relativo (percentil), no por un umbral fijo
    # arbitrario: el 20% de zonas con mayor concentración de riesgo = ALTO, el
    # siguiente 30% = MEDIO, el resto = BAJO. Esto siempre produce una
    # clasificación accionable ("visiten estas zonas primero") sin importar
    # qué tan dispersos o parejos estén los porcentajes en el dataset -- un
    # umbral fijo (ej. "1.5x el promedio") puede no discriminar nada si el
    # riesgo está repartido parejo entre zonas (como pasa en el dataset
    # sintético actual, donde la vulnerabilidad se asigna al azar por familia,
    # no por colonia -- una limitación honesta a declarar, no a ocultar).
    n_zonas = len(resumen)
    corte_alto = max(1, round(n_zonas * 0.2))
    corte_medio = max(corte_alto, round(n_zonas * 0.5))
    resumen["nivel_alerta_zona"] = "BAJO"
    resumen.loc[:corte_alto - 1, "nivel_alerta_zona"] = "ALTO"
    resumen.loc[corte_alto:corte_medio - 1, "nivel_alerta_zona"] = "MEDIO"

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "resumen_riesgo_por_zona.csv")
    resumen.to_csv(out_path, index=False, encoding="utf-8")

    return resumen


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
