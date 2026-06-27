"""
etl_pipeline.py
===============
Subcomponente B — Fase B-2/B-3: ETL (Extracción, Transformación, Carga) del
dataset de microdiagnóstico familiar del SUMS y definición de las listas de
features para el modelo de riesgo.

Fuente de datos:
  - Desarrollo: CSV sintético `data/synthetic_data.csv` (4000 familias).
  - Producción: la BD `centro_medico_2026` (query en `extract_from_db`, no usada
    aún porque IMSS-BIENESTAR no ha entregado datos reales).

ANTI-LEAKAGE (regla crítica del subcomponente):
  El label `nivel_riesgo` es una función determinística del score `score_total`,
  que a su vez es la suma de reglas sobre los features. Por eso:
    - Se EXCLUYE `score_total` de los features (es la respuesta casi directa).
    - Se EXCLUYEN identificadores de texto (`nombre_informante`, `domicilio`,
      `colonia`, `localidad`) que no son señal clínica y sí podrían filtrar
      información geográfica espuria.
  Es esperable y honesto que el accuracy sea alto: el label proviene de reglas.
  Esto se declara como LIMITACIÓN CONOCIDA, no se oculta.

pandas 3.0:
  - NO se usa `df[col].fillna(x, inplace=True)` (roto en 3.0). Se usa la forma
    `df[col] = df[col].fillna(x)`.
  - Booleanos: se fuerzan a bool real por si el motor de lectura los entrega como
    strings 'True'/'False'.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de columnas
# ─────────────────────────────────────────────────────────────────────────────

# Ruta por defecto al CSV sintético (relativa a la raíz del subcomponente).
DEFAULT_CSV = os.path.join("data", "synthetic_data.csv")

TARGET = "nivel_riesgo"

# Orden canónico de las clases (de menor a mayor riesgo) para reportes/gráficas.
CLASES_ORDEN = ["BAJO", "MEDIO", "ALTO"]

# Features NUMÉRICAS → SimpleImputer(mediana) + StandardScaler.
FEATURES_NUMERICAS = [
    "numero_cuartos",
    "numero_habitantes",
    "personas_por_cuarto",
    "count_enfermedades_cronicas",
    "count_toxicomanias",
    "avg_dias_proteina",
    "avg_dias_frutas_verduras",
    "avg_dias_cereales",
    "ingreso_nivel",
    "escolaridad_promedio",
    "total_integrantes",
]

# Features CATEGÓRICAS → SimpleImputer(moda) + OneHotEncoder(handle_unknown='ignore').
FEATURES_CATEGORICAS = [
    "material_techo",
    "material_paredes",
    "material_piso",
    "manejo_excretas",
    "cocina_ubicacion",
]

# Features BOOLEANAS → passthrough (ya son 0/1 una vez convertidas a bool).
FEATURES_BOOLEANAS = [
    "agua_entubada",
    "energia_electrica",
    "cocina_con_lena",
    "red_alcantarillado",
    "fosa_septica",
    "vacunacion_completa",
    "seguridad_social_jefe",
]

# Lista completa de features que entran al modelo (orden estable).
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS + FEATURES_BOOLEANAS

# Columnas EXCLUIDAS a propósito (documentado por anti-leakage / identificadores).
COLUMNAS_EXCLUIDAS = [
    "score_total",        # respuesta casi directa (el label se deriva de él)
    "nombre_informante",  # identificador de texto libre
    "domicilio",          # identificador de texto libre
    "colonia",            # geografía: ruido espurio para el clasificador
    "localidad",          # geografía: ruido espurio para el clasificador
]

# Columnas de identificación que SÍ se conservan para el reporte de visitas
# (no entran al modelo, pero sirven para la lista priorizada).
COLUMNAS_IDENTIFICACION = [
    "nombre_informante",
    "domicilio",
    "colonia",
    "localidad",
]

# Mapa robusto de strings → bool por si el CSV trae 'True'/'False' como texto.
_BOOL_MAP = {
    "True": True, "False": False,
    "true": True, "false": False,
    "1": True, "0": False,
    "VERDADERO": True, "FALSO": False,
    True: True, False: False,
    1: True, 0: False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Extracción
# ─────────────────────────────────────────────────────────────────────────────

def extract_synthetic(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    """Lee el CSV sintético en UTF-8 (modo desarrollo)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró el CSV sintético en '{csv_path}'. "
            "Genera el dataset con synthetic_generator.py o ajusta la ruta."
        )
    df = pd.read_csv(csv_path, encoding="utf-8")
    return df


def extract_from_db() -> pd.DataFrame:  # pragma: no cover - placeholder producción
    """
    Query real sobre `centro_medico_2026` (para cuando IMSS entregue datos reales).
    Se deja como referencia; en desarrollo se usa `extract_synthetic`.
    """
    raise NotImplementedError(
        "Extracción desde BD pendiente: usar extract_synthetic() en desarrollo."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Limpieza y transformación
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas booleanas a bool real. pandas 3.0 con el motor por
    defecto suele entregarlas ya como bool, pero por robustez aceptamos también
    'True'/'False' como strings (otros motores/exports).
    """
    for col in FEATURES_BOOLEANAS:
        if col in df.columns:
            if df[col].dtype != bool:
                df[col] = df[col].map(_BOOL_MAP)
                # Si quedó algún NaN tras el mapeo, lo imputamos a False (ausencia).
                df[col] = df[col].fillna(False).astype(bool)
            else:
                df[col] = df[col].astype(bool)
    return df


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza determinística del dataset:
      1. Normaliza booleanos a bool real.
      2. Imputa nulos (mediana en numéricas, moda en categóricas) — pandas 3.0 safe.
      3. Recalcula `personas_por_cuarto` por coherencia (hacinamiento).
    NO toca `score_total` ni `nivel_riesgo` (vienen ya calculados del generador).
    """
    df = df.copy()
    df = _coerce_booleans(df)

    # Imputación numérica (mediana). Forma pandas 3.0 (sin inplace).
    for col in FEATURES_NUMERICAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    # Imputación categórica (moda). Forma pandas 3.0 (sin inplace).
    for col in FEATURES_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype("object")
            if df[col].isna().any():
                moda = df[col].mode(dropna=True)
                relleno = moda.iloc[0] if len(moda) else "DESCONOCIDO"
                df[col] = df[col].fillna(relleno)

    # Recalcular hacinamiento por coherencia (evita división por cero).
    if {"numero_habitantes", "numero_cuartos"}.issubset(df.columns):
        cuartos = df["numero_cuartos"].replace(0, 1)
        df["personas_por_cuarto"] = (df["numero_habitantes"] / cuartos).astype(float)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de X, y
# ─────────────────────────────────────────────────────────────────────────────

def build_xy(df: pd.DataFrame):
    """
    Devuelve (X, y) listos para el Pipeline.
      - X: solo las columnas de FEATURES (sin score_total ni identificadores).
      - y: la columna TARGET como pandas Series de strings (ALTO/MEDIO/BAJO).
    """
    faltantes = [c for c in FEATURES if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas de features en el dataset: {faltantes}")
    if TARGET not in df.columns:
        raise KeyError(f"Falta la columna objetivo '{TARGET}' en el dataset.")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(str).copy()
    return X, y


def load_dataset(csv_path: str = DEFAULT_CSV):
    """
    Atajo end-to-end: extrae, limpia y devuelve (df_limpio, X, y).
    `df_limpio` conserva TODAS las columnas (incluye identificadores y score)
    para el reporte de visitas; X solo trae los features del modelo.
    """
    df_raw = extract_synthetic(csv_path)
    df = clean_and_transform(df_raw)
    X, y = build_xy(df)
    return df, X, y


# ─────────────────────────────────────────────────────────────────────────────
# EDA opcional (resúmenes ligeros sin dependencias de gráficos)
# ─────────────────────────────────────────────────────────────────────────────

def eda_summary(df: pd.DataFrame) -> dict:
    """
    Resumen exploratorio ligero (para imprimir en run_all o el notebook).
    Devuelve un dict con: distribución de clases, nulos, correlaciones top y
    estadísticas numéricas. No grafica; solo agrega texto/numérico.
    """
    resumen: dict = {}
    resumen["n_filas"] = int(len(df))
    resumen["n_columnas"] = int(df.shape[1])

    if TARGET in df.columns:
        dist = df[TARGET].value_counts()
        dist = dist.reindex([c for c in CLASES_ORDEN if c in dist.index])
        resumen["distribucion_clases"] = dist.to_dict()
        resumen["balance_clases_pct"] = (
            (dist / dist.sum() * 100).round(2).to_dict()
        )

    nulos = df.isnull().sum()
    resumen["columnas_con_nulos"] = nulos[nulos > 0].to_dict()

    num_cols = [c for c in FEATURES_NUMERICAS if c in df.columns]
    if num_cols:
        resumen["describe_numericas"] = (
            df[num_cols].describe().round(3).to_dict()
        )
        # Correlación de Pearson de las numéricas con el score (si existe).
        if "score_total" in df.columns:
            corr = (
                df[num_cols + ["score_total"]]
                .corr(numeric_only=True)["score_total"]
                .drop("score_total")
                .sort_values(ascending=False)
            )
            resumen["corr_numericas_vs_score"] = corr.round(3).to_dict()

    return resumen


if __name__ == "__main__":
    # Smoke test manual del ETL.
    df, X, y = load_dataset()
    print(f"Dataset cargado: {df.shape[0]} familias, {X.shape[1]} features.")
    print(f"Features numéricas:   {len(FEATURES_NUMERICAS)}")
    print(f"Features categóricas: {len(FEATURES_CATEGORICAS)}")
    print(f"Features booleanas:   {len(FEATURES_BOOLEANAS)}")
    print(f"Excluidas (anti-leakage/identificadores): {COLUMNAS_EXCLUIDAS}")
    print("\nDistribución del target:")
    print(y.value_counts())
    print("\nDtypes de X:")
    print(X.dtypes)
