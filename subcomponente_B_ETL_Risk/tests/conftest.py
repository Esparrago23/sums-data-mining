"""
conftest.py
===========
Fixtures compartidos para la suite de pytest del subcomponente B (ETL + modelo
de riesgo).

- Agrega `src/` al sys.path para que los tests puedan hacer
  `import etl_pipeline` / `import model_trainer` directamente, igual que
  run_all.py y los notebooks del subcomponente.
- Expone `df_sintetico_pequeno`: un DataFrame pequeño (180 filas, 60 por clase)
  que respeta EXACTAMENTE el esquema de columnas que espera el pipeline
  (etl_pipeline.FEATURES_NUMERICAS + FEATURES_CATEGORICAS + FEATURES_BOOLEANAS
  + la columna TARGET "nivel_riesgo"), construido con valores de catálogo
  reales (catalogos_sums.py) y de forma determinística (RandomState(0)) para
  que los tests sean reproducibles. No se usa el CSV sintético completo de
  4000 filas: esto mantiene la suite en segundos, no minutos.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from etl_pipeline import CLASES_ORDEN  # noqa: E402
from catalogos_sums import (  # noqa: E402
    CAT_MANEJO_EXCRETAS,
    CAT_MATERIAL_PISO,
    CAT_MATERIAL_TECHO_PAREDES,
)

# cocina_ubicacion no es un catálogo de BD (no vive en catalogos_sums.py): es
# el booleano de texto que captura la cédula SUMS directamente.
COCINA_UBICACION_VALORES = ["fuera_del_dormitorio", "dentro_del_dormitorio"]

FILAS_POR_CLASE = 60  # 60 * 3 clases = 180 filas totales (>> mínimo ~15/clase)

# Pesos de muestreo por clase de riesgo, en el MISMO orden que el catálogo real
# (de "mejor" a "peor" condición), para que el dataset sintético sea plausible:
# a mayor riesgo, mayor probabilidad de condiciones de vivienda precarias.
_PESOS_TECHO_PAREDES = {  # catálogo: ['Concreto o cemento', 'Madera', 'Lámina']
    "BAJO": [0.80, 0.15, 0.05],
    "MEDIO": [0.40, 0.40, 0.20],
    "ALTO": [0.10, 0.30, 0.60],
}
_PESOS_PISO = {  # catálogo: ['Concreto o cemento', 'Madera', 'Tierra']
    "BAJO": [0.80, 0.15, 0.05],
    "MEDIO": [0.40, 0.35, 0.25],
    "ALTO": [0.10, 0.20, 0.70],
}
_PESOS_EXCRETAS = {  # catálogo: ['WC', 'Letrina', 'Al ras de suelo']
    "BAJO": [0.85, 0.10, 0.05],
    "MEDIO": [0.45, 0.40, 0.15],
    "ALTO": [0.10, 0.35, 0.55],
}
_PESOS_COCINA = {  # ['fuera_del_dormitorio', 'dentro_del_dormitorio']
    "BAJO": [0.90, 0.10],
    "MEDIO": [0.60, 0.40],
    "ALTO": [0.25, 0.75],
}

# Rangos (min, max_exclusivo) por clase para cada feature numérica.
_RANGOS_NUMERICOS = {
    "BAJO": dict(
        numero_cuartos=(3, 7), numero_habitantes=(1, 4),
        count_enfermedades_cronicas=(0, 2), count_toxicomanias=(0, 1),
        avg_dias_proteina=(5.0, 7.0), avg_dias_frutas_verduras=(5.0, 7.0),
        avg_dias_cereales=(5.0, 7.0), ingreso_nivel=(3, 6),
        escolaridad_promedio=(4.0, 7.0),
    ),
    "MEDIO": dict(
        numero_cuartos=(2, 5), numero_habitantes=(3, 6),
        count_enfermedades_cronicas=(0, 3), count_toxicomanias=(0, 2),
        avg_dias_proteina=(3.0, 5.0), avg_dias_frutas_verduras=(3.0, 5.0),
        avg_dias_cereales=(3.0, 5.0), ingreso_nivel=(1, 4),
        escolaridad_promedio=(2.0, 5.0),
    ),
    "ALTO": dict(
        numero_cuartos=(1, 3), numero_habitantes=(5, 9),
        count_enfermedades_cronicas=(1, 5), count_toxicomanias=(0, 4),
        avg_dias_proteina=(0.0, 3.0), avg_dias_frutas_verduras=(0.0, 3.0),
        avg_dias_cereales=(0.0, 3.0), ingreso_nivel=(0, 2),
        escolaridad_promedio=(0.0, 3.0),
    ),
}

# Probabilidad de que cada booleano PROTECTOR tome el valor favorable (True).
# `cocina_con_lena` es un factor de RIESGO, así que se usa la probabilidad
# inversa (más riesgo -> más probable que cocinen con leña).
_P_BOOL_FAVORABLE = {"BAJO": 0.90, "MEDIO": 0.55, "ALTO": 0.20}


def _generar_bloque_clase(rng: np.random.RandomState, clase: str, n: int) -> pd.DataFrame:
    """Genera `n` filas sintéticas para la clase de riesgo `clase`."""
    r = _RANGOS_NUMERICOS[clase]

    numero_cuartos = rng.randint(*r["numero_cuartos"], size=n)
    numero_habitantes = rng.randint(*r["numero_habitantes"], size=n)
    cuartos_seguros = np.where(numero_cuartos == 0, 1, numero_cuartos)
    personas_por_cuarto = numero_habitantes / cuartos_seguros

    total_integrantes = numero_habitantes + rng.randint(0, 3, size=n)

    p_bool = _P_BOOL_FAVORABLE[clase]

    data = {
        "numero_cuartos": numero_cuartos,
        "numero_habitantes": numero_habitantes,
        "personas_por_cuarto": personas_por_cuarto,
        "count_enfermedades_cronicas": rng.randint(*r["count_enfermedades_cronicas"], size=n),
        "count_toxicomanias": rng.randint(*r["count_toxicomanias"], size=n),
        "avg_dias_proteina": rng.uniform(*r["avg_dias_proteina"], size=n),
        "avg_dias_frutas_verduras": rng.uniform(*r["avg_dias_frutas_verduras"], size=n),
        "avg_dias_cereales": rng.uniform(*r["avg_dias_cereales"], size=n),
        "ingreso_nivel": rng.randint(*r["ingreso_nivel"], size=n),
        "escolaridad_promedio": rng.uniform(*r["escolaridad_promedio"], size=n),
        "total_integrantes": total_integrantes,
        "material_techo": rng.choice(CAT_MATERIAL_TECHO_PAREDES, size=n, p=_PESOS_TECHO_PAREDES[clase]),
        "material_paredes": rng.choice(CAT_MATERIAL_TECHO_PAREDES, size=n, p=_PESOS_TECHO_PAREDES[clase]),
        "material_piso": rng.choice(CAT_MATERIAL_PISO, size=n, p=_PESOS_PISO[clase]),
        "manejo_excretas": rng.choice(CAT_MANEJO_EXCRETAS, size=n, p=_PESOS_EXCRETAS[clase]),
        "cocina_ubicacion": rng.choice(COCINA_UBICACION_VALORES, size=n, p=_PESOS_COCINA[clase]),
        "agua_entubada": rng.random_sample(size=n) < p_bool,
        "energia_electrica": rng.random_sample(size=n) < p_bool,
        "cocina_con_lena": rng.random_sample(size=n) >= p_bool,  # factor de riesgo (inverso)
        "red_alcantarillado": rng.random_sample(size=n) < p_bool,
        "fosa_septica": rng.random_sample(size=n) < p_bool,
        "vacunacion_completa": rng.random_sample(size=n) < p_bool,
        "seguridad_social_jefe": rng.random_sample(size=n) < p_bool,
        # Factores de riesgo (inverso, igual patrón que cocina_con_lena): más
        # probables en clases de mayor riesgo. Ver MEJORA en compute_risk.
        "tiene_menor_1_anio": rng.random_sample(size=n) >= p_bool,
        "tiene_mascota_sin_vacunar": rng.random_sample(size=n) >= p_bool,
        "nivel_riesgo": [clase] * n,
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def df_sintetico_pequeno() -> pd.DataFrame:
    """
    DataFrame sintético pequeño (180 filas, 60 por clase, balanceado) con el
    esquema exacto que espera el pipeline de riesgo. Determinístico
    (RandomState(0)): mismo resultado en cada corrida de la suite. scope
    "module" para no regenerarlo en cada test del mismo archivo.
    """
    rng = np.random.RandomState(0)
    bloques = [_generar_bloque_clase(rng, clase, FILAS_POR_CLASE) for clase in CLASES_ORDEN]
    df = pd.concat(bloques, ignore_index=True)

    # Mezcla determinística de las filas (no quedan agrupadas por clase).
    orden = rng.permutation(len(df))
    df = df.iloc[orden].reset_index(drop=True)
    return df
