"""
run_all.py
==========
Subcomponente B — Driver end-to-end del pipeline ETL + Modelo de Riesgo.

Ejecuta de punta a punta:
  1. ETL: carga + limpieza del CSV sintético, define features (anti-leakage).
  2. EDA: imprime resumen ligero (distribución de clases, correlaciones).
  3. Entrenamiento: DT vs RF vs XGB, métricas + CV, tabla comparativa, elección.
  4. Reporte: lista priorizada de visitas (familias ALTO por probabilidad).
  5. ASSERTS de cordura: los 3 modelos > 0.5 accuracy, tabla con 3 filas,
     archivos de salida existen, lista priorizada no vacía.

Uso (desde la raíz del subcomponente):
    python src/run_all.py

Sale con código 0 si todo está en verde.
"""

from __future__ import annotations

import os
import sys

# Permitir `python src/run_all.py` desde la raíz del subcomponente: agregamos
# la carpeta de este archivo (src/) al sys.path para resolver los imports locales.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# La raíz del subcomponente es el padre de src/. Trabajamos relativo a ella para
# que las rutas 'data/...' resuelvan sin importar desde dónde se invoque.
ROOT_DIR = os.path.dirname(THIS_DIR)
os.chdir(ROOT_DIR)

from etl_pipeline import (  # noqa: E402
    COLUMNAS_EXCLUIDAS,
    FEATURES,
    FEATURES_BOOLEANAS,
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
    eda_summary,
    load_dataset,
)
from model_trainer import train_and_evaluate  # noqa: E402
from risk_report import generar_lista_visitas, resumen_predicciones  # noqa: E402


def _print_header(titulo: str) -> None:
    print("\n" + "=" * 72)
    print(titulo)
    print("=" * 72)


def main() -> int:
    # ─── 1. ETL ──────────────────────────────────────────────────────────────
    _print_header("FASE 1/4 — ETL (carga + limpieza + features)")
    df, X, y = load_dataset()
    print(f"Familias cargadas        : {len(df)}")
    print(f"Features del modelo      : {len(FEATURES)} "
          f"({len(FEATURES_NUMERICAS)} num, {len(FEATURES_CATEGORICAS)} cat, "
          f"{len(FEATURES_BOOLEANAS)} bool)")
    print(f"Excluidas (anti-leakage) : {COLUMNAS_EXCLUIDAS}")

    # ─── 2. EDA ──────────────────────────────────────────────────────────────
    _print_header("FASE 2/4 — EDA (resumen exploratorio)")
    resumen = eda_summary(df)
    print(f"Distribución de clases   : {resumen.get('distribucion_clases')}")
    print(f"Balance (%)              : {resumen.get('balance_clases_pct')}")
    print(f"Columnas con nulos       : {resumen.get('columnas_con_nulos')}")
    corr = resumen.get("corr_numericas_vs_score", {})
    top_corr = dict(list(corr.items())[:5])
    print(f"Top corr num vs score    : {top_corr}")

    # ─── 3. Entrenamiento + evaluación comparativa ───────────────────────────
    _print_header("FASE 3/4 — Entrenamiento y evaluación comparativa")
    res = train_and_evaluate(X, y)
    comparison = res["comparison"]
    winner = res["winner"]
    print("\nTabla comparativa (ordenada por F1_Macro):\n")
    print(comparison.to_string())
    print(f"\n>>> MODELO GANADOR: {winner}")
    print(f"\nReporte de clasificación del ganador ({winner}):\n")
    print(res["reports"][winner])

    # ─── 4. Reporte: lista priorizada de visitas ─────────────────────────────
    _print_header("FASE 4/4 — Lista priorizada de visitas (familias ALTO)")
    pipe = res["fitted"][winner]
    le = res["label_encoder"] if winner == "XGBoost" else None
    resumen_pred = resumen_predicciones(df, pipe, le)
    print(f"Distribución predicha    : {resumen_pred['distribucion_predicha']}")
    lista = generar_lista_visitas(df, pipe, le)
    print(f"Familias en lista ALTO   : {len(lista)}")
    print("\nTop 10 visitas prioritarias:\n")
    print(lista.head(10).to_string())

    # ─── 5. ASSERTS de cordura ───────────────────────────────────────────────
    _print_header("VERIFICACIÓN — Asserts de cordura")
    paths = res["paths"]

    # (a) Los 3 modelos con accuracy > 0.5.
    assert comparison.shape[0] == 3, (
        f"La tabla comparativa debe tener 3 filas, tiene {comparison.shape[0]}."
    )
    bajos = comparison[comparison["Accuracy"] <= 0.5]
    assert bajos.empty, f"Modelos con accuracy <= 0.5: {bajos.index.tolist()}"
    print("[OK] Tabla comparativa con 3 modelos.")
    print("[OK] Los 3 modelos tienen accuracy > 0.5.")

    # (b) Archivos de salida existen.
    for clave, ruta in paths.items():
        assert os.path.exists(ruta), f"Falta el archivo de salida '{clave}': {ruta}"
    lista_path = os.path.join("data", "processed", "lista_visitas_prioritarias.csv")
    assert os.path.exists(lista_path), f"Falta la lista priorizada: {lista_path}"
    print("[OK] Archivos de salida presentes:")
    for clave, ruta in {**paths, "lista_visitas": lista_path}.items():
        print(f"     - {clave}: {ruta}")

    # (c) La lista priorizada no está vacía y está ordenada por probabilidad.
    assert len(lista) > 0, "La lista priorizada de visitas está vacía."
    probs = lista["probabilidad_alto"].tolist()
    assert probs == sorted(probs, reverse=True), (
        "La lista no está ordenada por probabilidad descendente."
    )
    print(f"[OK] Lista priorizada no vacía ({len(lista)} familias) y ordenada.")

    # (d) Sanidad extra: el ganador tiene macro-F1 razonable.
    f1_winner = float(comparison.loc[winner, "F1_Macro"])
    assert f1_winner > 0.5, f"macro-F1 del ganador demasiado bajo: {f1_winner}"
    print(f"[OK] macro-F1 del ganador = {f1_winner:.4f} (> 0.5).")

    _print_header("PIPELINE COMPLETO — TODO EN VERDE")
    print("run_all.py finalizó sin errores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
