"""
model_trainer.py
================
Subcomponente B — Fases B-3/B-4/B-5: preprocesamiento, entrenamiento y evaluación
comparativa de >=3 clasificadores de riesgo familiar (ALTO/MEDIO/BAJO).

Modelos comparados:
  1. DecisionTreeClassifier  — baseline visualizable.
  2. RandomForestClassifier  — modelo principal (ensemble).
  3. XGBClassifier           — comparación (gradient boosting).

Metodología (validación de la integración ML):
  - Pipeline = ColumnTransformer (OneHot categóricas + StandardScaler numéricas +
    passthrough booleanas) + clasificador. Todo dentro de Pipeline para evitar
    fuga de información en el preprocesamiento.
  - Split estratificado 80/20 + StratifiedKFold(5).
  - Métricas del PROBLEMA: accuracy, macro-F1 (importa por clase, no solo global)
    y CV macro-F1 (media ± desviación). Se decide el ganador por estas métricas.

xgboost 3.3:
  - El label string se codifica con LabelEncoder a entero para XGBClassifier y se
    mapea de vuelta a string para reportar. RandomForest/DecisionTree usan el
    label string directamente.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend headless (sin display) para correr en CI/nbconvert
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from etl_pipeline import (
    CLASES_ORDEN,
    FEATURES_BOOLEANAS,
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
)

RANDOM_STATE = 42
N_SPLITS = 5
PROCESSED_DIR = os.path.join("data", "processed")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocesador (ColumnTransformer)
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer:
      - num: StandardScaler sobre las numéricas (ya imputadas en el ETL).
      - cat: OneHotEncoder(handle_unknown='ignore') sobre las categóricas.
      - bool: passthrough (los bool entran como 0/1).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUMERICAS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                FEATURES_CATEGORICAS,
            ),
            ("bool", "passthrough", FEATURES_BOOLEANAS),
        ],
        remainder="drop",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Definición de los modelos
# ─────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Devuelve un dict {nombre: Pipeline} con los 3 clasificadores.
    Cada Pipeline incluye su propio preprocesador (evita estado compartido).
    """
    return {
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "clf",
                    DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        n_jobs=1,  # serial: evita ruido del resource-tracker de loky en Windows
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,  # serial: evita ruido del resource-tracker de loky en Windows
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Selección del LabelEncoder según el modelo ganador
# ─────────────────────────────────────────────────────────────────────────────

def resolver_label_encoder(winner: str, label_encoder: LabelEncoder):
    """Devuelve el LabelEncoder solo si el ganador es XGBoost (predice enteros
    y necesita mapear de vuelta a string); RF/Decision Tree predicen strings
    directamente, así que no necesitan encoder. Único punto de esta regla
    (antes duplicada en api_mineria.py y risk_report.py)."""
    return label_encoder if winner == "XGBoost" else None


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento + evaluación comparativa
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X, y, processed_dir: str = PROCESSED_DIR) -> dict:
    """
    Entrena y evalúa los 3 modelos. Devuelve un dict con:
      - 'comparison': DataFrame (accuracy, macro-F1, CV macro-F1 media/std).
      - 'winner': nombre del modelo ganador.
      - 'fitted': dict {nombre: pipeline entrenado} (XGB envuelto para predecir strings).
      - 'splits': (X_train, X_test, y_train, y_test).
      - 'reports': dict {nombre: classification_report (texto)}.
      - 'confusion': matriz de confusión del ganador (np.ndarray, orden CLASES_ORDEN).
    Además guarda en `processed_dir`:
      - model_comparison.csv
      - confusion_matrix.png  (ganador)
      - feature_importance.png (Random Forest)
    """
    os.makedirs(processed_dir, exist_ok=True)

    # LabelEncoder para XGBoost (necesita enteros). RF/DT usan strings directos.
    le = LabelEncoder().fit(y)

    # Split estratificado 80/20 (mismo split para los 3 modelos).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()

    filas = []
    fitted: dict = {}
    reports: dict = {}
    pred_test: dict = {}

    for nombre, pipe in models.items():
        es_xgb = nombre == "XGBoost"

        # Targets de entrenamiento: enteros para XGB, strings para el resto.
        y_tr = le.transform(y_train) if es_xgb else y_train.values

        # CV estratificada 5-fold SOLO sobre el TRAIN (macro-F1). Antes se le
        # pasaba (X, y) completos, que incluyen las filas de X_test: eso infla
        # la métrica de CV con fuga de datos (el criterio de selección del
        # ganador no se veía afectado porque usa Accuracy/F1 sobre X_test real,
        # pero CV_F1_Macro_Media/Std reportados eran optimistas).
        cv_scores = cross_val_score(
            pipe, X_train, y_tr, cv=skf, scoring="f1_macro", n_jobs=1
        )

        # Ajuste final sobre el train y predicción sobre el test.
        pipe.fit(X_train, y_tr)
        y_pred_raw = pipe.predict(X_test)
        y_pred = le.inverse_transform(y_pred_raw) if es_xgb else y_pred_raw

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")

        filas.append(
            {
                "Modelo": nombre,
                "Accuracy": round(float(acc), 4),
                "F1_Macro": round(float(f1m), 4),
                "CV_F1_Macro_Media": round(float(cv_scores.mean()), 4),
                "CV_F1_Macro_Std": round(float(cv_scores.std()), 4),
            }
        )

        fitted[nombre] = pipe
        pred_test[nombre] = y_pred
        reports[nombre] = classification_report(
            y_test, y_pred, labels=CLASES_ORDEN, digits=4, zero_division=0
        )

    comparison = pd.DataFrame(filas).set_index("Modelo")

    # Ganador: mayor macro-F1 en test; desempate por accuracy, luego por CV.
    comparison_sorted = comparison.sort_values(
        by=["F1_Macro", "Accuracy", "CV_F1_Macro_Media"], ascending=False
    )
    winner = comparison_sorted.index[0]

    # Persistir la tabla comparativa.
    comp_path = os.path.join(processed_dir, "model_comparison.csv")
    comparison_sorted.to_csv(comp_path, encoding="utf-8")

    # Matriz de confusión del ganador.
    cm = confusion_matrix(y_test, pred_test[winner], labels=CLASES_ORDEN)
    _plot_confusion_matrix(cm, winner, processed_dir)

    # Feature importance del Random Forest (modelo interpretable principal).
    _plot_feature_importance(fitted["Random Forest"], processed_dir)

    return {
        "comparison": comparison_sorted,
        "winner": winner,
        "fitted": fitted,
        "label_encoder": le,
        "splits": (X_train, X_test, y_train, y_test),
        "reports": reports,
        "confusion": cm,
        "paths": {
            "comparison_csv": comp_path,
            "confusion_png": os.path.join(processed_dir, "confusion_matrix.png"),
            "importance_png": os.path.join(processed_dir, "feature_importance.png"),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gráficas (PNG)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, modelo: str, processed_dir: str) -> str:
    """Heatmap de la matriz de confusión del modelo ganador."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASES_ORDEN,
        yticklabels=CLASES_ORDEN,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {modelo}")
    fig.tight_layout()
    out = os.path.join(processed_dir, "confusion_matrix.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _get_feature_names(pipe: Pipeline) -> list:
    """Recupera los nombres de features tras el ColumnTransformer."""
    pre: ColumnTransformer = pipe.named_steps["preprocessor"]
    nombres: list = []
    nombres.extend(FEATURES_NUMERICAS)
    ohe = pre.named_transformers_["cat"]
    try:
        cat_names = ohe.get_feature_names_out(FEATURES_CATEGORICAS).tolist()
    except Exception:
        cat_names = list(FEATURES_CATEGORICAS)
    nombres.extend(cat_names)
    nombres.extend(FEATURES_BOOLEANAS)
    return nombres


def _plot_feature_importance(rf_pipe: Pipeline, processed_dir: str, top_n: int = 15) -> str:
    """Barplot horizontal de la importancia de features del Random Forest."""
    clf: RandomForestClassifier = rf_pipe.named_steps["clf"]
    nombres = _get_feature_names(rf_pipe)
    importancias = clf.feature_importances_

    n = min(len(nombres), len(importancias))
    serie = (
        pd.Series(importancias[:n], index=nombres[:n])
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=serie.values, y=serie.index, hue=serie.index,
                palette="viridis", legend=False, ax=ax)
    ax.set_xlabel("Importancia (Gini)")
    ax.set_ylabel("Feature")
    ax.set_title("Importancia de features — Random Forest (top {})".format(top_n))
    fig.tight_layout()
    out = os.path.join(processed_dir, "feature_importance.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    # Smoke test del entrenamiento.
    from etl_pipeline import load_dataset

    _, X, y = load_dataset()
    res = train_and_evaluate(X, y)
    print("\n=== Tabla comparativa de modelos ===")
    print(res["comparison"])
    print(f"\nGanador: {res['winner']}")
    print(f"\nReporte de clasificación ({res['winner']}):")
    print(res["reports"][res["winner"]])
