"""
test_model_trainer.py
======================
Suite de regresión para model_trainer.py.

La prueba más importante es `test_cross_val_score_no_leakage`: verifica que
`train_and_evaluate` invoca `cross_val_score` SOLO con el split de
entrenamiento (X_train / y_tr), nunca con el X/y completos (que incluirían las
filas reservadas para test). Ese era el bug de fuga de datos ya corregido en
model_trainer.py (líneas ~183-190 de `train_and_evaluate`): antes se llamaba
`cross_val_score(pipe, X, y_cv, ...)`; ahora se llama
`cross_val_score(pipe, X_train, y_tr, ...)`.
"""

import pandas as pd
import pytest
from sklearn.model_selection import cross_val_score as _cross_val_score_real
from sklearn.model_selection import train_test_split as _train_test_split
from sklearn.preprocessing import LabelEncoder
from unittest.mock import patch

from etl_pipeline import CLASES_ORDEN, build_xy
from model_trainer import RANDOM_STATE, resolver_label_encoder, train_and_evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Regresión anti-fuga de datos en la validación cruzada
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_val_score_no_leakage(df_sintetico_pequeno, tmp_path):
    X, y = build_xy(df_sintetico_pequeno)

    # Reproducimos el MISMO split 80/20 estratificado que hace
    # train_and_evaluate (mismo random_state) para conocer el tamaño esperado
    # de X_train, y así saber con cuántas filas DEBERÍA llamarse cross_val_score.
    X_train_esperado, X_test_esperado, _, _ = _train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    tam_train_esperado = len(X_train_esperado)
    tam_total = len(X)
    assert 0 < tam_train_esperado < tam_total  # sanity check del split 80/20

    tamanos_capturados = []

    def _cross_val_score_espia(*args, **kwargs):
        # Firma real: cross_val_score(pipe, X_train, y_tr, cv=skf, scoring=..., n_jobs=...)
        X_usado = args[1] if len(args) > 1 else kwargs["X"]
        tamanos_capturados.append(len(X_usado))
        # Delegamos a la función real de sklearn para no romper el resto del
        # flujo (train_and_evaluate usa cv_scores.mean()/.std() del resultado).
        return _cross_val_score_real(*args, **kwargs)

    with patch(
        "model_trainer.cross_val_score", side_effect=_cross_val_score_espia
    ) as mock_cv:
        resultado = train_and_evaluate(X, y, processed_dir=str(tmp_path))

    # Un llamado por cada uno de los 3 modelos (Decision Tree, Random Forest, XGBoost).
    assert mock_cv.call_count == 3
    assert len(tamanos_capturados) == 3

    for tam in tamanos_capturados:
        assert tam == tam_train_esperado, (
            f"cross_val_score se llamó con {tam} filas; se esperaban "
            f"{tam_train_esperado} (tamaño de X_train). Si esto falla, revisar "
            "que train_and_evaluate NO esté pasando X/y completos a "
            "cross_val_score (eso sería fuga de datos: incluiría las filas "
            "reservadas para test)."
        )
        assert tam != tam_total

    assert resultado["winner"] in {"XGBoost", "Random Forest", "Decision Tree"}


# ─────────────────────────────────────────────────────────────────────────────
# resolver_label_encoder
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("winner", ["Random Forest", "Decision Tree", "Otro Modelo"])
def test_resolver_label_encoder_no_xgboost_devuelve_none(winner):
    le = LabelEncoder().fit(CLASES_ORDEN)
    assert resolver_label_encoder(winner, le) is None


def test_resolver_label_encoder_xgboost_devuelve_el_mismo_encoder():
    le = LabelEncoder().fit(CLASES_ORDEN)
    resultado = resolver_label_encoder("XGBoost", le)
    assert resultado is le


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test de train_and_evaluate
# ─────────────────────────────────────────────────────────────────────────────

def test_train_and_evaluate_smoke(df_sintetico_pequeno, tmp_path):
    X, y = build_xy(df_sintetico_pequeno)

    resultado = train_and_evaluate(X, y, processed_dir=str(tmp_path))

    assert set(resultado.keys()) >= {"winner", "comparison", "fitted"}
    assert resultado["winner"] in {"XGBoost", "Random Forest", "Decision Tree"}

    comparison = resultado["comparison"]
    assert isinstance(comparison, pd.DataFrame)
    assert comparison.shape[0] == 3
    assert set(comparison.index) == {"XGBoost", "Random Forest", "Decision Tree"}

    fitted = resultado["fitted"]
    assert set(fitted.keys()) == {"XGBoost", "Random Forest", "Decision Tree"}
    for pipe in fitted.values():
        # Cada pipeline debe estar entrenado: predict no debe fallar.
        pred = pipe.predict(X.iloc[:3])
        assert len(pred) == 3

    # El ganador debe ser consistente con la fila de mayor F1_Macro.
    max_f1 = comparison["F1_Macro"].max()
    f1_del_ganador = comparison.loc[resultado["winner"], "F1_Macro"]
    assert f1_del_ganador == max_f1
