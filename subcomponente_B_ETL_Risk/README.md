# Subcomponente B — ETL + Modelo de Riesgo Familiar

**Estado: integrado en la API (`POST /riesgo/predecir`, `/riesgo/predecir-lote`,
`/riesgo/lista`, `/riesgo/metricas`, `/riesgo/modelo-info`, `/riesgo/graficas/{tipo}`
en `integracion/api_mineria.py`).**

Objetivo: clasificar el nivel de riesgo (`BAJO` / `MEDIO` / `ALTO`) de una familia
a partir de sus condiciones de vivienda, salud y socioeconómicas, y generar una
lista priorizada de visitas domiciliarias preventivas.

## Cómo correrlo

```bash
cd subcomponente_B_ETL_Risk
python src/synthetic_generator.py --n 4000 --seed 42 --suchiapa-ratio 0.8  # genera data/synthetic_data.csv
python src/run_all.py   # ETL -> entrenamiento (3 modelos) -> reporte de visitas -> asserts
```

`run_all.py` termina con código 0 si: los 3 modelos superan 0.5 accuracy, la tabla
comparativa tiene 3 filas, los archivos de salida existen y la lista priorizada no
está vacía.

## Estructura

```
data/
  synthetic_data.csv           <- generado por synthetic_generator.py (gitignored)
  families_full.json           <- payloads completos (vivienda+integrantes+vacunación)
  processed/
    model_comparison.csv       <- Accuracy/F1_Macro/CV por modelo
    confusion_matrix.png       <- del modelo ganador
    feature_importance.png     <- Random Forest
    lista_visitas_prioritarias.csv
    modelo_riesgo_cache.joblib <- cache del modelo ganador (mejora M1, ver abajo)
src/
  catalogos_sums.py    <- catálogos oficiales de la cédula (coinciden con seeder.sql de sums-API)
  synthetic_generator.py <- genera familias sintéticas coherentes (label determinístico)
  etl_pipeline.py       <- carga + limpieza + FEATURES (excluye score_total e identificadores, anti-leakage)
  model_trainer.py      <- Decision Tree vs Random Forest vs XGBoost, CV 5-fold, gráficas
  risk_report.py        <- lista priorizada de visitas (familias ALTO por probabilidad)
  run_all.py             <- driver end-to-end
tests/
  test_etl_pipeline.py  <- clean_and_transform, build_xy
  test_model_trainer.py <- regresión: la CV no debe ver filas de X_test (fuga de datos)
```

## Notas importantes

- **Anti-leakage**: `score_total` y los identificadores de texto (`nombre_informante`,
  `domicilio`, `colonia`, `localidad`) se EXCLUYEN de los features del modelo
  (ver `etl_pipeline.COLUMNAS_EXCLUIDAS`). El accuracy alto es esperado porque el
  label se deriva de reglas sobre los features — es una limitación conocida, no oculta.
- **Fuga en la CV (corregida)**: `cross_val_score` se llama solo sobre `X_train`/`y_train`,
  nunca sobre el dataset completo — antes incluía por error las filas reservadas
  para test, lo que inflaba `CV_F1_Macro_Media/Std` (el criterio de selección del
  ganador, basado en Accuracy/F1 sobre test real, no se veía afectado).
- **PII en `/riesgo/lista`**: esa lista reincorpora nombre/domicilio en claro para
  que quien haga la visita sepa a quién buscar. La mitigación es que el endpoint
  requiere `X-API-Key` (ver `integracion/api_mineria.py`); con datos reales de
  producción, considerar cifrado en reposo o un `id_familia` resuelto aparte.
- **M1 — cache del modelo**: antes el microservicio reentrenaba los 3 modelos en
  CADA arranque de `uvicorn`. Ahora `api_mineria.py` cachea el modelo ganador en
  `data/processed/modelo_riesgo_cache.joblib` y solo reentrena si el CSV fuente
  cambió (por mtime).

Ver el plan completo en `sums-documentos/plan_mineria_datos.md` (Subcomponente B).
