# Subcomponente C — Motor de Búsqueda sobre Notas de Visita

**Estado: integrado en la API (`GET /buscar`, `/buscar/metricas`, `/corpus/estadisticas`,
`/corpus/documento/{doc_id}`, `/corpus/reindexar` en `integracion/api_mineria.py`).**

Objetivo: recuperar las notas de observación de visitas domiciliarias más relevantes
para una consulta en lenguaje natural (ej. "familias con desnutrición infantil y
vivienda precaria"), comparando tres motores.

## Cómo correrlo

```bash
cd subcomponente_C_busqueda
python src/run_all.py   # construye corpus -> preprocesa -> deriva qrels -> indexa -> evalúa -> asserts
```

## Motores

| Motor | Implementación | Nota |
|---|---|---|
| `tfidf` | TF-IDF + coseno, desde cero (`tfidf_engine.py`) | Baseline léxico |
| `bm25` | BM25 desde cero (`bm25_engine.py`) | Gana la evaluación léxica (nDCG@5) |
| `semantico` | Embeddings Sentence-BERT (`embeddings_engine.py`, opcional) | Encuentra sinónimos que bm25/tfidf no matchean (ej. "azúcar alta" ~ "diabetes"); requiere `sentence-transformers` instalado y el modelo descargado — si no está disponible, `/buscar?motor=semantico` responde 503 y el resto de la API sigue funcionando normal |

Los tres exponen la misma interfaz: `buscar_<motor>(consulta, k=5) -> [(score, id, titulo), ...]`.

## Estructura

```
data/
  corpus_crudo_sums.json        <- {id, titulo, texto} por nota
  corpus_procesado_sums.json    <- + tokens lematizados (preprocess.preprocesar)
  corpus_themes.json            <- tema dominante por nota (verdad para qrels)
  qrels_sums.json                <- relevancia graduada 0-3 por consulta de prueba
src/
  corpus_builder.py     <- genera notas sintéticas (150, 9 temas x ~17 c/u)
  preprocess.py          <- normalización + lematización spaCy (es_core_news_sm)
  tfidf_engine.py         <- MotorTFIDF (tf, idf, coseno — desde cero)
  bm25_engine.py           <- MotorBM25 (desde cero)
  embeddings_engine.py     <- MotorSemantico (Sentence-BERT, opcional)
  ir_metrics.py             <- P@k, R@k, MRR, MAP, nDCG@k — funciones puras
  qrels_builder.py          <- deriva relevancia graduada desde corpus_themes.json
  run_all.py                 <- driver end-to-end
tests/
  test_ir_metrics.py    <- casos borde de las 5 métricas (qrels vacío, empates, ranking corto, etc.)
  test_engines.py         <- N de BM25, orden de ranking, fixes de lematización
```

## Notas importantes

- **Lematización rota (parchada)**: spaCy corrompe algunas palabras de dominio
  ("dengue"→"denguir", "atención"→"atencionir", "ácido"→"acir", "letrina"→"letrinar"),
  incluso dentro de oraciones completas ya indexadas. Se corrige con un diccionario
  de excepciones (`EXCEPCIONES_LEMA` en `preprocess.py`) aplicado sobre el lema; es
  un parche, no una solución de fondo — el motor `semantico` evita este problema por
  completo al no lematizar (usa el texto crudo/normalizado).
- **Circularidad del benchmark**: el corpus y los qrels se derivan mecánicamente de
  la misma verdad (`corpus_themes.json`). Las métricas de `/buscar/metricas` son un
  benchmark relativo entre motores, no una medida de desempeño absoluto esperado en
  producción con notas reales.
- **`bm25.N`**: expone el número de documentos indexados (usado por
  `GET /corpus/estadisticas`).

Ver el plan completo en `sums-documentos/plan_mineria_datos.md` (Subcomponente C).
