# SUMS · Módulo de Minería de Datos

Módulo de minería de datos del proyecto **SUMS** (Sistema Unidad Médica Suchiapa — microdiagnóstico familiar para IMSS-BIENESTAR). Todo corre **gratis y en local**, sin APIs de pago.

Contiene tres subcomponentes; **dos están construidos, probados y verificados**:

| Subcomp. | Qué es | Estado | Integración académica válida |
|---|---|---|---|
| **A** | OCR de cédulas PDF | ⏳ pendiente (post-MVP) | — |
| **B** | ETL + **modelo predictivo de riesgo** (3 modelos comparados) | ✅ listo y verificado | opción ① (ML con ≥3 modelos) |
| **C** | **Motor de búsqueda** TF-IDF + BM25 con métricas de IR | ✅ listo y verificado | ✅ **opción ② — la integración central** |

> La integración que cuenta para la materia es el **Subcomponente C** (motor de búsqueda por keywords con sus métricas). El Subcomponente B aporta el modelo de riesgo, con la limitación declarada de que su etiqueta es de reglas (ver más abajo).

---

## 1. Puesta en marcha (una sola vez)

El entorno virtual vive **fuera de OneDrive** (para evitar bloqueos de sincronización):

```powershell
# Crear venv e instalar todo
python -m venv C:\Users\minis\.venvs\sums-mineria
$py = "C:\Users\minis\.venvs\sums-mineria\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m spacy download es_core_news_sm
```

A partir de aquí, **siempre** se usa ese intérprete:
`C:\Users\minis\.venvs\sums-mineria\Scripts\python.exe`

Versiones probadas: Python 3.13.7 · pandas 3.0.3 · scikit-learn 1.9.0 · xgboost 3.3.0 · spaCy 3.8.14 · FastAPI 0.138.1.

---

## 2. Subcomponente B — ETL + Modelo de Riesgo

**Qué hace:** toma las familias de microdiagnóstico, las limpia, y entrena un clasificador que predice el **nivel de riesgo familiar** (`ALTO` / `MEDIO` / `BAJO`) para producir una **lista priorizada de visitas preventivas**. Compara 3 modelos y elige el mejor por métricas.

**Cómo se usa:**
```powershell
cd sums-data-mining\subcomponente_B_ETL_Risk
& $py src\run_all.py        # ETL -> 3 modelos -> evaluación -> lista priorizada
```

**Qué produce** (`data/processed/`): `model_comparison.csv`, `confusion_matrix.png`, `feature_importance.png`, `lista_visitas_prioritarias.csv` (≈1.5k familias ALTO ordenadas por urgencia).

**Resultado (verificado):**

| Modelo | Accuracy | Macro-F1 | CV Macro-F1 |
|---|---|---|---|
| **XGBoost (ganador)** | **0.956** | **0.958** | **0.964 ± 0.006** |
| Random Forest | 0.905 | 0.908 | 0.897 ± 0.004 |
| Decision Tree | 0.816 | 0.817 | 0.821 ± 0.013 |

> **Limitación declarada (anti-leakage):** la etiqueta `nivel_riesgo` se calcula con reglas determinísticas, así que se **excluye `score_total`** y los identificadores del modelo. El accuracy alto es esperable; la validación clínica real requeriría etiquetas de expertos de IMSS-BIENESTAR. Documentado en el notebook.

**Cómo se conecta a datos reales:** el ETL ya tiene `extract_from_db()` (placeholder) con el query a `centro_medico_2026`. Cuando IMSS entregue datos reales, se cambia la fuente sin tocar el resto del pipeline.

Entregable académico: `notebooks/B_ETL_Risk_Model.ipynb` (ya ejecutado, sin errores).

---

## 3. Subcomponente C — Motor de Búsqueda (la integración válida)

**Qué hace:** indexa **notas de observación / visita domiciliaria** y permite buscarlas con **consultas en lenguaje natural** (ej. *"familias con desnutrición infantil"*, *"casos sospechosos de dengue"*). Implementa **TF-IDF + coseno** y **BM25** *desde cero*, y los evalúa con **métricas de IR** (P@k, R@k, MRR, MAP, nDCG) para decidir cuál es mejor con números.

**Cómo se usa:**
```powershell
cd sums-data-mining\subcomponente_C_busqueda
& $py src\run_all.py        # corpus -> preprocesa -> TF-IDF + BM25 -> métricas -> decisión
```

**Qué produce** (`data/`): `corpus_crudo_sums.json`, `corpus_procesado_sums.json`, `qrels_sums.json` + la tabla de métricas por consola.

**Resultado (verificado):**

| Sistema | P@5 | R@5 | MRR | MAP | nDCG@5 |
|---|---|---|---|---|---|
| TF-IDF + coseno | 0.925 | 0.115 | 0.917 | 0.685 | 0.520 |
| **BM25 (k1=2.0, b=0.75)** | **0.950** | **0.118** | **1.000** | **0.691** | **0.569** |

**Decisión:** se elige **BM25** porque gana en las 5 métricas (decisión por **nDCG@5**: 0.569 vs 0.520). Verificado además contra `sklearn.TfidfVectorizer` (el orden del ranking coincide).

> **Limitación natural de keywords:** la sinonimia no se resuelve (*"azúcar alta"* no recupera *"diabetes"*). Es la motivación para una futura extensión a **embeddings** (la otra mitad de la opción ②).

Entregable académico: `notebooks/C_Motor_Busqueda.ipynb` (ya ejecutado, sin errores).

Reglas de los labs respetadas: TF-IDF, BM25 y las 5 métricas **desde cero**; sklearn solo como verificación; consulta preprocesada con el **mismo** pipeline e IDF que el corpus; manejo de norma cero; negaciones conservadas (*"sin agua"*).

---

## 4. Integración al proyecto SUMS

La forma más limpia de meterlo al SUMS sin reescribir nada en Node es correr este módulo como un **microservicio Python** al lado de la `sums-API`, y que la **Web (React)** o la **app (Flutter)** lo consuman por HTTP.

```
  SUMS_WEB (React)  ─┐                 ┌─ sums-API (Node/TS)  → PostgreSQL centro_medico_2026
  sums-mobile (Flutter) ─┼─ HTTP ─────┤
                          └────────────┴─ api_mineria.py (FastAPI)  ← este módulo
                                            /buscar   (Subcomp. C)
                                            /riesgo/* (Subcomp. B)
```

### 4.1 Levantar la API de minería

```powershell
cd sums-data-mining\integracion
& $py -m uvicorn api_mineria:app --reload --port 8001
# Swagger interactivo:  http://localhost:8001/docs
```

Al arrancar carga el motor de búsqueda y entrena el modelo una vez. Endpoints:

| Método | Ruta | Para qué |
|---|---|---|
| `GET` | `/salud` | healthcheck |
| `GET` | `/buscar?q=...&motor=bm25&k=5` | **buscador de casos** (Subcomp. C) |
| `POST` | `/riesgo/predecir` | clasifica UNA familia (Subcomp. B) |
| `GET` | `/riesgo/lista?top=20` | lista priorizada de visitas (Subcomp. B) |

Ejemplos `curl`:
```bash
curl "http://localhost:8001/buscar?q=familias%20con%20desnutricion%20infantil&motor=bm25&k=5"
curl "http://localhost:8001/riesgo/lista?top=10"
curl -X POST "http://localhost:8001/riesgo/predecir" -H "Content-Type: application/json" \
  -d '{"numero_cuartos":1,"numero_habitantes":7,"personas_por_cuarto":7,"count_enfermedades_cronicas":3,"count_toxicomanias":2,"avg_dias_proteina":1,"avg_dias_frutas_verduras":1,"avg_dias_cereales":3,"ingreso_nivel":0,"escolaridad_promedio":0.5,"total_integrantes":7,"material_techo":"Lamina","material_paredes":"Madera","material_piso":"Tierra","manejo_excretas":"Letrina","cocina_ubicacion":"dentro_del_dormitorio","agua_entubada":false,"energia_electrica":false,"cocina_con_lena":true,"red_alcantarillado":false,"fosa_septica":false,"vacunacion_completa":false,"seguridad_social_jefe":false}'
# -> {"modelo":"XGBoost","nivel_riesgo":"ALTO","probabilidad_alto":1.0}
```

### 4.2 La vista en la Web (React) — "le pongo el buscador"

Ejemplo completo listo para copiar en `SUMS_WEB`: **`integracion/ejemplos/BuscadorCasos.tsx`**. Es una página con una caja de búsqueda que llama a `/buscar` y pinta los resultados. Resumen:

```tsx
const API = "http://localhost:8001";

async function buscar(q: string) {
  const r = await fetch(`${API}/buscar?q=${encodeURIComponent(q)}&motor=bm25&k=8`);
  return (await r.json()).resultados; // [{posicion, id, titulo, score, texto}]
}
// ...input + lista de tarjetas con titulo, score y texto de la nota.
```

Se monta como una página más del router (ej. `/buscador`) junto a `DashboardPage` y `CapturaOcrPage`, reutilizando el patrón atoms/molecules/organisms que ya tienes.

### 4.3 Conectar el buscador a datos REALES (campo `observaciones`)

Hoy el corpus son notas sintéticas. En producción, el corpus = el **texto libre que escriben los encuestadores** (campo `observaciones` de cada cédula, o un futuro campo "notas de visita"). Solo cambias la **fuente del corpus** y reusas todo lo demás:

```python
# build_corpus_desde_bd.py  (esquema)
import requests, json
cedulas = requests.get("https://sums-api.troy.engineer/sums/cedulas").json()
corpus = [{"id": f"ced{c['id']}", "titulo": c["familia"]["informante_nombre"],
           "texto": c.get("observaciones", "")} for c in cedulas if c.get("observaciones")]
json.dump(corpus, open("subcomponente_C_busqueda/data/corpus_crudo_sums.json","w"),
          ensure_ascii=False, indent=2)
# luego: python subcomponente_C_busqueda/src/preprocess.py  (re-indexa)
```
Cuanto más ricas sean las notas, mejor busca el motor. (El índice se puede reconstruir periódicamente, p.ej. tras cada sync de cédulas.)

### 4.4 La lista de riesgo en el Dashboard (Subcomp. B)

El `DashboardPage` ya tiene `MetricsGrid` y charts. Agrega un panel **"Visitas prioritarias"** que consuma `GET /riesgo/lista?top=20` y muestre nombre, domicilio, colonia y `probabilidad_alto`. O usa `POST /riesgo/predecir` para mostrar el riesgo **al terminar de capturar una cédula** (clasificación en caliente).

### 4.5 Flutter (opcional)

La app móvil consume los mismos endpoints con `http`/`dio` (igual que ya consume la `sums-API`). El buscador es útil offline-first: indexar localmente o pegarle al microservicio cuando hay red.

> **Producción:** para no reentrenar en cada arranque, persiste el modelo (`joblib.dump(pipe, "modelo.pkl")`) en `run_all.py` y cárgalo en la API. Y pon el microservicio detrás del mismo dominio/reverse-proxy que la `sums-API`.

---

## 5. Mapa de archivos

```
sums-data-mining/
├── requirements.txt              ← dependencias unificadas (este módulo)
├── README.md                     ← este archivo
├── integracion/
│   ├── api_mineria.py            ← FastAPI: /buscar + /riesgo/*  (verificado)
│   └── ejemplos/
│       └── BuscadorCasos.tsx     ← vista React de ejemplo para SUMS_WEB
├── subcomponente_B_ETL_Risk/
│   ├── src/  (synthetic_generator, etl_pipeline, model_trainer, risk_report, run_all)
│   ├── data/ (synthetic_data.csv, families_full.json, processed/…)
│   └── notebooks/B_ETL_Risk_Model.ipynb
└── subcomponente_C_busqueda/
    ├── src/  (corpus_builder, preprocess, tfidf_engine, bm25_engine, ir_metrics, qrels_builder, run_all)
    ├── data/ (corpus_crudo_sums.json, corpus_procesado_sums.json, qrels_sums.json)
    └── notebooks/C_Motor_Busqueda.ipynb
```

## 6. Reproducir todo de cero

```powershell
$py = "C:\Users\minis\.venvs\sums-mineria\Scripts\python.exe"
cd sums-data-mining\subcomponente_B_ETL_Risk; & $py src\run_all.py
cd ..\subcomponente_C_busqueda;              & $py src\run_all.py
cd ..\integracion; & $py -m uvicorn api_mineria:app --port 8001
```
