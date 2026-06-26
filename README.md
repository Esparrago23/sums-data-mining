# SUMS — Data Mining

Módulo de minería de datos del proyecto SUMS. Dos subcomponentes:

```
sums-data-mining/
├── subcomponente_A_OCR/          ← OCR de cédulas PDF  (PENDIENTE — se hace después)
│   ├── data/raw_pdfs/            ← 10 PDFs reales de evaluación
│   ├── data/ground_truth/        ← campos_esperados.json
│   ├── src/
│   └── notebooks/
│
└── subcomponente_B_ETL_Risk/     ← ETL + modelo de riesgo  (EN CURSO)
    ├── data/
    │   ├── families_full.json    ← familias anidadas (para la BD)   [generado]
    │   ├── synthetic_data.csv     ← features planas (para el modelo) [generado]
    │   └── processed/             ← logs de carga, datos post-ETL
    ├── src/
    │   ├── catalogos_sums.py      ← valores de catálogo EXACTOS + geografía Chiapas
    │   ├── synthetic_generator.py ← FASE B-1: genera las familias sintéticas
    │   ├── load_to_bd.py          ← carga a la BD vía API captura-completa (recomendado)
    │   └── load_to_bd_sql.py      ← carga directa a PostgreSQL (respaldo)
    ├── notebooks/                 ← B_ETL_Risk_Model.ipynb (fases B-2 a B-6)
    ├── BD_MAPPING.md              ← prueba de que los datos encajan en la BD
    └── requirements.txt
```

## Estado actual

- ✅ **Fase B-1** — Generador de familias sintéticas (Suchiapa / Tuxtla, Chiapas).
- ✅ **Carga a BD** — Scripts para poblar la BD real con los datos sintéticos.
- ⬜ Fases B-2 a B-6 — ETL, EDA, modelo de riesgo (notebook).
- ⬜ Subcomponente A (OCR) — después.

## Cómo correr (Fase B-1 + carga)

```bash
cd subcomponente_B_ETL_Risk/src

# 1) Generar familias sintéticas (sin dependencias externas)
python synthetic_generator.py --n 4000 --seed 42 --suchiapa-ratio 0.8

# 2) Instalar dependencias de carga
pip install requests

# 3) Prueba: cargar 5 familias a la API local
python load_to_bd.py --base-url http://localhost:3000/sums --limit 5

# 4) Carga completa (API desplegada)
python load_to_bd.py --base-url https://api-sums.troy.engineer/sums
```

> La carga es **reanudable**: si se corta o se vuelve a correr, omite las familias ya
> insertadas (registra cada una en `data/processed/load_results.jsonl`).

## Idea de los datos sintéticos

Familias **realistas para Suchiapa** (80%) y Tuxtla Gutiérrez (20%): calles y colonias
reales, nombres y apellidos comunes en Chiapas, lenguas indígenas de la región, y
correlaciones coherentes (piso de tierra + sin agua + ingreso bajo → riesgo ALTO).

Se usan **mientras IMSS-BIENESTAR entrega los datos reales**. Cuando lleguen, el mismo
pipeline (endpoint captura-completa / ETL) funciona sin cambios — solo se reemplaza la
fuente.
