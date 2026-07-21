#!/bin/sh
set -e

cd /app

if [ ! -f subcomponente_C_busqueda/data/corpus_procesado_sums.json ] || \
   [ ! -f subcomponente_C_busqueda/data/qrels_sums.json ]; then
  echo "[entrypoint] Construyendo corpus y entrenando el buscador Subcomponente C..."
  python subcomponente_C_busqueda/src/run_all.py
else
  echo "[entrypoint] Artefactos Subcomponente C ya existen."
fi

if [ ! -f subcomponente_B_ETL_Risk/data/processed/lista_visitas_prioritarias.csv ]; then
  echo "[entrypoint] Entrenando modelo de riesgo Subcomponente B..."
  python subcomponente_B_ETL_Risk/src/run_all.py
else
  echo "[entrypoint] Artefactos Subcomponente B ya existen."
fi

echo "[entrypoint] Iniciando API de minería en uvicorn..."
exec uvicorn integracion.api_mineria:app --host 0.0.0.0 --port 8001
