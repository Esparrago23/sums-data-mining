# -*- coding: utf-8 -*-
"""
api_mineria.py — API de ejemplo que EXPONE el módulo de minería al resto del SUMS
=================================================================================
Envuelve los dos componentes en endpoints HTTP que la Web (React) o la app
(Flutter) pueden consumir igual que consumen la sums-API de Node:

  GET  /salud                  -> healthcheck (qué cargó, cuántos documentos)
  GET  /buscar?q=...&motor=bm25&k=5
                               -> Subcomponente C: motor de búsqueda sobre notas
  POST /riesgo/predecir        -> Subcomponente B: clasifica UNA familia (ALTO/MEDIO/BAJO)
  GET  /riesgo/lista?top=20    -> Subcomponente B: lista priorizada de visitas

Levantar en local:
  cd sums-data-mining/integracion
  C:\\Users\\minis\\.venvs\\sums-mineria\\Scripts\\python.exe -m uvicorn api_mineria:app --reload --port 8001
  # Swagger interactivo en  http://localhost:8001/docs

NOTA de arquitectura: esto corre como un MICROSERVICIO Python al lado de la
sums-API (Node/TS). No reemplaza nada; el front llama a este servicio para las
funciones de minería. En producción el corpus del buscador se construye con el
campo `observaciones` real de las cédulas (ver build_corpus_desde_bd en el README).
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Rutas del módulo ─────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent.parent           # sums-data-mining/
B_DIR = RAIZ / "subcomponente_B_ETL_Risk"
C_DIR = RAIZ / "subcomponente_C_busqueda"
for p in (B_DIR / "src", C_DIR / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── Imports de los componentes (C carga spaCy al importar) ───────────────────
from tfidf_engine import MotorTFIDF        # noqa: E402
from bm25_engine import MotorBM25          # noqa: E402
from etl_pipeline import load_dataset, FEATURES   # noqa: E402
from model_trainer import train_and_evaluate      # noqa: E402
from risk_report import generar_lista_visitas, _predecir  # noqa: E402

ESTADO: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el motor de búsqueda y entrena el modelo de riesgo UNA vez al arrancar."""
    # --- Subcomponente C: motor de búsqueda ---
    proc = json.loads((C_DIR / "data" / "corpus_procesado_sums.json").read_text(encoding="utf-8"))
    crudo = json.loads((C_DIR / "data" / "corpus_crudo_sums.json").read_text(encoding="utf-8"))
    ESTADO["textos"] = {d["id"]: d.get("texto", "") for d in crudo}
    ESTADO["titulos"] = {d["id"]: d.get("titulo", "") for d in crudo}
    ESTADO["tfidf"] = MotorTFIDF(proc)
    ESTADO["bm25"] = MotorBM25(proc)

    # --- Subcomponente B: modelo de riesgo (entrena al arrancar) ---
    df, X, y = load_dataset(csv_path=str(B_DIR / "data" / "synthetic_data.csv"))
    res = train_and_evaluate(X, y, processed_dir=str(B_DIR / "data" / "processed"))
    winner = res["winner"]
    ESTADO["winner"] = winner
    ESTADO["pipe"] = res["fitted"][winner]
    ESTADO["le"] = res["label_encoder"] if winner == "XGBoost" else None
    ESTADO["lista"] = generar_lista_visitas(
        df, ESTADO["pipe"], ESTADO["le"], processed_dir=str(B_DIR / "data" / "processed")
    )
    yield
    ESTADO.clear()


app = FastAPI(title="SUMS · API de Minería (Buscador + Riesgo)", version="1.0", lifespan=lifespan)

# CORS abierto para desarrollo (la Web en localhost:5173 / Vite puede llamar).
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Salud
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/salud")
def salud():
    return {
        "ok": True,
        "buscador": ["bm25", "tfidf"],
        "modelo_riesgo": ESTADO.get("winner"),
        "n_documentos": len(ESTADO.get("textos", {})),
        "n_familias_alto": int(len(ESTADO.get("lista", []))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Subcomponente C — Buscador
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/buscar")
def buscar(q: str, motor: Literal["bm25", "tfidf"] = "bm25", k: int = 5):
    """Busca notas de observación relevantes a la consulta `q`.

    motor=bm25 (recomendado, ganó la evaluación) | tfidf. k = nº de resultados."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="La consulta 'q' no puede estar vacía.")
    k = max(1, min(k, 50))
    if motor == "tfidf":
        ranking = ESTADO["tfidf"].buscar_tfidf(q, k=k)
    else:
        ranking = ESTADO["bm25"].buscar_bm25(q, k=k, k1=2.0, b=0.75)  # mejor (k1,b)
    return {
        "consulta": q,
        "motor": motor,
        "k": k,
        "resultados": [
            {
                "posicion": i + 1,
                "id": doc_id,
                "titulo": titulo,
                "score": round(float(score), 4),
                "texto": ESTADO["textos"].get(doc_id, ""),
            }
            for i, (score, doc_id, titulo) in enumerate(ranking)
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Subcomponente B — Riesgo
# ─────────────────────────────────────────────────────────────────────────────
class FamiliaFeatures(BaseModel):
    """Features que el modelo necesita para clasificar una familia.
    Coinciden 1:1 con etl_pipeline.FEATURES (sin score_total ni identificadores)."""
    # numéricas
    numero_cuartos: int = Field(2, ge=1)
    numero_habitantes: int = Field(4, ge=1)
    personas_por_cuarto: float = 2.0
    count_enfermedades_cronicas: int = 0
    count_toxicomanias: int = 0
    avg_dias_proteina: float = 4.0
    avg_dias_frutas_verduras: float = 4.0
    avg_dias_cereales: float = 5.0
    ingreso_nivel: int = 2
    escolaridad_promedio: float = 2.0
    total_integrantes: int = 4
    # categóricas (valores oficiales de la cédula)
    material_techo: str = "Concreto o cemento"
    material_paredes: str = "Concreto o cemento"
    material_piso: str = "Concreto o cemento"
    manejo_excretas: str = "WC"
    cocina_ubicacion: str = "fuera_del_dormitorio"
    # booleanas
    agua_entubada: bool = True
    energia_electrica: bool = True
    cocina_con_lena: bool = False
    red_alcantarillado: bool = True
    fosa_septica: bool = False
    vacunacion_completa: bool = True
    seguridad_social_jefe: bool = False


@app.post("/riesgo/predecir")
def predecir(fam: FamiliaFeatures):
    """Clasifica el nivel de riesgo (ALTO/MEDIO/BAJO) de UNA familia + prob. de ALTO."""
    fila = pd.DataFrame([fam.model_dump()])[FEATURES]
    pred, prob_alto = _predecir(ESTADO["pipe"], fila, ESTADO["le"])
    return {
        "modelo": ESTADO["winner"],
        "nivel_riesgo": str(pred[0]),
        "probabilidad_alto": round(float(prob_alto[0]), 4),
    }


@app.get("/riesgo/lista")
def lista_visitas(top: int = 20):
    """Devuelve la lista priorizada de visitas (familias ALTO, más urgentes primero)."""
    lista = ESTADO.get("lista")
    if lista is None or len(lista) == 0:
        raise HTTPException(status_code=404, detail="Lista no disponible.")
    top = max(1, min(top, len(lista)))
    out = lista.head(top).reset_index()  # 'prioridad' pasa a columna
    return json.loads(out.to_json(orient="records", force_ascii=False))
