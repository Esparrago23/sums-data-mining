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
  set MINERIA_API_KEY=una-clave-larga-y-secreta
  set MINERIA_CORS_ORIGINS=http://localhost:5173,http://localhost:3000
  C:\\Users\\minis\\.venvs\\sums-mineria\\Scripts\\python.exe -m uvicorn api_mineria:app --reload --port 8001
  # Swagger interactivo en  http://localhost:8001/docs

Seguridad:
  - Todos los endpoints (excepto /salud) requieren el header `X-API-Key` con el
    valor de la variable de entorno MINERIA_API_KEY. Si esta variable no está
    definida, el servicio rechaza (503) todas las peticiones a esos endpoints.
  - CORS solo permite los orígenes listados en MINERIA_CORS_ORIGINS (separados
    por comas); si no se define, por defecto solo se permite el dev server de
    Vite en http://localhost:5173.

NOTA de arquitectura: esto corre como un MICROSERVICIO Python al lado de la
sums-API (Node/TS). No reemplaza nada; el front llama a este servicio para las
funciones de minería. En producción el corpus del buscador se construye con el
campo `observaciones` real de las cédulas (ver build_corpus_desde_bd en el README).
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("sums.mineria")

# ── Rutas del módulo ─────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent.parent           # sums-data-mining/
A_DIR = RAIZ / "subcomponente_A_OCR"
B_DIR = RAIZ / "subcomponente_B_ETL_Risk"
C_DIR = RAIZ / "subcomponente_C_busqueda"
for p in (A_DIR / "src", B_DIR / "src", C_DIR / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── Imports de los componentes (C carga spaCy al importar) ───────────────────
from tfidf_engine import MotorTFIDF        # noqa: E402
from bm25_engine import MotorBM25          # noqa: E402
from etl_pipeline import load_dataset, FEATURES   # noqa: E402
from model_trainer import train_and_evaluate      # noqa: E402
from risk_report import generar_lista_visitas, _predecir  # noqa: E402

# ── Imports del Subcomponente A (OCR) ────────────────────────────────────────
from pdf_renderer import render_pdf          # noqa: E402
from preprocessor import normalize_page      # noqa: E402
from field_extractor import extract_document, load_field_map  # noqa: E402
from evaluator import evaluate_checkbox_fields  # noqa: E402
from catalogos_sums import (  # noqa: E402
    CAT_MATERIAL_TECHO_PAREDES, CAT_MATERIAL_PISO, CAT_MANEJO_EXCRETAS,
)

# ── Motor de texto OCR (PaddleOCR si disponible, Tesseract como fallback) ────
# Importación condicional: la API arranca normalmente aunque PaddlePaddle
# no esté instalado; en ese caso el motor de texto cae automáticamente
# a Tesseract con recorte individual por bbox_rel.
try:
    from text_trainer import apply_paddle_text, apply_ocr_text  # noqa: E402
    from paddle_extractor import get_ocr_engine                  # noqa: E402
    _TEXT_ENGINE_AVAILABLE = True
except ImportError:
    _TEXT_ENGINE_AVAILABLE = False
    logger.warning(
        "[OCR] text_trainer / paddle_extractor no disponibles. "
        "Los campos de texto quedarán con value=null hasta que se instalen."
    )

ESTADO: dict = {}

# ── Seguridad: API key compartida (item 1 de la auditoría) ──────────────────
# El microservicio se autentica con una API-key compartida leída de entorno.
# Si la variable no está configurada, el servicio se niega a atender peticiones
# a endpoints protegidos (fail-closed) en vez de arrancar sin autenticación.
MINERIA_API_KEY = os.environ.get("MINERIA_API_KEY")


def verificar_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Dependencia FastAPI: exige el header `X-API-Key` con el valor de MINERIA_API_KEY.

    Se aplica a todos los endpoints que procesan o exponen datos (ocr/*, riesgo/*,
    corpus/*, buscar/*, catalogos, datos/*). `/salud` queda sin proteger para
    permitir monitoreo básico (no expone datos de pacientes)."""
    if not MINERIA_API_KEY:
        logger.error("MINERIA_API_KEY no está configurada; rechazando petición por seguridad.")
        raise HTTPException(status_code=503, detail="Servicio no configurado correctamente.")
    # Comparación de tiempo constante (hmac.compare_digest) para evitar timing
    # attacks; se descarta primero el caso x_api_key=None sin comparar cadenas.
    if x_api_key is None or not hmac.compare_digest(x_api_key, MINERIA_API_KEY):
        raise HTTPException(status_code=401, detail="API key inválida o faltante.")


REQUIERE_API_KEY = [Depends(verificar_api_key)]

# ── Seguridad: validación de subida de PDFs (item 4) ─────────────────────────
MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB
PDF_MAGIC = b"%PDF-"

# ── Seguridad: whitelist para segmentos de ruta usados en /ocr/roi (item 2) ──
ID_SEGURO_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


async def _leer_y_validar_pdf(archivo: UploadFile) -> bytes:
    """Lee el contenido del UploadFile validando tamaño máximo y firma %PDF- real
    (no solo la extensión del nombre de archivo)."""
    contenido = await archivo.read(MAX_PDF_BYTES + 1)
    if len(contenido) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo excede el tamaño máximo permitido ({MAX_PDF_BYTES // (1024 * 1024)} MB).",
        )
    if not contenido.startswith(PDF_MAGIC):
        raise HTTPException(status_code=400, detail="El archivo no es un PDF válido.")
    return contenido


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

    # --- Subcomponente A: OCR (precargar field map) ---
    field_map_path = A_DIR / "config" / "field_map_sums.json"
    if field_map_path.exists():
        ESTADO["ocr_field_map"] = load_field_map(str(field_map_path))
        ESTADO["ocr_processed_dir"] = str(A_DIR / "data" / "processed")
        ESTADO["ocr_raw_dir"] = str(A_DIR / "data" / "raw_pdfs")
        ESTADO["ocr_rendered_dir"] = str(A_DIR / "data" / "processed" / "rendered_pages")

    yield
    ESTADO.clear()


app = FastAPI(title="SUMS · API de Minería (Buscador + Riesgo)", version="1.0", lifespan=lifespan)

# CORS restringido: orígenes permitidos configurables por variable de entorno
# MINERIA_CORS_ORIGINS (lista separada por comas). Si no se define, por defecto
# solo se permite el origen de desarrollo local (Vite).
_cors_origins_env = os.environ.get("MINERIA_CORS_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or [
    "http://localhost:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Salud
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/salud")
def salud():
    """Healthcheck completo: qué componentes están cargados y cuántos datos hay."""
    return {
        "ok": True,
        "componentes": {
            "ocr": {
                "disponible": "ocr_field_map" in ESTADO,
                "n_campos_template": len(ESTADO.get("ocr_field_map", {}).get("datos_vivienda", [])) if isinstance(ESTADO.get("ocr_field_map"), dict) else 0,
            },
            "buscador": {
                "disponible": "bm25" in ESTADO,
                "motores": ["bm25", "tfidf"],
                "n_documentos": len(ESTADO.get("textos", {})),
            },
            "modelo_riesgo": {
                "disponible": "pipe" in ESTADO,
                "modelo_ganador": ESTADO.get("winner"),
                "n_familias_alto": int(len(ESTADO.get("lista", []))),
            },
        },
        "endpoints": [
            "/salud", "/catalogos", "/datos/estadisticas",
            "/ocr/procesar", "/ocr/resultados", "/ocr/resultados/{doc_id}",
            "/ocr/roi/{doc_id}/{field_id}", "/ocr/campos-template",
            "/buscar", "/buscar/metricas", "/corpus/estadisticas",
            "/corpus/documento/{doc_id}", "/corpus/reindexar",
            "/riesgo/predecir", "/riesgo/predecir-lote", "/riesgo/lista",
            "/riesgo/metricas", "/riesgo/modelo-info", "/riesgo/graficas/{tipo}",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Subcomponente C — Buscador
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/buscar", dependencies=REQUIERE_API_KEY)
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
    # categóricas (valores oficiales de la cédula; Literal validado contra el
    # catálogo real de subcomponente_B_ETL_Risk/src/catalogos_sums.py)
    material_techo: Literal[tuple(CAT_MATERIAL_TECHO_PAREDES)] = "Concreto o cemento"
    material_paredes: Literal[tuple(CAT_MATERIAL_TECHO_PAREDES)] = "Concreto o cemento"
    material_piso: Literal[tuple(CAT_MATERIAL_PISO)] = "Concreto o cemento"
    manejo_excretas: Literal[tuple(CAT_MANEJO_EXCRETAS)] = "WC"
    # cocina_ubicacion no tiene tabla de catálogo propia (no está en
    # catalogos_sums.py): es un enum fijo de 2 valores, ver BD_MAPPING.md.
    cocina_ubicacion: Literal["fuera_del_dormitorio", "dentro_del_dormitorio"] = "fuera_del_dormitorio"
    # booleanas
    agua_entubada: bool = True
    energia_electrica: bool = True
    cocina_con_lena: bool = False
    red_alcantarillado: bool = True
    fosa_septica: bool = False
    vacunacion_completa: bool = True
    seguridad_social_jefe: bool = False

    @field_validator("*", mode="before")
    @classmethod
    def _strip_str_fields(cls, v):
        """Item 7: recorta espacios de cualquier campo string antes de validar
        (incluye validación de Literal, para tolerar '  WC ' -> 'WC')."""
        if isinstance(v, str):
            return v.strip()
        return v


@app.post("/riesgo/predecir", dependencies=REQUIERE_API_KEY)
def predecir(fam: FamiliaFeatures):
    """Clasifica el nivel de riesgo (ALTO/MEDIO/BAJO) de UNA familia + prob. de ALTO."""
    fila = pd.DataFrame([fam.model_dump()])[FEATURES]
    pred, prob_alto = _predecir(ESTADO["pipe"], fila, ESTADO["le"])
    return {
        "modelo": ESTADO["winner"],
        "nivel_riesgo": str(pred[0]),
        "probabilidad_alto": round(float(prob_alto[0]), 4),
    }


@app.get("/riesgo/lista", dependencies=REQUIERE_API_KEY)
def lista_visitas(top: int = 20):
    """Devuelve la lista priorizada de visitas (familias ALTO, más urgentes primero)."""
    lista = ESTADO.get("lista")
    if lista is None or len(lista) == 0:
        raise HTTPException(status_code=404, detail="Lista no disponible.")
    top = max(1, min(top, len(lista)))
    out = lista.head(top).reset_index()  # 'prioridad' pasa a columna
    return json.loads(out.to_json(orient="records", force_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# Subcomponente A — OCR
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ocr/procesar", dependencies=REQUIERE_API_KEY)
async def ocr_procesar(archivo: UploadFile = File(...)):
    """Recibe un PDF de cédula escaneada, ejecuta el pipeline OCR completo.

    **Qué hace:** renderiza cada página a PNG, detecta el recuadro del formulario,
    extrae checkboxes (detección inteligente con score) y marca campos de texto/número
    como `needs_review=True` con su imagen ROI recortada para revisión humana.

    **Solo página 1** tiene campos definidos (45 campos: 37 checkboxes + 6 texto + 2 número).

    **Input:** archivo PDF (multipart/form-data, campo 'archivo').
    **Output:** JSON con todos los campos extraídos por página.

    Ejemplo frontend:
        const formData = new FormData();
        formData.append('archivo', pdfFile);
        const res = await fetch('http://localhost:8001/ocr/procesar', {method: 'POST', body: formData});
        const data = await res.json();
        // data.campos[field_id] = {type, value, confidence, needs_review, ...}
    """
    if not archivo.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos PDF.")

    field_map = ESTADO.get("ocr_field_map")
    if field_map is None:
        raise HTTPException(503, "OCR no disponible: field_map no cargado.")

    # Guardar PDF
    doc_id = f"upload_{uuid.uuid4().hex[:8]}"
    processed_dir = Path(ESTADO["ocr_processed_dir"])
    rendered_dir = processed_dir / "rendered_pages"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    tmp_pdf = processed_dir / f"{doc_id}.pdf"
    contenido = await _leer_y_validar_pdf(archivo)
    tmp_pdf.write_bytes(contenido)

    try:
        # 1. Renderizar PDF a PNGs
        page_paths = render_pdf(str(tmp_pdf), str(rendered_dir), dpi=180)

        # 2. Normalizar y extraer
        from preprocessor import normalize_page as _norm_page
        import re as _re

        def _page_num(p):
            m = _re.search(r"-(\d+)\.png$", str(p))
            return int(m.group(1)) if m else 0

        pages_sorted = sorted(page_paths, key=_page_num)
        pages_norm = [_norm_page(str(p), _page_num(p)) for p in pages_sorted]

        resultado = extract_document(doc_id, pages_norm, field_map, str(processed_dir))

        # ── Motor de texto: PaddleOCR o Tesseract (sólo campos type='text') ──
        if _TEXT_ENGINE_AVAILABLE:
            _preds = {doc_id: resultado}
            try:
                _engine = get_ocr_engine()
                if _engine.is_available:
                    logger.info("[OCR] Motor de texto: PaddleOCR + fuzzy mapping")
                    apply_paddle_text(_preds, engine=_engine, field_map=field_map, page_key="1")
                else:
                    logger.warning("[OCR] PaddleOCR no disponible, usando Tesseract OCR.")
                    apply_ocr_text(_preds, field_map=field_map, page_key="1")
            except Exception as _ocr_err:  # noqa: BLE001
                logger.error("[OCR] Error en motor de texto (%s), usando Tesseract OCR.", _ocr_err)
                apply_ocr_text(_preds, field_map=field_map, page_key="1")
        # ────────────────────────────────────────────────────────────────────

        return {
            "doc_id": doc_id,
            "archivo_original": archivo.filename,
            "n_paginas": len(pages_sorted),
            "campos": resultado.get("fields", {}),
            "resumen": {
                "total_campos": len(resultado.get("fields", {})),
                "necesitan_revision": sum(
                    1 for f in resultado.get("fields", {}).values()
                    if f.get("needs_review")
                ),
            },
        }
    except HTTPException:
        raise
    except Exception:
        # Item 3: no filtrar el detalle interno de la excepción al cliente;
        # se registra completo en el log del servidor.
        logger.exception("Error procesando OCR (doc_id=%s)", doc_id)
        raise HTTPException(500, "Error interno procesando el documento OCR.")


@app.post("/ocr/procesar-cedula", dependencies=REQUIERE_API_KEY)
async def ocr_procesar_cedula(archivo: UploadFile = File(...)):
    """Recibe un PDF de cédula escaneada y extrae todos sus campos estructurados.

    Endpoint principal del **Subcomponente A (OCR)**. Ejecuta el pipeline completo:

    1. **Renderizado:** convierte cada página del PDF a imágenes PNG (180 DPI).
    2. **Preprocesado:** detecta el recuadro del formulario y normaliza la imagen.
    3. **Extracción por plantilla:** aplica el modelo OCR sobre los 45 campos
       definidos en `field_map_sums.json` (37 checkboxes + 6 texto + 2 número).
    4. **Respuesta:** devuelve un JSON con `doc_id`, campos extraídos (`campos`)
       y un resumen de cuántos necesitan revisión humana (`needs_review=True`).

    **Input:** archivo PDF enviado como `multipart/form-data` (campo `archivo`).

    **Output:**
    ```json
    {
      "doc_id": "upload_a1b2c3d4",
      "archivo_original": "cedula_familia_007.pdf",
      "n_paginas": 2,
      "campos": {
        "vivienda.agua_entubada.si": {"type": "checkbox", "value": true, "confidence": 0.91, ...},
        "vivienda.numero_cuartos":   {"type": "number",   "value": null, "needs_review": true, ...}
      },
      "resumen": {"total_campos": 45, "necesitan_revision": 8}
    }
    ```

    **Errores:**
    - `400` — el archivo no es un PDF.
    - `503` — el módulo OCR no está disponible (field_map no cargado al arrancar).
    - `500` — error interno durante el procesamiento OCR.

    Ejemplo `curl`:
        curl -X POST "http://localhost:8001/ocr/procesar-cedula" \\
             -F "archivo=@cedula_001.pdf"
    """
    if not archivo.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos PDF.")

    field_map = ESTADO.get("ocr_field_map")
    if field_map is None:
        raise HTTPException(503, "OCR no disponible: field_map no cargado. Verifique que exista config/field_map_sums.json.")

    # Guardar PDF temporalmente con ID único
    doc_id = f"cedula_{uuid.uuid4().hex[:8]}"
    processed_dir = Path(ESTADO["ocr_processed_dir"])
    rendered_dir = processed_dir / "rendered_pages"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    tmp_pdf = processed_dir / f"{doc_id}.pdf"
    contenido = await _leer_y_validar_pdf(archivo)
    tmp_pdf.write_bytes(contenido)

    try:
        import re as _re

        def _page_num(p) -> int:
            m = _re.search(r"-(\d+)\.png$", str(p))
            return int(m.group(1)) if m else 0

        # 1. Renderizar PDF → PNGs
        page_paths = render_pdf(str(tmp_pdf), str(rendered_dir), dpi=180)

        # 2. Normalizar cada página con el preprocesador del Subcomponente A
        pages_sorted = sorted(page_paths, key=_page_num)
        pages_norm = [normalize_page(str(p), _page_num(p)) for p in pages_sorted]

        # 3. Extraer campos usando field_extractor.extract_document
        resultado = extract_document(doc_id, pages_norm, field_map, str(processed_dir))

        # ── Motor de texto: PaddleOCR o Tesseract (sólo campos type='text') ──
        if _TEXT_ENGINE_AVAILABLE:
            _preds = {doc_id: resultado}
            try:
                _engine = get_ocr_engine()
                if _engine.is_available:
                    logger.info("[OCR] Motor de texto: PaddleOCR + fuzzy mapping")
                    apply_paddle_text(_preds, engine=_engine, field_map=field_map, page_key="1")
                else:
                    logger.warning("[OCR] PaddleOCR no disponible, usando Tesseract OCR.")
                    apply_ocr_text(_preds, field_map=field_map, page_key="1")
            except Exception as _ocr_err:  # noqa: BLE001
                logger.error("[OCR] Error en motor de texto (%s), usando Tesseract OCR.", _ocr_err)
                apply_ocr_text(_preds, field_map=field_map, page_key="1")
        # ────────────────────────────────────────────────────────────────────

        # Re-leer campos desde resultado (puede haber sido enriquecido por el motor de texto).
        campos = resultado.get("fields", {})
        return {
            "doc_id": doc_id,
            "archivo_original": archivo.filename,
            "n_paginas": len(pages_sorted),
            "campos": campos,
            "resumen": {
                "total_campos": len(campos),
                "necesitan_revision": sum(1 for f in campos.values() if f.get("needs_review")),
            },
        }
    except HTTPException:
        raise
    except Exception:
        # Item 3: no filtrar el detalle interno de la excepción al cliente;
        # se registra completo en el log del servidor.
        logger.exception("Error en el pipeline OCR (doc_id=%s)", doc_id)
        raise HTTPException(500, "Error interno en el pipeline OCR.")


@app.get("/ocr/resultados", dependencies=REQUIERE_API_KEY)
def ocr_resultados():
    """Lista todos los documentos procesados por OCR con su resumen.

    Lee predictions.json si existe.
    Output: {"documentos": [{"doc_id": "Cédula_0001", "n_campos": 45, "n_review": 8}]}
    """
    processed = Path(ESTADO.get("ocr_processed_dir", ""))
    pred_path = processed / "predictions.json"
    if not pred_path.exists():
        return {"documentos": []}

    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    docs = []
    for doc_id, doc_data in preds.items():
        campos = doc_data.get("fields", {})
        docs.append({
            "doc_id": doc_id,
            "n_campos": len(campos),
            "n_review": sum(1 for f in campos.values() if f.get("needs_review")),
        })
    return {"documentos": docs}


@app.get("/ocr/resultados/{doc_id}", dependencies=REQUIERE_API_KEY)
def ocr_resultado_detalle(doc_id: str):
    """Devuelve los campos extraídos de un documento específico.

    Output: {"doc_id": "...", "campos": {"vivienda.agua_entubada.si": {type, value, confidence, ...}, ...}}
    """
    processed = Path(ESTADO.get("ocr_processed_dir", ""))
    pred_path = processed / "predictions.json"
    if not pred_path.exists():
        raise HTTPException(404, "No hay predicciones OCR disponibles.")

    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    if doc_id not in preds:
        raise HTTPException(404, f"Documento '{doc_id}' no encontrado.")

    doc_data = preds[doc_id]
    return {"doc_id": doc_id, "campos": doc_data.get("fields", {})}


@app.get("/ocr/roi/{doc_id}/{field_id}", dependencies=REQUIERE_API_KEY)
def ocr_roi_imagen(doc_id: str, field_id: str):
    """Sirve la imagen ROI recortada de un campo para revisión humana.

    Útil para mostrar al usuario la región de la cédula donde está el campo,
    para que corrija manualmente los campos marcados needs_review=True.

    Output: imagen PNG (FileResponse)
    """
    # Item 2: whitelist estricta contra path traversal (doc_id/field_id se usan
    # para construir rutas de archivo; '..', '/', '\\' quedan rechazados).
    if not ID_SEGURO_RE.match(doc_id) or not ID_SEGURO_RE.match(field_id):
        raise HTTPException(400, "doc_id/field_id inválidos: solo se permite [a-zA-Z0-9_-]+.")

    rois_base = Path(ESTADO.get("ocr_processed_dir", "")) / "rois"
    rois_base_resuelto = rois_base.resolve()
    roi_dir = (rois_base / doc_id).resolve()

    # Defensa en profundidad: aunque la regex ya bloquea '..', se verifica que
    # la ruta resultante siga dentro del directorio base esperado.
    if not roi_dir.is_relative_to(rois_base_resuelto):
        raise HTTPException(400, "Ruta fuera del directorio permitido.")

    if roi_dir.is_dir():
        for roi_file in roi_dir.rglob(f"{field_id}*.png"):
            roi_file_resuelto = roi_file.resolve()
            if not roi_file_resuelto.is_relative_to(rois_base_resuelto):
                continue
            return FileResponse(str(roi_file_resuelto), media_type="image/png")

    raise HTTPException(404, f"ROI no encontrado para {doc_id}/{field_id}")


@app.get("/ocr/campos-template", dependencies=REQUIERE_API_KEY)
def ocr_campos_template():
    """Devuelve la definición de campos del template OCR.

    Útil para que el frontend genere formularios de corrección/validación.
    Muestra qué campos se extraen, su tipo, y en qué página están.

    Output: {"paginas": {"datos_vivienda": [{"id": "vivienda.agua_entubada.si", "type": "checkbox", "bbox": [...]}, ...]}}
    """
    field_map = ESTADO.get("ocr_field_map")
    if not field_map:
        raise HTTPException(503, "Field map no cargado.")
    return field_map


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints Adicionales Riesgo (Subcomponente B)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/riesgo/metricas", dependencies=REQUIERE_API_KEY)
def riesgo_metricas():
    """Devuelve la tabla de comparación de los 3 modelos entrenados.

    Incluye accuracy, F1 macro, y validación cruzada 5-fold.

    Output: {
      "ganador": "XGBoost",
      "modelos": {
        "XGBoost": {"Accuracy": 0.9563, "F1_Macro": 0.9576, "CV_F1_Macro_Mean": 0.9637, "CV_F1_Macro_Std": 0.0055},
        "Random Forest": {...},
        "Decision Tree": {...}
      }
    }
    """
    csv_path = B_DIR / "data" / "processed" / "model_comparison.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Métricas no disponibles. El modelo no ha sido entrenado.")

    df = pd.read_csv(str(csv_path))
    modelos = {}
    # La primera columna es el nombre del modelo
    model_col = df.columns[0]
    for _, row in df.iterrows():
        nombre = row[model_col]
        modelos[nombre] = {col: round(float(row[col]), 4) for col in df.columns if col != model_col}

    return {"ganador": ESTADO.get("winner", "N/A"), "modelos": modelos}


class LoteFamilias(BaseModel):
    """Lista de familias para predicción en lote."""
    familias: list[FamiliaFeatures]


@app.post("/riesgo/predecir-lote", dependencies=REQUIERE_API_KEY)
def predecir_lote(lote: LoteFamilias):
    """Clasifica el riesgo de MÚLTIPLES familias en una sola llamada.

    Máximo 500 familias por lote. Útil para procesar cédulas en batch desde el frontend.

    Input: {"familias": [{...23 features...}, {...23 features...}]}
    Output: {"modelo": "XGBoost", "resultados": [
      {"indice": 0, "nivel_riesgo": "ALTO", "probabilidad_alto": 0.98},
      {"indice": 1, "nivel_riesgo": "BAJO", "probabilidad_alto": 0.05}
    ]}
    """
    if len(lote.familias) > 500:
        raise HTTPException(400, "Máximo 500 familias por lote.")
    if len(lote.familias) == 0:
        raise HTTPException(400, "La lista de familias no puede estar vacía.")

    filas = pd.DataFrame([f.model_dump() for f in lote.familias])[FEATURES]
    preds, probs = _predecir(ESTADO["pipe"], filas, ESTADO["le"])

    return {
        "modelo": ESTADO["winner"],
        "total": len(preds),
        "resultados": [
            {"indice": i, "nivel_riesgo": str(preds[i]), "probabilidad_alto": round(float(probs[i]), 4)}
            for i in range(len(preds))
        ],
    }


@app.get("/riesgo/modelo-info", dependencies=REQUIERE_API_KEY)
def modelo_info():
    """Información detallada del modelo de riesgo activo.

    Output: {
      "modelo_ganador": "XGBoost",
      "n_features": 23,
      "features": {"numericas": [...], "categoricas": [...], "booleanas": [...]},
      "clases": ["BAJO", "MEDIO", "ALTO"],
      "n_familias_alto": 1560,
      "dataset_origen": "synthetic_data.csv"
    }
    """
    from etl_pipeline import FEATURES_NUMERICAS, FEATURES_CATEGORICAS, FEATURES_BOOLEANAS

    return {
        "modelo_ganador": ESTADO.get("winner"),
        "n_features": len(FEATURES),
        "features": {
            "numericas": FEATURES_NUMERICAS,
            "categoricas": FEATURES_CATEGORICAS,
            "booleanas": FEATURES_BOOLEANAS,
        },
        "clases": ["BAJO", "MEDIO", "ALTO"],
        "n_familias_alto": len(ESTADO.get("lista", [])),
        "dataset_origen": "synthetic_data.csv",
    }


@app.get("/riesgo/graficas/{tipo}", dependencies=REQUIERE_API_KEY)
def riesgo_graficas(tipo: Literal["confusion_matrix", "feature_importance"]):
    """Sirve las gráficas PNG generadas durante el entrenamiento.

    tipo = "confusion_matrix" → matriz de confusión del modelo ganador
    tipo = "feature_importance" → importancia de features (Random Forest)

    Frontend: <img src="http://localhost:8001/riesgo/graficas/confusion_matrix" />
    """
    archivo = B_DIR / "data" / "processed" / f"{tipo}.png"
    if not archivo.exists():
        raise HTTPException(404, f"Gráfica '{tipo}' no encontrada. Entrene el modelo primero.")
    return FileResponse(str(archivo), media_type="image/png")


@app.get("/catalogos", dependencies=REQUIERE_API_KEY)
def obtener_catalogos():
    """Devuelve TODOS los catálogos oficiales de la cédula SUMS.

    Útil para poblar dropdowns/selects en el frontend cuando el usuario
    captura una cédula o hace una predicción de riesgo.

    Los valores coinciden exactamente con los de seeder.sql de la sums-API.

    Output: {"materiales_techo": ["Concreto o cemento", "Madera", "Lámina"], ...}
    """
    from catalogos_sums import (
        CAT_MATERIAL_TECHO_PAREDES, CAT_MATERIAL_PISO,
        CAT_MANEJO_EXCRETAS, CAT_ESTADO_CIVIL, CAT_PARENTESCO,
        CAT_TOXICOMANIA, CAT_ENFERMEDAD_CRONICA, CAT_INGRESO_SALARIAL,
        CAT_ESCOLARIDAD, CAT_LENGUA, CAT_ANIMAL,
        CAT_ATENCION_EMBARAZO, CAT_FRECUENCIA_SERVICIO_SALUD,
        VACUNAS, CAT_DOSIS,
    )
    return {
        "materiales_techo": CAT_MATERIAL_TECHO_PAREDES,
        "materiales_paredes": CAT_MATERIAL_TECHO_PAREDES,
        "materiales_piso": CAT_MATERIAL_PISO,
        "manejo_excretas": CAT_MANEJO_EXCRETAS,
        "estados_civiles": CAT_ESTADO_CIVIL,
        "parentescos": CAT_PARENTESCO,
        "toxicomanias": CAT_TOXICOMANIA,
        "enfermedades_cronicas": CAT_ENFERMEDAD_CRONICA,
        "ingresos_salariales": CAT_INGRESO_SALARIAL,
        "escolaridades": CAT_ESCOLARIDAD,
        "lenguas": CAT_LENGUA,
        "animales": CAT_ANIMAL,
        "atencion_embarazo": CAT_ATENCION_EMBARAZO,
        "frecuencia_servicio_salud": CAT_FRECUENCIA_SERVICIO_SALUD,
        "vacunas": VACUNAS,
        "dosis": CAT_DOSIS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints Adicionales Búsqueda (Subcomponente C)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/corpus/estadisticas", dependencies=REQUIERE_API_KEY)
def corpus_estadisticas():
    """Estadísticas del corpus de búsqueda indexado.

    Output: {
      "n_documentos": 150,
      "avg_tokens_por_doc": 42.5,
      "vocabulario_size": 620,
      "motores_disponibles": ["bm25", "tfidf"],
      "temas": ["vivienda", "agua", "cronica", ...]
    }
    """
    bm25 = ESTADO.get("bm25")
    if not bm25:
        raise HTTPException(503, "Motor de búsqueda no inicializado.")

    # Intentar cargar temas si existen
    temas_path = C_DIR / "data" / "corpus_themes.json"
    temas = []
    if temas_path.exists():
        themes_data = json.loads(temas_path.read_text(encoding="utf-8"))
        all_themes = set()
        for doc_themes in themes_data.values():
            all_themes.update(doc_themes)
        temas = sorted(all_themes)

    return {
        "n_documentos": bm25.N,
        "avg_tokens_por_doc": round(bm25.avgdl, 1),
        "vocabulario_size": len(bm25.IDF_BM25),
        "motores_disponibles": ["bm25", "tfidf"],
        "temas": temas,
    }


@app.get("/corpus/documento/{doc_id}", dependencies=REQUIERE_API_KEY)
def obtener_documento(doc_id: str):
    """Devuelve un documento específico del corpus por su ID.

    IDs tienen formato 'nXXX' (ej. 'n001', 'n150').

    Output: {"id": "n001", "titulo": "Nota de visita 001 - vivienda", "texto": "Visita domiciliaria en colonia..."}
    """
    texto = ESTADO["textos"].get(doc_id)
    if texto is None:
        raise HTTPException(404, f"Documento '{doc_id}' no encontrado. Los IDs tienen formato nXXX.")
    return {
        "id": doc_id,
        "titulo": ESTADO["titulos"].get(doc_id, ""),
        "texto": texto,
    }


@app.get("/buscar/metricas", dependencies=REQUIERE_API_KEY)
def buscar_metricas():
    """Ejecuta la evaluación IR de ambos motores con los 8 queries de prueba.

    Usa relevancia graduada (0-3) para nDCG. Los otros métricas usan binaria (>0).

    Output: {
      "tfidf": {"P@5": 0.925, "R@5": 0.115, "MRR": 0.917, "MAP": 0.685, "nDCG@5": 0.520},
      "bm25":  {"P@5": 0.950, "R@5": 0.118, "MRR": 1.000, "MAP": 0.691, "nDCG@5": 0.569},
      "ganador": "bm25",
      "criterio": "nDCG@5",
      "n_queries_evaluados": 8
    }
    """
    qrels_path = C_DIR / "data" / "qrels_sums.json"
    if not qrels_path.exists():
        raise HTTPException(404, "Archivo qrels no encontrado. Ejecute run_all.py del subcomponente C primero.")

    from ir_metrics import precision_at_k, recall_at_k, mrr, average_precision, ndcg_at_k

    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    k = 5

    def _evaluar_motor(motor_obj, buscar_fn_name):
        buscar = getattr(motor_obj, buscar_fn_name)
        metricas_acum = {"P@5": [], "R@5": [], "MRR": [], "MAP": [], "nDCG@5": []}
        for query_text, rels in qrels.items():
            # Buscar en todo el corpus
            ranking = buscar(query_text, k=150)  # todo el corpus
            metricas_acum["P@5"].append(precision_at_k(ranking, qrels, query_text, k))
            metricas_acum["R@5"].append(recall_at_k(ranking, qrels, query_text, k))
            metricas_acum["MRR"].append(mrr(ranking, qrels, query_text))
            metricas_acum["MAP"].append(average_precision(ranking, qrels, query_text))
            metricas_acum["nDCG@5"].append(ndcg_at_k(ranking, qrels, query_text, k))
        
        if not metricas_acum["P@5"]:
            return {}

        return {m: round(sum(vs)/len(vs), 4) for m, vs in metricas_acum.items()}

    met_tfidf = _evaluar_motor(ESTADO["tfidf"], "buscar_tfidf")
    met_bm25 = _evaluar_motor(ESTADO["bm25"], "buscar_bm25")

    ganador = "bm25" if met_bm25.get("nDCG@5", 0) >= met_tfidf.get("nDCG@5", 0) else "tfidf"

    return {
        "tfidf": met_tfidf,
        "bm25": met_bm25,
        "ganador": ganador,
        "criterio": "nDCG@5",
        "n_queries_evaluados": len(qrels),
    }


class CorpusConfig(BaseModel):
    """Configuración para reindexación del corpus."""
    fuente: Literal["archivo", "sintetico"] = "sintetico"
    n_documentos: int = Field(150, ge=10, le=500)


@app.post("/corpus/reindexar", dependencies=REQUIERE_API_KEY)
def corpus_reindexar(config: CorpusConfig):
    """Reconstruye el corpus y reindexa ambos motores de búsqueda.

    fuente='sintetico' → regenera notas sintéticas con corpus_builder.
    fuente='archivo' → recarga desde los JSON existentes en disco.

    ⚠️ Esta operación bloquea la API durante unos segundos.

    Output: {"ok": true, "n_documentos": 150, "mensaje": "Corpus reindexado exitosamente."}
    """
    from corpus_builder import main as build_corpus_main
    from preprocess import main as preprocess_main

    if config.fuente == "sintetico":
        # Regenerar corpus (usa data/ relativo al subcomponente)
        import os
        old_cwd = os.getcwd()
        os.chdir(str(C_DIR))
        try:
            build_corpus_main()  # genera corpus_crudo + corpus_themes
            preprocess_main()    # genera corpus_procesado
        finally:
            os.chdir(old_cwd)

    # Recargar desde archivos
    proc = json.loads((C_DIR / "data" / "corpus_procesado_sums.json").read_text(encoding="utf-8"))
    crudo = json.loads((C_DIR / "data" / "corpus_crudo_sums.json").read_text(encoding="utf-8"))

    # Reinicializar motores y lookups
    ESTADO["textos"] = {d["id"]: d.get("texto", "") for d in crudo}
    ESTADO["titulos"] = {d["id"]: d.get("titulo", "") for d in crudo}
    ESTADO["tfidf"] = MotorTFIDF(proc)
    ESTADO["bm25"] = MotorBM25(proc)

    return {"ok": True, "n_documentos": len(proc), "mensaje": "Corpus reindexado exitosamente."}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint de Datos / Generación
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/datos/estadisticas", dependencies=REQUIERE_API_KEY)
def datos_estadisticas():
    """Resumen del dataset usado para entrenar el modelo de riesgo.

    Output: {
      "n_familias": 4000,
      "distribucion_riesgo": {"ALTO": 1560, "MEDIO": 1200, "BAJO": 1240},
      "n_features": 23,
      "archivo_origen": "synthetic_data.csv"
    }
    """
    csv_path = B_DIR / "data" / "synthetic_data.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Dataset no encontrado. Ejecute run_all.py del subcomponente B.")

    df = pd.read_csv(str(csv_path))
    dist = df["nivel_riesgo"].value_counts().to_dict()

    return {
        "n_familias": len(df),
        "distribucion_riesgo": {k: int(v) for k, v in dist.items()},
        "n_features": len(FEATURES),
        "archivo_origen": "synthetic_data.csv",
    }
