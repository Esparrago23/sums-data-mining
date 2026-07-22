# Subcomponente A — OCR de Cédulas PDF

**Estado: FINALIZADO E INTEGRADO EN LA API.**

Los resultados del modelo se consumen directamente a través del endpoint
`POST /ocr/procesar-cedula` definido en `integracion/api_mineria.py`.
Consultar ese archivo y el `README.md` principal (sección 4.1) para ver
el ejemplo de uso con `curl` y la descripción completa de la respuesta JSON.

---

Objetivo: procesar los PDFs de cédulas escaneadas de `/data/raw_pdfs/`, extraer
los campos estructurados con el pipeline OCR (OpenCV + detección de checkboxes +
Tesseract para campos de texto) y devolver un JSON con los 45 campos definidos en
`config/field_map_sums.json` (37 checkboxes + 6 texto + 2 número).

Estructura:
```
config/
  field_map_sums.json   ← plantilla de campos y coordenadas relativas (bbox)
data/
  raw_pdfs/             ← PDFs de cédulas a procesar
  ground_truth/         ← campos_esperados.json (transcripción manual para métricas)
  processed/
    rendered_pages/     ← páginas PNG renderizadas (180 DPI)
    rois/               ← recortes de campos para revisión humana
    predictions.json    ← salida del pipeline (campos extraídos)
    report.json         ← resumen y métricas de accuracy
src/
  pdf_renderer.py       ← renderiza PDF → PNG con PyMuPDF
  preprocessor.py       ← normaliza imagen, detecta form_box, devuelve PageImage
  field_extractor.py    ← extrae campos por plantilla (extract_document, load_field_map)
  checkbox_model.py     ← modelo de detección de checkboxes por tinta/score
  evaluator.py          ← métricas checkbox vs. ground truth
  run_all.py            ← pipeline completo (batch sobre raw_pdfs/)
notebooks/
  A_OCR_Pipeline.ipynb  ← exploración y validación interactiva
```

Ver el plan completo en `sums-documentos/plan_mineria_datos.md` (Subcomponente A).

## Evaluación de motores OCR generales (docTR / PaddleOCR / DeepSeek-OCR)

Un profesor sugirió evaluar modelos de OCR general entrenados de forma más
amplia. Se evaluó **docTR** (Mindee) como motor alternativo — ver
`src/doctr_engine.py` y `src/comparar_motores_texto.py`.

**Por qué docTR y no los otros dos:**
- Reutiliza **PyTorch**, que ya es dependencia del proyecto (lo instala
  `sentence-transformers` para el buscador semántico del subcomponente C) — no
  agrega un segundo framework de deep learning como sí exigiría **PaddleOCR**
  (necesita `paddlepaddle` aparte).
- Licencia Apache 2.0, arquitecturas "mobile" livianas
  (`db_mobilenet_v3_large` + `crnn_mobilenet_v3_small`) — footprint en disco
  despreciable (el modelo se descargó sin mover el uso de disco de forma
  medible, con ~18GB libres en la máquina de desarrollo). **DeepSeek-OCR** es
  un VLM multimodal de varios GB pensado para GPU, inviable de evaluar con el
  margen de disco/tiempo de esta entrega.

**Alcance intencionalmente limitado**: docTR es un OCR de texto general — no
entiende "checkbox" ni "campo de formulario". Por eso **no reemplaza**
`checkbox_model.py` (el corazón del pipeline: 37 de los 45 campos). Se evaluó
únicamente como motor alternativo para los 5 campos de **texto libre**
(`TEXT_FIELDS` en `text_trainer.py`), sobre los MISMOS ROIs recortados por
plantilla que ya usa Tesseract — comparación justa, cabeza a cabeza.

**Bug real encontrado y corregido en el camino**: en esta máquina de
desarrollo Windows, `cv2.imwrite` fallaba **en silencio** (sin excepción) para
rutas con acentos (ej. `Cédula_0001`), así que **ningún ROI se guardaba en
disco** — ni de checkboxes ni de texto — antes de este fix. Se corrigió en
`field_extractor.py` (`_imwrite_unicode`: codifica con `cv2.imencode` y
escribe los bytes con `pathlib`, el mismo truco que ya usaban
`text_trainer._imread_gray`/`number_trainer._imread_gray` para LEER). Tras el
fix, `run_all.py` corre de punta a punta por primera vez en este entorno:
checkbox accuracy 0.908, number train/test 1.0/1.0, rol train/test 1.0/1.0.
También se instaló el binario de Tesseract (ausente en esta máquina) vía
`winget install --id UB-Mannheim.TesseractOCR -e`.

**Limitación descubierta (preexistente, no de este cambio)**: el ground truth
manual (`data/ground_truth/campos_esperados.json`) **no tiene etiquetas** para
los 5 campos de texto libre — solo cubre checkboxes, números y
`familia.rol_familiar`. Por eso nunca se pudo calcular un accuracy real de
Tesseract en esos campos (`text_split_metrics` sale `null`), ni antes ni
después de este cambio. Etiquetarlos a mano es trabajo futuro razonable.

**Comparación real ejecutada** (`comparar_motores_texto.py`, 200 ROIs = 5
campos × 40 documentos, sin ground truth disponible así que se mide
tasa-no-vacío / longitud / acuerdo / latencia en vez de accuracy):

| Métrica | Tesseract (actual) | docTR |
|---|---|---|
| Tasa de salida no vacía | 98.5% | 100% |
| Longitud promedio | 50.7 caracteres | 50.5 caracteres |
| ms por campo | 443 ms | 998 ms |

Acuerdo palabra-a-palabra (Jaccard) entre ambos motores sobre el mismo ROI:
**0.084** (muy bajo). Inspeccionando ejemplos lado a lado, ambos motores leen
mayoritariamente el **texto impreso de la plantilla** ("informacion de la
familia", "CARACTERISTICAS DE LA VIVIENDA", "Calle, numero, colonia...") en
vez de la respuesta manuscrita — indicio de que el ROI de estos 5 campos
captura demasiado contexto impreso alrededor de la zona manuscrita real. Esta
es una limitación honesta: **cambiar de motor OCR no resuelve un problema de
recorte de ROI** — haría falta ajustar el `bbox` de estos campos en
`field_map_sums.json` para acotar mejor la zona manuscrita antes de que
cualquiera de los dos motores rinda bien ahí.

**Veredicto**: docTR es una alternativa viable y más portátil (no requiere
instalar un binario de sistema aparte, a diferencia de Tesseract — de hecho
Tesseract ni siquiera estaba instalado en esta máquina hasta hoy), pero ~2.3x
más lento por campo y sin evidencia de mejor precisión dado el problema de ROI
compartido. Queda integrado como motor evaluado y documentado, no como
reemplazo de producción del pipeline de checkboxes.
