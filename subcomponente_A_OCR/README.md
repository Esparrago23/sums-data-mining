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
