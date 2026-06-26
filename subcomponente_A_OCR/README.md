# Subcomponente A — OCR de Cédulas PDF

**Estado: PENDIENTE.** Se desarrollará después del Subcomponente B.

Objetivo: procesar los 10 PDFs reales de `/Cedulas_pdf/`, extraer los campos con OCR
(EasyOCR + OpenCV + spaCy) y compararlos contra un ground truth manual para medir la
precisión. La inserción a la BD reutilizará el mismo endpoint `captura-completa` que ya
usa el Subcomponente B.

Estructura preparada:
```
data/raw_pdfs/        ← copiar aquí los 10 PDFs de /Cedulas_pdf/
data/ground_truth/    ← campos_esperados.json (transcripción manual)
src/                  ← preprocessor.py, ocr_engine.py, field_extractor.py, db_loader.py
notebooks/            ← A_OCR_Pipeline.ipynb
```

Ver el plan completo en `sums-documentos/plan_mineria_datos.md` (Subcomponente A).
