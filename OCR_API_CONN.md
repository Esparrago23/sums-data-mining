# API de análisis de PDFs SUMS

Este documento describe cómo conectar, enviar PDFs para análisis y obtener los resultados desde la API del modelo OCR/ETL del subcomponente A.

## 1. Requisitos previos

- Tener Python 3.10+ instalado.
- Tener acceso al repositorio y a la rama donde están los cambios.
- Tener el entorno virtual del proyecto activado.

## 2. Clonar o entrar al proyecto

```bash
cd /ruta/al/proyecto
source .venv/bin/activate
```

Si el entorno virtual no existe, crear uno e instalar dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Ejecutar la API

Desde la raíz del proyecto, ejecutar:

```bash
cd subcomponente_A_OCR
python src/run_all.py
```

Este comando procesa los PDFs que estén en:

- [subcomponente_A_OCR/data/raw_pdfs](subcomponente_A_OCR/data/raw_pdfs)

Y genera los resultados en:

- [subcomponente_A_OCR/data/processed/predictions.json](subcomponente_A_OCR/data/processed/predictions.json)
- [subcomponente_A_OCR/data/processed/report.json](subcomponente_A_OCR/data/processed/report.json)
- [subcomponente_A_OCR/data/processed/review_output.json](subcomponente_A_OCR/data/processed/review_output.json)

## 4. Cómo enviar un PDF para análisis

1. Colocar el archivo PDF en:
   - [subcomponente_A_OCR/data/raw_pdfs](subcomponente_A_OCR/data/raw_pdfs)

2. Asegurarse de que el nombre sea claro, por ejemplo:
   - `Cédula_0011.pdf`

3. Ejecutar nuevamente:

```bash
cd subcomponente_A_OCR
python src/run_all.py
```

## 5. Cómo obtener los resultados

Después de ejecutar el proceso, revisar los archivos generados:

- Predicciones estructuradas:
  - [subcomponente_A_OCR/data/processed/predictions.json](subcomponente_A_OCR/data/processed/predictions.json)

- Resumen del proceso:
  - [subcomponente_A_OCR/data/processed/report.json](subcomponente_A_OCR/data/processed/report.json)

- Resultado preparado para revisión:
  - [subcomponente_A_OCR/data/processed/review_output.json](subcomponente_A_OCR/data/processed/review_output.json)

## 6. Formato esperado de respuesta

El archivo `predictions.json` contiene una estructura por documento con campos como:

```json
{
  "Cédula_0011": {
    "fields": {
      "familia.nombre_informante": {
        "value": "texto extraído",
        "confidence": 0.65,
        "needs_review": false
      }
    }
  }
}
```

## 7. Recomendación para pruebas

Para pruebas iniciales, usar PDFs con la misma estructura de la cédula original y con buena calidad de escaneo.

## 8. Nota importante

El modelo actual está pensado como un pipeline de extracción semi-automática para pruebas y validación, no como una herramienta de OCR “100% automática” sin revisión.
