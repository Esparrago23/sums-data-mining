# -*- coding: utf-8 -*-
"""
conftest.py — configuración compartida para las pruebas de integración de
`integracion/api_mineria.py`.

MUY IMPORTANTE (orden de ejecución):
api_mineria.py lee `MINERIA_API_KEY` de `os.environ` A NIVEL DE MÓDULO (queda
guardada en la constante módulo `MINERIA_API_KEY` en el instante en que se
importa el módulo, ver línea `MINERIA_API_KEY = os.environ.get(...)`). Por eso
las variables de entorno se fijan AQUÍ, en conftest.py, ANTES de hacer
`import api_mineria` — y antes de que pytest importe cualquier test_*.py del
mismo directorio (pytest siempre carga conftest.py primero). Si se fijaran
después (p. ej. dentro de un test o de un fixture que corre tarde) el valor ya
quedaría "congelado" como None y todos los endpoints protegidos responderían
503 sin importar el header enviado.
"""
import os
import sys
from pathlib import Path

# Clave de API usada en toda la suite (se envía como header X-API-Key).
TEST_API_KEY = "test-clave-integracion-12345"

os.environ["MINERIA_API_KEY"] = TEST_API_KEY
os.environ.setdefault(
    "MINERIA_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
)

# integracion/ (el padre de esta carpeta tests/) debe estar en sys.path para
# poder hacer `import api_mineria` sin instalar el paquete.
INTEGRACION_DIR = Path(__file__).resolve().parent.parent
if str(INTEGRACION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRACION_DIR))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api_mineria  # noqa: E402  (import intencionalmente tardío: después de fijar env vars)


@pytest.fixture(scope="session")
def api_key() -> str:
    """La API key de prueba fijada en MINERIA_API_KEY para toda la sesión."""
    return TEST_API_KEY


@pytest.fixture(scope="session")
def client():
    """Cliente de prueba compartido por TODA la suite (scope="session").

    Se abre `with TestClient(app) as c:` UNA SOLA VEZ para disparar el
    `lifespan` de FastAPI (carga de spaCy/motores de búsqueda, entrenamiento o
    carga-desde-cache del modelo de riesgo, etc.) una sola vez en toda la
    corrida, en vez de repetir ese arranque costoso en cada test individual.
    """
    with TestClient(api_mineria.app) as c:
        yield c
