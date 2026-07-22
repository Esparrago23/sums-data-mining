# -*- coding: utf-8 -*-
"""
conftest.py - Subcomponente C (Motor de Busqueda SUMS)

Agrega ../src al sys.path para que los tests puedan importar los modulos
del motor de busqueda (ir_metrics, bm25_engine, tfidf_engine, preprocess)
sin necesidad de instalar el paquete.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
