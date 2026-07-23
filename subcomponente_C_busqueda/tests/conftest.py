# -*- coding: utf-8 -*-
"""
conftest.py - Subcomponente C (Motor de Busqueda SUMS)

Agrega ../src al sys.path para que los tests puedan importar los modulos
del motor de busqueda (ir_metrics, bm25_engine, tfidf_engine, preprocess)
sin necesidad de instalar el paquete.

Tambien agrega subcomponente_B_ETL_Risk/src: buscador_estructurado.py
reutiliza a proposito los catalogos (catalogos_sums.py) y las banderas de
grupos vulnerables (grupos_vulnerables.py) de B en vez de duplicarlos.
"""
import os
import sys

THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "..", "subcomponente_B_ETL_Risk", "src"))
