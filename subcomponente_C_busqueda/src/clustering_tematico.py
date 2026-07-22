# -*- coding: utf-8 -*-
"""
clustering_tematico.py - Subcomponente C (Motor de Busqueda SUMS)

Demostracion de APRENDIZAJE NO SUPERVISADO sobre el corpus.

Toda la evaluacion de C hasta aqui (TF-IDF/BM25/embeddings via /buscar/metricas)
es RECUPERACION supervisada por diseno: se sabe de antemano que documento debe
encontrar cada consulta (qrels_sums.json). Este script muestra el otro lado:
agrupar las notas SIN usar la etiqueta de tema (corpus_themes.json), y comparar
los grupos descubiertos contra el tema real SOLO como validacion posterior (el
algoritmo de clustering en si nunca ve corpus_themes.json).

Metodo: KMeans sobre los embeddings semanticos ya calculados por
embeddings_engine.MotorSemantico (mismo motor que usa /buscar?motor=semantico:
se reutiliza el mismo computo, no se agrega ninguna dependencia nueva -- sklearn
ya es dependencia base del subcomponente B).

Metricas de validacion EXTERNA (comparan los clusters contra el tema real):
  - Adjusted Rand Index (ARI): concordancia con el agrupamiento real corregida
    por azar; ~0 = azar, 1.0 = coincide exactamente.
  - Purity: fraccion de notas cuyo cluster coincide con el tema mayoritario de
    ese cluster (facil de explicar: "que tan puros quedaron los grupos").

Uso:  python clustering_tematico.py
"""
from __future__ import annotations

import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
DATA_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "data"))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from embeddings_engine import MotorSemantico


def purity_score(labels_true, labels_pred) -> float:
    """Fraccion de puntos correctamente agrupados si a cada cluster se le
    asigna la etiqueta real MAYORITARIA dentro de el (metrica de pureza,
    forma estandar de validar clustering contra etiquetas conocidas sin
    exigir que el numero/orden de clusters coincida con las clases)."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    total_correctas = 0
    for cluster in np.unique(labels_pred):
        mask = labels_pred == cluster
        if not mask.any():
            continue
        _, conteos = np.unique(labels_true[mask], return_counts=True)
        total_correctas += conteos.max()
    return total_correctas / len(labels_true)


def main() -> int:
    with open(os.path.join(DATA_DIR, "corpus_crudo_sums.json"), encoding="utf-8") as fh:
        crudo = json.load(fh)
    with open(os.path.join(DATA_DIR, "corpus_themes.json"), encoding="utf-8") as fh:
        temas_por_doc = json.load(fh)  # {id: [temas en orden de dominancia]}

    tema_dominante = {doc_id: temas[0] for doc_id, temas in temas_por_doc.items()}
    ids_ordenados = [d["id"] for d in crudo]
    y_real = [tema_dominante.get(doc_id, "desconocido") for doc_id in ids_ordenados]
    n_temas = len(set(y_real))

    print(f"Corpus: {len(crudo)} notas, {n_temas} temas reales distintos "
          f"({sorted(set(y_real))}).")
    print("Calculando embeddings semanticos (Sentence-BERT) para clustering "
          "(reutiliza embeddings_engine.MotorSemantico, el mismo del buscador)...")
    motor = MotorSemantico(crudo)

    print(f"Ejecutando KMeans(n_clusters={n_temas}) sobre los embeddings "
          f"-- SIN usar las etiquetas de tema en ningun momento...")
    kmeans = KMeans(n_clusters=n_temas, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(motor.embeddings)

    ari = adjusted_rand_score(y_real, y_pred)
    pureza = purity_score(y_real, y_pred)

    print(f"\nAdjusted Rand Index (clusters vs. tema real, solo validacion): {ari:.4f}")
    print(f"Purity (clusters vs. tema real, solo validacion):                {pureza:.4f}")
    print("\nLectura: ARI cercano a 1.0 / Purity cercana a 1.0 significa que el "
          "agrupamiento NO SUPERVISADO redescubrio, sin haberlos visto nunca, "
          "practicamente los mismos 9 temas que ya conociamos por construccion "
          "del corpus (corpus_builder.py). Un valor bajo indicaria que los "
          "embeddings no separan bien los temas y habria que revisar el modelo "
          "o el numero de clusters.")

    assert 0.0 <= pureza <= 1.0
    assert -1.0 <= ari <= 1.0
    return 0


if __name__ == "__main__":
    sys.exit(main())
