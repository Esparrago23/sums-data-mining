# -*- coding: utf-8 -*-
"""
grupos_vulnerables.py
======================
Subcomponente B — Banderas de PRIORIDAD por grupo vulnerable.

HALLAZGO que motiva este módulo: el modelo de riesgo (model_trainer.py) clasifica
BAJO/MEDIO/ALTO a partir de features AGREGADAS de la familia (conteos, promedios).
Al agregar, se pierde por completo la composición individual: si hay una mujer
embarazada, un bebé menor de 1 año, o un adulto mayor sin nadie más en el hogar
que lo apoye. Una familia con buena vivienda e ingreso puede salir BAJO/MEDIO y
aun así tener una embarazada sin control prenatal — un caso que, en la práctica
de salud comunitaria (IMSS-Bienestar y similares), es prioritario SIN IMPORTAR
el puntaje agregado. Embarazadas, menores de 5 y adultos mayores son justo los
"grupos prioritarios" estándar de esos programas.

Este módulo NO es un modelo de ML — son reglas determinísticas y auditables
sobre datos que YA se capturan por integrante (edad, atención de embarazo,
vacunas aplicadas) y por vivienda (mascotas) pero que
`synthetic_generator.familia_to_flat` agregaba y descartaba antes de llegar
al CSV/modelo. Se calculan una sola vez aquí y se reutilizan en:
  - `synthetic_generator.py` (columnas adicionales del CSV, fuera de FEATURES)
  - `risk_report.py` (para decidir qué entra a la lista priorizada de visitas,
    y para el resumen de riesgo por zona/colonia)
  - `subcomponente_C_busqueda/src/buscador_estructurado.py` (filtros
    "embarazo" y "mascotas")

MEJORA — riesgo zoonótico (mascotas sin vacunar): una familia puede tener
buena vivienda e ingreso y aun así convivir con un perro/gato dentro de casa
sin esquema de vacunación al corriente -- un canal real de zoonosis (rabia,
parásitos) que el modelo agregado tampoco ve. El dato ya existe en
`vivienda.perros_gatos_dentro` / `vivienda.mascotas_vacunas_corrientes`
(ver BD_MAPPING.md), solo no se usaba.
"""
from __future__ import annotations

EDAD_BEBE_MESES = 1  # "menor de 1 año" -> edad en años < 1
EDAD_LIMITE_ESQUEMA_INFANTIL = 5  # VACUNAS_INFANTILES se aplican hasta esta edad
EDAD_ADULTO_MAYOR = 60


def calcular_banderas(
    integrantes: list[dict],
    vacunas_aplicadas: list[dict] | None = None,
    vivienda: dict | None = None,
) -> dict:
    """Calcula las banderas de prioridad de UNA familia a partir de sus
    integrantes (lista de dicts con al menos 'edad'; opcionalmente
    'atencion_embarazo' si aplica), sus vacunas aplicadas (lista de dicts con
    'paciente' = nombre del integrante) y, opcionalmente, su vivienda (dict
    con 'perros_gatos_dentro' / 'mascotas_vacunas_corrientes') para la
    bandera zoonótica -- si no se pasa `vivienda`, esa bandera queda en False
    (caso del endpoint /riesgo/predecir, que recibe features planas y no la
    vivienda anidada; ahí el caller la provee directo como campo plano).

    Devuelve:
      tiene_embarazada                 -> algún integrante con atención de embarazo registrada
      tiene_menor_1_anio                -> algún integrante con edad < 1
      tiene_menor_5_sin_vacunas          -> algún integrante < 5 años SIN NINGÚN registro de vacuna
      tiene_adulto_mayor_solo            -> todos los integrantes tienen 60+ años (nadie más joven en el hogar)
      tiene_mascota_sin_vacunar          -> mascota dentro de la vivienda sin esquema de vacunación al corriente
      requiere_atencion_prioritaria      -> OR de todas las anteriores (bandera resumen)
    """
    vacunas_aplicadas = vacunas_aplicadas or []
    nombres_vacunados = {v.get("paciente") for v in vacunas_aplicadas if v.get("paciente")}

    tiene_embarazada = any(i.get("atencion_embarazo") for i in integrantes)
    tiene_menor_1_anio = any(i.get("edad", 99) < EDAD_BEBE_MESES for i in integrantes)
    tiene_menor_5_sin_vacunas = any(
        i.get("edad", 99) < EDAD_LIMITE_ESQUEMA_INFANTIL and i.get("nombre") not in nombres_vacunados
        for i in integrantes
    )
    tiene_adulto_mayor_solo = bool(integrantes) and all(
        i.get("edad", 0) >= EDAD_ADULTO_MAYOR for i in integrantes
    )
    tiene_mascota_sin_vacunar = bool(
        vivienda
        and vivienda.get("perros_gatos_dentro")
        and not vivienda.get("mascotas_vacunas_corrientes")
    )

    return {
        "tiene_embarazada": tiene_embarazada,
        "tiene_menor_1_anio": tiene_menor_1_anio,
        "tiene_menor_5_sin_vacunas": tiene_menor_5_sin_vacunas,
        "tiene_adulto_mayor_solo": tiene_adulto_mayor_solo,
        "tiene_mascota_sin_vacunar": tiene_mascota_sin_vacunar,
        "requiere_atencion_prioritaria": (
            tiene_embarazada or tiene_menor_1_anio or tiene_menor_5_sin_vacunas
            or tiene_adulto_mayor_solo or tiene_mascota_sin_vacunar
        ),
    }


BANDERAS_COLUMNAS = [
    "tiene_embarazada", "tiene_menor_1_anio", "tiene_menor_5_sin_vacunas",
    "tiene_adulto_mayor_solo", "tiene_mascota_sin_vacunar", "requiere_atencion_prioritaria",
]


def motivo_prioridad(banderas: dict, nivel_riesgo_ml: str | None = None) -> str:
    """Explicación legible de POR QUÉ una familia entró a la lista priorizada
    -- para que el equipo de salud sepa qué esperar antes de la visita, no
    solo que "salió priorizada"."""
    motivos = []
    if nivel_riesgo_ml == "ALTO":
        motivos.append("riesgo familiar ALTO (modelo)")
    if banderas.get("tiene_embarazada"):
        motivos.append("integrante embarazada")
    if banderas.get("tiene_menor_1_anio"):
        motivos.append("menor de 1 año en el hogar")
    if banderas.get("tiene_menor_5_sin_vacunas"):
        motivos.append("menor de 5 años sin vacunas registradas")
    if banderas.get("tiene_adulto_mayor_solo"):
        motivos.append("adulto(s) mayor(es) sin acompañante en el hogar")
    if banderas.get("tiene_mascota_sin_vacunar"):
        motivos.append("mascota en la vivienda sin vacunación al corriente (riesgo zoonótico)")
    return "; ".join(motivos) if motivos else "sin motivo de prioridad"
