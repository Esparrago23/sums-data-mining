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

MEJORA — hacinamiento severo: en `compute_risk` (synthetic_generator.py), la
densidad de personas por cuarto solo sumaba +1 punto de 12 con un único
umbral binario (>2.5), el mismo peso que cocinar con leña -- una familia con
2.6 personas/cuarto puntuaba IGUAL que una con 15. Se corrigió esa fórmula
para que la densidad severa pese más (ver compute_risk), pero además, igual
que con embarazada/adulto mayor solo, el hacinamiento severo es un factor de
riesgo respiratorio/de transmisión bien documentado que amerita prioridad de
visita SIN IMPORTAR el puntaje agregado del modelo -- de ahí esta bandera.

MEJORA — dos niveles de prioridad en vez de "cualquier bandera = urgente":
al agregar más banderas (zoonótica, hacinamiento) se volvió evidente un
problema de diseño: CUALQUIER bandera activa, sin importar qué tan leve,
forzaba prioridad_visita="URGENTE" -- una familia con un bebé sano de resto
bien salía "BAJO riesgo, 0% probabilidad" Y "VISITA URGENTE" en la misma
respuesta, contradictorio de cara al usuario, y le restaba peso a la
palabra "urgente" para los casos que sí lo son. Ahora las banderas se
dividen en dos niveles (BANDERAS_CRITICAS abajo):
  - CRÍTICAS (fuerzan URGENTE sin importar el score, como antes): embarazada,
    menor de 5 sin vacunas, adulto mayor solo, hacinamiento severo -- son
    condiciones tiempo-sensibles o estructurales de salud pública donde
    esperar a que el score agregado suba no es aceptable.
  - Solo-score (tiene_menor_1_anio, tiene_mascota_sin_vacunar): ya no fuerzan
    URGENTE por sí solas -- un bebé sano o una mascota sin vacunar en una
    familia por lo demás estable NO amerita visita urgente, pero SÍ debe
    subir el riesgo agregado. Por eso se promovieron a features reales del
    modelo (ver etl_pipeline.FEATURES_BOOLEANAS) y a puntos en compute_risk,
    en vez de quedarse solo como una alerta que no movía nada.
"""
from __future__ import annotations

EDAD_BEBE_MESES = 1  # "menor de 1 año" -> edad en años < 1
EDAD_LIMITE_ESQUEMA_INFANTIL = 5  # VACUNAS_INFANTILES se aplican hasta esta edad
EDAD_ADULTO_MAYOR = 60
UMBRAL_HACINAMIENTO_SEVERO = 3.0  # personas/cuarto -- referencia CONEVAL/OMS de hacinamiento crítico

# Banderas que fuerzan prioridad_visita="URGENTE" sin importar el nivel de
# riesgo ML (condiciones tiempo-sensibles/estructurales de salud pública que
# no deben esperar a que el score agregado suba). Las banderas de
# BANDERAS_COLUMNAS que NO están aquí (tiene_menor_1_anio,
# tiene_mascota_sin_vacunar) siguen calculándose y reportándose, pero solo
# influyen en el riesgo de forma gradual a través del modelo (ver
# etl_pipeline.FEATURES_BOOLEANAS), no como una alerta binaria.
BANDERAS_CRITICAS = {
    "tiene_embarazada",
    "tiene_menor_5_sin_vacunas",
    "tiene_adulto_mayor_solo",
    "tiene_hacinamiento_severo",
}


def calcular_banderas(
    integrantes: list[dict],
    vacunas_aplicadas: list[dict] | None = None,
    vivienda: dict | None = None,
) -> dict:
    """Calcula las banderas de prioridad de UNA familia a partir de sus
    integrantes (lista de dicts con al menos 'edad'; opcionalmente
    'atencion_embarazo' si aplica), sus vacunas aplicadas (lista de dicts con
    'paciente' = nombre del integrante) y, opcionalmente, su vivienda (dict
    con 'perros_gatos_dentro' / 'mascotas_vacunas_corrientes' / 'numero_cuartos'
    / 'numero_habitantes') para las banderas zoonótica y de hacinamiento -- si
    no se pasa `vivienda`, esas banderas quedan en False (caso del endpoint
    /riesgo/predecir, que recibe features planas y no la vivienda anidada;
    ahí el caller las provee directo como campos planos).

    Devuelve:
      tiene_embarazada                 -> algún integrante con atención de embarazo registrada
      tiene_menor_1_anio                -> algún integrante con edad < 1
      tiene_menor_5_sin_vacunas          -> algún integrante < 5 años SIN NINGÚN registro de vacuna
      tiene_adulto_mayor_solo            -> todos los integrantes tienen 60+ años (nadie más joven en el hogar)
      tiene_mascota_sin_vacunar          -> mascota dentro de la vivienda sin esquema de vacunación al corriente
      tiene_hacinamiento_severo          -> personas/cuarto por encima de UMBRAL_HACINAMIENTO_SEVERO
      requiere_atencion_prioritaria      -> OR de las banderas CRÍTICAS (BANDERAS_CRITICAS) -- NO de
                                            todas; tiene_menor_1_anio y tiene_mascota_sin_vacunar se
                                            calculan y devuelven igual (para reporte/motivo), pero ya
                                            no fuerzan esta bandera por sí solas (ver módulo).
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
    tiene_hacinamiento_severo = False
    if vivienda and vivienda.get("numero_cuartos"):
        personas_por_cuarto = vivienda.get("numero_habitantes", 0) / max(1, vivienda["numero_cuartos"])
        tiene_hacinamiento_severo = personas_por_cuarto > UMBRAL_HACINAMIENTO_SEVERO

    banderas = {
        "tiene_embarazada": tiene_embarazada,
        "tiene_menor_1_anio": tiene_menor_1_anio,
        "tiene_menor_5_sin_vacunas": tiene_menor_5_sin_vacunas,
        "tiene_adulto_mayor_solo": tiene_adulto_mayor_solo,
        "tiene_mascota_sin_vacunar": tiene_mascota_sin_vacunar,
        "tiene_hacinamiento_severo": tiene_hacinamiento_severo,
    }
    banderas["requiere_atencion_prioritaria"] = any(
        banderas[nombre] for nombre in BANDERAS_CRITICAS
    )
    return banderas


BANDERAS_COLUMNAS = [
    "tiene_embarazada", "tiene_menor_1_anio", "tiene_menor_5_sin_vacunas",
    "tiene_adulto_mayor_solo", "tiene_mascota_sin_vacunar", "tiene_hacinamiento_severo",
    "requiere_atencion_prioritaria",
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
    if banderas.get("tiene_hacinamiento_severo"):
        motivos.append(f"hacinamiento severo (más de {UMBRAL_HACINAMIENTO_SEVERO:.0f} personas por cuarto)")
    return "; ".join(motivos) if motivos else "sin motivo de prioridad"
