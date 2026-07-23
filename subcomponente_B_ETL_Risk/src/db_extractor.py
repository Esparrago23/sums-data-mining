# -*- coding: utf-8 -*-
"""
db_extractor.py
================
Subcomponente B — Extracción de familias REALES desde la BD de sums-API
(`centro_medico_<año>`) para entrenar el modelo de riesgo con datos reales
en vez de sintéticos.

DISEÑO (misma idea que synthetic_generator.familia_to_flat -- una sola
fuente de verdad): se consulta la BD por núcleo familiar, se arma la misma
estructura que espera `calcular_banderas()` (integrantes + vacunas_aplicadas
+ vivienda) y se reutilizan esa función y `compute_risk()` TAL CUAL -- así
el label y las banderas se calculan EXACTAMENTE igual sobre datos reales que
sobre datos sintéticos, sin duplicar la lógica de negocio en dos lugares.

ALCANCE / PRIVACIDAD: solo se leen columnas de vivienda/salud/composición
familiar (las de etl_pipeline.FEATURES) -- ningún campo cifrado en reposo
(nombres de persona, calle, número, colonia; ver
sums-API/src/shared/security/sensitiveFields.ts) es feature del modelo
(anti-leakage, ver etl_pipeline.COLUMNAS_EXCLUIDAS), así que este extractor
no necesita desencriptar nada ni maneja PII identificable more allá de IDs
numéricos internos (que se descartan, no se guardan en el CSV de salida).

CONFIGURACIÓN (variables de entorno, con default de desarrollo local vía
Docker Desktop -- ajustar en producción según dónde viva ese Postgres):
  SUMS_DB_HOST   (default: host.docker.internal)
  SUMS_DB_PORT   (default: 5433 -- DB_EXTERNAL_PORT de sums-API/.env.example)
  SUMS_DB_USER   (default: postgres)
  SUMS_DB_PASS   (default: postgres)
  SUMS_DB_NAME   (default: centro_medico_<año actual> -- ver
                   sums-API/src/core/db_postgresql.ts, rota cada año)

CALIDAD DE DATOS: solo se incluyen núcleos con cédula NO 'borrador' (evita
entrenar con capturas incompletas a medio llenar) y con vivienda con
numero_cuartos/numero_habitantes capturados (sin eso no se puede calcular
hacinamiento, el feature más correlacionado con el riesgo). El resto de
nulos (alimentación, ingreso, escolaridad no capturados para algún
integrante) se deja como NaN -- `etl_pipeline.clean_and_transform` ya sabe
imputar (mediana/moda), no se duplica esa lógica aquí.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import psycopg2

from catalogos_sums import ESCOLARIDAD_ORDINAL, INGRESO_ORDINAL
from grupos_vulnerables import calcular_banderas

# compute_risk vive en synthetic_generator.py -- se importa tal cual, no se
# reimplementa (misma fórmula de riesgo para datos sintéticos y reales).
from synthetic_generator import compute_risk

_QUERY_NUCLEOS = """
    SELECT DISTINCT nf.id_nucleo_familiar, nf.jefe_persona_id,
           v.numero_cuartos, v.numero_habitantes,
           v.agua_entubada, v.energia_electrica, v.cocina_ubicacion,
           v.cocina_con_lena, v.red_alcantarillado, v.fosa_septica,
           v.perros_gatos_dentro, v.mascotas_vacunas_corrientes,
           mt.nombre AS material_techo, mp.nombre AS material_paredes,
           mpi.nombre AS material_piso, me.nombre AS manejo_excretas,
           d.colonia, d.localidad
    FROM nucleo_familiar nf
    JOIN cedula c ON c.nucleo_familiar_id = nf.id_nucleo_familiar AND c.estado <> 'borrador'
    JOIN vivienda v ON v.nucleo_familiar_id = nf.id_nucleo_familiar
    LEFT JOIN cat_material mt ON mt.id_material = v.material_techo_id
    LEFT JOIN cat_material mp ON mp.id_material = v.material_paredes_id
    LEFT JOIN cat_material mpi ON mpi.id_material = v.material_piso_id
    LEFT JOIN cat_manejo_excretas me ON me.id_manejo_excretas = v.manejo_excretas_id
    LEFT JOIN nucleo_direccion nd ON nd.nucleo_familiar_id = nf.id_nucleo_familiar
    LEFT JOIN direccion d ON d.id_direccion = nd.direccion_id
    WHERE v.numero_cuartos IS NOT NULL AND v.numero_cuartos > 0
      AND v.numero_habitantes IS NOT NULL
"""

_QUERY_INTEGRANTES = """
    SELECT np.nucleo_familiar_id, p.id_persona,
           DATE_PART('year', AGE(CURRENT_DATE, p.fecha_nacimiento)) AS edad,
           esc.nombre AS escolaridad,
           ing.rango AS ingreso,
           ss.cuenta_seguridad_social AS seguridad_social,
           (sp.atencion_embarazo_id IS NOT NULL) AS atencion_embarazo,
           al.dias_proteina, al.dias_frutas_verduras, al.dias_cereales,
           (SELECT COUNT(*) FROM persona_toxicomania pt WHERE pt.persona_id = p.id_persona) AS n_toxicomanias,
           (SELECT COUNT(*) FROM persona_enfermedad_cronica pec WHERE pec.persona_id = p.id_persona) AS n_cronicas
    FROM nucleo_persona np
    JOIN persona p ON p.id_persona = np.persona_id
    LEFT JOIN LATERAL (
        SELECT ce.nombre FROM persona_escolaridad pe
        JOIN cat_escolaridad ce ON ce.id_escolaridad = pe.escolaridad_id
        WHERE pe.persona_id = p.id_persona
        ORDER BY pe.fecha_registro DESC NULLS LAST LIMIT 1
    ) esc ON true
    LEFT JOIN LATERAL (
        SELECT cis.rango FROM persona_ingreso pi
        JOIN cat_ingreso_salarial cis ON cis.id_ingreso_salarial = pi.ingreso_salarial_id
        WHERE pi.persona_id = p.id_persona
        ORDER BY pi.fecha_registro DESC NULLS LAST LIMIT 1
    ) ing ON true
    LEFT JOIN LATERAL (
        SELECT pss.cuenta_seguridad_social FROM persona_seguridad_social pss
        WHERE pss.persona_id = p.id_persona
        ORDER BY pss.fecha_registro DESC NULLS LAST LIMIT 1
    ) ss ON true
    LEFT JOIN LATERAL (
        SELECT psp.atencion_embarazo_id FROM persona_salud_preventiva psp
        WHERE psp.persona_id = p.id_persona
        ORDER BY psp.fecha_registro DESC NULLS LAST LIMIT 1
    ) sp ON true
    LEFT JOIN LATERAL (
        SELECT pa.dias_proteina, pa.dias_frutas_verduras, pa.dias_cereales
        FROM persona_alimentacion pa
        WHERE pa.persona_id = p.id_persona
        ORDER BY pa.fecha_registro DESC NULLS LAST LIMIT 1
    ) al ON true
    WHERE np.nucleo_familiar_id = ANY(%(nucleo_ids)s) AND np.fecha_salida IS NULL
"""

# Personas (no núcleos) con al menos una dosis registrada -- se necesita a
# nivel persona, no solo a nivel núcleo, porque calcular_banderas.
# tiene_menor_5_sin_vacunas revisa POR INTEGRANTE si esa persona específica
# tiene vacuna, no si "alguien en la familia" la tiene.
_QUERY_PERSONAS_VACUNADAS = """
    SELECT DISTINCT np.nucleo_familiar_id, np.persona_id
    FROM nucleo_persona np
    JOIN esquema_vacunacion ev ON ev.persona_id = np.persona_id
    JOIN inmunizacion i ON i.esquema_vacunacion_id = ev.id_esquema_vacunacion
    WHERE np.nucleo_familiar_id = ANY(%(nucleo_ids)s)
"""


def _env(nombre: str, default: str) -> str:
    """os.environ.get(nombre, default) no basta: docker-compose inyecta las
    variables ${VAR:-default} SIEMPRE (aunque el .env no las defina, quedan
    en el contenedor como cadena vacía), así que .get() nunca cae al default
    de Python -- se necesita el fallback explícito sobre valor falsy."""
    return os.environ.get(nombre) or default


def _conn():
    return psycopg2.connect(
        host=_env("SUMS_DB_HOST", "host.docker.internal"),
        port=int(_env("SUMS_DB_PORT", "5433")),
        user=_env("SUMS_DB_USER", "postgres"),
        password=_env("SUMS_DB_PASS", "postgres"),
        dbname=_env("SUMS_DB_NAME", f"centro_medico_{date.today().year}"),
        connect_timeout=10,
    )


def _persona_a_integrante(fila: pd.Series) -> dict:
    """Traduce una fila de la query de integrantes a la forma que espera
    calcular_banderas: dict con 'nombre' (aquí, el id_persona como string --
    solo se usa para cruzar contra vacunas_aplicadas, nunca se guarda en la
    salida), 'edad' y 'atencion_embarazo'."""
    return {
        "nombre": str(fila["id_persona"]),
        "edad": int(fila["edad"]) if pd.notna(fila["edad"]) else 99,
        "atencion_embarazo": bool(fila["atencion_embarazo"]) if pd.notna(fila["atencion_embarazo"]) else False,
    }


def extract_from_db() -> pd.DataFrame:
    """
    Extrae familias reales de `centro_medico_<año>`, las agrega por núcleo
    familiar con el mismo criterio que `synthetic_generator.familia_to_flat`
    y les aplica `calcular_banderas` + `compute_risk` (mismas funciones que
    usan los datos sintéticos).

    Devuelve un DataFrame con las mismas columnas que `synthetic_data.csv`
    (features + score_total + nivel_riesgo + banderas + identificadores),
    listo para pasar por `etl_pipeline.clean_and_transform` / `build_xy` sin
    ningún cambio adicional.
    """
    conn = _conn()
    try:
        nucleos = pd.read_sql(_QUERY_NUCLEOS, conn)
        if nucleos.empty:
            return pd.DataFrame()

        nucleo_ids = nucleos["id_nucleo_familiar"].tolist()
        integrantes = pd.read_sql(_QUERY_INTEGRANTES, conn, params={"nucleo_ids": nucleo_ids})
        vacunados = pd.read_sql(_QUERY_PERSONAS_VACUNADAS, conn, params={"nucleo_ids": nucleo_ids})
    finally:
        conn.close()

    # persona_id (str) -> set de núcleos donde esa persona tiene vacuna
    # aplicada -- en la práctica cada persona pertenece a un solo núcleo,
    # pero se agrupa por núcleo para construir vacunas_aplicadas por familia.
    vacunados_por_nucleo: dict = {}
    for _, fila in vacunados.iterrows():
        vacunados_por_nucleo.setdefault(fila["nucleo_familiar_id"], set()).add(str(fila["persona_id"]))

    filas_salida = []

    for _, nuc in nucleos.iterrows():
        nid = nuc["id_nucleo_familiar"]
        integ = integrantes[integrantes["nucleo_familiar_id"] == nid]
        if integ.empty:
            continue  # sin integrantes vigentes registrados -- no se puede evaluar

        n = len(integ)
        count_cronicas = int(integ["n_cronicas"].fillna(0).sum())
        count_toxico = int(integ["n_toxicomanias"].fillna(0).sum())
        avg_prot = float(integ["dias_proteina"].mean()) if integ["dias_proteina"].notna().any() else None
        avg_fv = float(integ["dias_frutas_verduras"].mean()) if integ["dias_frutas_verduras"].notna().any() else None
        avg_cer = float(integ["dias_cereales"].mean()) if integ["dias_cereales"].notna().any() else None

        # Ingreso del núcleo = máximo ingreso entre los integrantes con dato
        # capturado (igual criterio que familia_to_flat). Si nadie tiene
        # ingreso capturado, se deja en None (NaN) -- clean_and_transform lo
        # imputa con la mediana; NO se asume "sin ingresos" por falta de dato.
        ingresos = [INGRESO_ORDINAL[v] for v in integ["ingreso"].dropna() if v in INGRESO_ORDINAL]
        ingreso_nivel = max(ingresos) if ingresos else None

        adultos = integ[integ["edad"] >= 18]
        esc_adultos = [ESCOLARIDAD_ORDINAL[v] for v in adultos["escolaridad"].dropna() if v in ESCOLARIDAD_ORDINAL]
        escolaridad_prom = round(sum(esc_adultos) / len(esc_adultos), 2) if esc_adultos else None

        fila_jefe = integ[integ["id_persona"] == nuc["jefe_persona_id"]]
        if not fila_jefe.empty and pd.notna(fila_jefe.iloc[0]["seguridad_social"]):
            seguridad_social_jefe = bool(fila_jefe.iloc[0]["seguridad_social"])
        else:
            seguridad_social_jefe = False

        personas_por_cuarto = round(nuc["numero_habitantes"] / max(1, nuc["numero_cuartos"]), 2)

        flat = {
            "material_techo": nuc["material_techo"],
            "material_paredes": nuc["material_paredes"],
            "material_piso": nuc["material_piso"],
            "manejo_excretas": nuc["manejo_excretas"],
            "numero_cuartos": nuc["numero_cuartos"],
            "numero_habitantes": nuc["numero_habitantes"],
            "personas_por_cuarto": personas_por_cuarto,
            "agua_entubada": bool(nuc["agua_entubada"]) if pd.notna(nuc["agua_entubada"]) else None,
            "energia_electrica": bool(nuc["energia_electrica"]) if pd.notna(nuc["energia_electrica"]) else None,
            "cocina_con_lena": bool(nuc["cocina_con_lena"]) if pd.notna(nuc["cocina_con_lena"]) else None,
            "cocina_ubicacion": nuc["cocina_ubicacion"],
            "red_alcantarillado": bool(nuc["red_alcantarillado"]) if pd.notna(nuc["red_alcantarillado"]) else None,
            "fosa_septica": bool(nuc["fosa_septica"]) if pd.notna(nuc["fosa_septica"]) else None,
            "count_enfermedades_cronicas": count_cronicas,
            "count_toxicomanias": count_toxico,
            "avg_dias_proteina": avg_prot,
            "avg_dias_frutas_verduras": avg_fv,
            "avg_dias_cereales": avg_cer,
            "ingreso_nivel": ingreso_nivel,
            "escolaridad_promedio": escolaridad_prom,
            "total_integrantes": n,
            "seguridad_social_jefe": seguridad_social_jefe,
            # Identificadores de texto -- se excluyen del modelo (anti-leakage,
            # ver etl_pipeline.COLUMNAS_EXCLUIDAS). nombre_informante/domicilio
            # requerirían desencriptar (persona/direccion tienen campos
            # cifrados) y no aportan nada al entrenamiento -- se dejan vacíos.
            "nombre_informante": "",
            "domicilio": "",
            "colonia": nuc["colonia"] or "",
            "localidad": nuc["localidad"] or "",
        }

        vivienda_para_banderas = {
            "perros_gatos_dentro": nuc["perros_gatos_dentro"],
            "mascotas_vacunas_corrientes": nuc["mascotas_vacunas_corrientes"],
            "numero_cuartos": nuc["numero_cuartos"],
            "numero_habitantes": nuc["numero_habitantes"],
        }
        integrantes_para_banderas = [_persona_a_integrante(r) for _, r in integ.iterrows()]
        vacunas_del_nucleo = vacunados_por_nucleo.get(nid, set())
        vacunas_aplicadas = [{"paciente": pid} for pid in vacunas_del_nucleo]

        banderas = calcular_banderas(
            integrantes_para_banderas, vacunas_aplicadas=vacunas_aplicadas, vivienda=vivienda_para_banderas
        )
        flat.update(banderas)
        # vacunacion_completa: mismo criterio que synthetic_generator
        # (payload['vacunacion']['se_aplico_vacuna'] = len(vacunas_aplicadas) > 0)
        # -- "alguien en la familia tiene al menos una dosis registrada", no
        # literalmente "esquema completo" pese al nombre de la columna.
        flat["vacunacion_completa"] = len(vacunas_aplicadas) > 0

        score, nivel = compute_risk(flat)
        flat["score_total"] = score
        flat["nivel_riesgo"] = nivel

        filas_salida.append(flat)

    return pd.DataFrame(filas_salida)


if __name__ == "__main__":
    # Smoke test manual: conecta y muestra un resumen, sin entrenar nada.
    df = extract_from_db()
    print(f"Núcleos familiares extraídos: {len(df)}")
    if not df.empty:
        print("\nDistribución de riesgo (aplicando compute_risk a datos reales):")
        print(df["nivel_riesgo"].value_counts())
        print("\nNulos por columna (candidatos a imputación en clean_and_transform):")
        print(df.isnull().sum()[df.isnull().sum() > 0])
