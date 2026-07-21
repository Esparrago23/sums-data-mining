"""
load_to_bd_sql.py — Carga directa a PostgreSQL (RESPALDO / alternativa)
=======================================================================
Alternativa a load_to_bd.py para cuando NO hay una API corriendo, pero SÍ se tiene
acceso directo a la base de datos `centro_medico_2026`. Inserta cada familia en una
transacción propia (commit por familia, rollback ante error → nunca deja familias a
medias).

Es un port fiel de CapturaCompletaCedulaUseCase (sums-API). Inserta en orden:
  nucleo_familiar → direccion → nucleo_direccion → vivienda → familia_animal →
  (por integrante) persona + tablas satélite + nucleo_persona →
  cedula → esquema_vacunacion + inmunizacion

Resuelve catálogos con find-or-create (case-insensitive) igual que el backend, con
caché en memoria para no repetir SELECTs.

USO:
  python load_to_bd_sql.py --dbname centro_medico_2026 --host localhost \
      --user postgres --password TU_PASS --limit 5

Requiere: psycopg2-binary  (pip install psycopg2-binary)

NOTA: para datos "reales" preferir load_to_bd.py (reutiliza la lógica del backend).
      Este script es útil para cargas locales rápidas y transaccionales.
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import RealDictCursor
except ImportError:
    sys.exit("Falta 'psycopg2'. Instala con: pip install psycopg2-binary")

logger = logging.getLogger("sums.load_to_bd_sql")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ─── Validación de fechas (items 8 y 11) ──────────────────────────────────────

MAX_EDAD_ANIOS = 120
_WS_RE = re.compile(r"\s+")


def _normalizar_texto(value: str) -> str:
    """Item 10: normalización agresiva para comparar catálogos (strip + lower +
    colapsar espacios internos), evitando duplicados como 'Concreto  o cemento'
    vs 'Concreto o cemento'."""
    return _WS_RE.sub(" ", value.strip()).lower()


def validar_fecha_iso(valor, campo: str, *, permitir_futura: bool = False,
                       max_anios: int = MAX_EDAD_ANIOS, min_fecha: "date | None" = None):
    """Valida que `valor` sea una fecha ISO-8601 (YYYY-MM-DD) plausible.

    - Rechaza formato inválido.
    - Rechaza fechas futuras (salvo permitir_futura=True).
    - Rechaza fechas de más de `max_anios` años de antigüedad (evita typos como
      año 1899 en vez de 1999).
    - Si se da `min_fecha`, exige que `valor` no sea anterior a ella (usado para
      coherencia fecha_nacimiento <= fecha_aplicacion/fecha_registro).

    Lanza ValueError con mensaje claro ante cualquier violación.
    """
    if valor is None or str(valor).strip() == "":
        raise ValueError(f"Campo '{campo}' es obligatorio y no puede estar vacío.")
    try:
        fecha = date.fromisoformat(str(valor).strip())
    except ValueError as e:
        raise ValueError(f"Campo '{campo}' = '{valor}' no es una fecha ISO-8601 válida (YYYY-MM-DD): {e}")

    hoy = date.today()
    if not permitir_futura and fecha > hoy:
        raise ValueError(f"Campo '{campo}' = '{fecha}' no puede ser una fecha futura (hoy={hoy}).")

    limite_antiguo = date(hoy.year - max_anios, hoy.month, hoy.day)
    if fecha < limite_antiguo:
        raise ValueError(
            f"Campo '{campo}' = '{fecha}' implica más de {max_anios} años de antigüedad; revise el dato."
        )

    if min_fecha is not None and fecha < min_fecha:
        raise ValueError(f"Campo '{campo}' = '{fecha}' no puede ser anterior a {min_fecha}.")

    return fecha


# ─── Resolución de catálogos (find-or-create con caché) ──────────────────────

_cache = {}


def find_or_create(cur, table, id_col, label_col, value):
    """Resuelve (o crea) la fila de catálogo para `value` en `table`.

    Item 9: los nombres de tabla/columna (constantes internas del código, NUNCA
    provistas por el usuario) se interpolan con psycopg2.sql.Identifier() en vez
    de f-strings, como defensa adicional en profundidad.

    Item 10: la comparación normaliza agresivamente (strip + lower + colapsar
    espacios) tanto el valor de entrada como el valor almacenado, para no crear
    duplicados por diferencias triviales de espaciado/capitalización. Si aun así
    no hay coincidencia, se loguea explícitamente que se crea una fila nueva.
    """
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == 'na':
        return None

    normalizado = _normalizar_texto(value)
    key = (table, normalizado)
    if key in _cache:
        return _cache[key]

    select_query = sql.SQL(
        "SELECT {id_col} AS id FROM {table} "
        "WHERE LOWER(REGEXP_REPLACE(TRIM({label_col}), '\\s+', ' ', 'g')) = %s LIMIT 1;"
    ).format(
        id_col=sql.Identifier(id_col),
        table=sql.Identifier(table),
        label_col=sql.Identifier(label_col),
    )
    cur.execute(select_query, (normalizado,))
    row = cur.fetchone()
    if row:
        _cache[key] = row['id']
        return row['id']

    logger.info(
        "find_or_create: valor '%s' (normalizado='%s') no coincide con ninguna fila "
        "existente en '%s.%s'; se creará una entrada NUEVA en el catálogo.",
        value, normalizado, table, label_col,
    )
    insert_query = sql.SQL(
        "INSERT INTO {table} ({label_col}) VALUES (%s) "
        "ON CONFLICT ({label_col}) DO UPDATE SET {label_col} = EXCLUDED.{label_col} "
        "RETURNING {id_col} AS id;"
    ).format(
        table=sql.Identifier(table),
        label_col=sql.Identifier(label_col),
        id_col=sql.Identifier(id_col),
    )
    cur.execute(insert_query, (value,))
    new_id = cur.fetchone()['id']
    _cache[key] = new_id
    return new_id


# ─── Helpers ─────────────────────────────────────────────────────────────────

def split_name(nombre):
    parts = [p for p in nombre.split() if p]
    if len(parts) == 1:
        return parts[0], None, 'SIN APELLIDO', None
    if len(parts) == 2:
        return parts[0], None, parts[1], None
    return parts[0], ' '.join(parts[1:-2]) or None, parts[-2], parts[-1]


def tamizaje_bool(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ('si', 'sí', 'true', '1'):
        return True
    if v in ('no', 'false', '0'):
        return False
    return None


# ─── Inserción de una familia (en la transacción abierta) ────────────────────

def insertar_familia(cur, payload):
    fam = payload['familia']
    viv = payload['vivienda']

    # Items 8 y 11: la fecha de registro de la cédula debe ser una fecha ISO
    # plausible; la coherencia contra fecha_nacimiento de cada integrante se
    # valida más abajo, integrante por integrante.
    fecha_registro_cedula = validar_fecha_iso(payload['fecha_registro'], 'fecha_registro')

    # 1) nucleo_familiar
    comentarios = f"Informante: {fam['informante_nombre']} | Rol: {fam['rol_informante']}"
    cur.execute(
        "INSERT INTO nucleo_familiar (fecha_registro, comentarios) "
        "VALUES (NOW(), %s) RETURNING id_nucleo_familiar AS id;",
        (comentarios,),
    )
    nucleo_id = cur.fetchone()['id']

    # 2) direccion
    cur.execute(
        "INSERT INTO direccion (calle, numero_exterior, colonia, localidad, manzana, vivienda_referencia) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id_direccion AS id;",
        (fam['calle'], fam['numero_exterior'], fam['colonia'],
         fam['localidad'], fam['manzana'], fam.get('vivienda_referencia')),
    )
    direccion_id = cur.fetchone()['id']

    # 3) nucleo_direccion
    cur.execute(
        "INSERT INTO nucleo_direccion (nucleo_familiar_id, direccion_id, fecha_asociacion) "
        "VALUES (%s, %s, CURRENT_DATE);",
        (nucleo_id, direccion_id),
    )

    # 4) vivienda
    techo_id = find_or_create(cur, 'cat_material', 'id_material', 'nombre', viv['techo'])
    paredes_id = find_or_create(cur, 'cat_material', 'id_material', 'nombre', viv['paredes'])
    piso_id = find_or_create(cur, 'cat_material', 'id_material', 'nombre', viv['piso'])
    excretas_id = find_or_create(cur, 'cat_manejo_excretas', 'id_manejo_excretas', 'nombre', viv['excretas'])
    cur.execute(
        """INSERT INTO vivienda (
               nucleo_familiar_id, direccion_id, numero_cuartos, numero_habitantes,
               agua_entubada, energia_electrica, cocina_ubicacion, cocina_con_lena,
               manejo_excretas_id, red_alcantarillado, fosa_septica,
               material_techo_id, material_paredes_id, material_piso_id,
               perros_gatos_dentro, mascotas_vacunas_corrientes, mascotas_esterilizadas
           ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           RETURNING id_vivienda AS id;""",
        (nucleo_id, direccion_id, viv['numero_cuartos'], viv['numero_habitantes'],
         viv['agua_entubada'], viv['energia_electrica'], viv['cocina_ubicacion'],
         viv['cocina_con_lena'], excretas_id, viv['red_alcantarillado'], viv['fosa_septica'],
         techo_id, paredes_id, piso_id, viv['perros_gatos_dentro'],
         viv['mascotas_vacunas_corrientes'], viv['mascotas_esterilizadas']),
    )

    # 5) familia_animal
    for animal in viv.get('otros_animales', []):
        animal_id = find_or_create(cur, 'cat_animal', 'id_animal', 'nombre', animal)
        if animal_id:
            cur.execute(
                "INSERT INTO familia_animal (nucleo_familiar_id, animal_id) VALUES (%s, %s);",
                (nucleo_id, animal_id),
            )

    # 6) integrantes
    persona_by_name = {}
    nacimientos = {}
    for integrante in payload['integrantes']:
        persona_id, fecha_nacimiento = insertar_integrante(
            cur, nucleo_id, integrante, fecha_referencia=fecha_registro_cedula,
        )
        clave = integrante['nombre'].strip().lower()
        persona_by_name[clave] = persona_id
        nacimientos[clave] = fecha_nacimiento

    # 7) cedula
    cur.execute(
        """INSERT INTO cedula (unidad_salud_id, entrevistador_id, levantamiento_id,
               nucleo_familiar_id, fecha_registro, estado, observaciones)
           VALUES (%s, %s, NULL, %s, %s, %s, %s) RETURNING id_cedula AS id;""",
        (payload['unidad_salud_id'], payload['entrevistador_id'], nucleo_id,
         fecha_registro_cedula, payload['estado'], payload.get('observaciones')),
    )
    cedula_id = cur.fetchone()['id']

    # 8) vacunación
    n_inm = 0
    vac = payload.get('vacunacion', {})
    if vac.get('se_aplico_vacuna'):
        for v in vac.get('vacunas', []):
            paciente_key = str(v.get('paciente', '')).strip().lower()
            persona_id = persona_by_name.get(paciente_key)
            if not persona_id:
                continue
            vacuna_id = find_or_create(cur, 'vacuna', 'id_vacuna', 'nombre', v['vacuna'])
            dosis_id = find_or_create(cur, 'cat_dosis', 'id_dosis', 'nombre', v.get('dosis'))
            # Items 8 y 11: la fecha de aplicación debe ser ISO válida, no futura,
            # y no puede ser anterior a la fecha de nacimiento del paciente.
            fecha_aplicacion = validar_fecha_iso(
                v.get('fecha_aplicacion'), 'fecha_aplicacion',
                min_fecha=nacimientos.get(paciente_key),
            )
            cur.execute(
                "INSERT INTO esquema_vacunacion (persona_id, unidad_salud_id, fecha_registro) "
                "VALUES (%s, %s, CURRENT_DATE) RETURNING id_esquema_vacunacion AS id;",
                (persona_id, payload['unidad_salud_id']),
            )
            esquema_id = cur.fetchone()['id']
            cur.execute(
                "INSERT INTO inmunizacion (esquema_vacunacion_id, cedula_id, vacuna_id, dosis_id, fecha_aplicacion) "
                "VALUES (%s, %s, %s, %s, %s);",
                (esquema_id, cedula_id, vacuna_id, dosis_id, fecha_aplicacion),
            )
            n_inm += 1

    return {'cedula_id': cedula_id, 'nucleo_familiar_id': nucleo_id, 'n_inmunizaciones': n_inm}


def insertar_integrante(cur, nucleo_id, integrante, fecha_referencia=None):
    # Items 8 y 11: fecha_nacimiento debe ser ISO válida, plausible (no futura,
    # no > 120 años), y no posterior a `fecha_referencia` (la fecha de registro
    # de la cédula) — evita nacer "después" de que la familia fue registrada.
    fecha_nacimiento = validar_fecha_iso(integrante['fecha_nacimiento'], 'fecha_nacimiento')
    if fecha_referencia is not None and fecha_referencia < fecha_nacimiento:
        raise ValueError(
            f"Coherencia de fechas: fecha_registro ({fecha_referencia}) es anterior a la "
            f"fecha_nacimiento de '{integrante['nombre']}' ({fecha_nacimiento})."
        )

    estado_civil_id = find_or_create(cur, 'cat_estado_civil', 'id_estado_civil', 'nombre', integrante.get('estado_civil'))
    lengua_id = find_or_create(cur, 'cat_lengua', 'id_lengua', 'nombre', integrante.get('lengua'))
    escolaridad_id = find_or_create(cur, 'cat_escolaridad', 'id_escolaridad', 'nombre', integrante.get('escolaridad'))
    ingreso_id = find_or_create(cur, 'cat_ingreso_salarial', 'id_ingreso_salarial', 'rango', integrante.get('ingreso'))
    parentesco_id = find_or_create(cur, 'cat_parentesco', 'id_parentesco', 'nombre', integrante.get('parentesco'))

    pn, sn, ap, am = split_name(integrante['nombre'])
    cur.execute(
        """INSERT INTO persona (primer_nombre, segundo_nombre, apellido_paterno, apellido_materno,
               fecha_nacimiento, sexo, estado_civil_id, alfabetizacion, fecha_registro)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s, NOW()) RETURNING id_persona AS id;""",
        (pn, sn, ap, am, fecha_nacimiento, integrante['sexo'],
         estado_civil_id, integrante.get('alfabetizacion')),
    )
    persona_id = cur.fetchone()['id']

    if lengua_id:
        cur.execute(
            "INSERT INTO persona_lengua (persona_id, lengua_id, lengua_indigena_especificar, es_principal) "
            "VALUES (%s, %s, %s, true);",
            (persona_id, lengua_id, integrante.get('lengua_indigena_especificar')),
        )
    if escolaridad_id:
        cur.execute(
            "INSERT INTO persona_escolaridad (persona_id, escolaridad_id, fecha_registro) "
            "VALUES (%s, %s, CURRENT_DATE);",
            (persona_id, escolaridad_id),
        )
    if integrante.get('ocupacion'):
        cur.execute(
            "INSERT INTO persona_ocupacion (persona_id, ocupacion_id, ocupacion_texto, fecha_registro) "
            "VALUES (%s, NULL, %s, CURRENT_DATE);",
            (persona_id, integrante['ocupacion']),
        )
    if ingreso_id:
        cur.execute(
            "INSERT INTO persona_ingreso (persona_id, ingreso_salarial_id, fecha_registro) "
            "VALUES (%s, %s, CURRENT_DATE);",
            (persona_id, ingreso_id),
        )

    # Alimentación (columnas NOT NULL → default 0 si falta)
    cur.execute(
        "INSERT INTO persona_alimentacion (persona_id, dias_proteina, dias_frutas_verduras, dias_cereales, fecha_registro) "
        "VALUES (%s, %s, %s, %s, CURRENT_DATE);",
        (persona_id, integrante.get('dias_proteina', 0), integrante.get('dias_frutas_verduras', 0),
         integrante.get('dias_cereales', 0)),
    )
    if 'higiene' in integrante:
        cur.execute(
            "INSERT INTO persona_higiene (persona_id, higiene_bano_bucodental_diaria, fecha_registro) "
            "VALUES (%s, %s, CURRENT_DATE);",
            (persona_id, bool(integrante['higiene'])),
        )
    if 'seguridad_social' in integrante:
        cur.execute(
            "INSERT INTO persona_seguridad_social (persona_id, cuenta_seguridad_social, fecha_registro) "
            "VALUES (%s, %s, CURRENT_DATE);",
            (persona_id, bool(integrante['seguridad_social'])),
        )
    if integrante.get('presenta_discapacidad') or integrante.get('tipo_discapacidad'):
        cur.execute(
            "INSERT INTO persona_discapacidad (persona_id, presenta_discapacidad, tipo_discapacidad) "
            "VALUES (%s, %s, %s);",
            (persona_id, bool(integrante.get('presenta_discapacidad')), integrante.get('tipo_discapacidad')),
        )

    for tox in integrante.get('toxicomanias', []):
        tox_id = find_or_create(cur, 'cat_toxicomania', 'id_toxicomania', 'nombre', tox)
        if tox_id:
            cur.execute(
                "INSERT INTO persona_toxicomania (persona_id, toxicomania_id) VALUES (%s, %s);",
                (persona_id, tox_id),
            )
    for cron in integrante.get('enfermedades_cronicas', []):
        cron_id = find_or_create(cur, 'cat_enfermedad_cronica', 'id_enfermedad_cronica', 'nombre', cron)
        if cron_id:
            cur.execute(
                "INSERT INTO persona_enfermedad_cronica (persona_id, enfermedad_cronica_id) VALUES (%s, %s);",
                (persona_id, cron_id),
            )

    # Salud preventiva
    embarazo_id = find_or_create(cur, 'cat_atencion_embarazo', 'id_atencion_embarazo', 'nombre', integrante.get('atencion_embarazo'))
    cervico = tamizaje_bool(integrante.get('tamizaje_cervico_uterino'))
    mama = tamizaje_bool(integrante.get('tamizaje_cancer_mama'))
    if embarazo_id or cervico is not None or mama is not None:
        cur.execute(
            """INSERT INTO persona_salud_preventiva (persona_id, atencion_embarazo_id,
                   tamizaje_cervico_uterino, fecha_tamizaje_cervico_uterino,
                   tamizaje_cancer_mama, fecha_tamizaje_cancer_mama, fecha_registro)
               VALUES (%s, %s, %s, %s, %s, %s, CURRENT_DATE);""",
            (persona_id, embarazo_id, cervico, integrante.get('fecha_tamizaje_cervico_uterino'),
             mama, integrante.get('fecha_tamizaje_cancer_mama')),
        )

    # Servicio de salud
    frec_id = find_or_create(cur, 'cat_frecuencia_servicio_salud', 'id_frecuencia_servicio_salud', 'nombre',
                             integrante.get('frecuencia_servicio_salud'))
    if frec_id or integrante.get('motivo_uso'):
        cur.execute(
            "INSERT INTO persona_servicio_salud (persona_id, frecuencia_servicio_salud_id, motivo_uso, fecha_registro) "
            "VALUES (%s, %s, %s, CURRENT_DATE);",
            (persona_id, frec_id, integrante.get('motivo_uso')),
        )

    # nucleo_persona (idx único nucleo+persona)
    cur.execute(
        "INSERT INTO nucleo_persona (nucleo_familiar_id, persona_id, parentesco_id, fecha_registro) "
        "VALUES (%s, %s, %s, NOW()) ON CONFLICT (nucleo_familiar_id, persona_id) DO NOTHING;",
        (nucleo_id, persona_id, parentesco_id),
    )
    return persona_id, fecha_nacimiento


def main():
    parser = argparse.ArgumentParser(description='Carga directa a PostgreSQL (respaldo)')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--host', type=str, default=os.getenv('DB_HOST', 'localhost'))
    parser.add_argument('--port', type=int, default=int(os.getenv('DB_PORT', '5432')))
    parser.add_argument('--user', type=str, default=os.getenv('DB_USER', 'postgres'))
    parser.add_argument('--password', type=str, default=os.getenv('DB_PASS', ''))
    parser.add_argument('--dbname', type=str, default=os.getenv('DB_NAME', 'centro_medico_2026'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--start', type=int, default=1)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    input_path = Path(args.input) if args.input else (base_dir / 'data' / 'families_full.json')
    if not input_path.exists():
        sys.exit(f"No existe {input_path}. Corre primero: python synthetic_generator.py")

    with open(input_path, 'r', encoding='utf-8') as f:
        familias = json.load(f)

    conn = psycopg2.connect(host=args.host, port=args.port, user=args.user,
                            password=args.password, dbname=args.dbname)
    print(f"Conectado a {args.dbname}@{args.host}:{args.port}")

    inicio = args.start - 1
    fin = len(familias) if args.limit is None else min(len(familias), inicio + args.limit)
    ok = err = 0

    for i in range(inicio, fin):
        idx = i + 1
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                insertar_familia(cur, familias[i])
            conn.commit()
            ok += 1
        except Exception as e:  # noqa: BLE001
            conn.rollback()
            err += 1
            print(f"  [{idx}] ERROR: {e}")
        if idx % 50 == 0:
            print(f"  [{idx}/{fin}] ok={ok} err={err}")

    conn.close()
    print('=' * 60)
    print(f"  Cargadas OK: {ok}   Errores: {err}")
    print('=' * 60)


if __name__ == '__main__':
    main()
