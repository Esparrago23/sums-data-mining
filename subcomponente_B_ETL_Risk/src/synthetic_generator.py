"""
synthetic_generator.py — FASE B-1
=================================
Genera familias sintéticas de microdiagnóstico familiar contextualizadas en
Suchiapa / Tuxtla Gutiérrez (Chiapas), coherentes con la BD real del SUMS.

DISEÑO (una sola fuente de verdad):
  Se generan familias ANIDADAS completas (vivienda + integrantes + vacunación).
  De cada familia se derivan DOS salidas, garantizando consistencia total:

    1. data/families_full.json   → lista de payloads listos para el endpoint
                                    POST /sums/cedulas/captura-completa
                                    (lo que se inserta en la BD real).
    2. data/synthetic_data.csv   → features planas agregadas por familia
                                    (lo que entrena el modelo de riesgo, Fase B-3/B-4).

  El `nivel_riesgo` se calcula con reglas determinísticas (compute_risk) a partir
  de los atributos reales de la familia → el label SIEMPRE es coherente con lo que
  está en la BD. (Ver limitación del "label circular" en el notebook B / plan.)

COHERENCIA:
  Cada familia tiene un factor latente de vulnerabilidad `vuln` ∈ [0,1].
  Todos los atributos se muestrean condicionados a `vuln` (piso de tierra, sin agua,
  ingreso bajo, etc. son más probables cuando vuln es alto), de modo que las
  correlaciones salud-vivienda-socioeconómicas son realistas.

USO:
  python synthetic_generator.py --n 4000 --seed 42 --suchiapa-ratio 0.8
  (sin dependencias externas: solo librería estándar de Python)
"""

import argparse
import csv
import json
import random
from pathlib import Path

from catalogos_sums import (
    CAT_MATERIAL, CAT_MANEJO_EXCRETAS, CAT_ESTADO_CIVIL, CAT_TOXICOMANIA,
    CAT_ENFERMEDAD_CRONICA, CAT_INGRESO_SALARIAL, CAT_ESCOLARIDAD,
    CAT_ATENCION_EMBARAZO, CAT_FRECUENCIA_SERVICIO_SALUD, VACUNAS, CAT_DOSIS,
    INGRESO_ORDINAL, ESCOLARIDAD_ORDINAL,
    COLONIAS_SUCHIAPA, CALLES_SUCHIAPA, COLONIAS_TUXTLA, CALLES_TUXTLA,
    APELLIDOS_CHIAPAS, NOMBRES_M, NOMBRES_F, LENGUAS_INDIGENAS_CHIAPAS,
    MOTIVOS_SALUD, OCUPACIONES_TEXTO,
)
from grupos_vulnerables import calcular_banderas

REFERENCE_YEAR = 2026  # ancla determinística para calcular fechas de nacimiento

# Vacunas típicas del esquema infantil (para menores)
VACUNAS_INFANTILES = [
    'BCG', 'Hepatitis B', 'Hexavalente (DPaT+VPI+Hib+HepB)',
    'Rotavirus (RV1)', 'Neumocócica conjugada (13 valente)', 'SR (Sarampión,rubeola)',
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def chance(rng, p):
    """True con probabilidad p."""
    return rng.random() < p


def nombre_completo(rng, sexo):
    nombres = NOMBRES_F if sexo == 'femenino' else NOMBRES_M
    nombre = rng.choice(nombres)
    # 25% lleva segundo nombre
    if chance(rng, 0.25):
        nombre += ' ' + rng.choice(nombres)
    paterno = rng.choice(APELLIDOS_CHIAPAS)
    materno = rng.choice(APELLIDOS_CHIAPAS)
    return f"{nombre} {paterno} {materno}"


def fecha_nacimiento(rng, edad):
    anio = REFERENCE_YEAR - edad
    mes = rng.randint(1, 12)
    dia = rng.randint(1, 28)
    return f"{anio:04d}-{mes:02d}-{dia:02d}"


def fecha_reciente(rng):
    mes = rng.randint(1, 6)
    dia = rng.randint(1, 28)
    return f"{REFERENCE_YEAR:04d}-{mes:02d}-{dia:02d}"


def escolaridad_por_edad(rng, edad, vuln):
    """Escolaridad coherente con la edad y la vulnerabilidad (valores oficiales de la
    cédula). 'NA' = sin escolaridad; el endpoint lo omite (no crea fila)."""
    if edad < 3:
        return 'NA'
    if edad < 6:
        return 'Preescolar'
    if edad < 12:
        return 'Primaria'
    if edad < 15:
        return 'Secundaria' if chance(rng, 0.6) else 'Primaria'
    if edad < 18:
        return rng.choice(['Secundaria', 'Bachillerato', 'Bachillerato'])
    # Adultos: techo educativo más bajo cuando vuln es alto
    if vuln > 0.66:
        opciones = ['NA', 'Primaria', 'Primaria', 'Secundaria']
    elif vuln > 0.33:
        opciones = ['Primaria', 'Secundaria', 'Secundaria', 'Bachillerato']
    else:
        opciones = ['Secundaria', 'Bachillerato', 'Licenciatura', 'Licenciatura', 'Maestría', 'Doctorado']
    return rng.choice(opciones)


def ingreso_por_vuln(rng, edad, vuln, trabaja):
    if not trabaja or edad < 15:
        return 'No recibe ingresos'
    if vuln > 0.66:
        return rng.choice(['No recibe ingresos', 'Hasta un salario mínimo', 'Hasta un salario mínimo', '1 a 2'])
    if vuln > 0.33:
        return rng.choice(['Hasta un salario mínimo', '1 a 2', '1 a 2', '2 a 3'])
    return rng.choice(['1 a 2', '2 a 3', '3 a 5', 'Mayor a 5'])


def dias_alimento(rng, vuln, base_alto):
    """Días/semana (0-7). base_alto = valor típico cuando NO hay vulnerabilidad."""
    centro = base_alto - vuln * 4.0  # a más vuln, menos días
    val = round(rng.gauss(centro, 1.2))
    return max(0, min(7, val))


# ─────────────────────────────────────────────────────────────────────────────
# Observaciones sintéticas por familia (columna real `cedula.observaciones`,
# VARCHAR(300) en la BD del SUMS). MISMO banco de frases que usan la app web
# y móvil -- no se modifican, la consistencia de vocabulario entre repos
# importa para el proyecto.
# ─────────────────────────────────────────────────────────────────────────────
LIMITE_OBSERVACIONES = 300  # cedula.observaciones VARCHAR(300) en sumsAPI

FRASES_OBSERVACIONES = {
    'seguimiento': [
        "Se requiere visita de seguimiento la próxima semana.",
        "Programar seguimiento en los próximos días.",
        "Pendiente visita de control en una semana.",
    ],
    'enfermedad_rara': [
        "Se observó un padecimiento inusual que requiere valoración médica.",
        "Presenta un cuadro clínico poco común, referir a especialista.",
        "Posible enfermedad rara, se sugiere estudio adicional.",
    ],
    'embarazo': [
        "Hay una integrante embarazada, dar seguimiento prenatal.",
        "Integrante en gestación, sin control prenatal reciente.",
        "Embarazo detectado, canalizar a control prenatal.",
    ],
    'vacunas_pendientes': [
        "Faltan dosis de vacunación, llevar cartilla la próxima visita.",
        "Esquema de vacunación incompleto, traer cartilla pendiente.",
        "Pendientes dosis de vacuna, verificar en siguiente visita.",
    ],
    'vivienda_mal_estado': [
        "Vivienda con materiales precarios, requiere atención.",
        "Vivienda en malas condiciones estructurales.",
        "Techo y paredes en mal estado, riesgo para la familia.",
    ],
    'mascota_sin_vacunar': [
        "Hay mascotas sin vacunación al corriente en el hogar.",
        "Mascota dentro de la vivienda sin esquema de vacunación.",
        "Perro/gato sin vacunas vigentes, riesgo zoonótico.",
    ],
}

# Materiales de vivienda considerados precarios. El catálogo real (ver
# catalogos_sums.CAT_MATERIAL_TECHO_PAREDES / CAT_MATERIAL_PISO) usa
# 'Concreto o cemento' | 'Madera' | 'Lámina' para techo/paredes y agrega
# 'Tierra' para piso; se incluye también 'Adobe' por robustez ante una futura
# ampliación del catálogo (no rompe nada si nunca aparece).
MATERIALES_PRECARIOS = {'Lámina', 'Madera', 'Tierra', 'Adobe'}

# Probabilidades de inclusión de cada frase (ver docstring de generar_observaciones).
P_ALTA = 0.85              # la bandera correspondiente SÍ aplica
P_RUIDO = 0.08              # "ruido" realista cuando la bandera NO aplica (~0.06-0.10)
P_SEGUIMIENTO = 0.25        # no depende de bandera: cualquier visita puede requerir seguimiento
P_ENFERMEDAD_RARA = 0.05    # no depende de bandera: eventos raros por definición


def _truncar_en_limite_oracion(texto, limite):
    """Trunca `texto` a lo más `limite` caracteres sin cortar a la mitad de
    una palabra: recorta en el último punto de oración dentro del límite; si
    no hay ninguno, recorta en el último espacio."""
    if len(texto) <= limite:
        return texto
    corte = texto.rfind('.', 0, limite)
    if corte != -1:
        return texto[:corte + 1]
    corte = texto.rfind(' ', 0, limite)
    if corte != -1:
        return texto[:corte].rstrip()
    return texto[:limite]


def generar_observaciones(idx, banderas, colonia, localidad, material_techo, material_paredes):
    """Genera el texto sintético de `observaciones` de UNA familia (columna
    real `cedula.observaciones`, VARCHAR(300)), correlacionado con sus
    banderas de grupos vulnerables (`grupos_vulnerables.calcular_banderas`) y
    los materiales de su vivienda.

    Usa el MISMO banco de frases por categoría que la app web/móvil (no se
    modifica -- consistencia entre repos). Cada categoría se incluye con
    probabilidad ALTA (~0.85) si su condición aplica, o con probabilidad BAJA
    (~0.06-0.10, ruido realista -- notas reales a veces mencionan algo aunque
    no haya bandera) si no aplica. 'seguimiento' (~0.25) y 'enfermedad_rara'
    (~0.05) no dependen de ninguna bandera.

    INDEPENDENCIA DEL RNG (crítico -- ver docstring del módulo/plan de la
    tarea): usa su PROPIA instancia de `random.Random`, sembrada
    determinísticamente a partir de `idx`, en vez del `rng` compartido que
    `generar_familia` pasa secuencialmente a través de las 4000 familias en
    `main()`. Si esta función consumiera números del `rng` compartido,
    desplazaría el stream de aleatoriedad para TODAS las familias siguientes
    (idx+1 en adelante), cambiando sus materiales/vacunas/etc. sin necesidad.
    Con un RNG propio por idx, regenerar el dataset solo cambia el texto de
    `observaciones` y nada más."""
    rng_obs = random.Random(idx * 7919 + 42)

    frases = [f"Visita domiciliaria en colonia {colonia}, {localidad}."]

    vivienda_precaria = (
        material_techo in MATERIALES_PRECARIOS or material_paredes in MATERIALES_PRECARIOS
    )

    # Orden ESTABLE (no cambia entre corridas): seguimiento, enfermedad_rara,
    # embarazo, vacunas_pendientes, vivienda_mal_estado, mascota_sin_vacunar.
    categorias = [
        ('seguimiento', P_SEGUIMIENTO),
        ('enfermedad_rara', P_ENFERMEDAD_RARA),
        ('embarazo', P_ALTA if banderas.get('tiene_embarazada') else P_RUIDO),
        ('vacunas_pendientes', P_ALTA if banderas.get('tiene_menor_5_sin_vacunas') else P_RUIDO),
        ('vivienda_mal_estado', P_ALTA if vivienda_precaria else P_RUIDO),
        ('mascota_sin_vacunar', P_ALTA if banderas.get('tiene_mascota_sin_vacunar') else P_RUIDO),
    ]

    for categoria, prob in categorias:
        if chance(rng_obs, prob):
            frases.append(rng_obs.choice(FRASES_OBSERVACIONES[categoria]))

    texto = ' '.join(frases)
    return _truncar_en_limite_oracion(texto, LIMITE_OBSERVACIONES)


# ─────────────────────────────────────────────────────────────────────────────
# Generación de una persona (integrante)
# ─────────────────────────────────────────────────────────────────────────────

def generar_integrante(rng, rol, sexo, edad, vuln):
    nombre = nombre_completo(rng, sexo)

    es_adulto = edad >= 18
    es_mayor = edad >= 60
    trabaja = es_adulto and rol != 'Hijo(a)' or (rol == 'Hijo(a)' and edad >= 18 and chance(rng, 0.5))

    escolaridad = escolaridad_por_edad(rng, edad, vuln)
    ingreso = ingreso_por_vuln(rng, edad, vuln, trabaja)

    # Estado civil coherente con la edad
    if edad < 18:
        estado_civil = 'Soltero(a)'
    elif rol in ('Madre', 'Padre'):
        estado_civil = rng.choice(['Casado(a)', 'Unión libre', 'Unión libre'])
    elif es_mayor:
        estado_civil = rng.choice(['Casado(a)', 'Viudo(a)', 'Viudo(a)'])
    else:
        estado_civil = rng.choice(['Soltero(a)', 'Casado(a)', 'Unión libre'])

    # Lengua: mayoría español; en alta vulnerabilidad sube el uso de lengua indígena.
    # Valor oficial = 'Lengua indígena'; el nombre real va en lengua_indigena_especificar.
    lengua = 'Español'
    lengua_especificar = None
    if chance(rng, 0.10 + vuln * 0.25):
        lengua = 'Lengua indígena'
        lengua_especificar = rng.choice(LENGUAS_INDIGENAS_CHIAPAS)

    alfabetizacion = not (escolaridad == 'NA' and chance(rng, 0.5 + vuln * 0.3))

    # Alimentación (días/semana)
    dias_proteina = dias_alimento(rng, vuln, base_alto=6)
    dias_frutas_verduras = dias_alimento(rng, vuln, base_alto=6)
    dias_cereales = dias_alimento(rng, vuln, base_alto=7)

    higiene = not chance(rng, vuln * 0.3)
    seguridad_social = chance(rng, 0.55 - vuln * 0.35)

    # Enfermedades crónicas: por edad + vulnerabilidad
    enfermedades = []
    if es_adulto:
        riesgo_cronica = vuln * 0.4 + (0.3 if es_mayor else 0.0)
        if chance(rng, riesgo_cronica):
            enfermedades.append('Hipertensión')
        if chance(rng, riesgo_cronica * 0.8):
            enfermedades.append('Diabetes Mellitus tipo 2')
        if chance(rng, 0.15 + vuln * 0.2):
            enfermedades.append('Obesidad')
        if chance(rng, vuln * 0.15):
            enfermedades.append('Tosedor crónico')
    enfermedades = list(dict.fromkeys(enfermedades))  # dedup, conserva orden

    # Toxicomanías: adultos
    toxicomanias = []
    if es_adulto:
        if chance(rng, vuln * 0.35 + (0.1 if sexo == 'masculino' else 0.02)):
            toxicomanias.append('Alcoholismo')
        if chance(rng, vuln * 0.3 + (0.08 if sexo == 'masculino' else 0.03)):
            toxicomanias.append('Tabaquismo')

    # Discapacidad: poco frecuente, sube con edad
    presenta_disc = chance(rng, 0.02 + (0.05 if es_mayor else 0.0))
    tipo_disc = None
    if presenta_disc:
        tipo_disc = rng.choice([
            'Motriz', 'Visual', 'Auditiva',
            'Motriz parcial en extremidad inferior', 'Intelectual',
        ])

    integrante = {
        'nombre': nombre,
        'sexo': sexo,
        'fecha_nacimiento': fecha_nacimiento(rng, edad),
        'edad': edad,  # informativo; el endpoint recalcula desde fecha_nacimiento
        'parentesco': rol,
        'estado_civil': estado_civil,
        'lengua': lengua,
        'escolaridad': escolaridad,
        'ingreso': ingreso,
        'alfabetizacion': alfabetizacion,
        'ocupacion': rng.choice(OCUPACIONES_TEXTO) if (trabaja or edad >= 6) else None,
        'dias_proteina': dias_proteina,
        'dias_frutas_verduras': dias_frutas_verduras,
        'dias_cereales': dias_cereales,
        'higiene': higiene,
        'seguridad_social': seguridad_social,
        'toxicomanias': toxicomanias,
        'enfermedades_cronicas': enfermedades,
        'presenta_discapacidad': presenta_disc,
        'tipo_discapacidad': tipo_disc,
        'frecuencia_servicio_salud': rng.choice(CAT_FRECUENCIA_SERVICIO_SALUD),
        'motivo_uso': rng.choice(MOTIVOS_SALUD),
    }
    if lengua_especificar:
        integrante['lengua_indigena_especificar'] = lengua_especificar

    # Salud preventiva en mujeres en edad reproductiva / tamizajes
    if sexo == 'femenino' and 18 <= edad <= 65:
        cervico = 'si' if chance(rng, 0.6 - vuln * 0.3) else 'no'
        integrante['tamizaje_cervico_uterino'] = cervico
        if cervico == 'si':  # si hubo tamizaje, registrar la fecha
            integrante['fecha_tamizaje_cervico_uterino'] = fecha_reciente(rng)
        if edad >= 40:
            mama = 'si' if chance(rng, 0.5 - vuln * 0.3) else 'no'
            integrante['tamizaje_cancer_mama'] = mama
            if mama == 'si':
                integrante['fecha_tamizaje_cancer_mama'] = fecha_reciente(rng)
        if 18 <= edad <= 40 and chance(rng, 0.15):
            integrante['atencion_embarazo'] = rng.choice(CAT_ATENCION_EMBARAZO)

    return integrante


# ─────────────────────────────────────────────────────────────────────────────
# Generación de una familia completa
# ─────────────────────────────────────────────────────────────────────────────

def generar_familia(rng, idx, suchiapa_ratio):
    vuln = rng.random()  # factor latente de vulnerabilidad [0,1]

    # ── Composición del núcleo ──────────────────────────────────────────────
    n_hijos = rng.choices([0, 1, 2, 3, 4, 5], weights=[8, 18, 28, 22, 14, 10])[0]
    hay_pareja = chance(rng, 0.75)
    hay_abuelo = chance(rng, 0.18)
    numero_habitantes = 1 + (1 if hay_pareja else 0) + n_hijos + (1 if hay_abuelo else 0)

    integrantes = []

    # Jefe de familia (informante)
    jefe_sexo = rng.choice(['femenino', 'masculino'])
    jefe_rol = 'Madre' if jefe_sexo == 'femenino' else 'Padre'
    jefe_edad = rng.randint(28, 55)
    jefe = generar_integrante(rng, jefe_rol, jefe_sexo, jefe_edad, vuln)
    integrantes.append(jefe)

    # Pareja
    if hay_pareja:
        pareja_sexo = 'masculino' if jefe_sexo == 'femenino' else 'femenino'
        pareja_rol = 'Padre' if pareja_sexo == 'masculino' else 'Madre'
        pareja_edad = max(20, jefe_edad + rng.randint(-5, 6))
        integrantes.append(generar_integrante(rng, pareja_rol, pareja_sexo, pareja_edad, vuln))

    # Hijos
    for _ in range(n_hijos):
        hijo_sexo = rng.choice(['femenino', 'masculino'])
        hijo_edad = rng.randint(0, min(25, jefe_edad - 16))
        integrantes.append(generar_integrante(rng, 'Hijo(a)', hijo_sexo, hijo_edad, vuln))

    # Abuelo/a
    if hay_abuelo:
        ab_sexo = rng.choice(['femenino', 'masculino'])
        integrantes.append(generar_integrante(rng, 'Abuelo(a)', ab_sexo, rng.randint(62, 85), vuln))

    numero_habitantes = len(integrantes)

    # ── Vivienda (condicionada a vuln) ──────────────────────────────────────
    material_piso = 'Tierra' if chance(rng, vuln * 0.75) else rng.choice(['Concreto o cemento', 'Concreto o cemento', 'Madera'])
    material_techo = (
        rng.choice(['Lámina', 'Madera']) if chance(rng, vuln * 0.7)
        else 'Concreto o cemento'
    )
    material_paredes = (
        rng.choice(['Madera', 'Lámina']) if chance(rng, vuln * 0.6)
        else 'Concreto o cemento'
    )
    agua_entubada = not chance(rng, vuln * 0.6)
    energia_electrica = not chance(rng, vuln * 0.35)
    cocina_con_lena = chance(rng, vuln * 0.7)
    cocina_ubicacion = 'dentro_del_dormitorio' if chance(rng, vuln * 0.45) else 'fuera_del_dormitorio'
    if vuln > 0.66:
        manejo_excretas = rng.choice(['Letrina', 'Al ras de suelo', 'Letrina'])
    elif vuln > 0.33:
        manejo_excretas = rng.choice(['WC', 'Letrina'])
    else:
        manejo_excretas = 'WC'
    red_alcantarillado = not chance(rng, vuln * 0.7)
    fosa_septica = (not red_alcantarillado) and chance(rng, 0.5)

    # Cuartos: menos cuartos con más vulnerabilidad → hacinamiento
    max_cuartos = max(1, round(4 - vuln * 2.5))
    numero_cuartos = rng.randint(1, max(1, max_cuartos))

    # Animales ("otros animales" de la cédula: aves, bovinos, porcinos)
    otros_animales = []
    if chance(rng, 0.4 + vuln * 0.3):
        for animal in ['Aves de corral', 'Porcinos', 'Bovinos']:
            if chance(rng, 0.3):
                otros_animales.append(animal)
    perros_gatos = chance(rng, 0.5)

    # ── Vacunación de menores ───────────────────────────────────────────────
    vacunas_aplicadas = []
    for integrante in integrantes:
        if integrante['edad'] < 6 and chance(rng, 0.7):
            vac = rng.choice(VACUNAS_INFANTILES)
            vacunas_aplicadas.append({
                'paciente': integrante['nombre'],
                'vacuna': vac,
                'dosis': rng.choice(CAT_DOSIS),
                'fecha_aplicacion': fecha_reciente(rng),
            })

    # ── Domicilio (Suchiapa / Tuxtla) ───────────────────────────────────────
    if chance(rng, suchiapa_ratio):
        localidad = 'Suchiapa'
        colonia = rng.choice(COLONIAS_SUCHIAPA)
        calle = rng.choice(CALLES_SUCHIAPA)
    else:
        localidad = 'Tuxtla Gutiérrez'
        colonia = rng.choice(COLONIAS_TUXTLA)
        calle = rng.choice(CALLES_TUXTLA)
    numero_ext = str(rng.randint(1, 280))
    manzana = f"Mz. {rng.randint(1, 35)}"

    # ── Valores que antes se calculaban INLINE dentro del dict `payload` ────
    # Se extraen a variables, en el MISMO orden exacto en que ya se evaluaban
    # (misma cantidad y secuencia de llamadas al `rng` COMPARTIDO -- ni una
    # de más ni de menos), para poder calcular `banderas`/`observaciones`
    # (que necesitan la vivienda YA completa) ANTES de construir `payload`
    # sin desplazar el stream de aleatoriedad de ninguna familia siguiente.
    fecha_registro = fecha_reciente(rng)
    vivienda_referencia = f'Casa {rng.choice(["azul", "verde", "blanca", "de tabique"])}'
    mascotas_vacunas_corrientes = perros_gatos and chance(rng, 0.6)
    mascotas_esterilizadas = perros_gatos and chance(rng, 0.3)

    vivienda_dict = {
        'techo': material_techo,
        'paredes': material_paredes,
        'piso': material_piso,
        'excretas': manejo_excretas,
        'numero_cuartos': numero_cuartos,
        'numero_habitantes': numero_habitantes,
        'agua_entubada': agua_entubada,
        'energia_electrica': energia_electrica,
        'cocina_ubicacion': cocina_ubicacion,
        'cocina_con_lena': cocina_con_lena,
        'red_alcantarillado': red_alcantarillado,
        'fosa_septica': fosa_septica,
        'perros_gatos_dentro': perros_gatos,
        'mascotas_vacunas_corrientes': mascotas_vacunas_corrientes,
        'mascotas_esterilizadas': mascotas_esterilizadas,
        'otros_animales': otros_animales,
    }

    # Banderas de grupos vulnerables (embarazada / menor de 1 año / menor de 5
    # sin vacunas / adulto mayor solo / mascota sin vacunar) -- calculadas
    # ANTES del payload para poder generar observaciones sintéticas
    # correlacionadas con ellas. NO consumen `rng` (calcular_banderas es
    # lógica determinística pura sobre datos ya generados), así que calcularlas
    # aquí no afecta el stream de aleatoriedad.
    banderas = calcular_banderas(integrantes, vacunas_aplicadas, vivienda=vivienda_dict)
    observaciones = generar_observaciones(
        idx, banderas, colonia, localidad, material_techo, material_paredes
    )

    # ── Payload para captura-completa (estructura que lee el endpoint) ──────
    payload = {
        'unidad_salud_id': 1,
        'entrevistador_id': 1,
        'estado': 'validada',
        'fecha_registro': fecha_registro,
        'observaciones': observaciones,
        'familia': {
            'informante_nombre': jefe['nombre'],
            'rol_informante': jefe_rol,
            'calle': calle,
            'numero_exterior': numero_ext,
            'colonia': colonia,
            'localidad': localidad,
            'manzana': manzana,
            'vivienda_referencia': vivienda_referencia,
        },
        'vivienda': vivienda_dict,
        'integrantes': integrantes,
        'vacunacion': {
            'se_aplico_vacuna': len(vacunas_aplicadas) > 0,
            'vacunas': vacunas_aplicadas,
        },
    }
    return payload, vuln


# ─────────────────────────────────────────────────────────────────────────────
# Derivación de features planas + score de riesgo (única fuente de verdad)
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk(flat):
    """Reglas determinísticas de salud pública → (score, nivel_riesgo).
    Mismo criterio que el plan de minería (Fase B-2).

    MEJORA — hacinamiento severo (ver grupos_vulnerables.py): antes esta regla
    sumaba +1 punto de 12 con un único umbral binario (>2.5 personas/cuarto),
    el mismo peso que cocinar con leña -- una familia con 2.6 personas/cuarto
    puntuaba IGUAL que una con 15. Ahora es una escala graduada de 0 a 3
    puntos, alineada con UMBRAL_HACINAMIENTO_SEVERO de grupos_vulnerables.py,
    y el score máximo posible sube de 12 a 14 (los umbrales de nivel se
    reajustaron para conservar las mismas proporciones ~50% ALTO / ~25% MEDIO)."""
    s = 0
    # Vivienda
    s += 1 if flat['material_piso'] == 'Tierra' else 0
    s += 1 if flat['material_techo'] in ('Lámina', 'Madera') else 0
    s += 1 if not flat['agua_entubada'] else 0
    s += 1 if not flat['energia_electrica'] else 0
    # Hacinamiento (escala graduada, máx. 3 puntos en vez de 1 binario)
    ppc = flat['personas_por_cuarto']
    if ppc > 5.0:
        s += 3
    elif ppc > 3.0:
        s += 2
    elif ppc > 2.0:
        s += 1
    # Excretas
    s += 1 if flat['manejo_excretas'] != 'WC' else 0
    # Cocina con leña
    s += 1 if flat['cocina_con_lena'] else 0
    # Salud crónica
    s += 1 if flat['count_enfermedades_cronicas'] >= 2 else 0
    # Vacunación
    s += 1 if not flat['vacunacion_completa'] else 0
    # Nutrición
    s += 1 if (flat['avg_dias_proteina'] < 3 or flat['avg_dias_frutas_verduras'] < 3) else 0
    # Socioeconómico
    s += 1 if flat['ingreso_nivel'] <= 1 else 0
    # Toxicomanías
    s += 1 if flat['count_toxicomanias'] > 0 else 0

    # Score máximo ahora es 14 (11 reglas binarias + hasta 3 de hacinamiento).
    if s >= 7:
        nivel = 'ALTO'
    elif s >= 4:
        nivel = 'MEDIO'
    else:
        nivel = 'BAJO'
    return s, nivel


def familia_to_flat(payload):
    """Agrega la familia anidada a una fila plana para el modelo ML."""
    viv = payload['vivienda']
    fam = payload['familia']
    integrantes = payload['integrantes']
    n = len(integrantes)

    count_cronicas = sum(len(i['enfermedades_cronicas']) for i in integrantes)
    count_toxico = sum(len(i['toxicomanias']) for i in integrantes)
    avg_prot = sum(i['dias_proteina'] for i in integrantes) / n
    avg_fv = sum(i['dias_frutas_verduras'] for i in integrantes) / n
    avg_cer = sum(i['dias_cereales'] for i in integrantes) / n

    # Ingreso del núcleo = máximo ingreso entre adultos (mejor proxy del hogar)
    ingresos = [INGRESO_ORDINAL[i['ingreso']] for i in integrantes]
    ingreso_nivel = max(ingresos) if ingresos else 0

    # Escolaridad promedio de adultos
    esc_adultos = [ESCOLARIDAD_ORDINAL[i['escolaridad']] for i in integrantes if i['edad'] >= 18]
    escolaridad_prom = round(sum(esc_adultos) / len(esc_adultos), 2) if esc_adultos else 0

    personas_por_cuarto = round(viv['numero_habitantes'] / max(1, viv['numero_cuartos']), 2)
    vacunacion_completa = payload['vacunacion']['se_aplico_vacuna']
    seguridad_social_jefe = integrantes[0]['seguridad_social']

    flat = {
        'material_techo': viv['techo'],
        'material_paredes': viv['paredes'],
        'material_piso': viv['piso'],
        'manejo_excretas': viv['excretas'],
        'numero_cuartos': viv['numero_cuartos'],
        'numero_habitantes': viv['numero_habitantes'],
        'personas_por_cuarto': personas_por_cuarto,
        'agua_entubada': viv['agua_entubada'],
        'energia_electrica': viv['energia_electrica'],
        'cocina_con_lena': viv['cocina_con_lena'],
        'cocina_ubicacion': viv['cocina_ubicacion'],
        'red_alcantarillado': viv['red_alcantarillado'],
        'fosa_septica': viv['fosa_septica'],
        'count_enfermedades_cronicas': count_cronicas,
        'count_toxicomanias': count_toxico,
        'vacunacion_completa': vacunacion_completa,
        'avg_dias_proteina': round(avg_prot, 2),
        'avg_dias_frutas_verduras': round(avg_fv, 2),
        'avg_dias_cereales': round(avg_cer, 2),
        'ingreso_nivel': ingreso_nivel,
        'escolaridad_promedio': escolaridad_prom,
        'total_integrantes': n,
        'seguridad_social_jefe': seguridad_social_jefe,
        'nombre_informante': fam['informante_nombre'],
        'domicilio': f"{fam['calle']} #{fam['numero_exterior']}, Col. {fam['colonia']}",
        'colonia': fam['colonia'],
        'localidad': fam['localidad'],
    }
    score, nivel = compute_risk(flat)
    flat['score_total'] = score
    flat['nivel_riesgo'] = nivel

    # Banderas de grupos vulnerables (embarazada, menor de 1 año, menor de 5
    # sin vacunas, adulto mayor solo): se calculan de los integrantes ANTES de
    # que se pierdan en la agregación de arriba. NO son features del modelo
    # (no entran a etl_pipeline.FEATURES) -- se usan en risk_report.py para
    # decidir prioridad de visita independiente del nivel de riesgo ML.
    vacunas_aplicadas = payload['vacunacion']['vacunas']
    flat.update(calcular_banderas(integrantes, vacunas_aplicadas, vivienda=payload['vivienda']))

    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generador de familias sintéticas SUMS (Fase B-1)')
    parser.add_argument('--n', type=int, default=4000, help='Número de familias a generar')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para reproducibilidad')
    parser.add_argument('--suchiapa-ratio', type=float, default=0.8,
                        help='Proporción de familias en Suchiapa (resto Tuxtla)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Carpeta de salida (default: ../data relativo a src)')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parent.parent / 'data')
    out_dir.mkdir(parents=True, exist_ok=True)

    families = []
    flats = []
    for idx in range(1, args.n + 1):
        payload, _vuln = generar_familia(rng, idx, args.suchiapa_ratio)
        flat = familia_to_flat(payload)
        families.append(payload)
        flats.append(flat)

    # Guardar JSON anidado (para la BD)
    json_path = out_dir / 'families_full.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(families, f, ensure_ascii=False, indent=2)

    # Guardar CSV plano (para el modelo ML)
    csv_path = out_dir / 'synthetic_data.csv'
    fieldnames = list(flats[0].keys())
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flats)

    # Resumen
    dist = {'ALTO': 0, 'MEDIO': 0, 'BAJO': 0}
    for fl in flats:
        dist[fl['nivel_riesgo']] += 1
    total = len(flats)
    n_integrantes = sum(len(p['integrantes']) for p in families)
    n_vacunas = sum(len(p['vacunacion']['vacunas']) for p in families)

    print('=' * 60)
    print(f'  Familias generadas: {total}')
    print(f'  Integrantes (personas): {n_integrantes}')
    print(f'  Vacunas aplicadas: {n_vacunas}')
    print('-' * 60)
    print('  Distribucion de riesgo:')
    for nivel in ('ALTO', 'MEDIO', 'BAJO'):
        pct = 100 * dist[nivel] / total
        print(f'    {nivel:5s}: {dist[nivel]:5d}  ({pct:5.1f}%)')
    print('-' * 60)
    print(f'  -> {json_path}')
    print(f'  -> {csv_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
