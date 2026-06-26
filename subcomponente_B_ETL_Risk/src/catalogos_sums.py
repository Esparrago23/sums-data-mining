"""
catalogos_sums.py
=================
Fuente única de verdad de los valores de catálogo del SUMS.

IMPORTANTE: estos strings son COPIA EXACTA de `sums-API/database/seeder.sql`.
El endpoint `captura-completa` resuelve cada valor de texto contra su catálogo con
`findOrCreateCatalog` (búsqueda case-insensitive). Si un valor NO coincide exactamente
con uno ya sembrado, el endpoint CREA una entrada nueva → contamina el catálogo con
duplicados ("Concreto" vs "Concreto o cemento"). Por eso el generador sintético usa
únicamente los valores de este módulo.

También contiene la geografía real de Suchiapa / Tuxtla Gutiérrez y nombres/apellidos
comunes en Chiapas, para que las familias sintéticas sean verosímiles para la región.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Catálogos EXACTOS (seeder.sql) — no editar sin sincronizar con la BD
# ─────────────────────────────────────────────────────────────────────────────

# Material: catálogo único compartido por techo/paredes/piso. La RESTRICCIÓN por tipo
# (piso no puede ser Lámina; techo/paredes no pueden ser Tierra) la aplica el generador.
CAT_MATERIAL = ['Concreto o cemento', 'Madera', 'Lámina', 'Tierra', 'Otros (especifique)']
CAT_MATERIAL_TECHO_PAREDES = ['Concreto o cemento', 'Madera', 'Lámina']   # cédula: techo/paredes
CAT_MATERIAL_PISO = ['Concreto o cemento', 'Madera', 'Tierra']            # cédula: piso (sin lámina)
CAT_MANEJO_EXCRETAS = ['WC', 'Letrina', 'Al ras de suelo']
CAT_ESTADO_CIVIL = ['Soltero(a)', 'Casado(a)', 'Viudo(a)', 'Unión libre']
CAT_PARENTESCO = ['Madre', 'Padre', 'Hijo(a)', 'Abuelo(a)']
CAT_OCUPACION = ['Estudiante', 'Hogar', 'Desempleo']  # la cédula captura texto libre
CAT_TOXICOMANIA = ['Alcoholismo', 'Tabaquismo', 'Otras sustancias']
CAT_ENFERMEDAD_CRONICA = ['Obesidad', 'Hipertensión', 'Diabetes Mellitus tipo 2', 'Tosedor crónico']
# OJO: el rango de ingreso es por la columna `rango` (no `nombre`)
CAT_INGRESO_SALARIAL = ['Hasta un salario mínimo', '1 a 2', '2 a 3', '3 a 5', 'Mayor a 5', 'No recibe ingresos']
# Escolaridad oficial. 'NA' (sin escolaridad) NO es valor de catálogo: el endpoint lo
# omite (no crea fila persona_escolaridad). El generador puede emitir 'NA' para bebés.
CAT_ESCOLARIDAD = ['Preescolar', 'Primaria', 'Secundaria', 'Bachillerato', 'Licenciatura', 'Maestría', 'Doctorado']
# Lengua oficial: solo Español / Lengua indígena. El nombre va en lengua_indigena_especificar.
CAT_LENGUA = ['Español', 'Lengua indígena']
# "Otros animales" de la cédula (perros/gatos van en la tabla vivienda, no aquí).
CAT_ANIMAL = ['Aves de corral', 'Bovinos', 'Porcinos', 'Otros']
CAT_ATENCION_EMBARAZO = ['Sector Público', 'Sector Privado', 'Hogar']
CAT_FRECUENCIA_SERVICIO_SALUD = ['Mensual', 'Trimestral', 'Semestral', 'Anual']
# Esquema de vacunación oficial completo (17 entradas).
VACUNAS = [
    'BCG',
    'COVID-19',
    'DPT (Difteria, bordetella pertusis y tétanos)',
    'Hepatitis A',
    'Hepatitis B',
    'Hexavalente (DPaT+VPI+Hib+HepB)',
    'Influenza estacional',
    'Neumocócica conjugada (13 valente)',
    'Neumocócica Polisacárida (23 serotipos)',
    'Rotavirus (RV1)',
    'SR (Sarampión,rubeola)',
    'SRP Triple viral (Sarampión, rubeola y parotiditis)',
    'Td (Tétanos, difteria)',
    'Tdpa (Tétanos, difteria, tos ferina)',
    'VPH (Virus del Papiloma Humano)',
    'Varicela',
    'Otra',
]
CAT_DOSIS = ['Única', '1era', '2da', '3era', 'Refuerzo']

# ─────────────────────────────────────────────────────────────────────────────
# Mapas ordinales (para el feature engineering del modelo ML)
# El índice = severidad/nivel. Se usan tanto para generar como para el score.
# ─────────────────────────────────────────────────────────────────────────────

# 0 = sin ingreso ... 5 = mayor a 5 SM
INGRESO_ORDINAL = {
    'No recibe ingresos': 0,
    'Hasta un salario mínimo': 1,
    '1 a 2': 2,
    '2 a 3': 3,
    '3 a 5': 4,
    'Mayor a 5': 5,
}
# 0 = sin escolaridad (NA) ... 7 = doctorado
ESCOLARIDAD_ORDINAL = {
    'NA': 0,
    'Preescolar': 1,
    'Primaria': 2,
    'Secundaria': 3,
    'Bachillerato': 4,
    'Licenciatura': 5,
    'Maestría': 6,
    'Doctorado': 7,
}

# ─────────────────────────────────────────────────────────────────────────────
# Geografía real del área de cobertura (Suchiapa y Tuxtla Gutiérrez, Chiapas)
# ─────────────────────────────────────────────────────────────────────────────

COLONIAS_SUCHIAPA = [
    'Centro', 'San José', 'Pacú', 'San Jacinto', 'San Antonio',
    'San Roque', 'Las Palmas', 'Guadalupe', 'El Calvario', 'Linda Vista',
]
CALLES_SUCHIAPA = [
    'Calle Real', 'Calle Hidalgo', 'Calle Libertad', 'Avenida Emiliano Zapata',
    'Calle Morelos', 'Calle Juárez', '1ra Calle Norte', '2da Calle Norte',
    '3ra Calle Norte', 'Calle del Río', 'Avenida 20 de Noviembre',
    'Calle 5 de Mayo', 'Calle Allende', 'Calle Matamoros',
]
COLONIAS_TUXTLA = [
    'Centro', 'Las Granjas', 'Terán', 'Cuauhtémoc', 'El Bosque',
    'La Lomita', 'Colinas del Sur', 'Patria Nueva', 'Albania Baja', 'Penipak',
]
CALLES_TUXTLA = [
    'Avenida Central', 'Boulevard Ángel Albino Corzo', 'Calle 1 Norte',
    'Calle 2 Sur', 'Avenida de la Juventud', 'Calzada de las Etnias',
    '5a Avenida Sur', 'Libramiento Norte', 'Calle Central Oriente',
]

APELLIDOS_CHIAPAS = [
    'Jiménez', 'Pérez', 'López', 'Hernández', 'Cruz', 'González', 'Méndez',
    'Domínguez', 'Aguilar', 'Ruiz', 'Velázquez', 'Martínez', 'Gómez', 'García',
    'Torres', 'Gutiérrez', 'Sánchez', 'Morales', 'Castellanos', 'Zenteno',
    'Nucamendi', 'Chanona', 'Coutiño', 'Grajales', 'Penagos', 'Ozuna',
    'Vázquez', 'Ramírez', 'Flores', 'Espinosa', 'Moreno', 'Ovando',
]
NOMBRES_M = [
    'José', 'Juan', 'Miguel', 'Manuel', 'Pedro', 'Luis', 'Carlos', 'Roberto',
    'Francisco', 'Jesús', 'Antonio', 'Alberto', 'Marco', 'Sergio', 'Ramón',
    'Óscar', 'Rafael', 'Enrique', 'Javier', 'Eduardo',
]
NOMBRES_F = [
    'María', 'Guadalupe', 'Juana', 'Margarita', 'Rosa', 'Verónica', 'Patricia',
    'Leticia', 'Elena', 'Carmen', 'Adriana', 'Gabriela', 'Alejandra', 'Lucía',
    'Teresa', 'Sofía', 'Martha', 'Silvia', 'Claudia', 'Beatriz',
]

# Lenguas indígenas reales de Chiapas (van en lengua_indigena_especificar,
# con lengua = 'Otra' para no contaminar cat_lengua)
LENGUAS_INDIGENAS_CHIAPAS = ['Tzotzil', 'Tzeltal', 'Chol', 'Zoque', 'Tojolabal']

# Motivos de uso de servicios de salud (texto libre realista)
MOTIVOS_SALUD = [
    'Control de diabetes', 'Control de hipertensión', 'Consulta general',
    'Control prenatal', 'Vacunación', 'Enfermedad respiratoria',
    'Control de niño sano', 'Dolor crónico', 'Revisión de rutina',
]
OCUPACIONES_TEXTO = [
    'Campesino', 'Comerciante', 'Ama de casa', 'Albañil', 'Empleada doméstica',
    'Jornalero', 'Vendedor ambulante', 'Profesor', 'Chofer', 'Costurera',
    'Estudiante', 'Mecánico', 'Carpintero', 'Pescador',
]
