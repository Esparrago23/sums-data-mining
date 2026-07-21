# Estrategia para hojas horizontales de composición familiar

## Objetivo

Adaptar el pipeline de OCR para procesar las hojas horizontales de composición familiar, enfocadas en vivienda y uso de servicios de salud, sin romper el flujo actual de la cédula.

## Hipótesis de trabajo

Estas hojas son más adecuadas para un enfoque basado en:

- detección de página por estructura visual,
- normalización de orientación,
- extracción por ROI,
- y clasificación por catálogo para campos categóricos.

Esto es más robusto que intentar leer todo con OCR libre, especialmente cuando los campos son cortos y repetitivos.

## 1. Detección de página

### Nuevo tipo de página

Agregar un nuevo page kind para estas hojas, por ejemplo:

- `familia_horizontal`
- `servicios_salud_horizontal`

### Señales visuales para detectarlas

- aspecto ancho y relativamente bajo,
- caja del formulario más horizontal que vertical,
- presencia de múltiples columnas o bloques de texto,
- estructura tipo tabla o lista de miembros familiares.

## 2. Normalización de orientación

Como las hojas pueden venir en dos orientaciones, pero siempre horizontales:

1. probar la imagen en 0° y 180°,
2. estimar la caja del formulario en cada caso,
3. comparar con una plantilla de referencia basada en:
   - proporción del formulario,
   - posición de encabezados,
   - densidad de tinta en columnas,
   - alineación de bloques.
4. seleccionar la orientación con mejor score.

Esto evita que la extracción falle solo por un giro de página.

## 3. Extracción de campos

### Campos a extraer

- nombre o identificador del miembro familiar,
- parentesco,
- sexo,
- edad,
- estado civil,
- escolaridad,
- ocupación,
- servicio de salud / seguro,
- uso de servicios de salud,
- y otros campos categóricos.

### Estrategia por tipo de campo

- checkbox o sí/no: mantener el pipeline actual de checkbox.
- números: OCR ligero con preprocesamiento adaptado.
- texto corto categórico: matching por catálogo, no OCR libre.
- texto libre: usar OCR solo como respaldo, no como base principal.

## 4. Catálogo de opciones base

### Parentesco

- jefe(a)
- cónyuge
- hijo(a)
- padre/madre
- hermano(a)
- abuelo(a)
- nieto(a)
- suegro(a)
- otro

### Sexo

- hombre
- mujer

### Estado civil

- soltero(a)
- casado(a)
- unión libre
- viudo(a)
- divorciado(a)
- separado(a)

### Escolaridad

- sin escolaridad
- primaria
- primaria truncada
- secundaria
- secundaria truncada
- preparatoria
- preparatoria truncada
- bachillerato
- bachillerato truncado
- licenciatura
- licenciatura truncada

### Ocupación

- ama de casa
- estudiante
- trabajador(a)
- desempleado(a)
- jubilado(a)
- menor de edad
- empleado(a) privado(a)
- empleado(a) público(a)
- independiente
- otro

### Seguridad social

- sí
- no

### Frecuencia de servicios de salud

- anual
- mensual

## 5. Implementación recomendada

### Paso 1

Añadir un nuevo módulo de clasificación de páginas horizontales.

### Paso 2

Agregar una nueva configuración de campos para estas hojas.

### Paso 3

Implementar un extractor de catálogo para los campos categóricos.

### Paso 4

Validar con 5 a 10 muestras reales antes de ampliar.

## 6. Recomendación principal

La mejor estrategia para estas hojas no es forzar OCR libre sobre texto manuscrito, sino:

- detectar bien la página,
- normalizar la orientación,
- y usar catálogo + validación visual para los campos categóricos.

Eso tiene mejor rendimiento y más probabilidad de funcionar en hojas con layout repetitivo y relativamente limpio.
