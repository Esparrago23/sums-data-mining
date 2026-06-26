# Verificación: ¿los datos sintéticos encajan en la BD real?

Este documento prueba, campo por campo, que cada dato generado por
`synthetic_generator.py` tiene destino en la BD real (`centro_medico_2026`) a través
del endpoint `POST /sums/cedulas/captura-completa`.

> **Fuente de verdad consultada:** `sums-API/database/schema.sql`,
> `sums-API/database/seeder.sql` y `sums-API/src/Cedula/application/capturaCompletaCedula_UseCase.ts`.

---

## 1. Estrategia de carga

El generador produce familias **anidadas** con la MISMA forma que el endpoint
`captura-completa` espera. Ese endpoint ya descompone el payload en las ~20 tablas
involucradas y resuelve los catálogos solo. No reinventamos el mapeo: lo reutilizamos.

```
families_full.json  ──POST──►  /sums/cedulas/captura-completa  ──►  55 tablas
   (1 objeto/familia)            (CapturaCompletaCedulaUseCase)        PostgreSQL
```

---

## 2. Mapeo del payload → tablas

### `familia` → `nucleo_familiar` + `direccion` + `nucleo_direccion`
| Campo generado | Destino | Nota |
|---|---|---|
| `informante_nombre`, `rol_informante` | `nucleo_familiar.comentarios` | el endpoint los concatena |
| `calle`, `numero_exterior`, `colonia`, `localidad`, `manzana`, `vivienda_referencia` | `direccion.*` | crea fila y la liga vía `nucleo_direccion` |

### `vivienda` → `vivienda` (+ `familia_animal`)
| Campo generado | Columna BD | Resolución |
|---|---|---|
| `techo` / `paredes` / `piso` | `material_techo_id` / `_paredes_id` / `_piso_id` | FK → `cat_material` (find-or-create) |
| `excretas` | `manejo_excretas_id` | FK → `cat_manejo_excretas` |
| `numero_cuartos`, `numero_habitantes` | iguales | INT |
| `agua_entubada`, `energia_electrica`, `cocina_con_lena`, `red_alcantarillado`, `fosa_septica` | iguales | BOOLEAN |
| `cocina_ubicacion` | `cocina_ubicacion` | enum `fuera_del_dormitorio` / `dentro_del_dormitorio` ✔ |
| `perros_gatos_dentro`, `mascotas_vacunas_corrientes`, `mascotas_esterilizadas` | iguales | BOOLEAN |
| `otros_animales[]` | `familia_animal.animal_id` | FK → `cat_animal` |

### `integrantes[]` → `persona` + 11 tablas satélite
| Campo generado | Destino |
|---|---|
| `nombre` | `persona.primer_nombre/segundo_nombre/apellido_paterno/apellido_materno` (split) |
| `sexo` | `persona.sexo` (enum `masculino`/`femenino` ✔) |
| `fecha_nacimiento` | `persona.fecha_nacimiento` (la edad se calcula, no se guarda) |
| `estado_civil` | `persona.estado_civil_id` → `cat_estado_civil` |
| `alfabetizacion` | `persona.alfabetizacion` |
| `lengua` (+ `lengua_indigena_especificar`) | `persona_lengua` → `cat_lengua` |
| `escolaridad` | `persona_escolaridad` → `cat_escolaridad` |
| `ocupacion` | `persona_ocupacion.ocupacion_texto` (texto libre, `ocupacion_id` NULL) |
| `ingreso` | `persona_ingreso` → `cat_ingreso_salarial.rango` |
| `dias_proteina` / `dias_frutas_verduras` / `dias_cereales` | `persona_alimentacion.*` (INT NOT NULL) |
| `higiene` | `persona_higiene.higiene_bano_bucodental_diaria` |
| `seguridad_social` | `persona_seguridad_social.cuenta_seguridad_social` |
| `toxicomanias[]` | `persona_toxicomania` → `cat_toxicomania` |
| `enfermedades_cronicas[]` | `persona_enfermedad_cronica` → `cat_enfermedad_cronica` |
| `presenta_discapacidad`, `tipo_discapacidad` | `persona_discapacidad` |
| `atencion_embarazo`, `tamizaje_*` | `persona_salud_preventiva` → `cat_atencion_embarazo` |
| `frecuencia_servicio_salud`, `motivo_uso` | `persona_servicio_salud` → `cat_frecuencia_servicio_salud` |
| (relación) | `nucleo_persona` (parentesco) — índice único `(nucleo_familiar_id, persona_id)` ✔ |

### nivel cédula → `cedula` + `esquema_vacunacion` + `inmunizacion`
| Campo generado | Destino |
|---|---|
| `unidad_salud_id=1`, `entrevistador_id=1` | `cedula.*` (ambos existen por el seeder ✔) |
| `estado='validada'` | `cedula.estado` (enum ✔) |
| `fecha_registro` | `cedula.fecha_registro` |
| `vacunacion.vacunas[]` | `inmunizacion` (+ `esquema_vacunacion`) → `vacuna`, `cat_dosis` |
| `vacuna.paciente` | se empata por nombre con la `persona` ya creada ✔ |

---

## 3. Valores categóricos — alineados a la CÉDULA OFICIAL

Los catálogos se alinearon a la **Cédula de Microdiagnóstico Familiar oficial** (jun-2026).
Se actualizó `sums-API/database/seeder.sql` Y `catalogos_sums.py` con los MISMOS valores,
de modo que el generador emite exactamente lo que la BD tiene sembrado (cero duplicados).

| Catálogo | Valores oficiales (seeder.sql == catalogos_sums.py) | ¿Cambió vs seeder viejo? |
|---|---|---|
| `cat_material` (techo/paredes) | `Concreto o cemento`, `Madera`, `Lámina` | = (piso usa `Tierra` en vez de `Lámina`) |
| `cat_material` (piso) | `Concreto o cemento`, `Madera`, `Tierra` | = |
| `cat_manejo_excretas` | `WC`, `Letrina`, `Al ras de suelo` | = |
| `cat_estado_civil` | `Soltero(a)`, `Casado(a)`, `Viudo(a)`, `Unión libre` | = |
| `cat_parentesco` | `Madre`, `Padre`, `Hijo(a)`, `Abuelo(a)` | = |
| `cat_ingreso_salarial` (`rango`) | `No recibe ingresos`, `Hasta un salario mínimo`, `1 a 2`, `2 a 3`, `3 a 5`, `Mayor a 5` | = |
| `cat_enfermedad_cronica` | `Obesidad`, `Hipertensión`, `Diabetes Mellitus tipo 2`, `Tosedor crónico` | quitado `Ninguna` |
| `cat_toxicomania` | `Alcoholismo`, `Tabaquismo`, `Otras sustancias` | **CAMBIÓ** (`Otra`→`Otras sustancias`) |
| `cat_escolaridad` | `Preescolar`, `Primaria`, `Secundaria`, `Bachillerato`, `Licenciatura`, `Maestría`, `Doctorado` | **CAMBIÓ** |
| `cat_lengua` | `Español`, `Lengua indígena` (nombre en `lengua_indigena_especificar`) | **CAMBIÓ** |
| `cat_animal` | `Aves de corral`, `Bovinos`, `Porcinos`, `Otros` | **CAMBIÓ** (`Cerdos`→`Porcinos`) |
| `cat_atencion_embarazo` | `Sector Público`, `Sector Privado`, `Hogar` | **CAMBIÓ** |
| `cat_frecuencia_servicio_salud` | `Mensual`, `Trimestral`, `Semestral`, `Anual` | **CAMBIÓ** |
| `vacuna` | 17 vacunas oficiales completas | **CAMBIÓ** (se agregaron 7) |
| `cat_dosis` | `Única`, `1era`, `2da`, `3era`, `Refuerzo` | = |

> **Centinela `NA`**: para escolaridad/toxicomanías/enfermedades/atención-embarazo, cuando
> el campo es "NA" (no aplica) el generador NO emite valor de catálogo; el endpoint y el
> loader SQL omiten la fila (igual que la cédula en papel). Por eso `NA` no se siembra.

> **Re-seed**: como `seeder.sql` usa `ON CONFLICT DO NOTHING`, si la BD ya tenía los valores
> viejos hay que limpiar las tablas `cat_*` afectadas antes de re-sembrar (ver nota al inicio
> de `seeder.sql`).

---

## 4. Diferencias de diseño respecto al plan original

1. **No se usa SDV para poblar la BD.** SDV escala tablas *planas*; aquí necesitamos
   familias *anidadas* con personas individuales coherentes. Se generan por reglas
   (procedural) condicionadas a un factor de vulnerabilidad latente → correlaciones
   realistas y control total. SDV queda como opción para variar el CSV de ML.
2. **Una sola fuente de verdad.** El CSV de ML se *deriva* de las mismas familias que
   se insertan en la BD → lo que entrena el modelo es exactamente lo que está en la BD.
3. **Doble propósito cubierto:** poblar la BD (mientras IMSS entrega datos reales) y
   entrenar el modelo de riesgo.
