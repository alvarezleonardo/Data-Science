# Documentacion del caso Subtes

Desafio Profesional de la certificacion en Data Science. Caso elegido: **molinetes del subte de la
Ciudad de Buenos Aires**, con enfoque propio en el impacto del cambio de costumbres laborales
post-pandemia sobre el uso de la red y sobre la actividad gastronomica y hotelera de las zonas de
oficinas.

Enunciado del caso: [2.1 - Casos de Negocio - Subtes.md](<../2.1 - Casos de Negocio - Subtes.md>) ·
Consignas de las cuatro etapas: [../../../consignas/](../../../consignas/)

## Por donde empezar

| Documento | Que contiene |
|---|---|
| **[enfoque-analitico.md](enfoque-analitico.md)** | **Punto de entrada.** Pregunta de negocio, fuentes, limitaciones y plan de las cuatro etapas |
| [cobertura-entregables.md](cobertura-entregables.md) | Que esta hecho y que falta contra cada entregable de las consignas, y que decisiones quedan pendientes |
| [resultados-preliminares.md](resultados-preliminares.md) | Los numeros obtenidos hasta ahora y el estado de cada hipotesis |

## Diseno del analisis

| Documento | Que contiene |
|---|---|
| [diseno-analisis-flujo.md](diseno-analisis-flujo.md) | Metodologia: por que se comparan formas de perfil y no niveles, franjas horarias, clasificacion funcional de las 90 estaciones, y las seis hipotesis contrastables |
| [fuentes-externas.md](fuentes-externas.md) | Busqueda de fuentes oficiales de gastronomia y hoteleria: que se encontro, que se descarto y por que ninguna cubre todo |

## Calidad y preparacion del dato

| Documento | Que contiene |
|---|---|
| [problema-fechas.md](problema-fechas.md) | **El hallazgo tecnico central.** Tres anios tienen las fechas corrompidas de forma silenciosa; diagnostico, regla de correccion y validacion |
| [diccionario-estaciones.md](diccionario-estaciones.md) | Normalizacion de nombres de estacion: 12 alias verificados, encoding roto distinto en cada anio, y el padron de referencia |
| [serie-extendida-2022-2026.md](serie-extendida-2022-2026.md) | Ampliacion de la serie mas alla del material provisto, hasta junio de 2026, con los cambios de formato y de alcance de la red |

## Codigo

| Archivo | Que hace |
|---|---|
| [../scripts/etl_agregado_horario.py](../scripts/etl_agregado_horario.py) | ETL de los 13 CSV crudos al agregado horario. Resuelve los seis formatos, corrige las fechas y normaliza los nombres |

## Datos

Viven en `../datasets/` y **no se versionan**: los CSV crudos pesan unos 14 GB. La regla esta en el
`.gitignore` del repositorio.

| Archivo | Que es |
|---|---|
| `historico_2014.csv` .. `historico_2026.csv` | Molinetes crudos, 139.928.997 filas, tramos de 15 minutos |
| `agregado-horario.csv` | **Salida del ETL**: 6.740.017 filas, `fecha, anio, mes, dia_semana, hora, linea, estacion, pax` |
| `estaciones_de_subte.csv` (en `externos/`) | Padron oficial: 90 estaciones con coordenadas |
| `bocas-de-subte.csv` (en `externos/`) | 379 bocas de acceso, **con barrio y comuna** |
| `registro-historico-del-precio-del-boleto.csv` | Tarifa desde 1994, 305 registros. **Sin usar todavia** |
| `externos/idecba-*.xlsx` | Actividad de restaurantes y locales por comuna y rubro |

Para regenerar el agregado desde los crudos:

```bash
cd scripts
python3 etl_agregado_horario.py          # los 13 anios, unos 10 minutos
python3 etl_agregado_horario.py 2020      # un anio puntual
```

## Estado

La serie cubre **2014 a junio de 2026**: 13 anios, 139,9 millones de registros crudos,
3.023.825.954 pasajeros. El ETL esta validado contra la suma directa de la columna de pasajeros de
cada archivo de entrada, anio por anio.

Lo que falta, en orden: entorno con pandas, notebook de la Etapa 1 con sus visualizaciones, consultas
SQL y cierre de la Etapa 2, modelos de la Etapa 3 y red neuronal de la Etapa 4. El detalle esta en
[cobertura-entregables.md](cobertura-entregables.md).
