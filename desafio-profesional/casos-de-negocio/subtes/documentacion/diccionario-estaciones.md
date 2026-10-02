# Diccionario de normalizacion de estaciones

> Insumo de la **Etapa 2** del caso subtes. Sin esta normalizacion no existe serie historica
> comparable entre 2014 y hoy: el mismo anden aparece con nombres distintos segun el anio.
> Enfoque general del caso: [enfoque-analitico.md](enfoque-analitico.md).

## 1. Por que hace falta

Tres problemas se acumulan sobre la columna de estacion:

1. **Alias**: la misma estacion se nombra de dos formas (`FLORES` y `SAN JOSE DE FLORES`).
2. **Encoding roto**, distinto en cada anio (seccion 5.2 del enfoque): `Agüero` llega en hasta ocho
   grafias, y en 2020 con caracter de reemplazo, es decir con la informacion ya perdida.
3. **Mayusculas, minusculas y espacios finales** heterogeneos.

Un `groupby` sin resolver esto no agrupa estaciones: agrupa grafias.

## 2. Padron de referencia

Se toma `estaciones-accesibles.csv` como padron, porque es el unico archivo del caso que trae
**coordenadas** y porque sus nombres estan en UTF-8 correcto. Tiene dos salvedades verificadas:

- **93 filas, pero solo 85 estaciones unicas.** Ocho estaciones aparecen dos veces porque tienen dos
  grupos de accesos con coordenadas distintas, separadas unos 250 m: `CALLAO` (B), `PUEYRREDON` (B),
  `INDEPENDENCIA` (C), `RETIRO` (C), `CALLAO` (D), `PUEYRREDON` (D), `INDEPENDENCIA` (E) y
  `RETIRO` (E). Para el padron se deduplica por `linea + estacion`.
- **No es el padron completo de la red.** Al listar solo estaciones *accesibles*, omite cinco que
  operan y que si aparecen en los historicos (seccion 4).

## 3. Alias verificados

Obtenidos cruzando el padron contra las estaciones de `historico_2018.csv`, que es el anio con
encoding sano, **comparando por linea y nombre** (no solo por nombre, para no unir homonimos de
lineas distintas). De 87 combinaciones en 2018, 70 coinciden exactas con el padron; estas 12 son
alias reales:

| Linea | Como figura en los historicos | Nombre canonico (padron) |
|---|---|---|
| A | `FLORES` | `SAN JOSE DE FLORES` |
| B | `CALLAO.B` | `CALLAO` |
| B | `LEANDRO N. ALEM` | `LEANDRO ALEM` |
| B | `ROSAS` | `JUAN MANUEL DE ROSAS` |
| C | `GENERAL SAN MARTIN` | `SAN MARTIN` |
| C | `MARIANO MORENO` | `MORENO` |
| D | `MINISTRO CARRANZA` | `CARRANZA` |
| D | `PUEYRREDON.D` | `PUEYRREDON` |
| D | `SCALABRINI ORTIZ` | (sin equivalente: ver seccion 4) |
| E | `GENERAL BELGRANO` | `BELGRANO` |
| E | `INDEPENDENCIA.H` | `INDEPENDENCIA` |
| H | `HUMBERTO I` | `HUMBERTO 1°` |
| H | `PATRICIOS` | `PARQUE PATRICIOS` |

Dos observaciones sobre el sufijo de linea:

- `CALLAO.B` y `PUEYRREDON.D` desambiguan homonimos: `CALLAO` existe en B y en D, `PUEYRREDON`
  tambien. El sufijo es **redundante** con la columna de linea, asi que se descarta sin perdida.
- `INDEPENDENCIA.H` es **inconsistente**: aparece en registros de la **linea E**, y la estacion
  Independencia es combinacion C/E, no toca la linea H. El sufijo esta mal asignado en el origen.
  Se normaliza a `INDEPENDENCIA` conservando la linea que declara la fila, pero queda anotado como
  anomalia del dataset.

## 4. Estaciones sin coordenada en el padron

Operan y aparecen en los historicos, pero el padron de accesibles no las incluye (no tienen
ascensor). Necesitan coordenada de otra fuente antes de poder asignarles comuna:

| Linea | Estacion |
|---|---|
| A | `ALBERTI` |
| A | `PASCO` |
| A | `RIO DE JANEIRO` |
| D | `SCALABRINI ORTIZ` |
| E | `PZA. DE LOS VIRREYES` (terminal) |

## 5. Estaciones que no existian en 2018

Estan en el padron pero no en `historico_2018.csv`, porque corresponden a la **extension de la
linea E hacia Retiro**, habilitada despues:

- `E | CATALINAS`
- `E | CORREO CENTRAL`
- `E | RETIRO`

Son justamente estaciones del area de oficinas, asi que entran de lleno en las hipotesis H1 y H2.
**Consecuencia metodologica:** su serie arranca mas tarde que el resto, de modo que un cero en 2014
no es caida de demanda sino estacion inexistente. No pueden integrar la linea base pre-pandemia en
igualdad de condiciones con el resto, y hay que tratarlas aparte.

## 6. Validacion aritmetica del cierre

El conteo cuadra de forma exacta, lo que da confianza en que no quedaron alias sin detectar:

```
 85  estaciones unicas en el padron
 -3  que no existian en 2018 (Catalinas, Correo Central, Retiro E)
 +5  que operan pero no estan en el padron (Alberti, Pasco, Rio de Janeiro,
     Scalabrini Ortiz, Pza. de los Virreyes)
----
 87  estaciones en historico_2018.csv   <- coincide con lo observado
```

Total de la red una vez unificado: **90 estaciones** (85 del padron + 5 faltantes).

## 7. Como se aplica

Orden de las operaciones, que no es indistinto:

1. **Normalizar la linea** primero (cuatro nomenclaturas: `A`, `LINEA_A`, `LineaA`, y las mezclas de
   2015 y 2017) a una sola letra, porque el resto del diccionario esta indexado por linea.
2. **Resolver el encoding** por anio, con un diccionario explicito por grafia observada. No sirve
   intentar un `decode`: en 2020 el caracter ya se perdio.
3. **Pasar a mayusculas y recortar espacios** al final.
4. **Quitar el sufijo de linea** (`.B`, `.D`, `.H`).
5. **Aplicar la tabla de alias** de la seccion 3.
6. **Validar** que el resultado este contenido en el padron de 90. Toda estacion que quede afuera es
   un alias nuevo sin detectar, y debe hacer fallar el proceso en vez de pasar silenciosamente: es
   la unica forma de que un anio nuevo no introduzca un desdoblamiento inadvertido.

## 8. Registros a descartar

| Caso | Volumen | Criterio |
|---|---|---|
| Linea y estacion vacias | 78.207 filas en 2018 | Sin identificacion de origen, no imputable |
| Linea vacia con estacion presente | 56 filas en 2018, 1.836 en 2020 | Linea recuperable desde la estacion, salvo homonimos |
| `TALLER BONIFACIO` | 56 filas en 2018 | Taller, no estacion de pasajeros |

El total descartado en 2018 es del orden del 0,64 % del anio, proporcion que se informa en el
documento de la Etapa 2 en lugar de descartarse en silencio.

## 9. Pendiente

- [ ] Coordenadas de las 5 estaciones faltantes, desde el padron oficial de estaciones de la Ciudad
- [ ] Diccionario de grafias con encoding roto, anio por anio (requiere listar las variantes de cada uno)
- [ ] Verificar si los anios 2022+ introducen alias o estaciones nuevas
- [ ] Asignacion estacion -> comuna, una vez completas las 90 coordenadas
