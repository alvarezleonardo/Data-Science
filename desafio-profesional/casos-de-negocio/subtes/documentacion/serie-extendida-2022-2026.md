# Serie extendida 2022-2026: relevamiento y calidad

> Ampliacion del dataset de molinetes mas alla del material provisto con el caso, que terminaba el
> 31/12/2021. Fuente: dataset `subte-viajes-molinetes` del portal de datos abiertos de la Ciudad
> (SBASE), el mismo origen que los archivos originales. Complementa
> [enfoque-analitico.md](enfoque-analitico.md) y [diccionario-estaciones.md](diccionario-estaciones.md).

## 1. Que se incorporo

| Anio | Filas | Dias | Pasajeros | Observacion |
|---|---:|---:|---:|---|
| 2022 | 12.136.381 | 365 | 231.459.457 | |
| 2023 | 12.047.696 | 364 | 243.225.292 | falta el 13/08/2023 |
| 2024 | 11.440.440 | 363 | 201.386.688 | faltan el 09/05 y el 30/10/2024 |
| 2025 | 13.196.766 | 365 | 206.616.377 | **agosto con fechas corruptas**, ver seccion 3 |
| 2026 | 7.159.607 | 181 | 97.813.060 | **publicado solo hasta el 30/06/2026** |

La serie completa pasa de 83,9 millones de registros (2014-2021) a **139,9 millones** repartidos en
**12 anios y medio**. La granularidad se mantiene en tramos de 15 minutos por molinete.

**El corte real de la serie es el 30/06/2026**, no la fecha de hoy: julio en adelante todavia no esta
publicado. Toda conclusion sobre "la situacion actual" se refiere al primer semestre de 2026.

## 2. Cambios de formato respecto de 2014-2021

Los anios nuevos comparten entre si el formato de 2021 (separador `;`, fecha `D/M/YYYY` sin ceros),
pero traen novedades propias:

| Novedad | Desde | Implicancia |
|---|---|---|
| Un CSV por mes y grupo de lineas (`YYYYMM_PAX15min-ABC.csv` y `-DEH.csv`) | 2022 | El mes esta en el **nombre del archivo**: es informacion que no hay que perder al concatenar, y resulta clave para la seccion 3 |
| Sufijo `-INCLUYEOTROMODOSDEPAGO` en el nombre | 2024-12 | Posible cambio de definicion de la medicion |
| Archivos `-PM` con la linea `LineaPM` (Premetro) | 2025-06 | **La red cambia de alcance**: el Premetro no estaba antes |
| Filas con linea, molinete y estacion = `Prueba` | 2025-06 | Registros de prueba, hay que filtrarlos |
| BOM UTF-8, lineas `;;;;`, lineas en blanco (82.305 en 202501-DEH) | varios | Ruido de exportacion |
| Cuatro columnas extra al principio (`HORA;DIA;TIPO_DIA;¿Fuera de rango?`) en ISO-8859 | 4 archivos de 2022 | Rompen el indice de columnas si no se detectan |

### El cambio de diciembre de 2024 no produjo un salto visible

El sufijo `INCLUYEOTROMODOSDEPAGO` hacia temer una ruptura de serie: si desde diciembre de 2024 se
cuentan medios de pago que antes no se contaban, el nivel sube por un cambio de definicion y no por
mas demanda. **Se verifico sumando pasajeros mes a mes y no hay salto:**

```
2024-10  17.229.047
2024-11  16.697.874
2024-12  15.547.350   <- primer mes con el sufijo, y BAJA
2025-01  12.732.331
```

Diciembre baja respecto de noviembre, y enero mas, que es la estacionalidad normal de verano. No se
observa el escalon que un cambio de definicion habria producido. Igualmente **queda registrado como
riesgo**, porque no hay documentacion oficial de que se agrego, y conviene no apoyar ninguna
conclusion fina en la comparacion de niveles absolutos que cruce diciembre de 2024.

> Este es, de paso, el segundo argumento a favor de la decision de
> [comparar formas de perfil en vez de niveles](diseno-analisis-flujo.md): un cambio de definicion de
> la medicion altera el nivel de toda la red por igual, pero no altera la distribucion horaria de cada
> estacion. El indicador elegido es inmune a este problema.

## 3. El problema de agosto de 2025, y como se resuelve

**Los archivos de agosto de 2025 tienen las fechas en dos formatos mezclados**, parte en `D/M/YYYY` y
parte en `M/D/YYYY`, dentro del mismo archivo.

Lo verificado:

- **283.617 filas** del anio tienen el segundo campo mayor que 12 (por ejemplo `8/20/2025`), lo que
  solo es posible si ese campo es el dia y el primero el mes.
- Esas filas suman **3.513.067 pasajeros**.
- **Las 283.617 tienen primer campo igual a 8 sin excepcion**, de modo que el problema esta
  **confinado a los archivos de agosto** y no contamina al resto del anio.

El dano, si se lee ingenuamente con dia primero:

```
2025-07  19.936.627
2025-08  13.563.219   <- agosto aparece 32 % por debajo de julio
2025-09  19.713.080
```

Agosto no tuvo una caida del 32 %: perdio los 3,5 millones de las filas invertidas, y peor aun, las
filas ambiguas (`8/5/2025`) se contabilizan en **mayo**, inflando un mes que no corresponde. Un
analisis de series temporales sobre este dato veria un derrumbe y una recuperacion que nunca
ocurrieron.

### La solucion es univoca

El mes esta en el nombre del archivo (`202508_PAX15min-...`), de modo que **el mes es conocido y vale
8**. Como uno de los dos campos de la fecha es necesariamente el mes, el otro es el dia:

| Fecha cruda | Campo que vale 8 | Dia resultante |
|---|---|---|
| `8/20/2025` | el primero | 20 de agosto |
| `20/8/2025` | el segundo | 20 de agosto |
| `8/5/2025` | el primero | 5 de agosto |
| `5/8/2025` | el segundo | 5 de agosto |
| `8/8/2025` | ambos | 8 de agosto |

La regla resuelve el 100 % de los casos sin ambiguedad, **pero solo si se preserva el mes del nombre
del archivo**. Concatenar los CSV mensuales en un unico archivo sin agregar una columna de mes destruye
la informacion que permite la correccion. Los archivos `historico_2022..2026.csv` que se armaron por
concatenacion tienen ese defecto, asi que para 2025 hay que volver a los originales.

**Control posterior obligatorio:** una vez corregidas las fechas, contar duplicados por
`fecha + desde + molinete`. Hay indicios de que una parte de las filas puede estar repetida en los dos
formatos, lo que inflaria agosto en vez de hundirlo.

## 4. Padron oficial de estaciones: problema resuelto

El dataset `subte-estaciones` del mismo portal aporta lo que faltaba:

- **`estaciones_de_subte.csv`**: **90 filas, una por estacion**, con `id`, `estacion`, `linea` y
  `geometry` en formato `POINT (lon lat)`. Distribucion por linea: A 18, B 17, C 9, D 16, E 18, H 12,
  que coincide exactamente con el total de 90 deducido en
  [diccionario-estaciones.md](diccionario-estaciones.md).
- **Cubre las 5 estaciones que faltaban** en el padron de accesibles: `Alberti`, `Pasco`,
  `Rio De Janeiro`, `R.Scalabrini Ortiz` y `Plaza De Los Virreyes - Eva Perón`.

- **`bocas-de-subte.csv`**: 379 filas, una por boca de acceso, con columnas **`barrio` y `comuna`**.
  **Esto elimina la necesidad de hacer asignacion espacial punto-en-poligono**: la comuna viene dada.

Dos advertencias sobre estos archivos:

1. **Los nombres traen una tercera ortografia.** El padron oficial usa denominaciones ampliadas que no
   coinciden ni con los historicos ni con el archivo de accesibles: `Inclan - Mezquita Al Ahmad`,
   `Plaza De Los Virreyes - Eva Perón`, `R.Scalabrini Ortiz`, `TRIBUNALES - TEATRO COLÓN`. El
   diccionario de normalizacion necesita una entrada mas por cada uno de estos casos.
2. **Una misma estacion puede tener bocas en barrios distintos**, lo cual es correcto (una estacion
   bajo una avenida limite entre barrios). Para asignar una comuna por estacion hay que fijar un
   criterio: la comuna de la mayoria de sus bocas, o la de la boca principal. Conviene revisar a mano
   los casos de frontera, que son pocos.

## 5. Pendiente

- [ ] Rearmar 2025 desde los CSV mensuales originales, preservando el mes del nombre del archivo
- [ ] Aplicar la regla de correccion de agosto de 2025 y controlar duplicados
- [ ] Decidir si el Premetro entra en el analisis (aparece recien en 2025-06, de modo que no tiene
      linea base pre-pandemia y rompe la comparabilidad de la red; lo razonable es excluirlo)
- [ ] Filtrar las filas `Prueba`
- [ ] Extender el agregado horario a 2022-2026
- [ ] Crosswalk entre las tres ortografias de estacion: historicos, accesibles y padron oficial
- [ ] Asignar comuna por estacion desde `bocas-de-subte.csv`, resolviendo los casos de frontera
- [ ] Averiguar si el 13/08/2023, el 09/05/2024 y el 30/10/2024 son paros, feriados o faltantes
