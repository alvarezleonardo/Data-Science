# Caso Subtes — Enfoque analitico elegido

> Caso de negocio seleccionado para el Desafio Profesional. El brief original esta en
> [2.1 - Casos de Negocio - Subtes.md](<2.1 - Casos de Negocio - Subtes.md>); el enunciado habilita
> explicitamente a plantear preguntas propias ("tu podras buscar responder otras que consideres
> relevantes"). Este documento fija la pregunta de negocio, las hipotesis y el plan de las 4 etapas.

## 1. Pregunta de negocio

**Como impacto la pandemia en el uso del subte, y como se traslado ese cambio a la actividad
economica (gastronomia y hoteleria) de las zonas de oficinas de CABA?**

Se descompone en cuatro preguntas operativas:

1. **Magnitud de la caida.** Cuanto cayo el flujo de pasajeros por estacion y linea a partir de
   marzo de 2020, respecto de la linea base pre-pandemia.
2. **Asimetria territorial.** La caida fue homogenea en la red, o golpeo mas fuerte a las
   estaciones del area de oficinas (microcentro, Tribunales, Catalinas) que a las residenciales?
3. **Grado de recuperacion.** A partir de que momento y en que porcentaje se recupero el flujo, y
   ese porcentaje es distinto en estaciones de oficina que en el resto de la red (indicio de que el
   trabajo remoto no se revirtio del todo).
4. **Correlato economico.** La evolucion de locales gastronomicos y hoteleros por comuna acompana
   la evolucion del flujo de subte de esa comuna?

## 2. Hipotesis

| # | Hipotesis | Como se contrasta |
|---|---|---|
| H1 | La caida 2020 fue mas profunda en estaciones de oficina que en residenciales | Comparar la caida porcentual por grupo de estaciones contra la linea base 2019 |
| H2 | La recuperacion post-2020 es **parcial y desigual**: las estaciones de oficina recuperan un porcentaje menor | Ratio flujo recuperado / linea base, por grupo |
| H3 | El patron semanal cambio: cae mas el lunes-viernes que el fin de semana, y dentro de la semana cae mas el lunes y el viernes (hibrido 3x2) | Perfil dia-de-semana por periodo; comparar la forma de la curva, no solo el nivel |
| H4 | Los picos horarios (commuting 7-9 y 17-19) se achatan y se corren | Perfil horario por periodo |
| H5 | Las comunas con mayor caida sostenida de flujo muestran peor evolucion de locales gastronomicos | Correlacion entre variacion de flujo y variacion de locales, a nivel comuna |

H3 y H4 son clave: distinguen una caida generalizada de actividad de un **cambio estructural en el
motivo del viaje**. Es la evidencia que permite hablar de trabajo remoto y no solo de recesion.

## 3. Fuentes de datos

### 3.1 Provistas con el caso (en `datasets/`, no versionadas)

| Archivo | Contenido | Observacion |
|---|---|---|
| `historico_2014.csv` .. `historico_2021.csv` | Pasajeros por molinete, estacion, linea, en tramos de 15 minutos | ~8 GB sin comprimir. **Seis formatos distintos** entre los 8 anios: ver seccion 5 |
| `estaciones-accesibles.csv` | 94 filas con `long`, `lat`, `linea`, `estacion` | **Es la clave del cruce territorial**: aporta la coordenada de cada estacion |
| `lineas-de-subte.csv` | Trazado de las lineas en WKT | Para el mapa |
| `registro-historico-del-precio-del-boleto.csv` | 305 registros de tarifa desde 1994 | Permite deflactar y analizar elasticidad precio-demanda |
| `viajes_anual.csv` | 49 filas, total anual | Control de consistencia de los agregados propios |
| `BaseUnificadaEstaciones.csv` | 822.364 filas: `FECHA, ESTACION, LINEA, DESDE, HORA, CANTIDAD` | Subconjunto ya agregado (arranca en 2020). Util como validacion cruzada |

### 3.2 Fuentes externas a incorporar

**Actividad economica — Direccion General de Estadistica y Censos de CABA**
(`estadisticaciudad.gob.ar`, seccion Locales comerciales, series historica y no historica):
releva locales por comuna y por rubro, lo que permite aislar gastronomia y hoteleria.
Es la fuente que habilita la pregunta 4.

**Extension temporal de los molinetes** (ver seccion 4): los anios 2022 en adelante estan
publicados en el portal de datos abiertos de la Ciudad, con el mismo origen que los provistos.

### 3.3 Unidad de cruce: de estacion a comuna

El dataset de molinetes no trae comuna ni barrio. La cadena de union es:

```
molinete -> estacion -> (lat, long) de estaciones-accesibles.csv -> comuna / barrio -> locales por comuna
```

El ultimo paso requiere el poligono de comunas/barrios de CABA (dato abierto) y una asignacion
espacial punto-en-poligono. Con 94 estaciones el resultado es auditable fila por fila, asi que la
asignacion se revisa a mano antes de usarla.

Ademas de la comuna, cada estacion se clasifica por **funcion urbana** (oficinas / mixta /
residencial). Es un criterio del analista, no un dato del origen: queda documentado de forma
explicita y es el que sostiene H1 y H2.

## 4. Limitaciones conocidas

1. **El dataset provisto termina el 31/12/2021.** La pregunta del usuario incluye que la modalidad
   virtual "sigue", lo cual no se puede sostener con datos que cortan en 2021: 2021 todavia tiene
   restricciones vigentes, de modo que lo que se mide ahi es recuperacion temprana, no el nuevo
   equilibrio. **Decision propuesta:** incorporar los anios 2022+ desde el portal de datos abiertos
   de la Ciudad. Si se decide no ampliar, el alcance de la conclusion se acota a "recuperacion al
   cierre de 2021" y se declara como limitacion.
2. **Los molinetes miden ingresos, no viajes ni personas distintas.** Una persona que viaja ida y
   vuelta cuenta dos veces; una combinacion puede contar una sola. Es un proxy de circulacion, y
   como tal se lo nombra en todo el analisis.
3. **Correlacion no es causalidad.** La caida de locales gastronomicos en 2020 tiene como causa
   directa el cierre administrativo, no la ausencia de oficinistas. El analisis puede sostener
   co-movimiento y asimetria territorial; atribuir la variacion al trabajo remoto exige controlar
   por el calendario de restricciones.
4. **Cambios en la red.** Aperturas y renombramientos de estaciones entre 2014 y 2021 cortan
   algunas series; hay que detectarlos antes de interpretar un cero como caida de demanda.
5. **Obras y paros** generan ceros que no son caida de demanda. Se marcan como datos faltantes
   justificados, no como observaciones validas.

## 5. Hallazgo tecnico: seis formatos en ocho anios

Verificado leyendo los encabezados reales de cada archivo. Este es el nucleo del trabajo de la
Etapa 2:

| Anio | Separador | Columnas / particularidad | Formato de fecha | Ejemplo de linea y estacion |
|---|---|---|---|---|
| 2014 | `,` con comillas | `ID_ESTACION`, `PAX_PAGO` (singular), nulos como `NA` | ISO | `"B"`, `"FLORIDA"` |
| 2015 | `,` | `periodo` primero, todo minuscula, `total` | ISO | `LINEA_H`, `CASEROS` |
| 2016 | `,` con comillas | typo `PAX_FREQ` por `PAX_FRANQ` | `02/01/2016` | `"D"`, `"9 DE JULIO"` |
| 2017 | `,` con comillas | columna indice extra `V1`, mismo typo `PAX_FREQ` | ISO | idem 2016 |
| 2018 | `,` | `periodo` **al final** | ISO | `LineaA`, `Castro Barros` |
| 2019 | `,` | `periodo` primero | ISO | `LineaA`, `Lima` |
| 2020 | `,` | `pax_TOTAL`, sin `ID_ESTACION` | `01/01/2020` | mayusculas |
| 2021 | **`;`** | separador punto y coma | `1/1/2021` | mayusculas |

Tres ejes de heterogeneidad a normalizar: **separador**, **nomenclatura de columnas** (con dos
typos y un indice espurio) y, el mas delicado, **naming de linea y estacion**. La normalizacion de
estaciones es condicion necesaria para cualquier serie de 8 anios, y el enunciado del caso ya la
pide explicitamente.

### 5.1 Volumen y cobertura real, verificados

Relevado recorriendo los 8 archivos completos:

| Anio | Filas | Estaciones distintas | Cobertura |
|---|---:|---:|---|
| 2014 | 10.857.244 | 83 | anio completo (02/01 a 31/12) |
| 2015 | 10.958.582 | **168** | anio completo |
| 2016 | 11.542.322 | 83 | anio completo |
| 2017 | 11.938.476 | 83 | anio completo |
| 2018 | 12.136.454 | 89 | anio completo |
| 2019 | 12.662.343 | 94 | anio completo |
| 2020 | **5.781.006** | 91 | anio completo |
| 2021 | 8.071.680 | 96 | anio completo |
| **Total** | **83.948.107** | | |

Dos lecturas inmediatas:

- **2020 cae a menos de la mitad de 2019** (5,78 M contra 12,66 M filas). La caida aparece ya en el
  conteo de registros, antes de sumar un solo pasajero: con el subte cerrado o restringido no hay
  ni siquiera tramos de 15 minutos con actividad para registrar. 2021 recupera a 8,07 M, un 64 %
  de 2019, lo que apunta a recuperacion parcial y es consistente con H2.
- **2015 declara 168 estaciones** donde el resto declara entre 83 y 96. No se duplico la red: el
  anio trae dos grafias para la misma estacion (`SAENZ PEÑA` y `Saenz Peña `). Es el sintoma mas
  claro de que el conteo de estaciones unicas, sin normalizar, no significa nada.

Los archivos **no estan ordenados por fecha**: 2014 abre y cierra con registros del 09/04. No se
puede inferir el rango temporal mirando la primera y la ultima fila.

### 5.2 El problema serio: nombres de estacion con encoding roto

Los dos nombres con caracteres no ASCII (**Agüero** y **Saenz Peña**) aparecen corrompidos de forma
**distinta en cada anio**:

| Anio | Como llega |
|---|---|
| 2014, 2016, 2017 | latin1 / cp1252 sin convertir: `SAENZ PE?A` con byte invalido |
| 2015 | UTF-8 correcto, pero convive con una segunda grafia en mayusculas |
| 2018 | UTF-8 correcto (`Agüero`, `Saenz Peña `) |
| 2019 | mojibake y **doble** mojibake: `AgÃ¼ero`, `AgÃÂ¼ero`, `Saenz PeÃ±a `, `Saenz PeÃÂ±a ` |
| 2020 | caracter de reemplazo: `Ag?ero`, `Saenz Pe?a ` (informacion ya perdida) |
| 2021 | cuatro variantes mas: `Agero`, `Ag³ero`, `Saenz Pe¤a `, `Saenz Pe±a ` |

Sumado a esto: **mayusculas en 2014, 2016 y 2017** (`PALERMO`, `9 DE JULIO`) contra capitalizado en
el resto, y **espacios al final** (`Saenz Peña ` persiste en varios anios).

Consecuencia practica: un `groupby('estacion')` sobre la union de los 8 anios, sin normalizar,
parte Agüero en al menos ocho estaciones distintas y rompe la serie historica justamente en las
estaciones afectadas. El caso de 2020 es el peor, porque el caracter de reemplazo es irreversible:
la unica salida es un diccionario explicito, no un decode.

**Decision:** el normalizador de nombres no puede ser un `.str.upper().str.strip()`. Requiere un
diccionario de equivalencias construido a mano, validado contra las 94 estaciones de
`estaciones-accesibles.csv`, que es el padron de referencia. Ese diccionario es un entregable de la
Etapa 2.

### 5.3 Linea: cuatro nomenclaturas y valores vacios

| Anio | Valores de LINEA |
|---|---|
| 2014, 2016 | `A` .. `H` |
| 2015 | **12 valores**: `LINEA_A`..`LINEA_H` y `LineaA`..`LineaH` mezclados |
| 2017 | **mezcla parcial**: `B`, `D`, `E` sin prefijo; `LINEA_A`, `LINEA_C`, `LINEA_H` con prefijo |
| 2018, 2019, 2020, 2021 | `LineaA` .. `LineaH` |

Ademas hay **filas con LINEA vacia**: 78.263 en 2018 y 1.836 en 2020. En 2018, 78.207 de esas filas
tienen **tambien la estacion vacia**, es decir que son registros sin identificacion de origen: no se
pueden imputar y corresponde descartarlos, documentando el volumen (0,64 % del anio). Las 56 filas
de `Taller Bonifacio` en 2018 son de un taller, no de una estacion de pasajeros, y tambien salen.

## 6. Plan por etapas

### Etapa 1 — Exploracion visual

Con ~100 millones de filas no se carga todo en memoria. Estrategia:

1. Leer cada anio **por chunks**, aplicando el normalizador de su formato.
2. Agregar en la lectura a grano `fecha x estacion x linea x hora`, que es el grano minimo que
   sostiene todas las hipotesis (H3 necesita el dia, H4 la hora).
3. Persistir el agregado en **Parquet**, que pasa a ser la entrada de las etapas siguientes.
4. Sobre el agregado (ya manejable en memoria) correr el EDA: `info`, `describe`, nulos, outliers,
   serie temporal, perfil semanal, perfil horario, ranking de estaciones, heatmap de correlacion.

Entregables de la consigna: notebook documentada, visualizaciones y documento de hallazgos.

### Etapa 2 — Limpieza y transformacion

Formalizar el normalizador de los 6 formatos, el diccionario de estaciones, la asignacion
estacion-comuna, el tratamiento de ceros por obra o paro, y la integracion con la fuente de
locales comerciales. Salida: dataset analitico unico y reproducible.

### Etapa 3 — Machine Learning

Clustering de estaciones por perfil de uso (K-Means / DBSCAN, que el brief pide) comparando el
agrupamiento pre y post pandemia: si los clusters se reordenan, es evidencia directa de cambio
estructural. Mas regresion de demanda contra tarifa real, calendario y restricciones.

### Etapa 4 — Redes neuronales

Serie temporal con LSTM/GRU: entrenar con pre-pandemia, proyectar el contrafactual "sin pandemia" y
medir la brecha contra lo observado. La brecha es la cuantificacion del impacto, y responde el
"en que porcentaje" de la pregunta original.

## 7. Estado y proximos pasos

- [x] Caso elegido y datasets consolidados en `datasets/` (excluidos de git por `.gitignore`)
- [x] Formatos de los 8 anios relevados
- [x] Via de cruce territorial identificada (coordenadas por estacion)
- [x] Inventario de estaciones, lineas, volumen y cobertura por anio (seccion 5.1 a 5.3)
- [ ] Diccionario de normalizacion de estaciones, contra el padron de 94 de `estaciones-accesibles.csv`
- [ ] Decision sobre extender la serie a 2022+
- [ ] Descarga de la serie de locales comerciales por comuna y rubro
- [ ] Notebook de la Etapa 1
