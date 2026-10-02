# Fuentes externas para el correlato economico

> Caso subtes del Desafio Profesional. El dataset de molinetes mide circulacion de personas, pero no
> actividad economica: para responder la cuarta pregunta del
> [enfoque analitico](enfoque-analitico.md) hace falta una fuente externa de gastronomia y hoteleria.
> Este documento registra que se busco, que se encontro, que se verifico y que se descarto.

## 1. Conclusion, primero

**Ninguna fuente oficial cumple las cuatro condiciones ideales a la vez** (desagregacion por comuna +
gastronomia aislable + cobertura 2014-2026 + frecuencia al menos anual). Es una limitacion real del
dominio, no de la busqueda: el relevamiento de locales por comuna y rubro en CABA simplemente no se
hizo de forma continua.

La salida no es forzar una sola fuente, sino **combinar dos que se complementan**: una serie temporal
fina sin territorio, y un corte territorial sin serie. El subte aporta la unica variable que tiene
las dos cosas, y por eso queda como eje del analisis.

| Rol en el analisis | Fuente | Cubre | Lo que le falta |
|---|---|---|---|
| **Serie temporal de actividad** | Indice de Actividad en Restaurantes (IDECBA) | 2015-2026, **mensual** | Sin comuna (total Ciudad) |
| **Corte territorial actual** | Locales ocupados por comuna y rubro, ejes comerciales (IDECBA) | 2025-2026, cuatrimestral | Solo 5 cuatrimestres |
| **Corte territorial pre-pandemia** | Censo de locales comerciales por comuna y rubro (IDECBA) | **Solo 2019** | Una sola foto |
| **Estructura territorial del empleo** | Empleo formal AMBA por radio censal, CLAE 55 y 56 | Solo octubre 2021 | Una sola foto |
| Descartado como eje | Habilitaciones comerciales (AGC) | 2014-2024 | Ver seccion 5 |

## 2. Indice de Actividad en Restaurantes (la mejor fuente)

Archivo: `idecba-EE_indice_actividad_restaurante.xlsx`. Instituto de Estadistica y Censos de la
Ciudad (IDECBA), area tematica "Hoteles y Restaurantes".

- **Mide volumen de actividad real**, no aperturas: segun su ficha tecnica, el indice se construye
  sobre cubiertos servidos. Base **marzo 2015 = 100**.
- **Serie mensual verificada de 2015 a agosto de 2026**, con variacion intermensual e interanual ya
  calculadas.
- Cubre exactamente el periodo que el caso necesita, con frecuencia mas fina que la anual.

Valores de abril de cada anio, que al ser el mismo mes evitan el efecto estacional:

| Abril | Indice | Lectura |
|---|---:|---|
| 2015 | 112,3 | |
| 2018 | 128,5 | |
| **2019** | **139,4** | linea base pre-pandemia |
| 2020 | `///` | **sin dato**: actividad suspendida |
| 2021 | 63,7 | **-54 %** contra 2019 |
| 2022 | 164,1 | +18 % contra 2019 |
| 2023 | 182,1 | +31 % contra 2019, maximo de la serie |
| 2024 | 153,5 | |
| 2025 | 162,9 | |
| 2026 | 158,2 | +13 % contra 2019 |

Tres observaciones que condicionan la interpretacion del caso:

1. **Abril de 2020 no es un cero, es un dato ausente** (`///`). No corresponde imputarlo: la
   actividad estaba administrativamente suspendida, de modo que no hay valor que estimar. Es un hueco
   declarado, igual que los ceros por obra en los molinetes.
2. **La recuperacion supero el nivel pre-pandemia**, y con holgura: en 2023 el indice llega a 182
   contra 139 de 2019. Esto **tensiona la hipotesis H5** tal como estaba formulada: la gastronomia de
   la Ciudad en su conjunto no quedo deprimida.
3. **Desde 2023 el indice baja** (182 -> 153 -> 163 -> 158). O sea que el pico de recuperacion ya
   paso, y la caida posterior es un fenomeno distinto del efecto pandemia.

Consecuencia para el trabajo: si el total de la Ciudad se recupero pero el subte de las estaciones de
oficina no, entonces la actividad gastronomica **se redistribuyo territorialmente** en vez de caer.
Esa es una conclusion mas interesante que la hipotesis original, y es contrastable con el corte por
comuna de la seccion 3.

## 3. Locales ocupados por comuna y rubro (el corte territorial)

Archivo: `idecba-AC_EJ_2026_08.xlsx`. Relevamiento de **ejes comerciales** (no censo total): IDECBA
recorre 48 ejes y clasifica cada local por rubro.

- Desagregacion: **comuna 1 a 15 x rubro**, con el rubro **"Alojamiento y comidas"**, que es
  exactamente gastronomia mas hoteleria.
- Frecuencia **cuatrimestral**, cinco cortes: 1er cuatrimestre 2025 a 2do cuatrimestre 2026.
- Unidad: locales ocupados (stock), que es justamente lo que las habilitaciones no pueden medir.

Lo mas relevante de estos cinco cortes:

| Corte | Total locales ocupados CABA | "Alojamiento y comidas" CABA | Idem **Comuna 1** |
|---|---:|---:|---:|
| 1er cuatr. 2025 | 11.843 | 1.217 | 296 |
| 2do cuatr. 2025 | 11.819 | 1.220 | 310 |
| 3er cuatr. 2025 | 11.781 | 1.220 | 305 |
| 1er cuatr. 2026 | 11.605 | 1.264 | 315 |
| 2do cuatr. 2026 | 11.528 | 1.278 | 318 |

**El total de locales ocupados cae 2,7 %, mientras "Alojamiento y comidas" crece 5,0 %, y en la
Comuna 1 crece 7,4 %.** La Comuna 1 es el microcentro, es decir la zona de oficinas que el caso pone
en el centro de la hipotesis.

Es un resultado contraintuitivo que conviene tener a la vista desde el principio: en el periodo mas
reciente la gastronomia del microcentro **se expande** mientras el comercio en general se contrae. La
hipotesis de partida del caso (la ida al trabajo remoto deprime la gastronomia de oficinas) no se
sostiene para 2025-2026, al menos no en stock de locales. Puede sostenerse para 2020-2022, que es el
periodo donde el corte territorial no existe.

## 4. Censo de locales comerciales 2019 (la foto pre-pandemia)

Archivos: `idecba-EEF_EFIS_CenLoc_Dist_RubAg_Com.xlsx` (rubro agrupado x comuna) y
`idecba-EEF_EFIS_CenLoc_Dist_Rub.xlsx` (por rubro, total Ciudad).

- Unica foto **por comuna y rubro anterior a la pandemia**: el banco de datos historico de IDECBA
  tiene 18 conjuntos sobre locales comerciales y **todos los que cruzan comuna con rubro son de
  2019**.
- Rubro agrupado **"Gastronomia, esparcimiento y Hoteleria"**: 12,26 % de los locales de la Ciudad,
  y **20,17 % en la Comuna 1**, el valor mas alto junto con la Comuna 14 (20,36 %). Confirma con dato
  que el microcentro ya era gastronomia-intensivo antes de la pandemia.
- **Esta expresado en porcentajes, no en cantidades.** Para llevarlo a absolutos hay que combinarlo
  con "Locales comerciales y distribucion porcentual por comuna" del mismo relevamiento.

**Advertencia de comparabilidad:** 2019 es un *censo de locales* y 2025-2026 es un *relevamiento de
48 ejes comerciales*. Son operativos distintos, con universos distintos, y los rubros agrupados no
coinciden (`Gastronomia, esparcimiento y Hoteleria` contra `Alojamiento y comidas`). **No se pueden
restar.** Sirven para comparar **composicion relativa** entre comunas dentro de cada corte, nunca
para calcular una variacion 2019-2026.

## 5. Habilitaciones comerciales: evaluadas y descartadas como eje

Archivos descargados: `habilitaciones-2015-2018.csv` a `habilitaciones-2026.csv`, del portal de datos
abiertos de la Ciudad. Eran el candidato mas prometedor por tener fecha y domicilio en cada registro,
lo que permitiria agregar por comuna cualquier anio. **Se descartan como variable principal** por
cuatro razones verificadas:

1. **Miden aperturas, no actividad ni stock.** No hay registro de bajas, de modo que no informan
   cuanta actividad existe, solo cuantos locales nuevos se habilitan.
2. **La serie de gastronomia no arranca en 2014.** Revisando los 257 rubros distintos de las 7.845
   filas de 2014, **no hay ni un rubro gastronomico**: los unicos candidatos son falsos positivos
   (talabarteria por la palabra "cuero", bebidas envasadas que es comercio minorista). La proporcion
   de gastronomia sobre el total pasa de 0 % en 2014 a 0,7 % en 2015, 6 % en 2016 y 18,7 % en 2020.
   Esa progresion no es economia, es **cambio de nomenclador**: hasta 2018 rige el codificador viejo
   (`COM.MIN.DE ...`) y en paralelo aparece el nuevo (`Alimentacion en general y gastronomia`).
3. **No traen comuna.** Los archivos 2014-2024 tienen `calles`, seccion, manzana, parcela y partida
   matriz, pero no comuna ni barrio: habria que geocodificar cada domicilio.
4. **Los archivos 2025 y 2026 son otra cosa**: tienen comuna, pero **no tienen fecha**. Son un stock
   de habilitaciones vigentes (335.123 filas), no un flujo, asi que no forman serie.

Problemas de calidad adicionales, por si se los retoma como fuente secundaria: 2021 trae 588 columnas
`Unnamed` vacias al final; 2022 tiene 406 filas fechadas en el anio 1900; 2023 tiene 642 filas sin
fecha y 2015-2018 otras 1.851; 2024 esta en latin1 mientras el resto esta en UTF-8.

Distribucion real por anio de habilitacion, una vez leida la fecha de cada fila en vez de confiar en
el nombre del archivo:

| Anio | Habilitaciones | | Anio | Habilitaciones |
|---|---:|---|---|---:|
| 2014 | 7.845 (parcial) | | 2020 | 12.938 |
| 2015 | 21.051 | | 2021 | 31.829 |
| 2016 | 31.061 | | 2022 | 26.024 |
| 2017 | 37.222 | | 2023 | 4.421 |
| 2018 | 38.433 | | 2024 | 8.637 |
| 2019 | 23.203 | | | |

La caida de 2023-2024 a menos de un sexto de 2022 es sospechosa de archivo incompleto o de cambio de
tramite, no de derrumbe de habilitaciones. Sin aclarar eso, la serie no es publicable.

## 6. Otras fuentes relevadas

| Fuente | Que aporta | Por que no alcanza |
|---|---|---|
| Empleo formal AMBA por radio censal (Min. Produccion / SIPA) | Aisla CLAE 55 (alojamiento) y 56 (alimentos y bebidas) y llega a nivel de radio censal, agregable a comuna | **Una sola foto, octubre 2021.** Sirve como peso territorial, no como serie |
| Puestos de trabajo registrados por rama (IDECBA / SIPA) | Rama hoteles y restaurantes, anual desde 2003 y mensual desde 2022 | Solo total Ciudad, sin comuna |
| Tasa de ocupacion hotelera por categoria (IDECBA) | Mensual, enero 2008 a junio 2026 | Solo total Ciudad, sin comuna |
| Encuesta de Ocupacion Hotelera (INDEC) | Mensual, CABA como localidad | Sin desagregacion por comuna; publicada en PDF |
| Establecimientos gastronomicos (ENTUR) | Barrio, comuna y coordenadas de cada local | Foto unica sin fecha, marcada como discontinuada |
| Ejes comerciales 53 ejes (IDECBA) | Serie 2do cuatr. 2022 a 3er cuatr. 2025, la mas larga de ocupacion comercial | Sin comuna y sin rubro |

## 7. Nota de acceso

`estadisticaciudad.gob.ar` **no responde a `curl`** desde esta maquina: el handshake TLS se corta por
timeout, con o sin User-Agent de navegador, mientras `data.buenosaires.gob.ar` responde en 43 ms. No
es falta de conectividad sino un filtro del sitio. Las paginas y los xlsx si se obtienen desde el
navegador, que es la via por la que se consiguieron estos archivos.

`data.buenosaires.gob.ar` rechaza el User-Agent por defecto de `curl` con un error de WAF
("Request Rejected"); con User-Agent de Chrome funciona.

## 8. Diseno resultante

El analisis queda estructurado en dos planos, y conviene decirlo explicitamente en el informe final
para no aparentar una precision que los datos no tienen:

- **Plano temporal (2014-2026, fino).** Molinetes diarios contra el indice mensual de restaurantes.
  Permite medir co-movimiento, el desfasaje entre la caida del subte y la de la gastronomia, y la
  brecha contra la tendencia pre-pandemia. Sin desagregacion territorial.
- **Plano territorial (cortes 2019 y 2025-2026).** Flujo de subte por comuna, derivado de la
  asignacion estacion-comuna, contra composicion de locales por comuna. Permite contrastar la
  asimetria entre comunas de oficina y residenciales. Sin serie continua.

Ninguno de los dos planos solo responde la pregunta del caso; juntos si, y la honestidad sobre el
limite de cada uno es parte del entregable.

## 9. Pendiente

- [ ] Convertir los porcentajes de 2019 a cantidades absolutas, cruzando con el total por comuna
- [ ] Extraer el indice de restaurantes a un CSV limpio (el xlsx trae los anios como filas sueltas
      intercaladas entre los meses, no como columna)
- [ ] Decidir si se reformula H5 a la luz de que la gastronomia de la Comuna 1 crece en 2025-2026
- [ ] Verificar si el relevamiento de 48 ejes tiene cortes anteriores a 2025 con comuna y rubro
