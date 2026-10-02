# Diseno del analisis: como cambio el flujo

> Reformulacion del caso subtes a partir de la pregunta precisa: no interesa tanto si la gastronomia
> cayo o crecio en total, sino **como se redistribuyo el flujo de personas** — en el espacio (cerraron
> locales en el microcentro y abrieron en otros barrios) y en el tiempo (menos mediodia laboral, mas
> noche). Complementa [enfoque-analitico.md](enfoque-analitico.md) y
> [fuentes-externas.md](fuentes-externas.md).

## 1. La restriccion de medicion que define todo el diseno

**Los molinetes registran solo ingresos al sistema.** En el subte de Buenos Aires se paga al entrar y
la salida es libre, de modo que no existe registro de egreso. Esto se verifico sobre los datos: los
600 nombres de molinete de 2018 tienen la forma `LineaA_Acoyte_N_Turn01`, donde `N`, `S`, `E` y `O`
son la **orientacion del acceso** (norte, sur, este, oeste) y `Turn` o `Asc` el tipo de dispositivo.
No hay marca de entrada contra salida.

La consecuencia es que **cada viaje se registra en su estacion de origen, nunca en la de destino**.
Para alguien que vive en Villa Urquiza y trabaja en el microcentro:

```
 manana  ->  pasa el molinete en Juramento (su barrio)        -> registra en la estacion RESIDENCIAL
 tarde   ->  pasa el molinete en Catedral (su oficina)        -> registra en la estacion de OFICINA
 almuerzo -> si camina a comer y vuelve, NO genera registro   -> invisible al dataset
```

De ahi se siguen tres cosas que hay que tener claras antes de interpretar cualquier grafico:

1. **El flujo matutino de una estacion mide gente que sale de esa zona**, no que llega. El pico de
   7 a 9 en estaciones residenciales es gente yendo a trabajar.
2. **El flujo vespertino de una estacion de oficina mide gente que estuvo trabajando ahi.** Es el
   mejor proxy disponible de presencia de oficinistas en la zona.
3. **El almuerzo caminando no deja rastro.** La pregunta "donde almuerza la gente" no se responde de
   forma directa: se responde por la via de cuanta gente hay en cada zona en cada franja.

## 2. La decision metodologica central: comparar formas, no niveles

El nivel absoluto de pasajeros esta contaminado por todo lo demas: tarifa, poblacion, obras,
pandemia, crisis economica, cambios de red. Comparar 2019 contra 2026 en pasajeros absolutos mezcla
el efecto que interesa con otros cinco.

**La solucion es normalizar cada perfil por su propio total.** Para cada estacion, cada periodo y cada
franja horaria se calcula la **participacion** de esa franja en el total diario de esa estacion:

```
participacion(estacion, franja, periodo) = pax(estacion, franja) / pax(estacion, dia completo)
```

Asi se obtiene la **forma de la curva horaria**, que es adimensional y comparable entre anios y entre
estaciones. Si una estacion pierde la mitad de sus pasajeros pero mantiene la misma forma, su funcion
urbana no cambio: solo hay menos gente. Si en cambio la forma se achata en el mediodia y engorda en la
noche, **la zona cambio de uso**, y eso es exactamente lo que el caso quiere demostrar.

Esta es la razon por la que el analisis puede sostener conclusiones sobre cambio de costumbres aunque
el dato sea solo de ingresos: la forma del perfil es una propiedad de la estacion comparada consigo
misma, no una medicion de volumen.

## 3. Franjas horarias

Definidas por funcion, no en bloques iguales:

| Franja | Horas | Que capta |
|---|---|---|
| Temprano | 5 a 7 | Turnos que arrancan antes de la jornada de oficina |
| **Pico manana** | 7 a 10 | Viaje al trabajo o al estudio |
| Media manana | 10 a 12 | Tramites, compras, turnos |
| **Mediodia** | 12 a 15 | Almuerzo y salida de jornadas reducidas |
| Media tarde | 15 a 17 | |
| **Pico tarde** | 17 a 20 | Vuelta del trabajo |
| **Noche** | 20 a 23 | Cena, salida, ocio |

Las cuatro en negrita son las que sostienen las hipotesis. Dos indicadores derivados concentran casi
todo el interes:

- **Ratio noche / mediodia** por estacion: si crece, la zona se corrio al consumo nocturno.
- **Ratio pico manana / pico tarde** por estacion: distingue estaciones emisoras (residenciales, que
  despachan gente a la manana) de receptoras (de oficina, que la despachan a la tarde). Si una
  estacion de oficina pierde su asimetria vespertina, perdio su funcion de oficina.

## 4. Clasificacion funcional de estaciones

Criterio propio del analisis, no un dato del origen, y por eso se declara de forma explicita. Cuatro
grupos sobre las 90 estaciones:

**Microcentro y oficinas** (Comuna 1, area bancaria, administrativa y de tribunales): `CATEDRAL`,
`PERU`, `PLAZA DE MAYO`, `BOLIVAR`, `FLORIDA`, `LEANDRO ALEM`, `LAVALLE`, `DIAGONAL NORTE`,
`CARLOS PELLEGRINI`, `9 DE JULIO`, `TRIBUNALES`, `SAN MARTIN`, `CORREO CENTRAL`, `CATALINAS`,
`AVENIDA DE MAYO`, `PIEDRAS`, `LIMA`.

**Corredor gastronomico y nocturno** (Palermo, Villa Crespo, Abasto, Barrio Norte): `PLAZA ITALIA`,
`SCALABRINI ORTIZ`, `PALERMO`, `BULNES`, `AGÜERO`, `MALABIA`, `DORREGO`, `CARLOS GARDEL`, `MEDRANO`,
`ANGEL GALLARDO`, `SANTA FE`, `LAS HERAS`, `CORDOBA`.

**Residencial y periferico**: `SAN PEDRITO`, `CARABOBO`, `PUAN`, `PRIMERA JUNTA`, `SAN JOSE DE FLORES`,
`JURAMENTO`, `CONGRESO DE TUCUMAN`, `JOSE HERNANDEZ`, `OLLEROS`, `LOS INCAS`, `JUAN MANUEL DE ROSAS`,
`TRONADOR`, `ECHEVERRIA`, `VARELA`, `MEDALLA MILAGROSA`, `EMILIO MITRE`, `PLAZA DE LOS VIRREYES`,
`HOSPITALES`, `PARQUE PATRICIOS`, `CASEROS`, `INCLAN`, y el resto de las no listadas arriba.

**Nodos de transferencia regional** (combinan con ferrocarril o metrobus y reciben pasajeros del
conurbano): `CONSTITUCION`, `RETIRO`, `FEDERICO LACROZE`, `ONCE`, `PLAZA MISERERE`.

> Este cuarto grupo **se analiza aparte y no se promedia con los demas**. Su flujo refleja commuting
> regional y no la actividad del barrio donde esta la estacion, de modo que incluirlo en el promedio de
> "microcentro" contaminaria el indicador. Es una decision que conviene explicitar en el informe,
> porque Retiro y Constitucion son dos de las estaciones de mayor volumen de la red y su inclusion o
> exclusion mueve los resultados.

## 5. Hipotesis reformuladas

Reemplazan a H1-H5 del enfoque original, que estaban planteadas en terminos de nivel y no de patron.

| # | Hipotesis | Indicador | Como se falsea |
|---|---|---|---|
| **F1** | El microcentro perdio participacion en el total de la red | Share de pax del grupo microcentro sobre el total, por anio | Si el share se mantiene, no hubo redistribucion territorial |
| **F2** | En el microcentro, el **mediodia** cayo mas que el resto del dia | Participacion de la franja 12-15 en el total diario de esas estaciones, 2019 contra 2026 | Si la participacion del mediodia no baja, el almuerzo laboral no se perdio |
| **F3** | El corredor gastronomico **gano participacion nocturna** | Ratio noche/mediodia de ese grupo, por anio | Si el ratio no crece, no hubo corrimiento al consumo nocturno |
| **F4** | Las estaciones residenciales retuvieron flujo en el mediodia | Participacion de 12-15 en residenciales, 2019 contra 2026 | Si no sube, la gente no se quedo a almorzar en su barrio |
| **F5** | El patron semanal se volvio hibrido: martes a jueves sostienen el nivel, lunes y viernes caen | Pax por dia de la semana, normalizado al promedio semanal, por anio y grupo | Si los cinco dias caen parejo, es menos actividad y no trabajo hibrido |
| **F6** | La relacion entre flujo de subte y actividad gastronomica se desacoplo | Correlacion entre pax mensual de la red y el indice de restaurantes, por subperiodos | Si la correlacion se mantiene estable, el subte sigue explicando la gastronomia |

**F2 y F4 son el corazon del argumento**, y funcionan como par: si el mediodia se achata en el
microcentro **y** se engrosa en los barrios residenciales, eso es evidencia directa de que el almuerzo
laboral se desplazo geograficamente. Ninguna de las dos sola alcanza, porque una caida del mediodia en
el microcentro podria ser solo menos actividad; es el movimiento **simultaneo y de signo opuesto** lo
que sostiene la conclusion.

**F5 es el control mas fuerte contra la explicacion alternativa.** Si la caida fuera simplemente
recesion o menos poblacion, afectaria los cinco dias de la semana por igual. Un patron que castiga
lunes y viernes y preserva el centro de la semana solo se explica por esquemas hibridos de tres dias
presenciales.

## 6. Articulacion con las fuentes externas

El subte aporta la grilla fina (diaria y horaria, 2014-2026). Las fuentes externas aportan la
confirmacion de stock:

- **F1 y F2** se confirman contra los locales de "Alojamiento y comidas" por comuna de los
  relevamientos 2025-2026, y contra la foto de 2019.
- **F3** se confirma contra el indice mensual de restaurantes: si el indice total se sostiene mientras
  el mediodia del microcentro cae, la actividad se movio de franja o de lugar, no desaparecio.
- **F6** es directamente el cruce de las dos series.

## 7. Lo que este diseno NO puede afirmar

Conviene fijarlo ahora para no sobre-interpretar despues:

- **No mide comensales.** Mide ingresos al subte. Una zona puede ganar actividad gastronomica recibida
  en auto, a pie o en bicicleta sin que el subte lo registre.
- **No identifica locales que cerraron.** Las fuentes de stock por comuna son dos fotos y de operativos
  distintos, no comparables entre si (ver seccion 4 de fuentes-externas).
- **No aisla la causa.** El corrimiento de patron es compatible con trabajo hibrido, pero tambien con
  cambios de poder adquisitivo, de inseguridad percibida en el microcentro o de oferta de delivery. El
  informe debe nombrar estas alternativas en vez de atribuir todo al trabajo remoto.
- **No cubre 2020 de forma util en terminos de patron**: con el servicio restringido a trabajadores
  esenciales, el perfil horario de ese anio describe la restriccion administrativa, no las
  preferencias de la gente.

## 8. Pendiente

- [ ] Agregado base `fecha x hora x linea x estacion` (en curso)
- [ ] Asignar comuna a cada estacion para el cruce con los datos de locales
- [ ] Calcular los perfiles normalizados y los dos ratios por grupo y anio
- [ ] Definir los subperiodos de comparacion a partir de los quiebres detectados en la serie, en vez
      de fijarlos por calendario
