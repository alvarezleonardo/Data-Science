# Resultados preliminares

> Primeros calculos sobre el agregado horario de 139,9 millones de registros (2014 a junio de 2026).
> Metodologia en [diseno-analisis-flujo.md](diseno-analisis-flujo.md); calidad del dato en
> [problema-fechas.md](problema-fechas.md).
>
> **Comparacion usada:** enero a junio de 2019 contra enero a junio de 2026. Se recortan los mismos
> meses porque 2026 esta publicado solo hasta el 30/06, y comparar un semestre contra un anio completo
> mezclaria el efecto buscado con la estacionalidad.

## 1. La asimetria territorial existe, y es grande

Pasajeros en dias habiles, enero a junio:

| Grupo de estaciones | 2019 | 2026 | Variacion |
|---|---:|---:|---:|
| Microcentro y oficinas | 28.572.547 | 14.434.355 | **−49,5 %** |
| Corredor gastronomico | 22.585.503 | 12.656.781 | −44,0 % |
| Residencial y periferico | 33.560.206 | 22.417.344 | **−33,2 %** |

**El microcentro perdio la mitad de su flujo; las estaciones residenciales, un tercio.** La diferencia
de 16 puntos porcentuales entre los dos extremos es la medida de la redistribucion territorial, y
confirma la hipotesis F1.

## 2. El hallazgo principal: la semana dejo de ser plana

Pasajeros promedio por dia de la semana, microcentro, enero a junio:

| Dia | 2019 | 2026 | Caida | Indice 2026 |
|---|---:|---:|---:|---:|
| Lunes | 222.946 | 102.567 | −54,0 % | 0,866 |
| Martes | 214.745 | 119.868 | **−44,2 %** | 1,012 |
| Miercoles | 240.960 | 124.265 | −48,4 % | 1,050 |
| Jueves | 224.645 | 111.042 | −50,6 % | 0,938 |
| Viernes | 231.015 | 102.203 | **−55,8 %** | 0,863 |
| Sabado | 55.792 | 38.264 | −31,4 % | 0,323 |
| Domingo | 31.015 | 22.983 | −25,9 % | 0,194 |

El indice compara cada dia contra el promedio de martes a jueves **del mismo anio**, de modo que mide
forma y no nivel:

```
2019:  lunes 0,983   viernes 1,019     <- la semana laboral era PLANA
2026:  lunes 0,866   viernes 0,863     <- lunes y viernes quedan 14 % abajo
```

**En 2019 los cinco dias habiles eran equivalentes. En 2026 la semana tiene forma de campana, con
lunes y viernes hundidos.** Esa es la firma del esquema hibrido de tres dias presenciales: el martes es
el dia que menos cayo (−44,2 %) y el viernes el que mas (−55,8 %), con 11,6 puntos de diferencia entre
ellos.

Es el resultado mas solido del trabajo, por tres razones:

1. **Descarta la explicacion alternativa.** Una recesion, una suba de tarifa o una caida de poblacion
   afectarian los cinco dias por igual. Solo un cambio en la organizacion del trabajo castiga lunes y
   viernes y preserva el centro de la semana.
2. **Es independiente del nivel.** El indice esta normalizado contra el propio anio, asi que no lo
   afecta ni la caida general del transporte publico ni el posible cambio de definicion de la medicion
   de diciembre de 2024.
3. **El fin de semana cayo mucho menos** (−26 % y −31 %) **que los dias habiles** (−44 % a −56 %). El
   viaje que desaparecio es el laboral, no el recreativo.

## 3. El mediodia del microcentro cae, pero poco

Participacion de cada franja en el total diario de su propio grupo, dias habiles:

| Grupo | Franja | 2019 | 2026 | Delta |
|---|---|---:|---:|---:|
| Oficinas | Pico manana (7-10) | 7,43 % | 7,18 % | −0,25 pp |
| Oficinas | **Mediodia (12-15)** | 18,77 % | 17,42 % | **−1,35 pp** |
| Oficinas | Pico tarde (17-20) | 37,61 % | 37,66 % | +0,05 pp |
| Oficinas | Noche (20-23) | 8,83 % | 7,23 % | −1,59 pp |
| Residencial | **Mediodia** | 16,60 % | 16,75 % | **+0,15 pp** |
| Residencial | Pico manana | 35,44 % | 33,82 % | −1,63 pp |
| Gastronomico | Pico manana | 21,50 % | 19,23 % | −2,27 pp |
| Gastronomico | Pico tarde | 22,29 % | 24,25 % | +1,96 pp |

**F2 se confirma, pero debil.** El mediodia del microcentro pierde 1,35 puntos de participacion, que
es un movimiento real pero modesto. **F4 se confirma apenas**: el mediodia residencial sube 0,15
puntos, una diferencia demasiado chica para sostener por si sola que el almuerzo se mudo a los
barrios. El par F2 mas F4 apunta en la direccion esperada, pero la evidencia es debil y conviene
decirlo asi.

Dos observaciones que no estaban previstas:

- **El pico de la tarde del microcentro conserva su forma de manera exacta** (37,61 % contra 37,66 %).
  Quien sigue yendo a la oficina viaja igual que en 2019: lo que cambio es **cuanta** gente va, no
  **como** va. El cambio es de cantidad, no de habito, para los que mantienen la presencialidad.
- **En el corredor gastronomico el pico de la manana cae 2,27 puntos y el de la tarde crece 1,96.**
  Esas estaciones despachan menos gente a la manana, lo que es consistente con residentes que dejaron
  de viajar a trabajar.

## 4. Se refuta la hipotesis del corrimiento a la noche

El ratio **noche dividido mediodia**, que mide cuanto pesa la cena contra el almuerzo:

| Grupo | 2019 | 2026 | Variacion |
|---|---:|---:|---:|
| Oficinas | 0,470 | 0,415 | −11,7 % |
| Gastronomico | 0,512 | 0,497 | **−3,0 %** |
| Residencial | 0,295 | 0,273 | −7,3 % |

**La hipotesis F3 no se sostiene.** No hay corrimiento del dia hacia la noche: el ratio **baja** en los
tres grupos, incluso en el corredor gastronomico. La franja nocturna perdio participacion en todas
partes.

Esto no contradice que la actividad gastronomica se haya recuperado (el indice de restaurantes de
IDECBA esta 13 % por encima de 2019). Lo que indica es que **la cena recuperada no llega en subte**:
puede llegar a pie, en auto, en aplicacion de transporte, o concentrarse en zonas sin subte. Tambien
influye que el horario de servicio del subte limita el viaje nocturno de vuelta.

> Es un resultado valioso justamente por ser negativo: se planteo una hipotesis plausible, se la midio
> y los datos la rechazaron. Conviene reportarlo asi en el informe final, en lugar de eliminar la
> hipotesis como si nunca se hubiera formulado.

## 5. Sintesis

De las seis hipotesis, el estado preliminar es:

| | Hipotesis | Estado |
|---|---|---|
| F1 | El microcentro perdio participacion | **Confirmada**, con fuerza (−49,5 % contra −33,2 %) |
| F2 | Cayo el mediodia del microcentro | Confirmada, debil (−1,35 pp) |
| F3 | El corredor gastronomico gano la noche | **Refutada** (el ratio baja 3 %) |
| F4 | Los residenciales retuvieron el mediodia | Confirmada, muy debil (+0,15 pp) |
| F5 | La semana se volvio hibrida | **Confirmada**, con fuerza (lunes y viernes 14 % abajo) |
| F6 | Se desacoplo el subte de la gastronomia | Pendiente de calculo |

La conclusion que los datos sostienen con solidez no es la del enunciado original, y es mas precisa:
**lo que cambio no fue el horario de las comidas sino la cantidad de dias de presencia en la oficina.**
El microcentro no se vacio a la hora del almuerzo: se vacio los lunes y los viernes.

## 6. Pendiente

- [ ] F6: correlacion entre pax mensual y el indice de restaurantes, por subperiodos
- [ ] Cruce por comuna, usando `bocas-de-subte.csv` para asignar comuna a cada estacion
- [ ] Repetir el analisis con 2014-2019 como linea base, en lugar de 2019 solo, para descartar que
      2019 sea un anio atipico
- [ ] Verificar la sensibilidad de los resultados a la clasificacion de estaciones, en particular al
      criterio de excluir los nodos de transferencia
- [ ] Graficos: perfiles horarios superpuestos, serie mensual con los quiebres, mapa por comuna
