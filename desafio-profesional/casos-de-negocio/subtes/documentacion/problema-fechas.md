# El problema de las fechas: diagnostico y regla de correccion

> Hallazgo central del relevamiento del caso subtes. **Tres de los trece anios de la serie tienen las
> fechas corrompidas**, y el error es del tipo que no se nota: no produce excepciones al parsear,
> produce numeros plausibles pero equivocados. Afecta 2020, 2021 y agosto de 2025.
> Relacionado: [serie-extendida-2022-2026.md](serie-extendida-2022-2026.md),
> [diccionario-estaciones.md](diccionario-estaciones.md).

## 1. Por que importa

La fecha es la variable central del analisis: de ella salen el dia de la semana, el mes, los
subperiodos pre y post pandemia y toda la serie temporal. Un error de fechas no degrada el resultado
de forma visible, lo **invierte**.

El sintoma concreto en 2025, leyendo con dia primero:

```
2025-07  19.936.627
2025-08  13.563.219   <- caida del 32 %
2025-09  19.713.080   <- recuperacion
```

Esa caida y esa recuperacion **no existieron**. Un modelo de series temporales entrenado sobre esto
aprende un shock inexistente, y un informe que lo reporte estaria afirmando algo falso con total
seguridad aparente. Es exactamente el tipo de error que `success=true` no detecta.

## 2. El diagnostico: un bug de exportacion, siempre el mismo

En los tres casos el patron es identico, y es el bug clasico de las herramientas que interpretan
fechas segun una configuracion regional distinta de la del origen:

> **Las fechas ambiguas quedaron invertidas; las no ambiguas quedaron bien.**

Una fecha es **ambigua** cuando los dos campos son menores o iguales a 12, porque cualquiera de los
dos puede ser el mes (`8/5` puede ser 8 de mayo o 5 de agosto). Es **no ambigua** cuando uno de los
campos supera 12, porque entonces ese campo solo puede ser el dia (`8/20` solo puede ser 20 de
agosto).

La herramienta que exporto estos archivos dio vuelta las primeras y dejo intactas las segundas. El
resultado es un archivo con **dos formatos conviviendo, discriminados por el valor del dia**.

### La evidencia en 2021

Las fechas distintas, en orden de aparicion en el archivo:

```
1/1/2021 ... 31/1/2021       <- enero, formato D/M, correcto
2021-01-02                   <- cambia a ISO
2021-02-02
   ...
2021-12-02                   <- el TERCER campo es constante (02)
2021-02-13                   <- y aca el SEGUNDO campo pasa a ser el constante (02)
2021-02-14
```

El bloque es **febrero**. Las fechas `2021-01-02` a `2021-12-02` son los dias **1 a 12 de febrero**
escritos como `YYYY-DD-MM`, y desde `2021-02-13` son los dias 13 en adelante escritos como
`YYYY-MM-DD`. El corte cae exactamente en el dia 13, que es donde la ambiguedad desaparece.

Un parser ISO estandar lee las primeras como "2 de enero, 2 de febrero, ... 2 de diciembre" sin
emitir un solo error, y devuelve 344 dias distintos en el anio: un numero que parece razonable y esta
mal.

### La evidencia en 2020

Enero y febrero vienen como `DD/MM/YYYY` con ceros, y desde marzo como `M/D/YYYY` sin ceros. Hay
**234.288 filas con los dos campos de dos digitos y el segundo mayor que 12** (por ejemplo
`10/13/2020`, que es el 13 de octubre): esas filas refutan cualquier regla basada en la cantidad de
digitos.

Un parser con dia primero descarta 1.655.090 filas por "fecha invalida" (las que tienen dia mayor a
12) y **se come 10.454.088 pasajeros**, el 13 % del anio. Las ambiguas, en cambio, no fallan: se
invierten en silencio.

### La evidencia en agosto de 2025

283.617 filas en `M/D`, **todas con primer campo igual a 8**, lo que confina el problema a los
archivos de ese mes. Las ambiguas se contabilizan en el mes equivocado: `8/5/2025` leida como dia
primero cae en **mayo**, de modo que el error no solo hunde agosto sino que infla mayo.

## 3. La estructura que permite corregir

Los archivos no son una secuencia cronologica unica: son **bloques contiguos concatenados**, donde
cada bloque es un mes de un grupo de lineas. Se verifica mirando la linea de subte cuando la fecha
"se reinicia":

```
fila      2: 01/01/2020  LineaA  Acoyte      <- bloque enero, grupo ABC
fila 550101: 01/01/2020  LineaD  9 de julio  <- bloque enero, grupo DEH
```

Despues de enero del grupo ABC viene enero del grupo DEH, por eso la fecha vuelve al dia 1. Esto
tambien explica por que el rango de fechas no se puede deducir de la primera y la ultima fila.

**La propiedad clave: dentro de un bloque, el mes es constante.** Y como todo mes tiene dias mayores
a 12, **todo bloque contiene fechas no ambiguas**, que revelan cual es su mes sin ninguna duda.

## 4. La regla de correccion

```
Pasada 1: para cada bloque, deducir el mes a partir de sus fechas NO ambiguas
          (aquellas donde un campo supera 12; ese campo es el dia, el otro el mes).

Pasada 2: para cada fila:
            - no ambigua -> dia y mes se leen directamente
            - ambigua    -> el mes es el del bloque,
                            y el dia es el otro campo
```

Casos, en un bloque de agosto:

| Fecha cruda | Campo que vale el mes | Dia |
|---|---|---|
| `8/20/2025` | el primero | 20 |
| `20/8/2025` | el segundo | 20 |
| `8/5/2025` | el primero | 5 |
| `5/8/2025` | el segundo | 5 |
| `8/8/2025` | ambos | 8 |

La regla cubre el 100 % de los casos sin ambiguedad residual, y sirve igual para las tres formas
corrompidas (`DD/MM`, `M/D` e ISO invertido) porque no depende del formato declarado sino de la
constancia del mes dentro del bloque.

### Reglas que se evaluaron y se descartaron

- **"Si tiene dos digitos en ambos campos, es dia primero".** Refutada por las 234.288 filas de 2020
  del tipo `10/13/2020`, que tienen dos digitos y son mes primero.
- **Usar el mes del nombre del archivo.** Funciona para 2022-2026, donde el origen publica un CSV por
  mes, pero **no para 2020**: el zip oficial de ese anio contiene un unico `historico2.csv` de
  465.212.922 bytes, identico al que vino con el caso. No hay nombre de archivo del cual sacar el mes.
- **Confiar en la columna `periodo`.** Solo trae el anio, no el mes.

## 5. Validacion de la correccion

No alcanza con que el parseo no falle: hay que probar que el resultado es correcto. Dos controles:

1. **Dias contiguos por mes.** Despues de corregir, los dias de cada mes deben formar una secuencia
   que arranque en 1 y no tenga huecos sistematicos. Un mes que empieza en el dia 13, o al que le
   faltan justo los dias 1 a 12, delata que la correccion fallo en ese bloque: son precisamente los
   dias ambiguos.
2. **Suma de pasajeros contra la suma directa de la columna de origen, anio por anio.** Es
   independiente del parseo de fechas, asi que detecta cualquier fila perdida en el camino. Los
   totales de referencia, calculados con `awk` sobre los archivos crudos:

| Anio | Pasajeros | | Anio | Pasajeros |
|---|---:|---|---|---:|
| 2014 | 255.189.625 | | 2018 | 348.400.114 |
| 2015 | 282.519.120 | | 2019 | 341.340.501 |
| 2016 | 314.418.191 | | **2020** | **77.504.971** |
| 2017 | 328.701.725 | | **2021** | **96.077.158** |

La caida de 2019 a 2020 es de **341,3 a 77,5 millones, un 77 %**, y es el dato mas contundente de todo
el caso. Conviene notar que se obtiene de la suma directa de la columna de pasajeros, que no depende
de las fechas: es robusto al problema descripto en este documento.

## 6. Leccion metodologica

Este hallazgo justifica dos decisiones del diseno:

- **Validar contenido y no codigo de retorno.** Las tres corrupciones pasan silenciosamente por
  cualquier parser razonable. La unica forma de detectarlas fue mirar las fechas distintas en orden de
  aparicion y cruzar los totales.
- **Comparar formas de perfil en vez de niveles** (ver
  [diseno-analisis-flujo.md](diseno-analisis-flujo.md)). Un error de fechas desplaza pasajeros entre
  dias y meses, pero no altera la distribucion por hora del dia, porque la hora esta en otra columna y
  no sufrio el bug. El indicador elegido es el mas robusto de los disponibles frente a este problema.
