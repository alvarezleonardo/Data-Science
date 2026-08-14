# Perceptrón AND — Parte 1: Compuerta Lógica AND y primera iteración de entrenamiento

> Conversión a Markdown de la slide del curso. El PDF original está al lado.

## 1. Compuerta lógica AND

La compuerta **AND** es una función lógica de dos entradas (`x1`, `x2`) que devuelve `1` únicamente cuando **ambas entradas son 1**. Es uno de los ejemplos clásicos para mostrar cómo un **perceptrón** puede aprender una función linealmente separable.

Tabla de verdad de `x1 AND x2`:

| x1 | x2 | x1 AND x2 |
|----|----|-----------|
| 0  | 0  | 0         |
| 0  | 1  | 0         |
| 1  | 0  | 0         |
| 1  | 1  | 1         |

Graficando los cuatro puntos `(x1, x2)` en el plano, se observa que la clase **1** (el punto `(1,1)`) puede separarse de la clase **0** (los otros tres puntos) mediante una **línea recta**: la función AND es **linealmente separable**, condición necesaria para que un perceptrón simple pueda aprenderla.

El perceptrón, durante el entrenamiento, va ajustando los pesos de la recta de separación. Cada línea de la segunda imagen del PDF representa un intento de frontera de decisión en distintas iteraciones, acercándose progresivamente a una que separe correctamente el punto `(1,1)` del resto.

## 2. Estructura del perceptrón usado en el ejemplo

El perceptrón tiene:

- Dos entradas: `x1`, `x2`.
- Dos **pesos**: `w1`, `w2`.
- Un **umbral (bias)**: `θ` (theta).
- Una **suma ponderada**: `z = w1*x1 + w2*x2 + θ`.
- Una **función de activación escalón** `φ`, que devuelve `1` si `z` supera cierto punto de corte y `0` en caso contrario.

Parámetros iniciales del ejemplo:

- `w1 = 0.381`
- `w2 = 0.245`
- `θ = 0.196`
- Tasa de aprendizaje: `n = 0.045`

## 3. Ejemplo: 1° Iteración (primera pasada sobre las 4 combinaciones)

El ejemplo recorre las **cuatro filas** de la tabla de verdad, una por una, actualizando pesos y umbral después de cada fila cuando la predicción no coincide con el valor real.

Fórmulas utilizadas en cada paso:

```
Suma ponderada:        z = w1*x1 + w2*x2 + θ
Función de activación: y = φ(z)
Actualización de pesos: Δw_i = n * (y_real - y) * x_i
                        w_i_nuevo = w_i_anterior + Δw_i
Actualización de umbral: Δθ = n * (y_real - y)
                          θ_nuevo = θ_anterior + Δθ
```

> **Nota:** el PDF, al ser una exportación de slides con animaciones "build", repite la misma imagen del perceptrón varias veces por cada fila (mostrando primero la suma ponderada, luego la activación, luego la actualización de pesos y luego la del umbral como pasos incrementales). Acá se consolida cada fila en un único bloque de cálculo, sin duplicar las capturas intermedias idénticas.

### Fila 1: x1=0, x2=0 (esperado: 0)

```
Suma ponderada:
z = 0.381*0 + 0.245*0 + 0.196
z = 0.196

Función de activación:
y = φ(0.196) = 1
```

La predicción (`y=1`) **no coincide** con el valor real (`y_real=0`), así que se actualizan pesos y umbral:

```
Actualizar pesos:
Δw1 = 0.045 * (0 - 1) * 0 = 0        →  w1_nuevo = 0.381 + 0 = 0.381
Δw2 = 0.045 * (0 - 1) * 0 = 0        →  w2_nuevo = 0.245 + 0 = 0.245

Actualizar umbral:
Δθ = 0.045 * (0 - 1) = -0.045        →  θ_nuevo = 0.196 + (-0.045) = 0.151
```

Estado tras la fila 1: `w1 = 0.381`, `w2 = 0.245`, `θ = 0.151`.

### Fila 2: x1=0, x2=1 (esperado: 0)

```
Suma ponderada:
z = 0.381*0 + 0.245*1 + 0.151
z = 0.396

Función de activación:
y = φ(0.396) = 1
```

La predicción (`y=1`) tampoco coincide con el valor real (`y_real=0`):

```
Actualizar pesos:
Δw1 = 0.045 * (0 - 1) * 0 = 0        →  w1_nuevo = 0.381 + 0 = 0.381
Δw2 = 0.045 * (0 - 1) * 1 = -0.045   →  w2_nuevo = 0.245 + (-0.045) = 0.2

Actualizar umbral:
Δθ = 0.045 * (0 - 1) = -0.045        →  θ_nuevo = 0.151 + (-0.045) = 0.106
```

Estado tras la fila 2: `w1 = 0.381`, `w2 = 0.2`, `θ = 0.106`.

### Fila 3: x1=1, x2=0 (esperado: 0)

```
Suma ponderada:
z = 0.381*1 + 0.2*0 + 0.106
z = 0.487

Función de activación:
y = φ(0.487) = 1
```

La predicción (`y=1`) tampoco coincide con el valor real (`y_real=0`):

```
Actualizar pesos:
Δw1 = 0.045 * (0 - 1) * 1 = -0.045   →  w1_nuevo = 0.381 + (-0.045) = 0.336
Δw2 = 0.045 * (0 - 1) * 0 = 0        →  w2_nuevo = 0.2 + 0 = 0.2

Actualizar umbral:
Δθ = 0.045 * (0 - 1) = -0.045        →  θ_nuevo = 0.106 + (-0.045) = 0.061
```

Estado tras la fila 3: `w1 = 0.336`, `w2 = 0.2`, `θ = 0.061`.

### Fila 4: x1=1, x2=1 (esperado: 1)

```
Suma ponderada:
z = 0.336*1 + 0.2*1 + 0.061
z = 0.597

Función de activación:
y = φ(0.597) = 1
```

La predicción (`y=1`) **coincide** con el valor real (`y_real=1`), por lo tanto no hay error y los ajustes son nulos:

```
Actualizar pesos:
Δw1 = 0.045 * (1 - 1) * 1 = 0        →  w1_nuevo = 0.336 + 0 = 0.336
Δw2 = 0.045 * (1 - 1) * 1 = 0        →  w2_nuevo = 0.2 + 0 = 0.2

Actualizar umbral:
Δθ = 0.045 * (1 - 1) = 0             →  θ_nuevo = 0.061 + 0 = 0.061
```

Estado final tras la 1° iteración completa (una pasada por las 4 filas): `w1 = 0.336`, `w2 = 0.2`, `θ = 0.061`.

## 4. Cierre

Con estos valores actualizados de pesos y umbral, el perceptrón queda listo para una segunda iteración sobre las cuatro combinaciones de entrada, repitiendo el mismo procedimiento hasta que las predicciones coincidan con los valores reales en todas las filas (frontera de decisión encontrada).

> **Nota:** el material original de esta parte del curso llega hasta acá (cierra con una slide de "Validamos el resultado" y agradecimiento). No incluye una segunda iteración completa ni la verificación final de convergencia; esos pasos corresponderían a una parte posterior del material.
