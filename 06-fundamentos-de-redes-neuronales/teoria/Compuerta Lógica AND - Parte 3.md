# Perceptrón AND — Parte 3: Iteraciones 4 y 5 del entrenamiento

> Conversión a Markdown de la slide del curso. El PDF original está al lado.

> **Nota:** el PDF exportado incluye, a partir de la slide "¡Muchas gracias!", una serie de slides plantilla genéricas (con texto de relleno tipo *Lorem ipsum*, ejemplos de JavaScript, SQL y evaluación de la materia) que no pertenecen al contenido de esta Parte 3 sobre la compuerta AND. Esas slides fueron excluidas de esta conversión por no tener relación con el tema. También se consolidaron en menos pasos las slides que mostraban la misma iteración construida progresivamente (efecto de animación "build"), evitando duplicar el mismo cálculo repetido en páginas consecutivas.

## 1. Contexto: perceptrón para la compuerta AND

Se continúa el entrenamiento manual de un **perceptrón** que aprende la función lógica **AND**, usando la regla de aprendizaje del perceptrón (regla delta).

**Tabla de verdad de la compuerta AND:**

| x1 | x2 | x1 AND x2 |
|----|----|-----------|
| 0  | 0  | 0         |
| 0  | 1  | 0         |
| 1  | 0  | 0         |
| 1  | 1  | 1         |

**Parámetros del modelo al inicio de esta parte:**

- Tasa de aprendizaje: `n = 0.045`
- Pesos: `w1 = 0.246`, `w2 = 0.11`
- Umbral (bias): `θ = -0.164`
- Función de activación: **función escalón** — `y = φ(z)`, donde `φ(z) = 1` si `z >= 0` y `φ(z) = 0` si `z < 0`.

**Fórmulas utilizadas en cada paso:**

- Suma ponderada: `z = w1*x1 + w2*x2 + θ`
- Salida: `y = φ(z)`
- Actualización de pesos: `Δw = n * (d - y) * x`, con `w_nuevo = w + Δw` (donde `d` es la salida esperada según la tabla de verdad)
- Actualización del umbral: `Δθ = n * (d - y)`, con `θ_nuevo = θ + Δθ`

## 2. Cuarta iteración

Se recorren las cuatro combinaciones de la tabla de verdad en orden, actualizando pesos y umbral después de cada una.

**Fila (x1=0, x2=0), esperado 0:**

```
z = 0.246*0 + 0.11*0 - 0.164 = -0.164
y = φ(-0.164) = 0   → coincide con lo esperado (0)

Δw1 = 0.045*(0-0)*0 = 0        → w1_nuevo = 0.246
Δw2 = 0.045*(0-0)*0 = 0        → w2_nuevo = 0.11
Δθ  = 0.045*(0-0)   = 0        → θ_nuevo  = -0.164
```

**Fila (x1=0, x2=1), esperado 0:**

```
z = 0.246*0 + 0.11*1 - 0.164 = -0.054
y = φ(-0.054) = 0   → coincide con lo esperado (0)

Δw1 = 0.045*(0-0)*0 = 0        → w1_nuevo = 0.246
Δw2 = 0.045*(0-0)*1 = 0        → w2_nuevo = 0.11
Δθ  = 0.045*(0-0)   = 0        → θ_nuevo  = -0.164
```

**Fila (x1=1, x2=0), esperado 0:**

```
z = 0.246*1 + 0.11*0 - 0.164 = 0.082
y = φ(0.082) = 1    → NO coincide con lo esperado (0): hay error, se ajustan pesos y umbral

Δw1 = 0.045*(0-1)*1 = -0.045   → w1_nuevo = 0.246 + (-0.045) = 0.201
Δw2 = 0.045*(0-1)*0 = 0        → w2_nuevo = 0.11
Δθ  = 0.045*(0-1)   = -0.045   → θ_nuevo  = -0.164 + (-0.045) = -0.209
```

**Fila (x1=1, x2=1), esperado 1:**

```
z = 0.201*1 + 0.11*1 - 0.209 = 0.102
y = φ(0.102) = 1    → coincide con lo esperado (1)

Δw1 = 0.045*(1-1)*1 = 0        → w1_nuevo = 0.201
Δw2 = 0.045*(1-1)*1 = 0        → w2_nuevo = 0.11
Δθ  = 0.045*(1-1)   = 0        → θ_nuevo  = -0.209
```

**Estado de los parámetros al finalizar la 4° iteración:** `w1 = 0.201`, `w2 = 0.11`, `θ = -0.209`.

## 3. Quinta iteración

Se repite el recorrido de las cuatro filas con los parámetros actualizados en la iteración anterior.

**Fila (x1=0, x2=0), esperado 0:**

```
z = 0.201*0 + 0.11*0 - 0.209 = -0.209
y = φ(-0.209) = 0   → coincide con lo esperado (0), sin ajustes
```

**Fila (x1=0, x2=1), esperado 0:**

```
z = 0.201*0 + 0.11*1 - 0.209 = -0.099
y = φ(-0.099) = 0   → coincide con lo esperado (0), sin ajustes
```

**Fila (x1=1, x2=0), esperado 0:**

```
z = 0.201*1 + 0.11*0 - 0.209 = -0.008
y = φ(-0.008) = 0   → coincide con lo esperado (0), sin ajustes
```

> **Nota:** esta misma fila (x1=1, x2=0) había sido clasificada de forma incorrecta durante la 4° iteración (salida 1 en lugar de 0). Tras el ajuste de pesos y umbral, en la 5° iteración el perceptrón ya la clasifica correctamente.

**Fila (x1=1, x2=1), esperado 1:**

```
z = 0.201*1 + 0.11*1 - 0.209 = 0.102
y = φ(0.102) = 1    → coincide con lo esperado (1), sin ajustes
```

Como en las cuatro filas de esta iteración la salida obtenida coincide con la esperada (`Δw1 = Δw2 = Δθ = 0` en todos los casos), los parámetros no vuelven a modificarse: `w1 = 0.201`, `w2 = 0.11`, `θ = -0.209`. El perceptrón **converge**: clasifica correctamente las cuatro combinaciones de la compuerta AND.

## 4. Validación del resultado

Con los pesos finales (`w1 = 0.201`, `w2 = 0.11`, `θ = -0.209`) el perceptrón reproduce correctamente la tabla de verdad de la compuerta AND para las cuatro combinaciones de entrada, quedando validado el resultado del entrenamiento.
