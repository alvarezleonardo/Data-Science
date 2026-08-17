# Perceptrón AND — Parte 2: Iteraciones de Entrenamiento (2ª y 3ª Pasada)

> Conversión a Markdown de la slide del curso. El PDF original está al lado.

## 1. Punto de partida

Esta parte continúa el entrenamiento del **perceptrón** para aprender la **compuerta lógica AND**, retomando los pesos y el umbral obtenidos al cierre de la iteración anterior:

- `w1 = 0.336`
- `w2 = 0.2`
- `θ (umbral) = 0.061`
- Tasa de aprendizaje: `n = 0.045`

La tabla de verdad de referencia (la misma para todo el entrenamiento) es:

| x1 | x2 | x1 AND x2 |
|----|----|-----------|
| 0  | 0  | 0         |
| 0  | 1  | 0         |
| 1  | 0  | 0         |
| 1  | 1  | 1         |

El procedimiento se repite para cada fila de la tabla:

1. Calcular la **suma ponderada** `z`.
2. Aplicar la **función de activación** (función escalón) para obtener `y`.
3. Comparar `y` contra el valor esperado y **actualizar los pesos**.
4. **Actualizar el umbral**.

## 2. Segunda iteración

### 2.1 Fila (x1=0, x2=0) — esperado 0

Pesos y umbral de entrada: `w1 = 0.336`, `w2 = 0.2`, `θ = 0.061`.

```
Suma ponderada:
z = 0.336*0 + 0.2*0 + 0.061
z = 0.061

Función de activación:
y = phi(0.061) = 1   (esperado: 0 → error)

Actualizar pesos:
Δw1 = 0.045 * (0 - 1) * 0 = 0        → w1_nuevo = 0.336 + 0 = 0.336
Δw2 = 0.045 * (0 - 1) * 0 = 0        → w2_nuevo = 0.2 + 0 = 0.2

Actualizar umbral:
Δθ = 0.045 * (0 - 1) = -0.045
θ_nuevo = 0.061 + (-0.045) = 0.016
```

### 2.2 Fila (x1=0, x2=1) — esperado 0

Pesos y umbral de entrada: `w1 = 0.336`, `w2 = 0.2`, `θ = 0.016`.

```
Suma ponderada:
z = 0.336*0 + 0.2*1 + 0.016
z = 0.216

Función de activación:
y = phi(0.216) = 1   (esperado: 0 → error)

Actualizar pesos:
Δw1 = 0.045 * (0 - 1) * 0 = 0         → w1_nuevo = 0.336 + 0 = 0.336
Δw2 = 0.045 * (0 - 1) * 1 = -0.045    → w2_nuevo = 0.2 + (-0.045) = 0.155

Actualizar umbral:
Δθ = 0.045 * (0 - 1) = -0.045
θ_nuevo = 0.016 + (-0.045) = -0.029
```

### 2.3 Fila (x1=1, x2=0) — esperado 0

Pesos y umbral de entrada: `w1 = 0.336`, `w2 = 0.155`, `θ = -0.029`.

```
Suma ponderada:
z = 0.336*1 + 0.155*0 + (-0.029)
z = 0.307

Función de activación:
y = phi(0.307) = 1   (esperado: 0 → error)

Actualizar pesos:
Δw1 = 0.045 * (0 - 1) * 1 = -0.045    → w1_nuevo = 0.336 + (-0.045) = 0.291
Δw2 = 0.045 * (0 - 1) * 0 = 0         → w2_nuevo = 0.155 + 0 = 0.155

Actualizar umbral:
Δθ = 0.045 * (0 - 1) = -0.045
θ_nuevo = -0.029 + (-0.045) = -0.074
```

### 2.4 Fila (x1=1, x2=1) — esperado 1

Pesos y umbral de entrada: `w1 = 0.291`, `w2 = 0.155`, `θ = -0.074`.

```
Suma ponderada:
z = 0.291*1 + 0.155*1 + (-0.074)
z = 0.372

Función de activación:
y = phi(0.372) = 1   (esperado: 1 → correcto, sin error)

Actualizar pesos:
Δw1 = 0.045 * (1 - 1) * 1 = 0    → w1_nuevo = 0.291
Δw2 = 0.045 * (1 - 1) * 1 = 0    → w2_nuevo = 0.155

Actualizar umbral:
Δθ = 0.045 * (1 - 1) = 0
θ_nuevo = -0.074 + 0 = -0.074
```

Al cierre de la 2ª iteración, los parámetros quedan en: `w1 = 0.291`, `w2 = 0.155`, `θ = -0.074`.

## 3. Tercera iteración

### 3.1 Fila (x1=0, x2=0) — esperado 0

Pesos y umbral de entrada: `w1 = 0.291`, `w2 = 0.155`, `θ = -0.074`.

```
Suma ponderada:
z = 0.291*0 + 0.155*0 + (-0.074)
z = -0.074

Función de activación:
y = phi(-0.074) = 0   (esperado: 0 → correcto, sin error)

Actualizar pesos:
Δw1 = 0.045 * (0 - 0) * 0 = 0    → w1_nuevo = 0.291
Δw2 = 0.045 * (0 - 0) * 0 = 0    → w2_nuevo = 0.155

Actualizar umbral:
Δθ = 0.045 * (0 - 0) = 0
θ_nuevo = -0.074 + 0 = -0.074
```

### 3.2 Fila (x1=0, x2=1) — esperado 0

Pesos y umbral de entrada: `w1 = 0.291`, `w2 = 0.155`, `θ = -0.074`.

```
Suma ponderada:
z = 0.291*0 + 0.155*1 + (-0.074)
z = 0.081

Función de activación:
y = phi(0.081) = 1   (esperado: 0 → error)

Actualizar pesos:
Δw1 = 0.045 * (0 - 1) * 0 = 0         → w1_nuevo = 0.291
Δw2 = 0.045 * (0 - 1) * 1 = -0.045    → w2_nuevo = 0.155 + (-0.045) = 0.11

Actualizar umbral:
Δθ = 0.045 * (0 - 1) = -0.045
θ_nuevo = -0.074 + (-0.045) = -0.119
```

### 3.3 Fila (x1=1, x2=0) — esperado 0

Pesos y umbral de entrada: `w1 = 0.291`, `w2 = 0.11`, `θ = -0.119`.

```
Suma ponderada:
z = 0.291*1 + 0.11*0 + (-0.119)
z = 0.172

Función de activación:
y = phi(0.172) = 1   (esperado: 0 → error)

Actualizar pesos:
Δw1 = 0.045 * (0 - 1) * 1 = -0.045    → w1_nuevo = 0.291 + (-0.045) = 0.246
Δw2 = 0.045 * (0 - 1) * 0 = 0         → w2_nuevo = 0.11

Actualizar umbral:
Δθ = 0.045 * (0 - 1) = -0.045
θ_nuevo = -0.119 + (-0.045) = -0.164
```

### 3.4 Fila (x1=1, x2=1) — esperado 1

Pesos y umbral de entrada: `w1 = 0.246`, `w2 = 0.11`, `θ = -0.164`.

```
Suma ponderada:
z = 0.246*1 + 0.11*1 + (-0.164)
z = 0.192

Función de activación:
y = phi(0.192) = 1   (esperado: 1 → correcto, sin error)

Actualizar pesos:
Δw1 = 0.045 * (1 - 1) * 1 = 0    → w1_nuevo = 0.246
Δw2 = 0.045 * (1 - 1) * 1 = 0    → w2_nuevo = 0.11

Actualizar umbral:
Δθ = 0.045 * (1 - 1) = 0
θ_nuevo = -0.164 + 0 = -0.164
```

## 4. Resultado

Al cierre de la 3ª iteración, los parámetros del perceptrón quedan en: `w1 = 0.246`, `w2 = 0.11`, `θ = -0.164`.

Con estos valores el modelo predice correctamente las cuatro combinaciones de la compuerta AND: `y = phi(w1*x1 + w2*x2 + θ)` da 0 para (0,0), (0,1) y (1,0), y 1 para (1,1). El proceso de entrenamiento se detiene aquí porque ya no hay error de clasificación en ninguna fila de la tabla de verdad.
