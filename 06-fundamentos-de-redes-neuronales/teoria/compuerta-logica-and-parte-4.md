# Perceptrón AND — Parte 4: Validación de la convergencia (6.ª iteración)

> Conversión a Markdown de la slide del curso. El PDF original está en [`material/`](material/).

> **Nota:** el PDF de origen es un deck de slides con animaciones tipo "build": la slide "Ejemplo: 6° Iteración" se repite 4 veces, agregando en cada página una fila más de la verificación del error. Se consolidó el contenido en una sola sección con las 4 filas, evitando duplicar la tabla y las fórmulas que se repiten idénticas en cada página.

## 1. Contexto: última iteración del entrenamiento

Esta parte retoma el ejemplo del **Perceptrón: Ejemplo Compuerta AND**, en la **6.ª iteración** del entrenamiento. En este punto los parámetros del perceptrón ya fueron ajustados por las iteraciones anteriores y quedaron fijos en:

- `w1 = 0.201`
- `w2 = 0.11`
- `bias = -0.209`
- Tasa de aprendizaje: `n = 0.045`

El objetivo de esta iteración es recorrer las **4 combinaciones** de la tabla de verdad de la compuerta **AND** y verificar si el perceptrón, con estos pesos, ya clasifica correctamente a todas (es decir, si el **error** es `0` en todos los casos).

Tabla de verdad de la compuerta AND:

| x1 | x2 | x1 AND x2 |
|----|----|-----------|
| 0  | 0  | 0         |
| 0  | 1  | 0         |
| 1  | 0  | 0         |
| 1  | 1  | 1         |

Fórmulas utilizadas en cada paso:

- Suma ponderada: `z = w1*x1 + w2*x2 + bias`
- Función de activación (escalón): `y = φ(z)`
- Error: `e = y_real - y`

## 2. Verificación fila por fila

### Fila 1: x1 = 0, x2 = 0

```
z = 0.201*0 + 0.11*0 - 0.209
z = -0.209

y = φ(-0.209) = 0

e = y_real - y = 0 - 0 = 0
```

### Fila 2: x1 = 0, x2 = 1

```
z = 0.201*0 + 0.11*1 - 0.209
z = -0.099

y = φ(-0.099) = 0

e = y_real - y = 0 - 0 = 0
```

### Fila 3: x1 = 1, x2 = 0

```
z = 0.201*1 + 0.11*0 - 0.209
z = -0.008

y = φ(-0.008) = 0

e = y_real - y = 0 - 0 = 0
```

### Fila 4: x1 = 1, x2 = 1

```
z = 0.201*1 + 0.11*1 - 0.209
z = 0.102

y = φ(0.102) = 1

e = y_real - y = 1 - 1 = 0
```

## 3. Resultado: el perceptrón convergió

Con los pesos `w1 = 0.201`, `w2 = 0.11` y `bias = -0.209`, el perceptrón clasifica correctamente las **4 combinaciones** de la compuerta AND, obteniendo **error `0` en todos los casos**.

Como no hay error en ninguna fila, no es necesario actualizar los pesos: el entrenamiento **converge** en esta 6.ª iteración y el modelo queda validado como una correcta representación de la función lógica AND.
