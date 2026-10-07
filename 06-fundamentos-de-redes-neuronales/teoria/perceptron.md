# Perceptrón

> Conversión a Markdown de la slide del curso. El PDF original está en [`material/`](material/).

## 1. ¿Qué es el perceptrón?

El perceptrón es un **algoritmo de aprendizaje supervisado** utilizado para la **clasificación binaria**. Fue propuesto por **Frank Rosenblatt en 1957** y es uno de los **modelos más simples de una red neuronal**.

## 2. Ejemplo: clasificación de imágenes (perro vs. gato)

La slide ilustra el proceso con un ejemplo de clasificación binaria entre imágenes de perro y gato:

- Cada imagen se representa como un vector de **características** (celdas coloreadas), que actúan como entradas del modelo.
- A cada imagen le corresponde una **etiqueta** de clase (perro o gato).
- Al graficar los pares de características de cada imagen en un plano `(x1, x2)`, las observaciones de cada clase tienden a agruparse en regiones distintas del espacio.

## 3. La función lógica AND como caso de estudio

El ejemplo utiliza la **función lógica AND** para mostrar cómo el perceptrón aprende un límite de decisión:

| x1 | x2 | Etiqueta |
|----|----|----------|
| 0  | 0  | 0 |
| 0  | 1  | 0 |
| 1  | 0  | 0 |
| 1  | 1  | 1 |

Solo el par `(1, 1)` pertenece a la clase positiva (etiqueta 1); el resto pertenece a la clase negativa (etiqueta 0).

## 4. Separación lineal de las clases

Al graficar los cuatro puntos de la tabla AND en el plano `x1`-`x2`:

- Los puntos `(0,0)`, `(0,1)` y `(1,0)` (etiqueta 0) quedan de un lado.
- El punto `(1,1)` (etiqueta 1) queda del otro lado.
- El perceptrón encuentra una **recta** (límite de decisión) que separa ambas clases en el plano.

Esto ilustra la idea central del perceptrón: es un **clasificador lineal**, capaz de separar dos clases cuando estas son **linealmente separables** (se puede trazar una única línea recta que las divida por completo).

> **Nota:** la slide original solo desarrolla el ejemplo de la función AND; no incluye en este punto la fórmula matemática del perceptrón (suma ponderada, umbral y función de activación). Esa formalización se desarrolla en el documento complementario "Perceptrón - Estructura y Fórmulas".
