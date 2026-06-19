<!-- Convertido automáticamente desde: Modelos de Regresión.pdf -->

Modelos de
Regresión

01 ¿Qué es una
Regresión?

La regresión es una técnica estadística que se
utiliza para modelar y analizar la relación entre
una variable dependiente (también llamada
variable de respuesta) y una o más variables
independientes (también llamadas variables
predictoras o explicativas). El objetivo principal
de la regresión es entender cómo la variable
dependiente cambia cuando las variables
independientes se modifican y, a partir de esto,
hacer predicciones.

●
Regresión Lineal Simple: Se utiliza cuando se desea modelar
la relación entre dos variables: una variable dependiente y una
variable independiente. La relación se modela con una línea
recta.
|     |     |     | La fórmula es: 𝑦=𝛽 |     |     | +𝛽 𝑥+𝜖 |
| --- | --- | --- | ------------------ | --- | --- | ------ |
0 1
donde 𝑦 es la variable dependiente, 𝑥 es la variable independiente,
| 𝛽    es la intersección con el eje y, 𝛽 |     |     |     |     | es la pendiente de la línea, y 𝜖  |     |
| --------------------------------------- | --- | --- | --- | --- | --------------------------------- | --- |
| 0                                       |     |     |     |     | 1                                 |     |
es el término de error.
●
Regresión Lineal Múltiple: Se utiliza cuando hay más de una
Tipos de
variable independiente. La fórmula se extiende a:
Regresión
|         |         |                                      | 𝑦=𝛽 +𝛽 | 𝑥 +𝛽 | 𝑥 +⋯+𝛽 | 𝑥 +𝜖 |
| ------- | ------- | ------------------------------------ | ------ | ---- | ------ | ---- |
|         |         |                                      | 0      | 1 1  | 2 2    | 𝑛 𝑛  |
| Aquí, 𝑥 | ,𝑥 ,…,𝑥 |    son las variables independientes. |        |      |        |      |
|         | 1 2     | 𝑛                                    |        |      |        |      |

●
Regresión Polinómica: Es una extensión de la regresión lineal que
permite modelar relaciones no lineales entre la variable dependiente
y las independientes al incluir términos polinómicos.
●
Regresión Logística: Utilizada para modelar una variable
dependiente categórica, es decir, cuando la variable de respuesta es
binaria (por ejemplo, éxito/fallo). La fórmula utiliza una función
logística para modelar la probabilidad de uno de los dos posibles
resultados.
Tipos de
Regresión

02 Proceso de Regresión

Definición del modelo
Elegir el tipo de regresión y definir la fórmula que se utilizará.
Estimación de parámetros
Utilizar métodos estadísticos (como el método de los mínimos cuadrados)
para estimar los coeficientes 𝛽.
Evaluación del Modelo
Proceso de
Analizar el ajuste del modelo utilizando métricas como el coeficiente de
Regresión
determinación (R²), el error cuadrático medio (MSE), entre otros.
2020
Predicción
Utilizar el modelo para predecir valores de la variable dependiente basándose
en nuevos datos de las variables independientes.
Subtítulo
Todo valor definido y
actualizado en el contexto
podrá ser usado por los
componentes incluidos
dentro de tal contexto.

¡Muchas gracias!
