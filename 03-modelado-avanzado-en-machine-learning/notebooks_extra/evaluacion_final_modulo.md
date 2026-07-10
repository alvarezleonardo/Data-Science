# Evaluación Final — Módulo 03 (consignas)

Para el desarrollo de este último cuestionario usaremos el dataset **`housing.csv`**
del Censo de California, disponible en el repositorio (`../datasets/housing.csv`).

Los datos contienen información del censo de California de 1990. Refieren a las casas
que se encuentran en un determinado distrito de California y algunas estadísticas
resumidas. **Los datos no están depurados**, por lo que se requieren algunos pasos de
preprocesamiento.

## Columnas

| Español | Nombre en el dataset |
|---------|----------------------|
| longitud | `longitude` |
| latitud | `latitude` |
| edad media vivienda | `housing_median_age` |
| total habitaciones | `total_rooms` |
| total dormitorios | `total_bedrooms` |
| población | `population` |
| hogares | `households` |
| ingresos medianos | `median_income` |
| valor medio de la vivienda | `median_house_value` |
| proximidad al océano | `ocean_proximity` |

## ¿Qué es necesario hacer antes de resolver el cuestionario?

1. **Rellenar los valores faltantes** con la media de la variable.
2. **Estandarizar** las variables numéricas con `StandardScaler()`. Recordar dropear la
   variable categórica.
3. **Codificar** la variable categórica con `OneHotEncoder()`.
4. **Unificar** los datos en un único dataframe distinto del original.

## ¿Cómo distribuir los datos para entrenar y testear?

Dividir el dataset con un **20% de elementos para el testeo**, usando una **semilla de
valor 42** para la aleatoriedad.

## Nota

A medida que se avance en el cuestionario se pedirá realizar más cosas (creación y
evaluación de modelos), y en base a ello resolver las distintas consignas.

## Cuestionario

### Pregunta 1 — Árbol de decisión

Crear un árbol de decisión que permita realizar una regresión con como máximo **5 de
profundidad** (`max_depth=5`) y que el número mínimo de muestras necesarias para
dividir un nodo interno sea de **20** (`min_samples_split=20`), con **semilla de
aleatoriedad 20** (`random_state=20`). Luego de entrenar y evaluar el modelo, ¿cuál es
el valor de **R²**?

**Respuesta:** R² ≈ **0.6098** (0.60975).

### Pregunta 2 — Duplicar la profundidad

Si se duplica la profundidad del árbol creado en el punto anterior (`max_depth=10`), el
modelo: ¿empeoraría, no tendría cambios o mejoraría?

**Respuesta: Mejoraría.** El R² sube de **0.6098** (`max_depth=5`) a **0.7193**
(`max_depth=10`). Con más profundidad el árbol captura relaciones más finas y mejora la
performance en test (todavía sin sobreajustar respecto de test en este caso).

### Pregunta 3 — Regresión lineal (por defecto)

Crear un regresor lineal (`LinearRegression()`) con valores por defecto para aproximar el
valor de la vivienda. Luego de entrenar y evaluar, el R² es:
- Mayor a 70%
- Menor al 35%
- Entre el 40% y 50%
- Ninguno de los anteriores

**Respuesta: Ninguno de los anteriores.** R² ≈ **0.6257** (62.57%), que no cae en ninguno
de los rangos ofrecidos.

### Pregunta 4 — Término independiente

Luego de crear el regresor lineal del punto anterior, el término independiente
(intercepto) del modelo es aproximado a:
- 0.27
- 0.32
- 0.21
- Ninguno de los anteriores

**Respuesta: 0.27.** El intercepto es ≈ **0.2734**. (Esto asume el preprocesamiento de la
consigna, que estandariza **todas** las variables numéricas incluido el target; por eso el
intercepto es un número chico. Si el target quedara sin escalar, el intercepto sería
≈ 238.401 y la respuesta "Ninguno".)

### Pregunta 5 — ¿El coeficiente más grande es mayor a 1?

El coeficiente más grande del regresor lineal del punto anterior, ¿es mayor a 1?
- Verdadero
- Falso

**Respuesta: Verdadero.** El coeficiente más grande es el de `ocean_proximity_ISLAND` ≈
**1.0156**, que es mayor a 1 (el resto está por debajo; le sigue `median_income` ≈ 0.65).

### Pregunta 6 — SVR (por defecto)

Crear un `SVR()` con valores por defecto. Luego de entrenar y testear, hasta este punto,
una máquina de Soporte Vectorial es la que:
- Peor resultado da.
- Da igual que un árbol de decisión.
- No se puede utilizar una máquina de Soporte Vectorial.
- Mejor resultado da.

**Respuesta: Mejor resultado da.** El SVR por defecto da **R² ≈ 0.7502**, el más alto de
todos los modelos probados hasta ahora:

| Modelo | R² |
|--------|-----|
| **SVR (default)** | **0.7502** |
| Árbol (`max_depth=10`) | 0.7193 |
| Regresión lineal | 0.6257 |
| Árbol (`max_depth=5`) | 0.6098 |
