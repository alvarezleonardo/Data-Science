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
