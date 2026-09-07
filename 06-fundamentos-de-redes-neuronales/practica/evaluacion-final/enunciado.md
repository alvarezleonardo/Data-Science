# Evaluación final — Cuestionario sobre el dataset de reseñas de vinos

> Transcripción del enunciado entregado por el curso. Módulo 06 — Fundamentos de redes neuronales.

## Dataset

Se usa **`winemag-data-130k-v2.csv`**, descargable del repositorio de GitHub del curso.

Los datos contienen información sobre reseñas de vinos, incluyendo país de origen, variedad de uva, puntuación y precio.

| Columna | Descripción |
|---|---|
| `country` | País de origen del vino |
| `description` | Descripción de la reseña del vino |
| `designation` | Nombre específico del vino |
| `points` | Puntuación dada al vino |
| `price` | Precio del vino |
| `province` | Provincia o región de origen del vino |
| `region_1` | Región específica dentro de la provincia |
| `region_2` | Subregión dentro de la región específica |
| `variety` | Variedad de uva |
| `winery` | Nombre de la bodega que produce el vino |

## Preparación previa

Asumiendo que en `datos` está almacenado el DataFrame original, sin alteraciones, generar una copia:

```python
datos_copy = datos.copy(deep = True)
```

Luego, eliminar las columnas:

- `region_2`
- `taster_twitter_handle`
- `designation`
- `Unnamed: 0`

Después se crean **dos DataFrames nuevos**: `data_eliminados` y `data_imputados`.

### `data_eliminados`

1. Eliminar todos los valores ausentes, considerando las filas.
2. Codificar las variables categóricas mediante una **codificación de etiquetas**.

```python
features = ['price', 'points', 'province', 'region_1','taster_name','variety', 'winery']
target = 'country'

X_eliminados = data_eliminados[features]
y_eliminados = data_eliminados[target]
```

### `data_imputados`

1. Completar con el valor de la **mediana** el campo `points`.
2. Completar con el valor de la **media** el campo `price`.
3. Eliminar todos los valores ausentes, considerando las filas.
4. Codificar las variables categóricas mediante una **codificación de etiquetas**.

```python
features = ['country', 'points', 'province', 'region_1','taster_name','variety', 'winery']
target = 'price'

X_imputados = data_imputados[features]
y_imputados = data_imputados[target]
```

## División del conjunto de datos

Dividir en **70/30**, con **semilla de aleatoriedad igual a 17**.

## Nota sobre el alcance

A medida que se avance en el cuestionario se pedirán más cosas, como creación y evaluación de modelos, y en base a ello resolver las distintas consignas.
