# Evaluación final — Reseñas de vinos

Trabajo para la evaluación final del módulo, sobre el dataset `winemag-data-130k-v2.csv`.

| Archivo | Contenido |
|---|---|
| [`enunciado.md`](enunciado.md) | Consigna del curso, transcripta |
| [`evaluacion_final_vinos.ipynb`](evaluacion_final_vinos.ipynb) | Resolución |

## Antes de ejecutar

El dataset **no está versionado** (los archivos de datos quedan fuera del repo). Descargalo del repositorio de GitHub del curso y dejalo en:

```
datasets/winemag-data-130k-v2.csv
```

## Estado

La notebook cubre la preparación completa que pide el enunciado:

1. Carga y exploración de nulos.
2. `datos_copy` y descarte de `region_2`, `taster_twitter_handle`, `designation` y `Unnamed: 0`.
3. `data_eliminados` — descarte de nulos, codificación de etiquetas, `X`/`y` con target `country` (clasificación).
4. `data_imputados` — `points` con la mediana, `price` con la media, descarte del resto, codificación, `X`/`y` con target `price` (regresión).
5. División 70/30 con `random_state=17` en ambos conjuntos.

6. **Consignas del cuestionario**, cada una con su enunciado textual, el código que la resuelve y la respuesta.

### Respuestas

| # | Consigna | Respuesta |
|---|---|---|
| 1 | MLP `(6, 12)`, `max_iter=100`, `random_state=17` sobre `data_eliminados` — Accuracy | **0,5856 (58,56%)** |
| 2 | ¿Cuántos clasificó como Argentina? | **Solo a 1** (y esa única predicción fue un error) |
| 3 | País con la precision más alta | **Italy** (0,84) |
| 4 | Diferencia de Accuracy al pasar a `(100, 200)` | **≈ 38%** (37,89 puntos: 58,56% → 96,45%) |
| 5 | F1 más bajo y Recall más alto con `(100, 200)` | **Ninguna de las anteriores** — es Canada (0,5946) y US (0,9898) |
| 6 | R² del `MLPRegressor` sobre `data_imputados` | **Ninguno de los anteriores** — da 0,0415 |
| 7 | Accuracy "del modelo anterior" | **96%** si se refiere al `MLPClassifier` (100, 200) — ver la nota: un regresor no tiene accuracy |

La notebook está **ejecutada con el dataset real** y con los outputs guardados.

### Números que salen de la preparación

| Concepto | Valor |
|---|---|
| Filas del dataset original | 129.971 |
| Mediana de `points` (imputación) | 88,0 |
| Media de `price` (imputación) | 35,3634 |
| `data_eliminados` | 77.267 filas (se descarta el 40,55%) |
| `data_imputados` | 82.847 filas |
| Train / test eliminados | 54.086 / 23.181 |
| Train / test imputados | 57.992 / 24.855 |

> **Hallazgo a tener en cuenta.** El `dropna` no descarta solo filas: baja los países de **43 a 7**. La causa es `region_1`, que solo está poblada para esos 7. Además quedan muy desbalanceados — US aporta 37.255 filas y Canadá 253 —, así que la exactitud sola va a estar dominada por US.

[← Volver al módulo](../../README.md)
