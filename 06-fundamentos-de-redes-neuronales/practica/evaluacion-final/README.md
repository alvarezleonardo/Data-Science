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

La sección de modelos y evaluación queda abierta, para completar con cada consigna del cuestionario.

> Verificado ejecutando la notebook de punta a punta contra un dataset sintético con la misma estructura y columnas que el original: corre sin errores ni advertencias. Los números reales van a salir al correrla con el CSV verdadero.

[← Volver al módulo](../../README.md)
