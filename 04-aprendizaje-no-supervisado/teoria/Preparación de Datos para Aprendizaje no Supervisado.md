# Preparación de Datos para Aprendizaje no Supervisado

> Conversión a Markdown de la slide del curso (Clase 6). El PDF original está al lado.

## 1. Limpieza y preparación de datos

En 2009 **Mike Driscoll** (data scientist y CEO de Metamarkets) popularizó el término **"data munging"** para referirse al arduo proceso de **limpiar, preparar y validar** los datos.

Es la etapa que más tiempo consume en el trabajo de un data scientist. Reparto aproximado del tiempo:

| Tarea | Tiempo |
|-------|:------:|
| Limpiar y organizar los datos | 60% |
| Recolectar los datos | 19% |
| Minar patrones de los datos | 9% |
| Refinar los algoritmos | 5% |
| Construir datos de prueba | ~3% |
| Otros | ~5% |

## 2. Limpieza de datos

Tareas típicas de limpieza:

- **Resolver problemas de formato y asignar los tipos correctos:** p. ej. al pasar de CSV a Pandas una fecha que no se importa bien (`20090609231247` en lugar de `2009-06-09 23:12:47`).
- **Estandarizar categorías:** cuando el sistema de origen no tipifica valores, una misma categoría puede aparecer escrita distinto (ej. `Arg`, `AR`, `Argentina`).
- **Corregir valores erróneos:** un valor numérico o inválido para describir el género, o una edad negativa o mucho mayor a 100.
- **Completar datos faltantes:** los datasets reales traen faltantes (información que se perdió o nunca se recolectó). Al proceso de completarlos se lo llama **imputación**.
- **Organizar correctamente el dataset:** estructurar filas y columnas de la forma más conveniente, aplicando las reglas del **tidy data**.

## 3. ¿Por qué normalizar?

Manejo de cantidades en **diferentes unidades o escalas**. Para **no favorecer a ninguna variable** en particular al calcular la distancia, hay que **estandarizar y deshacerse de las unidades**.

> Clave en aprendizaje no supervisado: los algoritmos (clustering, PCA, etc.) trabajan con **distancias**, así que una variable con rango grande dominaría el cálculo si no se normaliza.

## 4. ¿Qué es un hiperparámetro?

En machine learning, un **hiperparámetro** es un parámetro cuyo valor se usa para **controlar el proceso de aprendizaje**. Por el contrario, los **parámetros** del modelo se derivan a través del entrenamiento.

Se clasifican en:
- **Hiperparámetros de modelo**
- **Hiperparámetros de algoritmo**

Diferentes algoritmos requieren diferentes hiperparámetros; algunos simples (como la regresión de **mínimos cuadrados ordinarios**) no requieren ninguno.
