# Notas sobre los recursos de R (Módulo 03, Sesión 08)

Este apunte documenta los dos archivos `.Rmd` de la sesión 8 del módulo de Gestión de Proyectos de IA. **No se convirtieron a notebook ni se ejecutaron** — R no forma parte del stack del curso (Python/Jupyter), así que el objetivo acá es explicar qué hacen, con qué herramientas, y dar la equivalencia conceptual con lo que ya se usa en Python en el resto del programa.

Archivos cubiertos:
- `OD_GPIA_ESP_M03_S08_R_para_analisis_estadistico_viz_shinny.Rmd`
- `OD_GPIA_ESP_M03_S08_Caso práctico_uso de Rstudio_chat_gpt.Rmd`

## ¿Qué es un `.Rmd` (R Markdown)?

R Markdown es un formato de documento que mezcla texto en Markdown con bloques de código ejecutable (` ```{r} ... ``` `), pensado para producir reportes reproducibles: al "tejer" (knit) el documento, cada bloque de código corre y su output (tablas, gráficos, texto) se inserta en el documento final (HTML, PDF o Word). Es el equivalente de R a un Jupyter Notebook: donde en Python el .ipynb intercala celdas de código y markdown dentro de RStudio o Jupyter, un .Rmd intercala bloques de R (`chunks`) y texto dentro de RStudio. Ambos dos archivos de este módulo están configurados como `output: html_notebook`, la variante interactiva de RStudio que muestra el resultado de cada bloque a medida que se ejecuta, sin necesidad de tejer todo el documento.

## Qué hace cada archivo

### `R_para_analisis_estadistico_viz_shinny.Rmd`

Recorrido guiado por el dataset `iris` (el "hola mundo" de la estadística, análogo al `iris` de `sklearn.datasets` o al que se usa en pandas/seaborn para practicar EDA):

1. Carga librerías: `tidyverse`, `dplyr`, `ggplot2`, `plotly`.
2. Carga y explora `iris` (`str()`, `summary()`, chequeo de nulos).
3. Renombra columnas a español y **redondea las 4 variables numéricas a enteros** con `as.integer(round(...))`.
4. Capitaliza los nombres de especie con una función custom.
5. Matriz de correlación (`corrplot`) y matriz de pares (`GGally::ggpairs`).
6. Exporta el dataset transformado a CSV y el gráfico a PNG.
7. Visualizaciones interactivas: histograma con facetas por especie (`ggplot2` + `plotly::ggplotly`), boxplot filtrado a una especie, tabla interactiva (`DT::datatable`), mapa (`leaflet`), gráfico interactivo (`ggiraph`).
8. Cierra con una mini app Shiny de ejemplo (histograma con un slider de número de observaciones), sin relación directa con `iris`.

> **Problema metodológico — redondear las medidas de `iris` a enteros destruye casi toda la variabilidad.** Las 4 columnas numéricas de `iris` están en centímetros con un rango angosto (largo de sépalo entre 4,3 y 7,9 cm, ancho de pétalo entre 0,1 y 2,5 cm). Redondear con `as.integer(round(...))` colapsa el ancho de pétalo (0,1–2,5) a apenas 3 valores posibles (0, 1, 2, 3), y el resto de las columnas a un puñado de enteros. Esto no rompe el código, pero para cualquier análisis posterior de correlación o clustering sobre esas columnas redondeadas, se pierde gran parte de la señal que distingue a las tres especies — es el mismo tipo de error que "tratar una variable continua como si tuviera mucha menos resolución de la que realmente tiene". No se explica en el material por qué se redondea.

### `Caso práctico_uso de Rstudio_chat_gpt.Rmd`

Modelo de regresión logística sobre el dataset `tips` (de la librería `reshape2`, propinas en un restaurante) para predecir si la propina va a ser "alta" o "baja", envuelto en una app Shiny interactiva:

1. Instala/carga `shiny`, `reshape2`, `ggplot2`.
2. Crea la variable binaria `HighTip` (1 si la propina está por encima del promedio, 0 si no).
3. Entrena `glm(HighTip ~ total_bill + size + sex + smoker + day + time, family = binomial)` — regresión logística clásica de R.
4. Define una app Shiny con un formulario (monto de la cuenta, tamaño del grupo, sexo, fumador, día, horario) que llama a `predict()` sobre el modelo y muestra la clase predicha más un gráfico de densidad.

El título del archivo dice "uso de Rstudio_chat_gpt" — es, explícitamente, código generado con ayuda de un LLM (coherente con el resto del módulo de Gestión de Proyectos de IA, que usa ChatGPT como herramienta de desarrollo).

## Paquetes de R usados y su rol

| Paquete | Para qué se usa |
|---|---|
| `tidyverse` | Meta-paquete que agrupa `dplyr`, `ggplot2`, `tidyr`, etc. — el "pandas + matplotlib + más" de R |
| `dplyr` | Manipulación de datos (filtrar, agrupar, transformar) — equivalente a los métodos de `pandas.DataFrame` |
| `ggplot2` | Gráficos estáticos declarativos por capas |
| `plotly` (`ggplotly()`) | Convierte un gráfico de `ggplot2` en uno interactivo (zoom, tooltips) — como pasar de `matplotlib` a `plotly.express` |
| `corrplot`, `GGally` | Visualización de matrices de correlación y de pares de variables |
| `DT` | Tablas HTML interactivas (bindeo de la librería JS DataTables) |
| `leaflet` | Mapas interactivos (bindeo de Leaflet.js) |
| `ggiraph` | Vuelve interactivos los gráficos de `ggplot2` (tooltips, hover) |
| `shiny` | Framework para apps web interactivas en R, sin escribir HTML/JS a mano |
| `reshape2` | Utilidades de reshape de datos; acá se usa solo por traer el dataset `tips` |

## ¿Qué es Shiny? (para quien viene de Python)

**Shiny** es un framework de R para construir aplicaciones web interactivas con controles (sliders, selectores, botones) que disparan cómputo en el servidor y actualizan la interfaz en tiempo real — todo en R, sin escribir JavaScript. Tiene dos partes: `ui` (qué se ve: layout, inputs, outputs) y `server` (la lógica: cómo reaccionar a los inputs y qué renderizar). `shinyApp(ui, server)` levanta la app.

Es el análogo directo a **Streamlit** o **Dash** en Python:

| Concepto | Shiny (R) | Streamlit (Python) | Dash (Python) |
|---|---|---|---|
| Definición de la interfaz | `ui <- fluidPage(...)` | Se arma línea a línea con `st.sidebar`, `st.slider`, etc. | `app.layout = html.Div([...])` |
| Lógica reactiva | `server <- function(input, output) {...}` con `eventReactive`/`renderPlot` | El script entero se re-ejecuta de arriba a abajo en cada interacción | Callbacks explícitos con `@app.callback` |
| Arrancar la app | `shinyApp(ui, server)` | `streamlit run app.py` | `app.run_server()` |
| Curva de aprendizaje | Media — hay que pensar en términos de reactividad explícita | Baja — se escribe como un script normal | Media-alta — más control, más boilerplate |

Los dos `.Rmd` de este módulo usan Shiny **dentro** del notebook interactivo de RStudio (no como una app desplegada aparte), algo parecido a usar `ipywidgets` dentro de un Jupyter Notebook para armar controles interactivos sin salir del notebook — de hecho el notebook de RRHH del mismo módulo (`OD_GPIA_ESP_M04_S11...`) usa exactamente `ipywidgets` con ese mismo propósito.

## Equivalencias conceptuales R ↔ Python (stack del curso)

| R | Python (stack del curso) |
|---|---|
| `data.frame` | `pandas.DataFrame` |
| `dplyr` (`filter`, `mutate`, `group_by`) | métodos de `DataFrame` / `groupby` en pandas |
| `ggplot2` (gramática de gráficos por capas) | `matplotlib` / `seaborn`, o `plotnine` si se quiere la misma sintaxis de capas |
| `plotly` en R (`ggplotly`) | `plotly.express` en Python |
| `glm(family = binomial)` | `sklearn.linear_model.LogisticRegression` o `statsmodels.api.Logit` |
| `shiny` | `streamlit` o `dash` |
| `install.packages()` / `library()` | `pip install` / `import` |
| R Markdown (`.Rmd`) | Jupyter Notebook (`.ipynb`) |
| RStudio | Jupyter/VS Code |

## Errores marcados en el material

- **Redondeo de `iris` a enteros** en `R_para_analisis_estadistico_viz_shinny.Rmd` — ver nota metodológica arriba; no está justificado y degrada la resolución de los datos antes de calcular correlaciones.
- Ninguno de los dos archivos fija una semilla (`set.seed()`); no es crítico acá porque ni el `glm` ni las visualizaciones dependen de aleatoriedad, pero es una ausencia notable si se compara con la insistencia en reproducibilidad (`random_state`) que sí aparece en el resto del módulo en Python.
- El archivo de Shiny + `tips` no separa datos de entrenamiento y validación para el modelo de regresión logística — entrena y predice sobre el mismo dataset completo, sin medir performance fuera de muestra en ningún momento del `.Rmd`.
