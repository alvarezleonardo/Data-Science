# Clustering Jerárquico

> Conversión a Markdown de la slide del curso (Clase 10). El PDF original está al lado.

## 1. ¿Qué es el hierarchical clustering?

En minería de datos y estadística, el **clustering jerárquico** (análisis jerárquico de clusters o **HCA**) es un método que busca construir una **jerarquía de clusters**.

Es útil cuando se desea observar el **orden en que se generan** los clusters: permite obtener el **historial de conglomeración** o **dendrograma**.

### Dos enfoques

- **Aglomerativo (ascendente / bottom-up):** cada observación comienza en su **propio conglomerado**, y los pares se **fusionan** a medida que se asciende en la jerarquía. Se parte de tantos clusters como datos y se agrupan hasta terminar en **uno único**.
- **Divisivo (descendente / top-down):** todas las observaciones comienzan en **un solo conglomerado** y se **dividen** recursivamente al descender en la jerarquía, hasta dar origen a un cluster por dato.

> No requiere fijar el número de clusters de antemano (se decide después, cortando el dendrograma), pero puede ser **computacionalmente costoso**.

## 2. Dendrograma

El clustering jerárquico se representa con un **dendrograma**, que muestra en qué **orden** se unieron los clusters y el **grado de proximidad** entre los que se unen.

- Los **nodos hoja** se corresponden con los **elementos individuales**.
- El **nodo raíz** representa el cluster que contiene a **todos** los elementos.
- El resto de nodos son los clusters que se van formando.

### Cómo se construye

Cada fusión ocurre a una **altura** (nivel) que refleja la distancia entre los clusters unidos. Ejemplo de alturas de fusión: `(a,b)` a 1,00; luego con `c` a 2,25; `(e,f)` a 1,50; `(d,(e,f))` a 3,50; etc. Cuanto **más abajo** la fusión, más parecidos los elementos.

### Cómo se genera un agrupamiento (podar el dendrograma)

El dendrograma puede usarse para generar **distintos agrupamientos**:
- Se **selecciona un nivel** y se **poda** el dendrograma, descartando los hijos de los nodos con nivel igual o superior al seleccionado. Los **nodos hoja** del árbol resultante dan el agrupamiento buscado.
- Según el nivel elegido se obtienen clusters **más o menos compactos**.

Ejemplos (con 8 elementos a–h):
- **Nivel = 6:** grupos `(a,b)`, `c`, `d`, `(e,f)`, `g`, `h`.
- **Nivel = 4:** grupos `(a,b,c)`, `(d,e,f)`, `g`, `h`.

### Cómo leer la altura del corte

- La **altura** a la que se fusionan dos clusters indica su **distancia/disimilitud**: alta = clusters distantes; baja = clusters cercanos.
- Un **corte más bajo** → **más** clusters; un **corte más alto** → **menos** clusters.
- Si dos clusters se fusionan en un **nivel bajo**, están **muy relacionados**; si se fusionan en un **nivel alto**, están más **distantes**.

> Práctica asociada: [notebooks/clustering_jerarquico_practica.ipynb](../notebooks/clustering_jerarquico_practica.ipynb) — dendrogramas con `scipy` (`linkage`, `dendrogram`, `fcluster`), variando `k` y los métodos de enlace (single, complete, average, centroid, ward).
