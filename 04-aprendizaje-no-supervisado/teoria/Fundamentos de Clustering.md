# Fundamentos de Clustering

> Conversión a Markdown de la slide del curso (Clase 9). El PDF original está al lado.

## 1. ¿Qué es clustering?

Es un conjunto de procesos que agrupan a **individuos no etiquetados** para crear subconjuntos de datos. Cada subconjunto recibe el nombre de **clúster**: una colección de objetos que **guardan similitudes entre ellos**, pero con suficientes características diferenciadoras respecto al resto como para justificar un segmento independiente.

### Análisis de conglomerados

**Objetivo:**
- Identificar grupos relativamente **homogéneos** con base en un conjunto de variables específicas.
- Clasificar las observaciones en grupos donde cada conglomerado sea **homogéneo internamente** y **heterogéneo** respecto de los demás.

### ¿Cómo se agrupan?

El criterio básico es la **distancia**:
- Objetos **cercanos** → mismo clúster.
- Objetos **lejanos** → clústeres diferentes.

Se distingue la variabilidad **intraclúster** (dentro de un grupo) de la **interclúster** (entre grupos). Los algoritmos más usados son **K-means** y **Hierarchical clustering**.

## 2. Tipos de distancias entre individuos

Según el tipo de variables (cuantitativas, cualitativas o rangos) se elige la distancia idónea para obtener la **matriz de distancias** (o proximidades).

- **Distancia Euclídea (Euclidiana):** la distancia **más corta** (recta) entre dos objetos en el espacio de variables. La más apropiada para variables **cuantitativas**. Entre dos puntos hay **una sola** recta posible.
- **Distancia de Manhattan:** distancia siguiendo el **reticulado** de la cuadrícula (movimientos horizontales/verticales). A diferencia de la euclídea, puede haber **varias rutas** con la misma longitud.

## 3. Tipos de agrupamientos (métodos de enlace / linkage)

Se pueden analizar alternativas para distintos números de grupos; según el problema se elige un método de agrupamiento. Los principales:

### Enlace simple (Single Linkage) — vecino más próximo
La distancia entre dos clusters es la **distancia mínima** entre un objeto de un cluster y un objeto del otro. También llamado **Enlace Mínimo (MIN)**.
- **Ventajas:** maneja formas **no elípticas**; bueno para clusters de **diferentes tamaños**.
- **Desventajas:** **sensible al ruido** y a los outliers (puede agrupar mal si hay ruido entre clusters).

### Enlace completo (Complete Linkage) — vecino más lejano
La distancia entre dos clusters es la **distancia máxima** entre un objeto de un cluster y uno del otro. También llamado **Enlace Máximo (MAX)**.
- **Ventajas:** **menos susceptible** al ruido y outliers.
- **Desventajas:** tiende a **romper** los conglomerados grandes; se inclina hacia clusters **globulares**.

### Enlace promedio (Average Linkage) — vinculación intergrupo
La distancia entre dos clusters es la **distancia media** entre cada punto de un cluster y cada punto del otro (grupos de pares no ponderados con media aritmética).
- **Ventajas:** funciona bien separando clusters con ruido.
- **Desventajas:** sesgado hacia conglomerados **globulares**.

### Método del centroide (Centroid Linkage)
La distancia entre dos clusters es la **distancia entre sus centroides** (vectores medios). En cada etapa se combinan los dos con la menor distancia entre centroides.
- **Ventajas:** separa bien con ruido.
- **Desventajas:** sesgado hacia conglomerados **globulares**.

### Método de Ward (varianza mínima)
Se basa en el **aumento del error cuadrático** al fusionar dos conglomerados (similar a la media de grupo si la distancia es al cuadrado).
- **Ventajas:** suele producir **mejores jerarquías**; menos susceptible al ruido y outliers.
- **Desventajas:** sesgado hacia clusters **globulares**.

> Regla práctica: **Ward** y **Complete** tienden a clusters compactos/globulares y son robustos al ruido; **Single** captura formas irregulares pero sufre con el ruido.
