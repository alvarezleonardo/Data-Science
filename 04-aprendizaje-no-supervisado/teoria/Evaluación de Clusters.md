# Evaluación de Clusters

> Conversión a Markdown de la slide del curso (Clase 12). El PDF original está al lado.

## Elegir el `k` óptimo

La elección del hiperparámetro `k` es **vital**: indica de qué forma se busca la estructura interna de los datos. Como debe explicitarse en la llamada al algoritmo, es un valor que hay que determinar por **métodos externos a KMeans**. Dos métodos clásicos: la **curva del codo** y el **método de la silueta**.

## 1. Elbow Curve (curva del codo)

Es la representación más utilizada: una curva que expresa la **inercia** del agrupamiento en función del número de clusters. Hay que buscar el **"codo" de la función** (el cambio brusco): a partir de ese `k`, la inercia se **estabiliza** e indica los mejores valores de `k`. A veces el codo no es evidente, así que se prueban varios `k`.

**WCSS / inercia:**
- **WCSS** (Within-Cluster Sum of Squares) = suma de distancias cuadradas de cada punto a su centroide.
- A medida que se agregan centroides, el WCSS **disminuye** (cada punto queda más cerca de un centroide).
- ⚠️ Caso extremo: si hay **tantos centroides como puntos** (un cluster por dato), WCSS = 0, pero eso es **sobre-agrupar** (overfitting): el modelo ni siquiera está agrupando.

**Cómo leerla:** se busca el punto donde ocurre el **último cambio sustancial** y a partir del cual las reducciones de WCSS son graduales/insignificantes. Ese punto (el "codo") da el equilibrio entre número de clusters y compacidad. No siempre es obvio: el data scientist decide usándolo como pista.

## 2. Método de la silueta (Silhouette)

Es una medida similar a la inercia que mide la variación **intracluster**, pero además mide **qué tan seguro** es que un punto pertenezca a su cluster. Así se identifican puntos **mal asignados** o que caen **entre dos clusters**.

Procedimiento: recorrer todos los posibles valores de `k`, calcular el **coeficiente de Silhouette** y quedarse con el mejor agrupamiento. **Cuanto más alto el coeficiente, mejor el agrupamiento.**

**Fórmula** (por punto), con dos distancias:
- **`a`** = distancia media **intra-clúster** (a los puntos de su propio grupo).
- **`b`** = distancia media a las observaciones del **clúster más cercano**.

```
S = (b − a) / max(a, b)
```

Varía entre **-1 y 1**: cercano a 1 → bien agrupado; cercano a -1 → mal agrupado.

**Gráficos de silueta:** en el eje X el coeficiente de silueta y en el Y el número del clúster; una **línea vertical discontinua** marca el coeficiente **medio** de todos los clústeres. Se compara el promedio entre distintos `k` y se elige el más alto. Cada barra muestra además el **tamaño** del cluster (nº de datos). Ejemplo del curso: `k=3` da el mejor coeficiente medio (0.662) → **mejor valor de K = 3**.
