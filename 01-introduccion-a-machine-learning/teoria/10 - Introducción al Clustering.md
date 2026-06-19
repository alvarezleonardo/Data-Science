<!-- Convertido automáticamente desde: 10 - Introducción al Clustering.pdf -->

Introducción al
Clustering

01 Clustering

Clustering
Es el proceso de organizar los objetos en grupos cuyos miembros son similares y los
grupos distintos entre sí. Llamamos cluster a la colección de datos similares entre
ellos.

¿Cuál es la diferencia entre Clustering y
Clasificación ?
En la clasificación, estamos agrupando los datos de acuerdo con un conjunto de grupos
predefinidos por la clase. (Supervisado). En clustering agrupamos los datos con características
similares. En otras palabras, tenemos que descubrir las propias clases. (No Supervisado)
Clustering Classification
● Los datos no están etiquetados. ● Puntos de datos etiquetados
● Agrupa puntos que están "cercanos" entre sí. ● Quiere una "regla" que asigne etiquetas a nuevos puntos.
● Identificar estructuras o patrones en los datos. ● Aprendizaje Supervisado
● Aprendizaje No Supervisado.

Aplicaciones
//
➔ Retail: Se utiliza para agrupar clientes según su comportamiento
de compra, preferencias de productos, etc.
➔ Salud: Se puede aplicar clustering para agrupar perfiles de
pacientes según características clínicas, historial médico, etc.
También se puede utilizar para agrupar genes o proteínas según
sus funciones, lo que puede ayudar en la identificación de
patrones genéticos.
➔ Banca: Se puede hacer análisis de transacciones para detectar
patrones de comportamiento anómalos.

Como el objetivo del clustering es encontrar grupos de datos similares entre sí, los resultados dependen de cómo medimos
dicha similaridad.
En clustering, la distancia es la medida más común. La usamos en los algoritmos K-means y Clustering Jerárquico.
Otra forma de agrupar es determinando cómo se distribuyen las observaciones en el espacio de dimensiones. Lo llamamos
estimación de la densidad.

En los algoritmos basados en distancia, se deben agrupar las observaciones tales que:
➔ Las distancias internas deberían ser pequeñas. Dentro del cluster.
➔ Las distancias externas deberían ser grandes. Entre los clusters.

02 ¿Qué es K-Means?

K-means
//
K-means es uno de los algoritmos más conocidos para identificar clusters, por su
simplicidad y eficiencia.
● Simplicidad: fácil de entender y de implementar.
● Eficiencia: la complejidad en tiempo de cómputo es O(tkn), donde n es el tamaño
de la muestra, k el número de clusters y t es el número de iteraciones.
○ Como k y t son pequeños valores, se considera un algoritmo lineal.
El número de clusters es fijado a priori y los clusters son disjuntos.
El criterio para agrupar los datos es por su cercanía a ciertos puntos en el espacio
llamados centroides

K-means
//
Este metodo, como los de particiones en general, asume que el numero final de
clusters (k) es conocido.
El procedimiento conlleva los siguientes pasos:
1. Determinar los centroides iniciales o semillas de los k grupos.
2. Asignar cada observacion al centroide más cercano segun la distancia.
3. Con las observaciones clasificadas se recalculan los centroides de los k grupos.
Si las distancias entre los nuevos y viejos centroides es mayor que un criterio de
convergencia establecido, se vuelve al paso 2.
4. El proceso termina cuando se cumple el criterio de convergencia. Además, se
fija un número de iteraciones que no debe superarse.

K-means - Ejemplo - Parte 1
//
Paso 1 Paso 2 Paso 3
Se generan centroides Se asignar cada observacion al centroide Con las observaciones clasificadas se
aleatorios, en este más cercano segun la distancia recalculan los centroides de los k grupos.
caso 4

K-means - Ejemplo - Parte 2
//
Paso 2 Paso 3
Se asignar cada observacion al centroide Con las observaciones clasificadas se
más cercano segun la distancia recalculan los centroides de los k grupos.

K-means - Ejemplo - Parte 3
//
|                |  X1         |          | X2  |
| -------------- | ----------- | -------- | --- |
| Centro_0       |  0.929175   | 4.290231 |     |
| Centro_1       |  2.019427   | 0.874748 |     |
| Centro_2       | -1.498074   | 2.901649 |     |
| Centro_3       | -1.272839   | 7.846515 |     |
Paso 4
El criterio de convergencia se cumplió,
termina el proceso y se muestran los 4
centroides finales.
Paso 2 Paso 3
Se asignar cada observacion al centroide  Con las observaciones clasificadas se
más cercano segun la distancia recalculan los centroides de los k grupos.

K-means
El algoritmo tiene como objetivo minimizar la función objetivo:
Donde 𝑐⃗ 𝑗 es el centroide del cluster 𝑆𝑗
● Se busca minimizar la función J que es la suma de las distancias al cuadrado de
los puntos al centroide de su cluster.
● K-means no necesariamente selecciona el mínimo de la función objetivo, ya que
es sensible a la selección inicial de los centroides.

Cómo evaluamos si el valor K cantidad de clusters es el óptimo?
Regla del codo (inercia):
○ A medida que aumenta el número de clusters, la suma de distancias cuadráticas se achica.
○ Evidentemente, si k fuera igual a n, habría un centroide para cada punto y las distancias
serían todas iguales a cero.
○ Pero podemos aplicar un criterio para elegir el k mirando este gráfico. Lo llamamos la regla del codo
(Elbow method).
Se elige el k a partir del cual la curva se aplana, "donde se encuentra el codo".
Número óptimo
de clusters

Cómo evaluamos si el valor K cantidad de clusters es el óptimo?
Silhouette score:
○ El coeficiente Silhouette mide cuán cercano es un punto al resto de los que están en su mismo
cluster, en relación a cuán cercano es a los puntos del cluster más próximo.
○ Es decir que para cada punto 𝑖 , se tiene un coeficiente 𝑠𝑖
en donde 𝑎(𝑖) es la distancia promedio a los otros puntos del mismo cluster y 𝑏(𝑖) es la distancia
promedio a todos los puntos del cluster vecino más cercano.
De esta manera

Cómo evaluamos si el valor K cantidad de clusters es el óptimo?
Silhouette score:
● Si
● 𝑠 es cercano a 1, significa que el punto 𝑖 está correctamente etiquetado
● 𝑠 cercano a cero indica que estaría igual de bien (o mal) etiquetarlo con el
cluster vecino más cercano
● 𝑠 cercano a -1 significa que el punto 𝑖 está mal etiquetado y pertenece al
cluster vecino.
Para obtener un score del clustering promediamos los coeficientes:
en donde 𝑠¯𝑗 es el coeficiente promedio de todos los puntos dentro del cluster j-ésimo.

Cómo evaluamos si el valor K cantidad de clusters es el óptimo?
Número óptimo
de clusters

Conclusiones
El clustering es una herramienta poderosa que nos
permite descubrir patrones ocultos y estructuras
subyacentes en nuestros conjuntos de datos. Hemos
explorado K-means como técnica y métricas de
evaluación que nos ayudan a entender la calidad y
coherencia de los agrupamientos obtenidos.
