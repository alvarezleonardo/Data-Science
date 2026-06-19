<!-- Convertido automáticamente desde: 9 - Aprendizaje No supervisado.pdf -->

Introducción al
Aprendizaje No
Supervisado

Índice

01

02
03

¿Qué es el Aprendizaje No
Supervisado?
Clustering
Reducción de la dimensionalidad

01

¿Qué es el Aprendizaje
No Supervisado?

El aprendizaje no
supervisado es como
explorar un mapa sin
un destino
predeterminado,
buscando entender la
topografía y los
puntos de interés por
nosotros mismos.

Aprendizaje No Supervisado

➔ En el aprendizaje no supervisado del campo
del Machine Learning, nos adentramos en la
exploración y descubrimiento de patrones
ocultos en los datos sin la ayuda de
respuestas predeﬁnidas o etiquetas
especíﬁcas.

➔ A diferencia del aprendizaje supervisado,
donde se nos proporcionan ejemplos de
entrada y las salidas correspondientes que
queremos que el modelo aprenda, en el
aprendizaje no supervisado nos
enfrentamos a conjuntos de datos que no
tienen etiquetas claras. En lugar de eso,
nuestro objetivo es encontrar patrones
interesantes o agrupar los datos en
categorías sin tener una guía explícita.

Clusterización
Reducción de la Dimensión

Aprendizaje No Supervisado

Columnas

Característica 1 Característica 2 Característica 3

…

Característica N

Filas
Registros
Rows

Características o Features

02

Clustering

Clustering

● El clustering es una técnica fundamental en el aprendizaje no supervisado, que se utiliza
para organizar un conjunto de datos en grupos basados en las similitudes entre las
observaciones. Este enfoque es especialmente útil cuando no se tiene información previa
sobre las categorías a las que pertenecen los datos.

○ El objetivo del clustering es agrupar elementos similares en un mismo grupo y separar

aquellos que son diferentes en distintos grupos.

● El proceso de clustering implica la identiﬁcación de patrones y estructuras ocultas dentro de
los datos, permitiendo a los analistas descubrir relaciones y tendencias que no son claros a
simple vista. Esta técnica se aplica en diversas áreas, como el análisis de mercado, la
biología, la medicina, la detección de fraudes, y muchas más.

○ Por ejemplo, en el análisis de mercado, el clustering puede ayudar a segmentar a los

clientes en diferentes grupos basados en su comportamiento de compra, permitiendo
estrategias de marketing más personalizadas.

//

Algoritmos de clustering

● K-Means: Este es uno de los algoritmos más populares y ampliamente utilizados.

○ El algoritmo comienza seleccionando 𝐾 centroides aleatorios, que representan

el centro de cada grupo.

○ Luego, asigna cada punto de datos al centroide más cercano y recalculan los

centroides como el promedio de los puntos asignados a cada grupo.

○ Este proceso se repite iterativamente hasta que los centroides ya no cambian

signiﬁcativamente, indicando que los grupos se han estabilizado.

● Clustering jerárquico (Hierarchical Clustering): Este método no requiere la

especiﬁcación previa del número de grupos.

○ El proceso comienza considerando cada punto de datos como su propio grupo y,
progresivamente, fusiona los grupos más cercanos en un proceso jerárquico.

○ El resultado es un árbol de grupos, llamado dendrograma, que muestra cómo se

agrupan los datos en diferentes niveles de similitud.

//

03

Reducción de la
dimensionalidad

Reducción de la dimensionalidad

● La reducción de dimensionalidad (dimensionality reduction) es una técnica empleada

para disminuir la cantidad de características o variables en un conjunto de
datos.

○ El objetivo principal es simpliﬁcar la representación de los datos mientras se

conserva la mayor cantidad posible de información relevante.

● En muchos conjuntos de datos, especialmente aquellos con numerosas

características, puede haber redundancia o correlación entre las variables, lo que
complica el análisis y la visualización.

○ La reducción de dimensionalidad aborda este problema al transformar los datos

originales en un espacio de menor dimensión.

○ En este espacio reducido, las nuevas variables (llamadas componentes

principales o características latentes) representan una combinación de las
variables originales.

//

Análisis de Componentes Principales (PCA)

El Análisis de Componentes Principales (PCA, por sus siglas en inglés)  se utiliza para transformar un conjunto de
datos posiblemente correlacionados en un conjunto de valores no correlacionados llamados componentes principales.
Estos componentes principales son nuevas variables obtenidas a partir de combinaciones lineales de las variables
originales del conjunto de datos.

Ventajas :

● Reducción de Dimensionalidad: Simpliﬁca los datos, reduciendo el número de variables necesarias

para describirlos.

● Mejora del Rendimiento Computacional: Disminuye el tiempo y los recursos necesarios para el

procesamiento de datos.

● Eliminación de Redundancia: Al combinar variables correlacionadas, elimina la redundancia de

información.

Limitaciones:

● Linealidad: Solo captura relaciones lineales entre variables, por lo que puede no ser efectivo para datos

con relaciones no lineales.

● Sensibilidad a la Escala: Las variables deben estar en la misma escala, ya que PCA es sensible a las

magnitudes de las variables.

Conclusiones

El aprendizaje no supervisado es una herramienta

poderosa y versátil en el análisis de datos. Al permitir la

identiﬁcación de patrones y estructuras ocultas sin la

necesidad de datos etiquetados, ofrece una forma

eﬁciente y efectiva de abordar problemas complejos y

explorar datos de manera signiﬁcativa.


