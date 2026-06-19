<!-- Convertido automáticamente desde: 8 - Árboles de Decisión.pdf -->

Árboles de Decisión

Índice

01
02
03

¿Qué es un Árbol de Decisión?
CART
¿Cómo decide un árbol donde
ramiﬁcarse?

01

¿Qué es un Árbol de
Decisión?

Árboles

Un árbol es una estructura jerárquica que se utiliza para
representar relaciones entre diferentes elementos. Está
compuesto por nodos y aristas (o ramas) que conectan estos
nodos. Aquí están los elementos clave de un árbol:

➔ Nodo Raíz: Es el nodo superior del árbol, del cual parten
todas las demás ramas. Solo hay uno en cada árbol.

➔ Nodos Internos: Son los nodos que tienen al menos un
nodo hijo. Representan puntos de decisión o categorías
intermedias.

➔ Nodos Hojas: Son los nodos que no tienen nodos hijos.
Representan el ﬁnal de una rama y suelen contener la
información ﬁnal o el resultado.

➔ Ramas: Conectan los nodos entre sí y representan la

relación o el camino entre los nodos.

Árbol de decisión

Es un modelo predictivo que utiliza una estructura de árbol
para tomar decisiones basadas en los valores de los
atributos de los datos.

Aquí cada nodo interno plantea una pregunta sobre un
atributo, las ramas representan las posibles respuestas,
y los nodos hojas representan la decisión ﬁnal o la
predicción basada en el conjunto de respuestas anteriores.

Esta estructura permite desglosar un problema complejo en
una serie de preguntas binarias o categóricas más
simples, facilitando el proceso de toma de decisiones.

02

CART

CART (Classiﬁcation and Regression Trees)

Es un algoritmo de machine learning que se utiliza tanto para problemas de clasiﬁcación como de
regresión.

➔ Inicio en la raíz: El algoritmo comienza en la raíz del árbol y selecciona la mejor característica

(feature) para dividir los datos en grupos (hijos).

➔ Criterio de división: Para problemas de clasiﬁcación, se usan el índice de Gini o la Entropía

para medir la pureza de la división.

➔ División recursiva: La mejor división es la que maximiza la homogeneidad dentro de los grupos
y la heterogeneidad entre ellos. Este proceso se repite recursivamente para cada nodo hijo,
creando el árbol de decisiones.

➔ Criterios de parada: Para evitar que el árbol crezca indeﬁnidamente, se utilizan criterios de

parada como:

● Alcanzar una profundidad máxima del árbol.

● Tener un número mínimo de observaciones en un nodo.

● No lograr una mejora signiﬁcativa en la medida de pureza.

Hiperparámetros

➔ ¿Hasta qué nivel puede crecer el árbol?

➔ ¿Cuántos atributos debemos considerar en un

árbol?

➔ ¿A partir de cuántas observaciones podemos seguir

dividiendo?

➔ ¿Con cuántas observaciones dejamos de

subdividir?

Hiperparámetros

➔ Se divide la población o muestra en conjuntos homegéneos basados en la variable de entrada más

signiﬁcativa

➔ La construcción del árbol sigue un enfoque de división binaria recursiva (top-down greddy

approach). Greedy -> analiza la mejor variable para ramiﬁcación sólo en el proceso de división
actual.

Ventajas

➔ Fácil de entender

➔ Útil en exploración de datos:identiﬁcar importancia

de variables a partir de cientos de variables.

➔ Menos limpieza de datos: outliers y valores faltantes

no inﬂuencian el modelo (A un cierto grado)

➔ El tipo de datos no es una restricción

➔ Es un método no paramétrico (i.e., no hay

suposición acerca del espacio de distribución y la
estructura del clasiﬁcador)

Desventajas

➔ Sobreajuste

➔ Pérdida de información al categorizar variables

continuas

➔ Precisión: métodos como SVM y clasiﬁcadores tipo

ensamblador a menudo tienen tasas de error 30%
más bajas que CART (Classiﬁcation and Regression
Trees)

➔ Inestabilidad: un pequeño cambio en los datos

puede modiﬁcar ampliamente la estructura del árbol.
Por lo tanto la interpretación no es tan directa como
parece.

03

¿Cómo decide un árbol
donde ramiﬁcarse?

¿Cómo decide un árbol donde ramiﬁcarse?

➔ La decisión de hacer divisiones estratégicas afecta altamente la precisión del árbol.

➔ Los criterios de decisión son diferentes para árboles de clasiﬁcación y regresión.

➔ Existen varios algoritmos para decidir la ramiﬁcación.

➔ La creación de subnodos incrementa la homogeneidad de los subnodos resultantes. Es

decir, la pureza del nodo se incrementa respecto a la variable objetivo.

➔ Se prueba la división con todas las variables y se escoge la que produce subnodos más

homogéneos.

Algunos algoritmos más comunes para la selección: Indice Gini, Entropía, Chi Cuadrado,

Ganancia de la información y Reducción en la varianza

Métricas de división para clasiﬁcación

GINI:

El índice de Gini se encuentra en el rango entre 0 y 1 y suele utilizarse por defecto el índice de Gini para
medir la impureza de una división.

➔ D: Conjunto de datos sobre el cual queremos medir la pureza.

➔ k: Número de clases.

➔ pi: Proporción de ejemplos en el nodo actual que pertenecen a la clase i.

La pureza mide qué tan homogéneos son los datos en cada nodo, y se calcula usando estas
proporciones.

A mayor valor de índice Gini, mayor la homogeneidad

Métricas de división para clasiﬁcación

Entropía:

➔ Un nodo menos impuro requiere menos información para ser descrito mientras un nodo más impuro

necesita más información.

➔ La teoría de la información es una medida para deﬁnir este grado de desorganización en un sistema

denominado como Entropía.

➔ Muestra completamente homogénea = entropía 0.

➔ Muestra igualmente dividida (50% – 50%) = entropía 1.

Conclusiones

Los árboles de decisión para clasiﬁcación son modelos

poderosos que dividen los datos en segmentos más

simples basados en características, buscando maximizar

la pureza en cada división. Utilizan criterios como el

índice de Gini o la Entropía para medir la homogeneidad

de los grupos resultantes. Son fáciles de interpretar y

pueden manejar datos numéricos y categóricos.

¡Muchas gracias!


