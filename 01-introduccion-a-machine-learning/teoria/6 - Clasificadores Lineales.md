<!-- Convertido automáticamente desde: 6 - Clasificadores Lineales.pdf -->

Clasiﬁcadores
Lineales

Los clasiﬁcadores lineales son modelos de aprendizaje automático
utilizados para la clasiﬁcación de datos. Se llaman "lineales" porque
toman decisiones basadas en una combinación lineal de las
características de los datos de entrada.

//

Hiperplano:

Un hiperplano es una generalización de un plano en
espacios de dimensiones superiores. En dos
dimensiones, un hiperplano es una línea, en tres
dimensiones es un plano, y en dimensiones
superiores, sigue la misma lógica.

Función de Decisión

x es el vector de características.
𝑤 es el vector de pesos.
𝑏 es el sesgo (bias).
La clasiﬁcación se realiza evaluando el signo de 𝑓(𝑥):
● Si 𝑓(𝑥)≥0, la muestra se clasiﬁca en una clase.
● Si 𝑓(𝑥)<0, se clasiﬁca en la otra clase.

Conceptos
Básicos

Perceptrón:

El perceptrón es uno de los clasiﬁcadores lineales más simples
y fue uno de los primeros algoritmos de aprendizaje
automático. Se ajusta mediante una regla de actualización que
modiﬁca los pesos en función de los errores de clasiﬁcación.

Regresión Logística

Aunque su nombre sugiere regresión, se utiliza para
clasiﬁcación. Utiliza una función logística para modelar la
probabilidad de que una muestra pertenezca a una clase
particular.

Máquinas de Vectores de Soporte
(SVM)

SVM encuentra el hiperplano que
maximiza la distancia (margen) entre
las dos clases. Soluciona un
problema de optimización
cuadrática para encontrar el
hiperplano óptimo.

Clasiﬁcadores
Lineales
Comunes

La regresión logística nos permite modelar la probabilidad de que la
variable objetivo y pertenezca a una determinada categoría, dados los
valores de las variables X.
¿Podríamos usar una regresión lineal?
Podríamos, pero vamos a obtener valores fuera del rango [0,1], diﬁcultando
la interpretación como probabilidad.

¿Por qué no podemos estimar la probabilidad usando una regresión
lineal simple?

Si estimamos p(y=1 | X) con una regresión lineal, nuestro modelo asumiría la siguiente forma:.

Esto arrojaría valores fuera del rango válido para una probabilidad [0,1].

Cuando el problema de clasiﬁcación es
multiclase, el modelo interpretaría a las
diferentes clases como valores numéricos.

La función logística.

//

Sin importar qué valores tome X siempre vamos a predecir valores dentro del rango [0,1].

¡Muchas gracias!


