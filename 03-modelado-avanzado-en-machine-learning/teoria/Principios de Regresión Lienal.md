<!-- Convertido automáticamente desde: Principios de Regresión Lienal.pdf -->

Principios de
Regresión Lineal

01

¿Qué es una Regresión
Lineal?

La regresión lineal simple intenta predecir una respuesta
cuantitativa Y en base a una única variable predictora X.
Asume que hay aproximadamente una relación lineal
entre X e Y. Matemáticamente:

𝑌=𝛽

0+𝛽

1.𝑋

Podemos leer esta expresión como "se modela
aproximadamente como".

1 son dos constantes que representan el intercepto y

0 y 𝛽
𝛽
la pendiente en el modelo lineal.
0 y 𝛽
𝛽

1 son conocidos como los parámetros del modelo.

Entrenar el modelo equivale a calcular sus parámetros, que para la
regresión lineal son los estimadores  𝛽0 y  𝛽1  de los coeﬁcientes de la
regresión.

donde  𝑦̂ indica una predicción de Y basada en un valor particular  𝑥.
Aquí usamos un símbolo  ^ para denotar el valor estimado para un
parámetro o coeﬁciente desconocido, o para denotar el valor predicho de
la respuesta.

Regresión
Lineal

Residuo o error de predicción:

Suma residual de cuadrados (RSS):

Estimación de
los
coeﬁcientes

02

Evaluación del Modelo

Error Absoluto Medio
(Mean Absolute Error,
MAE):

El Error Absoluto Medio (Mean Absolute Error, MAE) es
una medida utilizada para evaluar la precisión de un
modelo de regresión. Proporciona una idea clara de
cuánto diﬁeren, en promedio, las predicciones de los
valores reales observados. El MAE es fácil de
interpretar y comparar entre diferentes modelos.

Error Cuadrático Medio
(Mean Squared Error,
MSE):

El Error Cuadrático Medio (Mean Squared Error,
MSE) es otra métrica comúnmente utilizada para
evaluar la precisión de un modelo de regresión. Es
especialmente útil porque penaliza los errores
grandes más que los pequeños, debido a la
elevación al cuadrado de las diferencias.

Raíz del Error Cuadrático
Medio (Root Mean
Squared Error, RMSE):

La Raíz del Error Cuadrático Medio (Root Mean
Squared Error, RMSE) es una medida utilizada para
evaluar la precisión de un modelo de regresión. Es la
raíz cuadrada del Error Cuadrático Medio (MSE) y
proporciona una medida de error en las mismas
unidades que los valores predichos, lo que facilita su
interpretación.

R^2 (Coeﬁciente de
Determinación):

El coeﬁciente de determinación es una medida
estadística que indica qué tan bien se ajusta un modelo
de regresión a los datos observados. Proporciona la
proporción de la varianza en la variable dependiente
que es predecible a partir de las variables
independientes. Es una métrica clave en la evaluación
del desempeño de los modelos de regresión.

Valor 𝑅^2 de 1

■ Un valor de  𝑅^2 de 1.0 es lo mejor que podemos conseguir. Indica

que no hay ningún error en la regresión.

Valor 𝑅^2 de 0

■ Un valor de  𝑅^2 de 0 signiﬁca que la regresión no es mejor que tomar
el valor medio, es decir no estamos usando ninguna información de
otras variables.

Valor negativo de 𝑅^2

■ Un valor negativo de  𝑅^2, signiﬁca que estamos estimando peor que

usando la media.

Suma Total de los Cuadrados
(Total Sum of Squares, TSS):

La Suma Total de los Cuadrados (Total Sum of
Squares, TSS) es una medida de la variabilidad
total en los datos observados. Se utiliza en el
contexto de la regresión para cuantiﬁcar la
cantidad total de variación presente en los valores
de la variable dependiente. La TSS se
descompone en dos partes: la Suma de los
Cuadrados de los Residuos (Residual Sum of
Squares, RSS) y la Suma de los Cuadrados
Explicada (Explained Sum of Squares, ESS).

Suma de los Cuadrados de los
Residuos (Residual Sum of
Squares, RSS):

La Suma de los Cuadrados de los Residuos
(Residual Sum of Squares, RSS), también
conocida como el Error Cuadrático Residual,
mide la variabilidad en los datos que no puede ser
explicada por el modelo de regresión. Es una
métrica que ayuda a evaluar la precisión de un
modelo, indicando cuánto los valores predichos
se desvían de los valores observados.

Intervalo de conﬁanza para  𝛽1

Test de hipótesis sobre los coeﬁcientes
estimados

(¿Existe evidencias para aﬁrmar que hay relación
entre X e Y?)

Hipótesis Nula

No hay relación entre X e Y; entonces

El test de signiﬁcación
individual tiene las
siguientes hipótesis

Hipótesis Alternativa

Hay alguna relación entre X e Y; entonces

2020

Subtítulo

Todo valor deﬁnido y

actualizado en el contexto

podrá ser usado por los

componentes incluidos

dentro de tal contexto.

Si  𝛽1=0 , entonces el modelo se reduce a  𝑌=𝛽0+𝜖
Por lo tanto  𝑋  no estaría asociado a  𝑌

Necesitamos determinar si nuestro estimador para  𝛽1
está lo suﬁcientemente lejos de cero, de forma que
podamos estar seguros de que  𝛽1 no es cero.
Un p-value pequeño indica que es poco probable
observar un valor del estadístico como el observado o
más extremo asumiendo que  𝐻0 es verdadera.
Por lo tanto, si el p-value es chico podemos rechazar  𝐻0
con baja probabilidad de equivocarnos.

¡Muchas gracias!


