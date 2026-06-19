<!-- Convertido automáticamente desde: 5 - Preparación de Datos.pdf -->

Preparación de
Datos

Limpieza y preparación de datos

En 2009 Mike Driscoll (data scientist y CEO de Metamarkets) popularizó el término "data
munging" para referirse al arduo proceso de limpiar, preparar y validar los datos.

Resolver problemas de formato y asignar los
tipos correctos de datos:

Por ejemplo cuando al pasar de CSV a Pandas una
fecha no se importa correctamente. Ej:
20090609231247 en lugar de 2009-06-09 23:12:47.

Estandarizar categorías:

Cuando los datos se recolectaron con un sistema
que no tiene los valores tipiﬁcados, valores que
representan las mismas categorías pueden estar
expresados de forma distinta, por ejemplo Arg, AR
y Argentina.

Limpieza de
Datos

Corregir valores erróneos:
Por ejemplo un valor numérico o inválido para describir el
género. O una edad representada por un número negativo o
mucho mayor que 100

Completar datos faltantes:
Los datasets del mundo real suelen venir con datos faltantes
que responden a información que se perdió o nunca se
recolectó.
Existen varias técnicas para completar datos faltantes. Al
proceso de completar datos faltantes se lo
llama "imputación".

Organizar correctamente el dataset:
Es importante estructurar las ﬁlas y
columnas de la forma más conveniente.
Para hacerlo se pueden aplicar las reglas
del "tidy data".

Limpieza de
Datos

Manejo de cantidades en diferentes
unidades o escalas.
Para no favorecer a ninguna variable en
particular al calcular la distancia,
debemos estandarizar y deshacernos
de las unidades.

¿Por qué
normalizar?

En machine learning, un hiperparámetro es un
parámetro cuyo valor se utiliza para controlar el proceso
de aprendizaje. Por el contrario, los valores de los
parámetros se derivan a través del entrenamiento.

Los hiperparámetros se pueden clasificar como:

• hiperparámetros de modelo
• hiperparámetros de algoritmo

¿Qué es un
hiperparámetro?

Diferentes algoritmos de aprendizaje requieren
diferentes hiperparámetros, algunos algoritmos simples
(como la regresión de mínimos cuadrados ordinarios)
no requieren ninguno.

Por ejemplo, LASSO es un algoritmo que agrega un hiperparámetro de regularización a la
regresión de mínimos cuadrados ordinarios, que debe establecerse antes de estimar los
parámetros a través del algoritmo de entrenamiento.

(Puede ser que no quede claro en este momento este ejemplo, pero lo será en unas
pocas clases)

Usando hiperparámetros distintos podemos crear instancias distintas de la misma clase
del modelo seleccionado.

Los valores de los hiperparámetros deben ser establecidos antes de que el modelo sea
ajustado a los datos.

Una misma clase de modelo con dos conﬁguraciones de hiperparámetros distintas
puede llevar a resultados muy disímiles entre sí.

Preparar los datos en una matriz de features y un vector
target

Como vimos anteriormente, para el caso del aprendizaje supervisado requerimos la
matriz de features (las columnas con las características de las observaciones) y una
variable target con los labels que representan la clasiﬁcación de cada observación.

La matriz de features, X, puede ser un array bidimensional de Numpy o un DataFrame de
Pandas, mientras que el vector objetivo, y, puede ser un array de Numpy o una Series de
Pandas.

Para los modelos de aprendizaje no supervisado, no se requiere este paso.

Separar los sets de entrenamiento y de testing

El objetivo de un modelo de machine learning es generalizar a partir de lo aprendido, y ser
capaz predecir la variable target de observaciones nunca antes vistas por el modelo.

Entonces el modelo debe aprender primero a partir de los datos que disponemos.

Y para evaluar qué tan buenas resultan sus predicciones, podemos comparar los valores
reales y los que predice.

Para tener noción de cómo se comporta frente a datos nunca antes vistos dividimos
nuestros datos en dos conjuntos disjuntos: entrenamiento y testing.

Preparar los datos en una matriz de features y un vector
target

Para resolver este problema, vamos a dividir los datos de X en un training set y un testing
set. El modelo aprenderá a partir de los datos del training set.

Luego del entrenamiento, aplicamos el modelo a las observaciones de la X del conjunto
de testing set, y los valores predichos de y son comparados con los valores reales.

La división entre train y test se podría hacer a mano, pero Scikit-learn nos ofrece la
función train_test_split(), que importaremos del módulo de model_selection de
Scikit-Learn, el cual contiene herramientas para la selección y evaluación de modelos

¡Muchas gracias!


