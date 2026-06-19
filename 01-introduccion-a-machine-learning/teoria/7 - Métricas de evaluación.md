<!-- Convertido automáticamente desde: 7 - Métricas de evaluación.pdf -->

Métricas para la
Evaluación de
Modelos de
clasiﬁcación

Índice

01
02

03

Matriz de Confusión
Accuracy, Precisión, Recall, y
F1-score.
ROC y AUC

01

Matriz de Confusión

Matriz de Confusión

➔ TP = Verdaderos positivos (True positives)

➔ FN = Falsos negativos (False Negative)

➔ FP = Falsos positivos (False positives)

➔ TN  = Verdaderos negativos (True negatives)

02

Accuracy, Precisión,
Recall, y F1-score.

Accuracy

➔ Es la exactitud global del modelo corresponde a la proporción de datos que fueron correctamente
clasiﬁcados, independiente de la categoría, por eso se considera la métrica de exactitud global.

Precisión

➔ La proporción de ejemplos correctamente clasiﬁcados como positivos (verdaderos positivos) en
comparación con todos los ejemplos clasiﬁcados como positivos (verdaderos positivos y falsos
positivos).

Recall

➔ Es la proporción de ejemplos correctamente clasiﬁcados como positivos (verdaderos positivos) en
comparación con todos los ejemplos que realmente son positivos (verdaderos positivos y falsos
negativos). Esto reﬂeja la capacidad del modelo para identiﬁcar correctamente los ejemplos positivos.

F1 Score

➔ Es una métrica que combina la precisión y la sensibilidad en una sola medida. Es útil cuando se busca
un equilibrio entre la precisión y la sensibilidad, ya que tiene en cuenta tanto los falsos positivos como
los falsos negativos.

03

ROC y AUC

Curva ROC (Receiver Operating Characteristic)

➔ La curva ROC (Receiver Operating Characteristic) es una

representación gráﬁca utilizada para evaluar el rendimiento
de un modelo de clasiﬁcación binaria. La curva ROC muestra
la relación entre la tasa de verdaderos positivos (TPR, True
Positive Rate) y la tasa de falsos positivos (FPR, False
Positive Rate) a diferentes umbrales de clasiﬁcación.

○ La TPR, también conocida como sensibilidad o recall, se
calcula como el número de verdaderos positivos dividido
por la suma de verdaderos positivos y falsos negativos.

○ La FPR se calcula como el número de falsos positivos
dividido por la suma de falsos positivos y verdaderos
negativos.

➔ Para crear la curva ROC, se trazan los valores de TPR en el

eje y (vertical) contra los valores de FPR en el eje x
(horizontal) para cada posible umbral de clasiﬁcación.

AUC (Área Bajo la Curva)

➔ El AUC (Área Bajo la Curva) ROC es una métrica utilizada
para evaluar el rendimiento de un modelo de clasiﬁcación
binaria. Esta métrica representa el área delimitada por la
curva ROC y el eje diagonal.

○ Cuanto mayor sea el AUC, mejor será el rendimiento del
modelo. Un AUC de 1 indica un modelo perfecto, lo que
signiﬁca que puede clasiﬁcar todas las instancias
correctamente. Por otro lado, un AUC de 0.5 indica un
modelo que no es mejor que una clasiﬁcación aleatoria.

➔ El AUC se calcula integrando el área bajo la curva ROC. En
términos prácticos, esto se puede hacer utilizando métodos
numéricos, como la regla del trapecio, o mediante funciones
especíﬁcas en bibliotecas como scikit-learn en Python.

Conclusiones

Las métricas de evaluación de modelos son
herramientas fundamentales en el desarrollo y la
validación de modelos de machine learning. Estas
métricas proporcionan una evaluación objetiva del
rendimiento del modelo, lo que permite ajustar y mejorar
nuestros modelos.


