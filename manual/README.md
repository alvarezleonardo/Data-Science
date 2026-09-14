# Manual de Data Science — Digital House

> Manual de estudio del programa, ordenado **de lo más básico a lo más complejo**: cada parte se apoya en la anterior. No sigue el orden de la cursada — para eso está la [tabla de equivalencia](#equivalencia-con-los-módulos-del-programa) más abajo.
>
> Cada tema combina **dos criterios**: los **apuntes** (la teoría explicada para entender y estudiar) y, en la Parte VIII, la **referencia técnica** (cuándo aplicar cada técnica, hiperparámetros, cómo se evalúa y snippets de `scikit-learn`).
>
> Cada PDF del curso tiene además su conversión 1:1 en un `.md` al lado del archivo, en la carpeta del módulo. Donde el material de origen tenía errores, se corrigen y se marcan con **Nota**. Las notas que dicen **verificado en notebook** contienen resultados medidos, no citados.
>
> El manual está dividido en archivos por parte, en esta carpeta `manual/`. Este índice enlaza a cada capítulo en su archivo correspondiente.

## Índice

**[Parte I — Fundamentos](01-fundamentos.md)**
[1. ¿Qué es Machine Learning?](01-fundamentos.md#1-qué-es-machine-learning) · [2. Datasets, features y labels](01-fundamentos.md#2-datasets-features-y-labels) · [3. Tipos de aprendizaje](01-fundamentos.md#3-tipos-de-aprendizaje) · [4. El ciclo de vida de un proyecto de ML](01-fundamentos.md#4-el-ciclo-de-vida-de-un-proyecto-de-ml) · [5. Preparación de datos y EDA](01-fundamentos.md#5-preparación-de-datos-y-eda)

**[Parte II — Cómo se evalúa un modelo](02-como-se-evalua-un-modelo.md)**
[6. Train/test y validación cruzada](02-como-se-evalua-un-modelo.md#6-traintest-y-validación-cruzada) · [7. Sobreajuste, subajuste y sesgo-varianza](02-como-se-evalua-un-modelo.md#7-sobreajuste-subajuste-y-el-compromiso-sesgo-varianza) · [8. Métricas de clasificación](02-como-se-evalua-un-modelo.md#8-métricas-de-clasificación) · [9. Métricas de regresión](02-como-se-evalua-un-modelo.md#9-métricas-de-regresión)

**[Parte III — Modelos lineales](03-modelos-lineales.md)**
[10. Clasificadores lineales](03-modelos-lineales.md#10-clasificadores-lineales) · [11. Regresión lineal](03-modelos-lineales.md#11-regresión-lineal) · [12. Inferencia sobre los coeficientes](03-modelos-lineales.md#12-inferencia-sobre-los-coeficientes)

**[Parte IV — Modelos no lineales](04-modelos-no-lineales.md)**
[13. Árboles de decisión](04-modelos-no-lineales.md#13-árboles-de-decisión) · [14. Ensambles y XGBoost](04-modelos-no-lineales.md#14-ensambles-y-xgboost)

**[Parte V — Complejidad del modelo y selección](05-complejidad-del-modelo-y-seleccion.md)**
[15. Regularización: Ridge, Lasso y Elastic Net](05-complejidad-del-modelo-y-seleccion.md#15-regularización-ridge-lasso-y-elastic-net) · [16. Selección de variables](05-complejidad-del-modelo-y-seleccion.md#16-selección-de-variables) · [17. Criterios de selección de modelos (AIC y BIC)](05-complejidad-del-modelo-y-seleccion.md#17-criterios-de-selección-de-modelos-aic-y-bic)

**[Parte VI — Aprendizaje no supervisado](06-aprendizaje-no-supervisado.md)**
[18. Clustering con K-means](06-aprendizaje-no-supervisado.md#18-clustering-con-k-means) · [19. Clustering jerárquico](06-aprendizaje-no-supervisado.md#19-clustering-jerárquico) · [20. DBSCAN](06-aprendizaje-no-supervisado.md#20-dbscan-clustering-por-densidad) · [21. Evaluación de clusters](06-aprendizaje-no-supervisado.md#21-evaluación-de-clusters) · [22. La maldición de la dimensión](06-aprendizaje-no-supervisado.md#22-la-maldición-de-la-dimensión) · [23. Reducción de dimensionalidad](06-aprendizaje-no-supervisado.md#23-reducción-de-dimensionalidad)

**[Parte VII — Redes neuronales](07-redes-neuronales.md)**
[24. Fundamentos biológicos](07-redes-neuronales.md#24-fundamentos-biológicos) · [25. Historia](07-redes-neuronales.md#25-historia-de-las-redes-neuronales) · [26. El perceptrón](07-redes-neuronales.md#26-el-perceptrón-estructura-y-fórmulas) · [27. Entrenamiento: la compuerta AND](07-redes-neuronales.md#27-entrenamiento-del-perceptrón-la-compuerta-and) · [28. Limitaciones](07-redes-neuronales.md#28-limitaciones-del-perceptrón) · [29. Implementación con scikit-learn](07-redes-neuronales.md#29-implementación-con-scikit-learn) · [30. El perceptrón multicapa](07-redes-neuronales.md#30-el-perceptrón-multicapa-mlp) · [31. Grafos y capa densa](07-redes-neuronales.md#31-grafos-y-capa-densa) · [32. Funciones de activación](07-redes-neuronales.md#32-funciones-de-activación) · [33. Diseño de la arquitectura](07-redes-neuronales.md#33-diseño-de-la-arquitectura-de-la-red) · [34. Funciones de pérdida](07-redes-neuronales.md#34-funciones-de-pérdida) · [35. Optimización y descenso de gradiente](07-redes-neuronales.md#35-optimización-y-descenso-de-gradiente) · [36. Regularización en redes](07-redes-neuronales.md#36-regularización-en-redes-neuronales) · [37. Backpropagation](07-redes-neuronales.md#37-backpropagation) · [38. Persistencia de modelos](07-redes-neuronales.md#38-persistencia-de-modelos)

**[Parte IX — Deep Learning con frameworks](08-deep-learning-con-frameworks.md)**
[39. Por qué hacen falta TensorFlow y PyTorch](08-deep-learning-con-frameworks.md#39-por-qué-hacen-falta-tensorflow-y-pytorch) · [40. TensorFlow y Keras](08-deep-learning-con-frameworks.md#40-tensorflow-y-keras) · [41. PyTorch](08-deep-learning-con-frameworks.md#41-pytorch) · [42. Keras y PyTorch lado a lado](08-deep-learning-con-frameworks.md#42-keras-y-pytorch-lado-a-lado) · [43. Redes convolucionales](08-deep-learning-con-frameworks.md#43-redes-convolucionales-cnns) · [44. La capa convolucional](08-deep-learning-con-frameworks.md#44-la-capa-convolucional) · [45. Agrupamiento, aplanamiento y densas](08-deep-learning-con-frameworks.md#45-agrupamiento-aplanamiento-y-capas-densas) · [46. Transfer learning](08-deep-learning-con-frameworks.md#46-transfer-learning-reutilizar-una-red-ya-entrenada) · [47. Data augmentation](08-deep-learning-con-frameworks.md#47-data-augmentation-más-datos-sin-salir-a-buscarlos) · [48. Redes recurrentes](08-deep-learning-con-frameworks.md#48-redes-recurrentes-rnns) · [49. GRU](08-deep-learning-con-frameworks.md#49-gru-unidades-recurrentes-con-compuertas) · [50. LSTM](08-deep-learning-con-frameworks.md#50-lstm-memoria-a-largo-plazo) · [51. RNNs en la práctica](08-deep-learning-con-frameworks.md#51-rnns-en-la-práctica-trabajar-con-texto) · [52. Procesamiento de lenguaje natural](08-deep-learning-con-frameworks.md#52-procesamiento-de-lenguaje-natural) · [53. Seq2Seq](08-deep-learning-con-frameworks.md#53-seq2seq-y-el-problema-del-cuello-de-botella) · [54. Mecanismos de atención](08-deep-learning-con-frameworks.md#54-mecanismos-de-atención) · [55. Transformadores](08-deep-learning-con-frameworks.md#55-transformadores) · [56. Autoencoders](08-deep-learning-con-frameworks.md#56-autoencoders) · [57. Redes Adversarias Generativas (GANs)](08-deep-learning-con-frameworks.md#57-redes-adversarias-generativas-gans) · [58. De los Transformadores a los LLMs](08-deep-learning-con-frameworks.md#58-de-los-transformadores-a-los-llms)

**[Parte VIII — Referencia técnica](09-referencia-tecnica.md)** · **[Desafíos profesionales](09-referencia-tecnica.md#desafíos-profesionales)** · **[Glosario](09-referencia-tecnica.md#glosario-rápido)**

## Equivalencia con los módulos del programa

El manual reordena el contenido por dificultad. Esta tabla mapea cada módulo de la cursada a los capítulos donde quedó su material.

| Módulo del programa | Capítulos |
|---|---|
| **01** — Introducción a Machine Learning | 1-5, 6, 8, 10, 13, 14 |
| **02** — Desafío Profesional (Etapa 1) | [Desafíos profesionales](09-referencia-tecnica.md#desafíos-profesionales) |
| **03** — Modelado avanzado en ML | 9, 11, 12 |
| **04** — Aprendizaje no supervisado | 15, 16, 17, 18-23 |
| **05** — Desafío Profesional (Etapa 2) | [Desafíos profesionales](09-referencia-tecnica.md#desafíos-profesionales) |
| **06** — Fundamentos de redes neuronales | 24-38 |
| **07** — Fundamentos de deep learning | 39-57 |
| **08** — Gestión de proyectos de IA | pendiente |
| *fuera del programa* | 46, 47 (transfer learning y data augmentation) · 58 (LLMs y RAG) |

> **Capítulos que no vienen de una slide.** El **7** (sobreajuste y sesgo-varianza) es una ampliación propia: el material del curso lo da por sabido. Los **46** y **47** (transfer learning y data augmentation) tampoco están en el programa, pero son de uso constante en la práctica con CNNs y se agregan junto al bloque de convolucionales. El **58** (LLMs y RAG) está **fuera del programa** — el curso termina en autoencoders y GANs, capítulos **56** y **57** — y se agrega porque es la continuación directa de los Transformadores y lo que se encuentra hoy en el trabajo real. Los capítulos 5, 6, 10, 14, 19, 21 y los del módulo 07 están ampliados bastante más allá de lo que cubren las slides.
