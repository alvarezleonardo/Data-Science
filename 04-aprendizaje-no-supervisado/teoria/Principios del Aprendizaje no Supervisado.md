# Principios del Aprendizaje no Supervisado

> Conversión a Markdown de la slide del curso (Clase 3). El PDF original está al lado.

## 1. ¿Qué es el Aprendizaje Automático?

El **aprendizaje automático** (Machine Learning) es una rama de la inteligencia artificial que permite que los sistemas **aprendan automáticamente** a partir de los datos, mejorando su rendimiento sin ser explícitamente programados.

Los algoritmos de ML construyen **modelos matemáticos** basados en datos de entrenamiento, permitiendo que el sistema prediga o tome decisiones sin intervención humana.

> "El Aprendizaje Automático (ML) es una rama de la inteligencia artificial que permite a las máquinas aprender de datos y mejorar su rendimiento en tareas específicas sin ser programadas explícitamente para ello."
> — Arthur L. Samuel (1901-1990)

### Programación clásica vs. Aprendizaje automático

Es la diferencia clave sobre **qué se programa** y **qué se obtiene**:

- **Programación clásica:** el programador define las **reglas**; con esas reglas + los datos, el sistema produce **respuestas**.
  - `Reglas + Datos → Respuestas`
- **Aprendizaje automático:** se le dan **datos** y **respuestas** (los resultados esperados) y el sistema **infiere las reglas** (el modelo, con sus parámetros `w`).
  - `Datos + Respuestas → Reglas`

En ML el ciclo de **entrenamiento** ajusta los parámetros `w` del algoritmo: se compara el output contra un **target** con una **métrica**, y un **optimizador** actualiza `w` para mejorar. La **inferencia/predicción** es usar el modelo ya entrenado sobre datos nuevos.

## 2. Tipos de Aprendizaje Automático

- **Aprendizaje Supervisado:** clasificación y regresión.
- **Aprendizaje No Supervisado:** clustering y reducción de dimensionalidad.
- **Aprendizaje por Refuerzo:** decisiones en tiempo real, agentes, etc.

## 3. Aprendizaje Supervisado

El modelo se entrena con un **conjunto de datos etiquetados**, donde las **respuestas correctas son conocidas**.

- Los datos se organizan en una tabla: **filas** (registros/observaciones) y **columnas**. Las columnas de entrada son las **características o features** y la columna a predecir es la **etiqueta o label**.
- Con datos nuevos, el **modelo predictivo** entrega el resultado.

Dos tareas según el tipo de etiqueta:
- **Clasificación:** la etiqueta es **categórica** (separar clases con una frontera de decisión).
- **Regresión:** la etiqueta es **continua** (ajustar una recta/curva a los datos).

## 4. Aprendizaje No Supervisado

El modelo se entrena con un **conjunto de datos sin etiquetar** y busca **patrones o estructuras ocultas** en los datos.

- Los datos tienen **features** pero **no hay columna de etiqueta**.
- Tareas principales:
  - **Clusterización (clustering):** agrupar observaciones similares.
  - **Reducción de dimensión:** comprimir/proyectar los datos manteniendo su estructura.

## 5. Aprendizaje por Refuerzo

Modelos y funciones enfocadas en **maximizar una medida de recompensa**, basados en **acciones** y en el **ambiente** en el que el agente inteligente se desempeñará.

- Ciclo **agente ↔ ambiente**: el agente observa un **estado**, elige una **acción** (según un algoritmo de selección), recibe una **recompensa** y busca la **mejor acción** para maximizar la recompensa acumulada.

## Conclusiones

Mientras que el aprendizaje **supervisado** se centra en **predecir etiquetas conocidas**, el aprendizaje **no supervisado** se enfoca en **descubrir patrones ocultos y estructuras subyacentes** en datos sin etiquetas.

Ambos enfoques son fundamentales en el aprendizaje automático y tienen aplicaciones variadas en diferentes campos y sectores. La elección entre uno u otro **depende de la naturaleza del problema** y de los objetivos específicos del análisis de datos.
