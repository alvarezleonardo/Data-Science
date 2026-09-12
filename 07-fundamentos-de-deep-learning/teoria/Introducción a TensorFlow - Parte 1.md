# Introducción a TensorFlow - Parte 1

> Conversión a Markdown de las slides del curso (Clase 3 — Introducción a TensorFlow, bloque 1). El PDF original está en [`material/`](material/).

## 1. Qué es TensorFlow

**TensorFlow** es una biblioteca de código abierto desarrollada por Google que se utiliza para el desarrollo y entrenamiento de modelos de aprendizaje automático y redes neuronales. Fue diseñada para facilitar la construcción de modelos complejos a gran escala y está optimizada para ejecutar cálculos numéricos en diversas plataformas, como CPUs, GPUs y TPUs (unidades de procesamiento tensorial).

| Característica | Detalle |
|---|---|
| **Flexibilidad y escalabilidad** | Es altamente flexible y escalable. |
| **Ecosistema** | Amplio ecosistema de herramientas, bibliotecas y recursos, ideal para el desarrollo de aplicaciones de inteligencia artificial. |
| **Producción** | Ofrece herramientas y funcionalidades para el despliegue de modelos de aprendizaje automático en entornos de producción. |
| **Comunidad** | Gran comunidad de usuarios y desarrolladores que contribuyen con código, documentación y recursos educativos. |

## 2. Fundamentos

| Concepto | Qué es |
|---|---|
| **Tensores** | Los datos se representan como tensores, que son arreglos multidimensionales con un tipo de datos específico. Son fundamentales porque todas las operaciones y cálculos se realizan sobre ellos. |
| **Gráficos computacionales** | Representación estructurada de las operaciones matemáticas que se ejecutarán sobre los datos. El gráfico se construye de manera **estática antes de la ejecución**. |
| **Diferenciación automática** | TensorFlow calcula automáticamente gradientes de funciones, esencial para optimizar modelos mediante técnicas como el descenso de gradiente. |
| **Variables** | Objetos que pueden cambiar de valor a medida que se ejecuta el programa. |

> **Nota:** la slide describe el grafo computacional como algo que "se construye de manera estática antes de la ejecución". Eso corresponde al modo clásico de TensorFlow 1.x (`tf.Graph` + sesión). Desde TensorFlow 2.x el modo por defecto es **eager execution** (las operaciones se ejecutan inmediatamente, como en NumPy); el grafo estático sigue existiendo pero se arma de forma optativa con el decorador `@tf.function`, no es el flujo obligatorio como antes.

## 3. Keras

**Keras** es una biblioteca de código abierto de alto nivel que simplifica la construcción y el entrenamiento de modelos de aprendizaje profundo. Se integra con varias bibliotecas de backend para la ejecución de operaciones matemáticas complejas.

| Característica | Detalle |
|---|---|
| **Facilidad de uso** | Diseñado para ser fácil de usar y desarrollar rápidamente modelos de aprendizaje profundo. |
| **Arquitecturas** | Ofrece soporte nativo para varios tipos de arquitecturas de redes neuronales profundas. |
| **Backends** | Compatible con múltiples backends. |
| **Escala de entrenamiento** | Permite entrenar modelos en una variedad de entornos, desde computadoras personales hasta clústeres distribuidos en la nube. |

> **Nota:** desde TensorFlow 2.0, Keras dejó de ser un proyecto externo que corría "sobre" TensorFlow para convertirse en su API de alto nivel oficial (`tf.keras`). Con **Keras 3** (instalado en este entorno, ver `Configuración del entorno.md`), volvió a desacoplarse y hoy soporta como backend no solo TensorFlow sino también PyTorch y JAX — el punto de "compatible con múltiples backends" de la slide es, si acaso, más cierto ahora que cuando se escribió el material.

## 4. Relación con el resto del módulo

- Esta parte da el marco conceptual (tensores, grafo, Keras); la **Parte 2** entra en el código concreto: cargar datos, definir el modelo y compilarlo.
- La **Parte 3** cierra con el entrenamiento (`model.fit`) y la evaluación (`model.evaluate`).
- Los tensores y la diferenciación automática son la contraparte en TensorFlow de lo que en el módulo anterior (06) se vio "a mano": backpropagation es, en esencia, la diferenciación automática que TensorFlow calcula por vos.
