# Integración de Tecnologías de IA

> Conversión a Markdown de las slides del curso (Clases 6 a 8 — Integración de Tecnologías de IA, **Módulo 3**). El PDF original está en [`material/`](<material/Integración de Tecnologías de IA .pdf>) (nota: el nombre del archivo original trae un espacio antes de la extensión `.pdf`).

## 1. Machine Learning, Deep Learning, NLP y Visión por Computadora

El material repasa, sin definirlas de nuevo (ya se vieron en la Clase 2), las cuatro tecnologías centrales de la IA aplicada, con foco en **por qué son importantes hoy** — es decir, en la justificación de negocio de cada una, más que en su funcionamiento técnico.

### Machine Learning

- Impulsa la innovación en sectores como medicina, educación, transporte y energía.
- Optimiza procesos empresariales y operativos, ahorrando tiempo y recursos.
- Mejora la toma de decisiones mediante análisis predictivo y modelos avanzados de datos.
- Fomenta tecnologías emergentes: vehículos autónomos, asistentes virtuales, diagnóstico médico.

### Deep Learning

- **Maneja datos no estructurados:** a diferencia del ML clásico, que funciona mejor con datos estructurados (tablas), el Deep Learning procesa directamente imágenes, videos y texto, descubriendo patrones complejos sin necesidad de preprocesamiento manual.
- **Reduce la dependencia de la ingeniería de características (*feature engineering*):** en ML tradicional hay que definir y extraer manualmente qué variables importan; el Deep Learning automatiza esa detección, permitiendo un aprendizaje más eficiente y profundo.

> **Nota:** este es el punto técnico más preciso del material en esta sección, y conecta directamente con lo visto en el módulo anterior de deep learning: las CNNs, por ejemplo, aprenden automáticamente qué filtros/bordes/texturas son relevantes en una imagen, en lugar de que una persona programe esas reglas a mano.

### Procesamiento del Lenguaje Natural (NLP)

- **Aplicación directa en múltiples idiomas y dialectos:** a diferencia de otros métodos de IA que requieren grandes cantidades de datos etiquetados específicos por tarea, el NLP aprende y generaliza de textos en diferentes idiomas, facilitando su aplicación global.
- **Mejora de la comprensión contextual:** utiliza técnicas avanzadas como **transformers** y **redes neuronales recurrentes (RNNs)** para capturar la estructura y el contexto del lenguaje, mejorando la precisión y relevancia de las respuestas generadas.

> **Nota:** el material vuelve a mencionar transformers y RNNs sin profundizar — ambos ya se desarrollaron en detalle en el módulo anterior ([07-fundamentos-de-deep-learning/teoria/](<../../07-fundamentos-de-deep-learning/teoria/>), clases 15 a 26). Acá se los cita como las técnicas que sostienen al NLP moderno, no como contenido nuevo.

### Visión por Computadora (Computer Vision, CV)

- **Capacidad de interpretar datos visuales complejos:** a diferencia de métodos basados en datos numéricos o textuales, la visión por computadora procesa y analiza datos visuales no estructurados (fotos, videos, flujos en tiempo real), detectando patrones complejos y sutiles en la información visual.
- **Automatización de tareas visuales humanas:** identificación de objetos, reconocimiento facial, análisis de imágenes médicas y conducción autónoma, superando en muchos casos la precisión del ojo humano.

## 2. Tendencias actuales

El material menciona brevemente los **modelos generativos** (GPT-4, DALL-E) como la tendencia emergente de mayor impacto, sin desarrollarla más allá de nombrarla en el índice de la Clase 6.

> **Nota:** el material no profundiza qué distingue a un modelo generativo de los modelos discriminativos vistos hasta acá (clasificación, regresión). En términos generales: un modelo generativo aprende la distribución de los datos para poder *generar* contenido nuevo (texto con GPT-4, imágenes con DALL-E), mientras que los modelos vistos en módulos anteriores (CNNs para clasificación, RNNs para predicción) son discriminativos — aprenden a distinguir o predecir sobre datos existentes, no a crear datos nuevos. Los autoencoders y GANs del módulo anterior ([07-fundamentos-de-deep-learning](<../../07-fundamentos-de-deep-learning/teoria/Redes Adversarias Generativas (GANs).md>)) son la base técnica de esta familia de modelos generativos.

## 3. Herramientas de Desarrollo de IA (Clase 7)

### Lenguajes de programación

- **Python:** lenguaje predominante en IA, con bibliotecas como NumPy, pandas y scikit-learn.
- **R:** utilizado principalmente en análisis estadístico y visualización de datos.

### Frameworks y bibliotecas

- **TensorFlow:** plataforma de código abierto para el desarrollo de modelos de aprendizaje automático y profundo.
- **Keras:** interfaz de alto nivel para TensorFlow que simplifica la construcción de modelos.
- **PyTorch:** framework flexible y popular para investigación y producción en aprendizaje profundo.

> Estos tres frameworks ya se usaron en profundidad en el módulo anterior ([07-fundamentos-de-deep-learning](<../../07-fundamentos-de-deep-learning/README.md>)), donde se implementó la misma red en TensorFlow/Keras y en PyTorch en paralelo, para comparar ambos flujos de trabajo.

### Plataformas en la nube

El material dedica una slide a cada una de las tres plataformas principales:

- **Google Cloud AI** (lanzada en 2017): ofrece herramientas y servicios para desarrollar, entrenar y desplegar modelos de IA en la nube, aprovechando la infraestructura escalable de Google. Incluye **AutoML** para crear modelos personalizados sin necesidad de conocimientos avanzados, y **AI Platform** para gestionar el ciclo de vida completo de los modelos.
- **AWS SageMaker** (lanzado en 2017 por Amazon Web Services): servicio totalmente administrado que permite construir, entrenar y desplegar modelos de machine learning rápidamente, con herramientas integradas para preparar datos, elegir algoritmos, ajustar modelos y ponerlos en producción.
- **Azure Machine Learning** (lanzado en 2015 por Microsoft): entorno integral para construir, entrenar y desplegar modelos de machine learning, con soporte para múltiples lenguajes de programación y frameworks, gestionando todo el ciclo de vida desde la preparación de datos hasta el despliegue en producción.

> **Nota:** el material no compara estas tres plataformas entre sí (costos, curva de aprendizaje, integración con el resto del stack de cada nube) — las presenta como opciones equivalentes sin una recomendación de cuándo usar cada una. La elección en la práctica suele depender de qué proveedor de nube ya usa la organización, más que de diferencias técnicas profundas entre las tres.

## 4. Prácticas integrales del módulo (Clase 8)

El material presenta dos casos prácticos, sin desarrollo técnico en la slide misma:

- **RStudio + Shiny:** uso de RStudio para crear modelos predictivos y de clasificación, con **Shiny** para construir aplicaciones web interactivas en R.
- **ChatGPT como complemento:** uso de ChatGPT para complementar desarrollos propios de modelos de aprendizaje automático — resolver problemas, generar ideas y automatizar tareas de manera más eficiente en el trabajo de un científico de datos.

> **Nota:** este es el primer punto del curso donde aparece R con un enfoque de aplicación web (Shiny), un desvío del resto del programa (centrado en Python). El material no explica por qué introduce R específicamente acá — probablemente como complemento porque R es fuerte en estadística aplicada y Shiny facilita prototipos de dashboards interactivos sin un frontend dedicado, algo que Python no resuelve de forma tan directa.

## 5. Relación con el resto del módulo

Esta clase profundiza el "cómo" técnico de la Fase 3 (Desarrollo y Entrenamiento de Modelos) vista en la Clase 4: qué lenguaje, framework y plataforma en la nube elegir para llevar un proyecto de IA de la idea a un modelo entrenado. El Módulo 4 retoma el hilo de gestión (equipos, roles) y el Módulo 5 vuelve a lo técnico con el despliegue y la escalabilidad — de modo que estas herramientas (frameworks + nube) son el puente entre "definir el problema" y "ponerlo en producción".
