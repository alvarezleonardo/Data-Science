# Gestión de proyectos de IA

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 7 | 18 | 🟨 En curso — teoría de los módulos 2 a 6 documentada |

> Continúa [Fundamentos de Deep Learning](../07-fundamentos-de-deep-learning/): de entrenar redes neuronales se pasa a la **gestión integral de proyectos de IA** — definición del problema, herramientas y frameworks, gestión de equipos, despliegue, escalabilidad y MLOps.

## Teoría

> Los **6 PDF y 1 PPTX originales** convertidos hasta ahora están en [`teoria/material/`](teoria/material/); en `teoria/` quedan sus conversiones a Markdown.

- [`0 - Programa del módulo.md`](<teoria/0 - Programa del módulo.md>) — las 18 clases en 7 módulos
- [`Introducción a la IA y su evolución.md`](<teoria/Introducción a la IA y su evolución.md>) — Clases 2 a 4, Módulo 2: definición de IA, ramas (ML/DL/IA simbólica), breve historia, IA débil vs. fuerte, casos de impacto sectorial y las cuatro fases de un proyecto de IA
- [`Integración de Tecnologías de IA.md`](<teoria/Integración de Tecnologías de IA.md>) — Clases 6 a 8, Módulo 3: ML/DL/NLP/CV aplicados, lenguajes y frameworks (Python, R, TensorFlow, Keras, PyTorch), plataformas en la nube (Google Cloud AI, AWS SageMaker, Azure ML), y el caso práctico de RStudio + Shiny + ChatGPT
- [`Desarrollo y Gestión de Proyectos de IA.md`](<teoria/Desarrollo y Gestión de Proyectos de IA.md>) — Clases 10 y 11, Módulo 4: definición del problema con un caso de transfer learning / neural style transfer, y gestión de equipos (roles, habilidades, comunicación, contratación asistida por IA)
- [`Despliegue y Escalabilidad de Proyectos de IA.md`](<teoria/Despliegue y Escalabilidad de Proyectos de IA.md>) — Clase 13, Módulo 5: escalabilidad horizontal y vertical, recursos en la nube, reducción de latencia, GPU vs. CPU y contenedores (Docker)
- [`Introducción a MLOps.md`](<teoria/Introducción a MLOps.md>) — Clase 15, Módulo 6: definición de MLOps, experiment tracking y control de versiones de modelos (W&B, MLflow), y el stack de CI/CD y monitoreo (Jenkins, GitHub Actions, Prometheus, Grafana)

Queda pendiente convertir la teoría del módulo 7 (Cierre de curso) a medida que se sume el material correspondiente.

## Notebooks

Los **7 notebooks** de [`notebooks/`](notebooks/) están documentados, con los hallazgos sobre el material anotados en cada uno.

- [`M02_S03 — Análisis de imágenes en el sector industrial`](<notebooks/OD_GPIA_ESP_M02_S03_Análisis de imágenes en el sector industrial - Recurso descargable.ipynb>) — KNN sobre imágenes de piezas fundidas
- [`M02_S03 — Diagnóstico de diabetes`](<notebooks/OD_GPIA_ESP_M02_S03_Desarrollo de proyecto_ Diagnóstico de diabetes  - Recurso descargable.ipynb>) — red neuronal sobre Pima Indians
- [`M03_S07 — Herramientas de Desarrollo de IA`](<notebooks/OD_GPIA_ESP_M03_S07_Herramientas de Desarrollo de IA - Recurso descargable.ipynb>) — sweetviz, dataprep, plotly, bokeh, polars
- [`M03_S07 — TensorFlow, Keras y PyTorch`](<notebooks/OD_GPIA_ESP_M03_S07_Tensorflow_Keras_Pytorch_dentro_del_deep_learning - Recurso descargable.ipynb>) — la misma red en los dos frameworks
- [`M04_S10 — Caso de Arte`](<notebooks/OD_GPIA_ESP_M04_S10_Caso de Arte - Recurso descargable.ipynb>) — transferencia de estilo con VGG-19
- [`M04_S11 — Contratación con herramientas de IA/ML`](<notebooks/OD_GPIA_ESP_M04_S11_¿Cómo contratar mediante herramientas de AI _ ML al mejor capital humano_ - Recurso Descargable.ipynb>) — RandomForest sobre datos de RRHH
- [`M06_S15 — Prácticas y herramientas en MLOps`](<notebooks/OD_GPIA_ESP_M06_S15_Prácticas y herramientas en MLOps - Recurso descargable.ipynb>) — forecasting de MELI con ARIMA y tracking en W&B

> **Nota de seguridad.** El recurso de MLOps venía con la clave de API de Weights & Biases del autor del material hardcodeada en el código y grabada en el output. Se reemplazó por un marcador antes de versionarlo: es la única modificación al código original del curso en todo el repositorio. El detalle está en el propio notebook.

## Material en R

Las Clases 8 y 9 del Módulo 3 son el único material en R del programa. Los dos `.Rmd` están en `notebooks/`, junto a un apunte que los explica para quien viene de Python: [`R — notas de los .Rmd`](<notebooks/OD_GPIA_ESP_M03_S08_R_notas_de_los_Rmd.md>).

> El dataset de la Clase 3 (`Datos - Análisis de imágenes en el sector industrial/`, ~168 MB descomprimidos) está ignorado en `.gitignore` por su peso: no se versiona la carpeta, solo el código que la referencia.

[← Volver al índice](../README.md)
