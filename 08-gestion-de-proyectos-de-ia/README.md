# Gestión de proyectos de IA

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 7 | 18 | 🟨 En curso — teoría de los módulos 2 a 6 documentada |

> Continúa [Fundamentos de Deep Learning](../07-fundamentos-de-deep-learning/): de entrenar redes neuronales se pasa a la **gestión integral de proyectos de IA** — definición del problema, herramientas y frameworks, gestión de equipos, despliegue, escalabilidad y MLOps.

## Teoría

> Los **6 PDF y 1 PPTX originales** convertidos hasta ahora están en [`teoria/material/`](teoria/material/); en `teoria/` quedan sus conversiones a Markdown.

- [`0-programa-del-modulo.md`](<teoria/0-programa-del-modulo.md>) — las 18 clases en 7 módulos
- [`introduccion-a-la-ia-y-su-evolucion.md`](<teoria/introduccion-a-la-ia-y-su-evolucion.md>) — Clases 2 a 4, Módulo 2: definición de IA, ramas (ML/DL/IA simbólica), breve historia, IA débil vs. fuerte, casos de impacto sectorial y las cuatro fases de un proyecto de IA
- [`integracion-de-tecnologias-de-ia.md`](<teoria/integracion-de-tecnologias-de-ia.md>) — Clases 6 a 8, Módulo 3: ML/DL/NLP/CV aplicados, lenguajes y frameworks (Python, R, TensorFlow, Keras, PyTorch), plataformas en la nube (Google Cloud AI, AWS SageMaker, Azure ML), y el caso práctico de RStudio + Shiny + ChatGPT
- [`desarrollo-y-gestion-de-proyectos-de-ia.md`](<teoria/desarrollo-y-gestion-de-proyectos-de-ia.md>) — Clases 10 y 11, Módulo 4: definición del problema con un caso de transfer learning / neural style transfer, y gestión de equipos (roles, habilidades, comunicación, contratación asistida por IA)
- [`despliegue-y-escalabilidad-de-proyectos-de-ia.md`](<teoria/despliegue-y-escalabilidad-de-proyectos-de-ia.md>) — Clase 13, Módulo 5: escalabilidad horizontal y vertical, recursos en la nube, reducción de latencia, GPU vs. CPU y contenedores (Docker)
- [`introduccion-a-mlops.md`](<teoria/introduccion-a-mlops.md>) — Clase 15, Módulo 6: definición de MLOps, experiment tracking y control de versiones de modelos (W&B, MLflow), y el stack de CI/CD y monitoreo (Jenkins, GitHub Actions, Prometheus, Grafana)

Queda pendiente convertir la teoría del módulo 7 (Cierre de curso) a medida que se sume el material correspondiente.

## Notebooks

Los **7 notebooks** de [`notebooks/`](notebooks/) están documentados, con los hallazgos sobre el material anotados en cada uno.

- [`M02_S03 — Análisis de imágenes en el sector industrial`](<notebooks/od-gpia-esp-m02-s03-analisis-de-imagenes-en-el-sector-industrial-recurso-descargable.ipynb>) — KNN sobre imágenes de piezas fundidas
- [`M02_S03 — Diagnóstico de diabetes`](<notebooks/od-gpia-esp-m02-s03-desarrollo-de-proyecto-diagnostico-de-diabetes-recurso-descargable.ipynb>) — red neuronal sobre Pima Indians
- [`M03_S07 — Herramientas de Desarrollo de IA`](<notebooks/od-gpia-esp-m03-s07-herramientas-de-desarrollo-de-ia-recurso-descargable.ipynb>) — sweetviz, dataprep, plotly, bokeh, polars
- [`M03_S07 — TensorFlow, Keras y PyTorch`](<notebooks/od-gpia-esp-m03-s07-tensorflow-keras-pytorch-dentro-del-deep-learning-recurso-descargable.ipynb>) — la misma red en los dos frameworks
- [`M04_S10 — Caso de Arte`](<notebooks/od-gpia-esp-m04-s10-caso-de-arte-recurso-descargable.ipynb>) — transferencia de estilo con VGG-19
- [`M04_S11 — Contratación con herramientas de IA/ML`](<notebooks/od-gpia-esp-m04-s11-como-contratar-mediante-herramientas-de-ai-ml-al-mejor-capital-humano-recurso-descargable.ipynb>) — RandomForest sobre datos de RRHH
- [`M06_S15 — Prácticas y herramientas en MLOps`](<notebooks/od-gpia-esp-m06-s15-practicas-y-herramientas-en-mlops-recurso-descargable.ipynb>) — forecasting de MELI con ARIMA y tracking en W&B

> **Nota de seguridad.** El recurso de MLOps venía con la clave de API de Weights & Biases del autor del material hardcodeada en el código y grabada en el output. Se reemplazó por un marcador antes de versionarlo: es la única modificación al código original del curso en todo el repositorio. El detalle está en el propio notebook.

## Material en R

Las Clases 8 y 9 del Módulo 3 son el único material en R del programa. Los dos `.Rmd` están en `notebooks/`, junto a un apunte que los explica para quien viene de Python: [`R — notas de los .Rmd`](<notebooks/od-gpia-esp-m03-s08-r-notas-de-los-rmd.md>).

> El dataset de la Clase 3 (`Datos - Análisis de imágenes en el sector industrial/`, ~168 MB descomprimidos) está ignorado en `.gitignore` por su peso: no se versiona la carpeta, solo el código que la referencia.

[← Volver al índice](../README.md)
