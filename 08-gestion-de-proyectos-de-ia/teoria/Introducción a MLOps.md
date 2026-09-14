# Introducción a MLOps

> Conversión a Markdown de las slides del curso (Clase 15 — Fundamentos de MLOps e implementaciones, **Módulo 6**). El PDF original está en [`material/`](<material/Introducción a MLOps.pdf>).

> **Nota sobre el material:** este es el PDF más escueto de todo el módulo — 8 slides, de las cuales solo una (la slide 6) tiene contenido propio, y ese contenido es una introducción genérica que remite el desarrollo real a un notebook de Jupyter de la clase práctica ("vamos a codear una serie temporal... en donde aplicaremos al final del código el concepto que concierne a esta unidad MLOps"). El PDF **no define qué es MLOps**, ni desarrolla ninguna de las herramientas que el programa del curso promete para este módulo (Jenkins, GitHub Actions, Prometheus, Grafana, wandb — ver [`0 - Programa del módulo.md`](<0 - Programa del módulo.md>), Módulo 6). Este apunte amplía el contenido para que quede autocontenido, marcando explícitamente qué viene del material original y qué es ampliación.

## 1. Qué dice el material

El índice de la clase (slide 4) anuncia dos puntos:

1. Definición y Objetivos de MLOps.
2. Prácticas y Herramientas en MLOps.

Pero la única slide de desarrollo (slide 6, título "MLOps") no cubre ninguno de los dos con una definición formal. Textualmente, dice que la unidad va a "codear una serie temporal" y aplicar, al final del código, "el concepto que concierne a esta unidad": MLOps. Agrega que se van a explicar "dentro de nuestra Jupyter las bases teóricas que hablan sobre este módulo", y cierra invitando a "descubrir juntos la magia de compartir de manera interesante los resultados de nuestros proyectos de Machine Learning y AI gracias al MLOps", con un enlace de referencia a un artículo sobre **W&B (Weights & Biases)** como plataforma de MLOps (`npogeant.medium.com/wandb-the-best-mlops-platform`).

La conclusión de la clase (slide 7) sí es más específica y menciona una herramienta concreta:

> "El uso de MLOps en series temporales y AI con herramientas como W&B (Weights and Biases) facilita la monitorización, gestión y mejora continua de modelos en producción. Permite el seguimiento detallado de experimentos, ajuste de hiperparámetros y control de versiones, mejorando la precisión y eficiencia en el manejo de datos temporales. Además, optimiza el ciclo de vida del modelo, asegurando despliegues robustos y eficientes en entornos reales."

> **Nota:** en las slides no aparece contenido teórico sobre Jenkins, GitHub Actions, Prometheus ni Grafana, a pesar de que el programa oficial del módulo los promete. Es probable que se desarrollen únicamente en el notebook de la clase práctica (que no forma parte del material entregado como PDF), o que el programa esté desactualizado respecto a la versión final de la clase. Quedan documentados igual en la sección 3, con las salvedades correspondientes, porque son herramientas centrales del ecosistema MLOps y el módulo ya las nombró como objetivo de aprendizaje.

## 2. Qué es MLOps (ampliación)

Dado que el material no lo define, va la definición estándar del concepto: **MLOps** (*Machine Learning Operations*) es la aplicación de las prácticas de **DevOps** (integración y despliegue continuo, automatización, monitoreo) al ciclo de vida de modelos de Machine Learning. Nace porque entrenar un modelo en un notebook es solo una fracción del trabajo real: un modelo de producción necesita, además, versionado de datos y de modelo, pipelines reproducibles de entrenamiento, pruebas automatizadas, despliegue controlado, monitoreo de su desempeño en producción y un proceso claro de reentrenamiento cuando el modelo se degrada (*model drift*).

Las tres actividades que menciona la conclusión del material — seguimiento de experimentos, ajuste de hiperparámetros y control de versiones — son, efectivamente, tres de los pilares centrales de MLOps:

- **Seguimiento de experimentos (*experiment tracking*):** registrar de forma sistemática qué configuración de datos, hiperparámetros y código produjo cada modelo entrenado, junto con sus métricas de resultado, para poder comparar corridas y reproducir la mejor.
- **Ajuste de hiperparámetros:** ya visto en el Módulo 5 como técnica de mejora de rendimiento — en MLOps se automatiza y se registra como parte del tracking de experimentos.
- **Control de versiones:** no solo del código (como en desarrollo de software tradicional con Git), sino también de los **datos** de entrenamiento y de los **modelos** entrenados en sí, porque un mismo código con datos distintos produce modelos distintos.

## 3. Herramientas del ecosistema MLOps (ampliación)

El material nombra explícitamente solo W&B; el programa del módulo agrega Jenkins, GitHub Actions, Prometheus, Grafana. Van explicadas, agrupadas por el rol que cumple cada una — algo que ni el material ni el programa aclaran:

| Herramienta | Rol en MLOps |
|---|---|
| **Weights & Biases (W&B)** | Plataforma de *experiment tracking*: registra automáticamente hiperparámetros, métricas y artefactos (modelos, gráficos) de cada corrida de entrenamiento, y permite compararlas en un dashboard. Es la única que el material desarrolla mínimamente. |
| **MLflow** | Alternativa (a menudo open source, autohospedable) a W&B para lo mismo: tracking de experimentos, registro y versionado de modelos. Ya había aparecido en el Módulo 2 de este mismo curso, sección de Fase 4 (Despliegue y Monitoreo), sin desarrollo técnico en ese momento tampoco. |
| **Jenkins** | Servidor de automatización de **CI/CD** (integración y despliegue continuo) autohospedado. En MLOps se usa para automatizar pipelines: cada vez que hay código o datos nuevos, dispara automáticamente pasos como reentrenar el modelo, correr pruebas y desplegarlo. |
| **GitHub Actions** | Equivalente a Jenkins pero integrado nativamente en GitHub: define flujos de CI/CD como archivos de configuración dentro del propio repositorio, sin necesidad de mantener un servidor aparte. |
| **Docker** (visto en el Módulo 5) | Empaqueta el modelo con su entorno exacto de dependencias para que el despliegue sea reproducible entre el entorno de desarrollo y el de producción. Es habitual que los pipelines de Jenkins o GitHub Actions terminen construyendo una imagen Docker con el modelo listo para desplegar. |
| **Prometheus** | Sistema de recolección de métricas en el tiempo (series temporales de métricas de sistema y de la aplicación): cuántas requests recibe el modelo en producción, cuánto tarda en responder, cuánta memoria usa, etc. |
| **Grafana** | Herramienta de visualización que se conecta a Prometheus (u otras fuentes de métricas) para armar dashboards y configurar alertas — por ejemplo, avisar si la latencia del modelo supera un umbral, retomando el concepto de latencia visto en el Módulo 5. |

> **Nota:** Prometheus y Grafana no son específicas de Machine Learning — son herramientas estándar de monitoreo de infraestructura en general (el mismo stack se usa para monitorear cualquier aplicación web), que en MLOps se aplican también a modelos en producción monitoreando tanto métricas de sistema (latencia, uso de CPU/memoria) como métricas propias del modelo (precisión, distribución de las predicciones a lo largo del tiempo, para detectar *model drift*). El material no hace esta distinción entre "monitoreo de infraestructura general" y "monitoreo específico de modelos", y es un matiz importante: un modelo puede estar respondiendo rápido y sin errores (buena salud de infraestructura) y al mismo tiempo estar prediciendo cada vez peor porque el mundo real cambió respecto a los datos con los que se entrenó (mala salud del modelo). MLOps necesita vigilar ambas cosas.

## 4. Relación con el resto del módulo

Este módulo cierra el recorrido técnico del curso: retoma el monitoreo mencionado ya en la Fase 4 del Módulo 2, la infraestructura de Docker/GPU/CPU del Módulo 5, y les da nombre formal como disciplina — MLOps — que asegura que un modelo entrenado por el equipo del Módulo 4 no solo se despliegue una vez, sino que se mantenga, se monitoree y se reentrene de forma sostenible en producción. El Módulo 7 cierra el curso con una evaluación integral de todo lo anterior.
