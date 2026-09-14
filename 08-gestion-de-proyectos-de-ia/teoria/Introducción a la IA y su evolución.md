# Introducción a la IA y su evolución

> Conversión a Markdown de las slides del curso (Clase 2 — Introducción al Desarrollo de Proyectos de IA, arranca el **Módulo 2**). El PDF original está en [`material/`](<material/Introducción al desarrollo de Proyectos de IA.pdf>).

## 1. Qué es la Inteligencia Artificial

El material define la **Inteligencia Artificial (IA)** como un campo de estudio de la informática centrado en la creación de sistemas capaces de realizar tareas que normalmente requieren inteligencia humana: reconocimiento de voz, toma de decisiones, traducción de idiomas y visión por computadora.

> **Nota:** la definición del material es intencionalmente amplia — describe la IA por sus tareas, no por un mecanismo técnico único. Eso es correcto como punto de partida, pero conviene tener presente que "IA" hoy es paraguas de técnicas muy distintas (desde árboles de decisión hasta redes neuronales de miles de millones de parámetros), y el resto de la clase se dedica justamente a desglosar esas ramas.

## 2. Ramas principales

El material ubica tres grandes círculos, unos dentro de otros:

```
Inteligencia Artificial
  └── Machine Learning (ML)
        └── Deep Learning (DL)
  └── Symbolic AI (IA simbólica, basada en reglas)
```

- **Machine Learning (ML):** algoritmos que aprenden patrones a partir de datos, sin ser programados explícitamente para cada regla. Ejemplos del material: detección de fraudes, recomendaciones de productos, sistemas de reconocimiento de voz.
- **Deep Learning (DL):** subcampo del ML basado en redes neuronales profundas (ver el módulo anterior, [07-fundamentos-de-deep-learning](<../../07-fundamentos-de-deep-learning/>)). Ejemplos del material: conducción autónoma, diagnósticos médicos, asistentes virtuales.
- **Symbolic AI (IA simbólica):** el material la ubica como una rama separada dentro de la IA, sin desarrollarla — corresponde al enfoque más antiguo de la IA, basado en reglas lógicas explícitas y sistemas expertos (ver la sección de historia más abajo, con MYCIN en 1970), en contraste con el aprendizaje estadístico del ML.

> **Nota:** el material no explica por qué el Deep Learning aparece *dentro* de Machine Learning y no al lado. La relación es de inclusión: toda red neuronal profunda es, técnicamente, un modelo de Machine Learning (aprende de datos etiquetados o no), pero no todo modelo de ML es una red neuronal — el diagrama de círculos anidados representa exactamente esa relación de subconjunto.

## 3. Breve historia de la Inteligencia Artificial

El material presenta una línea de tiempo con los hitos más citados del campo, dividida en dos tablas.

### Orígenes (1950–2006)

| Año | Hito |
|---|---|
| 1950 | Alan Turing publica *Computing Machinery and Intelligence*, proponiendo el famoso **Test de Turing**. |
| 1956 | John McCarthy organiza la conferencia de Dartmouth, donde se acuña el término **"Inteligencia Artificial"**. |
| 1966 | Se desarrolla **ELIZA**, uno de los primeros programas de procesamiento del lenguaje natural. |
| 1970 | Aparecen los **sistemas expertos**, como MYCIN, utilizado en medicina. |
| 1986 | Geoffrey Hinton y otros popularizan el uso de las **redes neuronales** mediante el algoritmo de **retropropagación**. |
| 1997 | **Deep Blue** de IBM derrota al campeón mundial de ajedrez Garry Kasparov. |
| 2006 | Geoffrey Hinton introduce el concepto de **"aprendizaje profundo"** (deep learning). |

### Consolidación y explosión moderna (2011–2023)

| Año | Hito |
|---|---|
| 2011 | **IBM Watson** gana el concurso Jeopardy!, superando a los campeones humanos. |
| 2012 | **AlexNet** gana el concurso ImageNet, marcando un hito en el reconocimiento de imágenes con redes neuronales profundas. |
| 2016 | **AlphaGo** de DeepMind vence al campeón mundial de Go, un juego mucho más complejo que el ajedrez. |
| 2020 | **GPT-3**, un modelo de lenguaje con 175 mil millones de parámetros, es lanzado por OpenAI. |
| 2023 | **ChatGPT-4** es lanzado, mostrando avances significativos en generación de texto natural y conversación. |

> **Nota:** el material salta de 2006 (aprendizaje profundo como concepto) a 2011 (Watson) sin mencionar 2012 como el año de AlexNet en la misma tabla que 2006 — quedan repartidos en dos tablas del PDF original por razones de diagramación de slide, no por un corte conceptual real. Se muestran acá igual, como en el original, para no perder la referencia de fuente ("Elaboración propia - Julio Paredes").

## 4. Tipos de IA: débil vs. fuerte

El material distingue dos categorías según la **amplitud de las capacidades** del sistema:

| Aspecto | IA Específica (IA Débil) | IA General (IA Fuerte) |
|---|---|---|
| Propósito y alcance | Tareas específicas y limitadas | Capacidades cognitivas amplias, similar a la humana |
| Flexibilidad y adaptabilidad | Limitada a su dominio | Capacidad de aprender y adaptarse a múltiples dominios |
| Ejemplos | Chatbots, asistentes virtuales, IA en juegos | No existe aún, es un concepto teórico |
| Nivel de autonomía | Funciona dentro de su programación | Autonomía completa, aprende y toma decisiones por sí misma |
| Implicaciones éticas y sociales | Menos complejas, específicas | Desafíos éticos y sociales significativos, control y seguridad |

- **IA débil (weak AI):** toda la IA que existe hoy en producción. Ejemplos del material: asistentes virtuales (Siri, Alexa), algoritmos de recomendación (Netflix, Amazon), sistemas de visión artificial para reconocimiento de imágenes o rostros.
- **IA fuerte (strong AI):** un objetivo teórico de investigación, no un producto existente. La idea es una máquina que pueda pensar, razonar y tener conciencia, con autonomía comparable a la humana.

> **Nota:** el material es explícito y correcto en un punto clave que conviene remarcar: **la IA fuerte no existe actualmente**. Todo lo que se llama "IA" en la industria — incluidos los modelos de lenguaje grandes como GPT-4, mencionados como hito de 2023 — es IA débil: sistemas entrenados para tareas o familias de tareas específicas, por más versátiles que parezcan, sin comprensión ni conciencia en el sentido que implica la IA general.

## 5. Impacto de la IA en sectores clave (Clase 3)

El material presenta tres casos de estudio, sin desarrollo técnico en las slides mismas (quedan para las clases prácticas en video):

- **Diagnóstico de diabetes en la comunidad pima indígena de Norteamérica:** un caso clásico de datasets tabulares en ML (Pima Indians Diabetes Dataset) usado como ejemplo de código de una red neuronal profunda aplicada a un problema real de salud, con foco en el impacto social de la IA en una comunidad específica y una mirada desde la óptica empresarial.
- **Análisis de imágenes en el sector industrial:** aplicaciones de *computer vision* — el material muestra como ejemplo visual una imagen con **segmentación semántica** (clasificación de cada píxel de una escena urbana en categorías como `car`, `building`, `road`, `tree`, `sky`), ilustrando qué tipo de tarea resuelve la visión por computadora en un contexto industrial/urbano.
- **Estudio de caso — DILLO.AI:** plataforma en desarrollo para fomentar la comunicación entre personas sordas y oyentes mediante un intérprete de IA (Dillo.Ai), presentado como ejemplo de aplicación de IA con impacto social directo en la comunidad hipoacúsica.

> **Nota:** el material no detalla la arquitectura ni el dataset de segmentación semántica de la imagen industrial, ni el modelo detrás de Dillo.Ai — son estudios de caso a nivel de "qué problema resuelve la IA", no implementaciones técnicas documentadas en la slide. El dataset del diagnóstico de diabetes en la comunidad pima sí tiene una implementación en código asociada a este módulo, que se documenta por separado en los notebooks.

## 6. Fases del Proyecto de IA (Clase 4)

El material presenta cuatro fases con su descripción, pasos clave y una quinta transversal (integración ágil):

### Fase 1 — Definición del Problema

**Descripción:** se centra en entender el problema a resolver, identificando los objetivos y requisitos específicos del sistema de IA.

**Pasos clave:**
- **Recolección de requisitos:** documentar qué necesita el cliente o usuario.
- **Análisis de viabilidad:** evaluar si el problema se puede resolver con IA.
- **Definición de métricas de éxito:** determinar cómo se medirá el éxito del proyecto.

### Fase 2 — Recopilación de Datos

**Descripción:** obtener los datos necesarios y asegurarse de que sean de buena calidad, esencial para entrenar modelos de IA.

**Pasos clave:** identificar fuentes de datos, adquirir datos relevantes, aplicar técnicas de limpieza como eliminación de valores nulos y corrección de inconsistencias.

### Fase 3 — Desarrollo y Entrenamiento de Modelos

**Descripción:** se seleccionan los algoritmos más adecuados, se entrenan los modelos con los datos y se validan para asegurar su rendimiento.

**Pasos clave:** selección del algoritmo, entrenamiento del modelo con datos de entrenamiento, validación usando técnicas como validación cruzada. El material ilustra el split habitual de **80% entrenamiento / 20% test** (`Train Data` / `Test Data`).

### Fase 4 — Despliegue y Monitoreo

**Descripción:** implementar el modelo en el entorno de producción, configurar sistemas de monitoreo para detectar fallos o desviaciones, y ajustar el modelo según sea necesario.

**Herramientas mencionadas por el material:**
- **Grafana** y **Prometheus** — monitoreo y alertas.
- **MLflow** — gestión del ciclo de vida de modelos de ML (tracking de experimentos, versionado, registro de modelos).

> **Nota:** el material nombra Grafana, Prometheus y MLflow en la fase de despliegue sin explicar el rol específico de cada una. En términos generales: Prometheus recolecta métricas en el tiempo, Grafana las visualiza en dashboards, y MLflow versiona modelos y experimentos — son herramientas complementarias, no alternativas entre sí, un detalle que el material no aclara.

### Transversal — Integración con las metodologías ágiles

**Descripción:** enfoques de gestión de proyectos que promueven la entrega incremental y continua de valor mediante ciclos cortos de desarrollo, colaboración constante con los interesados y adaptación al cambio. Ejemplos del material: **Scrum**, **Kanban** y **Extreme Programming (XP)**.

**Pasos clave:**
- **Sprints de desarrollo (Scrum):** ciclos cortos para diseñar, entrenar y ajustar modelos de IA.
- **Revisiones y retrospectivas (Scrum):** evaluar los resultados del sprint y ajustar el enfoque.
- **Kanban:** pasos similares pero con controles mediante tableros interactivos, acotado a que el objetivo es realizar proyectos de manera organizada y en equipo.

> **Nota:** el material no profundiza en Extreme Programming (XP) más allá de nombrarlo como ejemplo — a diferencia de Scrum y Kanban, no aparece desarrollado en pasos clave dentro de esta slide.

## 7. Relación con el resto del módulo

Estas cuatro fases (definición del problema → datos → modelo → despliegue) son el hilo conductor de todo el curso: los módulos siguientes profundizan cada una — el Módulo 3 amplía las herramientas técnicas de la fase de desarrollo (frameworks, plataformas en la nube), el Módulo 4 profundiza la gestión de equipos alrededor de estas fases, el Módulo 5 se enfoca en el despliegue y la escalabilidad, y el Módulo 6 cierra con MLOps como la disciplina que formaliza el monitoreo y mantenimiento continuo mencionado en la Fase 4.
