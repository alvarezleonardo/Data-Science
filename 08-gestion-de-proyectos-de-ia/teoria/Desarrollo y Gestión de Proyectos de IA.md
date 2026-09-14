# Desarrollo y Gestión de Proyectos de IA

> Conversión a Markdown de las slides del curso — **Módulo 4**. Clase 10 (Definición del Problema y Objetivos) está en el PDF [`material/Desarrollo y Gestión de Proyectos de IA - Parte 1.pdf`](<material/Desarrollo y Gestión de Proyectos de IA - Parte 1.pdf>); Clase 11 (Gestión de Equipos) está en el PPTX [`material/Desarrollo de Gestión de Proyectos de IA - Parte 2.pptx`](<material/Desarrollo de Gestión de Proyectos de IA - Parte 2.pptx>). Nótese que el título del archivo PPTX difiere levemente del PDF ("de Gestión" en vez de "y Gestión") — es un error de tipeo del material original, no de esta conversión.

> **Nota sobre el material:** ambos archivos son considerablemente más escuetos que el resto del módulo — el PDF de la Clase 10 tiene apenas 8 slides (portada, índice, dos citas motivacionales, título de sección, una única slide de contenido, conclusión y cierre) y el PPTX de la Clase 11 tiene 10 slides con contenido real recién a partir de la slide 4. El contenido efectivo remite en varios puntos a "los próximos videos" de la clase práctica — es decir, buena parte del desarrollo queda fuera de las slides y vive en el video de la clase, no documentado acá.

## 1. Definición del Problema y Objetivos (Clase 10)

El material organiza esta clase en cuatro puntos, presentados como agenda de "lo que se profundiza en los próximos videos" más que como contenido desarrollado en la slide misma:

- **Fases críticas en el desarrollo de proyectos de IA:** desde la idea hasta el prototipo.
- **Identificación del Problema:** cómo identificar un problema que puede ser abordado con IA.
- **Formulación de objetivos:** definición clara y específica de los objetivos del proyecto.
- **Ejemplo práctico:** aprendizaje por transferencia (*transfer learning*) en *computer vision*, con un caso de estilo fotografía y arte.

> **Nota:** el material no explica ninguno de estos cuatro puntos en la slide — los nombra como índice de lo que se va a cubrir en video. Para no dejar el apunte vacío, van las definiciones que corresponden a cada término:
> - **Identificación del problema en IA:** no todo problema es apto para IA. El criterio habitual es evaluar si existe suficiente volumen de datos históricos relevantes, si el problema tiene un patrón aprendible (no es puramente aleatorio) y si el costo de una solución basada en IA se justifica frente a alternativas más simples (reglas de negocio, heurísticas).
> - **Aprendizaje por transferencia (*transfer learning*):** técnica de Deep Learning en la que se reutiliza un modelo ya entrenado en una tarea grande (por ejemplo, clasificación de imágenes sobre millones de fotos) como punto de partida para una tarea nueva y más específica, en vez de entrenar una red desde cero. Ahorra tiempo de entrenamiento y funciona bien con datasets chicos, porque las primeras capas de la red ya "saben" reconocer patrones visuales genéricos (bordes, texturas, formas).
> - **Transferencia de estilo neural (*neural style transfer*):** el caso concreto que cita el material (fotografía + arte) es una aplicación puntual de transfer learning: una red entrenada en reconocimiento de imágenes se usa para combinar el contenido de una foto con el estilo visual de una pintura (el ejemplo clásico es aplicar el estilo de *La noche estrellada* de Van Gogh a una foto de San Francisco). El material enlaza como referencia un hilo del foro de fast.ai (`forums.fast.ai/t/neural-style-transfer-using-s4tf/45128`), sin desarrollarlo en la slide.

### Conclusión de la clase

El material cierra con una cita atribuida a Julio Paredes (docente/autor del contenido) que resume el enfoque práctico de la clase: la idea de "seguir el código, transcribir, modificar, intuir y explorar" un proyecto de transfer learning de punta a punta, desde la negociación con el cliente hasta la entrega del producto final, como forma de simular un proyecto real de IA.

## 2. Gestión de Equipos (Clase 11)

El material organiza la clase en cuatro puntos, desarrollados esta vez con una slide de contenido por punto:

### 2.1 Roles y responsabilidades

El material define tres roles clave en un equipo de IA:

| Rol | Responsabilidad según el material |
|---|---|
| **Data Scientist** | Analiza datos y desarrolla modelos predictivos. |
| **Ingeniero de Datos** | Construye y mantiene la infraestructura de datos, asegurando su calidad y disponibilidad. |
| **Desarrollador de Software** | Integra los modelos en aplicaciones prácticas, asegurando su despliegue y escalabilidad. |

El material agrega que, desde Recursos Humanos, es clave definir estos roles con claridad, promover la colaboración entre ellos y fomentar un entorno de aprendizaje continuo.

> **Nota:** esta es una simplificación razonable pero incompleta de los roles típicos en un equipo de IA. En equipos más grandes suele aparecer también el **MLOps Engineer** (responsable específicamente de la infraestructura de despliegue y monitoreo de modelos en producción, tema que retoma este módulo más adelante) y el **Product Manager de IA** (traduce necesidades de negocio en requisitos técnicos). El material los omite, probablemente porque los absorbe dentro de "Desarrollador de Software" e "Ingeniero de Datos".

### 2.2 Formación y habilidades

El material distingue dos categorías de habilidades necesarias en el equipo:

- **Habilidades técnicas:** estadística, machine learning, programación, gestión de datos y herramientas de ETL (según el rol).
- **Habilidades blandas:** pensamiento crítico para resolver problemas complejos, comunicación efectiva para traducir conceptos técnicos a no técnicos, colaboración interdisciplinaria, adaptabilidad ante cambios tecnológicos y curiosidad para el aprendizaje continuo.

> **Nota:** el material nombra "herramientas de ETL" sin explicarlas. **ETL** significa *Extract, Transform, Load*: el proceso de extraer datos de una o más fuentes (bases de datos, APIs, archivos), transformarlos (limpieza, normalización, agregación) y cargarlos en un destino final (un data warehouse, por ejemplo) listo para análisis o entrenamiento de modelos. Es la base de cualquier pipeline de datos previo al modelado.

### 2.3 Comunicación y colaboración

El material propone tres categorías de herramientas para sostener la comunicación del equipo:

- **Gestión de proyectos:** Trello, Asana.
- **Mensajería y videoconferencia:** Slack, Microsoft Teams.
- **Repositorios de código compartido:** GitHub.

Como técnicas complementarias menciona reuniones regulares de sincronización, metodologías ágiles (ver Módulo 2, Fase transversal de integración ágil) y revisiones de código.

> **Nota:** el material lista estas herramientas sin distinguir su propósito, aunque en este caso son bastante autoexplicativas por categoría. Vale aclarar igual que "revisiones de código" (*code review*) no es solo una práctica de calidad: en equipos de IA cumple también un rol de transferencia de conocimiento entre roles con perfiles distintos (un Data Scientist revisando el código de un Ingeniero de Datos, por ejemplo), algo que el material no explicita.

### 2.4 Ejemplo práctico: contratación asistida por IA/ML

El material presenta como cuarto punto un caso práctico: usar herramientas de IA/ML para asistir en la selección del mejor capital humano, con un ejercicio de análisis de un caso real de selección de personal que se desarrolla en los videos de la clase (no en la slide).

> **Nota:** el material no aclara qué técnica de ML se usa en ese caso de selección de personal (¿clasificación de CVs?, ¿scoring de candidatos?, ¿NLP sobre entrevistas?) — queda pendiente de los videos prácticos, fuera del alcance de este documento. Vale una advertencia que el material tampoco menciona: el uso de IA en procesos de contratación es un área con riesgo conocido de **sesgo algorítmico** (por ejemplo, el caso público de Amazon en 2018, que descontinuó una herramienta de screening de CVs por discriminar candidatas mujeres al haber sido entrenada con datos históricos sesgados). Es un punto relevante de gestión de proyectos de IA — no solo técnico sino ético y legal — que el material no toca.

### Conclusión de la clase

El material cierra resumiendo que gestionar un equipo de IA de forma efectiva requiere: roles claros (Data Scientists, Ingenieros de Datos, Desarrolladores de Software) con habilidades técnicas específicas; formación continua y desarrollo de habilidades blandas; herramientas y técnicas de comunicación clara; y el uso de IA/ML en la propia selección de talento para conformar un equipo competente y diverso.

## 3. Relación con el resto del módulo

Este es el módulo que le da nombre al curso: retoma las cuatro fases de un proyecto de IA vistas en el Módulo 2 (definición del problema → datos → modelo → despliegue) y las profundiza desde dos ángulos — la definición técnica del problema con un caso concreto de transfer learning (Clase 10), y la gestión humana del equipo que lo ejecuta (Clase 11). El Módulo 5 continúa con la fase de despliegue y escalabilidad, y el Módulo 6 cierra con MLOps como la disciplina que formaliza el monitoreo y mantenimiento continuo del modelo ya en producción.
