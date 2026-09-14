# Despliegue y Escalabilidad de Proyectos de IA

> Conversión a Markdown de las slides del curso (Clase 13 — Técnicas y Herramientas para el Despliegue, **Módulo 5**). El PDF original está en [`material/`](<material/Despliegue y Escalabilidad de Proyectos de IA.pdf>).

## 1. Escalabilidad y rendimiento: definiciones

El material define **escalabilidad** como la capacidad de un sistema o modelo de IA para manejar un aumento en la carga de trabajo o volumen de datos sin perder eficiencia — es decir, que el modelo siga funcionando bien cuando se le agregan más datos o se le piden más tareas.

Como ejemplo práctico, el material propone un asistente virtual que responde preguntas: si solo una persona pregunta, responde rápido; si de repente 1.000 personas preguntan al mismo tiempo y el asistente sigue respondiendo sin perder velocidad, es escalable.

> **Nota:** el material no define **rendimiento** (*performance*) como concepto separado en esta sección, aunque lo usa en el título y lo retoma más adelante (sección 3). Conviene distinguirlos: la escalabilidad es sobre *crecer sin degradarse*, mientras que el rendimiento es sobre *qué tan rápido y eficiente es el sistema en un punto dado*, sin importar si crece o no. Un sistema puede tener buen rendimiento con poca carga y ser igual poco escalable (degradarse mal al crecer), o viceversa.

## 2. Importancia de la escalabilidad y el rendimiento

El material justifica ambos conceptos en términos de impacto de negocio:

- **Escalabilidad:** garantiza que el modelo de IA pueda crecer junto con el proyecto. Un modelo no escalable puede volverse lento o ineficiente frente a mayores volúmenes de datos, afectando la experiencia del usuario y los resultados del proyecto.
- **Rendimiento:** asegura que la IA haga su trabajo de manera efectiva. Un modelo con buen rendimiento reduce el tiempo de espera y aumenta la satisfacción del usuario, ayudando al proyecto a alcanzar sus objetivos más rápido.

## 3. Escalabilidad de Modelos de IA

El material distingue dos estrategias clásicas de escalado, y agrega el uso de recursos en la nube como una tercera vía que combina ambas.

### 3.1 Escalabilidad horizontal vs. vertical

| Tipo | Descripción según el material | Ejemplo del material |
|---|---|---|
| **Horizontal** | Añadir más instancias o servidores para distribuir la carga de trabajo. | Un motor de recomendación de productos ejecutándose en múltiples servidores, para manejar más usuarios simultáneos. |
| **Vertical** | Mejorar la capacidad de las instancias existentes (más memoria, CPU, almacenamiento). | Actualizar un servidor que procesa imágenes médicas para que tenga más potencia de cálculo y analice imágenes más rápido. |

> **Nota:** el material no menciona el trade-off más relevante entre ambas estrategias. El escalado vertical tiene un techo físico (hay un límite de CPU/memoria que se le puede agregar a una sola máquina) y un punto único de falla (si ese servidor cae, cae todo). El escalado horizontal no tiene ese techo y es más tolerante a fallas (si un servidor cae, los demás siguen respondiendo), pero es más complejo de implementar porque requiere balanceo de carga y, en muchos casos, que el sistema sea *stateless* (sin estado guardado en memoria de un servidor puntual) para repartir pedidos entre instancias sin problema.

### 3.2 Uso de recursos en la nube

El material propone aprovechar servicios en la nube (nombra **AWS**, **Azure** y **Google Cloud**, ya vistos como plataformas de IA en el Módulo 3) para escalar modelos de forma flexible y eficiente, combinando escalado horizontal (agregar instancias según demanda) y vertical (mejorar instancias existentes) de forma automática.

Ventajas que señala el material:

- **Flexibilidad:** ajuste dinámico de recursos según la carga.
- **Costos eficientes:** pagar solo por los recursos utilizados.

> **Nota:** el término técnico para el ajuste dinámico y automático que describe el material es **auto-scaling** (autoescalado): un mecanismo que, según reglas configuradas (por ejemplo, uso de CPU por encima del 70%), agrega o quita instancias automáticamente sin intervención manual. Los tres proveedores que nombra el material ofrecen esta funcionalidad, aunque con nombres de producto distintos entre sí — el material no entra en ese nivel de detalle, y este apunte tampoco lo hace por no poder verificar la oferta comercial vigente de cada proveedor al momento de escribir esto.

## 4. Rendimiento de Modelos de IA

El material se enfoca en la **reducción de latencia** como técnica central de mejora de rendimiento. Define **latencia** como el tiempo que un modelo de IA tarda en producir un resultado después de recibir una entrada, y propone tres técnicas para reducirla:

- **Optimización del código:** mejorar la eficiencia del código del modelo, eliminando operaciones innecesarias o complejas.
- **Uso de hardware especializado:** implementar aceleradores como **GPU** (unidades de procesamiento gráfico) o **TPU** (unidades de procesamiento tensorial) para cálculos más rápidos.
- **Ajuste de hiperparámetros:** variables que controlan el comportamiento del modelo de IA, como la tasa de aprendizaje o el número de capas de una red neuronal.

La conclusión general del material (sección 6) agrega dos técnicas más que no llegan a desarrollarse en una slide propia: **caché de resultados** y **preprocesamiento eficiente**.

> **Nota:** el material nombra GPU y TPU sin explicar la diferencia entre ellas. Una **GPU** (*Graphics Processing Unit*) es un procesador de propósito más general optimizado para operaciones matriciales masivamente paralelas — de ahí que sirva tanto para gráficos como para entrenar redes neuronales. Una **TPU** (*Tensor Processing Unit*) es un chip diseñado específicamente por Google para operaciones de tensores propias de Deep Learning (multiplicación de matrices a gran escala), lo que la hace más eficiente que una GPU para ese caso de uso puntual, pero menos flexible para otras cargas de trabajo.
>
> Sobre **caché de resultados**: consiste en guardar la salida de un modelo para una entrada ya procesada, de forma que si esa misma entrada (o una muy similar) vuelve a pedirse, se devuelve el resultado guardado en vez de volver a correr el modelo — reduce latencia a costa de memoria adicional y de la posibilidad de servir una respuesta desactualizada si el modelo se reentrena. El material lo menciona solo en la conclusión, sin slide de desarrollo propia.

## 5. Consideraciones de infraestructura

El material compara dos tipos de hardware y dos técnicas de gestión de infraestructura:

- **GPU:** ideal para tareas con procesamiento paralelo intensivo, como entrenamiento de redes neuronales profundas.
- **CPU** (Unidad de Procesamiento Central): adecuada para cargas de trabajo secuenciales y tareas más ligeras de procesamiento.
- **Balance de carga dinámico:** distribuir las tareas entre múltiples GPUs o CPUs para evitar sobrecarga en un solo recurso.
- **Virtualización y contenedores:** el material nombra específicamente **Docker** para aislar aplicaciones y maximizar la utilización del hardware disponible.

> **Nota:** el material menciona Docker sin explicar qué es. **Docker** es una plataforma de **contenedores**: empaqueta una aplicación junto con todas sus dependencias (librerías, versión de Python, variables de entorno) en una unidad portable que corre igual en cualquier máquina que tenga Docker instalado, sin los conflictos de "en mi máquina funciona" que genera instalar dependencias directamente sobre el sistema operativo. Es una pieza central en el despliegue de modelos de IA porque permite mover un modelo entrenado (con su entorno exacto de librerías) desde la laptop de desarrollo hasta un servidor de producción sin sorpresas de compatibilidad. El material lo nombra solo de pasada, en el contexto de virtualización de infraestructura, pero su uso real en MLOps es más amplio de lo que sugiere la slide — retoma protagonismo en el Módulo 6.

## 6. Conclusión de la clase

El material cierra remarcando que escalabilidad y rendimiento son esenciales para el éxito de los proyectos de IA: la escalabilidad (horizontal y vertical) permite ajustar recursos según necesidad, mientras que mejorar el rendimiento (latencia, hiperparámetros, caché, preprocesamiento) asegura respuestas rápidas y precisas. La correcta selección de infraestructura (GPU vs. CPU) junto con estrategias de optimización de recursos maximiza la eficiencia de los sistemas de IA.

## 7. Relación con el resto del módulo

Esta clase profundiza técnicamente la Fase 4 (Despliegue y Monitoreo) del proyecto de IA vista en el Módulo 2, y complementa la gestión de equipos del Módulo 4 con la infraestructura técnica que ese equipo necesita operar. El Módulo 6 retoma varios de estos conceptos (Docker en particular) bajo el paraguas de MLOps, formalizando cómo se automatiza y monitorea todo esto en producción.
