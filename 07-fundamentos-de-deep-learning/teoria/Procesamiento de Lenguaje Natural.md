# Procesamiento de Lenguaje Natural

> Conversión a Markdown de las slides del curso (Clase 21 — Procesamiento de Lenguaje Natural, NLP). El PDF original está al lado.

## 1. Qué es el NLP

El procesamiento del lenguaje natural (NLP, por *Natural Language Processing*) es una rama del aprendizaje automático enfocada en definir mecanismos con los que las computadoras puedan comprender y manipular el lenguaje natural de las personas: texto y habla tal como los produce alguien, con toda su ambigüedad, en oposición a un lenguaje formal diseñado para máquinas.

El material ubica el NLP como uno de los campos que más se desarrolló y popularizó en los últimos años, y menciona sus dos piezas técnicas centrales, que son el resto de esta clase:

- Representaciones numéricas conocidas como **embeddings**.
- Un procedimiento previo llamado **tokenización**, con tres variantes posibles: a nivel de carácter, de palabra o de subpalabra.

La slide se queda en esa enumeración. El resto de esta sección desarrolla cada punto, porque son la base que hace falta entender antes de llegar a los modelos seq2seq y de atención de las próximas dos clases.

> **Nota:** todo lo que sigue en las secciones 2, 3 y 4 es ampliación fuera de la slide — la slide original tiene una sola diapositiva de contenido (la segunda de cuatro páginas del PDF es la portada de sección, y la cuarta es el cierre "¡Muchas gracias!"). Se marca explícitamente dónde termina lo dado por el curso y dónde empieza el desarrollo agregado.

## 2. Por qué hace falta tokenizar

Una red neuronal no puede recibir texto crudo: necesita números. La **tokenización** es el primer paso de cualquier pipeline de NLP, y consiste en partir una cadena de texto en unidades discretas, llamadas *tokens*, que después se van a mapear a números (ver la sección de embeddings). La pregunta de fondo es: ¿cuál es la unidad mínima con la que trabaja el modelo? De esa decisión depende el tamaño del vocabulario, el largo de las secuencias y qué tan bien el modelo generaliza a palabras que nunca vio.

El material menciona tres estrategias, sin desarrollarlas. Acá se comparan:

| Estrategia | Unidad | Tamaño de vocabulario | Largo de secuencia | Problema principal |
|---|---|---|---|---|
| Por palabra | Cada palabra completa | Muy grande (decenas o cientos de miles de términos) | Corto | Palabras nuevas o poco frecuentes quedan fuera del vocabulario |
| Por carácter | Cada letra o símbolo | Mínimo (decenas de símbolos) | Muy largo | La secuencia se vuelve larguísima y le cuesta mucho más a la red aprender relaciones de largo alcance |
| Por subpalabra (BPE, WordPiece) | Fragmentos de palabra, aprendidos a partir de la frecuencia con la que aparecen | Intermedio (miles a decenas de miles) | Intermedio | Resuelve el compromiso anterior a costa de que un token individual no siempre tenga significado propio |

### Tokenización por palabra

Cada palabra del vocabulario recibe un identificador. Es la opción más intuitiva y la que produce las secuencias más cortas, pero el vocabulario crece sin límite práctico: cualquier corpus grande tiene decenas de miles de palabras distintas, y siempre va a aparecer una palabra nueva que el modelo nunca vio en entrenamiento (un nombre propio, un neologismo, una palabra mal escrita). A esa palabra no reconocida se la llama **fuera de vocabulario** (*out-of-vocabulary*, OOV), y la solución tradicional es reemplazarla por un token especial `<UNK>` (*unknown*). El problema de fondo es que `<UNK>` no lleva ninguna información: todas las palabras desconocidas colapsan en el mismo símbolo, y el modelo pierde toda posibilidad de razonar sobre ellas.

### Tokenización por carácter

En el extremo opuesto, cada carácter individual es un token. El vocabulario se reduce a un puñado de símbolos (letras, números, signos de puntuación), así que el problema de OOV prácticamente desaparece: cualquier palabra, por rara que sea, se puede construir con los caracteres del alfabeto. El costo es que las secuencias se vuelven mucho más largas (una palabra de ocho letras pasa de ser 1 token a ser 8), lo que le exige mucho más a una arquitectura secuencial como una RNN, GRU o LSTM para mantener relaciones entre palabras que ahora están muy lejos entre sí en términos de pasos de tiempo.

### Tokenización por subpalabra

Es el punto intermedio, y la estrategia que usan los modelos modernos (los Transformadores incluidos). Algoritmos como **BPE** (*Byte Pair Encoding*) o **WordPiece** aprenden, a partir de las frecuencias del corpus de entrenamiento, un vocabulario de fragmentos de palabra: las palabras muy frecuentes quedan como un solo token completo, y las palabras raras o desconocidas se descomponen en fragmentos más chicos que sí están en el vocabulario. Por ejemplo, una palabra inventada como "tokenización" podría partirse en fragmentos ya conocidos como "token" + "ización". Esto casi elimina el problema de OOV sin llegar al extremo de secuencias carácter por carácter: cualquier palabra se puede representar como una combinación de subpalabras conocidas, y las palabras comunes siguen siendo un solo token.

> **Nota:** esta sección completa (tokenización por palabra, por carácter y por subpalabra, BPE/WordPiece, OOV y `<UNK>`) es ampliación: la slide original solo nombra las tres categorías en una línea, sin desarrollarlas.

## 3. De tokens a embeddings

Una vez que el texto está partido en tokens, cada token se representa como un vector numérico. La forma más simple de hacerlo es **one-hot encoding**: un vector del tamaño del vocabulario, con un 1 en la posición del token y 0 en el resto. Funciona, pero tiene dos problemas serios: el vector es enorme (tan largo como el vocabulario) y no dice nada sobre el significado del token, porque la distancia entre cualquier par de vectores one-hot es siempre la misma, sin importar si las palabras están relacionadas o no.

Los **embeddings** resuelven ambos problemas: son vectores densos (de un tamaño fijo, mucho más chico que el vocabulario, típicamente algunos cientos de dimensiones) donde cada componente no tiene un significado individual claro, pero la posición relativa de los vectores sí captura relaciones semánticas. Palabras con significados parecidos terminan con vectores cercanos en ese espacio. Estos vectores se aprenden: o bien como parte del entrenamiento de la red (una capa de embedding más), o bien se usan embeddings ya entrenados sobre corpus enormes, como **Word2Vec** o **GloVe**, que se pueden reutilizar en un modelo nuevo sin volver a entrenarlos desde cero.

Una limitación de Word2Vec y GloVe es que asignan **un único vector por palabra**, sin importar el contexto: la palabra "banco" tiene el mismo embedding tanto si aparece en "banco de plaza" como en "banco central". Los **embeddings contextuales**, que son los que producen arquitecturas como los Transformadores (tema de la próxima clase del módulo), resuelven esto generando un vector distinto para la misma palabra según las palabras que la rodean en cada oración puntual. Esa distinción — embedding fijo por palabra frente a embedding que cambia según el contexto — es una de las razones por las que los Transformadores superan a los enfoques anteriores en tareas de NLP.

> **Nota:** esta sección (one-hot, embeddings densos, Word2Vec/GloVe y embeddings contextuales) es ampliación: la slide original solo menciona la palabra "embeddings" en una línea, sin definirla ni compararla con nada.

## 4. Relación con el resto del módulo

- Esta clase es la puerta de entrada a la parte del módulo dedicada a NLP con arquitecturas secuenciales y con Transformadores. El manual del curso (`APUNTES-DATA-SCIENCE.md`, capítulo 49) documenta cómo se aplican estos mismos conceptos — vocabulario, padding y embeddings — en la práctica con Keras/TensorFlow, incluyendo un bug real del recurso de la Clase 18 donde un `Tokenizer` nuevo genera identificadores de token que no corresponden al vocabulario con el que se había entrenado el modelo; es la materialización práctica de por qué el vocabulario y la tokenización no son un detalle menor.
- Las arquitecturas que ya se vieron en el módulo — [RNN clásica](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md), [GRU](<Unidades Recurrentes con Compuertas (GRU).md>) y [LSTM](<Memoria a Largo Plazo (LSTM).md>) — son justamente los bloques que reciben esta secuencia de embeddings como entrada, un token (o su vector) por paso de tiempo.
- El siguiente tema, el [Modelo Secuencia a Secuencia](Modelo%20Secuencia%20a%20Secuencia.md), usa estos mismos tokens y embeddings como entrada de un encoder y salida de un decoder, para tareas donde la entrada y la salida tienen largos distintos (como la traducción automática).
