# Mecanismos de Atención

> Conversión a Markdown de las slides del curso (Clase 23 — Mecanismos de Atención). El PDF original está al lado.

## 1. La idea central

El mecanismo de atención permite que, en lugar de usar solo un vector de contexto, el decoder **preste atención** a diferentes partes de la secuencia de entrada. El decoder puede acceder a **todos los estados ocultos del encoder** y asignarles diferentes pesos según la relevancia de cada estado en un momento dado.

Esta idea ataca directamente el cuello de botella señalado al cierre de la clase anterior: en el [Modelo Secuencia a Secuencia](Modelo%20Secuencia%20a%20Secuencia.md) básico, toda la secuencia de entrada se comprime en un único vector de contexto de tamaño fijo, y ese vector se degrada con secuencias largas. La atención elimina ese cuello de botella: en vez de comprimir todo en un solo vector antes de que el decoder empiece a generar, el decoder mira, en cada paso de generación, el conjunto completo de estados ocultos que produjo el encoder, y decide — mediante pesos aprendidos — cuáles de esos estados le importan más para generar la palabra que le toca en ese momento.

## 2. El diagrama del material

El material ilustra el mecanismo con un ejemplo de traducción: la entrada "Good Food" (inglés) se traduce a "Buena Comida" (español). El diagrama muestra dos bloques:

- **Encoder** (recuadro naranja): dos celdas **GRU Cell**, en secuencia. La primera recibe la palabra "Good", la segunda recibe "Food". Cada celda GRU produce un estado oculto que se pasa a la celda siguiente (flecha naranja horizontal) — es la misma recurrencia de una GRU vista en la clase correspondiente del módulo. La flecha naranja que sale de la última celda del encoder continúa hacia la primera celda del decoder: es el vector de contexto tradicional, que en este esquema sigue existiendo, pero deja de ser la única fuente de información del decoder.
- **Decoder** (recuadro verde): tres celdas GRU Cell en secuencia. La primera recibe el token especial `<SOS>` (*start of sequence*, inicio de secuencia) y produce "Buena"; la segunda produce "Comida"; la tercera produce el token especial `<EOS>` (*end of sequence*, fin de secuencia), que le indica al modelo que debe dejar de generar.

El elemento clave del diagrama son las **flechas curvas verdes** que conectan la salida de cada celda del decoder con la entrada de la celda siguiente, formando arcos que van "hacia atrás y hacia adelante" entre los pasos del decoder. Esas curvas representan el mecanismo de atención en acción: antes de producir su salida, cada celda del decoder no depende solamente de su propio estado anterior en línea recta, sino que su cómputo incorpora información combinada de los estados relevantes disponibles, con pesos que cambian según qué se está generando en cada paso — a diferencia del esquema seq2seq simple, donde la única conexión entre encoder y decoder era el vector de contexto único señalado con flechas punteadas hacia todos los pasos por igual.

> **Nota:** el material no incluye las ecuaciones formales de la atención (el cálculo de los *scores* de relevancia, la normalización con softmax para obtener los pesos, ni la suma ponderada de los estados del encoder). El diagrama transmite la idea mediante las celdas GRU y las flechas curvas, pero no desarrolla la fórmula matemática del mecanismo. Lo que sigue en la sección 3 es una ampliación para cubrir ese mecanismo con el nivel de detalle que el diagrama sugiere pero no explicita.

## 3. Cómo funciona el mecanismo, en términos generales

> **Nota:** toda esta sección es ampliación; no proviene de la slide, que se limita al párrafo y el diagrama descriptos arriba.

La atención, en su formulación general, agrega tres piezas al esquema encoder-decoder:

1. **Puntajes de relevancia (*scores*)**: en cada paso de generación del decoder, se calcula un puntaje entre el estado actual del decoder y **cada uno** de los estados ocultos del encoder (no solo el último). Ese puntaje mide qué tan relevante es cada palabra de la entrada para la palabra que se está por generar.
2. **Pesos de atención**: esos puntajes se normalizan con una función softmax, de modo que el resultado es una **distribución de probabilidad que suma 1** entre todos los estados del encoder. Un peso cercano a 1 en un estado particular significa que el decoder está prestando casi toda su atención a esa palabra de la entrada en ese paso puntual; los pesos se reparten entre varios estados cuando la relevancia está más distribuida.
3. **Vector de contexto dinámico**: en lugar de un único vector de contexto fijo para toda la generación (como en el seq2seq simple), se calcula un vector de contexto distinto **en cada paso del decoder**, como la suma ponderada de todos los estados del encoder, usando los pesos de atención de ese paso. Ese vector de contexto dinámico, junto con el estado anterior del decoder, es lo que alimenta la generación de la palabra siguiente.

Con este mecanismo, si la entrada es larga, el decoder no depende de que todo ese contenido haya sobrevivido comprimido en un solo vector: en cada paso puede volver a "consultar" directamente los estados del encoder y quedarse con los que importan para esa palabra puntual.

## 4. Por qué importa: rendimiento e interpretabilidad

La atención aporta dos beneficios distintos, y el material solo señala el primero:

- **Resuelve el cuello de botella del vector de contexto único**: al dar acceso directo a todos los estados del encoder en cada paso, ya no hace falta condensar la oración entera en un vector de tamaño fijo. Esto mejora especialmente el desempeño en secuencias largas, que era justo el punto débil señalado al cierre de la clase de [Modelo Secuencia a Secuencia](Modelo%20Secuencia%20a%20Secuencia.md).
- **Da interpretabilidad**: como los pesos de atención son una distribución explícita sobre las palabras de entrada, se puede visualizar, para cada palabra generada por el decoder, a qué palabra o palabras de la entrada le prestó más atención el modelo. En el ejemplo del diagrama, sería observable que al generar "Buena" el modelo pesa fuerte sobre "Good", y al generar "Comida" pesa fuerte sobre "Food" — algo que en el seq2seq simple es imposible de observar, porque toda la entrada pasó por el mismo vector de contexto indiferenciado.

> **Nota:** esta sección es ampliación; el material no desarrolla la interpretabilidad ni la relación explícita entre pesos de atención y palabras concretas del ejemplo, aunque el propio ejemplo del diagrama (Good Food → Buena Comida) se presta directamente a esa lectura.

## 5. Relación con el resto del módulo

- Esta clase resuelve el problema planteado al cierre de [Modelo Secuencia a Secuencia](Modelo%20Secuencia%20a%20Secuencia.md): el vector de contexto único como cuello de botella de la información. La atención reemplaza esa compresión total por acceso directo y ponderado a todos los estados del encoder.
- El diagrama usa celdas GRU explícitamente para el encoder y el decoder, la arquitectura recurrente vista en [Unidades Recurrentes con Compuertas (GRU)](<Unidades Recurrentes con Compuertas (GRU).md>), reforzando que el mecanismo de atención se agrega **sobre** una arquitectura recurrente ya existente (no la reemplaza).
- El mecanismo de atención es, históricamente y dentro de este módulo, el puente hacia el **Transformador**: la arquitectura que generaliza la atención a un mecanismo de "autoatención" (*self-attention*) sin necesidad de celdas recurrentes en absoluto, y que es el tema que continúa el bloque de "Transformadores y Procesamiento de Lenguaje Natural" al que pertenecen estas tres clases (según indica el título compartido de las tres slides: "Transformadores y Procesamiento de Lenguaje Natural").
- Junto con [Procesamiento de Lenguaje Natural](Procesamiento%20de%20Lenguaje%20Natural.md) (tokenización y embeddings) y [Modelo Secuencia a Secuencia](Modelo%20Secuencia%20a%20Secuencia.md) (encoder-decoder), esta clase completa el bloque conceptual que antecede a los Transformadores: cómo representar texto, cómo mapear secuencias de largo distinto, y cómo dejar de depender de un único vector de contexto.
