# Introducción a las redes neuronales convolucionales — Parte 2

> Conversión a Markdown de las slides del curso (Clase 8 — Introducción a las CNNs, bloque 02). El PDF original está en [`material/`](material/).

## 1. Las CNN y la corteza visual humana

Las CNN se inspiran directamente en la manera en que la **corteza visual humana** procesa la información visual. Ambos sistemas procesan la información de forma **jerárquica**: arrancan por características simples y las combinan gradualmente en representaciones más complejas.

El diagrama del curso muestra el camino de la señal visual, desde el ojo hasta las áreas del cerebro:

```
Entrada (imagen) → Retina → NGL → V1 → V2 → V4 → IT
```

- **Retina**: capta la imagen (en el ejemplo, un perro).
- **NGL** (núcleo geniculado lateral): retransmite la señal desde la retina hacia la corteza.
- **V1**: primera área de la corteza visual. Sus neuronas responden a estímulos de una región acotada del campo visual, el **campo receptivo**, y detectan rasgos elementales como bordes y orientaciones.
- **V2, V4, IT**: áreas sucesivas donde las respuestas se van combinando en patrones cada vez más complejos, hasta representaciones de objetos completos en la corteza inferotemporal (IT).

Las flechas del diagrama van en ambos sentidos entre V1-V2-V4-IT: además de la vía de avance (de lo simple a lo complejo), hay conexiones de vuelta entre áreas.

Es exactamente la misma progresión que sigue una CNN: las primeras capas convolucionales detectan características básicas en las imágenes (bordes, contrastes), y las capas posteriores combinan esas características simples para formar representaciones más complejas (texturas, formas, objetos). Este enfoque es lo que permite que las CNN aprendan y reconozcan patrones visuales de manera similar al cerebro humano, y les da además la capacidad de ser **invariantes a transformaciones** (reconocer un patrón sin importar su posición exacta en la imagen).

## 2. Qué es una imagen

Una imagen se describe como una **matriz bidimensional de valores numéricos**. Cada valor recibe el nombre de **píxel** (la unidad más pequeña de una imagen digital), y representa el color o la intensidad de luz en un punto específico de la imagen.

La slide lo muestra con la imagen clásica de prueba en visión por computadora (la fotografía de Lena): al recortar una región chica de la versión en escala de grises y ampliarla, se ve la matriz de números que hay detrás, con valores entre 0 y 255 (por ejemplo 34, 29, 42, 104, 130).

> La cantidad de píxeles que componen una imagen determina su **resolución**: a más píxeles, más detalle puede representar la imagen.

## 3. Canales de color

En una imagen en escala de grises, un solo valor por píxel alcanza para representar la intensidad de luz. Las imágenes a color, en cambio, se componen de **varios canales**. Los más comunes son los canales **RGB** (rojo, verde y azul): en una imagen RGB cada píxel se define por una combinación de tres valores, uno por canal.

La slide lo ilustra separando la misma foto en sus tres canales (rojo, verde, azul) y mostrando después las tres matrices numéricas superpuestas, una por canal, con valores independientes en cada una (por ejemplo 253, 14, 165 para el mismo punto en rojo, verde y azul respectivamente).

Esto es lo que en los notebooks del curso aparece como la dimensión de **canales de entrada**: una imagen RGB de 32×32 se representa como un tensor de forma `(3, 32, 32)` — 3 canales, cada uno una matriz de 32×32 valores.

## 4. Relación con el resto del módulo

- Este bloque cierra la introducción conceptual de la Clase 8; retoma la inspiración biológica ya mencionada en [`01 - Introducción a las redes neuronales convolucionales - Parte 1.md`](01%20-%20Introducci%C3%B3n%20a%20las%20redes%20neuronales%20convolucionales%20-%20Parte%201.md) y agrega el detalle de las áreas de la corteza visual (V1, V2, V4, IT).
- La representación de la imagen como matriz numérica con canales es la base sobre la que trabaja la **capa convolucional**, desarrollada en [`Capa convolucional - P1.md`](Capa%20convolucional%20-%20P1.md) y [`Capa convolucional - P2.md`](Capa%20convolucional%20-%20P2.md): un filtro convolucional recorre exactamente esta matriz de píxeles y canales.
- En los notebooks del curso (`../notebooks/CNN_pytorch.ipynb`), el primer bloque convolucional recibe como entrada un tensor de 3 canales (RGB), coincidiendo con lo descrito acá.
