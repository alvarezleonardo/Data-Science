# Limitaciones del Perceptrón

> Conversión a Markdown de la slide del curso. El PDF original está en [`material/`](material/).

## 1. Separabilidad lineal y el problema del XOR

El perceptrón resuelve problemas de clasificación **linealmente separables**. Su limitación clásica es el **problema del XOR**: un conjunto de datos cuyas clases no pueden separarse mediante una única línea (o hiperplano), y que por lo tanto el perceptrón simple no puede resolver.

> **Nota:** la slide original enuncia el punto como "Resuelve problemas de clasificación linealmente separables. Problema del XOR", una redacción ambigua tal como aparece en el material. Se aclara aquí el sentido: el perceptrón **solo** resuelve casos linealmente separables, y el XOR es el ejemplo típico de problema que **no** lo es, quedando fuera de su alcance.

## 2. Convergencia no garantizada

No siempre garantiza la **convergencia** en un número finito de pasos, en especial si los datos no son linealmente separables.

## 3. Capacidad limitada para relaciones complejas

Tiene **capacidad restringida** para capturar relaciones complejas y patrones en los datos.

## 4. Ajuste de hiperparámetros

Aunque el perceptrón tiene menos hiperparámetros que otros modelos más complejos, ajustar estos hiperparámetros (como la **tasa de aprendizaje**) sigue siendo un desafío.
