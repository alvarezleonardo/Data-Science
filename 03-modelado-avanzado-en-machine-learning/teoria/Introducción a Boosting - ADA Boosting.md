<!-- Convertido automáticamente desde: Introducción a Boosting - ADA Boosting.pdf -->

ADA Boost

- Creadores: Yoav Freund y Robert Schapire.
- Año: 1996.
- Importancia: marcó un avance significativo en los métodos de boosting.

Funcionamiento de AdaBoost:
- Inicialización: asignar pesos iguales a todas las muestras de entrenamiento.
- Iteración: entrenar un clasificador débil y evaluar su error.
- Actualizar los pesos: aumentar los pesos de las muestras mal clasificadas y
  disminuir los de las correctamente clasificadas.
- Combinación: los clasificadores se combinan ponderando su importancia según su
  precisión.

Cada predictor se entrena de forma secuencial; el peso de su voto es `α = η·α`.

Ventajas:
- Precisión: puede mejorar significativamente la precisión de los modelos base.
- Simplicidad: fácil de implementar con clasificadores básicos.
- Flexibilidad: puede usarse con una variedad de clasificadores débiles.

Desventajas:
- Sensibilidad a ruido: los datos ruidosos pueden afectar negativamente el
  rendimiento.
- Sobreajuste: posible si se usan demasiados clasificadores débiles.

Aplicaciones de AdaBoost:
- Reconocimiento de caras: identificación y verificación de rostros en imágenes.
- Detección de objetos: localización y reconocimiento de objetos en imágenes.
- Clasificación de texto: clasificación de documentos y correos electrónicos.
