<!-- Convertido automáticamente desde: Dilema Sesgo - Varianza.pdf -->

Dilema Sesgo - Varianza

- Si el modelo es demasiado simple (**tiene pocos grados de libertad**),
  entonces no importa cuán grande sea la muestra: tenemos sesgo o error
  sistemático.

      E[f̂(x)] ≠ E[f(x)]

- Si el modelo es demasiado complejo (**tiene demasiados grados de libertad**),
  entonces el estimador puede ajustar regularidades espurias de la muestra,
  generando sobreajuste.
- Por lo tanto, el modelo no debe ser ni muy simple ni muy complejo.

Ejemplo (ajuste de un polinomio):

- Grado 1: underfit (mucho sesgo), MSE alto.
- Grado 4: buen ajuste, MSE mínimo.
- Grado 15: overfit (mucha varianza), MSE altísimo sobre datos nuevos.

Descomposición del error

A medida que aumenta la complejidad del modelo:

- El **sesgo (bias²)** baja.
- La **varianza (variance)** sube.
- El **error total** (sesgo² + varianza + error irreducible) tiene forma de U:
  baja y luego vuelve a subir.

El punto donde el error total es mínimo es la **complejidad óptima**: a la
izquierda hay underfit, a la derecha overfit.
