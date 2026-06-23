<!-- Convertido automáticamente desde: Regresión Lineal Múltiple.pdf -->

Regresión Lineal Múltiple

01 Regresión lineal múltiple

Supongamos que tenemos `p` predictores distintos, entonces el modelo de
regresión lineal múltiple toma la siguiente forma:

    Y = β0 + β1·X1 + β2·X2 + ... + βp·Xp + ε

La "línea" de regresión mínimo cuadrática se vuelve un plano.
El plano buscado minimiza la suma de los cuadrados de las distancias entre
cada observación y el plano.

02 Estimadores y Predicción para regresión lineal múltiple

Dados estimadores de los coeficientes de pendiente podemos pronosticar la
variable de respuesta para una observación con valores dados de los
predictores como:

    ŷ = β̂0 + β̂1·x1 + β̂2·x2 + ... + β̂p·xp

Elegimos los valores para los estimadores de los coeficientes que minimizan
la suma de residuos al cuadrado:

    RSS = Σ (yi − ŷi)²
        = Σ (yi − β̂0 − β̂1·xi1 − β̂2·xi2 − ... − β̂p·xip)²

Test de hipótesis sobre los coeficientes estimados

¿Hay alguna relación entre la variable objetivo y todas las variables
explicativas?

- H0: β1 = β2 = ... = βp = 0
- H1: al menos un βj ≠ 0

Usamos el estadístico F:

    F = ((TSS − RSS) / p) / (RSS / (n − p − 1))

Donde:

    TSS = Σ (yi − ȳ)²
    RSS = Σ (yi − ŷi)²

¡Muchas gracias!
