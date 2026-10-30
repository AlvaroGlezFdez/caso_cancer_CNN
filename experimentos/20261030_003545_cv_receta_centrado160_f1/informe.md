# Informe: 20261030_003545_cv_receta_centrado160_f1

- Validación interna: fold 1 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 12; mejor época: 12 (criterio de selección `ultima`).
- Optimizador AdamW, weight decay 0.01; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.540 |
| average precision (AP) | 0.318 |
| AP del azar (tasa real de pCR) | 0.283 |
| sensibilidad | 0.048 |
| especificidad | 0.962 |
| accuracy | 0.703 |
| precisión | 0.333 |
| Brier | 0.213 |
| probabilidad media predicha | 0.296 |
| tasa real de pCR | 0.283 |
| AUC solo cohorte | 0.526 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 7 | 0.333 | 0.418 | 0.000 | 0.929 | 0.619 | 0.000 | 0 | 13 | 1 | 7 |
| I-SPY2 | 156 | 46 | 0.295 | 0.553 | 0.043 | 0.964 | 0.692 | 0.333 | 2 | 106 | 4 | 44 |
| Duke | 42 | 9 | 0.214 | 0.471 | 0.111 | 0.970 | 0.786 | 0.500 | 1 | 32 | 1 | 8 |
| Todas | 219 | 62 | 0.283 | 0.540 | 0.048 | 0.962 | 0.703 | 0.333 | 3 | 151 | 6 | 59 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.229 |
| I-SPY2 | 0.328 |
| Duke | 0.210 |

AUC solo cohorte: **0.526** frente a **0.540** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
