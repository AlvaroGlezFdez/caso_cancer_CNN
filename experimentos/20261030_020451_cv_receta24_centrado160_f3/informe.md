# Informe: 20261030_020451_cv_receta24_centrado160_f3

- Validación interna: fold 3 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 24; mejor época: 24 (criterio de selección `ultima`).
- Optimizador AdamW, weight decay 0.01; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.572 |
| average precision (AP) | 0.390 |
| AP del azar (tasa real de pCR) | 0.341 |
| sensibilidad | 0.093 |
| especificidad | 0.952 |
| accuracy | 0.659 |
| precisión | 0.500 |
| Brier | 0.236 |
| probabilidad media predicha | 0.282 |
| tasa real de pCR | 0.341 |
| AUC solo cohorte | 0.593 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 3 | 0.143 | 0.500 | 0.000 | 0.944 | 0.810 | 0.000 | 0 | 17 | 1 | 3 |
| I-SPY2 | 157 | 63 | 0.401 | 0.616 | 0.111 | 0.957 | 0.618 | 0.636 | 7 | 90 | 4 | 56 |
| Duke | 42 | 9 | 0.214 | 0.370 | 0.000 | 0.939 | 0.738 | 0.000 | 0 | 31 | 2 | 9 |
| Todas | 220 | 75 | 0.341 | 0.572 | 0.093 | 0.952 | 0.659 | 0.500 | 7 | 138 | 7 | 68 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.277 |
| I-SPY2 | 0.301 |
| Duke | 0.210 |

AUC solo cohorte: **0.593** frente a **0.572** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
