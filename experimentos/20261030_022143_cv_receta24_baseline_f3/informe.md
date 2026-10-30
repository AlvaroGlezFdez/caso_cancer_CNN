# Informe: 20261030_022143_cv_receta24_baseline_f3

- Validación interna: fold 3 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 24; mejor época: 24 (criterio de selección `ultima`).
- Optimizador AdamW, weight decay 0.01; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.519 |
| average precision (AP) | 0.360 |
| AP del azar (tasa real de pCR) | 0.341 |
| sensibilidad | 0.040 |
| especificidad | 0.972 |
| accuracy | 0.655 |
| precisión | 0.429 |
| Brier | 0.239 |
| probabilidad media predicha | 0.273 |
| tasa real de pCR | 0.341 |
| AUC solo cohorte | 0.593 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 3 | 0.143 | 0.407 | 0.000 | 1.000 | 0.857 | — | 0 | 18 | 0 | 3 |
| I-SPY2 | 157 | 63 | 0.401 | 0.558 | 0.048 | 0.957 | 0.592 | 0.429 | 3 | 90 | 4 | 60 |
| Duke | 42 | 9 | 0.214 | 0.333 | 0.000 | 1.000 | 0.786 | — | 0 | 33 | 0 | 9 |
| Todas | 220 | 75 | 0.341 | 0.519 | 0.040 | 0.972 | 0.655 | 0.429 | 3 | 141 | 4 | 72 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.277 |
| I-SPY2 | 0.301 |
| Duke | 0.210 |

AUC solo cohorte: **0.593** frente a **0.519** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
