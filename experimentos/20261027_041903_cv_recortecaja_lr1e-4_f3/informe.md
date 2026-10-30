# Informe: 20261027_041903_cv_recortecaja_lr1e-4_f3

- Validación interna: fold 3 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 40; mejor época: 17 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.582 |
| average precision (AP) | 0.429 |
| AP del azar (tasa real de pCR) | 0.341 |
| sensibilidad | 0.080 |
| especificidad | 0.952 |
| accuracy | 0.655 |
| precisión | 0.462 |
| Brier | 0.249 |
| probabilidad media predicha | 0.202 |
| tasa real de pCR | 0.341 |
| AUC solo cohorte | 0.593 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 3 | 0.143 | 0.741 | 0.333 | 0.944 | 0.857 | 0.500 | 1 | 17 | 1 | 2 |
| I-SPY2 | 157 | 63 | 0.401 | 0.581 | 0.063 | 0.947 | 0.592 | 0.444 | 4 | 89 | 5 | 59 |
| Duke | 42 | 9 | 0.214 | 0.498 | 0.111 | 0.970 | 0.786 | 0.500 | 1 | 32 | 1 | 8 |
| Todas | 220 | 75 | 0.341 | 0.582 | 0.080 | 0.952 | 0.655 | 0.462 | 6 | 138 | 7 | 69 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.277 |
| I-SPY2 | 0.301 |
| Duke | 0.210 |

AUC solo cohorte: **0.593** frente a **0.582** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
