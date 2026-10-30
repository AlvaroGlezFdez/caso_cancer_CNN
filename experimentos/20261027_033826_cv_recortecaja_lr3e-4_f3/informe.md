# Informe: 20261027_033826_cv_recortecaja_lr3e-4_f3

- Validación interna: fold 3 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 40; mejor época: 8 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.592 |
| average precision (AP) | 0.447 |
| AP del azar (tasa real de pCR) | 0.341 |
| sensibilidad | 0.280 |
| especificidad | 0.814 |
| accuracy | 0.632 |
| precisión | 0.438 |
| Brier | 0.225 |
| probabilidad media predicha | 0.379 |
| tasa real de pCR | 0.341 |
| AUC solo cohorte | 0.593 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 3 | 0.143 | 0.444 | 0.333 | 0.778 | 0.714 | 0.200 | 1 | 14 | 4 | 2 |
| I-SPY2 | 157 | 63 | 0.401 | 0.585 | 0.270 | 0.819 | 0.599 | 0.500 | 17 | 77 | 17 | 46 |
| Duke | 42 | 9 | 0.214 | 0.636 | 0.333 | 0.818 | 0.714 | 0.333 | 3 | 27 | 6 | 6 |
| Todas | 220 | 75 | 0.341 | 0.592 | 0.280 | 0.814 | 0.632 | 0.438 | 21 | 118 | 27 | 54 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.277 |
| I-SPY2 | 0.301 |
| Duke | 0.210 |

AUC solo cohorte: **0.593** frente a **0.592** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
