# Informe: 20261027_051343_cv_baseline_lr3e-4_f2

- Validación interna: fold 2 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 40; mejor época: 18 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.627 |
| average precision (AP) | 0.405 |
| AP del azar (tasa real de pCR) | 0.268 |
| sensibilidad | 0.763 |
| especificidad | 0.410 |
| accuracy | 0.505 |
| precisión | 0.321 |
| Brier | 0.334 |
| probabilidad media predicha | 0.611 |
| tasa real de pCR | 0.268 |
| AUC solo cohorte | 0.538 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 7 | 0.333 | 0.541 | 0.286 | 0.643 | 0.524 | 0.286 | 2 | 9 | 5 | 5 |
| I-SPY2 | 158 | 45 | 0.285 | 0.658 | 0.867 | 0.381 | 0.519 | 0.358 | 39 | 43 | 70 | 6 |
| Duke | 41 | 7 | 0.171 | 0.542 | 0.571 | 0.412 | 0.439 | 0.167 | 4 | 14 | 20 | 3 |
| Todas | 220 | 59 | 0.268 | 0.627 | 0.763 | 0.410 | 0.505 | 0.321 | 45 | 66 | 95 | 14 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.229 |
| I-SPY2 | 0.331 |
| Duke | 0.220 |

AUC solo cohorte: **0.538** frente a **0.627** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
