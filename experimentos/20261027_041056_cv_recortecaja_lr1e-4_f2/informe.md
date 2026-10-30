# Informe: 20261027_041056_cv_recortecaja_lr1e-4_f2

- Validación interna: fold 2 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 40; mejor época: 3 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.542 |
| average precision (AP) | 0.330 |
| AP del azar (tasa real de pCR) | 0.268 |
| sensibilidad | 0.085 |
| especificidad | 0.919 |
| accuracy | 0.695 |
| precisión | 0.278 |
| Brier | 0.205 |
| probabilidad media predicha | 0.326 |
| tasa real de pCR | 0.268 |
| AUC solo cohorte | 0.538 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 7 | 0.333 | 0.490 | 0.000 | 0.929 | 0.619 | 0.000 | 0 | 13 | 1 | 7 |
| I-SPY2 | 158 | 45 | 0.285 | 0.556 | 0.111 | 0.903 | 0.677 | 0.312 | 5 | 102 | 11 | 40 |
| Duke | 41 | 7 | 0.171 | 0.492 | 0.000 | 0.971 | 0.805 | 0.000 | 0 | 33 | 1 | 7 |
| Todas | 220 | 59 | 0.268 | 0.542 | 0.085 | 0.919 | 0.695 | 0.278 | 5 | 148 | 13 | 54 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.229 |
| I-SPY2 | 0.331 |
| Duke | 0.220 |

AUC solo cohorte: **0.538** frente a **0.542** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
