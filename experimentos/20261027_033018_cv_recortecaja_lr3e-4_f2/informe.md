# Informe: 20261027_033018_cv_recortecaja_lr3e-4_f2

- Validación interna: fold 2 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 40; mejor época: 24 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.581 |
| average precision (AP) | 0.363 |
| AP del azar (tasa real de pCR) | 0.268 |
| sensibilidad | 0.254 |
| especificidad | 0.845 |
| accuracy | 0.686 |
| precisión | 0.375 |
| Brier | 0.229 |
| probabilidad media predicha | 0.256 |
| tasa real de pCR | 0.268 |
| AUC solo cohorte | 0.538 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 7 | 0.333 | 0.500 | 0.429 | 0.571 | 0.524 | 0.333 | 3 | 8 | 6 | 4 |
| I-SPY2 | 158 | 45 | 0.285 | 0.603 | 0.267 | 0.867 | 0.696 | 0.444 | 12 | 98 | 15 | 33 |
| Duke | 41 | 7 | 0.171 | 0.462 | 0.000 | 0.882 | 0.732 | 0.000 | 0 | 30 | 4 | 7 |
| Todas | 220 | 59 | 0.268 | 0.581 | 0.254 | 0.845 | 0.686 | 0.375 | 15 | 136 | 25 | 44 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.229 |
| I-SPY2 | 0.331 |
| Duke | 0.220 |

AUC solo cohorte: **0.538** frente a **0.581** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
