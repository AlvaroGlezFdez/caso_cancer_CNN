# Informe: 20261027_040247_cv_recortecaja_lr1e-4_f1

- Validación interna: fold 1 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 40; mejor época: 6 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.561 |
| average precision (AP) | 0.329 |
| AP del azar (tasa real de pCR) | 0.283 |
| sensibilidad | 0.145 |
| especificidad | 0.904 |
| accuracy | 0.689 |
| precisión | 0.375 |
| Brier | 0.213 |
| probabilidad media predicha | 0.304 |
| tasa real de pCR | 0.283 |
| AUC solo cohorte | 0.526 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 7 | 0.333 | 0.520 | 0.143 | 0.929 | 0.667 | 0.500 | 1 | 13 | 1 | 6 |
| I-SPY2 | 156 | 46 | 0.295 | 0.566 | 0.152 | 0.873 | 0.660 | 0.333 | 7 | 96 | 14 | 39 |
| Duke | 42 | 9 | 0.214 | 0.529 | 0.111 | 1.000 | 0.810 | 1.000 | 1 | 33 | 0 | 8 |
| Todas | 219 | 62 | 0.283 | 0.561 | 0.145 | 0.904 | 0.689 | 0.375 | 9 | 142 | 15 | 53 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.229 |
| I-SPY2 | 0.328 |
| Duke | 0.210 |

AUC solo cohorte: **0.526** frente a **0.561** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
