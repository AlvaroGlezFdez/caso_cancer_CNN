# Informe: 20261024_071822_cv_recortecaja_f3

- Validación interna: fold 3 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 30; mejor época: 24 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.600 |
| average precision (AP) | 0.476 |
| AP del azar (tasa real de pCR) | 0.341 |
| sensibilidad | 0.213 |
| especificidad | 0.938 |
| accuracy | 0.691 |
| precisión | 0.640 |
| Brier | 0.251 |
| probabilidad media predicha | 0.198 |
| tasa real de pCR | 0.341 |
| AUC solo cohorte | 0.593 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 3 | 0.143 | 0.648 | 0.333 | 0.889 | 0.810 | 0.333 | 1 | 16 | 2 | 2 |
| I-SPY2 | 157 | 63 | 0.401 | 0.606 | 0.222 | 0.936 | 0.650 | 0.700 | 14 | 88 | 6 | 49 |
| Duke | 42 | 9 | 0.214 | 0.566 | 0.111 | 0.970 | 0.786 | 0.500 | 1 | 32 | 1 | 8 |
| Todas | 220 | 75 | 0.341 | 0.600 | 0.213 | 0.938 | 0.691 | 0.640 | 16 | 136 | 9 | 59 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.277 |
| I-SPY2 | 0.301 |
| Duke | 0.210 |

AUC solo cohorte: **0.593** frente a **0.600** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
