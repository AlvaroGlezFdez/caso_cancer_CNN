# Informe: 20261030_004227_cv_receta_recortecaja_f2

- Validación interna: fold 2 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 12; mejor época: 12 (criterio de selección `ultima`).
- Optimizador AdamW, weight decay 0.01; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.515 |
| average precision (AP) | 0.327 |
| AP del azar (tasa real de pCR) | 0.268 |
| sensibilidad | 0.102 |
| especificidad | 0.944 |
| accuracy | 0.718 |
| precisión | 0.400 |
| Brier | 0.208 |
| probabilidad media predicha | 0.324 |
| tasa real de pCR | 0.268 |
| AUC solo cohorte | 0.538 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 7 | 0.333 | 0.347 | 0.000 | 0.857 | 0.571 | 0.000 | 0 | 12 | 2 | 7 |
| I-SPY2 | 158 | 45 | 0.285 | 0.535 | 0.133 | 0.938 | 0.709 | 0.462 | 6 | 106 | 7 | 39 |
| Duke | 41 | 7 | 0.171 | 0.483 | 0.000 | 1.000 | 0.829 | — | 0 | 34 | 0 | 7 |
| Todas | 220 | 59 | 0.268 | 0.515 | 0.102 | 0.944 | 0.718 | 0.400 | 6 | 152 | 9 | 53 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.229 |
| I-SPY2 | 0.331 |
| Duke | 0.220 |

AUC solo cohorte: **0.538** frente a **0.515** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
