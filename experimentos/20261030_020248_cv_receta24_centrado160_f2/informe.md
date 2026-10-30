# Informe: 20261030_020248_cv_receta24_centrado160_f2

- Validación interna: fold 2 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 24; mejor época: 24 (criterio de selección `ultima`).
- Optimizador AdamW, weight decay 0.01; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.528 |
| average precision (AP) | 0.307 |
| AP del azar (tasa real de pCR) | 0.268 |
| sensibilidad | 0.136 |
| especificidad | 0.882 |
| accuracy | 0.682 |
| precisión | 0.296 |
| Brier | 0.212 |
| probabilidad media predicha | 0.318 |
| tasa real de pCR | 0.268 |
| AUC solo cohorte | 0.538 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 7 | 0.333 | 0.531 | 0.143 | 0.857 | 0.619 | 0.333 | 1 | 12 | 2 | 6 |
| I-SPY2 | 158 | 45 | 0.285 | 0.534 | 0.133 | 0.858 | 0.652 | 0.273 | 6 | 97 | 16 | 39 |
| Duke | 41 | 7 | 0.171 | 0.458 | 0.143 | 0.971 | 0.829 | 0.500 | 1 | 33 | 1 | 6 |
| Todas | 220 | 59 | 0.268 | 0.528 | 0.136 | 0.882 | 0.682 | 0.296 | 8 | 142 | 19 | 51 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.229 |
| I-SPY2 | 0.331 |
| Duke | 0.220 |

AUC solo cohorte: **0.538** frente a **0.528** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
