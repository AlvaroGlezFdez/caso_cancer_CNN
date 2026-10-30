# Informe: 20261022_054946_cv_aumento_f1

- Validación interna: fold 1 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 23; mejor época: 13 (criterio de selección `auc`).
- Optimizador AdamW, weight decay 0; aumento de datos: volteo_h0.5_rot10.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.569 |
| average precision (AP) | 0.345 |
| AP del azar (tasa real de pCR) | 0.283 |
| sensibilidad | 0.306 |
| especificidad | 0.771 |
| accuracy | 0.639 |
| precisión | 0.345 |
| Brier | 0.231 |
| probabilidad media predicha | 0.413 |
| tasa real de pCR | 0.283 |
| AUC solo cohorte | 0.526 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 7 | 0.333 | 0.551 | 0.143 | 0.929 | 0.667 | 0.500 | 1 | 13 | 1 | 6 |
| I-SPY2 | 156 | 46 | 0.295 | 0.565 | 0.326 | 0.745 | 0.622 | 0.349 | 15 | 82 | 28 | 31 |
| Duke | 42 | 9 | 0.214 | 0.603 | 0.333 | 0.788 | 0.690 | 0.300 | 3 | 26 | 7 | 6 |
| Todas | 219 | 62 | 0.283 | 0.569 | 0.306 | 0.771 | 0.639 | 0.345 | 19 | 121 | 36 | 43 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.229 |
| I-SPY2 | 0.328 |
| Duke | 0.210 |

AUC solo cohorte: **0.526** frente a **0.569** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
