# Informe: 20261022_052003_cv_baseline_f3

- Validación interna: fold 3 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 26; mejor época: 16 (criterio de selección `auc`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.540 |
| average precision (AP) | 0.363 |
| AP del azar (tasa real de pCR) | 0.341 |
| sensibilidad | 0.693 |
| especificidad | 0.407 |
| accuracy | 0.505 |
| precisión | 0.377 |
| Brier | 0.295 |
| probabilidad media predicha | 0.548 |
| tasa real de pCR | 0.341 |
| AUC solo cohorte | 0.593 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 3 | 0.143 | 0.444 | 0.333 | 0.500 | 0.476 | 0.100 | 1 | 9 | 9 | 2 |
| I-SPY2 | 157 | 63 | 0.401 | 0.537 | 0.730 | 0.383 | 0.522 | 0.442 | 46 | 36 | 58 | 17 |
| Duke | 42 | 9 | 0.214 | 0.455 | 0.556 | 0.424 | 0.452 | 0.208 | 5 | 14 | 19 | 4 |
| Todas | 220 | 75 | 0.341 | 0.540 | 0.693 | 0.407 | 0.505 | 0.377 | 52 | 59 | 86 | 23 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.277 |
| I-SPY2 | 0.301 |
| Duke | 0.210 |

AUC solo cohorte: **0.593** frente a **0.540** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
