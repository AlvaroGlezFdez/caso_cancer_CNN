# Informe: 20261030_003010_cv_receta_baseline_f3

- Validación interna: fold 3 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 12; mejor época: 12 (criterio de selección `ultima`).
- Optimizador AdamW, weight decay 0.01; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.506 |
| average precision (AP) | 0.380 |
| AP del azar (tasa real de pCR) | 0.341 |
| sensibilidad | 0.013 |
| especificidad | 0.986 |
| accuracy | 0.655 |
| precisión | 0.333 |
| Brier | 0.236 |
| probabilidad media predicha | 0.269 |
| tasa real de pCR | 0.341 |
| AUC solo cohorte | 0.593 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 3 | 0.143 | 0.574 | 0.000 | 1.000 | 0.857 | — | 0 | 18 | 0 | 3 |
| I-SPY2 | 157 | 63 | 0.401 | 0.516 | 0.016 | 0.989 | 0.599 | 0.500 | 1 | 93 | 1 | 62 |
| Duke | 42 | 9 | 0.214 | 0.380 | 0.000 | 0.970 | 0.762 | 0.000 | 0 | 32 | 1 | 9 |
| Todas | 220 | 75 | 0.341 | 0.506 | 0.013 | 0.986 | 0.655 | 0.333 | 1 | 143 | 2 | 74 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.277 |
| I-SPY2 | 0.301 |
| Duke | 0.210 |

AUC solo cohorte: **0.593** frente a **0.506** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
