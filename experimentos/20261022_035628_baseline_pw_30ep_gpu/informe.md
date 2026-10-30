# Informe: 20261022_035628_baseline_pw_30ep_gpu

- Validación interna: fold 0 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: 2.4002; épocas ejecutadas: 17; mejor época: 10 (criterio de selección `auc`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.579 |
| average precision (AP) | 0.333 |
| AP del azar (tasa real de pCR) | 0.292 |
| sensibilidad | 0.844 |
| especificidad | 0.258 |
| accuracy | 0.429 |
| precisión | 0.320 |
| Brier | 0.325 |
| probabilidad media predicha | 0.620 |
| tasa real de pCR | 0.292 |
| AUC solo cohorte | 0.534 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 4 | 0.190 | 0.750 | 1.000 | 0.588 | 0.667 | 0.364 | 4 | 10 | 7 | 0 |
| I-SPY2 | 156 | 49 | 0.314 | 0.563 | 0.837 | 0.168 | 0.378 | 0.315 | 41 | 18 | 89 | 8 |
| Duke | 42 | 11 | 0.262 | 0.598 | 0.818 | 0.387 | 0.500 | 0.321 | 9 | 12 | 19 | 2 |
| Todas | 219 | 64 | 0.292 | 0.579 | 0.844 | 0.258 | 0.429 | 0.320 | 54 | 40 | 115 | 10 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.265 |
| I-SPY2 | 0.323 |
| Duke | 0.198 |

AUC solo cohorte: **0.534** frente a **0.579** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
