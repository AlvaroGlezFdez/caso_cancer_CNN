# Informe: 20261027_043517_cv_baseline_lr3e-4_f0

- Validación interna: fold 0 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 40; mejor época: 9 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.542 |
| average precision (AP) | 0.304 |
| AP del azar (tasa real de pCR) | 0.292 |
| sensibilidad | 0.531 |
| especificidad | 0.548 |
| accuracy | 0.543 |
| precisión | 0.327 |
| Brier | 0.259 |
| probabilidad media predicha | 0.491 |
| tasa real de pCR | 0.292 |
| AUC solo cohorte | 0.534 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 4 | 0.190 | 0.397 | 0.000 | 0.647 | 0.524 | 0.000 | 0 | 11 | 6 | 4 |
| I-SPY2 | 156 | 49 | 0.314 | 0.550 | 0.571 | 0.542 | 0.551 | 0.364 | 28 | 58 | 49 | 21 |
| Duke | 42 | 11 | 0.262 | 0.566 | 0.545 | 0.516 | 0.524 | 0.286 | 6 | 16 | 15 | 5 |
| Todas | 219 | 64 | 0.292 | 0.542 | 0.531 | 0.548 | 0.543 | 0.327 | 34 | 85 | 70 | 30 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.265 |
| I-SPY2 | 0.323 |
| Duke | 0.198 |

AUC solo cohorte: **0.534** frente a **0.542** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
