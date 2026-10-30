# Informe: 20261024_072430_cv_recortecaja_f4

- Validación interna: fold 4 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 15; mejor época: 4 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.589 |
| average precision (AP) | 0.370 |
| AP del azar (tasa real de pCR) | 0.283 |
| sensibilidad | 0.016 |
| especificidad | 0.981 |
| accuracy | 0.708 |
| precisión | 0.250 |
| Brier | 0.200 |
| probabilidad media predicha | 0.293 |
| tasa real de pCR | 0.283 |
| AUC solo cohorte | 0.554 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 20 | 5 | 0.250 | 0.547 | 0.000 | 1.000 | 0.750 | — | 0 | 15 | 0 | 5 |
| I-SPY2 | 157 | 49 | 0.312 | 0.567 | 0.000 | 0.972 | 0.669 | 0.000 | 0 | 105 | 3 | 49 |
| Duke | 42 | 8 | 0.190 | 0.629 | 0.125 | 1.000 | 0.833 | 1.000 | 1 | 34 | 0 | 7 |
| Todas | 219 | 62 | 0.283 | 0.589 | 0.016 | 0.981 | 0.708 | 0.250 | 1 | 154 | 3 | 61 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.250 |
| I-SPY2 | 0.324 |
| Duke | 0.216 |

AUC solo cohorte: **0.554** frente a **0.589** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
