# Informe: 20261022_050538_cv_baseline_f2

- Validación interna: fold 2 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 30; mejor época: 22.
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.608 |
| sensibilidad | 0.458 |
| especificidad | 0.677 |
| accuracy | 0.618 |
| precisión | 0.342 |
| Brier | 0.229 |
| probabilidad media predicha | 0.431 |
| tasa real de pCR | 0.268 |
| AUC solo cohorte | 0.538 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 7 | 0.333 | 0.592 | 0.571 | 0.786 | 0.714 | 0.571 | 4 | 11 | 3 | 3 |
| I-SPY2 | 158 | 45 | 0.285 | 0.628 | 0.444 | 0.664 | 0.601 | 0.345 | 20 | 75 | 38 | 25 |
| Duke | 41 | 7 | 0.171 | 0.508 | 0.429 | 0.676 | 0.634 | 0.214 | 3 | 23 | 11 | 4 |
| Todas | 220 | 59 | 0.268 | 0.608 | 0.458 | 0.677 | 0.618 | 0.342 | 27 | 109 | 52 | 32 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.229 |
| I-SPY2 | 0.331 |
| Duke | 0.220 |

AUC solo cohorte: **0.538** frente a **0.608** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Calibración](calibracion.png)
