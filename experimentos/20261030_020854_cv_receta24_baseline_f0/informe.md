# Informe: 20261030_020854_cv_receta24_baseline_f0

- Validación interna: fold 0 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 24; mejor época: 24 (criterio de selección `ultima`).
- Optimizador AdamW, weight decay 0.01; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.557 |
| average precision (AP) | 0.331 |
| AP del azar (tasa real de pCR) | 0.292 |
| sensibilidad | 0.031 |
| especificidad | 0.961 |
| accuracy | 0.689 |
| precisión | 0.250 |
| Brier | 0.210 |
| probabilidad media predicha | 0.290 |
| tasa real de pCR | 0.292 |
| AUC solo cohorte | 0.534 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 4 | 0.190 | 0.368 | 0.000 | 0.882 | 0.714 | 0.000 | 0 | 15 | 2 | 4 |
| I-SPY2 | 156 | 49 | 0.314 | 0.575 | 0.020 | 0.972 | 0.673 | 0.250 | 1 | 104 | 3 | 48 |
| Duke | 42 | 11 | 0.262 | 0.575 | 0.091 | 0.968 | 0.738 | 0.500 | 1 | 30 | 1 | 10 |
| Todas | 219 | 64 | 0.292 | 0.557 | 0.031 | 0.961 | 0.689 | 0.250 | 2 | 149 | 6 | 62 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.265 |
| I-SPY2 | 0.323 |
| Duke | 0.198 |

AUC solo cohorte: **0.534** frente a **0.557** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
