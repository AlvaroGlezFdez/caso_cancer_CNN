# Informe: 20261030_004004_cv_receta_recortecaja_f0

- Validación interna: fold 0 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 12; mejor época: 12 (criterio de selección `ultima`).
- Optimizador AdamW, weight decay 0.01; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.570 |
| average precision (AP) | 0.356 |
| AP del azar (tasa real de pCR) | 0.292 |
| sensibilidad | 0.062 |
| especificidad | 0.961 |
| accuracy | 0.699 |
| precisión | 0.400 |
| Brier | 0.208 |
| probabilidad media predicha | 0.282 |
| tasa real de pCR | 0.292 |
| AUC solo cohorte | 0.534 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 4 | 0.190 | 0.735 | 0.000 | 1.000 | 0.810 | — | 0 | 17 | 0 | 4 |
| I-SPY2 | 156 | 49 | 0.314 | 0.533 | 0.061 | 0.944 | 0.667 | 0.333 | 3 | 101 | 6 | 46 |
| Duke | 42 | 11 | 0.262 | 0.613 | 0.091 | 1.000 | 0.762 | 1.000 | 1 | 31 | 0 | 10 |
| Todas | 219 | 64 | 0.292 | 0.570 | 0.062 | 0.961 | 0.699 | 0.400 | 4 | 149 | 6 | 60 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.265 |
| I-SPY2 | 0.323 |
| Duke | 0.198 |

AUC solo cohorte: **0.534** frente a **0.570** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
