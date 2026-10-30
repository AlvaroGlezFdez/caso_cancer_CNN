# Informe: 20261024_075132_cv_realce_f3

- Validación interna: fold 3 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce sí), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 27; mejor época: 16 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.544 |
| average precision (AP) | 0.374 |
| AP del azar (tasa real de pCR) | 0.341 |
| sensibilidad | 0.547 |
| especificidad | 0.538 |
| accuracy | 0.541 |
| precisión | 0.380 |
| Brier | 0.296 |
| probabilidad media predicha | 0.498 |
| tasa real de pCR | 0.341 |
| AUC solo cohorte | 0.593 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 3 | 0.143 | 0.389 | 0.000 | 0.556 | 0.476 | 0.000 | 0 | 10 | 8 | 3 |
| I-SPY2 | 157 | 63 | 0.401 | 0.512 | 0.571 | 0.489 | 0.522 | 0.429 | 36 | 46 | 48 | 27 |
| Duke | 42 | 9 | 0.214 | 0.562 | 0.556 | 0.667 | 0.643 | 0.312 | 5 | 22 | 11 | 4 |
| Todas | 220 | 75 | 0.341 | 0.544 | 0.547 | 0.538 | 0.541 | 0.380 | 41 | 78 | 67 | 34 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.277 |
| I-SPY2 | 0.301 |
| Duke | 0.210 |

AUC solo cohorte: **0.593** frente a **0.544** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
