# Informe: 20261022_060652_cv_aumento_f3

- Validación interna: fold 3 (220 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 22; mejor época: 12.
- Optimizador AdamW, weight decay 0; aumento de datos: volteo_h0.5_rot10.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.549 |
| sensibilidad | 0.053 |
| especificidad | 0.945 |
| accuracy | 0.641 |
| precisión | 0.333 |
| Brier | 0.233 |
| probabilidad media predicha | 0.313 |
| tasa real de pCR | 0.341 |
| AUC solo cohorte | 0.593 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 3 | 0.143 | 0.667 | 0.333 | 0.944 | 0.857 | 0.500 | 1 | 17 | 1 | 2 |
| I-SPY2 | 157 | 63 | 0.401 | 0.559 | 0.048 | 0.947 | 0.586 | 0.375 | 3 | 89 | 5 | 60 |
| Duke | 42 | 9 | 0.214 | 0.418 | 0.000 | 0.939 | 0.738 | 0.000 | 0 | 31 | 2 | 9 |
| Todas | 220 | 75 | 0.341 | 0.549 | 0.053 | 0.945 | 0.641 | 0.333 | 4 | 137 | 8 | 71 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.277 |
| I-SPY2 | 0.301 |
| Duke | 0.210 |

AUC solo cohorte: **0.593** frente a **0.549** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Calibración](calibracion.png)
