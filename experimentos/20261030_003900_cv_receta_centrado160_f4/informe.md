# Informe: 20261030_003900_cv_receta_centrado160_f4

- Validación interna: fold 4 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 12; mejor época: 12 (criterio de selección `ultima`).
- Optimizador AdamW, weight decay 0.01; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.596 |
| average precision (AP) | 0.388 |
| AP del azar (tasa real de pCR) | 0.283 |
| sensibilidad | 0.065 |
| especificidad | 0.968 |
| accuracy | 0.712 |
| precisión | 0.444 |
| Brier | 0.199 |
| probabilidad media predicha | 0.304 |
| tasa real de pCR | 0.283 |
| AUC solo cohorte | 0.554 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 20 | 5 | 0.250 | 0.560 | 0.200 | 1.000 | 0.800 | 1.000 | 1 | 15 | 0 | 4 |
| I-SPY2 | 157 | 49 | 0.312 | 0.553 | 0.041 | 0.954 | 0.669 | 0.286 | 2 | 103 | 5 | 47 |
| Duke | 42 | 8 | 0.190 | 0.699 | 0.125 | 1.000 | 0.833 | 1.000 | 1 | 34 | 0 | 7 |
| Todas | 219 | 62 | 0.283 | 0.596 | 0.065 | 0.968 | 0.712 | 0.444 | 4 | 152 | 5 | 58 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.250 |
| I-SPY2 | 0.324 |
| Duke | 0.216 |

AUC solo cohorte: **0.554** frente a **0.596** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
