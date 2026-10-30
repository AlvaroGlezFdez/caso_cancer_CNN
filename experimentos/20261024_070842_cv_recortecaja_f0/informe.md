# Informe: 20261024_070842_cv_recortecaja_f0

- Validación interna: fold 0 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce no), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 13; mejor época: 13 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.563 |
| average precision (AP) | 0.335 |
| AP del azar (tasa real de pCR) | 0.292 |
| sensibilidad | 0.047 |
| especificidad | 0.935 |
| accuracy | 0.676 |
| precisión | 0.231 |
| Brier | 0.226 |
| probabilidad media predicha | 0.215 |
| tasa real de pCR | 0.292 |
| AUC solo cohorte | 0.534 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 4 | 0.190 | 0.500 | 0.250 | 1.000 | 0.857 | 1.000 | 1 | 17 | 0 | 3 |
| I-SPY2 | 156 | 49 | 0.314 | 0.530 | 0.041 | 0.916 | 0.641 | 0.182 | 2 | 98 | 9 | 47 |
| Duke | 42 | 11 | 0.262 | 0.648 | 0.000 | 0.968 | 0.714 | 0.000 | 0 | 30 | 1 | 11 |
| Todas | 219 | 64 | 0.292 | 0.563 | 0.047 | 0.935 | 0.676 | 0.231 | 3 | 145 | 10 | 61 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.265 |
| I-SPY2 | 0.323 |
| Duke | 0.198 |

AUC solo cohorte: **0.534** frente a **0.563** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
