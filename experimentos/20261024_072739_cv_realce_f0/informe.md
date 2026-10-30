# Informe: 20261024_072739_cv_realce_f0

- Validación interna: fold 0 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros, 4 bloques, realce sí), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 21; mejor época: 10 (criterio de selección `auc_suavizado`).
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.544 |
| average precision (AP) | 0.330 |
| AP del azar (tasa real de pCR) | 0.292 |
| sensibilidad | 0.141 |
| especificidad | 0.916 |
| accuracy | 0.689 |
| precisión | 0.409 |
| Brier | 0.218 |
| probabilidad media predicha | 0.336 |
| tasa real de pCR | 0.292 |
| AUC solo cohorte | 0.534 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 4 | 0.190 | 0.500 | 0.000 | 0.941 | 0.762 | 0.000 | 0 | 16 | 1 | 4 |
| I-SPY2 | 156 | 49 | 0.314 | 0.555 | 0.143 | 0.916 | 0.673 | 0.438 | 7 | 98 | 9 | 42 |
| Duke | 42 | 11 | 0.262 | 0.499 | 0.182 | 0.903 | 0.714 | 0.400 | 2 | 28 | 3 | 9 |
| Todas | 219 | 64 | 0.292 | 0.544 | 0.141 | 0.916 | 0.689 | 0.409 | 9 | 142 | 13 | 55 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.265 |
| I-SPY2 | 0.323 |
| Duke | 0.198 |

AUC solo cohorte: **0.534** frente a **0.544** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Curva precisión-sensibilidad](curva_pr.png)

![Calibración](calibracion.png)
