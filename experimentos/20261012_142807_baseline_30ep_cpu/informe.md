# Informe: 20261012_142807_baseline_30ep_cpu

- Validación interna: fold 0 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros), dispositivo cpu (AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx).
- pos_weight: no; épocas ejecutadas: 17; mejor época: 10.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.603 |
| sensibilidad | 0.453 |
| especificidad | 0.697 |
| accuracy | 0.626 |
| precisión | 0.382 |
| Brier | 0.228 |
| probabilidad media predicha | 0.442 |
| tasa real de pCR | 0.292 |
| AUC solo cohorte | 0.534 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 4 | 0.190 | 0.662 | 0.000 | 0.941 | 0.762 | 0.000 | 0 | 16 | 1 | 4 |
| I-SPY2 | 156 | 49 | 0.314 | 0.608 | 0.469 | 0.682 | 0.615 | 0.404 | 23 | 73 | 34 | 26 |
| Duke | 42 | 11 | 0.262 | 0.563 | 0.545 | 0.613 | 0.595 | 0.333 | 6 | 19 | 12 | 5 |
| Todas | 219 | 64 | 0.292 | 0.603 | 0.453 | 0.697 | 0.626 | 0.382 | 29 | 108 | 47 | 35 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.265 |
| I-SPY2 | 0.323 |
| Duke | 0.198 |

AUC solo cohorte: **0.534** frente a **0.603** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Calibración](calibracion.png)
