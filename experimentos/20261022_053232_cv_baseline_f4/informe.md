# Informe: 20261022_053232_cv_baseline_f4

- Validación interna: fold 4 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 15; mejor época: 5.
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.611 |
| sensibilidad | 0.161 |
| especificidad | 0.924 |
| accuracy | 0.708 |
| precisión | 0.455 |
| Brier | 0.203 |
| probabilidad media predicha | 0.368 |
| tasa real de pCR | 0.283 |
| AUC solo cohorte | 0.554 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 20 | 5 | 0.250 | 0.547 | 0.000 | 0.933 | 0.700 | 0.000 | 0 | 14 | 1 | 5 |
| I-SPY2 | 157 | 49 | 0.312 | 0.599 | 0.204 | 0.907 | 0.688 | 0.500 | 10 | 98 | 10 | 39 |
| Duke | 42 | 8 | 0.190 | 0.728 | 0.000 | 0.971 | 0.786 | 0.000 | 0 | 33 | 1 | 8 |
| Todas | 219 | 62 | 0.283 | 0.611 | 0.161 | 0.924 | 0.708 | 0.455 | 10 | 145 | 12 | 52 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.250 |
| I-SPY2 | 0.324 |
| Duke | 0.216 |

AUC solo cohorte: **0.554** frente a **0.611** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Calibración](calibracion.png)
