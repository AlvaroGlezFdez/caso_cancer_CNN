# Informe: 20261022_044451_cv_baseline_f0

- Validación interna: fold 0 (219 pacientes). Test no se ha usado.
- Modelo: CNNBase (98.049 parámetros), dispositivo cuda (AMD Radeon RX 6700 XT).
- pos_weight: no; épocas ejecutadas: 20; mejor época: 10.
- Optimizador AdamW, weight decay 0; aumento de datos: no.
- Métricas por paciente: agregación `mean` de las probabilidades de sus cortes, umbral 0.5.

## Métricas del mejor modelo

| métrica | valor |
|---|---|
| AUC | 0.598 |
| sensibilidad | 0.062 |
| especificidad | 0.929 |
| accuracy | 0.676 |
| precisión | 0.267 |
| Brier | 0.209 |
| probabilidad media predicha | 0.302 |
| tasa real de pCR | 0.292 |
| AUC solo cohorte | 0.534 |

## Por cohorte

| cohorte | pacientes | pCR_1 | tasa_pCR | auc | sensibilidad | especificidad | accuracy | precision | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY1 | 21 | 4 | 0.190 | 0.485 | 0.000 | 0.824 | 0.667 | 0.000 | 0 | 14 | 3 | 4 |
| I-SPY2 | 156 | 49 | 0.314 | 0.601 | 0.041 | 0.953 | 0.667 | 0.286 | 2 | 102 | 5 | 47 |
| Duke | 42 | 11 | 0.262 | 0.657 | 0.182 | 0.903 | 0.714 | 0.400 | 2 | 28 | 3 | 9 |
| Todas | 219 | 64 | 0.292 | 0.598 | 0.062 | 0.929 | 0.676 | 0.267 | 4 | 144 | 11 | 60 |

## Referencia "solo cohorte"

A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.

| cohorte | tasa_pCR_entrenamiento |
|---|---|
| I-SPY1 | 0.265 |
| I-SPY2 | 0.323 |
| Duke | 0.198 |

AUC solo cohorte: **0.534** frente a **0.598** del modelo.

## Gráficas

![Curvas de aprendizaje](curvas_aprendizaje.png)

![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)

![Calibración](calibracion.png)
