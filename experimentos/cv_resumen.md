# Validación cruzada de 5 folds: línea base frente a aumento

AUC de validación por paciente (agregación media) del mejor modelo de cada ejecución.
La desviación típica es la muestral (ddof=1) sobre los 5 folds.

| fold | AUC línea base | AUC aumento | aumento - línea base | AUC solo cohorte |
|---|---|---|---|---|
| 0 | 0.598 | 0.577 | -0.021 | 0.534 |
| 1 | 0.626 | 0.569 | -0.057 | 0.526 |
| 2 | 0.608 | 0.600 | -0.008 | 0.538 |
| 3 | 0.540 | 0.549 | +0.009 | 0.593 |
| 4 | 0.611 | 0.614 | +0.004 | 0.554 |
| media | 0.597 | 0.582 | -0.015 | 0.549 |
| desv_tipica | 0.033 | 0.026 | 0.026 | 0.027 |

El aumento supera a la línea base en 2 de 5 folds.

## Ejecuciones usadas

| configuración | fold | ejecución | mejor época | épocas ejecutadas |
|---|---|---|---|---|
| linea_base | 0 | 20261022_044451_cv_baseline_f0 | 10 | 20 |
| linea_base | 1 | 20261022_045427_cv_baseline_f1 | 13 | 23 |
| linea_base | 2 | 20261022_050538_cv_baseline_f2 | 22 | 30 |
| linea_base | 3 | 20261022_052003_cv_baseline_f3 | 16 | 26 |
| linea_base | 4 | 20261022_053232_cv_baseline_f4 | 5 | 15 |
| aumento | 0 | 20261022_053954_cv_aumento_f0 | 10 | 20 |
| aumento | 1 | 20261022_054946_cv_aumento_f1 | 13 | 23 |
| aumento | 2 | 20261022_060058_cv_aumento_f2 | 2 | 12 |
| aumento | 3 | 20261022_060652_cv_aumento_f3 | 12 | 22 |
| aumento | 4 | 20261022_061736_cv_aumento_f4 | 1 | 11 |
