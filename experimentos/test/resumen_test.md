# Evaluación única en test del modelo final

- Fecha: 2026-10-30T03:23:33; commit 5f3469220f44f7e6d7f5cebce77104c3e47ef14c; comando `python evaluar_test.py --confirmo-evaluacion-unica`.
- CNN: 20261030_003434_cv_receta_centrado160_f0 (configuración receta_centrado160, fold de validación 0, época 12, criterio ultima); entrada 3 × 160 × 160; preprocesado: ventana de 160 px centrada en la caja del tumor, sin reescalar.
- SHA-256: modelo.pt e86a36a9b9e7b28a1aa5e30e3121d74a5b185eff88b824011ce9d4e6a7cd6e1a; fusion.json 42dc2d907e00b9b78ef009e191908a11cfc9fe2590c671bb98b939376def3769.
- Test: 176 pacientes (53 con pCR, tasa 0.301) y 1758 cortes. Nada se ha ajustado ni elegido con test: modelos, escaladores, coeficientes y umbrales vienen de fusion.json (solo train).
- Por paciente (métrica principal): media de las probabilidades de sus cortes. IC95: bootstrap por paciente, 2000 remuestreos, semilla 42, los mismos para todos los modelos (diferencias pareadas); por corte, cada remuestreo toma pacientes con todos sus cortes.
- La CNN de test es una sola ejecución (20261030_003434_cv_receta_centrado160_f0); en la validación cruzada de fusion.json cada corte de train tenía la probabilidad out-of-fold de la CNN de su fold.

- La caja del tumor cabe entera en la ventana en el 95 % de las pacientes de test.

## Por paciente (métrica principal)

| modelo | AUC (IC95) | AP | Brier | prob. media |
|---|---|---|---|---|
| solo cohorte | 0.543 (0.461 a 0.622) | 0.321 | 0.210 | 0.281 |
| clínico | 0.674 (0.590 a 0.758) | 0.400 | 0.197 | 0.299 |
| solo CNN | 0.572 (0.477 a 0.661) | 0.349 | 0.209 | 0.289 |
| combinado | 0.679 (0.596 a 0.760) | 0.406 | 0.196 | 0.293 |

Azar: AP = tasa de pCR 0.301; Brier de predecir siempre la tasa 0.210.

Diferencias de AUC (pareadas):

| diferencia | valor (IC95) |
|---|---|
| clínico − solo cohorte | +0.131 (+0.013 a +0.250) |
| combinado − clínico | +0.005 (-0.021 a +0.029) |
| combinado − solo cohorte | +0.136 (+0.017 a +0.249) |
| combinado − solo CNN | +0.107 (-0.000 a +0.218) |

Con el umbral congelado (positivo si prob ≥ umbral):

| modelo | umbral | VP | FN | FP | VN | sensibilidad | especificidad | precisión | VPN |
|---|---|---|---|---|---|---|---|---|---|
| solo cohorte | 0.3309 | 0 | 53 | 0 | 123 | 0.000 (0.000 a 0.000) | 1.000 (1.000 a 1.000) | — (— a —) | 0.699 (0.631 a 0.767) |
| clínico | 0.4340 | 13 | 40 | 18 | 105 | 0.245 (0.132 a 0.367) | 0.854 (0.790 a 0.914) | 0.419 (0.240 a 0.606) | 0.724 (0.649 a 0.796) |
| solo CNN | 0.3258 | 5 | 48 | 12 | 111 | 0.094 (0.019 a 0.177) | 0.902 (0.847 a 0.952) | 0.294 (0.071 a 0.529) | 0.698 (0.625 a 0.771) |
| combinado | 0.4311 | 15 | 38 | 18 | 105 | 0.283 (0.161 a 0.411) | 0.854 (0.794 a 0.913) | 0.455 (0.278 a 0.629) | 0.734 (0.662 a 0.809) |

AUC por cohorte:

| cohorte | pacientes | pacientes con pCR | solo cohorte | clínico | solo CNN | combinado |
|---|---|---|---|---|---|---|
| I-SPY1 | 35 | 12 | 0.500 (constante en la cohorte) | 0.801 (0.632 a 0.947) | 0.449 (0.254 a 0.653) | 0.783 (0.614 a 0.927) |
| I-SPY2 | 99 | 32 | 0.500 (constante en la cohorte) | 0.728 (0.622 a 0.822) | 0.612 (0.489 a 0.727) | 0.736 (0.635 a 0.827) |
| Duke | 42 | 9 | 0.500 (constante en la cohorte) | 0.492 (0.259 a 0.728) | 0.633 (0.432 a 0.819) | 0.515 (0.287 a 0.747) |

## Por corte

| modelo | AUC (IC95) | AP | Brier | prob. media |
|---|---|---|---|---|
| solo cohorte | 0.543 (0.461 a 0.621) | 0.321 | 0.210 | 0.281 |
| clínico | 0.674 (0.590 a 0.757) | 0.400 | 0.197 | 0.299 |
| solo CNN | 0.570 (0.478 a 0.658) | 0.339 | 0.209 | 0.289 |
| combinado | 0.679 (0.596 a 0.760) | 0.397 | 0.196 | 0.294 |

Azar: AP = tasa de pCR 0.301; Brier de predecir siempre la tasa 0.211.

Diferencias de AUC (pareadas):

| diferencia | valor (IC95) |
|---|---|
| clínico − solo cohorte | +0.131 (+0.013 a +0.249) |
| combinado − clínico | +0.005 (-0.020 a +0.029) |
| combinado − solo cohorte | +0.135 (+0.017 a +0.248) |
| combinado − solo CNN | +0.108 (+0.002 a +0.216) |

Con el umbral congelado (positivo si prob ≥ umbral):

| modelo | umbral | VP | FN | FP | VN | sensibilidad | especificidad | precisión | VPN |
|---|---|---|---|---|---|---|---|---|---|
| solo cohorte | 0.3309 | 0 | 530 | 0 | 1228 | 0.000 (0.000 a 0.000) | 1.000 (1.000 a 1.000) | — (— a —) | 0.699 (0.630 a 0.767) |
| clínico | 0.4358 | 130 | 400 | 170 | 1058 | 0.245 (0.132 a 0.367) | 0.862 (0.801 a 0.921) | 0.433 (0.250 a 0.621) | 0.726 (0.651 a 0.797) |
| solo CNN | 0.3266 | 52 | 478 | 107 | 1121 | 0.098 (0.032 a 0.171) | 0.913 (0.870 a 0.951) | 0.327 (0.119 a 0.532) | 0.701 (0.630 a 0.771) |
| combinado | 0.4330 | 146 | 384 | 180 | 1048 | 0.275 (0.164 a 0.397) | 0.853 (0.795 a 0.911) | 0.448 (0.280 a 0.621) | 0.732 (0.659 a 0.804) |

AUC por cohorte:

| cohorte | pacientes | pacientes con pCR | solo cohorte | clínico | solo CNN | combinado |
|---|---|---|---|---|---|---|
| I-SPY1 | 35 | 12 | 0.500 (constante en la cohorte) | 0.801 (0.632 a 0.947) | 0.437 (0.266 a 0.614) | 0.778 (0.609 a 0.919) |
| I-SPY2 | 99 | 32 | 0.500 (constante en la cohorte) | 0.728 (0.622 a 0.822) | 0.609 (0.488 a 0.721) | 0.736 (0.634 a 0.825) |
| Duke | 42 | 9 | 0.500 (constante en la cohorte) | 0.491 (0.257 a 0.726) | 0.626 (0.431 a 0.804) | 0.512 (0.283 a 0.743) |

## CV vs test

Validación cruzada de 5 folds en train (fusion.json; umbrales elegidos y medidos con las mismas predicciones, así que su sensibilidad es optimista) frente a test con esos umbrales congelados.

Por paciente:

| modelo | AUC CV (IC95) | AUC test (IC95) | test − CV | AP CV / test | Brier CV / test | sensib. CV / test | especif. CV / test |
|---|---|---|---|---|---|---|---|
| solo cohorte | 0.523 (0.484 a 0.560) | 0.543 (0.461 a 0.622) | +0.021 | 0.301 / 0.321 | 0.206 / 0.210 | 0.140 / 0.000 | 0.854 / 1.000 |
| clínico | 0.686 (0.653 a 0.720) | 0.674 (0.590 a 0.758) | -0.012 | 0.435 / 0.400 | 0.189 / 0.197 | 0.320 / 0.245 | 0.850 / 0.854 |
| solo CNN | 0.545 (0.508 a 0.584) | 0.572 (0.477 a 0.661) | +0.027 | 0.325 / 0.349 | 0.207 / 0.209 | 0.202 / 0.094 | 0.850 / 0.902 |
| combinado | 0.691 (0.659 a 0.725) | 0.679 (0.596 a 0.760) | -0.012 | 0.452 / 0.406 | 0.188 / 0.196 | 0.335 / 0.283 | 0.850 / 0.854 |

Combinado − clínico: CV +0.006 (-0.005 a +0.015); test +0.005 (-0.021 a +0.029).

Por corte:

| modelo | AUC CV (IC95) | AUC test (IC95) | test − CV | AP CV / test | Brier CV / test | sensib. CV / test | especif. CV / test |
|---|---|---|---|---|---|---|---|
| solo cohorte | 0.522 (0.484 a 0.560) | 0.543 (0.461 a 0.621) | +0.021 | 0.301 / 0.321 | 0.207 / 0.210 | 0.140 / 0.000 | 0.854 / 1.000 |
| clínico | 0.686 (0.653 a 0.720) | 0.674 (0.590 a 0.757) | -0.012 | 0.435 / 0.400 | 0.189 / 0.197 | 0.314 / 0.245 | 0.851 / 0.862 |
| solo CNN | 0.544 (0.507 a 0.582) | 0.570 (0.478 a 0.658) | +0.027 | 0.321 / 0.339 | 0.207 / 0.209 | 0.197 / 0.098 | 0.850 / 0.913 |
| combinado | 0.691 (0.658 a 0.724) | 0.679 (0.596 a 0.760) | -0.012 | 0.447 / 0.397 | 0.188 / 0.196 | 0.327 / 0.275 | 0.850 / 0.853 |

Combinado − clínico: CV +0.005 (-0.005 a +0.015); test +0.005 (-0.020 a +0.029).

Figuras: roc_test.png, confusion_test.png y calibracion_test.png. Métricas en formato largo: metricas_test.csv.
