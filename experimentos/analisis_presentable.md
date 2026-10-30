# ¿Hay alguna variante presentable? (validación cruzada de 5 folds)

Generado por `python analisis_presentable.py`. Solo validación interna (train); test no se usa.

## Criterios

- C1: inicio del sobreajuste en la época 11 o después, de media (pérdida de validación > su mínimo hasta entonces + 0.03 dos épocas seguidas).
- C2: el IC95 de la diferencia de AUC con "solo cohorte" queda por encima de 0.
- C3: el límite inferior del IC95 de la sensibilidad supera la del azar con la misma especificidad.

| variante | C1 sobreajuste ≥ 11 | C2 IC dif. > 0 | C3 sens > azar |
|---|---|---|---|
| cv_baseline_suav | no | no | sí |
| cv_aumento | no | no | sí |
| cv_5bloques | no | no | sí |
| cv_realce | no | no | sí |
| cv_recortecaja | no | no | sí |

## 1. Curvas: inicio del sobreajuste por fold

Época de inicio; "≥ n" = no observado en las n-1 épocas entrenadas (el early stopping corta las curvas, así que es una cota inferior). Bajada = pérdida de entrenamiento de la época 1 menos la de la época de inicio (o la última).

| variante | fold | epocas | inicio | bajada_entr |
|---|---|---|---|---|
| cv_baseline_suav | 0 | 21 | 8 | 0.041 |
| cv_baseline_suav | 1 | 24 | 3 | 0.013 |
| cv_baseline_suav | 2 | 30 | 12 | 0.067 |
| cv_baseline_suav | 3 | 30 | 3 | 0.014 |
| cv_baseline_suav | 4 | 17 | 8 | 0.044 |
| cv_aumento | 0 | 20 | 11 | 0.037 |
| cv_aumento | 1 | 23 | 3 | 0.011 |
| cv_aumento | 2 | 12 | ≥ 13 | 0.031 |
| cv_aumento | 3 | 22 | 3 | 0.011 |
| cv_aumento | 4 | 11 | 7 | 0.017 |
| cv_5bloques | 0 | 13 | 4 | 0.030 |
| cv_5bloques | 1 | 22 | 3 | 0.021 |
| cv_5bloques | 2 | 30 | 6 | 0.063 |
| cv_5bloques | 3 | 18 | 4 | 0.027 |
| cv_5bloques | 4 | 17 | 2 | 0.013 |
| cv_realce | 0 | 21 | 6 | 0.038 |
| cv_realce | 1 | 13 | 3 | 0.016 |
| cv_realce | 2 | 15 | 6 | 0.035 |
| cv_realce | 3 | 27 | 4 | 0.023 |
| cv_realce | 4 | 12 | 8 | 0.044 |
| cv_recortecaja | 0 | 13 | 8 | 0.053 |
| cv_recortecaja | 1 | 21 | 8 | 0.072 |
| cv_recortecaja | 2 | 12 | 3 | 0.029 |
| cv_recortecaja | 3 | 30 | 8 | 0.050 |
| cv_recortecaja | 4 | 15 | 7 | 0.049 |

## 2 y 3. Predicciones de los 5 folds juntas (mejor época de cada fold)

AUC por paciente, IC95 por bootstrap (2000 remuestreos por paciente, mismos remuestreos para el modelo y "solo cohorte"). Umbral: el de mayor sensibilidad con especificidad ≥ 0.85, elegido sobre estas mismas predicciones (optimista). Sensibilidad y precisión con IC95 de Wilson; azar con la misma especificidad: sensibilidad 1 - especificidad, precisión = prevalencia.

| variante | pacientes | inicio_sobreajuste_medio | bajada_entr_media | AUC (IC95) | solo cohorte (IC95) | diferencia (IC95) | sens (IC95) / azar | prec (IC95) / azar |
|---|---|---|---|---|---|---|---|---|
| cv_baseline_suav | 1097 | 6.8 | 0.036 | 0.551 (0.513-0.591) | 0.523 (0.485-0.561) | 0.028 (-0.018-0.075) | 0.214 (0.173-0.262) / 0.150 | 0.373 (0.307-0.445) / 0.294 |
| cv_aumento | 1097 | ≥ 7.4 | 0.021 | 0.565 (0.529-0.601) | 0.523 (0.485-0.561) | 0.042 (-0.003-0.090) | 0.193 (0.153-0.239) / 0.150 | 0.348 (0.282-0.421) / 0.294 |
| cv_5bloques | 1097 | 3.8 | 0.031 | 0.538 (0.502-0.575) | 0.523 (0.485-0.561) | 0.016 (-0.044-0.071) | 0.202 (0.162-0.249) / 0.150 | 0.359 (0.293-0.431) / 0.294 |
| cv_realce | 1097 | 5.4 | 0.031 | 0.567 (0.531-0.604) | 0.523 (0.485-0.561) | 0.045 (-0.012-0.101) | 0.202 (0.162-0.249) / 0.150 | 0.359 (0.293-0.431) / 0.294 |
| cv_recortecaja | 1097 | 6.8 | 0.051 | 0.551 (0.511-0.587) | 0.523 (0.485-0.561) | 0.028 (-0.019-0.073) | 0.236 (0.193-0.285) / 0.150 | 0.396 (0.329-0.466) / 0.294 |

Matrices de confusión (filas real, columnas predicho):

- cv_baseline_suav: umbral 0.559, especificidad 0.850. pCR=1: 69 VP / 253 FN; pCR=0: 116 FP / 659 VN.
- cv_aumento: umbral 0.403, especificidad 0.850. pCR=1: 62 VP / 260 FN; pCR=0: 116 FP / 659 VN.
- cv_5bloques: umbral 0.315, especificidad 0.850. pCR=1: 65 VP / 257 FN; pCR=0: 116 FP / 659 VN.
- cv_realce: umbral 0.447, especificidad 0.850. pCR=1: 65 VP / 257 FN; pCR=0: 116 FP / 659 VN.
- cv_recortecaja: umbral 0.394, especificidad 0.850. pCR=1: 76 VP / 246 FN; pCR=0: 116 FP / 659 VN.
