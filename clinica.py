"""Variables clínicas para el modelo combinado (imagen + clínica).

Única fuente de verdad de cómo se construyen las variables clínicas: la usan
fusion_clinica.py, exportar_fusion.py y la app (app/nucleo/fusion.py), igual que
preprocesado.py para las imágenes. Solo depende de numpy y pandas.

Variables (todas conocidas antes del tratamiento):
- subtipo (HR_HER2_STATUS): HRposHER2neg, HER2pos o TripleNeg, en indicadores 0/1;
- log(1 + tum_vol): volumen del tumor;
- age: edad.
No se usan: menopause (12 % de valores faltantes), race_* (la guía advierte de
que tratarlas como causales es un error metodológico) ni las de adquisición.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUBTIPOS = ["HRposHER2neg", "HER2pos", "TripleNeg"]
COLUMNAS_ENTRADA = ["HR_HER2_STATUS", "tum_vol", "age"]


def logit(p):
    """log(p / (1 - p)), con p recortada a [1e-6, 1 - 1e-6]: la entrada de la CNN
    en el modelo combinado."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la matriz de variables clínicas (sin rellenar faltantes)."""
    faltan = [c for c in COLUMNAS_ENTRADA if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas clínicas: {faltan}")
    X = pd.DataFrame(index=df.index)
    sub = df["HR_HER2_STATUS"]
    for s in SUBTIPOS:
        X[f"sub_{s}"] = np.where(sub.isna(), np.nan, (sub == s).astype(float))
    X["log_tum_vol"] = np.log1p(pd.to_numeric(df["tum_vol"], errors="coerce"))
    X["age"] = pd.to_numeric(df["age"], errors="coerce")
    return X


class Escalador:
    """Rellena faltantes con la media de entrenamiento y estandariza.

    Se ajusta SOLO con las pacientes de entrenamiento de cada fold, para no
    filtrar información de validación.
    """

    def ajustar(self, X: pd.DataFrame) -> "Escalador":
        self.media = X.mean()
        Xr = X.fillna(self.media)
        self.desv = Xr.std(ddof=0).replace(0, 1.0)
        return self

    def transformar(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X.fillna(self.media) - self.media) / self.desv

    def a_dict(self) -> dict:
        """Media y desviación por variable, para guardarlas en JSON (modelo_app/fusion.json)."""
        return {"media": {k: float(v) for k, v in self.media.items()},
                "desv": {k: float(v) for k, v in self.desv.items()}}

    @classmethod
    def desde_dict(cls, d: dict) -> "Escalador":
        esc = cls()
        esc.media, esc.desv = pd.Series(d["media"], dtype=float), pd.Series(d["desv"], dtype=float)
        return esc
