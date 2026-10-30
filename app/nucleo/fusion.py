"""Modelo final combinado (variables clínicas + CNN), de modelo_app/fusion.json
(exportar_fusion.py).

    x  = clinica.construir_variables(paciente)  +  logit_cnn = clinica.logit(prob. de la CNN)
    z  = clinica.Escalador (media y desviación de fusion.json; faltantes -> media)
    P(pCR) = sigmoide(intercepto + suma(coef_j * z_j))      positivo si P >= umbral

Es la misma cuenta que LogisticRegression.predict_proba en exportar_fusion.py
(prueba_app.py lo comprueba). La contribución de cada variable al logit es
coef_j * z_j: como z = 0 es la media de entrenamiento, el intercepto es el logit
de una paciente «media» y cada contribución dice cuánto se aparta de él. El
subtipo son tres indicadores; su contribución es la suma de los tres.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import rutas
from clinica import COLUMNAS_ENTRADA, Escalador, construir_variables, logit  # noqa: E402

GRUPOS = {"Subtipo": ("sub_HRposHER2neg", "sub_HER2pos", "sub_TripleNeg"), "Volumen tumoral": ("log_tum_vol",),
          "Edad": ("age",), "Imagen (CNN)": ("logit_cnn",)}
PATRON_CV = re.compile(r"(?:^|_)cv_(?P<config>.+)_f[0-4]$")


class ErrorFusion(Exception):
    """modelo_app/fusion.json está incompleto o no es coherente con clinica.py."""


@dataclass
class ResultadoFusion:
    probabilidad: float              # modelo combinado: la principal
    logit: float
    umbral: float
    clase: int                       # 1 = pCR estimada (probabilidad >= umbral)
    prob_imagen: float               # la CNN sola
    prob_clinica: float              # el modelo clínico solo (referencia)
    intercepto: float
    contribuciones: dict[str, float] = field(default_factory=dict)   # grupo -> aporte al logit
    por_variable: dict[str, dict] = field(default_factory=dict)      # variable -> x, z, coef, aporte
    imputadas: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModeloFusion:
    meta: dict
    variables: tuple
    escalador: Escalador
    coef: np.ndarray
    intercepto: float
    clinico: tuple                   # (variables, escalador, coef, intercepto) del modelo clínico

    @property
    def umbral(self) -> float:
        return float(self.meta["umbral"]["valor"])

    @property
    def configuracion(self) -> str:
        return self.meta["cnn"]["configuracion"]

    @property
    def provisional(self) -> bool:
        return bool(self.meta.get("provisional"))

    def coincide_cnn(self, ejecucion: str) -> bool:
        """¿La CNN del paquete (modelo.json) es de la configuración de la fusión?"""
        m = PATRON_CV.search(ejecucion)
        return bool(m) and m["config"] == self.configuracion

    def combinar(self, prob_cnn: float, valores: dict) -> ResultadoFusion:
        """`valores`: columnas de patients.csv (HR_HER2_STATUS, tum_vol, age); None = sin dato."""
        X = construir_variables(pd.DataFrame([{c: valores.get(c) for c in COLUMNAS_ENTRADA}]))
        X["logit_cnn"] = float(logit(np.float64(prob_cnn)))
        X = X[list(self.variables)].astype(float)
        imputadas = [v for v in self.variables if math.isnan(X[v].iloc[0])]
        z = self.escalador.transformar(X)[list(self.variables)].to_numpy(float)[0]
        lg = float(z @ self.coef + self.intercepto)
        prob = 1.0 / (1.0 + math.exp(-lg))
        aporte = z * self.coef
        por_variable = {v: {"x": float(X[v].iloc[0]), "z": float(z[i]), "coef": float(self.coef[i]),
                            "aporte": float(aporte[i])} for i, v in enumerate(self.variables)}
        contribuciones = {g: float(sum(por_variable[v]["aporte"] for v in vs if v in por_variable))
                          for g, vs in GRUPOS.items()}

        variables_c, esc_c, coef_c, int_c = self.clinico
        zc = esc_c.transformar(X[list(variables_c)])[list(variables_c)].to_numpy(float)[0]
        prob_c = 1.0 / (1.0 + math.exp(-float(zc @ coef_c + int_c)))
        return ResultadoFusion(probabilidad=prob, logit=lg, umbral=self.umbral, clase=int(prob >= self.umbral),
                               prob_imagen=float(prob_cnn), prob_clinica=prob_c, intercepto=self.intercepto,
                               contribuciones=contribuciones, por_variable=por_variable, imputadas=imputadas)


def cargar_fusion(directorio: Path = rutas.DIR_MODELO) -> ModeloFusion | None:
    """None si no hay fusion.json (la app usa entonces solo la imagen)."""
    ruta = directorio / "fusion.json"
    if not ruta.exists():
        return None
    try:
        meta = json.loads(ruta.read_text(encoding="utf-8"))
        m = meta["modelo"]
        variables = tuple(m["variables"])
        esc = Escalador.desde_dict(m["escalador"])
        coef = np.array([m["coeficientes"][v] for v in variables], dtype=float)
        intercepto = float(m["intercepto"])
        float(meta["umbral"]["valor"])
        meta["cnn"]["configuracion"]
        mc = meta["modelo_clinico"]
        clinico = (tuple(mc["variables"]), Escalador.desde_dict(mc["escalador"]),
                   np.array([mc["coeficientes"][v] for v in mc["variables"]], dtype=float), float(mc["intercepto"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        raise ErrorFusion(f"modelo_app/fusion.json está incompleto o dañado ({type(e).__name__}: {e}).") from e
    # Las variables tienen que ser las que construye clinica.py, más la de la CNN.
    esperadas = tuple(construir_variables(pd.DataFrame({c: [None] for c in COLUMNAS_ENTRADA})).columns) + ("logit_cnn",)
    if variables != esperadas or set(esc.media.index) != set(variables) or set(esc.desv.index) != set(variables):
        raise ErrorFusion(f"Las variables de fusion.json ({', '.join(variables)}) no son las de clinica.py "
                          f"({', '.join(esperadas)}): hay que volver a exportarlo (exportar_fusion.py).")
    return ModeloFusion(meta=meta, variables=variables, escalador=esc, coef=coef, intercepto=intercepto,
                        clinico=clinico)
