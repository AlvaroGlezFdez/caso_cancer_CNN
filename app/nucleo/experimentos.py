"""Resultados agregados de experimentos/ (en el repositorio): tabla de ejecuciones,
resumen de la validación cruzada por variante y curvas de aprendizaje.

Solo lee ficheros agregados que ya están en git (experimentos.csv y
metricas_epoca.csv de cada ejecución): nada de predicciones por paciente.

Inicio del sobreajuste: la misma regla que analisis_presentable.py, la primera
época en que la pérdida de validación, y también la de la época siguiente,
superan en más de 0,03 su mínimo hasta ese momento.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import rutas

UMBRAL_SOBREAJUSTE = 0.03
PATRON_EJECUCION = re.compile(r"^(?P<fecha>\d{8})_(?P<hora>\d{6})_(?P<nombre>.+)$")
PATRON_CV = re.compile(r"^(?P<variante>cv_.+)_f(?P<fold>[0-4])$")


def tabla() -> pd.DataFrame:
    """experimentos.csv con nombre, variante (cv_*), fold y fecha añadidos."""
    d = pd.read_csv(rutas.DIR_EXPERIMENTOS / "experimentos.csv", float_precision="round_trip")
    partes = d.ejecucion.str.extract(PATRON_EJECUCION)
    d["nombre"] = partes["nombre"]
    d["fecha"] = pd.to_datetime(partes["fecha"], format="%Y%m%d")
    cv = d.nombre.str.extract(PATRON_CV)
    d["variante"] = cv["variante"]
    d["fold"] = pd.to_numeric(cv["fold"])
    # Si un nombre se repite, vale la ejecución más reciente (como cv_resumen.py).
    return d.sort_values("ejecucion").drop_duplicates("nombre", keep="last").reset_index(drop=True)


def resumen_variantes(d: pd.DataFrame) -> pd.DataFrame:
    """Una fila por variante de validación cruzada con sus 5 folds completos."""
    filas = []
    for variante, g in d.dropna(subset=["variante"]).groupby("variante", sort=False):
        if sorted(g.fold.astype(int)) != [0, 1, 2, 3, 4]:
            continue
        filas.append({
            "variante": variante, "fecha": g.fecha.min(), "folds": len(g),
            "auc_media": g.auc_val.mean(), "auc_dt": g.auc_val.std(ddof=1),
            "auc_min": g.auc_val.min(), "auc_max": g.auc_val.max(),
            "ap_media": g.average_precision_val.mean() if g.average_precision_val.notna().all() else np.nan,
            "solo_cohorte_media": g.auc_solo_cohorte.mean(),
            "mejor_epoca_media": g.mejor_epoca.mean(), "epocas_media": g.epocas_ejecutadas.mean(),
            "lr": g.lr.iloc[0], "criterio": g.criterio.fillna("auc").iloc[0],
            "paciencia": g.paciencia.iloc[0], "epocas_max": g.epocas_max.iloc[0],
            "aumento": g.aumento_datos.notna().any(), "pos_weight": g.pos_weight.notna().any(),
            "bloques": int(g.bloques.fillna(4).iloc[0]), "realce": bool(g.realce.fillna(False).astype(bool).iloc[0]),
            "recorte_caja": bool(g.recorte_caja.fillna(False).astype(bool).iloc[0]),
        })
    return pd.DataFrame(filas)


def curvas(ejecucion: str) -> pd.DataFrame | None:
    ruta = rutas.DIR_EXPERIMENTOS / ejecucion / "metricas_epoca.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


def curvas_medias(variante: str) -> pd.DataFrame | None:
    """Media de los 5 folds de una variante (cv_<...>_f0..f4, la ejecución más reciente
    de cada uno) por época: pérdida de entrenamiento y de validación y AUC de
    validación. Solo las épocas que tienen los 5 folds (con parada temprana, las
    primeras). None si faltan folds."""
    d = tabla()
    filas = d[d.variante == variante].sort_values("fold")
    if sorted(filas.fold.astype(int)) != [0, 1, 2, 3, 4]:
        return None
    partes = [curvas(e) for e in filas.ejecucion]
    if any(p is None for p in partes):
        return None
    n = min(len(p) for p in partes)
    columnas = ["perdida_entr", "perdida_val", "auc_val"]
    media = sum(p[columnas].iloc[:n].reset_index(drop=True) for p in partes) / len(partes)
    return media.assign(epoca=np.arange(1, n + 1))


def inicio_sobreajuste(perdida_val: np.ndarray) -> int | None:
    minimo = np.minimum.accumulate(perdida_val)
    por_encima = perdida_val > minimo + UMBRAL_SOBREAJUSTE
    for i in range(len(perdida_val) - 1):
        if por_encima[i] and por_encima[i + 1]:
            return i + 1
    return None


def variante_de(ejecucion: str) -> tuple[str, int | None]:
    """Variante de validación cruzada de una ejecución, y su fold. Una ejecución
    que no es de validación cruzada se asocia a la línea base (cv_baseline)."""
    m = PATRON_EJECUCION.match(ejecucion)
    nombre = m.group("nombre") if m else ejecucion
    cv = PATRON_CV.match(nombre)
    return (cv.group("variante"), int(cv.group("fold"))) if cv else ("cv_baseline", None)


def sobreajuste_por_fold(d: pd.DataFrame, variante: str) -> pd.DataFrame:
    filas = []
    for fila in d[d.variante == variante].sort_values("fold").itertuples():
        c = curvas(fila.ejecucion)
        if c is None:
            continue
        inicio = inicio_sobreajuste(c.perdida_val.to_numpy())
        filas.append({"fold": int(fila.fold), "ejecucion": fila.ejecucion, "epocas": len(c),
                      "inicio": inicio, "observado": inicio is not None, "mejor_epoca": int(fila.mejor_epoca),
                      "auc": fila.auc_val})
    return pd.DataFrame(filas)
