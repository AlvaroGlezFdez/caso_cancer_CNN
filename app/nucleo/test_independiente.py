"""Evaluación única en test del modelo final (experimentos/test/, de evaluar_test.py).

Solo lee lo que ya está en git: metricas_test.csv (agregados en formato largo),
la cabecera de resumen_test.md (fecha y SHA-256 del paquete evaluado) y las
figuras roc_test.png, confusion_test.png y calibracion_test.png. No recalcula
nada ni carga etiquetas. El test es independiente (pacientes apartadas desde el
principio) pero de las mismas cohortes que train.
"""

from __future__ import annotations

import hashlib
import math
import re

import pandas as pd

from . import rutas

DIR_TEST = rutas.DIR_EXPERIMENTOS / "test"
MODELOS = ("cohorte", "clinico", "cnn", "combinado")
NIVELES = ("paciente", "corte")
COHORTES = ("I-SPY1", "I-SPY2", "Duke")
METRICAS_UMBRAL = ("sensibilidad", "especificidad", "precision", "vpn")
FIGURAS = {"roc": "roc_test.png", "confusion": "confusion_test.png", "calibracion": "calibracion_test.png"}


def _valor(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _ic(fila) -> list:
    return [_valor(fila.ic95_inf), _valor(fila.ic95_sup)]


def _cabecera() -> dict:
    ruta = DIR_TEST / "resumen_test.md"
    texto = ruta.read_text(encoding="utf-8") if ruta.exists() else ""
    fecha = re.search(r"Fecha: ([0-9T:-]+)", texto)
    sha = re.search(r"modelo\.pt ([0-9a-f]{64}); fusion\.json ([0-9a-f]{64})", texto)
    cnn = re.search(r"CNN: (\S+) \(", texto)
    return {"fecha": fecha[1] if fecha else None, "sha_modelo": sha[1] if sha else None,
            "sha_fusion": sha[2] if sha else None, "ejecucion": cnn[1] if cnn else None}


def cargar() -> dict | None:
    """None si no hay evaluación en test. Si no: {"niveles": {nivel: {...}}, "figuras": {...},
    fecha, sha_modelo, sha_fusion, ejecucion}. Cada nivel: n, pCR_1, modelos (auc, auc_ic95,
    average_precision, brier, prob_media, umbral, VP, FN, FP, VN, sensibilidad… con su IC95,
    auc_cv, auc_cv_ic95), diferencias (valor, ic95) y por_cohorte."""
    ruta = DIR_TEST / "metricas_test.csv"
    if not ruta.exists():
        return None
    d = pd.read_csv(ruta)
    niveles = {}
    for nivel in NIVELES:
        g = d[d.nivel == nivel]
        glob = g[g.ambito == "global"]
        totales = glob[glob.modelo.isna()].set_index("metrica").valor
        modelos = {}
        for k in MODELOS:
            filas = {f.metrica: f for f in glob[glob.modelo == k].itertuples()}
            m = {met: _valor(f.valor) for met, f in filas.items()}
            for c in ("VP", "FN", "FP", "VN"):
                m[c] = int(m[c])
            m["auc_ic95"] = _ic(filas["auc"])
            for met in METRICAS_UMBRAL:
                m[f"{met}_ic95"] = _ic(filas[met])
            cv = g[(g.ambito == "validacion_cruzada") & (g.modelo == k) & (g.metrica == "auc")]
            if len(cv):
                m["auc_cv"], m["auc_cv_ic95"] = _valor(cv.valor.iloc[0]), _ic(cv.iloc[0])
            modelos[k] = m
        diferencias = {f.modelo: {"valor": _valor(f.valor), "ic95": _ic(f)}
                       for f in glob[glob.metrica == "auc_diferencia"].itertuples()}
        por_cohorte = {}
        for c in COHORTES:
            s = g[g.ambito == c]
            if s.empty:
                continue
            t = s[s.modelo.isna()].set_index("metrica").valor
            por_cohorte[c] = {"n": int(t["n"]), "pacientes_pCR": int(t.get("pacientes_pCR", t["pCR_1"])),
                              **{k: {"auc": _valor(f.valor), "auc_ic95": _ic(f)}
                                 for f in s[s.metrica == "auc"].itertuples() for k in [f.modelo]}}
        niveles[nivel] = {"n": int(totales["n"]), "pCR_1": int(totales["pCR_1"]),
                          "tasa_pCR": float(totales["pCR_1"] / totales["n"]),
                          "modelos": modelos, "diferencias": diferencias, "por_cohorte": por_cohorte}
    return {"niveles": niveles, "figuras": {k: DIR_TEST / v for k, v in FIGURAS.items()}, **_cabecera()}


def sha256_texto(ruta) -> str | None:
    """SHA-256 de un fichero de texto con finales de línea LF (git los convierte a CRLF en
    Windows; evaluar_test.py lo calculó en Linux)."""
    if not ruta.exists():
        return None
    return hashlib.sha256(ruta.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def es_el_paquete(test: dict, sha_modelo: str) -> bool:
    """¿El test se hizo con el modelo.pt y el fusion.json que carga la app?"""
    return test.get("sha_modelo") == sha_modelo and test.get("sha_fusion") == sha256_texto(rutas.DIR_MODELO / "fusion.json")
