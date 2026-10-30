"""Evaluación ÚNICA en test del modelo final congelado (variables clínicas + CNN).

Uso, desde la raíz del repositorio (necesita breastdcedl/; el paquete va en git):

    python evaluar_test.py                                ensayo en seco: no predice ni escribe
    python evaluar_test.py --confirmo-evaluacion-unica    la evaluación única

Entrada: SOLO el paquete modelo_app/ (modelo.pt, modelo.json, fusion.json) y, de
breastdcedl/metadata/, las filas split == "test" de samples.csv y patients.csv.
No reentrena, no ajusta nada y no elige umbral: los cuatro modelos de
fusion.json (solo cohorte, clínico, solo CNN y combinado) y sus umbrales (por
corte y por paciente) se ajustaron y eligieron solo con train
(exportar_fusion.py), y aquí se aplican tal cual.

Cadena, la misma que la app: cada corte de test -> preprocesado.preprocesar
(con la caja de su paciente si la CNN recorta: caja_tumor.caja_en_png desde
patients.csv, como train.py) -> CNN del paquete en eval() y CPU
(nucleo.inferencia.predecir) -> probabilidad del corte -> cada modelo de
fusion.json (escalador y coeficientes guardados; el combinado se comprueba
contra nucleo.fusion de la app). Por paciente (métrica principal): la media de
las probabilidades de sus cortes.

Se niega a ejecutarse si fusion.json es provisional (o anterior a la versión 2,
sin umbrales por paciente ni los cuatro modelos), si el SHA-256 de modelo.pt no
es el de modelo.json, si la CNN del paquete no es de la configuración de
fusion.json, si hay .py sin commit o cambios sin commit en modelo_app/, o si ya
existe experimentos/test/. Exige --confirmo-evaluacion-unica; sin él hace un
ensayo en seco: comprueba todo lo anterior, carga el paquete, lee y preprocesa
todos los cortes de test y valida variables y cajas, pero no pasa ningún corte
de test por la red ni escribe nada.

Métricas, por paciente y por corte, para los cuatro modelos: AUC con IC95 por
bootstrap por paciente (2000 remuestreos, semilla 42; cada remuestreo toma
pacientes con todos sus cortes; los mismos para todos los modelos, así que las
diferencias combinado - clínico, etc. son pareadas), average precision, Brier,
matriz de confusión, sensibilidad, especificidad, precisión y VPN con el umbral
congelado (IC95 del mismo bootstrap), curva ROC, calibración y AUC por cohorte.
Tabla «CV vs test» con las métricas de validación cruzada de fusion.json.

Salida en experimentos/test/ (se escribe en una carpeta temporal y se renombra
al final): resumen_test.md, metricas_test.csv, roc_test.png, confusion_test.png
y calibracion_test.png. SOLO agregados: ni ids ni etiquetas por paciente; se
comprueba antes de terminar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import informe
from analisis_presentable import N_BOOTSTRAP, SEMILLA
from caja_tumor import caja_en_png
from clinica import COLUMNAS_ENTRADA, Escalador
from exportar_fusion import COHORTES, DIFERENCIAS, MODELOS, NOMBRES, PATRON_IDS, matriz
from preprocesado import FASES, preprocesar, ventana_centrada

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "app"))
from nucleo.fusion import ErrorFusion, cargar_fusion  # noqa: E402
from nucleo.inferencia import necesita_caja, predecir  # noqa: E402
from nucleo.paquete import ErrorPaquete, cargar_paquete  # noqa: E402

META = RAIZ / "breastdcedl" / "metadata"
DIR_PAQUETE = RAIZ / "modelo_app"
DIR_TEST = RAIZ / "experimentos" / "test"
VERSION_FUSION_MINIMA = 2
TOLERANCIA_APP = 1e-12             # combinado vectorizado frente a nucleo.fusion (mismo cálculo)
COLUMNAS_CAJA = ("n_xy", "sraw", "eraw", "scol", "ecol")
NIVELES = ("paciente", "corte")
COLOR = {"combinado": "#2a78d6", "clinico": "#eb6834", "cnn": "#1baf7a", "cohorte": informe.TINTA_2}
ESTILO = {"combinado": "-", "clinico": "-", "cnn": "-", "cohorte": "--"}
METRICAS_UMBRAL = ("sensibilidad", "especificidad", "precision", "vpn")


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------- #
# Comprobaciones previas (también en el ensayo en seco)
# --------------------------------------------------------------------------- #

def comprobar_paquete() -> tuple[list[str], dict | None, dict | None]:
    """Problemas que impiden la evaluación, modelo.json y fusion.json."""
    problemas = []
    rutas = {n: DIR_PAQUETE / n for n in ("modelo.pt", "modelo.json", "fusion.json")}
    faltan = [n for n, r in rutas.items() if not r.is_file()]
    if faltan:
        return [f"faltan en modelo_app/: {', '.join(faltan)}"], None, None
    meta = json.loads(rutas["modelo.json"].read_text(encoding="utf-8"))
    fusion = json.loads(rutas["fusion.json"].read_text(encoding="utf-8"))
    obtenido, esperado = sha256(rutas["modelo.pt"]), meta.get("modelo", {}).get("sha256")
    if obtenido != esperado:
        problemas.append(f"el SHA-256 de modelo.pt ({obtenido[:12]}…) no es el de modelo.json ({str(esperado)[:12]}…)")
    if fusion.get("provisional"):
        problemas.append("fusion.json es PROVISIONAL (probabilidad por paciente en cada corte): reexportarlo con "
                         "exportar_fusion.py sin --provisional")
    if int(fusion.get("version_fusion", 1)) < VERSION_FUSION_MINIMA or not {"umbrales", "modelos_finales"} <= set(fusion):
        problemas.append(f"fusion.json es de la versión {fusion.get('version_fusion', 1)}, sin umbrales por paciente "
                         f"ni los cuatro modelos ajustados con train: reexportarlo con exportar_fusion.py")
    else:
        for m in MODELOS:
            for nivel in NIVELES:
                if not isinstance(fusion["umbrales"].get(nivel, {}).get(m), (int, float)):
                    problemas.append(f"fusion.json no tiene el umbral por {nivel} de {NOMBRES[m]}")
    config_cnn = fusion.get("cnn", {}).get("configuracion")
    hallado = re.search(r"(?:^|_)cv_(?P<config>.+)_f[0-4]$", meta.get("ejecucion", ""))
    if not hallado or hallado["config"] != config_cnn:
        problemas.append(f"la CNN del paquete ({meta.get('ejecucion')}) no es de la configuración de fusion.json "
                         f"({config_cnn})")
    return problemas, meta, fusion


def comprobar_git() -> list[str]:
    cambios_py = git("status", "--porcelain", "--", "*.py")
    cambios_paquete = git("status", "--porcelain", "--", "modelo_app")
    if cambios_py is None or cambios_paquete is None:
        return ["git no responde: no se puede comprobar que el código y el paquete están en un commit"]
    problemas = []
    if cambios_py:
        problemas.append("hay .py sin commit: " + ", ".join(l[2:].strip() for l in cambios_py.splitlines()))
    if cambios_paquete:
        problemas.append("modelo_app/ tiene cambios sin commit (el paquete evaluado debe estar en un commit): "
                         + ", ".join(l[2:].strip() for l in cambios_paquete.splitlines()))
    return problemas


def cargar_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cortes y pacientes de test, con las columnas que se usan; nada de train."""
    pac = pd.read_csv(META / "patients.csv")
    mue = pd.read_csv(META / "samples.csv")
    pac = pac[pac.split == "test"].set_index("pid")
    cortes = mue[mue.split == "test"].reset_index(drop=True)
    return cortes, pac


def validar_test(cortes: pd.DataFrame, pac: pd.DataFrame, meta: dict) -> list[str]:
    """Formatos de las filas de test (sin mirar etiquetas más allá de que sean 0/1)."""
    problemas = []
    if cortes.empty or pac.empty:
        return ["no hay filas de test en samples.csv o patients.csv"]
    faltan = [c for c in ("dataset", *COLUMNAS_ENTRADA, *COLUMNAS_CAJA, "pCR") if c not in pac.columns]
    if faltan:
        problemas.append(f"a patients.csv le faltan columnas: {', '.join(faltan)}")
    if not set(cortes.patient_id) <= set(pac.index):
        problemas.append("hay cortes de test cuya paciente no está en patients.csv con split == test")
    if set(pac.index) - set(cortes.patient_id):
        problemas.append("hay pacientes de test sin cortes en samples.csv")
    if cortes.sample_id.duplicated().any():
        problemas.append("samples.csv repite sample_id en test")
    if not set(pac.dataset.dropna()) <= set(COHORTES) or pac.dataset.isna().any():
        problemas.append("cohortes (dataset) desconocidas o vacías en test")
    for col, tabla in (("pCR", pac), ("pCR", cortes)):
        if col in tabla and not tabla[col].isin([0, 1]).all():
            problemas.append(f"la columna {col} de test no es 0/1 en todas las filas")
    if "pCR" in pac and "pCR" in cortes and set(cortes.patient_id) <= set(pac.index):
        por_paciente = cortes.groupby("patient_id").pCR
        if not (por_paciente.nunique() == 1).all() or not (
                por_paciente.first() == pac.pCR.reindex(por_paciente.first().index)).all():
            problemas.append("la pCR de samples.csv y patients.csv no coincide en test")
    if necesita_caja(meta) and not faltan and pac[list(COLUMNAS_CAJA)].isna().any().any():
        problemas.append("la CNN recorta por la caja y hay pacientes de test sin caja")
    return problemas


def leer_entradas(cortes: pd.DataFrame, pac: pd.DataFrame, meta: dict) -> tuple[list[np.ndarray], list[str], dict]:
    """Preprocesa todos los cortes de test exactamente como train.py y la app."""
    preproc = meta["preprocesado"]
    lado = preproc.get("recorte_centrado")
    cajas = ({pid: caja_en_png(pac.loc[pid]) for pid in pac.index} if necesita_caja(meta) else None)
    entradas, problemas = [], []
    forma = tuple(meta["modelo"]["entrada"])
    for f in cortes.itertuples():
        rutas = [RAIZ / "breastdcedl" / r for r in (f.path_pre, f.path_early, f.path_late)]
        try:
            x = preprocesar(*rutas, caja=None if cajas is None else cajas[f.patient_id],
                            lado_centrado=int(lado) if lado else None)
        except (ValueError, OSError) as e:
            problemas.append(f"un corte de test no se puede preprocesar ({type(e).__name__}: {str(e)[:80]})")
            continue
        if x.shape != forma or not np.isfinite(x).all():
            problemas.append(f"un corte de test da una entrada {x.shape} (se esperaba {forma}) o no finita")
        entradas.append(x)
    resumen = {}
    if lado and cajas is not None:
        dentro = []
        for c in cajas.values():
            r0, k0 = ventana_centrada(c, int(lado))
            dentro.append(r0 <= c[0] and c[1] <= r0 + lado and k0 <= c[2] and c[3] <= k0 + lado)
        resumen["caja_dentro_de_la_ventana"] = float(np.mean(dentro))
    return entradas, sorted(set(problemas)), resumen


# --------------------------------------------------------------------------- #
# Predicción y métricas
# --------------------------------------------------------------------------- #

def probabilidades_fusion(filas: pd.DataFrame, fusion: dict) -> dict[str, np.ndarray]:
    """Los cuatro modelos de fusion.json (modelos_finales), sin reajustar nada."""
    salida = {}
    for m in MODELOS:
        mf = fusion["modelos_finales"][m]
        X = matriz(filas, m)
        if list(X.columns) != list(mf["variables"]):
            sys.exit(f"ERROR: las variables de {NOMBRES[m]} ({list(X.columns)}) no son las de fusion.json "
                     f"({mf['variables']})")
        z = Escalador.desde_dict(mf["escalador"]).transformar(X)[mf["variables"]].to_numpy(float)
        coef = np.array([mf["coeficientes"][v] for v in mf["variables"]], dtype=float)
        salida[m] = 1.0 / (1.0 + np.exp(-(z @ coef + float(mf["intercepto"]))))
    return salida


def con_umbral(y: np.ndarray, p: np.ndarray, umbral: float) -> dict:
    pred = p >= umbral
    vp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    fp, vn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"VP": vp, "FN": fn, "FP": fp, "VN": vn, "sensibilidad": div(vp, vp + fn),
            "especificidad": div(vn, vn + fp), "precision": div(vp, vp + fp), "vpn": div(vn, vn + fn)}


def bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], paciente: np.ndarray, umbrales: dict) -> dict:
    """Remuestreos por paciente con todos sus cortes (como exportar_fusion.bootstrap: misma
    semilla y mismo esquema), los mismos para todos los modelos: AUC y métricas con el umbral."""
    rng = np.random.default_rng(SEMILLA)
    grupos = list(pd.Series(np.arange(len(y))).groupby(paciente).indices.values())
    auc = {k: [] for k in preds}
    umb = {k: {n: [] for n in METRICAS_UMBRAL} for k in preds}
    while len(auc[MODELOS[0]]) < N_BOOTSTRAP:
        i = np.concatenate([grupos[k] for k in rng.integers(0, len(grupos), len(grupos))])
        yy = y[i]
        if yy.min() == yy.max():
            continue
        for k, p in preds.items():
            auc[k].append(roc_auc_score(yy, p[i]))
            c = con_umbral(yy, p[i], umbrales[k])
            for n in METRICAS_UMBRAL:
                umb[k][n].append(c[n])
    return {"auc": {k: np.array(v) for k, v in auc.items()},
            "umbral": {k: {n: np.array(v, dtype=float) for n, v in d.items()} for k, d in umb.items()}}


def ic(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    return tuple(float(x) for x in np.percentile(v, [2.5, 97.5])) if len(v) else (float("nan"), float("nan"))


def metricas_nivel(y: np.ndarray, preds: dict[str, np.ndarray], paciente: np.ndarray, umbrales: dict) -> dict:
    b = bootstrap(y, preds, paciente, umbrales)
    modelos = {}
    for k, p in preds.items():
        real, predicha = calibration_curve(y, p, n_bins=informe.N_GRUPOS_CALIBRACION, strategy="quantile")
        modelos[k] = {
            "auc": float(roc_auc_score(y, p)), "auc_ic95": ic(b["auc"][k]),
            "average_precision": float(average_precision_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
            "prob_media": float(np.mean(p)), "umbral": float(umbrales[k]), **con_umbral(y, p, umbrales[k]),
            **{f"{n}_ic95": ic(b["umbral"][k][n]) for n in METRICAS_UMBRAL},
            "roc": roc_curve(y, p)[:2], "calibracion": (predicha, real),
        }
    diferencias = {f"{a}_menos_{c}": {"valor": modelos[a]["auc"] - modelos[c]["auc"],
                                      "ic95": ic(b["auc"][a] - b["auc"][c])} for a, c in DIFERENCIAS}
    return {"n": int(len(y)), "pCR_1": int(y.sum()), "tasa_pCR": float(y.mean()),
            "brier_tasa_constante": float(y.mean() * (1 - y.mean())), "modelos": modelos, "diferencias": diferencias}


def por_cohorte(y: np.ndarray, preds: dict[str, np.ndarray], cohorte: np.ndarray, paciente: np.ndarray) -> dict:
    """AUC por cohorte; IC95 por bootstrap por paciente dentro de la cohorte (mismos remuestreos para los modelos)."""
    salida = {}
    for c in COHORTES.values():
        s = cohorte == c
        if not s.any():
            continue
        fila = {"n": int(s.sum()), "pacientes": int(len(np.unique(paciente[s]))), "pCR_1": int(y[s].sum()),
                "pacientes_pCR": int(len(np.unique(paciente[s & (y == 1)])))}
        if 0 < y[s].mean() < 1:
            sub = {k: p[s] for k, p in preds.items()}
            b = bootstrap(y[s], sub, paciente[s], {k: 0.5 for k in preds})
            fila.update({k: {"auc": float(roc_auc_score(y[s], sub[k])), "auc_ic95": ic(b["auc"][k])} for k in preds})
        salida[c] = fila
    return salida


# --------------------------------------------------------------------------- #
# Salida
# --------------------------------------------------------------------------- #

def f3(v, signo: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.3f}" if signo else f"{v:.3f}"


def fic(v, i, signo: bool = False) -> str:
    return f"{f3(v, signo)} ({f3(i[0], signo)} a {f3(i[1], signo)})"


def tabla(cabecera: list[str], filas: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(cabecera) + " |", "|" + "---|" * len(cabecera)]
                     + ["| " + " | ".join(f) + " |" for f in filas])


def filas_csv(res: dict, cohortes: dict, cv: dict) -> pd.DataFrame:
    filas = []

    def anadir(nivel, ambito, modelo, metrica, valor, ic95=(None, None)):
        filas.append({"nivel": nivel, "ambito": ambito, "modelo": modelo, "metrica": metrica, "valor": valor,
                      "ic95_inf": ic95[0], "ic95_sup": ic95[1]})

    for nivel in NIVELES:
        r = res[nivel]
        anadir(nivel, "global", "", "n", r["n"])
        anadir(nivel, "global", "", "pCR_1", r["pCR_1"])
        for k, m in r["modelos"].items():
            anadir(nivel, "global", k, "auc", m["auc"], m["auc_ic95"])
            for n in ("average_precision", "brier", "prob_media", "umbral", "VP", "FN", "FP", "VN"):
                anadir(nivel, "global", k, n, m[n])
            for n in METRICAS_UMBRAL:
                anadir(nivel, "global", k, n, m[n], m[f"{n}_ic95"])
            cvm = cv[nivel]["modelos"].get(k, {})
            anadir(nivel, "validacion_cruzada", k, "auc", cvm.get("auc"), tuple(cvm.get("auc_ic95", (None, None))))
        for d, v in r["diferencias"].items():
            anadir(nivel, "global", d, "auc_diferencia", v["valor"], v["ic95"])
        for c, fila in cohortes[nivel].items():
            anadir(nivel, c, "", "n", fila["n"])
            anadir(nivel, c, "", "pCR_1", fila["pCR_1"])
            anadir(nivel, c, "", "pacientes_pCR", fila["pacientes_pCR"])
            for k in MODELOS:
                if k in fila:
                    anadir(nivel, c, k, "auc", fila[k]["auc"], fila[k]["auc_ic95"])
    return pd.DataFrame(filas)


def figura_roc(res: dict, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    for ax, nivel in zip(ejes, NIVELES):
        r = res[nivel]
        ax.plot([0, 1], [0, 1], color=informe.TINTA_2, lw=1, ls=":", label="azar")
        for k in ("cohorte", "clinico", "cnn", "combinado"):
            m = r["modelos"][k]
            ax.plot(*m["roc"], color=COLOR[k], lw=2.2 if k == "combinado" else 1.8, ls=ESTILO[k],
                    label=f"{NOMBRES[k]} (AUC {m['auc']:.3f})".replace(".", ","))
        ax.set_xlim(*informe.ESCALA_PROBABILIDAD)
        ax.set_ylim(*informe.ESCALA_PROBABILIDAD)
        ax.set_aspect("equal")
        ax.set_xlabel("1 - especificidad")
        ax.set_ylabel("sensibilidad")
        ax.set_title(f"Test, por {nivel} (n = {r['n']})", fontsize=11)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        informe._estilo(ax)
    fig.suptitle("Curvas ROC en test: modelos ajustados solo con train", color=informe.TINTA)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def figura_confusion(res: dict, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(2, 4, figsize=(15, 7.6), constrained_layout=True)
    etiquetas = ["no pCR", "pCR"]
    for fila_ejes, nivel in zip(ejes, NIVELES):
        for ax, k in zip(fila_ejes, MODELOS):
            m = res[nivel]["modelos"][k]
            mc = np.array([[m["VN"], m["FP"]], [m["FN"], m["VP"]]])
            ax.imshow(mc, cmap=informe.CMAP_RECUENTOS, vmin=0, vmax=max(int(mc.max()), 1))
            for i, nombres in enumerate((("VN", "FP"), ("FN", "VP"))):
                for j in range(2):
                    claro = mc[i, j] > 0.55 * mc.max()
                    pct = f"{100 * mc[i, j] / mc[i].sum():.0f} %" if mc[i].sum() else "—"
                    ax.text(j, i, f"{nombres[j]} = {mc[i, j]}\n{pct} de la fila", ha="center", va="center",
                            fontsize=10, color="white" if claro else informe.TINTA)
            ax.set_xticks([0, 1], etiquetas)
            ax.set_yticks([0, 1], etiquetas)
            ax.set_xlabel("predicha", color=informe.TINTA_2)
            ax.set_ylabel("real", color=informe.TINTA_2)
            ax.axhline(0.5, color="white", lw=3)
            ax.axvline(0.5, color="white", lw=3)
            ax.spines[:].set_visible(False)
            ax.tick_params(length=0, colors=informe.TINTA_2)
            ax.set_title(f"{NOMBRES[k]}, por {nivel}\numbral congelado {m['umbral']:.4f}".replace(".", ","),
                         fontsize=10, color=informe.TINTA)
    fig.suptitle("Matrices de confusión en test con los umbrales elegidos en train (especificidad ≥ 0,85)",
                 color=informe.TINTA)
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def figura_calibracion(res: dict, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    for ax, nivel in zip(ejes, NIVELES):
        r = res[nivel]
        ax.plot([0, 1], [0, 1], color=informe.TINTA_2, lw=1, ls=":", label="calibración perfecta")
        ax.axhline(r["tasa_pCR"], color=informe.TINTA_2, lw=0.8, alpha=0.5)
        for k in ("cohorte", "clinico", "cnn", "combinado"):
            m = r["modelos"][k]
            ax.plot(*m["calibracion"], color=COLOR[k], lw=1.8, ls=ESTILO[k], marker="o", ms=5,
                    label=f"{NOMBRES[k]} (Brier {m['brier']:.3f})".replace(".", ","))
        ax.set_xlim(*informe.ESCALA_PROBABILIDAD)
        ax.set_ylim(*informe.ESCALA_PROBABILIDAD)
        ax.set_aspect("equal")
        ax.set_xlabel(f"probabilidad media predicha ({informe.N_GRUPOS_CALIBRACION} cuantiles)")
        ax.set_ylabel("fracción observada de pCR")
        ax.set_title(f"Test, por {nivel} (tasa de pCR {r['tasa_pCR']:.3f}; Brier de la tasa constante "
                     f"{r['brier_tasa_constante']:.3f})".replace(".", ","), fontsize=10)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        informe._estilo(ax)
    fig.suptitle("Calibración en test", color=informe.TINTA)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def resumen_md(res: dict, cohortes: dict, cv: dict, contexto: dict) -> str:
    L = [f"# Evaluación única en test del modelo final",
         "",
         f"- Fecha: {contexto['fecha']}; commit {contexto['commit']}; comando `{contexto['comando']}`.",
         f"- CNN: {contexto['ejecucion']} (configuración {contexto['configuracion']}, fold de validación "
         f"{contexto['fold']}, época {contexto['epoca']}, criterio {contexto['criterio']}); entrada "
         f"{contexto['entrada']}; preprocesado: {contexto['preprocesado']}.",
         f"- SHA-256: modelo.pt {contexto['sha_modelo']}; fusion.json {contexto['sha_fusion']}.",
         f"- Test: {res['paciente']['n']} pacientes ({res['paciente']['pCR_1']} con pCR, tasa "
         f"{res['paciente']['tasa_pCR']:.3f}) y {res['corte']['n']} cortes. Nada se ha ajustado ni elegido con test: "
         "modelos, escaladores, coeficientes y umbrales vienen de fusion.json (solo train).",
         f"- Por paciente (métrica principal): media de las probabilidades de sus cortes. IC95: bootstrap por "
         f"paciente, {N_BOOTSTRAP} remuestreos, semilla {SEMILLA}, los mismos para todos los modelos (diferencias "
         "pareadas); por corte, cada remuestreo toma pacientes con todos sus cortes.",
         f"- La CNN de test es una sola ejecución ({contexto['ejecucion']}); en la validación cruzada de fusion.json "
         "cada corte de train tenía la probabilidad out-of-fold de la CNN de su fold.",
         ""]
    if contexto.get("caja_dentro"):
        L.append(f"- La caja del tumor cabe entera en la ventana en el {100 * contexto['caja_dentro']:.0f} % de las "
                 "pacientes de test.")
        L.append("")
    for nivel in NIVELES:
        r = res[nivel]
        L += [f"## Por {nivel}" + (" (métrica principal)" if nivel == "paciente" else ""), "",
              tabla(["modelo", "AUC (IC95)", "AP", "Brier", "prob. media"],
                    [[NOMBRES[k], fic(m["auc"], m["auc_ic95"]), f3(m["average_precision"]), f3(m["brier"]),
                      f3(m["prob_media"])] for k, m in r["modelos"].items()]),
              "",
              f"Azar: AP = tasa de pCR {r['tasa_pCR']:.3f}; Brier de predecir siempre la tasa "
              f"{r['brier_tasa_constante']:.3f}.", "",
              "Diferencias de AUC (pareadas):", "",
              tabla(["diferencia", "valor (IC95)"],
                    [[d.replace("_menos_", " − ").replace("clinico", "clínico").replace("cnn", "solo CNN")
                      .replace("cohorte", "solo cohorte"), fic(v["valor"], v["ic95"], True)]
                     for d, v in r["diferencias"].items()]),
              "",
              "Con el umbral congelado (positivo si prob ≥ umbral):", "",
              tabla(["modelo", "umbral", "VP", "FN", "FP", "VN", "sensibilidad", "especificidad", "precisión", "VPN"],
                    [[NOMBRES[k], f"{m['umbral']:.4f}", str(m["VP"]), str(m["FN"]), str(m["FP"]), str(m["VN"])]
                     + [fic(m[n], m[f"{n}_ic95"]) for n in METRICAS_UMBRAL] for k, m in r["modelos"].items()]),
              "",
              "AUC por cohorte:", "",
              tabla(["cohorte", "pacientes", "pacientes con pCR", *[NOMBRES[k] for k in MODELOS]],
                    [[c, str(f["pacientes"]), str(f["pacientes_pCR"])]
                     + ["0.500 (constante en la cohorte)" if k == "cohorte" else
                        fic(f[k]["auc"], f[k]["auc_ic95"]) if k in f else "—" for k in MODELOS]
                     for c, f in cohortes[nivel].items()]),
              ""]
    L += ["## CV vs test", "",
          "Validación cruzada de 5 folds en train (fusion.json; umbrales elegidos y medidos con las mismas "
          "predicciones, así que su sensibilidad es optimista) frente a test con esos umbrales congelados.", ""]
    for nivel in NIVELES:
        filas = []
        for k in MODELOS:
            c, t = cv[nivel]["modelos"].get(k, {}), res[nivel]["modelos"][k]
            cu = c.get("con_umbral", {})
            filas.append([NOMBRES[k], fic(c.get("auc"), c.get("auc_ic95", (None, None))), fic(t["auc"], t["auc_ic95"]),
                          f3(t["auc"] - c["auc"], True) if "auc" in c else "—",
                          f"{f3(c.get('average_precision'))} / {f3(t['average_precision'])}",
                          f"{f3(c.get('brier'))} / {f3(t['brier'])}",
                          f"{f3(cu.get('sensibilidad'))} / {f3(t['sensibilidad'])}",
                          f"{f3(cu.get('especificidad'))} / {f3(t['especificidad'])}"])
        L += [f"Por {nivel}:", "",
              tabla(["modelo", "AUC CV (IC95)", "AUC test (IC95)", "test − CV", "AP CV / test", "Brier CV / test",
                     "sensib. CV / test", "especif. CV / test"], filas), ""]
        dc = cv[nivel].get("diferencias", {}).get("combinado_menos_clinico")
        dt = res[nivel]["diferencias"]["combinado_menos_clinico"]
        if dc:
            L += [f"Combinado − clínico: CV {fic(dc['valor'], dc['ic95'], True)}; test "
                  f"{fic(dt['valor'], dt['ic95'], True)}.", ""]
    L += ["Figuras: roc_test.png, confusion_test.png y calibracion_test.png. Métricas en formato largo: "
          "metricas_test.csv.", ""]
    return "\n".join(L)


def evaluar(cortes: pd.DataFrame, pac: pd.DataFrame, entradas: list[np.ndarray], paquete, modelo_fusion,
            fusion: dict, meta: dict, resumen_img: dict, destino: Path) -> dict:
    """CNN corte a corte (como la app), los cuatro modelos de fusion.json, métricas y
    salida en `destino` (carpeta temporal renombrada al final). Devuelve las métricas."""
    filas = cortes[["sample_id", "patient_id"]].join(pac[["dataset", *COLUMNAS_ENTRADA]], on="patient_id")
    prob_cnn = np.array([predecir(paquete, x)[0] for x in entradas], dtype=np.float64)
    filas = filas.assign(prob=prob_cnn)
    preds = probabilidades_fusion(filas, fusion)
    app = np.array([modelo_fusion.combinar(p, {c: (None if pd.isna(v) else v) for c, v in zip(COLUMNAS_ENTRADA, r)})
                    .probabilidad for p, r in zip(prob_cnn, filas[COLUMNAS_ENTRADA].itertuples(index=False))])
    if np.abs(app - preds["combinado"]).max() > TOLERANCIA_APP:
        sys.exit("ERROR: el combinado no coincide con el de la app (nucleo.fusion); no se escribe nada")
    print(f"CNN y los cuatro modelos aplicados a {len(filas)} cortes; combinado = app (diferencia máxima "
          f"{np.abs(app - preds['combinado']).max():.1e})")

    y = cortes.pCR.to_numpy().astype(int)
    paciente = cortes.patient_id.to_numpy()
    cohorte_corte = filas.dataset.map(COHORTES).to_numpy()
    medias = pd.DataFrame(preds).groupby(paciente).mean()
    y_pac = pd.Series(y).groupby(paciente).first().reindex(medias.index).to_numpy()
    cohorte_pac = pac.dataset.reindex(medias.index).map(COHORTES).to_numpy()
    preds_pac = {m: medias[m].to_numpy() for m in MODELOS}
    um = fusion["umbrales"]
    res = {"paciente": metricas_nivel(y_pac, preds_pac, medias.index.to_numpy(), um["paciente"]),
           "corte": metricas_nivel(y, preds, paciente, um["corte"])}
    cohortes = {"paciente": por_cohorte(y_pac, preds_pac, cohorte_pac, medias.index.to_numpy()),
                "corte": por_cohorte(y, preds, cohorte_corte, paciente)}
    cv = {"paciente": fusion["validacion_cruzada"]["por_paciente"], "corte": fusion["validacion_cruzada"]["por_corte"]}

    contexto = {
        "fecha": datetime.now().isoformat(timespec="seconds"), "commit": git("rev-parse", "HEAD"),
        "comando": "python evaluar_test.py --confirmo-evaluacion-unica",
        "ejecucion": meta["ejecucion"], "configuracion": fusion["cnn"]["configuracion"],
        "fold": meta.get("entrenamiento", {}).get("fold_validacion"), "epoca": meta["modelo"]["epoca"],
        "criterio": (meta["modelo"].get("seleccion") or {}).get("criterio"),
        "entrada": " × ".join(map(str, meta["modelo"]["entrada"])),
        "preprocesado": (f"ventana de {meta['preprocesado']['recorte_centrado']} px centrada en la caja del tumor, "
                         "sin reescalar" if meta["preprocesado"].get("recorte_centrado") else
                         "recorte por la caja" if meta["preprocesado"].get("recorte_caja") else "imagen completa"),
        "sha_modelo": meta["modelo"]["sha256"], "sha_fusion": sha256(DIR_PAQUETE / "fusion.json"),
        "caja_dentro": resumen_img.get("caja_dentro_de_la_ventana"),
    }

    # Escritura: carpeta temporal, comprobación de que solo hay agregados y renombrado.
    temporal = destino.parent / f".{destino.name}_en_curso_{datetime.now():%Y%m%d_%H%M%S}"
    temporal.mkdir(parents=True)
    try:
        (temporal / "resumen_test.md").write_text(resumen_md(res, cohortes, cv, contexto), encoding="utf-8")
        filas_csv(res, cohortes, cv).to_csv(temporal / "metricas_test.csv", index=False)
        figura_roc(res, temporal / "roc_test.png")
        figura_confusion(res, temporal / "confusion_test.png")
        figura_calibracion(res, temporal / "calibracion_test.png")
        ids = set(cortes.sample_id) | set(cortes.patient_id)
        for ruta in (temporal / "resumen_test.md", temporal / "metricas_test.csv"):
            texto = ruta.read_text(encoding="utf-8")
            if PATRON_IDS.search(texto) or any(i in texto for i in ids):
                raise RuntimeError(f"{ruta.name} contendría un id de paciente o de corte")
        metricas = pd.read_csv(temporal / "metricas_test.csv")
        if (metricas[metricas.metrica == "n"].valor < 2).any():   # un grupo de 1 revelaría una etiqueta
            raise RuntimeError("metricas_test.csv tiene un grupo de una sola paciente o un solo corte")
        os.replace(temporal, destino)
    except Exception:
        shutil.rmtree(temporal, ignore_errors=True)
        raise

    print(f"\nResultados en {destino.as_posix()}/ (solo agregados)")
    for nivel in NIVELES:
        r = res[nivel]
        print(f"  Por {nivel}: " + "; ".join(f"{NOMBRES[k]} {m['auc']:.3f} ({m['auc_ic95'][0]:.3f}-"
                                              f"{m['auc_ic95'][1]:.3f})" for k, m in r["modelos"].items()))
        d = r["diferencias"]["combinado_menos_clinico"]
        c = r["modelos"]["combinado"]
        print(f"    combinado - clínico {d['valor']:+.3f} ({d['ic95'][0]:+.3f} a {d['ic95'][1]:+.3f}); combinado con "
              f"umbral {c['umbral']:.4f}: sensibilidad {c['sensibilidad']:.3f}, especificidad {c['especificidad']:.3f}")
    return res


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmo-evaluacion-unica", action="store_true",
                    help="hace la evaluación única en test y escribe experimentos/test/ (sin él, ensayo en seco)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    seco = not args.confirmo_evaluacion_unica
    print("ENSAYO EN SECO: no se pasa ningún corte de test por la red ni se escribe nada.\n" if seco else
          "EVALUACIÓN ÚNICA EN TEST\n")

    problemas, meta, fusion = comprobar_paquete()
    problemas += comprobar_git()
    if DIR_TEST.exists():
        problemas.append(f"ya existe {DIR_TEST.relative_to(RAIZ).as_posix()}/: la evaluación en test es única")
    if meta is None:
        sys.exit("ERROR: " + "; ".join(problemas))

    # El paquete se carga con el mismo código que la app (verifica de nuevo el SHA-256).
    try:
        paquete = cargar_paquete(DIR_PAQUETE, "cpu")
        modelo_fusion = cargar_fusion(DIR_PAQUETE)
    except (ErrorPaquete, ErrorFusion) as e:
        problemas.append(str(e))
        paquete = modelo_fusion = None
    if paquete is not None:
        import torch
        with torch.no_grad():                           # una entrada de ceros, no de test
            forma = tuple(paquete.meta["modelo"]["entrada"])
            salida = paquete.modelo(torch.zeros((1, *forma)))
        if tuple(salida.shape) != (1, 1):
            problemas.append(f"la CNN da una salida {tuple(salida.shape)} con una entrada {forma}")

    cortes, pac = cargar_test()
    problemas += validar_test(cortes, pac, meta)
    entradas, problemas_img, resumen_img = ([], [], {})
    if not any("faltan columnas" in p or "no hay filas" in p for p in problemas):
        entradas, problemas_img, resumen_img = leer_entradas(cortes, pac, meta)
        problemas += problemas_img
        filas = cortes[["sample_id", "patient_id"]].join(pac[["dataset", *COLUMNAS_ENTRADA]], on="patient_id")
        for m in ("cohorte", "clinico"):
            columnas = list(matriz(filas.assign(prob=0.5), m).columns)
            if fusion and "modelos_finales" in fusion and columnas != fusion["modelos_finales"][m]["variables"]:
                problemas.append(f"las variables de {NOMBRES[m]} no son las de fusion.json")

    print(f"Paquete: {meta.get('ejecucion')} (época {meta['modelo'].get('epoca')}, criterio "
          f"{(meta['modelo'].get('seleccion') or {}).get('criterio', '?')}), entrada {meta['modelo'].get('entrada')}, "
          f"recorte por la caja {'sí' if meta['preprocesado'].get('recorte_caja') else 'no'}, ventana centrada "
          f"{meta['preprocesado'].get('recorte_centrado') or 'no'}")
    if fusion:
        print(f"Fusión: configuración {fusion.get('cnn', {}).get('configuracion')}, versión "
              f"{fusion.get('version_fusion', 1)}, provisional {'sí' if fusion.get('provisional') else 'no'}")
    print(f"Test: {pac.shape[0]} pacientes ({', '.join(f'{COHORTES.get(k, k)} {v}' for k, v in pac.dataset.value_counts().items())}), "
          f"{len(cortes)} cortes; {len(entradas)} cortes leídos y preprocesados con la forma esperada"
          + (f"; la caja cabe en la ventana en el {100 * resumen_img['caja_dentro_de_la_ventana']:.0f} % de las "
             "pacientes"
             if "caja_dentro_de_la_ventana" in resumen_img else ""))
    faltantes = {c: int(pac[c].isna().sum()) for c in COLUMNAS_ENTRADA if c in pac and pac[c].isna().any()}
    if faltantes:
        print(f"Variables vacías en test (se rellenan con la media de train, como en el ajuste): {faltantes}")

    if problemas:
        print("\nNO se puede hacer la evaluación:")
        for p in problemas:
            print(f"  - {p}")
        sys.exit(1)
    if seco:
        print("\nTodo listo. Para la evaluación ÚNICA: python evaluar_test.py --confirmo-evaluacion-unica")
        return

    evaluar(cortes, pac, entradas, paquete, modelo_fusion, fusion, meta, resumen_img, DIR_TEST)


if __name__ == "__main__":
    main()
