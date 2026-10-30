"""Análisis del último bloque de experimentos de imagen (la «receta regularizada»).

Se ejecuta en el portátil, después de traer del laboratorio experimentos/ (con
experimentos/oof/ y experimentos/oof_cortes/ de exportar_oof.py):

    python analisis_ultimo.py

Configuraciones: las 4 nuevas (cv_receta_baseline, cv_receta_centrado160,
cv_receta_recortecaja y cv_receta_centrales4) frente a tres de referencia
(cv_baseline_suav, cv_recortecaja y cv_recortecaja_lr3e-4). Las que falten se
indican y se omiten.

1. AUC por paciente (probabilidades out-of-fold de los 5 folds juntos,
   experimentos/oof/) con IC95 por bootstrap por paciente, el de «solo cohorte»
   (tasa de pCR de la cohorte en la partición de entrenamiento de cada fold) y
   la diferencia con cv_baseline_suav; los mismos remuestreos para todas
   (diferencias pareadas).
2. Sobreajuste: distancia entre la pérdida de validación y la de entrenamiento
   (val - entr) en la época 6 y en la última, media y DT de los 5 folds
   (metricas_epoca.csv), y figura con las curvas medias de los 5 folds
   (pérdidas y AUC por época) con las escalas fijas de informe.py.
3. Cortes centrales al agregar: con experimentos/oof_cortes/, AUC por paciente
   promediando solo sus k cortes centrales (muestreo.cortes_centrales, la misma
   regla que --cortes-centrales), k = 2, 4, 6 y todos.

Escribe solo agregados: experimentos/analisis_ultimo.md y
experimentos/analisis_ultimo_curvas.png. Las etiquetas (patients.csv) solo se
leen. Nunca usa test.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import informe
import muestreo

RAIZ = Path(__file__).resolve().parent
META = RAIZ / "breastdcedl" / "metadata"
EXP = RAIZ / "experimentos"
NUEVAS = ["receta_baseline", "receta_centrado160", "receta_recortecaja", "receta_centrales4"]
REFERENCIAS = ["baseline_suav", "recortecaja", "recortecaja_lr3e-4"]
REFERENCIA_DIFERENCIA = "baseline_suav"
KS = (2, 4, 6, None)                        # None = todos los cortes
N_BOOT = 2000
SEMILLA = 42
EPOCA_INTERMEDIA = 6
ESCALA_EPOCAS = (1, 40)                     # las de referencia con lr 3e-4 llegan a 40
SALIDA_MD = EXP / "analisis_ultimo.md"
SALIDA_PNG = EXP / "analisis_ultimo_curvas.png"
COHORTES = {"spy1": "I-SPY1", "spy2": "I-SPY2", "duke": "Duke"}


def pacientes_train() -> pd.DataFrame:
    """Una fila por paciente de train: pCR, cohorte, fold y la tasa de pCR de su cohorte
    en la partición de entrenamiento de su fold («solo cohorte»)."""
    pac = pd.read_csv(META / "patients.csv")
    mue = pd.read_csv(META / "samples.csv")
    fold = mue[mue.split == "train"].groupby("patient_id").fold.first()
    t = pac[pac.split == "train"].set_index("pid").join(fold, how="inner")
    t["cohorte"] = t.dataset.map(COHORTES)
    t["prob_cohorte"] = np.nan
    for f in range(5):
        tasa = t[t.fold != f].groupby("cohorte").pCR.mean()
        t.loc[t.fold == f, "prob_cohorte"] = t.loc[t.fold == f, "cohorte"].map(tasa)
    return t


def remuestreos(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(SEMILLA)
    return [rng.integers(0, n, n) for _ in range(N_BOOT)]


def auc_con_ic(y: np.ndarray, p: np.ndarray, muestras: list[np.ndarray]) -> tuple[float, np.ndarray]:
    """AUC y su distribución bootstrap (NaN en los remuestreos con una sola clase)."""
    b = np.array([roc_auc_score(y[i], p[i]) if 0 < y[i].mean() < 1 else np.nan for i in muestras])
    return float(roc_auc_score(y, p)), b


def ic(b: np.ndarray) -> str:
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return f"{lo:.3f}-{hi:.3f}"


def ic_signo(b: np.ndarray) -> str:
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return f"{lo:+.3f} a {hi:+.3f}"


def ejecuciones(config: str) -> dict[int, Path]:
    """fold -> carpeta de experimentos/ de cv_<config>_f<fold> (la más reciente si se repite)."""
    patron = re.compile(rf"^\d{{8}}_\d{{6}}_cv_{re.escape(config)}_f(?P<fold>[0-4])$")
    encontradas: dict[int, Path] = {}
    for d in sorted(EXP.iterdir()):
        m = patron.match(d.name)
        if m and (d / "metricas_epoca.csv").is_file():
            encontradas[int(m["fold"])] = d
    return encontradas


def seccion_auc(t: pd.DataFrame, configs: list[str]) -> tuple[list[str], dict]:
    lineas = ["## 1. AUC por paciente (5 folds juntos)", "",
              f"Probabilidades out-of-fold de experimentos/oof/. IC95 por bootstrap por paciente ({N_BOOT} "
              f"remuestreos, semilla {SEMILLA}, los mismos para todas). Diferencia pareada con "
              f"{REFERENCIA_DIFERENCIA}.", "",
              "| configuración | pacientes | AUC (IC95) | solo cohorte (IC95) | AUC - cohorte (IC95) | "
              f"AUC - {REFERENCIA_DIFERENCIA} (IC95) |", "|---|---|---|---|---|---|"]
    probs = {}
    for c in configs:
        ruta = EXP / "oof" / f"{c}.csv"
        if ruta.is_file():
            probs[c] = pd.read_csv(ruta).set_index("patient_id").prob
    faltan = [c for c in configs if c not in probs]
    comunes = t.index
    for p in probs.values():
        comunes = comunes.intersection(p.index)
    tt = t.loc[comunes]
    y = tt.pCR.to_numpy()
    muestras = remuestreos(len(tt))
    auc_c, b_c = auc_con_ic(y, tt.prob_cohorte.to_numpy(), muestras)
    boots = {}
    for c, p in probs.items():
        a, b = auc_con_ic(y, p.reindex(comunes).to_numpy(), muestras)
        boots[c] = (a, b)
    for c, (a, b) in boots.items():
        if REFERENCIA_DIFERENCIA in boots and c != REFERENCIA_DIFERENCIA:
            ar, br = boots[REFERENCIA_DIFERENCIA]
            dif = f"{a - ar:+.3f} ({ic_signo(b - br)})"
        else:
            dif = "—"
        lineas.append(f"| {c} | {len(tt)} | {a:.3f} ({ic(b)}) | {auc_c:.3f} ({ic(b_c)}) | "
                      f"{a - auc_c:+.3f} ({ic_signo(b - b_c)}) | {dif} |")
    if faltan:
        lineas += ["", "Sin experimentos/oof/ de: " + ", ".join(faltan) + ". Ejecuta exportar_oof.py en la "
                   "máquina con los pesos (experimentos/oof*/ no se versiona: lleva ids de pacientes)."]
    return lineas, boots


def seccion_sobreajuste(configs: list[str]) -> tuple[list[str], dict]:
    lineas = ["", "## 2. Sobreajuste: pérdida de validación - pérdida de entrenamiento", "",
              f"Media (DT) de los 5 folds en la época {EPOCA_INTERMEDIA} y en la última entrenada. En la receta, "
              "la pérdida de entrenamiento es la de los 3 cortes por paciente de cada época.", "",
              f"| configuración | folds | épocas | val - entr (época {EPOCA_INTERMEDIA}) | val - entr (última) | "
              "pérdida val (última) | AUC val (última) |", "|---|---|---|---|---|---|---|"]
    curvas = {}
    for c in configs:
        runs = ejecuciones(c)
        if len(runs) < 5:
            lineas.append(f"| {c} | {len(runs)} | pendiente | | | | |")
            continue
        tablas = [pd.read_csv(d / "metricas_epoca.csv") for d in runs.values()]
        curvas[c] = tablas
        gap = lambda m, e: float(m.perdida_val.iloc[e - 1] - m.perdida_entr.iloc[e - 1]) if len(m) >= e else np.nan  # noqa: E731
        g6 = np.array([gap(m, EPOCA_INTERMEDIA) for m in tablas])
        gu = np.array([gap(m, len(m)) for m in tablas])
        pv = np.array([m.perdida_val.iloc[-1] for m in tablas])
        av = np.array([m.auc_val.iloc[-1] for m in tablas])
        epocas = sorted({len(m) for m in tablas})
        texto_ep = str(epocas[0]) if len(epocas) == 1 else f"{epocas[0]}-{epocas[-1]}"
        lineas.append(f"| {c} | 5 | {texto_ep} | {np.nanmean(g6):+.3f} ({np.nanstd(g6, ddof=1):.3f}) | "
                      f"{gu.mean():+.3f} ({gu.std(ddof=1):.3f}) | {pv.mean():.3f} | {av.mean():.3f} |")
    lineas += ["", "En las de referencia la «última» es la última entrenada (con early stopping o 40 épocas), no la "
               "elegida por el criterio."]
    return lineas, curvas


def figura(curvas: dict, ruta: Path) -> None:
    n = len(curvas)
    fig, ejes = plt.subplots(2, max(n, 1), figsize=(3.4 * max(n, 1), 6.4), squeeze=False, layout="constrained")
    for j, (c, tablas) in enumerate(curvas.items()):
        largo = max(len(m) for m in tablas)
        def media(col):
            mat = np.full((len(tablas), largo), np.nan)
            for i, m in enumerate(tablas):
                mat[i, :len(m)] = m[col].to_numpy()
            return np.nanmean(mat, axis=0)
        x = np.arange(1, largo + 1)
        a1, a2 = ejes[0, j], ejes[1, j]
        a1.plot(x, media("perdida_entr"), color="#1f5fa8", lw=1.8, label="entrenamiento")
        a1.plot(x, media("perdida_val"), color="#c2410c", lw=1.8, label="validación")
        a1.set(xlim=ESCALA_EPOCAS, ylim=informe.ESCALA_PERDIDA, title=c, xlabel="época")
        a2.plot(x, media("auc_val"), color="#0e7490", lw=1.8)
        a2.axhline(0.5, color="grey", lw=0.8, ls=":")
        a2.set(xlim=ESCALA_EPOCAS, ylim=informe.ESCALA_AUC_EPOCA, xlabel="época")
        for a in (a1, a2):
            a.grid(alpha=0.3)
        if j == 0:
            a1.set_ylabel("pérdida (BCE), media de 5 folds")
            a2.set_ylabel("AUC val por paciente, media de 5 folds")
            a1.legend(fontsize=8)
    fig.suptitle("Curvas medias de los 5 folds (escalas fijas; en épocas con folds ya parados, media de los "
                 "que siguen)", fontsize=11)
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def seccion_centrales(t: pd.DataFrame, configs: list[str]) -> list[str]:
    lineas = ["", "## 3. AUC por paciente agregando solo los k cortes centrales", "",
              "Media de las probabilidades por corte (experimentos/oof_cortes/) de los k cortes con slice_index más "
              "cercano a la mediana de los de la paciente (muestreo.cortes_centrales). IC95 por bootstrap por "
              "paciente, los mismos remuestreos para todas las k.", "",
              "| configuración | " + " | ".join("todos" if k is None else f"k = {k}" for k in KS) + " |",
              "|---|" + "---|" * len(KS)]
    faltan = []
    muestras = remuestreos(len(t))
    for c in configs:
        ruta = EXP / "oof_cortes" / f"{c}.csv"
        if not ruta.is_file():
            faltan.append(c)
            continue
        cortes = pd.read_csv(ruta)
        celdas = []
        for k in KS:
            sel = cortes if k is None else muestreo.cortes_centrales(cortes, k)
            p = sel.groupby("patient_id").prob.mean().reindex(t.index)
            if p.isna().any():
                celdas.append("faltan pacientes")
                continue
            a, b = auc_con_ic(t.pCR.to_numpy(), p.to_numpy(), muestras)
            celdas.append(f"{a:.3f} ({ic(b)})")
        lineas.append(f"| {c} | " + " | ".join(celdas) + " |")
    if faltan:
        lineas += ["", "Sin experimentos/oof_cortes/ de: " + ", ".join(faltan) + ". Ejecuta exportar_oof.py en "
                   "la máquina con los pesos (experimentos/oof*/ no se versiona: lleva ids de pacientes)."]
    return lineas


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t = pacientes_train()
    configs = NUEVAS + REFERENCIAS
    lineas = ["# Último bloque de experimentos de imagen: la receta regularizada", "",
              "Receta (train.py): 12 épocas fijas sin elegir época (--criterio ultima), lr 1e-3 con bajada lineal "
              "a 0 (--lr-lineal), weight decay 1e-2, dropout 0,35 y 3 cortes al azar por paciente en cada época. "
              "Variantes: imagen completa, ventana centrada de 160 px sin reescalar, recorte por la caja y solo los "
              "4 cortes centrales. Validación cruzada de 5 folds por paciente, solo train; sin test.", ""]
    l1, _ = seccion_auc(t, configs)
    l2, curvas = seccion_sobreajuste(configs)
    l3 = seccion_centrales(t, configs)
    lineas += l1 + l2 + l3
    if curvas:
        figura(curvas, SALIDA_PNG)
        lineas += ["", f"Figura: {SALIDA_PNG.name}."]
    SALIDA_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    print(f"\nEscrito {SALIDA_MD.relative_to(RAIZ).as_posix()}" + (f" y {SALIDA_PNG.relative_to(RAIZ).as_posix()}"
                                                                    if curvas else ""))


if __name__ == "__main__":
    main()
