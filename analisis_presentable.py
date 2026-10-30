"""¿Es presentable alguna variante? Análisis de la validación cruzada de 5 folds.

Para cada variante (prefijo de nombre, p. ej. cv_recortecaja, con ejecuciones
<variante>_f0 ... _f4; si un nombre se repite, la más reciente):

1. Curvas: época de inicio del sobreajuste en cada fold, la primera época e en
   que la pérdida de validación supera en más de UMBRAL_SOBREAJUSTE su mínimo
   hasta esa época, y también en la e+1 (dos seguidas). Si no ocurre antes de
   la última época entrenada, el inicio es "no observado" y cuenta como
   (épocas entrenadas + 1), una cota inferior. Bajada de la pérdida de
   entrenamiento: la de la época 1 menos la de la época de inicio.
2. ROC: predicciones por paciente de la mejor época de cada fold
   (outputs/runs/.../predicciones_val.csv), juntas (cada paciente una vez).
   AUC con IC95 por bootstrap por paciente, el de "solo cohorte" (tasa de pCR
   de la cohorte en la partición de entrenamiento de su fold) y la diferencia,
   todos con los mismos remuestreos.
3. Confusión: el umbral (sobre las mismas predicciones juntas) con mayor
   sensibilidad entre los que dan especificidad >= ESPECIFICIDAD_OBJETIVO.
   Sensibilidad y precisión con IC95 de Wilson, frente al azar con la misma
   especificidad (sensibilidad 1 - especificidad, precisión = prevalencia).
4. Tabla de criterios. No elige ganador.

El umbral se elige con la validación interna (nunca con test).

Uso, desde la raíz:

    python analisis_presentable.py
    python analisis_presentable.py --variantes cv_recortecaja_lr3e-4 cv_baseline_lr3e-4 \\
        --salida experimentos/analisis_lr.md --comparar cv_recortecaja_lr3e-4=cv_recortecaja

--comparar nueva=referencia guarda además una figura con las curvas de
pérdida y AUC por época de los 5 folds de las dos variantes, con las escalas
fijas de informe.py.

Solo escribe resultados agregados (nunca predicciones por paciente).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import informe
from informe import RAIZ_REPO, uc

RAIZ_DATOS = RAIZ_REPO / "breastdcedl"
DIR_EXPERIMENTOS = RAIZ_REPO / "experimentos"
DIR_RUNS = RAIZ_REPO / "outputs" / "runs"
VARIANTES = ("cv_baseline_suav", "cv_aumento", "cv_5bloques", "cv_realce", "cv_recortecaja")
DATASET_A_COHORTE = {"spy1": "I-SPY1", "spy2": "I-SPY2", "duke": "Duke"}

UMBRAL_SOBREAJUSTE = 0.03
EPOCA_MINIMA_SOBREAJUSTE = 11
ESPECIFICIDAD_OBJETIVO = 0.85
N_BOOTSTRAP = 2000
SEMILLA = 42


def ejecuciones(variante: str) -> dict[int, str]:
    """fold -> nombre de carpeta (la más reciente si el nombre se repite)."""
    patron = re.compile(rf"^\d{{8}}_\d{{6}}_{re.escape(variante)}_f([0-4])$")
    encontradas = {}
    for d in sorted(p.name for p in DIR_EXPERIMENTOS.iterdir() if p.is_dir()):
        m = patron.match(d)
        if m:
            if int(m.group(1)) in encontradas:
                print(f"AVISO: {variante}_f{m.group(1)} repetida; se usa {d}")
            encontradas[int(m.group(1))] = d
    if sorted(encontradas) != list(range(5)):
        sys.exit(f"ERROR: {variante} no tiene los 5 folds (tiene {sorted(encontradas)})")
    return encontradas


def inicio_sobreajuste(perdida_val: np.ndarray) -> int | None:
    """Primera época (desde 1) cuya pérdida de validación, y la de la siguiente,
    superan en más de UMBRAL_SOBREAJUSTE el mínimo hasta cada una."""
    minimo = np.minimum.accumulate(perdida_val)
    por_encima = perdida_val > minimo + UMBRAL_SOBREAJUSTE
    for i in range(len(perdida_val) - 1):
        if por_encima[i] and por_encima[i + 1]:
            return i + 1
    return None


def curvas(variante: str, carpetas: dict[int, str]) -> pd.DataFrame:
    filas = []
    for fold, d in carpetas.items():
        m = pd.read_csv(DIR_EXPERIMENTOS / d / "metricas_epoca.csv")
        inicio = inicio_sobreajuste(m.perdida_val.to_numpy())
        efectivo = inicio if inicio is not None else len(m) + 1
        hasta = min(efectivo, len(m))
        filas.append({"variante": variante, "fold": fold, "epocas": len(m), "observado": inicio is not None,
                      "inicio": efectivo,
                      "bajada_entr": m.perdida_entr.iloc[0] - m.perdida_entr.iloc[hasta - 1]})
    return pd.DataFrame(filas)


def predicciones(carpetas: dict[int, str], samples: pd.DataFrame, tasa_por_fold: dict) -> pd.DataFrame:
    """Una fila por paciente de validación: fold, cohorte, pCR, prob y prob_cohorte."""
    partes = []
    for fold, d in carpetas.items():
        p = (pd.read_csv(DIR_RUNS / d / "predicciones_val.csv")
             .groupby("patient_id").agg(cohorte=("cohorte", "first"), pCR=("pCR", "first"),
                                        prob=("prob_paciente", "first")).reset_index())
        p["fold"] = fold
        p["prob_cohorte"] = p.cohorte.map(tasa_por_fold[fold])
        partes.append(p)
    pac = pd.concat(partes, ignore_index=True)
    if pac.patient_id.duplicated().any():
        sys.exit("ERROR: una paciente aparece en más de un fold")
    return pac


def ic_bootstrap(y, p_modelo, p_cohorte, rng) -> dict:
    n = len(y)
    auc_m, auc_c = roc_auc_score(y, p_modelo), roc_auc_score(y, p_cohorte)
    bm, bc = [], []
    while len(bm) < N_BOOTSTRAP:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        bm.append(roc_auc_score(y[i], p_modelo[i]))
        bc.append(roc_auc_score(y[i], p_cohorte[i]))
    bm, bc = np.array(bm), np.array(bc)
    q = lambda v: tuple(np.percentile(v, [2.5, 97.5]))  # noqa: E731
    return {"auc": auc_m, "auc_ic": q(bm), "auc_cohorte": auc_c, "auc_cohorte_ic": q(bc),
            "dif": auc_m - auc_c, "dif_ic": q(bm - bc)}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(0.975)
    p = k / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    radio = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centro - radio, centro + radio)


def confusion(y: np.ndarray, p: np.ndarray) -> dict:
    """Umbral con mayor sensibilidad entre los de especificidad >= objetivo
    (positivo si prob >= umbral)."""
    mejor = None
    for u in np.unique(p):
        pred = p >= u
        esp = np.mean(~pred[y == 0])
        if esp >= ESPECIFICIDAD_OBJETIVO:
            sens = np.mean(pred[y == 1])
            if mejor is None or sens > mejor[1]:
                mejor = (u, sens)
    u = mejor[0]
    pred = p >= u
    vp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    fp, vn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
    esp = vn / (vn + fp)
    return {"umbral": float(u), "vp": vp, "fn": fn, "fp": fp, "vn": vn, "especificidad": esp,
            "sensibilidad": vp / (vp + fn), "sens_ic": wilson(vp, vp + fn),
            "precision": vp / (vp + fp) if vp + fp else np.nan, "prec_ic": wilson(vp, vp + fp),
            "sens_azar": 1 - esp, "prec_azar": float(np.mean(y))}


def figura_comparada(nueva: str, ref: str, carpetas: dict[str, dict[int, str]], ruta: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(2, 5, figsize=(22, 8), layout="constrained", sharey="row")
    epocas_max = 1
    for fold in range(5):
        for variante, color in ((ref, "tab:gray"), (nueva, "tab:blue")):
            m = pd.read_csv(DIR_EXPERIMENTOS / carpetas[variante][fold] / "metricas_epoca.csv")
            epocas_max = max(epocas_max, int(m.epoca.max()))
            ejes[0, fold].plot(m.epoca, m.perdida_entr, color=color, ls="--", label=f"{variante} entr.")
            ejes[0, fold].plot(m.epoca, m.perdida_val, color=color, label=f"{variante} val.")
            ejes[1, fold].plot(m.epoca, m.auc_val, color=color, label=variante)
        ejes[0, fold].set_title(f"fold {fold}")
    for eje in ejes[0]:
        eje.set_ylim(*informe.ESCALA_PERDIDA)
    for eje in ejes[1]:
        eje.set_ylim(*informe.ESCALA_AUC_EPOCA)
        eje.axhline(0.5, color="k", lw=0.5)
        eje.set_xlabel("época")
    for eje in ejes.ravel():
        eje.set_xlim(informe.ESCALA_EPOCAS[0], max(informe.ESCALA_EPOCAS[1], epocas_max))
        eje.grid(alpha=0.3)
    ejes[0, 0].set_ylabel("pérdida (BCE)")
    ejes[1, 0].set_ylabel("AUC de validación por paciente")
    ejes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{nueva} (azul) frente a {ref} (gris)")
    fig.savefig(ruta, dpi=80)
    plt.close(fig)


def fmt_ic(v, ic) -> str:
    return f"{v:.3f} ({ic[0]:.3f}-{ic[1]:.3f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variantes", nargs="+", default=list(VARIANTES))
    ap.add_argument("--salida", default="experimentos/analisis_presentable.md")
    ap.add_argument("--comparar", nargs="*", default=[], metavar="NUEVA=REFERENCIA")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    samples = uc.cargar_samples(RAIZ_DATOS)
    cohorte = uc.cargar_patients(RAIZ_DATOS).set_index("pid").dataset.map(DATASET_A_COHORTE)
    samples["cohorte"] = samples.patient_id.map(cohorte)
    tasa_por_fold = {f: uc.particion(samples, fold_val=f)[0].drop_duplicates("patient_id")
                     .groupby("cohorte").pCR.mean() for f in range(5)}

    carpetas = {v: ejecuciones(v) for v in args.variantes}
    for par in args.comparar:
        for v in par.split("="):
            carpetas.setdefault(v, ejecuciones(v))

    tabla_curvas, resumen, detalle_conf = [], [], []
    for v in args.variantes:
        c = curvas(v, carpetas[v])
        tabla_curvas.append(c)
        pac = predicciones(carpetas[v], samples, tasa_por_fold)
        y, p, pc = pac.pCR.to_numpy(), pac.prob.to_numpy(), pac.prob_cohorte.to_numpy()
        roc = ic_bootstrap(y, p, pc, np.random.default_rng(SEMILLA))
        conf = confusion(y, p)
        detalle_conf.append((v, conf))
        no_obs = int((~c.observado).sum())
        resumen.append({
            "variante": v, "pacientes": len(pac),
            "inicio_sobreajuste_medio": ("≥ " if no_obs else "") + f"{c.inicio.mean():.1f}",
            "bajada_entr_media": round(c.bajada_entr.mean(), 3),
            "AUC (IC95)": fmt_ic(roc["auc"], roc["auc_ic"]),
            "solo cohorte (IC95)": fmt_ic(roc["auc_cohorte"], roc["auc_cohorte_ic"]),
            "diferencia (IC95)": fmt_ic(roc["dif"], roc["dif_ic"]),
            "sens (IC95) / azar": f"{fmt_ic(conf['sensibilidad'], conf['sens_ic'])} / {conf['sens_azar']:.3f}",
            "prec (IC95) / azar": f"{fmt_ic(conf['precision'], conf['prec_ic'])} / {conf['prec_azar']:.3f}",
            "C1 sobreajuste ≥ 11": "sí" if c.inicio.mean() >= EPOCA_MINIMA_SOBREAJUSTE else "no",
            "C2 IC dif. > 0": "sí" if roc["dif_ic"][0] > 0 else "no",
            "C3 sens > azar": "sí" if conf["sens_ic"][0] > conf["sens_azar"] else "no",
        })
    tabla_curvas = pd.concat(tabla_curvas)
    resumen = pd.DataFrame(resumen)

    figuras = []
    for par in args.comparar:
        nueva, ref = par.split("=")
        ruta = DIR_EXPERIMENTOS / f"curvas_{nueva}_vs_{ref}.png"
        figura_comparada(nueva, ref, carpetas, ruta)
        figuras.append(ruta.name)

    criterios = ["variante", "C1 sobreajuste ≥ 11", "C2 IC dif. > 0", "C3 sens > azar"]
    texto = [
        "# ¿Hay alguna variante presentable? (validación cruzada de 5 folds)",
        "",
        f"Generado por `python {' '.join(sys.argv)}`. Solo validación interna (train); test no se usa.",
        "",
        "## Criterios",
        "",
        f"- C1: inicio del sobreajuste en la época {EPOCA_MINIMA_SOBREAJUSTE} o después, de media "
        f"(pérdida de validación > su mínimo hasta entonces + {UMBRAL_SOBREAJUSTE} dos épocas seguidas).",
        "- C2: el IC95 de la diferencia de AUC con \"solo cohorte\" queda por encima de 0.",
        "- C3: el límite inferior del IC95 de la sensibilidad supera la del azar con la misma especificidad.",
        "",
        informe.tabla_md(resumen[criterios].set_index("variante")),
        "",
        "## 1. Curvas: inicio del sobreajuste por fold",
        "",
        "Época de inicio; \"≥ n\" = no observado en las n-1 épocas entrenadas (el early stopping corta las curvas, "
        "así que es una cota inferior). Bajada = pérdida de entrenamiento de la época 1 menos la de la época de "
        "inicio (o la última).",
        "",
        informe.tabla_md(tabla_curvas.assign(inicio=lambda t: np.where(t.observado, t.inicio.astype(str),
                                                                      "≥ " + t.inicio.astype(str)))
                         .drop(columns="observado").set_index(["variante", "fold"]), 3),
        "",
        "## 2 y 3. Predicciones de los 5 folds juntas (mejor época de cada fold)",
        "",
        f"AUC por paciente, IC95 por bootstrap ({N_BOOTSTRAP} remuestreos por paciente, mismos remuestreos para el "
        "modelo y \"solo cohorte\"). Umbral: el de mayor sensibilidad con especificidad ≥ "
        f"{ESPECIFICIDAD_OBJETIVO}, elegido sobre estas mismas predicciones (optimista). Sensibilidad y "
        "precisión con IC95 de Wilson; azar con la misma especificidad: sensibilidad 1 - especificidad, "
        "precisión = prevalencia.",
        "",
        informe.tabla_md(resumen.drop(columns=criterios[1:]).set_index("variante")),
        "",
        "Matrices de confusión (filas real, columnas predicho):",
        "",
    ]
    for v, c in detalle_conf:
        texto += [f"- {v}: umbral {c['umbral']:.3f}, especificidad {c['especificidad']:.3f}. "
                  f"pCR=1: {c['vp']} VP / {c['fn']} FN; pCR=0: {c['fp']} FP / {c['vn']} VN."]
    if figuras:
        texto += ["", "## Curvas comparadas", ""] + [f"![{f}]({f})" for f in figuras]
    ruta = RAIZ_REPO / args.salida
    ruta.write_text("\n".join(texto) + "\n", encoding="utf-8")
    print(informe.tabla_md(resumen.set_index("variante")))
    print(f"Escrito {ruta.relative_to(RAIZ_REPO).as_posix()}" + (f"; figuras {figuras}" if figuras else ""))


if __name__ == "__main__":
    main()
