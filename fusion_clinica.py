"""Modelo clínico y modelo combinado (CNN + clínica) con validación cruzada.

    python fusion_clinica.py

Usa los mismos 5 folds por paciente que la CNN (columna fold de samples.csv),
solo pacientes de train. Nunca toca test.

Modelos, todos con regresión logística (fusión tardía):
- solo cohorte (referencia);
- clínico: subtipo + log(volumen) + edad (clinica.py);
- solo CNN: la probabilidad out-of-fold de la CNN;
- combinado: clínico + logit de la probabilidad de la CNN.

Para cada configuración de experimentos/oof/*.csv: AUC con IC95 (bootstrap por
paciente), diferencia combinado - clínico con IC95 pareado, coeficientes y, con
especificidad >= 0,85, umbral, sensibilidad y precisión.

Salida (solo agregados, sin etiquetas): experimentos/fusion/resumen_fusion.md,
resumen_fusion.csv y roc_fusion.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from clinica import Escalador, construir_variables, logit

RAIZ = Path(__file__).resolve().parent
META = RAIZ / "breastdcedl" / "metadata"
OOF = RAIZ / "experimentos" / "oof"
SALIDA = RAIZ / "experimentos" / "fusion"
N_BOOT = 2000
SEMILLA = 42
ESPECIFICIDAD_OBJETIVO = 0.85


def cargar_train() -> pd.DataFrame:
    pac = pd.read_csv(META / "patients.csv")
    mue = pd.read_csv(META / "samples.csv")
    fold = mue[mue.split == "train"].groupby("patient_id").fold.first()
    t = pac[pac.split == "train"].set_index("pid").join(fold, how="inner")
    if (t.fold < 0).any() or t.fold.isna().any():
        sys.exit("Hay pacientes de train sin fold.")
    return t


def matriz(t: pd.DataFrame, modelo: str, cnn: pd.Series | None) -> pd.DataFrame:
    if modelo == "cohorte":
        return pd.DataFrame({c: (t.dataset == c).astype(float) for c in ["spy1", "spy2"]}, index=t.index)
    X = pd.DataFrame(index=t.index)
    if modelo in ("clinico", "combinado"):
        X = construir_variables(t)
    if modelo in ("cnn", "combinado"):
        X["logit_cnn"] = logit(cnn.reindex(t.index))
    return X


def validacion_cruzada(t: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    oof = pd.Series(np.nan, index=t.index)
    for f in range(5):
        tr, va = t.fold != f, t.fold == f
        esc = Escalador().ajustar(X[tr])
        lr = LogisticRegression(max_iter=1000).fit(esc.transformar(X[tr]), t.pCR[tr])
        oof[va] = lr.predict_proba(esc.transformar(X[va]))[:, 1]
    return oof


def coeficientes(t: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    esc = Escalador().ajustar(X)
    lr = LogisticRegression(max_iter=1000).fit(esc.transformar(X), t.pCR)
    return pd.Series(lr.coef_[0], index=X.columns)


def bootstrap(y: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEMILLA)
    res = {k: [] for k in preds}
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        for k, p in preds.items():
            res[k].append(roc_auc_score(y[i], p[i]))
    return {k: np.array(v) for k, v in res.items()}


def umbral_especificidad(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    negativos = np.sort(p[y == 0])
    k = int(np.ceil(ESPECIFICIDAD_OBJETIVO * len(negativos))) - 1
    umbral = float(negativos[min(k, len(negativos) - 1)])
    pred = p > umbral
    vp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    return {"umbral": umbral,
            "especificidad": float(1 - fp / (y == 0).sum()),
            "sensibilidad": float(vp / (y == 1).sum()),
            "precision": float(vp / max(vp + fp, 1)),
            "VP": vp, "FP": fp, "FN": int((y == 1).sum() - vp), "VN": int((y == 0).sum() - fp)}


def fmt(a: float, b: np.ndarray) -> str:
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{a:.3f} ({lo:.3f}-{hi:.3f})"


def main() -> None:
    t = cargar_train()
    y = t.pCR.to_numpy()
    print(f"{len(t)} pacientes de train, tasa de pCR {y.mean():.3f}")

    base = {"cohorte": validacion_cruzada(t, matriz(t, "cohorte", None)),
            "clinico": validacion_cruzada(t, matriz(t, "clinico", None))}

    configs = sorted(OOF.glob("*.csv")) if OOF.is_dir() else []
    if not configs:
        print("Aviso: no hay experimentos/oof/*.csv (ejecuta exportar_oof.py en la máquina con los pesos; no se "
              "versionan porque llevan ids de pacientes). Solo se evalúan cohorte y clínico.")

    filas, lineas, rocs = [], [], {}
    b0 = bootstrap(y, {k: v.to_numpy() for k, v in base.items()})
    for k in base:
        filas.append({"configuracion": "-", "modelo": k, "auc": roc_auc_score(y, base[k]),
                      "ic_inf": np.percentile(b0[k], 2.5), "ic_sup": np.percentile(b0[k], 97.5),
                      **umbral_especificidad(y, base[k].to_numpy())})
    rocs["clínico"] = base["clinico"]; rocs["solo cohorte"] = base["cohorte"]

    lineas += ["# Modelo clínico y modelo combinado (CNN + clínica)", "",
               f"Validación cruzada de 5 folds por paciente, {len(t)} pacientes de train, "
               f"IC95 por bootstrap por paciente ({N_BOOT} remuestreos). Sin test.", "",
               "Variables clínicas: subtipo HR/HER2, log(1 + volumen tumoral), edad.", "",
               "## Referencias", "",
               f"- Solo cohorte: AUC {fmt(roc_auc_score(y, base['cohorte']), b0['cohorte'])}",
               f"- Clínico: AUC {fmt(roc_auc_score(y, base['clinico']), b0['clinico'])}", "",
               "## Combinado frente a clínico, por configuración de la CNN", "",
               "| CNN | solo CNN | combinado | combinado - clínico (IC95 pareado) | coef. CNN (estand.) |",
               "|---|---|---|---|---|"]

    for ruta in configs:
        nombre = ruta.stem
        o = pd.read_csv(ruta).set_index("patient_id")
        faltan = t.index.difference(o.index)
        if len(faltan):
            print(f"  {nombre}: faltan {len(faltan)} pacientes, se omite")
            continue
        if not (o.fold.reindex(t.index).to_numpy() == t.fold.to_numpy()).all():
            print(f"  {nombre}: los folds no coinciden con samples.csv, se omite")
            continue
        cnn = o.prob
        Xc, Xm = matriz(t, "cnn", cnn), matriz(t, "combinado", cnn)
        p_cnn, p_mix = validacion_cruzada(t, Xc), validacion_cruzada(t, Xm)
        b = bootstrap(y, {"cnn": p_cnn.to_numpy(), "mix": p_mix.to_numpy(),
                          "clin": base["clinico"].to_numpy()})
        a_cnn, a_mix = roc_auc_score(y, p_cnn), roc_auc_score(y, p_mix)
        dif = b["mix"] - b["clin"]
        a_dif = a_mix - roc_auc_score(y, base["clinico"])
        coef = coeficientes(t, Xm)["logit_cnn"]
        lo, hi = np.percentile(dif, [2.5, 97.5])
        lineas.append(f"| {nombre} | {fmt(a_cnn, b['cnn'])} | {fmt(a_mix, b['mix'])} | "
                      f"{a_dif:+.3f} ({lo:+.3f} a {hi:+.3f}) | {coef:+.3f} |")
        for modelo, p, bb in (("cnn", p_cnn, b["cnn"]), ("combinado", p_mix, b["mix"])):
            filas.append({"configuracion": nombre, "modelo": modelo, "auc": roc_auc_score(y, p),
                          "ic_inf": np.percentile(bb, 2.5), "ic_sup": np.percentile(bb, 97.5),
                          **umbral_especificidad(y, p.to_numpy())})
        rocs[f"combinado ({nombre})"] = p_mix
        print(f"  {nombre}: solo CNN {a_cnn:.3f}, combinado {a_mix:.3f}, dif. {a_dif:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    res = pd.DataFrame(filas)
    lineas += ["", f"## Con especificidad >= {ESPECIFICIDAD_OBJETIVO}", "",
               "| configuración | modelo | umbral | sensibilidad | precisión | VP | FN | FP | VN |",
               "|---|---|---|---|---|---|---|---|---|"]
    for r in res.itertuples():
        lineas.append(f"| {r.configuracion} | {r.modelo} | {r.umbral:.3f} | {r.sensibilidad:.3f} | "
                      f"{r.precision:.3f} | {r.VP} | {r.FN} | {r.FP} | {r.VN} |")
    coef_clin = coeficientes(t, matriz(t, "clinico", None))
    lineas += ["", "## Coeficientes del modelo clínico (variables estandarizadas, todo train)", ""]
    lineas += [f"- {k}: {v:+.3f}" for k, v in coef_clin.items()]
    lineas += ["", "Nota: el umbral se elige sobre las mismas predicciones que se evalúan, así que "
               "sensibilidad y precisión son algo optimistas. El combinado usa probabilidades "
               "out-of-fold de la CNN, de modo que ninguna paciente se predice con una red que la "
               "haya visto al entrenar."]

    SALIDA.mkdir(parents=True, exist_ok=True)
    res.to_csv(SALIDA / "resumen_fusion.csv", index=False)
    (SALIDA / "resumen_fusion.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 6))
    for nombre, p in rocs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.6, label=f"{nombre} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", label="azar")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 - especificidad", ylabel="sensibilidad",
           title=f"ROC por paciente, validación cruzada (n = {len(t)})")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(SALIDA / "roc_fusion.png", dpi=150)
    print(f"Resultados en {SALIDA}")


if __name__ == "__main__":
    main()
