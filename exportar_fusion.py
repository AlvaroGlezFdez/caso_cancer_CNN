"""Modelo final combinado POR CORTE (variables clínicas + CNN): modelo_app/fusion.json.

Uso, desde la raíz del repositorio:

    python exportar_fusion.py recortecaja_lr3e-4
    python exportar_fusion.py baseline --provisional      (solo para desarrollar)

Una fila por corte de train: las variables clínicas de su paciente
(clinica.construir_variables: subtipo, log(1 + volumen), edad) y el logit de la
probabilidad out-of-fold de ese corte con la CNN
(experimentos/oof_cortes/<configuracion>.csv, de exportar_oof.py). Regresión
logística (fusión tardía), como fusion_clinica.py, pero por corte: la app
analiza un corte suelto.

Validación cruzada con los mismos 5 folds por paciente que la CNN (columna fold
de samples.csv; el escalador y la regresión se ajustan solo con los cortes de
entrenamiento de cada fold). Cuatro modelos: solo cohorte, clínico, solo CNN y
combinado. AUC por corte y por paciente (media de las probabilidades de sus
cortes) con IC95 por bootstrap por paciente (cada remuestreo toma pacientes con
todos sus cortes; los mismos remuestreos para todos los modelos, así que las
diferencias son pareadas).

Umbral: el de mayor sensibilidad entre los que dan especificidad >= 0,85 sobre
la probabilidad combinada out-of-fold de cada corte
(analisis_presentable.confusion, positivo si prob >= umbral). Optimista: se
elige con las mismas predicciones con las que se mide. Con el mismo criterio,
un umbral por paciente (sobre la media de las probabilidades out-of-fold de sus
cortes) y los umbrales de los otros tres modelos, por corte y por paciente
(umbrales): los usa, congelados, la evaluación única en test (evaluar_test.py).

Modelo final: la misma regresión ajustada con todos los cortes de train. Se
guarda en modelo_app/fusion.json con la configuración de la CNN, las variables,
la media y la desviación del escalador, los coeficientes, el umbral y las
métricas agregadas de la validación cruzada (AUC con IC95, AP, Brier y, con el
umbral de cada modelo, la matriz de confusión). También los cuatro modelos
ajustados con todo train (modelos_finales: solo cohorte, clínico, solo CNN y
combinado), para evaluarlos en test sin reajustar nada. Ni etiquetas ni ids: se
comprueba antes de terminar. Nunca usa test.

--provisional: sin oof_cortes (en el portátil no están las predicciones por
corte), usa en cada corte la probabilidad por paciente de
experimentos/oof/<configuracion>.csv. Solo sirve para desarrollar la app:
fusion.json queda marcado como provisional y la app lo avisa.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from analisis_presentable import ESPECIFICIDAD_OBJETIVO, N_BOOTSTRAP, SEMILLA, confusion
from clinica import COLUMNAS_ENTRADA, SUBTIPOS, Escalador, construir_variables, logit

RAIZ = Path(__file__).resolve().parent
META = RAIZ / "breastdcedl" / "metadata"
OOF = RAIZ / "experimentos" / "oof"
OOF_CORTES = RAIZ / "experimentos" / "oof_cortes"
DIR_PAQUETE = RAIZ / "modelo_app"
VERSION_FUSION = 2
MODELOS = ("cohorte", "clinico", "cnn", "combinado")
NOMBRES = {"cohorte": "solo cohorte", "clinico": "clínico", "cnn": "solo CNN", "combinado": "combinado"}
DIFERENCIAS = (("clinico", "cohorte"), ("combinado", "clinico"), ("combinado", "cohorte"), ("combinado", "cnn"))
COHORTES = {"spy1": "I-SPY1", "spy2": "I-SPY2", "duke": "Duke"}
CONSTRUCCION = {
    "cohorte": "indicadores 0/1 de la columna dataset de patients.csv (spy1, spy2; Duke = los dos a 0)",
    "clinico": "clinica.construir_variables",
    "cnn": "logit_cnn = clinica.logit(probabilidad de la CNN)",
    "combinado": "clinica.construir_variables + logit_cnn = clinica.logit(probabilidad de la CNN)",
}
MAX_PUNTOS_CURVA = 400
PATRON_IDS = re.compile(r"ISPY1_|ISPY2-|ACRIN-6698|Breast_MRI_|patient_id|sample_id", re.IGNORECASE)


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def commit_actual() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cargar_filas(config: str, provisional: bool = False) -> tuple[pd.DataFrame, Path]:
    """Una fila por corte de train con fold, slice_index, prob (CNN out-of-fold), pCR,
    cohorte (dataset) y las columnas de patients.csv que usa clinica.py."""
    pac = pd.read_csv(META / "patients.csv").set_index("pid")
    mue = pd.read_csv(META / "samples.csv")
    cortes = mue.loc[mue.split == "train", ["sample_id", "patient_id", "fold", "slice_index"]]
    if (cortes.fold < 0).any():
        sys.exit("ERROR: hay cortes de train sin fold en samples.csv")
    ruta = (OOF if provisional else OOF_CORTES) / f"{config}.csv"
    if not ruta.is_file():
        sys.exit(f"ERROR: falta {ruta.relative_to(RAIZ).as_posix()}: ejecuta exportar_oof.py en la máquina con los "
                 "pesos (outputs/runs/). experimentos/oof*/ no se versiona: lleva ids de pacientes.")
    o = pd.read_csv(ruta).rename(columns={"fold": "fold_oof"})
    clave = "patient_id" if provisional else "sample_id"
    if o[clave].duplicated().any():
        sys.exit(f"ERROR: {ruta.name} repite {clave}")
    if not o[clave].isin(cortes[clave]).all():
        sys.exit(f"ERROR: {ruta.name} tiene filas que no son de train")
    filas = cortes.merge(o[[clave, "fold_oof", "prob"]], on=clave, how="left")
    if filas.prob.isna().any():
        sys.exit(f"ERROR: a {ruta.name} le faltan {int(filas.prob.isna().sum())} cortes de train")
    if not (filas.fold_oof == filas.fold).all():
        sys.exit(f"ERROR: los folds de {ruta.name} no coinciden con samples.csv")
    filas = filas.drop(columns="fold_oof").join(pac[["pCR", "dataset", *COLUMNAS_ENTRADA]], on="patient_id")
    return filas.reset_index(drop=True), ruta


def matriz(filas: pd.DataFrame, modelo: str) -> pd.DataFrame:
    if modelo == "cohorte":
        return pd.DataFrame({c: (filas.dataset == c).astype(float) for c in ["spy1", "spy2"]}, index=filas.index)
    X = construir_variables(filas) if modelo in ("clinico", "combinado") else pd.DataFrame(index=filas.index)
    if modelo in ("cnn", "combinado"):
        X["logit_cnn"] = logit(filas.prob)
    return X


def ajustar(filas: pd.DataFrame, X: pd.DataFrame) -> tuple[Escalador, LogisticRegression]:
    esc = Escalador().ajustar(X)
    return esc, LogisticRegression(max_iter=1000).fit(esc.transformar(X), filas.pCR)


def validacion_cruzada(filas: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    oof = np.full(len(filas), np.nan)
    for f in range(5):
        tr, va = (filas.fold != f).to_numpy(), (filas.fold == f).to_numpy()
        esc, lr = ajustar(filas[tr], X[tr])
        oof[va] = lr.predict_proba(esc.transformar(X[va]))[:, 1]
    return oof


def bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], paciente: np.ndarray,
              umbral: float | None = None) -> dict:
    """Remuestreos por paciente (con todos sus cortes); los mismos para todos los modelos.
    Con `umbral`, también sensibilidad y precisión del combinado."""
    rng = np.random.default_rng(SEMILLA)
    grupos = list(pd.Series(np.arange(len(y))).groupby(paciente).indices.values())
    auc = {k: [] for k in preds}
    sens, prec = [], []
    while len(auc[next(iter(preds))]) < N_BOOTSTRAP:
        i = np.concatenate([grupos[k] for k in rng.integers(0, len(grupos), len(grupos))])
        yy = y[i]
        if yy.min() == yy.max():
            continue
        for k, p in preds.items():
            auc[k].append(roc_auc_score(yy, p[i]))
        if umbral is not None:
            pred = preds["combinado"][i] >= umbral
            sens.append(pred[yy == 1].mean())
            if pred.any():
                prec.append(yy[pred].mean())
    return {"auc": {k: np.array(v) for k, v in auc.items()}, "sens": np.array(sens), "prec": np.array(prec)}


def ic(v: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def curva_roc(y: np.ndarray, p: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    if len(fpr) > MAX_PUNTOS_CURVA:                      # menos puntos para la app; conserva los extremos
        k = np.unique(np.linspace(0, len(fpr) - 1, MAX_PUNTOS_CURVA).round().astype(int))
        fpr, tpr = fpr[k], tpr[k]
    return {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]}


def en_umbral(y: np.ndarray, p: np.ndarray, umbral: float) -> dict:
    """Matriz de confusión y métricas con un umbral dado (positivo si prob >= umbral)."""
    pred = p >= umbral
    vp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    fp, vn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
    return {"umbral": float(umbral), "matriz_confusion": {"VP": vp, "FN": fn, "FP": fp, "VN": vn},
            "sensibilidad": vp / (vp + fn) if vp + fn else None, "especificidad": vn / (vn + fp) if vn + fp else None,
            "precision": vp / (vp + fp) if vp + fp else None, "vpn": vn / (vn + fn) if vn + fn else None}


def metricas_nivel(y: np.ndarray, preds: dict[str, np.ndarray], paciente: np.ndarray,
                   umbrales: dict[str, float] | None = None) -> dict:
    """AUC con IC95 (bootstrap por paciente, pareado), AP, Brier y curvas ROC de cada
    modelo; con `umbrales` (modelo -> umbral elegido con estas mismas predicciones),
    la matriz de confusión de cada uno y, para el combinado, IC95 de sensibilidad y precisión."""
    umbral = None if umbrales is None else umbrales["combinado"]
    b = bootstrap(y, preds, paciente, umbral)
    auc = {k: float(roc_auc_score(y, p)) for k, p in preds.items()}
    salida = {
        "n": int(len(y)), "pCR_1": int(y.sum()), "tasa_pCR": float(y.mean()),
        "modelos": {k: {"nombre": NOMBRES[k], "auc": auc[k], "auc_ic95": ic(b["auc"][k]),
                        "average_precision": float(average_precision_score(y, p)),
                        "brier": float(np.mean((p - y) ** 2)), "prob_media": float(np.mean(p))}
                    for k, p in preds.items()},
        "diferencias": {f"{a}_menos_{c}": {"valor": auc[a] - auc[c], "ic95": ic(b["auc"][a] - b["auc"][c])}
                        for a, c in DIFERENCIAS},
        "curvas_roc": {k: curva_roc(y, p) for k, p in preds.items()},
    }
    if umbrales is not None:
        for k, p in preds.items():
            assert confusion(y, p)["umbral"] == umbrales[k]
            salida["modelos"][k]["con_umbral"] = en_umbral(y, p, umbrales[k])
        e = confusion(y, preds["combinado"])
        assert e["umbral"] == umbral
        salida["con_umbral"] = {
            "umbral": umbral, "matriz_confusion": {"VP": e["vp"], "FN": e["fn"], "FP": e["fp"], "VN": e["vn"]},
            "sensibilidad": e["sensibilidad"], "especificidad": e["especificidad"], "precision": e["precision"],
            "sensibilidad_ic95": ic(b["sens"]), "precision_ic95": ic(b["prec"]),
            "sensibilidad_azar": e["sens_azar"], "precision_azar": e["prec_azar"]}
    return salida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configuracion", help="configuración de validación cruzada de la CNN (p. ej. recortecaja_lr3e-4)")
    ap.add_argument("--provisional", action="store_true",
                    help="usa la probabilidad por paciente (experimentos/oof/) en cada corte; solo para desarrollar")
    ap.add_argument("--salida", type=Path, default=DIR_PAQUETE)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    filas, ruta_oof = cargar_filas(args.configuracion, args.provisional)
    y = filas.pCR.to_numpy().astype(int)
    paciente = filas.patient_id.to_numpy()
    print(f"{len(filas)} cortes de {filas.patient_id.nunique()} pacientes de train; CNN {args.configuracion}"
          + (" (PROVISIONAL: probabilidad por paciente en cada corte)" if args.provisional else ""))

    # Validación cruzada por corte, los cuatro modelos.
    X = {m: matriz(filas, m) for m in MODELOS}
    oof = {m: validacion_cruzada(filas, X[m]) for m in MODELOS}
    umbrales_corte = {m: confusion(y, oof[m])["umbral"] for m in MODELOS}
    umbral = umbrales_corte["combinado"]
    por_corte = metricas_nivel(y, oof, paciente, umbrales_corte)

    # Por paciente: la media de las probabilidades de sus cortes, con su propio umbral.
    medias = pd.DataFrame({m: oof[m] for m in MODELOS}).groupby(paciente).mean()
    y_pac = filas.groupby("patient_id").pCR.first().reindex(medias.index).to_numpy().astype(int)
    umbrales_paciente = {m: confusion(y_pac, medias[m].to_numpy())["umbral"] for m in MODELOS}
    por_paciente = metricas_nivel(y_pac, {m: medias[m].to_numpy() for m in MODELOS}, medias.index.to_numpy(),
                                  umbrales_paciente)
    cohorte_pac = filas.groupby("patient_id").dataset.first().reindex(medias.index).map(COHORTES).to_numpy()
    por_cohorte_pac = {}
    for c in COHORTES.values():
        s = cohorte_pac == c
        if s.any() and 0 < y_pac[s].mean() < 1:
            por_cohorte_pac[c] = {"pacientes": int(s.sum()), "pCR_1": int(y_pac[s].sum()),
                                  **{f"auc_{m}": float(roc_auc_score(y_pac[s], medias[m].to_numpy()[s]))
                                     for m in MODELOS}}

    # Por cohorte (solo el combinado y el clínico, por corte).
    cohorte = filas.dataset.map(COHORTES).to_numpy()
    por_cohorte = {}
    for c in COHORTES.values():
        s = cohorte == c
        if s.any() and 0 < y[s].mean() < 1:
            por_cohorte[c] = {"cortes": int(s.sum()), "pacientes": int(len(np.unique(paciente[s]))),
                              "auc_combinado": float(roc_auc_score(y[s], oof["combinado"][s])),
                              "auc_clinico": float(roc_auc_score(y[s], oof["clinico"][s]))}

    # Modelo final: todos los cortes de train.
    finales = {m: ajustar(filas, X[m]) for m in MODELOS}
    esc, lr = finales["combinado"]
    variables = list(X["combinado"].columns)
    esc_clin, lr_clin = finales["clinico"]

    referencia_pac = pd.read_csv(META / "patients.csv")
    referencia_pac = referencia_pac[referencia_pac.split == "train"]
    fusion = {
        "version_fusion": VERSION_FUSION,
        "exportado": datetime.now().isoformat(timespec="seconds"),
        "exportado_con": {"script": "exportar_fusion.py", "commit": commit_actual()},
        "provisional": bool(args.provisional),
        "nota_provisional": ("Sin probabilidades por corte: en cada corte se usó la probabilidad por paciente (media "
                             "de sus cortes) de la CNN. Solo para desarrollar la app; hay que reexportar con "
                             "oof_cortes.") if args.provisional else None,
        "cnn": {
            "configuracion": args.configuracion,
            "ejecuciones": f"cv_{args.configuracion}_f0 … f4 (probabilidades out-of-fold de la mejor época de cada fold)",
            "fuente": ruta_oof.relative_to(RAIZ).as_posix(), "sha256_fuente": sha256(ruta_oof),
            "nivel": "paciente (provisional)" if args.provisional else "corte",
            "paquete_esperado": f"una ejecución cv_{args.configuracion}_fN exportada con exportar_modelo.py",
        },
        "modelo": {
            "tipo": "regresión logística (sklearn LogisticRegression, L2, C = 1, lbfgs), fusión tardía",
            "unidad": "corte: una fila por corte de train, con las variables clínicas de su paciente",
            "entradas": COLUMNAS_ENTRADA, "subtipos": SUBTIPOS,
            "variables": variables,
            "construccion": "clinica.construir_variables + logit_cnn = clinica.logit(probabilidad de la CNN)",
            "escalador": {**esc.a_dict(),
                          "nota": "faltantes -> media de entrenamiento; z = (x - media) / desv (clinica.Escalador)"},
            "coeficientes": dict(zip(variables, map(float, lr.coef_[0]))),
            "intercepto": float(lr.intercept_[0]),
            "probabilidad": "sigmoide(intercepto + suma(coeficiente * z))",
            "cortes_entrenamiento": int(len(filas)), "pacientes_entrenamiento": int(filas.patient_id.nunique()),
        },
        "modelo_clinico": {
            "variables": list(X["clinico"].columns), "escalador": esc_clin.a_dict(),
            "coeficientes": dict(zip(X["clinico"].columns, map(float, lr_clin.coef_[0]))),
            "intercepto": float(lr_clin.intercept_[0]),
        },
        "umbral": {
            "valor": umbral,
            "regla": "el de mayor sensibilidad entre los que dan especificidad ≥ "
                     + f"{ESPECIFICIDAD_OBJETIVO:.2f}".replace(".", ",")
                     + " sobre la probabilidad combinada out-of-fold de cada corte de train; positivo si prob ≥ umbral",
            "especificidad_objetivo": ESPECIFICIDAD_OBJETIVO, "nivel": "corte",
            "nota": "elegido y medido con las mismas predicciones: sensibilidad y especificidad son optimistas",
            "valor_paciente": umbrales_paciente["combinado"],
            "regla_paciente": "el mismo criterio sobre la media de las probabilidades combinadas out-of-fold de los "
                              "cortes de cada paciente de train",
        },
        "umbrales": {
            "regla": "por modelo y nivel, el de mayor sensibilidad entre los que dan especificidad ≥ "
                     + f"{ESPECIFICIDAD_OBJETIVO:.2f}".replace(".", ",")
                     + " sobre sus probabilidades out-of-fold de train (por corte, o la media de los cortes de cada "
                       "paciente); positivo si prob ≥ umbral. Congelados para la evaluación en test.",
            "corte": umbrales_corte, "paciente": umbrales_paciente,
        },
        "modelos_finales": {
            m: {"nombre": NOMBRES[m], "construccion": CONSTRUCCION[m], "variables": list(X[m].columns),
                "escalador": finales[m][0].a_dict(),
                "coeficientes": dict(zip(X[m].columns, map(float, finales[m][1].coef_[0]))),
                "intercepto": float(finales[m][1].intercept_[0])}
            for m in MODELOS},
        "rango_entrenamiento": {
            c: {"p01": float(referencia_pac[c].quantile(0.01)), "p99": float(referencia_pac[c].quantile(0.99))}
            for c in ("tum_vol", "age")},
        "validacion_cruzada": {
            "folds": 5, "particion": "los mismos 5 folds por paciente que la CNN (samples.csv)",
            "ic95_metodo": f"bootstrap por paciente (cada remuestreo toma pacientes con todos sus cortes), {N_BOOTSTRAP} "
                           f"remuestreos, semilla {SEMILLA}, los mismos para todos los modelos (diferencias pareadas)",
            "por_corte": por_corte, "por_paciente": por_paciente, "por_cohorte_corte": por_cohorte,
            "por_cohorte_paciente": por_cohorte_pac,
        },
    }
    args.salida.mkdir(parents=True, exist_ok=True)
    ruta = args.salida / "fusion.json"
    texto = json.dumps(fusion, indent=2, ensure_ascii=False)
    if PATRON_IDS.search(texto):
        sys.exit("ERROR: fusion.json contendría un id de paciente o de corte; no se guarda")
    ruta.write_text(texto + "\n", encoding="utf-8")

    print(f"Modelo combinado en {ruta.resolve().as_posix()}")
    for nivel, m in (("corte", por_corte), ("paciente", por_paciente)):
        partes = [f"{NOMBRES[k]} {v['auc']:.3f} ({v['auc_ic95'][0]:.3f}-{v['auc_ic95'][1]:.3f})"
                  for k, v in m["modelos"].items()]
        d = m["diferencias"]["combinado_menos_clinico"]
        print(f"  AUC por {nivel}: " + "; ".join(partes)
              + f"; combinado - clínico {d['valor']:+.3f} ({d['ic95'][0]:+.3f} a {d['ic95'][1]:+.3f})")
    for nivel, m in (("corte", por_corte), ("paciente", por_paciente)):
        c = m["con_umbral"]
        print(f"  umbral por {nivel} {c['umbral']:.4f}: sensibilidad {c['sensibilidad']:.3f}, especificidad "
              f"{c['especificidad']:.3f}, precisión {c['precision']:.3f} ({c['matriz_confusion']})")
    print("  umbrales congelados (corte / paciente): "
          + "; ".join(f"{NOMBRES[m]} {umbrales_corte[m]:.4f} / {umbrales_paciente[m]:.4f}" for m in MODELOS))
    print("  coeficientes (variables estandarizadas): "
          + ", ".join(f"{k} {v:+.3f}" for k, v in fusion["modelo"]["coeficientes"].items()))


if __name__ == "__main__":
    main()
