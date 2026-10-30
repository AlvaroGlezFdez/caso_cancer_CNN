"""Crea el paquete del modelo que carga la app: modelo_app/modelo.pt y modelo.json.

Uso, desde la raíz del repositorio (en la máquina que tenga los pesos):

    python exportar_modelo.py baseline_30ep_cpu
    python exportar_modelo.py 20261022_050538_cv_baseline_f2

Acepta el nombre completo de la carpeta o solo el nombre de la ejecución (si se
repite, usa la más reciente). Lee:

    outputs/runs/<ejecución>/best.pt               pesos del mejor modelo
    outputs/runs/<ejecución>/predicciones_val.csv  predicciones de validación
    experimentos/<ejecución>/config.json           arquitectura, preprocesado,
                                                   hiperparámetros y entorno

y escribe en modelo_app/:

    modelo.pt    solo el state_dict de best.pt (los mismos tensores, bit a bit). No se
                 versiona: se sube a Hugging Face (AlvaroGlez/pcr-cnn-breastdcedl) y
                 la app lo descarga y lo verifica (app/nucleo/paquete.py)
    modelo.json  arquitectura, opciones de preprocesado, umbral, SHA-256 de
                 modelo.pt, ejecución, hiperparámetros y métricas de validación

Preprocesado: el de preprocesado.preprocesar, con la caja del tumor si la
ejecución usó --recorte-caja (preprocesado.recorte_caja) o --recorte-centrado N
(preprocesado.recorte_centrado = N, el lado de la ventana en px). Época: con
--criterio ultima, best.pt es la última época entrenada (épocas fijas, sin
elegir); modelo.json lo dice en modelo.seleccion (criterio y época usada).

Umbral: la app analiza un corte suelto, así que el umbral se elige POR CORTE:
el de mayor sensibilidad entre los que dan especificidad >= 0,85 sobre la
probabilidad de cada corte de validación de la propia ejecución
(analisis_presentable.confusion, positivo si prob >= umbral). Es optimista: se
elige con las mismas predicciones con las que se mide.

Métricas, SOLO AGREGADOS, por corte (las de la app; IC95 por bootstrap por
paciente, que remuestrea pacientes con todos sus cortes) y, como información
adicional, por paciente (media de sus cortes, con su propio umbral por el mismo
criterio): AUC con IC95 y el AUC "solo cohorte" (mismos remuestreos), puntos de
las curvas ROC (también la de "solo cohorte") y precisión-sensibilidad, curva de
calibración, histograma de probabilidades por clase (para mover el umbral en la
app), matriz de confusión, sensibilidad, especificidad y precisión con el
umbral, y AUC por cohorte. Nada de modelo_app/ contiene etiquetas ni ids de
pacientes: el script lo comprueba antes de terminar.
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
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

import informe
from analisis_presentable import ESPECIFICIDAD_OBJETIVO, N_BOOTSTRAP, SEMILLA, confusion, ic_bootstrap
from models.cnn_base import FILTROS_POR_BLOQUES, contar_parametros, crear_modelo

RAIZ_REPO = Path(__file__).resolve().parent
DIR_EXPERIMENTOS = RAIZ_REPO / "experimentos"
DIR_RUNS = RAIZ_REPO / "outputs" / "runs"
DIR_PAQUETE = RAIZ_REPO / "modelo_app"
VERSION_PAQUETE = 2
COHORTES = ("I-SPY1", "I-SPY2", "Duke")
BORDES_HISTOGRAMA = np.round(np.linspace(0, 1, 101), 2)       # intervalos de 0,01
DECIMALES_CURVAS = 4
# Prefijos de los ids de paciente: no pueden aparecer en modelo_app/.
PATRON_IDS = re.compile(r"ISPY1_|ISPY2-|ACRIN-6698|Breast_MRI_|patient_id|sample_id", re.IGNORECASE)


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def commit_actual() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ_REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def buscar_ejecucion(nombre: str) -> str:
    if (DIR_EXPERIMENTOS / nombre).is_dir():
        return nombre
    patron = re.compile(rf"^\d{{8}}_\d{{6}}_{re.escape(nombre)}$")
    candidatas = sorted(d.name for d in DIR_EXPERIMENTOS.iterdir() if d.is_dir() and patron.match(d.name))
    if not candidatas:
        sys.exit(f"ERROR: no hay ninguna ejecución '{nombre}' en experimentos/")
    if len(candidatas) > 1:
        print(f"AVISO: '{nombre}' se repite ({', '.join(candidatas)}); se usa la más reciente")
    return candidatas[-1]


def redondear(valores, decimales: int = DECIMALES_CURVAS) -> list[float]:
    return [round(float(v), decimales) for v in valores]


def _curvas_y_umbral(y: np.ndarray, p: np.ndarray, p_cohorte: np.ndarray, umbral: float) -> dict:
    """Lo común a los dos niveles (corte y paciente): curvas, calibración,
    histograma por clase y matriz de confusión con el umbral y con 0,5."""
    def en_umbral(u: float) -> dict:
        pred = p >= u
        vp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
        fp, vn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
        return {"umbral": u, "matriz_confusion": {"VP": vp, "FN": fn, "FP": fp, "VN": vn},
                "sensibilidad": vp / (vp + fn), "especificidad": vn / (vn + fp),
                "precision": vp / (vp + fp) if vp + fp else None}

    fpr, tpr, _ = roc_curve(y, p)
    fpr_c, tpr_c, _ = roc_curve(y, p_cohorte)
    precision, sensibilidad, _ = precision_recall_curve(y, p)
    precision_c, sensibilidad_c, _ = precision_recall_curve(y, p_cohorte)
    real, predicha = calibration_curve(y, p, n_bins=informe.N_GRUPOS_CALIBRACION, strategy="quantile")
    grupos = pd.qcut(p, informe.N_GRUPOS_CALIBRACION, duplicates="drop")
    por_grupo = pd.Series(p).groupby(grupos, observed=True).size().to_list()
    hist_0, _ = np.histogram(p[y == 0], bins=BORDES_HISTOGRAMA)
    hist_1, _ = np.histogram(p[y == 1], bins=BORDES_HISTOGRAMA)
    return {
        "average_precision": informe.ap_segura(y, p),
        "average_precision_solo_cohorte": informe.ap_segura(y, p_cohorte),
        "brier": float(np.mean((p - y) ** 2)),
        "prob_media": float(p.mean()),
        "con_umbral": en_umbral(umbral),
        "con_umbral_0_5": en_umbral(0.5),
        "curva_roc": {"fpr": redondear(fpr), "tpr": redondear(tpr)},
        "curva_roc_solo_cohorte": {"fpr": redondear(fpr_c), "tpr": redondear(tpr_c)},
        "curva_pr": {"sensibilidad": redondear(sensibilidad), "precision": redondear(precision)},
        "curva_pr_solo_cohorte": {"sensibilidad": redondear(sensibilidad_c), "precision": redondear(precision_c)},
        "calibracion": {"grupos": f"{informe.N_GRUPOS_CALIBRACION} cuantiles de la probabilidad",
                        "prob_media": redondear(predicha), "fraccion_pCR": redondear(real), "n": por_grupo},
        "histograma": {"bordes": BORDES_HISTOGRAMA.tolist(),
                       "pCR_0": hist_0.astype(int).tolist(), "pCR_1": hist_1.astype(int).tolist()},
    }


def bootstrap_por_paciente(y: np.ndarray, p: np.ndarray, p_cohorte: np.ndarray, paciente: np.ndarray,
                           umbral: float, rng) -> dict:
    """IC95 de las métricas por corte. Los cortes de una paciente no son
    independientes: cada remuestreo toma pacientes con reposición, con todos sus
    cortes. Los mismos remuestreos para el modelo y para «solo cohorte»."""
    indices = [np.flatnonzero(paciente == pid) for pid in np.unique(paciente)]
    bm, bc, sens, prec = [], [], [], []
    while len(bm) < N_BOOTSTRAP:
        i = np.concatenate([indices[k] for k in rng.integers(0, len(indices), len(indices))])
        yy = y[i]
        if yy.min() == yy.max():
            continue
        bm.append(roc_auc_score(yy, p[i]))
        bc.append(roc_auc_score(yy, p_cohorte[i]))
        pred = p[i] >= umbral
        sens.append(pred[yy == 1].mean())
        if pred.any():
            prec.append(yy[pred].mean())
    bm, bc = np.array(bm), np.array(bc)
    q = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
    return {"auc_ic95": q(bm), "auc_solo_cohorte_ic95": q(bc), "diferencia_ic95": q(bm - bc),
            "sensibilidad_ic95": q(sens), "precision_ic95": q(prec)}


def metricas_por_paciente(pac: pd.DataFrame, tasa_cohorte: dict) -> dict:
    """Métricas por paciente (media de las probabilidades de sus cortes, como
    train.py), con su propio umbral (mismo criterio). Información adicional: la
    app analiza cortes sueltos. `pac` tiene una fila por paciente y no sale del script."""
    y = pac.pCR.to_numpy().astype(int)
    p = pac.prob.to_numpy().astype(float)
    p_cohorte = pac.cohorte.map(tasa_cohorte).to_numpy().astype(float)
    auc = ic_bootstrap(y, p, p_cohorte, np.random.default_rng(SEMILLA))
    elegido = confusion(y, p)
    comunes = _curvas_y_umbral(y, p, p_cohorte, elegido["umbral"])
    comunes["con_umbral"].update({
        "sensibilidad_ic95": list(elegido["sens_ic"]), "precision_ic95": list(elegido["prec_ic"]),
        "ic95_metodo": "Wilson", "sensibilidad_azar": elegido["sens_azar"], "precision_azar": elegido["prec_azar"]})
    por_cohorte = {}
    for c in COHORTES:
        sub = pac[pac.cohorte == c]
        if len(sub):
            por_cohorte[c] = {"pacientes": int(len(sub)), "pCR_1": int(sub.pCR.sum()),
                              "auc": informe.auc_seguro(sub.pCR, sub.prob)}
    return {
        "nivel": "paciente: media de las probabilidades de sus cortes",
        "pacientes": int(len(pac)), "pCR_1": int(y.sum()), "tasa_pCR": float(y.mean()),
        "auc": auc["auc"], "auc_ic95": list(auc["auc_ic"]),
        "auc_solo_cohorte": auc["auc_cohorte"], "auc_solo_cohorte_ic95": list(auc["auc_cohorte_ic"]),
        "diferencia_auc_modelo_menos_cohorte": auc["dif"], "diferencia_ic95": list(auc["dif_ic"]),
        "ic95_metodo": f"bootstrap por paciente, {N_BOOTSTRAP} remuestreos, semilla {SEMILLA}, "
                       "los mismos para el modelo y para 'solo cohorte'",
        "umbral": elegido["umbral"],
        "umbral_regla": "el mismo criterio que el de la app, sobre las probabilidades por paciente",
        **comunes,
        "por_cohorte": por_cohorte,
    }


def metricas_validacion(pred: pd.DataFrame, tasa_cohorte: dict) -> tuple[dict, dict]:
    """Métricas agregadas y umbral. El umbral se elige POR CORTE, la misma entrada
    que analiza la app (un corte suelto): el de mayor sensibilidad entre los que
    dan especificidad >= 0,85 sobre la probabilidad de cada corte de validación.
    `pred` es predicciones_val.csv (una fila por corte); no sale del script."""
    y = pred.pCR.to_numpy().astype(int)
    p = pred.prob_corte.to_numpy().astype(float)
    p_cohorte = pred.cohorte.map(tasa_cohorte).to_numpy().astype(float)
    if np.isnan(p_cohorte).any():
        sys.exit("ERROR: hay cohortes de validación sin tasa de pCR de entrenamiento en config.json")
    paciente = pred.patient_id.to_numpy()

    elegido = confusion(y, p)
    umbral = elegido["umbral"]
    ic = bootstrap_por_paciente(y, p, p_cohorte, paciente, umbral, np.random.default_rng(SEMILLA))
    auc, auc_c = float(roc_auc_score(y, p)), float(roc_auc_score(y, p_cohorte))
    comunes = _curvas_y_umbral(y, p, p_cohorte, umbral)
    comunes["con_umbral"].update({
        "sensibilidad_ic95": ic["sensibilidad_ic95"], "precision_ic95": ic["precision_ic95"],
        "ic95_metodo": "bootstrap por paciente (todos sus cortes juntos)",
        "sensibilidad_azar": elegido["sens_azar"], "precision_azar": elegido["prec_azar"]})

    por_cohorte = {}
    for c in COHORTES:
        sub = pred[pred.cohorte == c]
        if len(sub):
            por_cohorte[c] = {"cortes": int(len(sub)), "pacientes": int(sub.patient_id.nunique()),
                              "pCR_1": int(sub.groupby("patient_id").pCR.first().sum()),
                              "auc": informe.auc_seguro(sub.pCR, sub.prob_corte)}
    pac = pred.groupby("patient_id").agg(cohorte=("cohorte", "first"), pCR=("pCR", "first"),
                                         prob=("prob_paciente", "first"))
    metricas = {
        "nivel": "corte: la probabilidad de cada corte de validación, la misma entrada que analiza la app",
        "pacientes": int(len(pac)), "cortes": int(len(pred)), "pCR_1": int(pac.pCR.sum()),
        "cortes_pCR_1": int(y.sum()), "tasa_pCR": float(y.mean()),
        "auc": auc, "auc_ic95": ic["auc_ic95"],
        "auc_solo_cohorte": auc_c, "auc_solo_cohorte_ic95": ic["auc_solo_cohorte_ic95"],
        "diferencia_auc_modelo_menos_cohorte": auc - auc_c, "diferencia_ic95": ic["diferencia_ic95"],
        "ic95_metodo": f"bootstrap por paciente (cada remuestreo toma pacientes con todos sus cortes), "
                       f"{N_BOOTSTRAP} remuestreos, semilla {SEMILLA}, los mismos para el modelo y para "
                       "'solo cohorte'",
        "tasa_pCR_entrenamiento_por_cohorte": tasa_cohorte,
        **comunes,
        "por_cohorte": por_cohorte,
        "por_paciente": metricas_por_paciente(pac, tasa_cohorte),
    }
    umbral_info = {
        "valor": umbral,
        "regla": "el de mayor sensibilidad entre los que dan especificidad ≥ "
                 + f"{ESPECIFICIDAD_OBJETIVO:.2f}".replace(".", ",")
                 + " en las probabilidades de validación por corte de esta ejecución; positivo si prob ≥ umbral",
        "especificidad_objetivo": ESPECIFICIDAD_OBJETIVO,
        "nivel": "corte: se elige con la probabilidad de cada corte de validación, la misma entrada que "
                 "analiza la app",
        "nota": "elegido y medido con las mismas predicciones: sensibilidad y especificidad son optimistas",
    }
    return metricas, umbral_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ejecucion", help="carpeta de experimentos/ o nombre de la ejecución")
    ap.add_argument("--salida", type=Path, default=DIR_PAQUETE)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ejecucion = buscar_ejecucion(args.ejecucion)
    dir_exp, dir_run = DIR_EXPERIMENTOS / ejecucion, DIR_RUNS / ejecucion
    for ruta in (dir_exp / "config.json", dir_run / "best.pt", dir_run / "predicciones_val.csv"):
        if not ruta.exists():
            sys.exit(f"ERROR: falta {ruta.relative_to(RAIZ_REPO).as_posix()} "
                     "(los pesos y las predicciones solo están en la máquina que entrenó)")
    config = json.loads((dir_exp / "config.json").read_text(encoding="utf-8"))
    hiper = config["hiperparametros"]

    sha_best = sha256(dir_run / "best.pt")
    sha_config = (config.get("sha256") or {}).get("best.pt")
    if sha_config and sha_config != sha_best:
        sys.exit(f"ERROR: el SHA-256 de best.pt ({sha_best[:12]}...) no es el de config.json ({sha_config[:12]}...)")

    # Modelo: arquitectura de best.pt (las ejecuciones antiguas no guardan bloques
    # ni realce: son la línea base).
    estado = torch.load(dir_run / "best.pt", map_location="cpu", weights_only=True)
    bloques = int(estado.get("bloques", hiper.get("bloques", 4)))
    realce = bool(estado.get("realce", hiper.get("realce", False)))
    recorte_caja = hiper.get("recorte_caja")
    recorte_centrado = (hiper.get("recorte_centrado") or {}).get("lado_px")
    # La app solo reproduce el preprocesado de preprocesado.preprocesar (sin recorte,
    # con el recorte por la caja o con la ventana centrada en ella).
    otros = sorted(k for k, v in hiper.items()
                   if k.startswith("recorte") and k not in ("recorte_caja", "recorte_centrado") and v)
    if otros:
        sys.exit(f"ERROR: la ejecución usa {', '.join(otros)}, que la app no sabe reproducir")
    if recorte_caja and recorte_centrado:
        sys.exit("ERROR: la ejecución usa a la vez recorte_caja y recorte_centrado")
    if bool(estado.get("recorte_caja", False)) != bool(recorte_caja):
        sys.exit("ERROR: best.pt y config.json no coinciden en recorte_caja")
    if estado.get("recorte_centrado") != recorte_centrado:
        sys.exit(f"ERROR: best.pt (recorte_centrado {estado.get('recorte_centrado')}) y config.json "
                 f"({recorte_centrado}) no coinciden")
    if recorte_centrado is not None:
        recorte_centrado = int(recorte_centrado)
    modelo = crear_modelo(bloques, realce, hiper.get("dropout", 0.3))
    modelo.load_state_dict(estado["modelo"])                    # estricto: mismos nombres y formas
    n_parametros = contar_parametros(modelo)

    args.salida.mkdir(parents=True, exist_ok=True)
    ruta_pt = args.salida / "modelo.pt"
    torch.save(estado["modelo"], ruta_pt)
    recargado = torch.load(ruta_pt, map_location="cpu", weights_only=True)
    if recargado.keys() != estado["modelo"].keys() or not all(
            torch.equal(recargado[k], estado["modelo"][k]) for k in recargado):
        sys.exit("ERROR: modelo.pt no reproduce los pesos de best.pt")

    # Métricas por corte (las de la app) y, como información adicional, por
    # paciente (la media de sus cortes, la misma agregación que train.py).
    pred = pd.read_csv(dir_run / "predicciones_val.csv")
    tasa_cohorte = config["datos"]["tasa_pCR_entrenamiento_por_cohorte"]
    metricas, umbral = metricas_validacion(pred, tasa_cohorte)

    resultado = config.get("resultado") or {}
    entorno = config.get("entorno") or {}
    criterio = hiper.get("criterio_seleccion", "auc")
    epoca = int(estado.get("epoca", resultado.get("mejor_epoca", 0)))
    if criterio == "ultima" and epoca != resultado.get("epocas_ejecutadas"):
        sys.exit(f"ERROR: con --criterio ultima best.pt debería ser la última época "
                 f"({resultado.get('epocas_ejecutadas')}) y es la {epoca}")
    if recorte_centrado:
        entrada = [3, recorte_centrado, recorte_centrado]
    else:
        entrada = [3, *((recorte_caja or {}).get("tamano_salida") or [256, 256])]
    paquete = {
        "version_paquete": VERSION_PAQUETE,
        "exportado": datetime.now().isoformat(timespec="seconds"),
        "exportado_con": {"script": "exportar_modelo.py", "commit": commit_actual()},
        "ejecucion": ejecucion,
        "origen": {"pesos": f"outputs/runs/{ejecucion}/best.pt", "sha256_best_pt": sha_best,
                   "config": f"experimentos/{ejecucion}/config.json",
                   "commit_entrenamiento": entorno.get("git_commit")},
        "modelo": {
            "fichero": "modelo.pt", "sha256": sha256(ruta_pt), "formato": "state_dict de PyTorch",
            "arquitectura": "CNNBase (models/cnn_base.py, crear_modelo)", "bloques": bloques,
            "filtros": list(FILTROS_POR_BLOQUES[bloques]), "realce": realce,
            "dropout": hiper.get("dropout", 0.3), "n_parametros": n_parametros,
            "entrada": entrada,
            "salida": "un logit; P(pCR) = sigmoide(logit)",
            "epoca": epoca,
            "seleccion": {
                "criterio": criterio, "epoca_usada": epoca, "epocas_ejecutadas": resultado.get("epocas_ejecutadas"),
                "descripcion": ("sin selección: best.pt es la última época entrenada (épocas fijas, sin early "
                                "stopping ni elección de época con la validación)") if criterio == "ultima" else
                               f"best.pt es la mejor época de validación según el criterio {criterio}",
            },
        },
        "preprocesado": {
            "funcion": "preprocesado.preprocesar",
            "fases": ["PRE", "EARLY", "LATE"],
            "formato_entrada": "3 PNG en escala de grises de 8 bits y 256x256",
            "normalizacion": hiper.get("normalizacion", "dividir entre 255 (preprocesado.preprocesar)"),
            "recorte_caja": recorte_caja,
            "recorte_centrado": recorte_centrado,
            "recorte_centrado_detalle": ({**hiper["recorte_centrado"], "funcion": "preprocesado.recortar_centrado",
                                          "llamada": "preprocesado.preprocesar(pre, early, late, caja=caja, "
                                                     "lado_centrado=recorte_centrado)"}
                                         if recorte_centrado else None),
        },
        "umbral": umbral,
        "entrenamiento": {
            "fold_validacion": config.get("fold_val"), "semilla": config.get("semilla"),
            "pacientes_entrenamiento": config["datos"].get("pacientes_entrenamiento"),
            "pacientes_validacion": config["datos"].get("pacientes_validacion"),
            "optimizador": hiper.get("optimizador"), "lr": hiper.get("lr"), "betas": hiper.get("betas"),
            "weight_decay": hiper.get("weight_decay"), "lr_lineal": hiper.get("lr_lineal"),
            "dropout": hiper.get("dropout", 0.3), "batch_size": hiper.get("batch_size"),
            "cortes_por_paciente": hiper.get("cortes_por_paciente"), "cortes_centrales": hiper.get("cortes_centrales"),
            "epocas_max": hiper.get("epocas_max"), "paciencia": hiper.get("paciencia"),
            "criterio_seleccion": criterio,
            "perdida": hiper.get("perdida"), "pos_weight": config.get("pos_weight"),
            "aumento_datos": hiper.get("aumento_datos"), "agregacion_paciente": hiper.get("agregacion_paciente"),
            "epocas_ejecutadas": resultado.get("epocas_ejecutadas"), "mejor_epoca": resultado.get("mejor_epoca"),
            "parada_temprana": resultado.get("parada_temprana"),
            "tiempo_total_min": (resultado["tiempo_total_s"] / 60) if resultado.get("tiempo_total_s") else None,
            "dispositivo": entorno.get("dispositivo"), "gpu": entorno.get("gpu"), "cpu": entorno.get("cpu"),
            "pytorch": entorno.get("pytorch"),
        },
        "metricas_validacion": metricas,
    }
    ruta_json = args.salida / "modelo.json"
    informe.guardar_json(ruta_json, paquete)

    # Ni etiquetas individuales ni ids: solo agregados.
    texto = ruta_json.read_text(encoding="utf-8")
    if PATRON_IDS.search(texto):
        ruta_json.unlink()
        sys.exit("ERROR: modelo.json contenía un id de paciente o de corte; se ha borrado")
    extra = sorted(p.name for p in args.salida.iterdir() if p.name not in ("modelo.pt", "modelo.json"))
    if extra:
        print(f"AVISO: {args.salida.name}/ tiene otros ficheros: {', '.join(extra)}")

    m, mp = metricas, metricas["por_paciente"]
    print(f"Paquete creado en {args.salida.resolve().as_posix()}/ desde {ejecucion}")
    print(f"  modelo.pt: {ruta_pt.stat().st_size / 1024:.0f} KB, SHA-256 {paquete['modelo']['sha256']}")
    print(f"  arquitectura: {bloques} bloques, realce {'sí' if realce else 'no'}, {n_parametros} parámetros, "
          f"recorte por la caja {'sí' if recorte_caja else 'no'}, recorte centrado {recorte_centrado or 'no'}; "
          f"entrada {' x '.join(map(str, entrada))}")
    print(f"  época {epoca} de {resultado.get('epocas_ejecutadas')} (criterio {criterio}"
          + (": la última, sin elegir" if criterio == "ultima" else "") + ")")
    print(f"  AUC validación por corte {m['auc']:.3f} (IC95 {m['auc_ic95'][0]:.3f}-{m['auc_ic95'][1]:.3f}); "
          f"solo cohorte {m['auc_solo_cohorte']:.3f}")
    print(f"  AUC validación por paciente {mp['auc']:.3f} (IC95 {mp['auc_ic95'][0]:.3f}-{mp['auc_ic95'][1]:.3f}); "
          f"solo cohorte {mp['auc_solo_cohorte']:.3f}; umbral por paciente {mp['umbral']:.4f}")
    c = m["con_umbral"]
    print(f"  umbral por corte {umbral['valor']:.4f}: sensibilidad {c['sensibilidad']:.3f}, especificidad "
          f"{c['especificidad']:.3f}, precisión {c['precision']:.3f} ({c['matriz_confusion']})")


if __name__ == "__main__":
    main()
