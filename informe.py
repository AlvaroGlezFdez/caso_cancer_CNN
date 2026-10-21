"""Informe de una ejecución a partir de las predicciones de validación del mejor modelo.

Lo llama train.py al terminar cada ejecución. Escribe en la carpeta del
experimento:

    curvas_aprendizaje.png   pérdida (entrenamiento y validación) y AUC por época
    matriz_confusion.png     por paciente, con el umbral y la agregación dados
    curva_roc.png            por paciente, junto a la referencia "solo cohorte"
    calibracion.png          curva de calibración e histograma por clase
    metricas_cohorte.csv     métricas por cohorte (I-SPY1, I-SPY2, Duke y todas)
    metricas_finales.json    métricas globales del mejor modelo
    informe.md               resumen con las tablas y las gráficas

Todo son resultados agregados: ninguna etiqueta ni predicción por paciente, así
que puede subirse a git.

Referencia "solo cohorte": a cada paciente de validación se le predice la tasa
de pCR de su cohorte, calculada sobre los pacientes de la partición de
entrenamiento. Si el modelo no la supera, puede estar aprendiendo la cohorte y
no la biología.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

RAIZ_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_REPO / "breastdcedl"))
import utils_caso as uc  # noqa: E402

COHORTES = ("I-SPY1", "I-SPY2", "Duke")
N_GRUPOS_CALIBRACION = 10

# Paleta de la auditoría: posiciones 1 y 2 de la paleta categórica de referencia
# y rampa secuencial azul para los recuentos.
AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2 = "#0b0b0b", "#52514e"
CMAP_RECUENTOS = LinearSegmentedColormap.from_list(
    "recuentos", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#104281"]
)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #

def auc_seguro(y, p) -> float:
    """AUC, o NaN si solo hay una clase."""
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def por_paciente(pred: pd.DataFrame, metodo: str, umbral: float) -> pd.DataFrame:
    """Una fila por paciente: pCR, cohorte y probabilidad agregada (`prob`)."""
    pac = pred.groupby("patient_id").agg(pCR=("pCR", "first"), cohorte=("cohorte", "first"))
    return pac.join(uc.agregar_por_paciente(pred.patient_id.values, pred.prob.values, metodo, umbral))


def metricas_por_cohorte(pred: pd.DataFrame, umbral: float, metodo: str) -> pd.DataFrame:
    filas = {}
    grupos = [(c, pred[pred.cohorte == c]) for c in COHORTES] + [("Todas", pred)]
    for nombre, sub in grupos:
        if sub.empty:
            continue
        r = uc.evaluar_por_paciente(sub.prob.values, sub, umbral=umbral, metodo=metodo)
        n_pcr = int(sub.groupby("patient_id").pCR.first().sum())
        filas[nombre] = {
            "pacientes": r["pacientes"], "pCR_1": n_pcr, "tasa_pCR": n_pcr / r["pacientes"],
            "auc": r["auc"], "sensibilidad": r["sensibilidad"], "especificidad": r["especificidad"],
            "accuracy": r["accuracy"], "precision": r["precision"], **r["matriz_confusion"],
        }
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "cohorte"
    return tabla


# --------------------------------------------------------------------------- #
# Gráficas
# --------------------------------------------------------------------------- #

def _estilo(ax):
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TINTA_2)
    ax.xaxis.label.set_color(TINTA_2)
    ax.yaxis.label.set_color(TINTA_2)
    ax.title.set_color(TINTA)


def figura_curvas(epocas: pd.DataFrame, mejor_epoca: int, auc_cohorte: float, ruta: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = epocas.epoca
    a1.plot(x, epocas.perdida_entr, color=AZUL, lw=2, marker="o", ms=5, label="entrenamiento")
    a1.plot(x, epocas.perdida_val, color=NARANJA, lw=2, marker="o", ms=5, label="validación")
    a1.set(title="Pérdida por época", xlabel="época", ylabel="BCE (media por corte)")
    a1.legend(frameon=False)

    a2.plot(x, epocas.auc_val, color=AZUL, lw=2, marker="o", ms=5, label="modelo")
    a2.axhline(0.5, color=TINTA_2, lw=1, ls=":", label="azar (0.5)")
    if np.isfinite(auc_cohorte):
        a2.axhline(auc_cohorte, color=TINTA_2, lw=1.2, ls="--", label=f"solo cohorte ({auc_cohorte:.3f})")
    a2.axvline(mejor_epoca, color=TINTA_2, lw=1, ls="-.", alpha=0.7, label=f"mejor época ({mejor_epoca})")
    a2.set(title="AUC de validación por paciente", xlabel="época", ylabel="AUC")
    a2.legend(frameon=False)
    for a in (a1, a2):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        _estilo(a)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def figura_confusion(mc: dict, umbral: float, metodo: str, ruta: Path):
    m = np.array([[mc["VN"], mc["FP"]], [mc["FN"], mc["VP"]]])
    nombres = np.array([["VN", "FP"], ["FN", "VP"]])
    fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    ax.imshow(m, cmap=CMAP_RECUENTOS, vmin=0, vmax=max(int(m.max()), 1))
    for i in range(2):
        fila = m[i].sum()
        for j in range(2):
            pct = f"{100 * m[i, j] / fila:.0f} % de la fila" if fila else "—"
            claro = m[i, j] > 0.55 * m.max()
            ax.text(j, i, f"{nombres[i, j]} = {m[i, j]}\n{pct}", ha="center", va="center",
                    fontsize=12, color="white" if claro else TINTA)
    etiquetas = ["no pCR (0)", "pCR (1)"]
    ax.set_xticks([0, 1], etiquetas)
    ax.set_yticks([0, 1], etiquetas)
    ax.set_xlabel("predicha", color=TINTA_2)
    ax.set_ylabel("real", color=TINTA_2)
    ax.axhline(0.5, color="white", lw=3)
    ax.axvline(0.5, color="white", lw=3)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, colors=TINTA_2)
    ax.set_title(f"Matriz de confusión por paciente (n = {m.sum()})\n"
                 f"umbral {umbral}, agregación {metodo}", color=TINTA)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def figura_roc(pac: pd.DataFrame, prob_cohorte: pd.Series, auc_modelo: float, auc_cohorte: float, ruta: Path):
    fig, ax = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color=TINTA_2, lw=1, ls=":", label="azar")
    if pac.pCR.nunique() == 2:
        fpr, tpr, _ = roc_curve(pac.pCR, pac.prob)
        ax.plot(fpr, tpr, color=AZUL, lw=2, label=f"modelo (AUC {auc_modelo:.3f})")
        fpr, tpr, _ = roc_curve(pac.pCR, prob_cohorte)
        ax.plot(fpr, tpr, color=NARANJA, lw=2, ls="--", marker="o", ms=7,
                label=f"solo cohorte (AUC {auc_cohorte:.3f})")
    else:
        ax.text(0.5, 0.5, "Solo hay una clase:\nla curva ROC no está definida", ha="center", color=TINTA)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 − especificidad", ylabel="sensibilidad",
           title=f"Curva ROC por paciente (n = {len(pac)})")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right")
    _estilo(ax)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def figura_calibracion(pac: pd.DataFrame, umbral: float, brier: float, ruta: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    real, predicha = calibration_curve(pac.pCR, pac.prob, n_bins=N_GRUPOS_CALIBRACION, strategy="quantile")
    a1.plot([0, 1], [0, 1], color=TINTA_2, lw=1, ls=":", label="calibración perfecta")
    a1.plot(predicha, real, color=AZUL, lw=2, marker="o", ms=7, label="modelo")
    a1.set(xlim=(0, 1), ylim=(0, 1), xlabel="probabilidad media predicha en el grupo",
           ylabel="fracción real de pCR en el grupo",
           title=f"Calibración por paciente ({N_GRUPOS_CALIBRACION} grupos de igual tamaño)\n"
                 f"Brier = {brier:.3f}")
    a1.legend(frameon=False, loc="upper left")

    bins = np.linspace(0, 1, 21)
    for clase, color, nombre in ((0, AZUL, "no pCR (0)"), (1, NARANJA, "pCR (1)")):
        p = pac.prob[pac.pCR == clase]
        a2.hist(p, bins=bins, color=color, alpha=0.3, histtype="stepfilled")
        a2.hist(p, bins=bins, color=color, lw=2, histtype="step", label=f"{nombre}, n = {len(p)}")
    a2.axvline(umbral, color=TINTA_2, lw=1, ls="--", label=f"umbral {umbral}")
    a2.set(xlim=(0, 1), xlabel="probabilidad de pCR (agregada por paciente)", ylabel="pacientes",
           title="Probabilidades por clase real")
    a2.yaxis.set_major_locator(MaxNLocator(integer=True))
    a2.legend(frameon=False)
    for a in (a1, a2):
        _estilo(a)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Informe
# --------------------------------------------------------------------------- #

def sin_nan(obj):
    """NaN -> None, para que el JSON sea estándar (null en vez de NaN)."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: sin_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sin_nan(v) for v in obj]
    return obj


def guardar_json(ruta: Path, obj):
    ruta.write_text(json.dumps(sin_nan(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def tabla_md(df: pd.DataFrame, decimales: int = 3) -> str:
    """DataFrame -> tabla Markdown (sin depender de tabulate)."""
    d = df.reset_index()

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else f"{v:.{decimales}f}"
        return str(v)
    cab = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    filas = ["| " + " | ".join(fmt(v) for v in fila) + " |" for fila in d.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def generar_informe(carpeta: Path, epocas: pd.DataFrame, pred: pd.DataFrame, tasa_cohorte_entr: pd.Series,
                    contexto: dict, umbral: float = 0.5, metodo: str = "mean") -> dict:
    """Genera gráficas y tablas en `carpeta` y devuelve las métricas finales.

    `pred` tiene una fila por corte de validación con patient_id, cohorte, pCR y
    prob (salida del mejor modelo). `tasa_cohorte_entr` es la tasa de pCR por
    cohorte en los pacientes de entrenamiento. `contexto` son datos de la
    ejecución para la cabecera del informe.
    """
    pac = por_paciente(pred, metodo, umbral)
    glob = uc.evaluar_por_paciente(pred.prob.values, pred, umbral=umbral, metodo=metodo)
    cohortes = metricas_por_cohorte(pred, umbral, metodo)
    prob_cohorte = pac.cohorte.map(tasa_cohorte_entr)
    auc_cohorte = auc_seguro(pac.pCR, prob_cohorte)
    brier = float(np.mean((pac.prob - pac.pCR) ** 2))

    final = {
        "pacientes": glob["pacientes"], "metodo": metodo, "umbral": umbral,
        "auc": glob["auc"], "sensibilidad": glob["sensibilidad"], "especificidad": glob["especificidad"],
        "accuracy": glob["accuracy"], "precision": glob["precision"], "brier": brier,
        "prob_media_predicha": float(pac.prob.mean()), "tasa_pCR_real": float(pac.pCR.mean()),
        "auc_solo_cohorte": auc_cohorte, "matriz_confusion": glob["matriz_confusion"],
        "auc_por_cohorte": {c: float(cohortes.auc.get(c, np.nan)) for c in COHORTES},
    }

    figura_curvas(epocas, contexto["mejor_epoca"], auc_cohorte, carpeta / "curvas_aprendizaje.png")
    figura_confusion(glob["matriz_confusion"], umbral, metodo, carpeta / "matriz_confusion.png")
    figura_roc(pac, prob_cohorte, glob["auc"], auc_cohorte, carpeta / "curva_roc.png")
    figura_calibracion(pac, umbral, brier, carpeta / "calibracion.png")
    cohortes.to_csv(carpeta / "metricas_cohorte.csv")
    guardar_json(carpeta / "metricas_finales.json", final)

    tasa = pd.DataFrame({"tasa_pCR_entrenamiento": tasa_cohorte_entr.reindex(list(COHORTES))})
    resumen = pd.DataFrame({"valor": {
        "AUC": final["auc"], "sensibilidad": final["sensibilidad"],
        "especificidad": final["especificidad"], "accuracy": final["accuracy"],
        "precisión": final["precision"], "Brier": brier,
        "probabilidad media predicha": final["prob_media_predicha"],
        "tasa real de pCR": final["tasa_pCR_real"], "AUC solo cohorte": auc_cohorte,
    }})
    resumen.index.name = "métrica"
    L = [
        f"# Informe: {contexto['ejecucion']}\n",
        f"- Validación interna: fold {contexto['fold']} ({final['pacientes']} pacientes). Test no se ha usado.",
        f"- Modelo: {contexto['modelo']} ({contexto['n_parametros']:,} parámetros)".replace(",", ".")
        + f", dispositivo {contexto['dispositivo']}.",
        f"- pos_weight: {contexto['pos_weight'] if contexto['pos_weight'] is not None else 'no'}; "
        f"épocas ejecutadas: {contexto['epocas_ejecutadas']}; mejor época: {contexto['mejor_epoca']}.",
        f"- Optimizador AdamW, weight decay {contexto.get('weight_decay', 0):g}; "
        f"aumento de datos: {contexto.get('aumento') or 'no'}.",
        f"- Métricas por paciente: agregación `{metodo}` de las probabilidades de sus cortes, umbral {umbral}.",
        "",
        "## Métricas del mejor modelo\n",
        tabla_md(resumen) + "\n",
        "## Por cohorte\n",
        tabla_md(cohortes) + "\n",
        "## Referencia \"solo cohorte\"\n",
        "A cada paciente de validación se le asigna la tasa de pCR de su cohorte en los pacientes "
        "de entrenamiento. Dentro de una cohorte esa referencia es constante, así que el AUC por "
        "cohorte del modelo mide lo que aporta más allá de saber de qué cohorte viene la imagen.\n",
        tabla_md(tasa) + "\n",
        f"AUC solo cohorte: **{auc_cohorte:.3f}** frente a **{final['auc']:.3f}** del modelo.\n",
        "## Gráficas\n",
        "![Curvas de aprendizaje](curvas_aprendizaje.png)\n",
        "![Matriz de confusión](matriz_confusion.png) ![Curva ROC](curva_roc.png)\n",
        "![Calibración](calibracion.png)\n",
    ]
    (carpeta / "informe.md").write_text("\n".join(L), encoding="utf-8")
    return final
