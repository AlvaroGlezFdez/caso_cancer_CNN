"""Resumen de la validación cruzada de 5 folds: línea base frente a aumento.

Uso, desde la raíz del repositorio, cuando hayan terminado las 10 ejecuciones
(cv_baseline_f0 ... cv_baseline_f4 y cv_aumento_f0 ... cv_aumento_f4):

    python cv_resumen.py

Lee experimentos/experimentos.csv (una fila por ejecución) y el
metricas_finales.json y metricas_epoca.csv de cada ejecución, y escribe:
  experimentos/cv_resumen.csv           por fold: AUC y average precision (AP) de
                                        validación por paciente de cada configuración,
                                        diferencia de AUC aumento - línea base, AUC
                                        "solo cohorte" y AP del azar (la tasa de pCR);
                                        más las filas media y desv_tipica (desviación
                                        típica muestral, ddof=1)
  experimentos/cv_criterio_suavizado.csv  qué época y qué AUC habría dado el criterio
                                        auc_suavizado (seleccion.py) en cada ejecución,
                                        reconstruido desde metricas_epoca.csv
  experimentos/cv_curvas_pr.png         curvas precisión-sensibilidad de los 5 folds de
                                        cada configuración (necesita las predicciones de
                                        outputs/runs/, que solo están en la máquina que
                                        entrenó; si faltan, no se dibuja y se avisa)
  experimentos/cv_resumen.md            todo lo anterior, legible

Son resultados agregados (sin etiquetas ni predicciones por paciente): pueden
subirse a git. Si un nombre aparece en varias ejecuciones (porque se relanzó),
se usa la más reciente y se avisa. Si falta alguna ejecución, no escribe nada.

Reconstrucción del criterio auc_suavizado: el criterio no cambia el
entrenamiento, solo qué época se guarda y cuándo se para, y en la GPU del
laboratorio el entrenamiento es determinista; así que basta con recorrer los AUC
por época guardados. Es exacta si la ejecución habría parado dentro de las
épocas que se entrenaron. Si no, falta al menos la época n+1 (el valor
suavizado de la última época entrenada necesita el AUC de la siguiente): se da
la elección provisional y el AUC que tendría que haber dado esa época n+1 para
cambiarla.
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

import informe
import seleccion

RAIZ_REPO = Path(__file__).resolve().parent
DIR_EXPERIMENTOS = RAIZ_REPO / "experimentos"
DIR_RUNS = RAIZ_REPO / "outputs" / "runs"
CSV_EXPERIMENTOS = DIR_EXPERIMENTOS / "experimentos.csv"

# configuración -> prefijo del --nombre de sus ejecuciones (<prefijo>_f<fold>)
CONFIGURACIONES = {"linea_base": "cv_baseline", "aumento": "cv_aumento"}
NOMBRES = {"linea_base": "Línea base", "aumento": "Aumento"}
FOLDS = range(5)

# Columnas que deben coincidir en las 10 ejecuciones: solo pueden diferir el
# fold y el aumento. Las tres últimas existen desde el 2026-10-22.
COLUMNAS_COMUNES = ["modelo", "n_parametros", "semilla", "optimizador", "lr", "weight_decay",
                    "batch_size", "epocas_max", "paciencia", "pos_weight", "dispositivo",
                    "criterio", "bloques", "realce"]

# Un color por fold: los 5 primeros puestos, en orden, de la paleta categórica
# de referencia (los dos primeros son AZUL y NARANJA de informe.py).
COLORES_FOLD = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def nombre_de(ejecucion: str) -> str:
    """'20261022_042121_cv_baseline_f0' -> 'cv_baseline_f0'."""
    return ejecucion.split("_", 2)[2]


def seleccionar(experimentos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (configuración, fold); sale con error si falta alguna o no cuadra."""
    experimentos = experimentos.assign(nombre=experimentos.ejecucion.map(nombre_de))
    indices, configs, folds, faltan = [], [], [], []
    for config, prefijo in CONFIGURACIONES.items():
        for fold in FOLDS:
            nombre = f"{prefijo}_f{fold}"
            candidatas = experimentos[experimentos.nombre == nombre].sort_values("fecha_hora")
            if candidatas.empty:
                faltan.append(nombre)
                continue
            if len(candidatas) > 1:
                print(f"Aviso: {nombre} tiene {len(candidatas)} ejecuciones; se usa la más reciente "
                      f"({candidatas.ejecucion.iloc[-1]})")
            fila = candidatas.iloc[-1]
            if fila.fold_val != fold:
                sys.exit(f"{fila.ejecucion}: el nombre dice fold {fold} pero fold_val es {fila.fold_val}")
            con_aumento = pd.notna(fila.aumento_datos) and fila.aumento_datos != ""
            if con_aumento != (config == "aumento"):
                sys.exit(f"{fila.ejecucion}: aumento_datos = {fila.aumento_datos!r} no cuadra con '{config}'")
            indices.append(fila.name)
            configs.append(config)
            folds.append(fold)
    if faltan:
        sys.exit("Faltan ejecuciones en experimentos.csv: " + ", ".join(faltan))

    # Se seleccionan las filas de experimentos.csv (y no se apilan Series sueltas)
    # para conservar los tipos: con columnas object, nunique(dropna=False) cuenta
    # cada NaN como un valor distinto.
    elegidas = experimentos.loc[indices].assign(configuracion=configs, fold=folds).reset_index(drop=True)
    distintas = [c for c in COLUMNAS_COMUNES if c in elegidas and elegidas[c].nunique(dropna=False) > 1]
    if distintas:
        sys.exit("Las ejecuciones difieren en algo más que el fold y el aumento: " + ", ".join(distintas))
    if elegidas.git_commit.nunique() > 1:
        print("Aviso: las ejecuciones no comparten commit: " + ", ".join(sorted(elegidas.git_commit.unique())))

    finales = [json.loads((DIR_EXPERIMENTOS / e / "metricas_finales.json").read_text(encoding="utf-8"))
               for e in elegidas.ejecucion]
    if any(f.get("average_precision") is None for f in finales):
        sys.exit("Falta average_precision en algún metricas_finales.json: python regenerar_informes.py")
    return elegidas.assign(ap_val=[f["average_precision"] for f in finales],
                           tasa_pCR_val=[f["tasa_pCR_real"] for f in finales])


def resumir(tabla: pd.DataFrame) -> pd.DataFrame:
    """Añade las filas media y desv_tipica (ddof=1) a una tabla con una fila por fold."""
    tabla.index = tabla.index.astype(str)
    resumen = pd.DataFrame({"media": tabla.mean(), "desv_tipica": tabla.std(ddof=1)}).T
    return pd.concat([tabla, resumen]).rename_axis("fold")


def tabla_resumen(elegidas: pd.DataFrame) -> pd.DataFrame:
    """Por fold: AUC y AP de cada configuración, diferencia de AUC, "solo cohorte" y AP del azar."""
    def por_fold(columna):
        return elegidas.pivot(index="fold", columns="configuracion", values=columna).astype(float)

    auc, ap, cohorte, tasa = por_fold("auc_val"), por_fold("ap_val"), por_fold("auc_solo_cohorte"), \
        por_fold("tasa_pCR_val")
    # "Solo cohorte" y la tasa de pCR dependen solo de la partición: deben coincidir.
    for nombre, t in (("AUC 'solo cohorte'", cohorte), ("tasa de pCR de validación", tasa)):
        if not np.allclose(t["linea_base"], t["aumento"]):
            sys.exit(f"{nombre} de un mismo fold difiere entre configuraciones:\n" + t.to_string())

    return resumir(pd.DataFrame({
        "auc_linea_base": auc["linea_base"],
        "auc_aumento": auc["aumento"],
        "dif_aumento_menos_linea_base": auc["aumento"] - auc["linea_base"],
        "auc_solo_cohorte": cohorte["linea_base"],
        "ap_linea_base": ap["linea_base"],
        "ap_aumento": ap["aumento"],
        "ap_azar": tasa["linea_base"],
    }))


# --------------------------------------------------------------------------- #
# Criterio auc_suavizado reconstruido
# --------------------------------------------------------------------------- #

def umbral_de_cambio(aucs: list[float], r: dict, paciencia: int) -> float | None:
    """En una reconstrucción incompleta (faltan épocas), el AUC que tendría que
    haber dado la época n+1, la primera que no se entrenó, para cambiar la época
    elegida. None si con la n+1 aún no se habría parado (harían falta más épocas).

    Con el AUC `a` de la época n+1: la época n vale (AUC[n-1] + AUC[n] + a) / 3;
    si no mejora, se cumple la paciencia y se para en n+1, que se fija con las
    disponibles, (AUC[n] + a) / 2. La elección cambia si alguna de las dos
    supera el mejor valor.
    """
    if r["sin_mejora"] + 1 < paciencia:
        return None
    n, mejor = len(aucs), r["valor"]
    return min(3 * mejor - aucs[n - 2] - aucs[n - 1], 2 * mejor - aucs[n - 1])


def reconstruir(elegidas: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for _, fila in elegidas.iterrows():
        aucs = pd.read_csv(DIR_EXPERIMENTOS / fila.ejecucion / "metricas_epoca.csv").auc_val.tolist()
        paciencia, maximo = int(fila.paciencia), int(fila.epocas_max)
        r = seleccion.simular(aucs, "auc_suavizado", paciencia, maximo)
        umbral = None if r["completa"] else umbral_de_cambio(aucs, r, paciencia)
        # Si ni un AUC de 1 en la época n+1 cambia la elección, el resultado también es seguro.
        segura = r["completa"] or (umbral is not None and umbral >= 1)
        filas.append({
            "configuracion": fila.configuracion, "fold": fila.fold, "ejecucion": fila.ejecucion,
            "paciencia": paciencia, "epocas_entrenadas": len(aucs),
            "mejor_epoca_auc": int(fila.mejor_epoca), "auc_criterio_auc": float(fila.auc_val),
            "mejor_epoca_suavizado": r["mejor_epoca"], "valor_suavizado": r["valor"],
            "auc_criterio_suavizado": aucs[r["mejor_epoca"] - 1],
            "exacta": segura,
            "auc_necesario_epoca_siguiente": None if segura else umbral,
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Curvas precisión-sensibilidad
# --------------------------------------------------------------------------- #

def figura_pr(elegidas: pd.DataFrame, ruta: Path) -> bool:
    """Una gráfica por configuración con las curvas de sus 5 folds. False si
    faltan predicciones (no se dibuja nada)."""
    rutas = {e: DIR_RUNS / e / "predicciones_val.csv" for e in elegidas.ejecucion}
    faltan = [e for e, r in rutas.items() if not r.exists()]
    if faltan:
        print("Aviso: no se dibuja cv_curvas_pr.png; faltan predicciones de: " + ", ".join(faltan))
        return False
    configs = list(CONFIGURACIONES)
    fig, ejes = plt.subplots(1, len(configs), figsize=(5.6 * len(configs), 5.6), constrained_layout=True)
    for ax, config in zip(ejes, configs):
        filas = elegidas[elegidas.configuracion == config].sort_values("fold")
        for _, fila in filas.iterrows():
            pac = pd.read_csv(rutas[fila.ejecucion]).groupby("patient_id").agg(
                pCR=("pCR", "first"), prob=("prob_paciente", "first"))
            color = COLORES_FOLD[fila.fold]
            ax.axhline(pac.pCR.mean(), color=color, lw=1, ls=":")
            informe.curva_pr(ax, pac.pCR, pac.prob, color, f"fold {fila.fold} (AP {fila.ap_val:.3f})", lw=1.8)
        ax.plot([], [], color=informe.TINTA_2, lw=1, ls=":", label="azar de cada fold (su tasa de pCR)")
        ax.set(xlim=informe.ESCALA_PROBABILIDAD, ylim=informe.ESCALA_PROBABILIDAD,
               xlabel="sensibilidad (recall)", ylabel="precisión (valor predictivo positivo)",
               title=f"{NOMBRES[config]}: AP media {filas.ap_val.mean():.3f} "
                     f"(DT {filas.ap_val.std(ddof=1):.3f})")
        ax.set_aspect("equal")
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        informe._estilo(ax)
    fig.suptitle("Precisión-sensibilidad por paciente, validación de cada fold", color=informe.TINTA)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Informe
# --------------------------------------------------------------------------- #

def escribir_md(tabla: pd.DataFrame, elegidas: pd.DataFrame, criterio: pd.DataFrame, con_figura: bool,
                ruta: Path) -> None:
    folds = [str(f) for f in FOLDS]
    dif = tabla.loc[folds, "dif_aumento_menos_linea_base"]

    def con_signo(fold):
        return ".3f" if fold == "desv_tipica" else "+.3f"

    lineas = [
        "# Validación cruzada de 5 folds: línea base frente a aumento",
        "",
        "Métricas de validación por paciente (agregación media) del mejor modelo de cada ejecución,",
        "elegido con el criterio con que se entrenó (`auc`: el AUC de cada época). La desviación",
        "típica es la muestral (ddof=1) sobre los 5 folds.",
        "",
        "## AUC",
        "",
        "| fold | AUC línea base | AUC aumento | aumento - línea base | AUC solo cohorte |",
        "|---|---|---|---|---|",
    ]
    for fold, fila in tabla.iterrows():
        lineas.append(f"| {fold} | {fila.auc_linea_base:.3f} | {fila.auc_aumento:.3f} | "
                      f"{fila.dif_aumento_menos_linea_base:{con_signo(fold)}} | {fila.auc_solo_cohorte:.3f} |")
    lineas += [
        "",
        f"El aumento supera a la línea base en {int((dif > 0).sum())} de {len(dif)} folds.",
        "",
        "## Average precision (área bajo la curva precisión-sensibilidad)",
        "",
        "La AP del azar es la tasa de pCR de la validación de cada fold.",
        "",
        "| fold | AP línea base | AP aumento | AP del azar |",
        "|---|---|---|---|",
    ]
    for fold, fila in tabla.iterrows():
        lineas.append(f"| {fold} | {fila.ap_linea_base:.3f} | {fila.ap_aumento:.3f} | {fila.ap_azar:.3f} |")
    lineas += ["", "![Curvas precisión-sensibilidad](cv_curvas_pr.png)" if con_figura
               else "(cv_curvas_pr.png no se ha dibujado: faltan predicciones en outputs/runs/)", ""]

    # Criterio auc_suavizado reconstruido
    lineas += [
        "## Criterio auc_suavizado, reconstruido",
        "",
        "Qué época habría guardado el criterio auc_suavizado (media del AUC de la época con la",
        "anterior y la siguiente; en los extremos, las disponibles) con la misma paciencia, y su",
        "AUC. Se reconstruye desde metricas_epoca.csv: el criterio no cambia el entrenamiento y en",
        "esta GPU el entrenamiento es determinista. Es exacta si la ejecución habría parado dentro",
        "de las épocas entrenadas. Si no, la época elegida es provisional: se mantiene salvo que el",
        "AUC de la primera época no entrenada hubiera superado el valor de la última columna.",
        "",
        "| configuración | fold | épocas entrenadas | criterio auc: época (AUC) | "
        "auc_suavizado: época (AUC; valor suavizado) | exacta | cambia si la época siguiente da AUC > |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, f in criterio.iterrows():
        umbral = f.auc_necesario_epoca_siguiente
        lineas.append(
            f"| {NOMBRES[f.configuracion]} | {f.fold} | {f.epocas_entrenadas} | "
            f"{f.mejor_epoca_auc} ({f.auc_criterio_auc:.3f}) | "
            f"{f.mejor_epoca_suavizado} ({f.auc_criterio_suavizado:.3f}; {f.valor_suavizado:.3f}) | "
            f"{'sí' if f.exacta else 'no'} | "
            f"{'—' if f.exacta else ('harían falta varias épocas' if pd.isna(umbral) else f'{umbral:.3f}')} |")
    medias = criterio.pivot(index="fold", columns="configuracion", values="auc_criterio_suavizado")
    d = medias["aumento"] - medias["linea_base"]
    n_exactas = int(criterio.exacta.sum())
    lineas += [
        "",
        f"Media de los 5 folds con auc_suavizado ({n_exactas} de {len(criterio)} reconstrucciones exactas; "
        "las demás, provisionales): " + ", ".join(
            f"{NOMBRES[c]} {medias[c].mean():.3f} (DT {medias[c].std(ddof=1):.3f})" for c in CONFIGURACIONES)
        + f"; aumento - línea base {d.mean():+.3f} (DT {d.std(ddof=1):.3f}).",
        "",
        "## Ejecuciones usadas",
        "",
        "| configuración | fold | ejecución | mejor época | épocas ejecutadas |",
        "|---|---|---|---|---|",
    ]
    for _, fila in elegidas.sort_values(["configuracion", "fold"], ascending=[False, True]).iterrows():
        lineas.append(f"| {NOMBRES[fila.configuracion]} | {fila.fold} | {fila.ejecucion} | {fila.mejor_epoca} | "
                      f"{fila.epocas_ejecutadas} |")
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    experimentos = pd.read_csv(CSV_EXPERIMENTOS, float_precision="round_trip")
    elegidas = seleccionar(experimentos)
    tabla = tabla_resumen(elegidas)
    criterio = reconstruir(elegidas)
    con_figura = figura_pr(elegidas, DIR_EXPERIMENTOS / "cv_curvas_pr.png")

    tabla.to_csv(DIR_EXPERIMENTOS / "cv_resumen.csv", float_format="%.6f")
    criterio.to_csv(DIR_EXPERIMENTOS / "cv_criterio_suavizado.csv", index=False, float_format="%.6f")
    escribir_md(tabla, elegidas, criterio, con_figura, DIR_EXPERIMENTOS / "cv_resumen.md")
    print(tabla.round(3).to_string())
    print()
    print(criterio.drop(columns="ejecucion").round(3).to_string(index=False))
    print(f"\nEscrito en {DIR_EXPERIMENTOS}: cv_resumen.csv, cv_criterio_suavizado.csv, cv_resumen.md"
          + (", cv_curvas_pr.png" if con_figura else ""))


if __name__ == "__main__":
    main()
