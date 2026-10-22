"""Resumen de la validación cruzada de 5 folds: línea base frente a aumento.

Uso, desde la raíz del repositorio, cuando hayan terminado las 10 ejecuciones
(cv_baseline_f0 ... cv_baseline_f4 y cv_aumento_f0 ... cv_aumento_f4):

    python cv_resumen.py

Lee experimentos/experimentos.csv (una fila por ejecución) y escribe:
  experimentos/cv_resumen.csv   por fold: AUC de validación por paciente de cada
                                configuración, diferencia aumento - línea base y
                                AUC "solo cohorte"; más las filas media y
                                desv_tipica (desviación típica muestral, ddof=1)
  experimentos/cv_resumen.md    la misma tabla legible y las ejecuciones usadas

Son resultados agregados (sin etiquetas ni predicciones por paciente): pueden
subirse a git. Si un nombre aparece en varias ejecuciones (porque se relanzó),
se usa la más reciente y se avisa. Si falta alguna ejecución, no escribe nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parent
DIR_EXPERIMENTOS = RAIZ_REPO / "experimentos"
CSV_EXPERIMENTOS = DIR_EXPERIMENTOS / "experimentos.csv"

# configuración -> prefijo del --nombre de sus ejecuciones (<prefijo>_f<fold>)
CONFIGURACIONES = {"linea_base": "cv_baseline", "aumento": "cv_aumento"}
FOLDS = range(5)

# Columnas que deben coincidir en las 10 ejecuciones: solo pueden diferir el
# fold y el aumento.
COLUMNAS_COMUNES = ["modelo", "n_parametros", "semilla", "optimizador", "lr", "weight_decay",
                    "batch_size", "epocas_max", "paciencia", "pos_weight", "dispositivo"]


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
    distintas = [c for c in COLUMNAS_COMUNES if elegidas[c].nunique(dropna=False) > 1]
    if distintas:
        sys.exit("Las ejecuciones difieren en algo más que el fold y el aumento: " + ", ".join(distintas))
    if elegidas.git_commit.nunique() > 1:
        print("Aviso: las ejecuciones no comparten commit: " + ", ".join(sorted(elegidas.git_commit.unique())))
    return elegidas


def tabla_resumen(elegidas: pd.DataFrame) -> pd.DataFrame:
    """Por fold: AUC de cada configuración, diferencia y "solo cohorte"; más media y desv_tipica."""
    auc = elegidas.pivot(index="fold", columns="configuracion", values="auc_val").astype(float)
    cohorte = elegidas.pivot(index="fold", columns="configuracion", values="auc_solo_cohorte").astype(float)
    # "Solo cohorte" depende solo de la partición, así que debe ser igual en las dos configuraciones.
    if not np.allclose(cohorte["linea_base"], cohorte["aumento"]):
        sys.exit("El AUC 'solo cohorte' de un mismo fold difiere entre configuraciones:\n" + cohorte.to_string())

    tabla = pd.DataFrame({
        "auc_linea_base": auc["linea_base"],
        "auc_aumento": auc["aumento"],
        "dif_aumento_menos_linea_base": auc["aumento"] - auc["linea_base"],
        "auc_solo_cohorte": cohorte["linea_base"],
    })
    tabla.index = tabla.index.astype(str)
    resumen = pd.DataFrame({"media": tabla.mean(), "desv_tipica": tabla.std(ddof=1)}).T
    return pd.concat([tabla, resumen]).rename_axis("fold")


def escribir_md(tabla: pd.DataFrame, elegidas: pd.DataFrame, ruta: Path) -> None:
    dif = tabla.loc[[str(f) for f in FOLDS], "dif_aumento_menos_linea_base"]
    lineas = [
        "# Validación cruzada de 5 folds: línea base frente a aumento",
        "",
        "AUC de validación por paciente (agregación media) del mejor modelo de cada ejecución.",
        "La desviación típica es la muestral (ddof=1) sobre los 5 folds.",
        "",
        "| fold | AUC línea base | AUC aumento | aumento - línea base | AUC solo cohorte |",
        "|---|---|---|---|---|",
    ]
    for fold, fila in tabla.iterrows():
        formato_dif = ".3f" if fold == "desv_tipica" else "+.3f"
        lineas.append(f"| {fold} | {fila.auc_linea_base:.3f} | {fila.auc_aumento:.3f} | "
                      f"{fila.dif_aumento_menos_linea_base:{formato_dif}} | {fila.auc_solo_cohorte:.3f} |")
    lineas += [
        "",
        f"El aumento supera a la línea base en {int((dif > 0).sum())} de {len(dif)} folds.",
        "",
        "## Ejecuciones usadas",
        "",
        "| configuración | fold | ejecución | mejor época | épocas ejecutadas |",
        "|---|---|---|---|---|",
    ]
    for _, fila in elegidas.sort_values(["configuracion", "fold"], ascending=[False, True]).iterrows():
        lineas.append(f"| {fila.configuracion} | {fila.fold} | {fila.ejecucion} | {fila.mejor_epoca} | "
                      f"{fila.epocas_ejecutadas} |")
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    experimentos = pd.read_csv(CSV_EXPERIMENTOS, float_precision="round_trip")
    elegidas = seleccionar(experimentos)
    tabla = tabla_resumen(elegidas)

    tabla.to_csv(DIR_EXPERIMENTOS / "cv_resumen.csv", float_format="%.6f")
    escribir_md(tabla, elegidas, DIR_EXPERIMENTOS / "cv_resumen.md")
    print(tabla.round(3).to_string())
    print(f"\nEscrito en {DIR_EXPERIMENTOS / 'cv_resumen.csv'} y {DIR_EXPERIMENTOS / 'cv_resumen.md'}")


if __name__ == "__main__":
    main()
