"""Exporta las probabilidades de validación (out-of-fold) de la CNN, por paciente y por corte.

Se ejecuta en el ordenador donde están outputs/runs/ (la GPU):

    python exportar_oof.py

Para cada configuración de validación cruzada con sus 5 folds completos
(ejecuciones llamadas cv_<configuracion>_f0 ... f4), lee
outputs/runs/<ejecución>/predicciones_val.csv (mejor época de cada fold) y escribe:

    experimentos/oof/<configuracion>.csv         patient_id, fold, prob
                                                 (prob_paciente: media de sus cortes)
    experimentos/oof_cortes/<configuracion>.csv  sample_id, patient_id, fold, slice_index, prob
                                                 (prob_corte: la de cada corte)

Los dos salen de la columna correspondiente de predicciones_val.csv, que tienen
todas las ejecuciones, también las antiguas. No contienen la etiqueta pCR, pero
sí ids de pacientes y de cortes: no se versionan (.gitignore) y se quedan en la
máquina con los pesos. Si una ejecución está repetida, se usa la más reciente.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
RUNS = RAIZ / "outputs" / "runs"
SALIDA = RAIZ / "experimentos" / "oof"
SALIDA_CORTES = RAIZ / "experimentos" / "oof_cortes"
PATRON = re.compile(r"^(?P<fecha>\d{8}_\d{6})_cv_(?P<config>.+)_f(?P<fold>[0-4])$")


def main() -> None:
    if not RUNS.is_dir():
        sys.exit(f"No existe {RUNS}: ejecuta esto en el ordenador donde se entrenó.")
    elegidas: dict[tuple[str, int], tuple[str, Path]] = {}
    for d in sorted(RUNS.iterdir()):
        m = PATRON.match(d.name)
        if not m or not (d / "predicciones_val.csv").is_file():
            continue
        clave = (m["config"], int(m["fold"]))
        if clave not in elegidas or m["fecha"] > elegidas[clave][0]:
            elegidas[clave] = (m["fecha"], d)

    configs = sorted({c for c, _ in elegidas})
    SALIDA.mkdir(parents=True, exist_ok=True)
    SALIDA_CORTES.mkdir(parents=True, exist_ok=True)
    for config in configs:
        folds = [f for f in range(5) if (config, f) in elegidas]
        if len(folds) < 5:
            print(f"  {config}: faltan folds ({folds}), se omite")
            continue
        partes, partes_cortes = [], []
        for f in folds:
            pred = pd.read_csv(elegidas[(config, f)][1] / "predicciones_val.csv")
            pac = (pred.groupby("patient_id", as_index=False)["prob_paciente"].first()
                       .rename(columns={"prob_paciente": "prob"}))
            pac.insert(1, "fold", f)
            partes.append(pac)
            cortes = pred[["sample_id", "patient_id", "slice_index", "prob_corte"]].rename(
                columns={"prob_corte": "prob"})
            cortes.insert(2, "fold", f)
            partes_cortes.append(cortes)
        oof = pd.concat(partes, ignore_index=True)
        if oof.patient_id.duplicated().any():
            sys.exit(f"{config}: hay pacientes en más de un fold. Revisa las ejecuciones.")
        oof_cortes = pd.concat(partes_cortes, ignore_index=True)
        if oof_cortes.sample_id.duplicated().any():
            sys.exit(f"{config}: hay cortes en más de un fold. Revisa las ejecuciones.")
        oof.to_csv(SALIDA / f"{config}.csv", index=False)
        oof_cortes.to_csv(SALIDA_CORTES / f"{config}.csv", index=False)
        print(f"  {config}: {len(oof)} pacientes -> {SALIDA / (config + '.csv')}; "
              f"{len(oof_cortes)} cortes -> {SALIDA_CORTES / (config + '.csv')}")


if __name__ == "__main__":
    main()
