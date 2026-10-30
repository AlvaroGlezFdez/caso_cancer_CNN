"""Rehace las gráficas e informes de las ejecuciones ya terminadas con el informe.py actual.

Uso, desde la raíz del repositorio (por ejemplo, tras cambiar las escalas o
añadir una gráfica):

    python regenerar_informes.py              # todas las carpetas de experimentos/
    python regenerar_informes.py 20261022_042121_baseline_30ep_gpu ...

Para cada ejecución:
  - Si están sus predicciones (outputs/runs/<ejecución>/predicciones_val.csv,
    que solo existen en la máquina que la entrenó), rehace el informe completo
    (informe.generar_informe) y comprueba que todas las métricas que ya estaban
    en metricas_finales.json salen idénticas; si alguna cambia, restaura los
    ficheros anteriores y termina con error.
  - Si no, solo rehace curvas_aprendizaje.png, que sale de metricas_epoca.csv;
    el resto de gráficas se quedan como estaban.
No toca metricas_epoca.csv, config.json, train.log ni experimentos.csv.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parent
RAIZ_DATOS = RAIZ_REPO / "breastdcedl"
sys.path.insert(0, str(RAIZ_DATOS))
import utils_caso as uc  # noqa: E402

import aumento  # noqa: E402
import informe  # noqa: E402

DIR_EXPERIMENTOS = RAIZ_REPO / "experimentos"
DIR_RUNS = RAIZ_REPO / "outputs" / "runs"
DATASET_A_COHORTE = {"spy1": "I-SPY1", "spy2": "I-SPY2", "duke": "Duke"}   # como en train.py


def tasa_cohorte_entrenamiento(samples: pd.DataFrame, fold: int) -> pd.Series:
    """Tasa de pCR por cohorte en la partición de entrenamiento, igual que train.py."""
    entrenamiento, _ = uc.particion(samples, fold_val=fold)
    return entrenamiento.drop_duplicates("patient_id").groupby("cohorte").pCR.mean()


def contexto_de(config: dict, epocas: pd.DataFrame) -> dict:
    """El `contexto` que train.py pasa a generar_informe, reconstruido desde config.json."""
    h, entorno = config["hiperparametros"], config["entorno"]
    pos_weight = config.get("pos_weight")
    return {
        "ejecucion": config["ejecucion"], "fold": config["fold_val"], "modelo": h["modelo"],
        "n_parametros": config["n_parametros"], "bloques": h.get("bloques", 4), "realce": h.get("realce", False),
        "criterio": h.get("criterio_seleccion", "auc"),
        "dispositivo": f"{entorno['dispositivo']} ({entorno.get('gpu') or entorno['cpu']})",
        "pos_weight": None if pos_weight is None else round(pos_weight, 4),
        "weight_decay": h.get("weight_decay", 0.0),
        "aumento": aumento.etiqueta() if h.get("aumento_datos") else None,
        "epocas_ejecutadas": len(epocas), "mejor_epoca": config["resultado"]["mejor_epoca"],
    }


def iguales(antes, despues) -> list[str]:
    """Claves de `antes` cuyo valor cambia en `despues` (las claves nuevas no cuentan).

    Tolerancia 1e-8: predicciones_val.csv guarda las probabilidades por corte
    con ~7 cifras, así que Brier y probabilidad media cambian hasta ~5e-10
    respecto a los calculados en memoria (comprobado el 2026-10-22). AUC,
    sensibilidad, especificidad y recuentos salen exactos.
    """
    distintas = []
    for clave, v in antes.items():
        w = despues.get(clave)
        if isinstance(v, dict):
            distintas += [f"{clave}.{c}" for c in iguales(v, w or {})]
        elif isinstance(v, float) or isinstance(w, float):
            if v is None or w is None or not np.isclose(v, w, rtol=0, atol=1e-8):
                distintas.append(clave)
        elif v != w:
            distintas.append(clave)
    return distintas


def regenerar(carpeta: Path, samples: pd.DataFrame) -> str:
    config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    epocas = pd.read_csv(carpeta / "metricas_epoca.csv")
    contexto = contexto_de(config, epocas)
    finales = json.loads((carpeta / "metricas_finales.json").read_text(encoding="utf-8"))
    predicciones = DIR_RUNS / carpeta.name / "predicciones_val.csv"

    if not predicciones.exists():
        auc_cohorte = finales["auc_solo_cohorte"]
        informe.figura_curvas(epocas, contexto["mejor_epoca"], np.nan if auc_cohorte is None else auc_cohorte,
                              carpeta / "curvas_aprendizaje.png", ponderada=contexto["pos_weight"] is not None,
                              criterio=contexto["criterio"])
        return "solo curvas_aprendizaje.png (sus predicciones no están en este ordenador)"

    pred = pd.read_csv(predicciones).rename(columns={"prob_corte": "prob"}).drop(columns="prob_paciente")
    copia = Path(tempfile.mkdtemp())
    shutil.copytree(carpeta, copia, dirs_exist_ok=True)
    final = informe.generar_informe(carpeta, epocas, pred, tasa_cohorte_entrenamiento(samples, contexto["fold"]),
                                    contexto, config["hiperparametros"]["umbral"],
                                    config["hiperparametros"]["agregacion_paciente"])
    distintas = iguales(finales, informe.sin_nan(final))
    if distintas:
        shutil.copytree(copia, carpeta, dirs_exist_ok=True)
        if not (copia / "curva_pr.png").exists():          # la figura nueva no estaba antes
            (carpeta / "curva_pr.png").unlink(missing_ok=True)
        sys.exit(f"{carpeta.name}: cambian métricas que ya estaban ({', '.join(distintas)}); "
                 "se han restaurado los ficheros anteriores")
    shutil.rmtree(copia)
    return f"informe completo (métricas anteriores idénticas; AP {final['average_precision']:.3f})"


def main():
    samples = uc.cargar_samples(RAIZ_DATOS)
    cohorte = uc.cargar_patients(RAIZ_DATOS).set_index("pid").dataset.map(DATASET_A_COHORTE)
    samples["cohorte"] = samples.patient_id.map(cohorte)
    nombres = sys.argv[1:] or sorted(p.name for p in DIR_EXPERIMENTOS.iterdir() if (p / "config.json").exists())
    for nombre in nombres:
        print(f"{nombre}: {regenerar(DIR_EXPERIMENTOS / nombre, samples)}")


if __name__ == "__main__":
    main()
