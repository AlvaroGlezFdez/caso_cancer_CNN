"""Resumen agregado del dataset para la página «Datos» de la app (app/datos/resumen_dataset.json).

Uso, desde la raíz del repositorio (necesita breastdcedl/):

    python preparar_datos_app.py

Solo recuentos, proporciones y cuantiles de train (pacientes, cortes, clases y
adquisición por cohorte, subtipo y pCR por cohorte, cortes por paciente, folds). De test, solo los
totales de metadata/statistics.json, los mismos que da la guía. Añade los
hallazgos de la auditoría visual (recuentos de la tabla HALLAZGOS, descritos en
el informe, docs/informe.pdf). Ningún id ni etiqueta individual: el script lo
comprueba antes de escribir.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

RAIZ_REPO = Path(__file__).resolve().parent
RAIZ_DATOS = RAIZ_REPO / "breastdcedl"
SALIDA = RAIZ_REPO / "app" / "datos" / "resumen_dataset.json"
sys.path.insert(0, str(RAIZ_DATOS))
import utils_caso as uc  # noqa: E402

COHORTES = {"spy1": "I-SPY1", "spy2": "I-SPY2", "duke": "Duke"}
SUBTIPOS = {"HRposHER2neg": "HR+/HER2−", "HER2pos": "HER2+", "TripleNeg": "Triple negativo"}
ORDEN = ["I-SPY1", "I-SPY2", "Duke"]
PATRON_IDS = re.compile(r"ISPY1_|ISPY2-|ACRIN-6698|Breast_MRI_|patient_id|sample_id")

# Auditoría visual (fuera del repositorio; descrita en el informe, docs/informe.pdf):
# 20 pacientes de train por cohorte, corte central, un anotador.
HALLAZGOS = {
    "fuente": "Auditoría visual de 60 pacientes de train (20 por cohorte), sobre el corte central y con un único "
              "anotador, con revisión asistida por IA; detalle en el informe (docs/informe.pdf)",
    "corazon": {"descripcion": "estructura grande y muy realzada detrás de la pared torácica (posible corazón)",
                "si": {"I-SPY1": 0, "I-SPY2": 13, "Duke": 13},
                "dudoso": {"I-SPY1": 9, "I-SPY2": 3, "Duke": 1}, "por_cohorte": 20},
    "movimiento": {"descripcion": "bordes dobles rojo/azul en el realce EARLY-PRE (posible movimiento entre fases)",
                   "si": {"I-SPY1": 6, "I-SPY2": 3, "Duke": 4},
                   "dudoso": {"I-SPY1": 3, "I-SPY2": 7, "Duke": 4}, "por_cohorte": 20},
    "lesion_delimitada": {"descripcion": "lesión focal delimitada visible en el corte central",
                          "si": {"I-SPY1": 15, "I-SPY2": 13, "Duke": 12},
                          "dudoso": {"I-SPY1": 4, "I-SPY2": 6, "Duke": 6}, "por_cohorte": 20},
    "caja_tumor": {"descripcion": "la caja del tumor de patients.csv (caja_tumor.caja_en_png) cae sobre la lesión",
                   "pacientes_con_lesion_clara": 40, "acierta": 40, "errores": 0,
                   "dentro_recorte_central": {"128": 0.386, "160": 0.613, "192": 0.831},
                   "dentro_recorte_128_por_cohorte": {"I-SPY1": 0.365, "I-SPY2": 0.270, "Duke": 0.828}},
    "relacion_pcr": {"descripcion": "AUC como predictor único de pCR, en train (IC95 por bootstrap)",
                     "xy_spacing": [0.499, 0.460, 0.537], "lado_caja_columnas": [0.440, 0.405, 0.475],
                     "lado_caja_filas": [0.465, 0.428, 0.499]},
}


def cuantiles(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"mediana": float(np.median(x)), "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
            "min": float(x.min()), "max": float(x.max())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    samples = uc.cargar_samples(RAIZ_DATOS)
    patients = uc.cargar_patients(RAIZ_DATOS)
    patients["cohorte"] = patients.dataset.map(COHORTES)
    estadisticas = json.loads((RAIZ_DATOS / "metadata" / "statistics.json").read_text(encoding="utf-8"))
    tr_s = samples[samples.split == "train"]
    tr_p = patients[patients.split == "train"]
    te_p = patients[patients.split == "test"]
    tr_p = tr_p.assign(cortes=tr_p.pid.map(tr_s.groupby("patient_id").size()))
    tr_p = tr_p.assign(fold=tr_p.pid.map(tr_s.drop_duplicates("patient_id").set_index("patient_id").fold))
    # El PNG es una ventana de 256 px sin reescalar (n_xy >= 256) o el corte ampliado a 256 (n_xy = 128).
    tr_p = tr_p.assign(pixel_png=np.where(tr_p.n_xy >= 256, tr_p.xy_spacing, tr_p.xy_spacing * tr_p.n_xy / 256))
    tr_p = tr_p.assign(campo_mm=256 * tr_p.pixel_png)

    resumen = {
        "nota": "Solo agregados de train y los totales de test de statistics.json. Ningún id ni etiqueta individual.",
        "train": {"pacientes": int(len(tr_p)), "cortes": int(len(tr_s)),
                  "pCR_1": int(tr_p.pCR.sum()), "pCR_0": int((tr_p.pCR == 0).sum())},
        "test": {"pacientes": estadisticas["patients_by_split"]["test"],
                 "cortes": estadisticas["samples_by_split"]["test"],
                 "pCR_1": estadisticas["patients_by_split_and_pcr"]["test"]["pCR_1"],
                 "pCR_0": estadisticas["patients_by_split_and_pcr"]["test"]["pCR_0"]},
        "por_cohorte": {},
        "cortes_por_paciente": {str(k): int(v) for k, v in tr_p.cortes.value_counts().sort_index().items()},
        "folds": {},
    }
    for c in ORDEN:
        g = tr_p[tr_p.cohorte == c]
        resumen["por_cohorte"][c] = {
            "pacientes_train": int(len(g)), "cortes_train": int(g.cortes.sum()),
            "pCR_1": int(g.pCR.sum()), "tasa_pCR": float(g.pCR.mean()),
            "pacientes_test": int((te_p.cohorte == c).sum()),
            "n_xy": {str(int(k)): int(v) for k, v in g.n_xy.value_counts().sort_index().items()},
            "xy_spacing_mm": cuantiles(g.xy_spacing), "pixel_png_mm": cuantiles(g.pixel_png),
            "grosor_corte_mm": cuantiles(g.slice_thick),
            "campo_png_mm": cuantiles(g.campo_mm),
        }
    # Subtipo (HR_HER2_STATUS de patients.csv) y pCR por cohorte: solo recuentos.
    subtipo = tr_p.HR_HER2_STATUS.map(SUBTIPOS).fillna("Sin dato")
    resumen["por_cohorte_subtipo"] = {
        c: {s: {"pCR_0": int(((tr_p.cohorte == c) & (subtipo == s) & (tr_p.pCR == 0)).sum()),
                "pCR_1": int(((tr_p.cohorte == c) & (subtipo == s) & (tr_p.pCR == 1)).sum())}
            for s in [*SUBTIPOS.values(), "Sin dato"] if ((tr_p.cohorte == c) & (subtipo == s)).any()}
        for c in ORDEN}
    for f, g in tr_p.groupby("fold"):
        resumen["folds"][str(int(f))] = {"pacientes": int(len(g)), "pCR_1": int(g.pCR.sum()),
                                         "tasa_pCR": float(g.pCR.mean())}
    resumen["hallazgos_auditoria"] = HALLAZGOS

    texto = json.dumps(resumen, indent=2, ensure_ascii=False)
    if PATRON_IDS.search(texto):
        sys.exit("ERROR: el resumen contiene un id; no se escribe")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(texto, encoding="utf-8")
    print(f"Resumen en {SALIDA.relative_to(RAIZ_REPO).as_posix()}: train {resumen['train']}, test {resumen['test']}")
    for c, v in resumen["por_cohorte"].items():
        print(f"  {c}: {v['pacientes_train']} pacientes, tasa pCR {v['tasa_pCR']:.3f}, campo mediano "
              f"{v['campo_png_mm']['mediana']:.0f} mm")


if __name__ == "__main__":
    main()
