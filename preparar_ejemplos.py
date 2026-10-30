"""Prepara los 3 cortes de ejemplo de la app (app/ejemplos/), sin etiqueta ni id.

Uso, desde la raíz del repositorio (necesita breastdcedl/):

    python preparar_ejemplos.py

Son tres cortes de entrenamiento, uno por cohorte, todos de pacientes del fold
2: la validación de cv_baseline_f2, el modelo final previsto, que por tanto no
los ha visto al entrenar. De cada paciente se toma el corte central (cortes
ordenados por slice_index, el de posición len // 2), con una lesión visible en
la auditoría visual (../exploracion/03_caja_tumor).

Cada ejemplo queda en app/ejemplos/ejemplo_N/ como ejemplo_N_PRE.png,
ejemplo_N_EARLY.png y ejemplo_N_LATE.png: los mismos píxeles que el dataset
(se comprueba), guardados de nuevo con Pillow para no arrastrar metadatos, y con
nombres que no dicen de qué paciente son. app/ejemplos/indice.json guarda el
título, la cohorte, una descripción, la caja del tumor en el PNG
(caja_tumor.caja_en_png, de patients.csv), su centro como punto sugerido para
la curva cinética y las variables de la paciente que usa el modelo combinado
(subtipo, volumen tumoral, edad y la caja de patients.csv: n_xy, sraw, eraw,
scol, ecol y xy_spacing); nunca la pCR ni el id.

Del ejemplo 1 se guarda además app/ejemplos/ejemplo_1/pila.npz: todos los cortes
de su paciente (PRE, EARLY y LATE en uint8, ordenados por altura), su posición
relativa, cuál es el del ejemplo, el grosor de corte y el tamaño del píxel en mm
(xy_spacing, que es el del PNG cuando n_xy >= 256). Lo usa la vista 3D de la
página «Datos».

Imágenes derivadas de BreastDCEDL (Fridman et al., 2026), licencia CC BY-NC 4.0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ_REPO = Path(__file__).resolve().parent
RAIZ_DATOS = RAIZ_REPO / "breastdcedl"
DIR_EJEMPLOS = RAIZ_REPO / "app" / "ejemplos"
sys.path.insert(0, str(RAIZ_DATOS))
import utils_caso as uc  # noqa: E402

from caja_tumor import caja_en_png  # noqa: E402
from preprocesado import FASES, leer_fase  # noqa: E402

# Paciente de origen (solo en este script) y lo que la app puede mostrar.
EJEMPLOS = [
    ("ISPY2-166412", {"titulo": "Ejemplo 1", "cohorte": "I-SPY2",
                      "descripcion": "Lesión con realce intenso y, detrás de la pared torácica, una "
                                     "estructura grande muy realzada (corazón o tórax)."}),
    ("ISPY1_1038", {"titulo": "Ejemplo 2", "cohorte": "I-SPY1",
                    "descripcion": "Nódulo redondeado con realce intenso en una mama densa."}),
    ("Breast_MRI_534", {"titulo": "Ejemplo 3", "cohorte": "Duke",
                        "descripcion": "Lesión pequeña y centrada; la pared torácica queda fuera del campo."}),
]
FOLD_ESPERADO = 2
VARIABLES = ("HR_HER2_STATUS", "tum_vol", "age", "n_xy", "sraw", "eraw", "scol", "ecol", "xy_spacing")
EJEMPLO_CON_PILA = 1
LICENCIA = ("Imágenes derivadas de BreastDCEDL (N. Fridman et al., Scientific Data, 2026; "
            "doi:10.1038/s41597-026-06589-6), licencia CC BY-NC 4.0. Uso docente y de investigación.")


def guardar_pila(cortes, paciente, destino: Path) -> None:
    fases = {f.lower(): np.stack([leer_fase(RAIZ_DATOS / getattr(fila, f"path_{f.lower()}"), f)
                                  for fila in cortes.itertuples()]) for f in FASES}
    pixel_mm = paciente.xy_spacing if paciente.n_xy >= 256 else paciente.xy_spacing * paciente.n_xy / 256
    np.savez_compressed(destino, **fases,
                        posicion=(cortes.slice_index - cortes.slice_index.min()).to_numpy(np.int16),
                        indice_ejemplo=np.int16(len(cortes) // 2),
                        grosor_mm=np.float32(paciente.slice_thick), pixel_mm=np.float32(pixel_mm))


def variables_paciente(paciente) -> dict:
    """Las columnas de patients.csv que pide el modelo combinado (sin pCR ni id)."""
    return {c: (None if paciente[c] != paciente[c] else paciente[c] if isinstance(paciente[c], str)
                else float(paciente[c])) for c in VARIABLES}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    samples = uc.cargar_samples(RAIZ_DATOS)
    patients = uc.cargar_patients(RAIZ_DATOS).set_index("pid")
    train = samples[samples.split == "train"]
    indice = []
    for n, (pid, info) in enumerate(EJEMPLOS, start=1):
        cortes = train[train.patient_id == pid].sort_values("slice_index")
        if cortes.empty:
            sys.exit(f"ERROR: la paciente del ejemplo {n} no está en train")
        if set(cortes.fold) != {FOLD_ESPERADO}:
            sys.exit(f"ERROR: la paciente del ejemplo {n} no es del fold {FOLD_ESPERADO}")
        fila = cortes.iloc[len(cortes) // 2]
        clave = f"ejemplo_{n}"
        carpeta = DIR_EJEMPLOS / clave
        carpeta.mkdir(parents=True, exist_ok=True)
        for fase, ruta in zip(FASES, (fila.path_pre, fila.path_early, fila.path_late)):
            original = leer_fase(RAIZ_DATOS / ruta, fase)            # uint8 (256, 256), modo L
            destino = carpeta / f"{clave}_{fase}.png"
            Image.fromarray(original).save(destino, format="PNG", optimize=True)   # uint8 2D -> modo L
            if not np.array_equal(leer_fase(destino, fase), original):
                sys.exit(f"ERROR: {destino.name} no reproduce los píxeles del dataset")
        caja = [round(float(v), 1) for v in caja_en_png(patients.loc[pid])]
        indice.append({"clave": clave, **info,
                       "corte": f"corte central ({len(cortes) // 2 + 1} de {len(cortes)} de la paciente)",
                       "caja_tumor": caja,
                       "punto_sugerido": [int(round((caja[0] + caja[1]) / 2)), int(round((caja[2] + caja[3]) / 2))],
                       "variables": variables_paciente(patients.loc[pid])})
        if n == EJEMPLO_CON_PILA:
            guardar_pila(cortes, patients.loc[pid], carpeta / "pila.npz")
            indice[-1]["pila"] = f"pila.npz: los {len(cortes)} cortes de la paciente de este ejemplo"
        print(f"{clave}: {info['cohorte']}, {indice[-1]['corte']}, caja {caja}")
    (DIR_EJEMPLOS / "indice.json").write_text(
        json.dumps({"licencia": LICENCIA, "ejemplos": indice}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Ejemplos en {DIR_EJEMPLOS.relative_to(RAIZ_REPO).as_posix()}/ (píxeles idénticos al dataset)")


if __name__ == "__main__":
    main()
