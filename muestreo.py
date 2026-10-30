"""Qué cortes de cada paciente se usan para entrenar (train.py --cortes-centrales y
--cortes-por-paciente). La validación usa siempre todos los cortes.

Los cortes de una paciente están muy correlacionados (son vecinos del mismo
tumor) y en los extremos del tumor la lesión casi no se ve (auditoría visual).
Inspiración: estrategias de compañeros para datos escasos y correlacionados
(pocos cortes por paciente) y el repositorio de los autores del dataset
(naomifridman/BreastDCEDL: solo los cortes centrales del tumor).

    cortes_centrales(filas, k)        los k cortes de cada paciente cuyo slice_index
                                      está más cerca de la mediana de los suyos
                                      (empates: el de slice_index menor)
    CortesPorPaciente(filas, n, s)    sampler: en cada época, n cortes al azar de
                                      cada paciente (todos si tiene menos), en orden
                                      aleatorio; distintos en cada época y
                                      reproducibles (generador propio con semilla s)

Si se usan los dos, primero se filtran los k centrales y después se eligen n al
azar entre ellos.

Comprobación, desde la raíz del repositorio (necesita breastdcedl/):

    python muestreo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


def cortes_centrales(filas: pd.DataFrame, k: int) -> pd.DataFrame:
    """Las filas de los k cortes centrales de cada paciente, en el orden original."""
    mediana = filas.groupby("patient_id").slice_index.transform("median")
    orden = (filas.assign(_distancia=(filas.slice_index - mediana).abs())
                  .sort_values(["patient_id", "_distancia", "slice_index"], kind="mergesort"))
    elegidos = orden.groupby("patient_id", sort=False).head(k).index
    return filas[filas.index.isin(elegidos)]


class CortesPorPaciente(Sampler[int]):
    """Índices (posiciones en `filas`) de n cortes al azar por paciente, en orden
    aleatorio, distintos en cada época. Usa su propio generador: no altera el
    generador global (dropout) ni ningún otro."""

    def __init__(self, filas: pd.DataFrame, n: int, semilla: int):
        self.grupos = [np.asarray(v) for v in
                       filas.reset_index(drop=True).groupby("patient_id", sort=True).indices.values()]
        self.n = n
        self.generador = torch.Generator().manual_seed(semilla)

    def __len__(self) -> int:
        return sum(min(self.n, len(g)) for g in self.grupos)

    def __iter__(self):
        elegidos: list[int] = []
        for g in self.grupos:
            if len(g) <= self.n:
                elegidos.extend(g.tolist())
            else:
                elegidos.extend(g[torch.randperm(len(g), generator=self.generador)[: self.n].numpy()].tolist())
        orden = torch.randperm(len(elegidos), generator=self.generador).tolist()
        return iter([elegidos[i] for i in orden])


def main():
    import sys
    from pathlib import Path

    raiz = Path(__file__).resolve().parent
    sys.path.insert(0, str(raiz / "breastdcedl"))
    import utils_caso as uc

    sys.stdout.reconfigure(encoding="utf-8")
    entrenamiento, _ = uc.particion(uc.cargar_samples(raiz / "breastdcedl"), fold_val=0)
    n_pac = entrenamiento.patient_id.nunique()
    fallos = []

    def comprobar(nombre: str, bien: bool, detalle: str = ""):
        print(f"  {'OK   ' if bien else 'FALLO'} {nombre}" + (f": {detalle}" if detalle else ""))
        if not bien:
            fallos.append(nombre)

    s = CortesPorPaciente(entrenamiento, 3, 44)
    e1, e2 = list(s), list(s)
    por_pac = entrenamiento.reset_index(drop=True).patient_id.iloc[e1].value_counts()
    comprobar("--cortes-por-paciente 3: ~3 x pacientes imágenes por época", len(e1) == len(s) == len(set(e1))
              and por_pac.max() <= 3, f"{len(e1)} imágenes de {n_pac} pacientes (3 x {n_pac} = {3 * n_pac})")
    comprobar("distintos en cada época", set(e1) != set(e2), f"{len(set(e1) & set(e2))} cortes repetidos entre la "
              "época 1 y la 2")
    comprobar("reproducibles con la semilla", list(CortesPorPaciente(entrenamiento, 3, 44)) == e1)

    c = cortes_centrales(entrenamiento, 4)
    # Ningún corte descartado está más cerca de la mediana que uno elegido. (Con slice_index
    # consecutivos son los 4 del medio; 37 pacientes tienen huecos y entonces cuenta la distancia.)
    ok, consecutivos = True, True
    for pid, g in entrenamiento.groupby("patient_id"):
        d = (g.slice_index - g.slice_index.median()).abs()
        elegido = g.sample_id.isin(c.sample_id)
        ok &= elegido.sum() == min(4, len(g)) and (elegido.all() or d[elegido].max() <= d[~elegido].min())
        orden = sorted(g.slice_index)
        if len(orden) % 2 == 0 and len(orden) >= 4 and orden[-1] - orden[0] == len(orden) - 1:
            i = (len(orden) - 4) // 2
            consecutivos &= set(g.slice_index[elegido]) == set(orden[i:i + 4])
    comprobar("--cortes-centrales 4: los 4 cortes más cercanos a la mediana de cada paciente", ok and consecutivos,
              f"{len(c)} cortes de {c.patient_id.nunique()} pacientes; con cortes consecutivos, los 4 del medio")
    ambos = CortesPorPaciente(c, 3, 44)
    elegidos = set(c.reset_index(drop=True).sample_id.iloc[list(ambos)])
    comprobar("las dos juntas: 3 al azar entre los 4 centrales", elegidos <= set(c.sample_id)
              and len(elegidos) == len(ambos), f"{len(ambos)} imágenes por época")
    if fallos:
        raise SystemExit("FALLAN: " + "; ".join(fallos))
    print("Todas las comprobaciones OK.")


if __name__ == "__main__":
    main()
