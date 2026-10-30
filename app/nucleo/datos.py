"""Datos agregados del dataset para la página «Datos» y la pila de cortes del ejemplo 1.

app/datos/resumen_dataset.json lo escribe preparar_datos_app.py (solo recuentos y
medianas, sin ids ni etiquetas individuales). La pila (app/ejemplos/ejemplo_1/
pila.npz) son los cortes de la paciente del ejemplo 1, de preparar_ejemplos.py.
"""

from __future__ import annotations

import json

import numpy as np

from . import rutas

RUTA_RESUMEN = rutas.RAIZ_APP / "datos" / "resumen_dataset.json"


def resumen() -> dict:
    return json.loads(RUTA_RESUMEN.read_text(encoding="utf-8"))


def pila(clave: str = "ejemplo_1") -> dict | None:
    """{"pre", "early", "late": uint8 (n, 256, 256), "posicion": (n,), "indice_ejemplo",
    "grosor_mm", "pixel_mm"} o None si no hay pila para ese ejemplo."""
    ruta = rutas.DIR_EJEMPLOS / clave / "pila.npz"
    if not ruta.exists():
        return None
    with np.load(ruta, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
