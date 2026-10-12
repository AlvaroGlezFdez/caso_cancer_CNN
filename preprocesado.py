"""Preprocesado único del proyecto: lo usan el entrenamiento, la evaluación y la app.

Un corte son 3 PNG en escala de grises de 8 bits y 256x256, uno por fase DCE.
Se apilan en el orden PRE, EARLY, LATE y se dividen entre 255. Nada más: ni
normalización ImageNet, ni estandarización por canal, ni recortes.

Da exactamente el mismo resultado que utils_caso.cargar_imagen (train.py lo
comprueba al arrancar), pero acepta también ficheros abiertos (lo que sube la
app) y rechaza con un mensaje claro lo que no sea un PNG de ese formato, en vez
de convertirlo en silencio. No depende de utils_caso, que no se sube a git.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, UnidentifiedImageError

FASES = ("PRE", "EARLY", "LATE")
TAMANO = (256, 256)


def leer_fase(fuente, fase: str = "") -> np.ndarray:
    """Lee un PNG (ruta o fichero abierto) y devuelve (256, 256) en uint8."""
    try:
        with Image.open(fuente) as png:
            formato, modo, tamano = png.format, png.mode, png.size
            arr = np.asarray(png)
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"{fase}: no se puede leer como imagen ({e})") from e
    if formato != "PNG":
        raise ValueError(f"{fase}: se esperaba un PNG y es {formato}")
    if modo != "L":
        raise ValueError(f"{fase}: se esperaba escala de grises de 8 bits (modo L) y es modo {modo}")
    if tamano != TAMANO:
        raise ValueError(f"{fase}: se esperaba {TAMANO[0]}x{TAMANO[1]} y es {tamano[0]}x{tamano[1]}")
    return arr


def preprocesar(pre, early, late) -> np.ndarray:
    """3 PNG (rutas o ficheros abiertos) -> (3, 256, 256) float32 en [0, 1].

    El orden de los argumentos es el orden de los canales que recibe la red.
    """
    fases = [leer_fase(f, nombre) for f, nombre in zip((pre, early, late), FASES)]
    return np.stack(fases).astype(np.float32) / 255.0
