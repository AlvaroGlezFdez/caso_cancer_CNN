"""Preprocesado único del proyecto: lo usan el entrenamiento, la evaluación y la app.

Un corte son 3 PNG en escala de grises de 8 bits y 256x256, uno por fase DCE.
Se apilan en el orden PRE, EARLY, LATE y se dividen entre 255. Nada más: ni
normalización ImageNet, ni estandarización por canal.

Opcionalmente (train.py --recorte-caja) recorta un cuadrado centrado en la caja
del tumor (caja_tumor.caja_en_png), de lado el mayor de la caja más un 25 %
(mínimo 64 px), y lo reescala a 128x128, igual en las tres fases. Lo que cae
fuera del PNG se rellena con ceros. La caja es un dato de la paciente, el mismo
que recibirá la app.

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
LADO_RECORTE = 128     # lado del recorte reescalado
MARGEN_RECORTE = 1.25  # lado del cuadrado = lado mayor de la caja x 1,25
LADO_MINIMO = 64       # lado mínimo del cuadrado, en px del PNG


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


def cuadrado_recorte(caja) -> tuple[float, float, float]:
    """caja (fila0, fila1, col0, col1) en el PNG -> (fila_centro, col_centro, lado)."""
    f0, f1, c0, c1 = caja
    lado = max(MARGEN_RECORTE * max(f1 - f0, c1 - c0), LADO_MINIMO)
    return (f0 + f1) / 2, (c0 + c1) / 2, lado


def recortar(x: np.ndarray, caja) -> np.ndarray:
    """(3, 256, 256) -> (3, 128, 128): el cuadrado de cuadrado_recorte(caja),
    reescalado con interpolación bilineal de PIL (con antialiasing al reducir),
    con la misma geometría en las tres fases y ceros fuera de la imagen."""
    fc, cc, lado = cuadrado_recorte(caja)
    # Relleno de ceros suficiente para que el cuadrado quede dentro.
    relleno = int(np.ceil(max(lado / 2 - min(fc, cc), fc + lado / 2 - TAMANO[0],
                              cc + lado / 2 - TAMANO[1], 0))) + 1
    caja_pil = (cc - lado / 2 + relleno, fc - lado / 2 + relleno,   # (izq, arriba, der, abajo)
                cc + lado / 2 + relleno, fc + lado / 2 + relleno)
    salida = [np.asarray(Image.fromarray(np.pad(fase, relleno))
                         .resize((LADO_RECORTE, LADO_RECORTE), Image.BILINEAR, box=caja_pil))
              for fase in x]
    return np.stack(salida).astype(np.float32)


def preprocesar(pre, early, late, caja=None) -> np.ndarray:
    """3 PNG (rutas o ficheros abiertos) -> (3, 256, 256) float32 en [0, 1].

    El orden de los argumentos es el orden de los canales que recibe la red.
    Con `caja` (la de caja_tumor.caja_en_png de la paciente) devuelve el recorte
    (3, 128, 128) de recortar().
    """
    fases = [leer_fase(f, nombre) for f, nombre in zip((pre, early, late), FASES)]
    x = np.stack(fases).astype(np.float32) / 255.0
    return x if caja is None else recortar(x, caja)
