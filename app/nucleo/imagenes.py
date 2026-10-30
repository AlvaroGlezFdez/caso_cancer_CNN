"""Colores e imágenes de la app: las mismas escalas fijas de la auditoría visual.

    Fases          gris, de 0 a 1 (los 0-255 originales)
    Realce         EARLY - PRE con RdBu_r de -0,5 a 0,5: azul, se oscurece con el
                   contraste; blanco, no cambia; rojo, se ilumina
    Atención       Grad-CAM con la escala «inferno», de 0 a 1, superpuesto al gris

Sin Streamlit: lo usan las páginas y el informe descargable.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

ESCALA_REALCE = 0.5
COLORES_REALCE = ("#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
                  "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f")
COLORES_ATENCION = ("#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446",
                    "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4")                       # inferno


def tabla_colores(colores, n: int = 256) -> np.ndarray:
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in colores], dtype=float)
    x = np.linspace(0, 1, len(colores))
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, x, rgb[:, k]) for k in range(3)], axis=1).round().astype(np.uint8)


TABLA_REALCE = tabla_colores(COLORES_REALCE)
TABLA_ATENCION = tabla_colores(COLORES_ATENCION)


def escala_plotly(colores) -> list[list]:
    """Lista de colores -> colorscale de plotly, repartidos de 0 a 1."""
    return [[i / (len(colores) - 1), c] for i, c in enumerate(colores)]


def a_uint8(fase: np.ndarray) -> np.ndarray:
    """Fase preprocesada en [0, 1] -> los 0-255 originales, exactos."""
    return np.round(np.clip(fase, 0, 1) * 255).astype(np.uint8)


def colorear_realce(realce: np.ndarray) -> np.ndarray:
    """EARLY - PRE (float) -> RGB uint8 con la escala fija de -0,5 a 0,5."""
    t = np.clip((realce + ESCALA_REALCE) / (2 * ESCALA_REALCE), 0, 1)
    return TABLA_REALCE[np.round(t * 255).astype(int)]


def superponer_atencion(fondo: np.ndarray, mapa: np.ndarray, opacidad: float) -> np.ndarray:
    """Fondo en [0, 1] (gris) y mapa en [0, 1] -> RGB uint8. Cada píxel del mapa
    tapa el fondo en proporción a su valor, para que lo irrelevante no se tiña."""
    gris = np.repeat(a_uint8(fondo)[..., None], 3, axis=2).astype(float)
    color = TABLA_ATENCION[np.round(np.clip(mapa, 0, 1) * 255).astype(int)].astype(float)
    alfa = (opacidad * np.clip(mapa, 0, 1))[..., None]
    return np.round(gris * (1 - alfa) + color * alfa).astype(np.uint8)


def png(arr: np.ndarray) -> bytes:
    """Array uint8 (H, W) o (H, W, 3) -> PNG en memoria."""
    salida = io.BytesIO()
    Image.fromarray(arr).save(salida, format="PNG")
    return salida.getvalue()
