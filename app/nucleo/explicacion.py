"""Ayudas para interpretar un corte: curva cinética, superficie de realce y Grad-CAM.

Curva cinética. Media de PRE, EARLY y LATE en una ventana cuadrada alrededor de
un punto. Se clasifica como en la resonancia de mama (curvas de tipo I, II y
III), con el cambio de la fase tardía respecto a la temprana:
    ascendente (persistente)   LATE más de un 10 % por encima de EARLY
    meseta                     LATE dentro de ±10 % de EARLY
    lavado                     LATE más de un 10 % por debajo de EARLY
Si EARLY apenas supera a PRE (menos de 0,05) la zona no tiene realce apreciable.
Las intensidades del dataset están normalizadas por volumen (ventana común a las
tres fases), así que los porcentajes son orientativos.

Grad-CAM (Selvaraju et al., 2017) sobre la salida del último bloque
convolucional, tras la ReLU y antes del MaxPool: los gradientes del logit (o de
su opuesto, para «no pCR») respecto a esos mapas, promediados por canal, pesan
cada mapa; se suma, se aplica ReLU, se interpola al tamaño de la entrada y se
escala a [0, 1]. El recorrido hacia delante se hace a mano, con los mismos
módulos del modelo, para no registrar hooks en el modelo que comparten todas las
sesiones de la app; el logit resultante debe coincidir con el de la inferencia.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .paquete import PaqueteModelo

UMBRAL_TEJIDO = 0.1
CAMBIO_TARDIO = 0.10            # ±10 % de EARLY
REALCE_MINIMO = 0.05            # EARLY - PRE por debajo de esto: sin realce apreciable
INTENSIDAD_MINIMA = 0.02        # para no dividir por casi cero


def media_local(imagen: np.ndarray, radio: int) -> np.ndarray:
    """Media en una ventana (2r+1) x (2r+1), con los bordes replicados (imagen integral)."""
    if radio <= 0:
        return imagen.astype(np.float64)
    a = np.pad(imagen.astype(np.float64), radio, mode="edge")
    integral = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * radio + 1
    suma = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return suma / (k * k)


def punto_maximo_realce(x: np.ndarray, radio: int = 2) -> tuple[int, int]:
    """(fila, columna) del máximo del realce medio en una ventana, dentro del tejido."""
    realce = media_local(x[1] - x[0], radio)
    tejido = (x[0] > UMBRAL_TEJIDO) | (x[1] > UMBRAL_TEJIDO)
    if not tejido.any():
        tejido = np.ones_like(tejido)
    realce = np.where(tejido, realce, -np.inf)
    fila, columna = np.unravel_index(int(np.argmax(realce)), realce.shape)
    return int(fila), int(columna)


@dataclass
class CurvaCinetica:
    fila: int
    columna: int
    radio: int
    pre: float
    early: float
    late: float
    realce_inicial: float | None       # (EARLY - PRE) / PRE
    cambio_tardio: float | None        # (LATE - EARLY) / EARLY
    tipo: str                          # ascendente, meseta, lavado o sin realce apreciable


def curva_cinetica(x: np.ndarray, fila: int, columna: int, radio: int = 2) -> CurvaCinetica:
    alto, ancho = x.shape[1:]
    fila, columna = int(np.clip(fila, 0, alto - 1)), int(np.clip(columna, 0, ancho - 1))
    f0, f1 = max(fila - radio, 0), min(fila + radio + 1, alto)
    c0, c1 = max(columna - radio, 0), min(columna + radio + 1, ancho)
    pre, early, late = (float(x[k, f0:f1, c0:c1].mean()) for k in range(3))
    inicial = (early - pre) / pre if pre > INTENSIDAD_MINIMA else None
    tardio = (late - early) / early if early > INTENSIDAD_MINIMA else None
    if early - pre < REALCE_MINIMO or tardio is None:
        tipo = "sin realce apreciable"
    elif tardio > CAMBIO_TARDIO:
        tipo = "ascendente"
    elif tardio < -CAMBIO_TARDIO:
        tipo = "lavado"
    else:
        tipo = "meseta"
    return CurvaCinetica(fila, columna, radio, pre, early, late, inicial, tardio, tipo)


def reducir(imagen: np.ndarray, factor: int) -> np.ndarray:
    """Media por bloques factor x factor (para la superficie 3D)."""
    if factor <= 1:
        return imagen
    alto, ancho = (s - s % factor for s in imagen.shape)
    return imagen[:alto, :ancho].reshape(alto // factor, factor, ancho // factor, factor).mean(axis=(1, 3))


@dataclass
class MapaAtencion:
    mapa: np.ndarray                   # (H, W) en [0, 1], del tamaño de la entrada de la red
    resolucion: tuple[int, int]        # tamaño del mapa del último bloque antes de interpolar
    logit: float                       # logit del recorrido manual (= el de la inferencia)
    hacia_pcr: bool


def grad_cam(paquete: PaqueteModelo, entrada: np.ndarray, hacia_pcr: bool = True) -> MapaAtencion:
    modelo = paquete.modelo
    modelo.eval()
    x = torch.from_numpy(np.ascontiguousarray(entrada, dtype=np.float32)).unsqueeze(0).to(paquete.dispositivo)
    x.requires_grad_(True)                 # para que exista el grafo: los pesos no tienen gradiente
    with torch.enable_grad():
        h = modelo.entrada(x)
        bloques = list(modelo.extractor)
        for bloque in bloques[:-1]:
            h = bloque(h)
        conv, normalizacion, relu, pool = bloques[-1]
        activaciones = relu(normalizacion(conv(h)))
        activaciones.retain_grad()
        logit = modelo.cabeza(pool(activaciones)).squeeze()
        (logit if hacia_pcr else -logit).backward()
    pesos = activaciones.grad.mean(dim=(2, 3), keepdim=True)
    mapa = torch.relu((pesos * activaciones.detach()).sum(dim=1, keepdim=True))
    resolucion = tuple(mapa.shape[-2:])
    mapa = F.interpolate(mapa, size=entrada.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    maximo = float(mapa.max())
    mapa = (mapa / maximo if maximo > 0 else mapa).detach().cpu().numpy()
    return MapaAtencion(mapa=mapa, resolucion=resolucion, logit=float(logit.detach()), hacia_pcr=hacia_pcr)
