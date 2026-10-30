"""La arquitectura del modelo exportado, capa a capa, medida sobre el propio modelo.

Construye un modelo nuevo con models.cnn_base.crear_modelo (no el de la app, que
comparten todas las sesiones) y pasa una entrada de ceros con hooks para leer la
forma de salida de cada capa. Campo receptivo de cada capa, en píxeles de la
entrada: con convoluciones 3x3 de paso 1 y MaxPool 2x2 de paso 2,
    RF_salida = RF_entrada + (k - 1) * salto,    salto_salida = salto * paso.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from . import rutas  # noqa: F401  (añade la raíz al sys.path)
from models.cnn_base import Realce, crear_modelo  # noqa: E402


@dataclass
class Capa:
    etapa: str                 # "Entrada", "Bloque 1", ..., "Cabeza"
    nombre: str                # p. ej. "Conv2d 3×3"
    salida: tuple[int, ...]    # (canales, alto, ancho) o (características,)
    parametros: int
    campo_receptivo: int | None
    salto: int | None


def _nombre(capa: nn.Module) -> str:
    if isinstance(capa, nn.Conv2d):
        return f"Conv2d {capa.kernel_size[0]}×{capa.kernel_size[1]}, {capa.in_channels}→{capa.out_channels}"
    if isinstance(capa, nn.BatchNorm2d):
        return f"BatchNorm2d ({capa.num_features})"
    if isinstance(capa, nn.ReLU):
        return "ReLU"
    if isinstance(capa, nn.MaxPool2d):
        return "MaxPool2d 2×2"
    if isinstance(capa, nn.AdaptiveAvgPool2d):
        return "Promedio global (GAP)"
    if isinstance(capa, nn.Flatten):
        return "Aplanado"
    if isinstance(capa, nn.Dropout):
        return f"Dropout (p = {capa.p:g})"
    if isinstance(capa, nn.Linear):
        return f"Lineal {capa.in_features}→{capa.out_features}"
    if isinstance(capa, Realce):
        return "Realce: PRE, EARLY−PRE, LATE−EARLY"
    return type(capa).__name__


def capas(meta: dict) -> list[Capa]:
    modelo = crear_modelo(int(meta["modelo"]["bloques"]), bool(meta["modelo"]["realce"]),
                          float(meta["modelo"].get("dropout", 0.3))).eval()
    entrada = tuple(meta["modelo"]["entrada"])
    orden: list[tuple[str, nn.Module]] = []
    if isinstance(modelo.entrada, Realce):
        orden.append(("Entrada", modelo.entrada))
    for i, bloque in enumerate(modelo.extractor, start=1):
        orden += [(f"Bloque {i}", capa) for capa in bloque]
    orden += [("Cabeza", capa) for capa in modelo.cabeza]

    formas: dict[int, tuple] = {}
    ganchos = [capa.register_forward_hook(lambda m, e, s, i=i: formas.__setitem__(i, tuple(s.shape[1:])))
               for i, (_, capa) in enumerate(orden)]
    with torch.no_grad():
        modelo(torch.zeros(1, *entrada))
    for g in ganchos:
        g.remove()

    resultado = [Capa("Entrada", "Imagen PRE, EARLY, LATE", entrada, 0, 1, 1)]
    rf, salto = 1, 1
    for i, (etapa, capa) in enumerate(orden):
        if isinstance(capa, nn.Conv2d):
            rf += (capa.kernel_size[0] - 1) * salto
            salto *= capa.stride[0]
        elif isinstance(capa, nn.MaxPool2d):
            k = capa.kernel_size if isinstance(capa.kernel_size, int) else capa.kernel_size[0]
            s = capa.stride if isinstance(capa.stride, int) else capa.stride[0]
            rf += (k - 1) * salto
            salto *= s
        espacial = etapa != "Cabeza"
        resultado.append(Capa(etapa, _nombre(capa), formas[i], sum(p.numel() for p in capa.parameters()),
                              rf if espacial else None, salto if espacial else None))
    return resultado


def resumen_bloques(lista: list[Capa]) -> list[dict]:
    """Una fila por etapa: salida, parámetros y campo receptivo al final de la etapa."""
    filas: dict[str, dict] = {}
    for c in lista:
        f = filas.setdefault(c.etapa, {"etapa": c.etapa, "capas": [], "parametros": 0})
        f["capas"].append(c.nombre)
        f["parametros"] += c.parametros
        f["salida"] = c.salida
        if c.campo_receptivo is not None:
            f["campo_receptivo"] = c.campo_receptivo
            f["salto"] = c.salto
    return list(filas.values())
