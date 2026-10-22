"""Línea base: CNN 2D desde cero que predice pCR a partir de un corte DCE-MRI.

Entrada (B, 3, 256, 256): canales PRE, EARLY, LATE, en ese orden. Son fases
temporales, no colores. Salida (B, 1): un único logit; P(pCR=1) = sigmoid(logit).
Se entrena con BCEWithLogitsLoss.

    4 bloques  Conv2d 3x3 (padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
               filtros 3->16->32->64->128, resolución 256->128->64->32->16
    cabeza     AdaptiveAvgPool2d(1) -> Dropout(0.3) -> Linear(128, 1)

98.049 parámetros entrenables. Inicialización: la de PyTorch por defecto.

Variantes (cada una cambia una sola cosa respecto a la línea base):
  bloques=5    un quinto bloque idéntico, de 128 a 128 filtros, antes del GAP
               (resolución 16->8). 245.889 parámetros.
  realce=True  dentro del modelo, antes del primer bloque, la entrada pasa a ser
               PRE, EARLY-PRE y LATE-EARLY (en ese orden) en lugar de PRE, EARLY
               y LATE. Sin parámetros: siguen siendo 98.049. La entrada del
               modelo y el preprocesado no cambian.

Comprobación, desde la raíz del repositorio:

    python -m models.cnn_base
"""

from __future__ import annotations

import torch
from torch import nn

FILTROS_POR_BLOQUES = {4: (16, 32, 64, 128), 5: (16, 32, 64, 128, 128)}
FILTROS = FILTROS_POR_BLOQUES[4]
PARAMETROS_ESPERADOS = {4: 98_049, 5: 245_889}


def bloque(entrada: int, salida: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel_size=3, padding=1),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Realce(nn.Module):
    """(PRE, EARLY, LATE) -> (PRE, EARLY-PRE, LATE-EARLY). Sin parámetros."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre, early, late = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return torch.cat([pre, early - pre, late - early], dim=1)


class CNNBase(nn.Module):
    def __init__(self, canales: int = 3, filtros: tuple[int, ...] = FILTROS, dropout: float = 0.3,
                 realce: bool = False):
        super().__init__()
        # Identity y Realce no tienen parámetros ni consumen números aleatorios: la
        # inicialización y los nombres del state_dict de la línea base no cambian.
        self.entrada = Realce() if realce else nn.Identity()
        capas, entrada = [], canales
        for salida in filtros:
            capas.append(bloque(entrada, salida))
            entrada = salida
        self.extractor = nn.Sequential(*capas)
        self.cabeza = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(entrada, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.extractor(self.entrada(x)))     # (B, 1) logits


def crear_modelo(bloques: int = 4, realce: bool = False) -> CNNBase:
    """La línea base (bloques=4, realce=False) o una de sus variantes."""
    if bloques not in FILTROS_POR_BLOQUES:
        raise ValueError(f"bloques debe ser uno de {sorted(FILTROS_POR_BLOQUES)}")
    return CNNBase(filtros=FILTROS_POR_BLOQUES[bloques], realce=realce)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(crear_modelo())
    x = torch.rand(2, 3, 256, 256)
    for bloques, realce in ((4, False), (5, False), (4, True)):
        modelo = crear_modelo(bloques, realce).eval()
        n = contar_parametros(modelo)
        salida = modelo(x)
        print(f"bloques={bloques} realce={realce}: {n:_} parámetros entrenables, ".replace("_", ".")
              + f"salida {tuple(salida.shape)}")
        assert n == PARAMETROS_ESPERADOS[bloques], f"se esperaban {PARAMETROS_ESPERADOS[bloques]} parámetros, hay {n}"
        assert salida.shape == (2, 1)
    r = Realce()(x)
    assert torch.equal(r[:, 0], x[:, 0]) and torch.equal(r[:, 1], x[:, 1] - x[:, 0])
    assert torch.equal(r[:, 2], x[:, 2] - x[:, 1])
    print("Realce: canales PRE, EARLY-PRE, LATE-EARLY comprobados")
