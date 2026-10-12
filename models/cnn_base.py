"""Línea base: CNN 2D desde cero que predice pCR a partir de un corte DCE-MRI.

Entrada (B, 3, 256, 256): canales PRE, EARLY, LATE, en ese orden. Son fases
temporales, no colores. Salida (B, 1): un único logit; P(pCR=1) = sigmoid(logit).
Se entrena con BCEWithLogitsLoss.

    4 bloques  Conv2d 3x3 (padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
               filtros 3->16->32->64->128, resolución 256->128->64->32->16
    cabeza     AdaptiveAvgPool2d(1) -> Dropout(0.3) -> Linear(128, 1)

98.049 parámetros entrenables. Inicialización: la de PyTorch por defecto.

Comprobación, desde la raíz del repositorio:

    python -m models.cnn_base
"""

from __future__ import annotations

import torch
from torch import nn

FILTROS = (16, 32, 64, 128)
PARAMETROS_ESPERADOS = 98_049


def bloque(entrada: int, salida: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel_size=3, padding=1),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNBase(nn.Module):
    def __init__(self, canales: int = 3, filtros: tuple[int, ...] = FILTROS, dropout: float = 0.3):
        super().__init__()
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
        return self.cabeza(self.extractor(x))           # (B, 1) logits


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


if __name__ == "__main__":
    modelo = CNNBase()
    print(modelo)
    n = contar_parametros(modelo)
    salida = modelo.eval()(torch.zeros(2, 3, 256, 256))
    print(f"Parámetros entrenables: {n:,}".replace(",", "."))
    print(f"Salida para una entrada (2, 3, 256, 256): {tuple(salida.shape)}")
    assert n == PARAMETROS_ESPERADOS, f"se esperaban {PARAMETROS_ESPERADOS} parámetros, hay {n}"
    assert salida.shape == (2, 1)
