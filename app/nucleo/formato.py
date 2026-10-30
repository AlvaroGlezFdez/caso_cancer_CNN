"""Números y textos en formato español (coma decimal), para todos los textos de la app."""

from __future__ import annotations

import re

import pandas as pd


def num(x: float, decimales: int = 3) -> str:
    return f"{x:.{decimales}f}".replace(".", ",")


def pct(x: float, decimales: int = 0) -> str:
    return f"{100 * x:.{decimales}f}".replace(".", ",") + " %"


def entero(n: int) -> str:
    """12345 -> «12.345»."""
    return f"{int(n):,}".replace(",", ".")


def numero_g(x: float) -> str:
    """Formato corto (como :g) con coma decimal: 0.001 -> «0,001», 1e-4 -> «0,0001»."""
    texto = f"{x:.10f}".rstrip("0").rstrip(".") if x != 0 else "0"
    return texto.replace(".", ",")


def mayuscula(texto: str) -> str:
    """Primera letra en mayúscula sin tocar el resto (str.capitalize pasaría
    «AUC por paciente» a «Auc por paciente»)."""
    return texto[:1].upper() + texto[1:]


def nombre_corto(ejecucion: str) -> str:
    """«20261012_142807_baseline_30ep_cpu» -> «baseline_30ep_cpu (12-10-2026)»."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_\d{6}_(.+)$", ejecucion)
    return f"{m.group(4)} ({m.group(3)}-{m.group(2)}-{m.group(1)})" if m else ejecucion


def tabla_texto(df: pd.DataFrame, decimales: dict[str, int]) -> pd.DataFrame:
    """Copia con las columnas indicadas como texto en formato español; los
    valores que faltan quedan como «—»."""
    salida = df.copy()
    for columna, d in decimales.items():
        salida[columna] = [("—" if pd.isna(v) else num(float(v), d)) for v in df[columna]]
    return salida
