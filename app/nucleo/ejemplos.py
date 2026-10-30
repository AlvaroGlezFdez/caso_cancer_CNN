"""Los 3 cortes de ejemplo de app/ejemplos/ (preparar_ejemplos.py), sin etiqueta ni id.

Solo se leen los ficheros que figuran en indice.json: el usuario elige un
ejemplo de la lista, nunca una ruta.
"""

from __future__ import annotations

import json
import re

from . import rutas
from preprocesado import FASES  # noqa: E402

PATRON_CLAVE = re.compile(r"^ejemplo_\d+$")


def indice() -> dict:
    """{"licencia": str, "ejemplos": [{"clave", "titulo", "cohorte", "descripcion", "corte"}, ...]}"""
    return json.loads((rutas.DIR_EJEMPLOS / "indice.json").read_text(encoding="utf-8"))


def cargar(clave: str) -> dict[str, tuple[str, bytes]]:
    """fase -> (nombre, contenido) de un ejemplo del índice."""
    if not PATRON_CLAVE.match(clave) or clave not in {e["clave"] for e in indice()["ejemplos"]}:
        raise ValueError(f"No existe el ejemplo «{clave}».")
    carpeta = rutas.DIR_EJEMPLOS / clave
    return {f: (f"{clave}_{f}.png", (carpeta / f"{clave}_{f}.png").read_bytes()) for f in FASES}
