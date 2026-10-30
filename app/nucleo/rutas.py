"""Carpetas del repositorio. Importarlo añade la raíz al sys.path, para que la app
use preprocesado.py y models/cnn_base.py del proyecto en lugar de copias."""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ_APP = Path(__file__).resolve().parents[1]
RAIZ_REPO = RAIZ_APP.parent
DIR_MODELO = RAIZ_REPO / "modelo_app"
DIR_EJEMPLOS = RAIZ_APP / "ejemplos"
DIR_EXPERIMENTOS = RAIZ_REPO / "experimentos"

if str(RAIZ_REPO) not in sys.path:
    sys.path.insert(0, str(RAIZ_REPO))
