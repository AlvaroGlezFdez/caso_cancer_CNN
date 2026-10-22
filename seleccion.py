"""Criterio de selección de época: decide el best.pt y el early stopping.

Criterios (train.py --criterio):
  auc_suavizado  (por defecto) media del AUC de validación por paciente de la
                 época con el de la anterior y el de la siguiente. En los
                 extremos (la primera época y la última que se ejecuta, también
                 cuando para el early stopping) se usan solo las disponibles.
  auc            el AUC de la propia época (criterio anterior al 2026-10-22).

Con auc_suavizado el valor de la época e solo se conoce al terminar la e+1,
así que la decisión sobre la época e (¿es la mejor?, ¿una época más sin
mejorar?) se toma un paso después; train.py guarda los pesos de la época
anterior hasta entonces. La paciencia cuenta épocas cuyo valor ya está fijado.

La misma clase la usa cv_resumen.py para reconstruir, a partir de los
metricas_epoca.csv guardados, qué habría elegido cada criterio.
"""

from __future__ import annotations

import numpy as np

CRITERIOS = ("auc_suavizado", "auc")
DESCRIPCION = {
    "auc_suavizado": "media móvil del AUC de validación por paciente con la época anterior y la siguiente "
                     "(en los extremos, las disponibles)",
    "auc": "AUC de validación por paciente de la propia época",
}


def valor(aucs, i: int, criterio: str) -> float:
    """Valor del criterio para la época de índice i (0 = primera) con los AUC
    conocidos hasta ahora. -inf si no hay ningún AUC finito (validación con una
    sola clase, posible en pruebas recortadas)."""
    if criterio == "auc":
        ventana = np.asarray(aucs[i: i + 1], dtype=float)
    elif criterio == "auc_suavizado":
        ventana = np.asarray(aucs[max(0, i - 1): i + 2], dtype=float)
    else:
        raise ValueError(f"criterio desconocido: {criterio}")
    ventana = ventana[np.isfinite(ventana)]
    return float(ventana.mean()) if len(ventana) else -np.inf


class Seleccion:
    """Sigue el criterio época a época.

        sel = Seleccion("auc_suavizado", paciencia=10)
        for epoca ...:
            for e, v, mejora in sel.anadir(auc):    # épocas cuyo valor queda fijado ahora
                ...
            if sel.parar: break
        for e, v, mejora in sel.cerrar():           # al terminar: las que queden pendientes
            ...
    """

    def __init__(self, criterio: str, paciencia: int | None):
        if criterio not in CRITERIOS:
            raise ValueError(f"criterio debe ser uno de {CRITERIOS}")
        self.criterio, self.paciencia = criterio, paciencia
        self.retraso = 1 if criterio == "auc_suavizado" else 0   # épocas que hay que esperar
        self.aucs: list[float] = []
        self.valores: dict[int, float] = {}                       # época (desde 1) -> valor fijado
        self.mejor_epoca, self.mejor_valor, self.sin_mejora = None, -np.inf, 0

    def _fijar(self, epoca: int) -> tuple[int, float, bool]:
        v = valor(self.aucs, epoca - 1, self.criterio)
        self.valores[epoca] = v
        mejora = self.mejor_epoca is None or v > self.mejor_valor
        if mejora:
            self.mejor_epoca, self.mejor_valor, self.sin_mejora = epoca, v, 0
        else:
            self.sin_mejora += 1
        return epoca, v, mejora

    def anadir(self, auc: float) -> list[tuple[int, float, bool]]:
        """Registra el AUC de la época siguiente y fija las que ya se pueden fijar."""
        self.aucs.append(float(auc))
        epoca = len(self.aucs) - self.retraso
        return [self._fijar(epoca)] if epoca >= 1 else []

    def cerrar(self) -> list[tuple[int, float, bool]]:
        """Fin del entrenamiento: fija las épocas pendientes con los AUC disponibles."""
        pendientes = range(len(self.valores) + 1, len(self.aucs) + 1)
        return [self._fijar(e) for e in pendientes]

    @property
    def parar(self) -> bool:
        return bool(self.paciencia) and self.sin_mejora >= self.paciencia


def simular(aucs, criterio: str, paciencia: int | None, epocas_max: int) -> dict:
    """Recorre una serie de AUC por época como lo haría train.py.

    Devuelve la época en que habría parado, la mejor época y su valor, y
    `completa`. Si es False, la ejecución habría seguido más allá de las épocas
    de la serie (sin parar por paciencia y sin llegar a epocas_max): entonces
    no se cierra la serie, y la mejor época es provisional, la mejor de las
    épocas cuyo valor ya está fijado; `sin_mejora` dice cuántas épocas fijadas
    llevaba sin mejorar.
    """
    sel = Seleccion(criterio, paciencia)
    parada = None
    for epoca, auc in enumerate(aucs, 1):
        sel.anadir(auc)
        if sel.parar or epoca >= epocas_max:
            parada = epoca
            break
    if parada is not None:
        sel.cerrar()
    return {"epoca_parada": parada, "mejor_epoca": sel.mejor_epoca, "valor": sel.mejor_valor,
            "completa": parada is not None, "sin_mejora": sel.sin_mejora}
