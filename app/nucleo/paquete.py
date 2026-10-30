"""Carga del paquete del modelo (modelo_app/modelo.pt y modelo.json).

Los pesos (modelo.pt) no están en el repositorio: están en Hugging Face
(REPO_HF, en la revisión fijada REVISION_HF). Si modelo_app/modelo.pt no existe,
se descarga de ahí (solo biblioteca estándar, con tiempo de espera), se comprueba
su SHA-256 contra el de modelo.json ANTES de guardarlo y se deja en
modelo_app/modelo.pt (o, si no se puede escribir, en una carpeta temporal). Si
el SHA-256 no coincide o Hugging Face no responde, no se carga nada y se explica
por qué. modelo.json y fusion.json sí están en el repositorio.

Antes de usar los pesos comprueba que el SHA-256 de modelo.pt es el de
modelo.json, reconstruye la arquitectura con models.cnn_base.crear_modelo (la
misma función que train.py), carga el state_dict en modo estricto, comprueba el
número de parámetros y deja el modelo en model.eval() y sin gradientes, en CPU.
La app lo llama desde una función con st.cache_resource: se descarga una vez.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

from . import rutas
from models.cnn_base import contar_parametros, crear_modelo  # noqa: E402  (rutas añade la raíz)


REPO_HF = "AlvaroGlez/pcr-cnn-breastdcedl"
# Commit del repositorio de Hugging Face (https://huggingface.co/api/models/<REPO_HF>, campo «sha»):
# fijarlo impide que un cambio posterior en Hugging Face cambie el modelo de la app sin avisar.
REVISION_HF = "fcf55eeca9ac944575fc8623da279efa7482a826"
URL_HF = "https://huggingface.co/{repo}/resolve/{revision}/{fichero}"
ESPERA_DESCARGA_S = 60
DIR_TEMPORAL = Path(tempfile.gettempdir()) / "pcr_cnn_breastdcedl"


class ErrorPaquete(Exception):
    """El paquete del modelo falta, está incompleto o no coincide con su SHA-256."""


@dataclass(frozen=True)
class PaqueteModelo:
    modelo: torch.nn.Module
    meta: dict
    sha256: str
    dispositivo: torch.device

    @property
    def umbral(self) -> float:
        return float(self.meta["umbral"]["valor"])


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar_pesos(destino: Path, esperado: str, url: str | None = None,
                    espera: float = ESPERA_DESCARGA_S) -> Path:
    """Descarga los pesos de Hugging Face, comprueba su SHA-256 contra `esperado` antes de
    guardarlos y los deja en `destino` (o en DIR_TEMPORAL si no se puede escribir ahí).
    Devuelve la ruta final. Si algo falla, ErrorPaquete con un mensaje en español y sin
    dejar ningún fichero."""
    url = url or URL_HF.format(repo=REPO_HF, revision=REVISION_HF, fichero=destino.name)
    try:
        with urllib.request.urlopen(url, timeout=espera) as respuesta:
            datos = respuesta.read()
    except urllib.error.HTTPError as e:
        raise ErrorPaquete(f"Hugging Face respondió con un error al descargar los pesos del modelo (HTTP {e.code}). "
                           "Vuelva a cargar la página en unos minutos.") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        motivo = getattr(e, "reason", e)
        raise ErrorPaquete(f"No se pudieron descargar los pesos del modelo de Hugging Face ({motivo}). Compruebe la "
                           "conexión y vuelva a cargar la página en unos minutos.") from e
    obtenido = hashlib.sha256(datos).hexdigest()
    if obtenido != esperado:
        raise ErrorPaquete(f"Los pesos descargados de Hugging Face (SHA-256 {obtenido[:12]}…) no coinciden con los de "
                           f"modelo_app/modelo.json ({esperado[:12]}…): no se carga el modelo. Los pesos del "
                           f"repositorio {REPO_HF} no son los exportados.")
    for carpeta in (destino.parent, DIR_TEMPORAL):
        final = carpeta / destino.name
        parcial = final.with_name(final.name + ".descarga")
        try:
            carpeta.mkdir(parents=True, exist_ok=True)
            parcial.write_bytes(datos)
            os.replace(parcial, final)                     # el fichero aparece completo o no aparece
            return final
        except OSError:
            parcial.unlink(missing_ok=True)
    raise ErrorPaquete("Se descargaron los pesos del modelo, pero no se pudieron guardar en disco.")


def _pesos(ruta_pt: Path, esperado: str) -> Path:
    """La ruta de los pesos: el fichero local; si no está, la copia temporal de una descarga
    anterior con el SHA-256 esperado; si tampoco, los descarga de Hugging Face."""
    if ruta_pt.exists():
        return ruta_pt
    temporal = DIR_TEMPORAL / ruta_pt.name
    if temporal.exists() and sha256(temporal) == esperado:
        return temporal
    return descargar_pesos(ruta_pt, esperado)


def cargar_paquete(directorio: Path = rutas.DIR_MODELO, dispositivo: str = "cpu") -> PaqueteModelo:
    ruta_json = directorio / "modelo.json"
    if not ruta_json.exists():
        raise ErrorPaquete("No se encuentra modelo_app/modelo.json: falta exportar el modelo "
                           "(python exportar_modelo.py <ejecución>).")
    try:
        meta = json.loads(ruta_json.read_text(encoding="utf-8"))
        ruta_pt = directorio / meta["modelo"]["fichero"]
        esperado = meta["modelo"]["sha256"]
        bloques, realce = int(meta["modelo"]["bloques"]), bool(meta["modelo"]["realce"])
        n_esperado = int(meta["modelo"]["n_parametros"])
        float(meta["umbral"]["valor"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        raise ErrorPaquete(f"modelo_app/modelo.json está incompleto o dañado ({type(e).__name__}: {e}).") from e
    ruta_pt = _pesos(ruta_pt, esperado)
    obtenido = sha256(ruta_pt)
    if obtenido != esperado:
        raise ErrorPaquete(f"El SHA-256 de modelo_app/{ruta_pt.name} ({obtenido[:12]}…) no coincide con el de "
                           f"modelo.json ({esperado[:12]}…): los pesos no son los exportados.")

    disp = torch.device(dispositivo)
    modelo = crear_modelo(bloques, realce, float(meta["modelo"].get("dropout", 0.3)))  # sin efecto en eval()
    estado = torch.load(ruta_pt, map_location=disp, weights_only=True)
    modelo.load_state_dict(estado)                            # estricto
    if contar_parametros(modelo) != n_esperado:
        raise ErrorPaquete(f"El modelo tiene {contar_parametros(modelo)} parámetros y modelo.json dice {n_esperado}.")
    modelo.to(disp).eval()
    for p in modelo.parameters():
        p.requires_grad_(False)
    return PaqueteModelo(modelo=modelo, meta=meta, sha256=obtenido, dispositivo=disp)
