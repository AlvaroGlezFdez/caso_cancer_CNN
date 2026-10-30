"""Del corte subido a la probabilidad de pCR, con la misma cadena que train.py:

    validacion.validar_corte  ->  preprocesado.preprocesar (opciones de modelo.json)
    ->  comprobaciones del preprocesado  ->  modelo en eval(), sin gradientes
    ->  sigmoide del logit

La probabilidad es la del corte. El umbral de modelo.json se eligió por corte
(exportar_modelo.py) y aquí se aplica al corte: positivo si prob >= umbral.

Esta es la probabilidad de la imagen sola. El modelo final combina después esta
probabilidad con las variables clínicas de la paciente (nucleo/fusion.py).
DatosPaciente lleva las variables de la paciente (nucleo/variables.py, desde un
CSV o el formulario). Un modelo entrenado con --recorte-caja o con
--recorte-centrado N necesita la caja del tumor de la paciente: sin ella no
predice (no hay recorte de recambio, como centrar en el medio de la imagen).
Con --recorte-centrado la llamada es la de train.py:
preprocesado.preprocesar(pre, early, late, caja=caja, lado_centrado=N).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field, fields

import numpy as np
import torch

from . import rutas  # noqa: F401  (añade la raíz al sys.path)
from .paquete import PaqueteModelo
from .validacion import CORTE, ERROR, OK, Informe, comprobar_preprocesado, validar_corte
from preprocesado import FASES, preprocesar, ventana_centrada  # noqa: E402

CAMPOS_CAJA = ("n_xy", "sraw", "eraw", "scol", "ecol")
# Opciones de preprocesado de modelo.json que la app sabe reproducir.
PREPROCESADO_CONOCIDO = {"funcion", "fases", "formato_entrada", "normalizacion", "recorte_caja", "recorte_centrado",
                         "recorte_centrado_detalle"}


def necesita_caja(meta: dict) -> bool:
    """¿El preprocesado del modelo recorta por la caja del tumor (--recorte-caja o --recorte-centrado)?"""
    preproc = meta.get("preprocesado", {})
    return bool(preproc.get("recorte_caja") or preproc.get("recorte_centrado"))


@dataclass
class DatosPaciente:
    """Variables de la paciente, con los nombres de patients.csv. La caja del
    tumor (n_xy, sraw, eraw, scol, ecol) basta para caja_tumor.caja_en_png; el
    resto (edad, subtipo, volumen…) irá en `clinicas` cuando se use."""
    n_xy: float | None = None
    sraw: float | None = None
    eraw: float | None = None
    scol: float | None = None
    ecol: float | None = None
    clinicas: dict = field(default_factory=dict)

    @classmethod
    def desde_fila(cls, fila: dict) -> "DatosPaciente":
        """Desde una fila de patients.csv (dict columna -> valor)."""
        nombres = {f.name for f in fields(cls)} - {"clinicas"}
        return cls(**{k: fila.get(k) for k in nombres},
                   clinicas={k: v for k, v in fila.items() if k not in nombres})

    def tiene_caja(self) -> bool:
        return all(getattr(self, c) is not None for c in CAMPOS_CAJA)

    def caja_png(self) -> tuple[float, float, float, float]:
        from caja_tumor import caja_en_png
        return caja_en_png(self)


@dataclass
class Resultado:
    probabilidad: float        # sigmoide del logit, el float32 de la red
    logit: float
    umbral: float
    clase: int                 # 1 = pCR estimada (probabilidad >= umbral)
    tiempo_inferencia_ms: float
    tiempo_total_ms: float     # validación + preprocesado + inferencia
    dispositivo: str


@dataclass
class Analisis:
    informe: Informe
    nombres: dict = field(default_factory=dict)
    imagen: np.ndarray | None = None       # (3, 256, 256) preprocesada: lo que se muestra
    entrada_red: np.ndarray | None = None  # lo que recibe la red (igual salvo recorte por la caja)
    resultado: Resultado | None = None
    caja: tuple | None = None
    ventana: tuple | None = None           # (fila0, col0, lado) de la ventana de --recorte-centrado


def predecir(paquete: PaqueteModelo, entrada: np.ndarray) -> tuple[float, float, float]:
    """(probabilidad, logit, milisegundos). Un corte, modelo en eval(), sin gradientes."""
    x = torch.from_numpy(np.ascontiguousarray(entrada, dtype=np.float32)).unsqueeze(0).to(paquete.dispositivo)
    paquete.modelo.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        logit = paquete.modelo(x).squeeze(1)
        prob = torch.sigmoid(logit)
    ms = (time.perf_counter() - t0) * 1000
    return float(prob.cpu().numpy()[0]), float(logit.cpu().numpy()[0]), ms


def analizar(paquete: PaqueteModelo, ficheros: dict[str, tuple[str, bytes] | None],
             datos_paciente: DatosPaciente | None = None) -> Analisis:
    """`ficheros`: fase -> (nombre, contenido) o None. Nunca lanza por culpa de la
    entrada: los problemas quedan como errores en el informe."""
    t0 = time.perf_counter()
    informe, validadas = validar_corte(ficheros)
    analisis = Analisis(informe=informe, nombres={f: ficheros[f][0] for f in FASES if ficheros.get(f)})
    if not informe.valido:
        return analisis

    caja = None
    preproc = paquete.meta.get("preprocesado", {})
    desconocidas = sorted(k for k, v in preproc.items() if k not in PREPROCESADO_CONOCIDO and v)
    if desconocidas:
        informe.anadir(CORTE, "Preprocesado del modelo", ERROR,
                       f"El modelo usa un preprocesado que la app no sabe reproducir ({', '.join(desconocidas)}).")
        return analisis
    lado = preproc.get("recorte_centrado")
    if preproc.get("recorte_caja") and lado:
        informe.anadir(CORTE, "Preprocesado del modelo", ERROR,
                       "modelo.json pide a la vez el recorte por la caja y la ventana centrada: no es coherente.")
        return analisis
    if necesita_caja(paquete.meta):
        if datos_paciente is None or not datos_paciente.tiene_caja():
            como = (f"usa una ventana de {lado} × {lado} px centrada en la caja del tumor" if lado else
                    "recorta la imagen alrededor de la caja del tumor")
            informe.anadir(CORTE, "Datos de la paciente", ERROR,
                           f"Este modelo {como} y necesita la caja de la paciente: suba el CSV con su fila de "
                           "patients.csv o rellene el formulario (n_xy, sraw, eraw, scol, ecol y xy_spacing). "
                           "Sin la caja no se predice: centrar la ventana en otro sitio daría una entrada distinta "
                           "de la del entrenamiento.")
            return analisis
        caja = datos_paciente.caja_png()
        analisis.caja = caja
        if lado:
            lado = int(lado)
            analisis.ventana = (*ventana_centrada(caja, lado), lado)

    def fuentes():
        return [io.BytesIO(validadas[f].datos_png) for f in FASES]

    try:
        imagen = preprocesar(*fuentes())
        entrada = imagen if caja is None else preprocesar(*fuentes(), caja=caja, lado_centrado=lado or None)
    except ValueError as e:                     # preprocesar vuelve a comprobar el formato
        informe.anadir(CORTE, "Preprocesado", ERROR, f"preprocesado.preprocesar rechazó la entrada: {e}")
        return analisis
    informe.anadir(CORTE, "Preprocesado", OK,
                   "preprocesado.preprocesar: PRE, EARLY y LATE apiladas en ese orden y divididas entre 255"
                   + ("" if caja is None else
                      f", ventana de {lado} × {lado} px centrada en la caja del tumor, sin reescalar" if lado else
                      ", recortadas por la caja del tumor"))
    comprobar_preprocesado(imagen, informe)
    analisis.imagen = imagen
    if not informe.valido:
        return analisis

    analisis.entrada_red = entrada
    prob, logit, ms = predecir(paquete, entrada)
    umbral = paquete.umbral
    analisis.resultado = Resultado(probabilidad=prob, logit=logit, umbral=umbral, clase=int(prob >= umbral),
                                   tiempo_inferencia_ms=ms, tiempo_total_ms=(time.perf_counter() - t0) * 1000,
                                   dispositivo=str(paquete.dispositivo))
    return analisis
