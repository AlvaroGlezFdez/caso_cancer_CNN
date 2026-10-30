"""Piezas comunes de la interfaz: modelo y cálculos cacheados, estilo, barra
superior, barra lateral, cabeceras, tiles y la envoltura que evita que un error
muestre una traza."""

from __future__ import annotations

import base64
import hashlib
import html
import logging
import re

import streamlit as st

from nucleo import rutas
from nucleo.explicacion import MapaAtencion, grad_cam
from nucleo.fusion import ErrorFusion, ModeloFusion, cargar_fusion
from nucleo.inferencia import Analisis, DatosPaciente, analizar
from nucleo.paquete import ErrorPaquete, PaqueteModelo, cargar_paquete
from preprocesado import FASES

log = logging.getLogger("app")

AVISO_USO = ("Uso exclusivamente educativo. Sin validez clínica: no es un dispositivo médico y no sirve para "
             "diagnosticar ni para decidir tratamientos.")


@st.cache_resource(show_spinner="Cargando el modelo…")
def paquete_modelo() -> PaqueteModelo:
    """Una sola carga por proceso: todas las sesiones comparten el modelo (solo lectura)."""
    return cargar_paquete()


def modelo_o_parar() -> PaqueteModelo:
    try:
        return paquete_modelo()
    except ErrorPaquete as e:
        st.error(f"No se puede cargar el modelo. {e}", icon=":material/error:")
    except Exception:
        log.exception("Error inesperado al cargar el modelo")
        st.error("No se puede cargar el modelo: error inesperado al leer modelo_app/.", icon=":material/error:")
    st.stop()


@st.cache_resource(show_spinner=False)
def _modelo_fusion() -> ModeloFusion | None:
    return cargar_fusion()


def fusion_o_none() -> ModeloFusion | None:
    """El modelo final combinado (modelo_app/fusion.json), o None si no hay o está dañado:
    entonces la app usa solo la imagen (y lo dice)."""
    try:
        return _modelo_fusion()
    except ErrorFusion as e:
        log.error("fusion.json no válido: %s", e)
    except Exception:
        log.exception("Error inesperado al cargar fusion.json")
    return None


def clave_entrada(entrada: tuple) -> str:
    """Huella corta del contenido de los tres ficheros (para claves de estado)."""
    h = hashlib.sha1()
    for e in entrada:
        h.update(b"-" if e is None else e[1])
    return h.hexdigest()[:12]


@st.cache_data(max_entries=32, show_spinner="Analizando el corte…")
def analizar_cacheado(sha_modelo: str, entrada: tuple, caja: tuple | None = None) -> Analisis:
    """Cacheado por contenido: la misma muestra da siempre el mismo análisis.
    `sha_modelo` forma parte de la clave para no mezclar modelos. `caja`: (n_xy, sraw,
    eraw, scol, ecol) de la paciente, solo si el modelo recorta por la caja."""
    ficheros = {fase: (None if e is None else (e[0], e[1])) for fase, e in zip(FASES, entrada)}
    datos = None if caja is None else DatosPaciente(*caja)
    return analizar(paquete_modelo(), ficheros, datos)


@st.cache_data(max_entries=64, show_spinner="Calculando el mapa de atención…")
def grad_cam_cacheado(sha_modelo: str, entrada: tuple, hacia_pcr: bool,
                      caja: tuple | None = None) -> MapaAtencion | None:
    analisis = analizar_cacheado(sha_modelo, entrada, caja)
    if analisis.entrada_red is None:
        return None
    return grad_cam(paquete_modelo(), analisis.entrada_red, hacia_pcr)


# --------------------------------------------------------------------------- #
# Textos y piezas de la página
# --------------------------------------------------------------------------- #

def encabezado(titulo: str, antetitulo: str | None = None, subtitulo: str | None = None) -> None:
    if antetitulo:
        st.html(f'<div class="antetitulo">{html.escape(antetitulo)}</div>')
    st.title(titulo)
    if subtitulo:
        st.caption(subtitulo)


def tile(contenedor, etiqueta: str, valor, help: str | None = None, delta: str | None = None, **kw) -> None:
    """Métrica como «tile»: etiqueta en tipo oración y cifra en Inter (estilo.css)."""
    contenedor.metric(etiqueta, valor, help=help, delta=delta,
                      delta_color="off", delta_arrow="off", **kw)


def aviso_discreto(texto: str = AVISO_USO) -> None:
    st.html(f'<div class="aviso-discreto"><span class="ms">info</span><span>{html.escape(texto)}</span></div>')


def mas_detalle(titulo: str = "Más detalle"):
    return st.expander(titulo, icon=":material/unfold_more:")


def tarjetas(lista: list[dict], columnas_min: int = 210) -> None:
    """Rejilla de tarjetas HTML: icono (Material), cifra, título, texto y miniatura (data URI)."""
    partes = []
    for t in lista:
        partes.append(
            f'<div class="tarjeta{" aviso" if t.get("aviso") else ""}">'
            + (f'<img class="mini" src="{t["imagen"]}" alt="">' if t.get("imagen") else "")
            + (f'<span class="ms icono">{t["icono"]}</span>' if t.get("icono") else "")
            + (f'<div class="et">{html.escape(t["etiqueta"])}</div>' if t.get("etiqueta") else "")
            + (f'<div class="cifra">{t["cifra"]}</div>' if t.get("cifra") else "")
            + f'<h4>{t["titulo"]}</h4><p>{t["texto"]}</p></div>')
    st.html(f'<div class="rejilla-tarjetas" style="grid-template-columns: repeat(auto-fit, minmax({columnas_min}px, 1fr))">'
            + "".join(partes) + "</div>")


def ejecutar(contenido) -> None:
    """Ejecuta una página; un error inesperado se muestra con un mensaje en español
    y nunca con una traza (st.stop y st.rerun no son Exception: no se capturan)."""
    try:
        contenido()
    except Exception:
        log.exception("Error inesperado en una página")
        st.error("Se ha producido un error inesperado en esta sección. La app sigue funcionando: puede volver a "
                 "intentarlo o pasar a otra sección.", icon=":material/error:")


# --------------------------------------------------------------------------- #
# Marco: estilo, barra superior, barra lateral y pie
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def _estaticos() -> tuple[str, str]:
    carpeta = rutas.RAIZ_APP / "estaticos"
    logo = base64.b64encode((carpeta / "logo.svg").read_bytes()).decode("ascii")
    return (carpeta / "estilo.css").read_text(encoding="utf-8"), f"data:image/svg+xml;base64,{logo}"


def _paquete_o_none() -> PaqueteModelo | None:
    try:
        return paquete_modelo()
    except Exception:
        return None


def marco() -> None:
    """Estilo propio, barra superior y barra lateral. Lo llama streamlit_app.py en cada ejecución."""
    css, logo = _estaticos()
    paquete = _paquete_o_none()
    fusion = fusion_o_none()
    if paquete is not None:
        pastillas = (
            '<span class="pastilla"><span class="punto"></span><b>Modelo cargado</b></span>'
            f'<span class="pastilla opcional" title="SHA-256 de modelo.pt: {paquete.sha256}">SHA-256 '
            f'<b>{paquete.sha256[:8]}</b></span>'
            f'<span class="pastilla opcional">{html.escape(str(paquete.dispositivo).upper())}</span>')
        if fusion is None:
            pastillas += ('<span class="pastilla aviso" title="No hay modelo_app/fusion.json: la app usa solo la '
                          'imagen.">Solo imagen</span>')
        elif fusion.provisional or not fusion.coincide_cnn(paquete.meta["ejecucion"]):
            motivo = (fusion.meta.get("nota_provisional") or "") + (
                "" if fusion.coincide_cnn(paquete.meta["ejecucion"]) else
                f" La CNN del paquete ({paquete.meta['ejecucion']}) no es de la configuración de la fusión "
                f"({fusion.configuracion}).")
            pastillas += f'<span class="pastilla aviso" title="{html.escape(motivo.strip())}">Fusión provisional</span>'
        else:
            pastillas += '<span class="pastilla opcional"><b>Clínico + CNN</b></span>'
    else:
        pastillas = '<span class="pastilla mal"><span class="punto"></span><b>Modelo no disponible</b></span>'
    pastillas += '<span class="pastilla aviso" title="' + html.escape(AVISO_USO) + '">Uso educativo</span>'
    st.html(f"<style>{css}</style>"
            f'<div class="barra-superior"><div class="marca"><img src="{logo}" width="34" height="34" alt="">'
            f'<div class="nombre">Estación pCR · DCE-MRI'
            f'<small>Análisis de resonancia de mama · herramienta docente</small></div></div>'
            f'<div class="pastillas">{pastillas}</div></div>')
    barra_lateral(paquete, fusion)


AVISO_PIE = "Uso exclusivamente educativo, sin validez clínica · no es un dispositivo médico"


def barra_lateral(paquete: PaqueteModelo | None, fusion: ModeloFusion | None = None) -> None:
    """Una línea sobria con el modelo en uso; el detalle está en «El modelo» y en el panel técnico."""
    if paquete is None:
        texto = "Modelo no disponible"
    elif fusion is None:
        texto = "Modelo: solo imagen (CNN)"
    else:
        texto = "Modelo final: clínico + CNN" + (" (provisional)" if fusion.provisional else "")
    with st.sidebar:
        st.html(f'<div class="lateral-modelo">{html.escape(texto)}</div>')


def pie() -> None:
    st.html(f'<div class="pie-app">{html.escape(AVISO_PIE)}</div>')
