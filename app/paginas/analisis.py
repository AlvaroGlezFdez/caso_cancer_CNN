"""Página principal: análisis de un caso (3 PNG de un corte o un ejemplo), como un visor tipo PACS.

Entrada compacta en una fila y, debajo, las variables de la paciente (CSV con su
fila de patients.csv, formulario o las del ejemplo) -> nucleo.variables.validar
-> nucleo.inferencia.analizar (validación, preprocesado.preprocesar con la caja
si el modelo recorta por ella o usa la ventana centrada en ella, CNN) -> nucleo.fusion (modelo final: variables clínicas +
CNN) -> tiras de comprobaciones, visor de 4 paneles, resultado (anillo del
modelo combinado, la imagen sola y la contribución de cada parte) y pestañas:
realce 3D, curva cinética, mapa de atención y datos técnicos. Sin variables, el
resultado es el de la imagen sola y se dice que no es el modelo final. Cualquier fallo inesperado se muestra con un
mensaje en español, nunca con una traza, y la app sigue lista para la siguiente
muestra.
"""

from __future__ import annotations

import html
import math
import platform
from datetime import datetime
from pathlib import PurePath

import pandas as pd
import streamlit as st
import torch

from clinica import SUBTIPOS
from nucleo import ejemplos, variables as var
from nucleo.explicacion import curva_cinetica, grad_cam, punto_maximo_realce, reducir
from nucleo.formato import entero, mayuscula, num, numero_g, pct
from nucleo.fusion import ModeloFusion, ResultadoFusion
from nucleo.inferencia import Analisis, necesita_caja as _necesita_caja
from nucleo.informe_caso import informe_html
from nucleo.validacion import ERROR
from paginas import comun, graficos as g, visor
from preprocesado import FASES

ORIGEN_SUBIR, ORIGEN_EJEMPLO = "Subir", "Ejemplo"
VAR_CSV, VAR_FORMULARIO, VAR_EJEMPLO, VAR_NINGUNA = "CSV", "Formulario", "Ejemplo", "Ninguna"
DESCRIPCION_FASE = {"PRE": "antes del contraste", "EARLY": "poscontraste temprana", "LATE": "poscontraste tardía"}
EXPLICACION_TIPO = {
    "ascendente": "Sigue subiendo en la fase tardía (persistente, tipo I).",
    "meseta": "Se mantiene entre la fase temprana y la tardía (meseta, tipo II).",
    "lavado": "Baja en la fase tardía (lavado, tipo III), un patrón que en clínica se asocia más a lesiones "
              "sospechosas.",
    "sin realce apreciable": "La zona apenas se ilumina con el contraste: no hay curva que clasificar.",
}
OPACIDAD_ATENCION = 0.6
PESTANAS = ["Realce 3D", "Cinética", "Atención", "Técnico"]


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #

def _nuevo_caso() -> None:
    st.session_state.caso = st.session_state.get("caso", 0) + 1


def _entrada_subida() -> dict:
    caso = st.session_state.setdefault("caso", 0)
    ficheros = {}
    for columna, fase in zip(st.columns(3, gap="small"), FASES):
        with columna:
            subido = st.file_uploader(
                fase, key=f"subida_{fase}_{caso}", max_upload_size=10,
                help=f"Fase {fase} ({DESCRIPCION_FASE[fase]}): PNG en escala de grises de 8 bits y 256 × 256. "
                     "Máximo 2 MB.")
            ficheros[fase] = None if subido is None else (subido.name, subido.getvalue())
    return ficheros


def _entrada_ejemplo(indice: dict) -> tuple[dict, dict]:
    lista = indice["ejemplos"]
    elegido = st.radio("Ejemplo de entrenamiento", [e["clave"] for e in lista], horizontal=True, key="ejemplo",
                       format_func=lambda c: next(f"{e['titulo']} · {e['cohorte']}" for e in lista if e["clave"] == c),
                       help="Cortes de entrenamiento sin etiqueta ni identificador de la paciente. "
                            + indice["licencia"])
    info = next(e for e in lista if e["clave"] == elegido)
    return ejemplos.cargar(elegido), info


def _ejemplo_de_la_url(indice: dict) -> None:
    """?ejemplo=ejemplo_2 abre la página con ese ejemplo (solo claves del índice)."""
    if st.session_state.get("url_atendida"):
        return
    st.session_state.url_atendida = True
    clave = st.query_params.get("ejemplo")
    if clave in {e["clave"] for e in indice["ejemplos"]}:
        st.session_state.origen = ORIGEN_EJEMPLO
        st.session_state.ejemplo = clave


def _formulario(base: str, inicial: dict, necesita_caja: bool, rango: dict | None) -> var.VariablesPaciente | None:
    """Formulario (st.form: los valores se aplican al pulsar). None si está vacío."""
    def campo(col, etiqueta, clave, ayuda, paso=1.0, formato="%.2f"):
        return col.number_input(etiqueta, value=inicial.get(clave), step=paso, format=formato, help=ayuda,
                                key=f"f_{clave}_{base}", placeholder="sin dato")

    with st.form(f"formulario_{base}", border=False):
        fila = st.columns([1.3, 1, 1, 1], gap="small",
                          vertical_alignment="bottom")
        actual = inicial.get("HR_HER2_STATUS")
        subtipo = fila[0].selectbox(
            "Subtipo HR/HER2", [None, *SUBTIPOS], index=([None, *SUBTIPOS].index(actual) if actual in SUBTIPOS else 0),
            format_func=lambda s: "sin dato" if s is None else var.NOMBRE_SUBTIPO[s],
            key=f"f_subtipo_{base}", help="HR_HER2_STATUS de patients.csv.")
        valores = {"HR_HER2_STATUS": subtipo,
                   "tum_vol": campo(fila[1], "Volumen tumoral", "tum_vol", "tum_vol de patients.csv (mismas unidades)."),
                   "age": campo(fila[2], "Edad (años)", "age", "age de patients.csv.", formato="%.1f")}
        if necesita_caja:
            valores["xy_spacing"] = campo(fila[3], "xy_spacing (mm)", "xy_spacing", "Tamaño del píxel del corte "
                                          "original, en mm.", paso=0.01, formato="%.4f")
            fila = st.columns([1, 1, 1, 1, 1, 1], gap="small", vertical_alignment="bottom")
            for col, c in zip(fila, ("n_xy", "sraw", "eraw", "scol", "ecol")):
                valores[c] = campo(col, c, c, "Caja del tumor en píxeles del corte original (patients.csv): filas "
                                              "sraw-eraw, columnas scol-ecol, corte de n_xy × n_xy.", formato="%.0f")
            boton = fila[5]
        else:
            boton = fila[3]
        boton.form_submit_button("Aplicar", icon=":material/check:", width="stretch")
    if all(v is None for v in valores.values()):
        return None
    return var.validar(valores, necesita_caja, rango, origen="formulario")


def _panel_variables(info_ejemplo: dict | None, necesita_caja: bool,
                     fusion: ModeloFusion | None) -> var.VariablesPaciente | None:
    """Variables de la paciente: CSV (su fila de patients.csv), formulario, las del
    ejemplo o ninguna. Devuelve None si no hay variables."""
    opciones = ([VAR_EJEMPLO] if info_ejemplo else [VAR_CSV]) + [VAR_FORMULARIO, VAR_NINGUNA]
    if st.session_state.get("fuente_variables") not in opciones:
        st.session_state.fuente_variables = opciones[0]
    rango = fusion.meta.get("rango_entrenamiento") if fusion else None
    caso = st.session_state.get("caso", 0)
    base = info_ejemplo["clave"] if info_ejemplo else f"subida_{caso}"
    with st.container(key="panel_variables"):
        izquierda, derecha = st.columns([2.1, 3.5], gap="medium", vertical_alignment="bottom")
        with izquierda:
            fuente = st.segmented_control(
                "Variables", opciones, required=True, key="fuente_variables",
                help="Variables de la paciente para el modelo final (clínico + CNN): subtipo HR/HER2, volumen "
                     "tumoral y edad" + (" y, como esta CNN recorta la imagen, la caja del tumor" if necesita_caja
                                          else "") + ". Sin ellas, solo la probabilidad de la imagen.")
        with derecha:
            if fuente == VAR_CSV:
                subido = st.file_uploader(
                    "Variables (CSV)", key=f"subida_csv_{caso}", max_upload_size=1,
                    help="Una fila con las columnas de patients.csv (cabecera incluida). Se usan HR_HER2_STATUS, "
                         "tum_vol y age" + (" y n_xy, sraw, eraw, scol, ecol y xy_spacing" if necesita_caja else "")
                         + "; el resto se ignora, también la pCR.")
                st.download_button("Descargar plantilla CSV", var.plantilla_csv(necesita_caja),
                                   file_name="plantilla_variables.csv", mime="text/csv", icon=":material/download:",
                                   key="plantilla_panel", type="tertiary",
                                   help="Cabecera exacta y una fila de ejemplo inventada: cámbiela por los datos de "
                                        "la paciente.")
                if subido is None:
                    return None
                return var.desde_csv(subido.name, subido.getvalue(), necesita_caja, rango)
            if fuente == VAR_EJEMPLO:
                v = info_ejemplo.get("variables") or {}
                sub = v.get("HR_HER2_STATUS")
                partes = [var.NOMBRE_SUBTIPO.get(sub, "subtipo sin dato"),
                          f"volumen {num(v['tum_vol'], 1)}" if v.get("tum_vol") is not None else "volumen sin dato",
                          f"{num(v['age'], 0)} años" if v.get("age") is not None else "edad sin dato"]
                if necesita_caja and v.get("n_xy") is not None:
                    partes.append(f"caja {v['sraw']:g}-{v['eraw']:g} × {v['scol']:g}-{v['ecol']:g} de {v['n_xy']:g} px")
                st.html('<div class="tira"><span class="resumen">Del ejemplo</span>'
                        + "".join(f'<span class="chip">{html.escape(p)}</span>' for p in partes) + "</div>")
                return var.validar(v, necesita_caja, rango, origen="ejemplo")
            if fuente == VAR_FORMULARIO:
                inicial = dict(info_ejemplo.get("variables") or {}) if info_ejemplo else {}
                return _formulario(base, inicial, necesita_caja, rango)
            st.caption("Sin variables de la paciente no se puede analizar: esta CNN mira una ventana centrada en "
                       "la caja del tumor y la necesita. Con la caja válida y sin el resto de variables, se da solo "
                       "la probabilidad de la imagen, que no es el modelo final." if necesita_caja else
                       "Sin variables de la paciente: se calcula solo la probabilidad de la imagen, que no es el "
                       "modelo final.")
            return None


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #

def _frase(r) -> str:
    if r.clase:
        return ("<b>Por encima del umbral</b>: el modelo agrupa este corte con los que alcanzaron la pCR; "
                "es una estimación orientativa, no un diagnóstico.")
    return ("<b>Por debajo del umbral</b>: el modelo agrupa este corte con los que no alcanzaron la pCR; "
            "es una estimación orientativa, no un diagnóstico.")


def _frase_combinado(c: ResultadoFusion) -> str:
    lado = "Por encima del umbral" if c.clase else "Por debajo del umbral"
    grupo = "los que alcanzaron" if c.clase else "los que no alcanzaron"
    return (f"<b>{lado}</b>: con la imagen y las variables clínicas, el modelo agrupa este caso con {grupo} la pCR; "
            "es una estimación orientativa, no un diagnóstico.")


def _texto_rellenadas(c: ResultadoFusion, fusion: ModeloFusion) -> str:
    """Qué variable clínica faltaba y con qué valor se ha rellenado (la media de entrenamiento
    del escalador de fusion.json, como en el ajuste)."""
    media = fusion.escalador.media
    partes = []
    if any(v.startswith("sub_") for v in c.imputadas):
        partes.append("**subtipo HR/HER2**: cada indicador toma su proporción en entrenamiento ("
                      + ", ".join(f"{var.NOMBRE_SUBTIPO[s]} {num(media[f'sub_{s}'], 2)}" for s in SUBTIPOS) + ")")
    if "log_tum_vol" in c.imputadas:
        partes.append(f"**volumen tumoral**: media de entrenamiento de log(1 + volumen), {num(media['log_tum_vol'], 2)} "
                      f"(equivale a un volumen de {num(math.expm1(media['log_tum_vol']), 1)})")
    if "age" in c.imputadas:
        partes.append(f"**edad**: media de entrenamiento, {num(media['age'], 1)} años")
    return ("Valor que faltaba, rellenado como en el ajuste del modelo. " if len(partes) == 1 else
            "Valores que faltaban, rellenados como en el ajuste del modelo. ") + "; ".join(partes) + \
        ". El resultado combinado es menos fiable."


def _resultado(analisis: Analisis, paquete, combinado: ResultadoFusion | None, fusion: ModeloFusion | None,
               motivo_solo_imagen: str, entrada_texto: str, clave: str, licencia: str | None,
               info_ejemplo: dict | None) -> None:
    r, meta = analisis.resultado, paquete.meta
    with st.container(key="panel_resultado"):
        if combinado is not None:
            st.markdown(visor.anillo(combinado.probabilidad, combinado.umbral, combinado.clase,
                                     _frase_combinado(combinado), etiqueta="Modelo combinado", tipo="combinado",
                                     datos={"imagen": num(r.probabilidad, 3), "imagen-exacta": repr(r.probabilidad)}),
                        unsafe_allow_html=True)
            celdas = [
                ("Solo imagen", num(r.probabilidad, 3), f"umbral {num(r.umbral, 3)}",
                 "Probabilidad de la CNN sola para este corte, con su propio umbral (por corte). No es el modelo final."),
                ("Solo clínico", num(combinado.prob_clinica, 3), "sin imagen",
                 "Referencia: el modelo clínico (subtipo, volumen y edad) sin la imagen, ajustado igual."),
                ("Umbral", num(combinado.umbral, 3), f"{num(r.tiempo_total_ms, 0)} ms",
                 f"Positivo (pCR) si la probabilidad combinada ≥ umbral. Umbral {fusion.meta['umbral']['regla']}. "
                 f"{mayuscula(fusion.meta['umbral']['nota'])}. Debajo, el tiempo de validación, preprocesado e "
                 "inferencia."),
            ]
            st.html('<div class="mini-cifras">' + "".join(
                f'<div class="mc" title="{html.escape(ayuda)}"><div class="et">{html.escape(et)}</div>'
                f'<div class="va">{va}</div><div class="de">{html.escape(de)}</div></div>'
                for et, va, de, ayuda in celdas) + "</div>")
            g.mostrar(g.figura_contribuciones(combinado), key="contribuciones")
            if combinado.imputadas:
                st.warning(_texto_rellenadas(combinado, fusion), icon=":material/edit_note:")
        else:
            st.markdown(visor.anillo(r.probabilidad, r.umbral, r.clase, _frase(r), etiqueta="Solo imagen",
                                     tipo="imagen"), unsafe_allow_html=True)
            comun.aviso_discreto(f"Probabilidad de la imagen sola: no es el modelo final. {motivo_solo_imagen}")
            c1, c2 = st.columns(2, gap="small")
            comun.tile(c1, "Umbral", num(r.umbral, 3),
                       help=f"Positivo (pCR) si la probabilidad ≥ umbral. Umbral {meta['umbral']['regla']}. "
                            f"{mayuscula(meta['umbral']['nota'])}.")
            comun.tile(c2, "Tiempo", f"{num(r.tiempo_total_ms, 0)} ms",
                       help=f"{num(r.tiempo_inferencia_ms, 1)} ms de inferencia en {r.dispositivo.upper()}; el resto, "
                            "validación y preprocesado.")
        _boton_informe(analisis, paquete, combinado, fusion, entrada_texto, clave, licencia, info_ejemplo)


def _boton_informe(analisis: Analisis, paquete, combinado: ResultadoFusion | None, fusion: ModeloFusion | None,
                   origen: str, clave: str, licencia: str | None, info_ejemplo: dict | None) -> None:
    """El HTML se genera solo al pulsar (descarga diferida), con el punto de la curva
    cinética y los ajustes del mapa de atención elegidos en sus pestañas o, si no
    se han abierto, los iniciales."""
    def generar() -> bytes:
        curva = st.session_state.get(f"cinetica_{clave}") or curva_cinetica(
            analisis.imagen, *_punto_inicial(analisis.imagen, info_ejemplo))
        hacia_pcr, opacidad = st.session_state.get(f"atencion_{clave}", (True, OPACIDAD_ATENCION))
        mapa = grad_cam(paquete, analisis.entrada_red, hacia_pcr)
        return informe_html(analisis, paquete.meta, paquete.sha256, origen, cinetica=curva, atencion=mapa,
                            opacidad_atencion=opacidad, licencia=licencia, combinado=combinado,
                            fusion_meta=None if fusion is None else fusion.meta).encode("utf-8")

    st.download_button("Informe del caso (HTML)", data=generar, width="stretch",
                       file_name=f"informe_caso_{datetime.now():%Y%m%d_%H%M%S}.html", mime="text/html",
                       icon=":material/download:", key=f"descarga_{clave}",
                       help="Un único HTML con las imágenes, el resultado, la validación, la curva cinética, el mapa "
                            "de atención, los datos del modelo y el aviso de uso. Se puede imprimir o guardar como PDF.")


def _tabla_informe(analisis: Analisis, solo_presentes: bool = False, extra: list | None = None) -> None:
    pasos = analisis.informe.pasos + list(extra or [])
    if solo_presentes:
        pasos = [p for p in pasos if not (p.comprobacion in ("Fichero recibido", "Tres fases presentes")
                                          and p.estado == ERROR)]
    tabla = pd.DataFrame([{"Ámbito": p.ambito, "Comprobación": p.comprobacion,
                           "Resultado": visor.ETIQUETA_ESTADO[p.estado], "Detalle": p.detalle} for p in pasos])
    st.dataframe(tabla, hide_index=True, width="stretch",
                 column_config={"Detalle": st.column_config.TextColumn(width="large")})


# --------------------------------------------------------------------------- #
# Pestañas (solo se ejecuta la que está abierta)
# --------------------------------------------------------------------------- #

def _punto_inicial(x, info_ejemplo: dict | None) -> tuple[int, int]:
    """Centro de la caja del tumor en los ejemplos; en lo subido, el máximo realce del tejido."""
    sugerido = info_ejemplo.get("punto_sugerido") if info_ejemplo else None
    return tuple(sugerido) if sugerido else punto_maximo_realce(x)


def _pestana_superficie(analisis: Analisis) -> None:
    completa = st.toggle("Resolución completa (256 × 256, más lenta de girar)", key="superficie_completa",
                         persist_state="page")
    realce = analisis.imagen[1] - analisis.imagen[0]
    g.mostrar(g.figura_superficie(realce if completa else reducir(realce, 2)), key="superficie")
    st.caption("Altura y color: realce EARLY − PRE. Rojo, se ilumina con el contraste; azul, se oscurece. "
               "Arrastre para girar.")


def _pestana_cinetica(analisis: Analisis, clave: str, info_ejemplo: dict | None) -> None:
    x = analisis.imagen
    k_punto, k_sel, k_graf = f"punto_{clave}", f"ultima_sel_{clave}", f"selector_{clave}"
    k_fila, k_col = f"fila_{clave}", f"columna_{clave}"
    if k_punto not in st.session_state:
        st.session_state[k_punto] = _punto_inicial(x, info_ejemplo)
        st.session_state[k_fila], st.session_state[k_col] = st.session_state[k_punto]

    # Un clic en la imagen (selección de plotly) cambia el punto antes de dibujar nada.
    try:
        puntos = list(st.session_state[k_graf]["selection"]["points"])
    except (KeyError, TypeError):
        puntos = []
    if puntos:
        nuevo = (int(round(puntos[0]["y"])), int(round(puntos[0]["x"])))
        if st.session_state.get(k_sel) != nuevo:
            st.session_state[k_sel] = nuevo
            st.session_state[k_punto] = nuevo
            st.session_state[k_fila], st.session_state[k_col] = nuevo

    def desde_campos():
        st.session_state[k_punto] = (int(st.session_state[k_fila]), int(st.session_state[k_col]))

    izquierda, derecha = st.columns([1, 1.15], gap="large")
    with izquierda:
        c1, c2, c3 = st.columns(3)
        c1.number_input("Fila", 0, x.shape[1] - 1, key=k_fila, on_change=desde_campos, persist_state="page")
        c2.number_input("Columna", 0, x.shape[2] - 1, key=k_col, on_change=desde_campos, persist_state="page")
        radio = c3.select_slider("Ventana", options=[0, 1, 2, 3, 4, 5], value=2, key=f"radio_{clave}",
                                 format_func=lambda r: f"{2 * r + 1}×{2 * r + 1}",
                                 help="Lado de la zona cuadrada que se promedia, en píxeles.", persist_state="page")
        fila, columna = st.session_state[k_punto]
        g.mostrar(g.figura_selector_punto(x[1] - x[0], (fila, columna), radio), key=k_graf,
                  on_select="rerun", selection_mode="points")
        st.caption("Pulse sobre el mapa de realce para elegir la zona.")
    curva = curva_cinetica(x, fila, columna, radio)
    tejido = x[0] > 0.1
    referencia = [float(x[k][tejido].mean()) for k in range(3)] if tejido.sum() > 500 else None
    with derecha:
        c1, c2, c3 = st.columns(3, gap="small")
        comun.tile(c1, "Tipo de curva", curva.tipo,
                   help="Orientativo. Cambio de LATE respecto a EARLY: más de +10 % ascendente, entre −10 % y "
                        "+10 % meseta, menos de −10 % lavado. " + EXPLICACION_TIPO[curva.tipo])
        comun.tile(c2, "Realce inicial", "—" if curva.realce_inicial is None else pct(curva.realce_inicial),
                   help="(EARLY − PRE) / PRE en la zona elegida.")
        comun.tile(c3, "Cambio tardío", "—" if curva.cambio_tardio is None else pct(curva.cambio_tardio),
                   help="(LATE − EARLY) / EARLY en la zona elegida.")
        g.mostrar(g.figura_curva_cinetica(curva, referencia), key=f"curva_{clave}")
        st.caption("Intensidades normalizadas por volumen: porcentajes aproximados, curva orientativa.")
    st.session_state[f"cinetica_{clave}"] = curva


def _pestana_atencion(analisis: Analisis, paquete, entrada: tuple, clave: str, caja_red: tuple | None) -> None:
    izquierda, derecha = st.columns([1.5, 1], gap="large")
    with derecha:
        hacia = st.segmented_control("Explicar", ["pCR", "no pCR"], default="pCR", required=True, key="atencion_hacia",
                                     help="Zonas que empujan la predicción hacia esa clase.", persist_state="page")
        fondo_nombre = st.segmented_control("Fondo", ["PRE", "EARLY", "LATE"], default="EARLY", required=True,
                                            key="atencion_fondo", persist_state="page")
        opacidad = st.slider("Opacidad del mapa", 0, 100, round(100 * OPACIDAD_ATENCION), 5, format="%d %%",
                             key="atencion_opacidad", persist_state="page")
    mapa = comun.grad_cam_cacheado(paquete.sha256, entrada, hacia == "pCR", caja_red)
    if mapa is None:
        st.info("No hay mapa de atención para esta muestra.")
        return
    fondo = analisis.entrada_red[FASES.index(fondo_nombre)]
    with izquierda:
        g.mostrar(g.figura_atencion(fondo, mapa.mapa, opacidad / 100, fondo_nombre), key="atencion")
    with derecha:
        st.markdown(f"Zonas claras: las que más empujan la salida hacia **{hacia}** (Grad-CAM, último bloque). "
                    "**Orientativo**: no señala el tumor ni la causa de la respuesta.",
                    help=f"Mapa de {mapa.resolucion[0]} × {mapa.resolucion[1]} posiciones interpolado a "
                         f"{fondo.shape[0]} × {fondo.shape[1]}; lo que queda por debajo del 10 % del máximo no se "
                         "colorea. Con un modelo de rendimiento modesto puede señalar estructuras ajenas a la lesión "
                         "(corazón, piel, bordes).")
        coincide = "sí" if mapa.logit == analisis.resultado.logit else "no"
        st.html(f'<div class="exacta">logit Grad-CAM {num(mapa.logit, 6)} · igual al de la inferencia: {coincide}</div>')
    st.session_state[f"atencion_{clave}"] = (hacia == "pCR", opacidad / 100)


def _filas_combinado(c: ResultadoFusion, fusion: ModeloFusion, paquete) -> list[tuple[str, str]]:
    m = fusion.meta
    vc = m["validacion_cruzada"]
    aportes = " · ".join(f"{g.NOMBRE_VARIABLE.get(v, v)} {d['aporte']:+.4f}" for v, d in c.por_variable.items())
    return [
        ("Modelo final", f"regresión logística por corte: variables clínicas + CNN {m['cnn']['configuracion']}"
                         + (" · PROVISIONAL" if fusion.provisional else "")
                         + ("" if fusion.coincide_cnn(paquete.meta["ejecucion"]) else
                            f" · la CNN del paquete ({paquete.meta['ejecucion']}) no es de esa configuración")),
        ("Probabilidad combinada", f"{c.probabilidad!r}"),
        ("Logit combinado", f"{c.logit!r} = intercepto {c.intercepto:+.4f} + suma de aportes"),
        ("Aportes al logit", aportes),
        ("Clase (combinado)", f"{'pCR' if c.clase else 'no pCR'} (prob {'≥' if c.clase else '<'} umbral {c.umbral!r})"),
        ("Solo clínico", f"{c.prob_clinica!r}"),
        ("Rellenadas", ", ".join(c.imputadas) if c.imputadas else "ninguna"),
        ("Validación del combinado", f"5 folds · AUC por corte {num(vc['por_corte']['modelos']['combinado']['auc'])}"
                                     f" · por paciente {num(vc['por_paciente']['modelos']['combinado']['auc'])} · "
                                     f"exportado {m['exportado']}"),
    ]


def _pestana_tecnico(analisis: Analisis, paquete, combinado: ResultadoFusion | None,
                     fusion: ModeloFusion | None, variables: var.VariablesPaciente | None) -> None:
    r, meta = analisis.resultado, paquete.meta
    e, m = meta["entrenamiento"], meta["metricas_validacion"]
    c = m["con_umbral"]
    filas = (_filas_combinado(combinado, fusion, paquete) if combinado is not None else []) + [
        ("SHA-256 modelo.pt", f"{paquete.sha256} · coincide con modelo.json"),
        ("Ejecución", f"{meta['ejecucion']} · época {meta['modelo']['epoca']}"
                      + (" (la última: épocas fijas, sin elegir)"
                         if meta["modelo"].get("seleccion", {}).get("criterio") == "ultima" else "")
                      + f" · paquete v{meta['version_paquete']}"),
        ("Probabilidad (imagen)", f"{r.probabilidad!r}"),
        ("Logit (imagen)", f"{r.logit!r}"),
        ("Clase (imagen)", f"{'pCR' if r.clase else 'no pCR'} (prob {'≥' if r.clase else '<'} umbral {r.umbral!r})"),
        ("Dispositivo", f"{r.dispositivo} · PyTorch {torch.__version__} · {torch.get_num_threads()} hilos · "
                        f"Python {platform.python_version()} · Streamlit {st.__version__}"),
        ("Tiempo", f"{num(r.tiempo_inferencia_ms, 2)} ms inferencia · {num(r.tiempo_total_ms, 2)} ms total"),
        ("Preprocesado", f"{meta['preprocesado']['funcion']} · {meta['preprocesado']['normalizacion']} · "
                         + _texto_recorte(meta["preprocesado"], analisis)),
        ("Entrada de la red", " × ".join(map(str, meta["modelo"]["entrada"])) + " (PRE, EARLY, LATE)"),
        ("Modelo", f"CNNBase · {meta['modelo']['bloques']} bloques · filtros "
                   f"{'-'.join(map(str, meta['modelo']['filtros']))} · {entero(meta['modelo']['n_parametros'])} "
                   "parámetros"),
        ("Umbral", f"{num(meta['umbral']['valor'], 4)} · {meta['umbral']['regla']}"),
        ("Validación por corte", f"fold {e['fold_validacion']} · {entero(m['cortes'])} cortes de {m['pacientes']} "
                                 f"pacientes · AUC {num(m['auc'])} · sensibilidad {num(c['sensibilidad'])} · "
                                 f"especificidad {num(c['especificidad'])}"),
        ("Entrenamiento", f"{e['optimizador']} · lr {numero_g(e['lr'])} · batch {e['batch_size']} · mejor época "
                          f"{e['mejor_epoca']} de {e['epocas_ejecutadas']} · {e['dispositivo']}"),
    ]
    def celda(v) -> str:                         # el SHA-256, en monoespaciada (la única de la app)
        texto = html.escape(str(v))
        return texto.replace(paquete.sha256, f'<span class="sha">{paquete.sha256}</span>')

    st.html('<table class="tabla-tecnica">' + "".join(
        f"<tr><th>{html.escape(k)}</th><td>{celda(v)}</td></tr>" for k, v in filas) + "</table>")
    with comun.mas_detalle("Las comprobaciones de la entrada, una a una"):
        _tabla_informe(analisis, extra=variables.informe.pasos if variables else [])


# --------------------------------------------------------------------------- #

def _texto_recorte(preproc: dict, analisis: Analisis) -> str:
    if preproc.get("recorte_centrado"):
        lado = preproc["recorte_centrado"]
        texto = f"ventana centrada en la caja: {lado} × {lado} px, sin reescalar"
        if analisis.ventana is not None:
            texto += f" (origen fila {analisis.ventana[0]}, columna {analisis.ventana[1]})"
        return texto
    return f"recorte por la caja: {'sí' if preproc.get('recorte_caja') else 'no'}"


def _guia(necesita_caja: bool) -> None:
    """Bloque plegable con los cuatro pasos para analizar una muestra sin ayuda."""
    with st.expander("Cómo analizar una muestra", icon=":material/help:"):
        st.markdown(
            "1. **Imágenes.** Suba los tres PNG del mismo corte, cada uno en su casilla: **PRE**, **EARLY** y "
            "**LATE** (256 × 256, escala de grises).\n"
            "2. **Variables de la paciente.** Con el **formulario** o con un **CSV de una fila** con las columnas de "
            "patients.csv (descargue la plantilla).\n"
            "3. **Análisis.** Empieza solo en cuanto están las tres fases; con el formulario, pulse **Aplicar**.\n"
            "4. **Resultado.** Lea la **probabilidad** de pCR, el **umbral** y la **clase** (pCR si la probabilidad "
            "llega al umbral).")
        if necesita_caja:
            st.markdown("**Imprescindible:** la caja del tumor (**n_xy, sraw, eraw, scol, ecol** y xy_spacing): sin "
                        "ella no se analiza. **Pueden quedar vacías** el subtipo, el volumen y la edad: se rellenan con "
                        "la media de entrenamiento y se avisa.")
        else:
            st.markdown("**Pueden quedar vacías** el subtipo, el volumen y la edad: se rellenan con la media de "
                        "entrenamiento y se avisa. Sin ninguna variable se da solo la probabilidad de la imagen.")
        st.download_button("Descargar plantilla CSV", var.plantilla_csv(necesita_caja),
                           file_name="plantilla_variables.csv", mime="text/csv", icon=":material/download:",
                           key="plantilla_guia")


def _registrar(clave: str, variables, analisis: Analisis, combinado: ResultadoFusion | None) -> None:
    """Apunta el caso en el registro de la sesión (solo en st.session_state: nada se guarda en el
    servidor ni en disco). Un caso (mismas imágenes y variables) se apunta una sola vez."""
    firma = (clave, variables.clave() if variables is not None else None)
    vistos = st.session_state.setdefault("registro_firmas", set())
    if firma in vistos:
        return
    vistos.add(firma)
    r = analisis.resultado
    prob, umbral, clase = ((combinado.probabilidad, combinado.umbral, combinado.clase) if combinado is not None
                           else (r.probabilidad, r.umbral, r.clase))
    st.session_state.setdefault("registro_sesion", []).append({
        "Hora": datetime.now().strftime("%H:%M:%S"),
        "Ficheros": " · ".join(PurePath(n).name for n in analisis.nombres.values()),
        "Modelo": "Combinado (clínico + CNN)" if combinado is not None else "Solo imagen",
        "Probabilidad": num(prob, 3), "Umbral": num(umbral, 3), "Clase": "pCR" if clase else "No pCR"})


def _registro() -> None:
    filas = st.session_state.get("registro_sesion", [])
    if not filas:
        return
    with st.expander(f"Casos analizados en esta sesión ({len(filas)})", icon=":material/history:"):
        tabla = pd.DataFrame(filas)
        st.dataframe(tabla, hide_index=True, width="stretch")
        st.download_button("Descargar registro (CSV)", tabla.to_csv(index=False, sep=";").encode("utf-8-sig"),
                           file_name=f"casos_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv",
                           icon=":material/download:", key="registro_csv")
        st.caption("Solo en esta pestaña del navegador: no se guarda en el servidor y se borra al cerrarla.")


def _lista(pasos) -> str:
    return "\n".join(f"- **{p.ambito} · {p.comprobacion}**: {p.detalle}" for p in pasos)


def _contenido() -> None:
    paquete = comun.modelo_o_parar()
    fusion = comun.fusion_o_none()
    necesita_caja = _necesita_caja(paquete.meta)
    indice = ejemplos.indice()
    _ejemplo_de_la_url(indice)
    st.html('<div class="antetitulo" style="margin:0 0 .5rem">Análisis de un caso</div>')
    _guia(necesita_caja)

    st.session_state.setdefault("origen", ORIGEN_SUBIR)          # sin default=: la URL puede fijarlo
    if st.session_state.origen not in (ORIGEN_SUBIR, ORIGEN_EJEMPLO):
        st.session_state.origen = ORIGEN_SUBIR
    info_ejemplo = None
    with st.container(key="panel_entrada"):
        izquierda, centro, derecha = st.columns([1.2, 3.4, 1.0], gap="medium", vertical_alignment="bottom")
        with izquierda:
            origen = st.segmented_control("Origen", [ORIGEN_SUBIR, ORIGEN_EJEMPLO], required=True, key="origen")
        with centro:
            if origen == ORIGEN_EJEMPLO:
                ficheros, info_ejemplo = _entrada_ejemplo(indice)
                texto_origen = f"{info_ejemplo['titulo']} ({info_ejemplo['cohorte']}, corte de entrenamiento)"
            else:
                ficheros = _entrada_subida()
                texto_origen = "ficheros subidos"
        with derecha:
            if origen == ORIGEN_SUBIR:
                st.button("Nuevo caso", icon=":material/restart_alt:", on_click=_nuevo_caso, width="stretch",
                          help="Vacía las casillas (imágenes y variables) para cargar otra muestra.")
    variables = _panel_variables(info_ejemplo, necesita_caja, fusion)
    validas = variables is not None and variables.valido

    presentes = [f for f in FASES if ficheros.get(f) is not None]
    if not presentes:
        if variables is not None:
            visor.tira(variables.informe.pasos, "Variables")
        izquierda, derecha = st.columns([1.75, 1], gap="large")
        with izquierda:
            visor.visor_vacio()
        with derecha, st.container(key="panel_espera"):
            st.markdown("##### Cargue un caso")
            st.caption("Los tres PNG del mismo corte, cada uno en su casilla, o uno de los ejemplos, y las variables "
                       "de la paciente para el modelo final. Todo se valida antes de analizar.")
            comun.aviso_discreto()
        _registro()
        return
    entrada = tuple(None if ficheros.get(f) is None else (ficheros[f][0], ficheros[f][1]) for f in FASES)
    clave = comun.clave_entrada(entrada)
    # La caja solo entra en el análisis (y en la clave de la caché) si la CNN recorta por ella.
    # Basta con que la caja sea válida (caja_cruda es None si no lo es): si fallan otras
    # variables, se da la probabilidad de la imagen sola, con el aviso de que no es el modelo final.
    caja_red = variables.caja_cruda() if (necesita_caja and variables is not None) else None
    analisis = comun.analizar_cacheado(paquete.sha256, entrada, caja_red)

    if len(presentes) < 3:
        faltan = [f for f in FASES if f not in presentes]
        st.info(f"Falta{'n' if len(faltan) > 1 else ''} {', '.join(faltan)}. El análisis empieza con las tres fases.",
                icon=":material/hourglass_empty:")
        visor.tira([p for p in analisis.informe.pasos if not (
            p.comprobacion in ("Fichero recibido", "Tres fases presentes") and p.estado == ERROR)])
        if variables is not None:
            visor.tira(variables.informe.pasos, "Variables")
        return

    if analisis.resultado is None:
        errores = analisis.informe.errores + (variables.informe.errores if variables is not None else [])
        st.error("No se puede analizar la muestra:\n\n" + _lista(errores), icon=":material/error:")
        visor.tira(analisis.informe.pasos)
        if variables is not None:
            visor.tira(variables.informe.pasos, "Variables")
        with comun.mas_detalle("Las comprobaciones, una a una"):
            _tabla_informe(analisis, extra=variables.informe.pasos if variables else [])
        if analisis.imagen is not None:
            visor.visor(analisis.imagen)
        return

    combinado = None
    if fusion is None:
        motivo = "No hay modelo combinado en modelo_app/ (fusion.json)."
    elif variables is None:
        motivo = "Añada las variables de la paciente (CSV o formulario) para el modelo combinado."
    elif not validas:
        motivo = "Las variables de la paciente tienen errores: corríjalas para el modelo combinado."
    else:
        motivo = ""
        combinado = fusion.combinar(analisis.resultado.probabilidad, variables.valores)

    visor.tira(analisis.informe.pasos)
    if variables is not None:
        visor.tira(variables.informe.pasos, "Variables")
        if variables.informe.errores:
            st.error("Variables de la paciente con errores: se muestra solo la probabilidad de la imagen.\n\n"
                     + _lista(variables.informe.errores), icon=":material/error:")
    avisos = analisis.informe.avisos + (variables.informe.avisos if variables is not None else [])
    if avisos:
        st.warning(_lista(avisos), icon=":material/warning:")

    izquierda, derecha = st.columns([1.75, 1], gap="large")
    with izquierda:
        caja_png = (info_ejemplo or {}).get("caja_tumor")
        if caja_png is None and validas and variables.tiene_caja():
            caja_png = variables.datos_paciente().caja_png()
        caja = None
        a, b = st.columns([1.7, 2.6], vertical_alignment="center")
        if caja_png is not None and a.toggle(
                "Caja del tumor", key="ver_caja", persist_state="page",
                help="Caja de la lesión según patients.csv (dato de la paciente, no del modelo)."):
            caja = caja_png
        if info_ejemplo:
            b.caption(f"{info_ejemplo['titulo']} · {info_ejemplo['cohorte']}: {info_ejemplo['descripcion']}")
        visor.visor(analisis.imagen, caja, analisis.ventana)
        if analisis.ventana is not None:
            f0, c0, lado = analisis.ventana
            st.caption(f"Recuadro: la ventana de {lado} × {lado} px que recibe la red (filas {f0} a {f0 + lado - 1}, "
                       f"columnas {c0} a {c0 + lado - 1}), centrada en la caja del tumor y sin reescalar; lo que cae "
                       "fuera del PNG entra como ceros. Lo atenuado no lo ve la red.")
    with derecha:
        _resultado(analisis, paquete, combinado, fusion, motivo, texto_origen, clave,
                   indice["licencia"] if info_ejemplo else None, info_ejemplo)
    _registrar(clave, variables, analisis, combinado)
    _registro()

    # Pestañas perezosas: cada interacción solo dibuja la pestaña abierta.
    pestanas = st.tabs(PESTANAS, key="pestana_analisis", on_change="rerun")
    contenidos = [
        lambda: _pestana_superficie(analisis),
        lambda: _pestana_cinetica(analisis, clave, info_ejemplo),
        lambda: _pestana_atencion(analisis, paquete, entrada, clave, caja_red),
        lambda: _pestana_tecnico(analisis, paquete, combinado, fusion, variables),
    ]
    for pestana, contenido in zip(pestanas, contenidos):
        if pestana.open:
            with pestana:
                contenido()


comun.ejecutar(_contenido)
