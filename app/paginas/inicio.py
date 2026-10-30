"""Inicio, la portada y página por defecto: titular, una frase, tres cifras clave (AUC en
test, pacientes de test y parámetros de la CNN, leídas de experimentos/test/ y de
modelo.json), el botón «Analizar un caso» y, en grande, el volumen 3D del realce de
una paciente de ejemplo. Debajo, el esquema imagen -> red -> variables -> probabilidad."""

from __future__ import annotations

import html
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from nucleo import datos, ejemplos, test_independiente as ti
from nucleo.explicacion import media_local, reducir
from nucleo.formato import entero, num
from nucleo.imagenes import a_uint8
from nucleo.inferencia import necesita_caja
from paginas import comun, graficos as g, visor
from preprocesado import FASES, preprocesar

FACTOR = 4              # 256 -> 64 píxeles por lado: unos 41.000 vóxeles (ligero en Community Cloud)
EXAGERACION = 3         # los cortes están a ~2 mm; se separan para ver el volumen
# Mitad positiva de la escala del realce (se ilumina con el contraste), de oscuro a claro.
ESCALA_VOLUMEN = [[0, "#67001f"], [0.35, "#b2182b"], [0.7, "#ef8a62"], [1, "#fddbc7"]]


@st.cache_data(show_spinner="Preparando el volumen…")
def _figura_volumen(alto_figura: int = 560) -> go.Figure | None:
    p = datos.pila("ejemplo_1")
    if p is None:
        return None
    realce = (p["early"].astype(np.float32) - p["pre"].astype(np.float32)) / 255
    vol = np.stack([media_local(reducir(r, FACTOR), 1) for r in realce])          # (cortes, 64, 64)
    # Suavizado entre cortes vecinos (1-2-1): menos ruido, superficies más limpias.
    vol = (np.concatenate([vol[:1], vol[:-1]]) + 2 * vol + np.concatenate([vol[1:], vol[-1:]])) / 4
    n, alto, ancho = vol.shape
    px = float(p["pixel_mm"]) * FACTOR
    z = np.asarray(p["posicion"], dtype=float) * float(p["grosor_mm"]) * EXAGERACION
    zz, yy, xx = np.meshgrid(z, np.arange(alto) * px, np.arange(ancho) * px, indexing="ij")
    # Isosuperficies del realce (tres niveles, de translúcido a opaco), con iluminación.
    fig = go.Figure(go.Isosurface(
        x=xx.ravel().round(1), y=yy.ravel().round(1), z=zz.ravel().round(1), value=vol.ravel().round(3),
        isomin=0.15, isomax=0.4, surface=dict(count=3), opacity=0.5, colorscale=ESCALA_VOLUMEN,
        cmin=0.15, cmax=0.4, caps=dict(x_show=False, y_show=False, z_show=False), flatshading=False,
        lighting=dict(ambient=0.45, diffuse=0.9, specular=0.4, roughness=0.5, fresnel=0.3),
        lightposition=dict(x=2000, y=-3000, z=4000),
        colorbar=g.colorbar(title=dict(text="realce", side="right", font=dict(size=11, color=g.TEXTO_2)),
                            len=0.5, tickvals=[0.15, 0.25, 0.35]),
        hovertemplate="realce %{value:+.2f}<extra></extra>"))
    # Suelo: el corte del ejemplo (EARLY) en gris, para situar la anatomía.
    k = int(p["indice_ejemplo"])
    suelo = reducir(p["early"][k].astype(np.float32) / 255, FACTOR // 2)
    lado = np.arange(suelo.shape[0]) * px / 2
    fig.add_trace(go.Surface(x=lado, y=lado, z=np.full(suelo.shape, z.min() - 10), surfacecolor=suelo,
                             colorscale=g.ESCALA_GRIS, cmin=0, cmax=1, showscale=False,
                             opacityscale=[[0, 0], [0.12, 0], [0.35, 0.55], [1, 0.8]],
                             hoverinfo="skip", lighting=dict(ambient=1, diffuse=0, specular=0)))
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed"), zaxis=dict(visible=False),
        aspectmode="data", camera=dict(eye=dict(x=0.78, y=-0.88, z=0.74), center=dict(x=0, y=0, z=-0.05))))
    g.estilo(fig, alto=alto_figura, leyenda=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


@st.cache_data(show_spinner=False)
def _miniaturas() -> list[str]:
    """Las tres fases del ejemplo 1 a 256 x 256, sin pasar por la red (no dependen del modelo ni de su caja)."""
    fich = ejemplos.cargar("ejemplo_1")
    x = preprocesar(*(io.BytesIO(fich[f][1]) for f in FASES))
    return [visor.data_uri(a_uint8(x[k])) for k in range(3)]


def _entrada(clave: str) -> tuple:
    fich = ejemplos.cargar(clave)
    return tuple((fich[f][0], fich[f][1]) for f in FASES)


def _esquema(paquete, fusion) -> None:
    """Imagen -> red -> (variables clínicas) -> probabilidad, con el ejemplo 1."""
    info = next(e for e in ejemplos.indice()["ejemplos"] if e["clave"] == "ejemplo_1")
    v = info.get("variables") or {}
    caja = (tuple(v[c] for c in ("n_xy", "sraw", "eraw", "scol", "ecol"))
            if necesita_caja(paquete.meta) else None)
    analisis = comun.analizar_cacheado(paquete.sha256, _entrada("ejemplo_1"), caja)
    if analisis.resultado is None:
        return
    combinado = None if fusion is None else fusion.combinar(analisis.resultado.probabilidad, v)
    mini = "".join(f'<img src="{s}" alt="{f}" title="{f}">' for s, f in zip(_miniaturas(), ("PRE", "EARLY", "LATE")))
    capas = "".join(f'<span style="display:inline-block;width:{w}px;height:{h}px;margin:0 3px;border-radius:2px;'
                    f'background:{c};vertical-align:middle"></span>'
                    for w, h, c in ((10, 64, "#1F3651"), (12, 52, "#176E66"), (14, 40, "#1F8A7F"), (16, 28, "#38C6B6"),
                                    (18, 16, "#62D6C9"), (8, 8, "#EDF2F7")))
    mo = paquete.meta["modelo"]
    if combinado is not None:
        clinico = ('<div class="flecha"><span class="ms">add</span></div>'
                   '<div class="paso"><div class="et">3 · Paciente</div><h4>Variables clínicas</h4>'
                   '<div class="cifra" style="font-size:1.25rem;line-height:1.6">subtipo<br>volumen · edad</div>'
                   '<p>Regresión logística que suma la imagen y las variables de la paciente.</p></div>')
        final = ('<div class="paso"><div class="et">4 · Probabilidad</div><h4>Modelo combinado</h4>'
                 f'<div class="cifra">{num(combinado.probabilidad, 3)}</div>'
                 f'<p>Ejemplo 1: la imagen sola da {num(analisis.resultado.probabilidad, 3)}.</p></div>')
    else:
        clinico = ''
        final = ('<div class="paso"><div class="et">3 · Probabilidad</div><h4>Estimación de pCR</h4>'
                 f'<div class="cifra">{num(analisis.resultado.probabilidad, 3)}</div>'
                 '<p>Probabilidad del ejemplo 1, comparada con un umbral.</p></div>')
    st.html(f"""
<div class="esquema{' cuatro' if combinado is not None else ''}">
  <div class="paso"><div class="et">1 · Imagen</div><h4>Tres fases de un corte</h4>
    <div class="miniaturas">{mini}</div><p>PRE, EARLY y LATE: antes y después del contraste.</p></div>
  <div class="flecha"><span class="ms">arrow_forward</span></div>
  <div class="paso"><div class="et">2 · Red</div><h4>CNN desde cero</h4>
    <div style="height:76px;display:flex;align-items:center;justify-content:center">{capas}</div>
    <p>{mo['bloques']} bloques · {entero(mo['n_parametros'])} parámetros · mismo preprocesado que el entrenamiento.</p></div>
  {clinico}
  <div class="flecha"><span class="ms">arrow_forward</span></div>
  {final}
</div>""")


def _cifras(paquete) -> list[tuple[str, str, str]]:
    """(cifra, rótulo, detalle al pasar el ratón): AUC en test, pacientes de test y parámetros."""
    test = ti.cargar()
    mo = paquete.meta["modelo"]
    cifras = []
    if test is not None:
        p = test["niveles"]["paciente"]
        c = p["modelos"]["combinado"]
        cifras += [(num(c["auc"], 2), "AUC en test",
                    f"Modelo final (clínico + CNN), por paciente, en {p['n']} pacientes de test evaluadas una sola "
                    f"vez. IC95 {num(c['auc_ic95'][0])}–{num(c['auc_ic95'][1])}; 0,5 es el azar."),
                   (entero(p["n"]), "pacientes de test", f"{p['pCR_1']} con pCR, apartadas desde el principio.")]
    cifras.append((entero(mo["n_parametros"]), "parámetros, CNN desde cero",
                   f"{mo['bloques']} bloques convolucionales, sin modelos preentrenados."))
    return cifras


def _portada(paquete) -> None:
    # Sin vertical_alignment="center": plotly calcula mal el tamaño dentro de columnas centradas.
    izquierda, derecha = st.columns([1.05, 1], gap="large")
    with izquierda:
        # Bloques <div> y un espacio entre cifra y rótulo: se leen separados aunque falte el estilo.
        cifras = "".join(f'<div class="cifra-portada" title="{html.escape(ayuda)}"><div class="valor">'
                         f'{html.escape(valor)}</div> <div class="rotulo">{html.escape(rotulo)}</div></div>'
                         for valor, rotulo, ayuda in _cifras(paquete))
        st.html(f"""
<div class="portada">
  <div class="antetitulo">Herramienta docente · resonancia de mama</div>
  <h1>Predicción de la respuesta patológica completa con <span class="sin-corte">DCE-MRI</span></h1>
  <p class="sub">Una red neuronal entrenada desde cero lee un corte de resonancia y, junto con el subtipo, el
  volumen del tumor y la edad, estima la probabilidad de respuesta completa al tratamiento.</p>
  <div class="cifras-portada">{cifras}</div>
</div>""")
        if st.button("Analizar un caso", type="primary", icon=":material/arrow_forward:", key="cta_analizar"):
            st.switch_page("paginas/analisis.py")
        st.caption("Uso exclusivamente educativo, sin validez clínica.")
    with derecha, st.container(key="negro_volumen"):
        fig = _figura_volumen()
        if fig is None:
            st.info("No está la pila de cortes del ejemplo 1.")
        else:
            g.mostrar(fig, key="volumen")
            st.caption("Realce tras el contraste en los 10 cortes de una paciente de entrenamiento. Arrastre para girar.")


def _contenido() -> None:
    paquete = comun.modelo_o_parar()
    fusion = comun.fusion_o_none()
    _portada(paquete)

    st.html('<div class="antetitulo" style="margin:2.2rem 0 .9rem">Cómo funciona</div>')
    _esquema(paquete, fusion)

    with comun.mas_detalle("Qué es la pCR y qué es la resonancia DCE"):
        c1, c2 = st.columns(2, gap="large")
        c1.markdown("**pCR** (respuesta patológica completa): tras el tratamiento previo a la cirugía, el patólogo no "
                    "encuentra cáncer invasivo residual. Es un resultado favorable y se determina estudiando el "
                    "tejido, no con la imagen.")
        c2.markdown("**DCE-MRI**: resonancia con contraste en varios instantes. **PRE** antes del contraste, "
                    "**EARLY** poco después y **LATE** minutos más tarde. El **realce** EARLY − PRE muestra dónde "
                    "entra el contraste, que llega antes a los tejidos con más vasos, como los tumores.")


comun.ejecutar(_contenido)
