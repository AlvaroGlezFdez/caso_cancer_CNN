"""Gráficos de la app con plotly: una plantilla propia («estacion»), una paleta y coma decimal.

Plantilla: fondo transparente sobre los paneles azul marino, rejilla sutil e Inter
en todo (sin fuente monoespaciada). Paleta: un único acento teal (el modelo, lo
seleccionado); gris para las referencias (azar, «solo cohorte», siempre con línea
discontinua y rotulada); ámbar solo para avisos. Para categorías (cohortes,
subtipos, tipos de capa) hay tres tonos validados para daltonismo sobre el fondo
(OKLab: ΔE ≥ 15 sin daltonismo y ≥ 12 con deuteranopía entre el acento, el violeta y
el magenta). Las imágenes usan las escalas fijas de la auditoría
(nucleo.imagenes): gris 0-1, realce RdBu ±0,5 y atención «inferno».
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

from nucleo.imagenes import COLORES_ATENCION, COLORES_REALCE, ESCALA_REALCE, escala_plotly

FONDO = "#0C1B2E"
PANEL = "#12253C"
BORDE = "#1F3651"
BORDE_2 = "#2C4A6B"
REJILLA = "#172C45"
TEXTO = "#EDF2F7"
TEXTO_2 = "#A9B7C9"
TEXTO_3 = "#7A8DA6"
ACENTO = "#38C6B6"
ACENTO_OSCURO = "#1F8A7F"
AVISO = "#F59E0B"
REFERENCIA = "#7D8796"                      # azar y «solo cohorte» (siempre discontinua)
NO_PCR = "#4A5A70"                          # la clase «no pCR» en histogramas y recuentos
# Categorías: teal oscuro, violeta y magenta.
CATEGORIAS = ["#1F8A7F", "#9085e9", "#d55181"]
# Sin nombres con cifras sin comillas («Source Sans 3»): invalidarían la lista entera en el SVG.
FUENTE = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"

ESCALA_REALCE_PLOTLY = escala_plotly(COLORES_REALCE)
ESCALA_ATENCION_PLOTLY = escala_plotly(COLORES_ATENCION)
ESCALA_GRIS = [[0, "#000000"], [1, "#ffffff"]]

CONFIG = {"displaylogo": False, "scrollZoom": False,
          "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
          "toImageButtonOptions": {"format": "png", "scale": 2}}

_EJE = dict(gridcolor=REJILLA, zerolinecolor=BORDE_2, linecolor=BORDE, tickcolor=BORDE, ticks="outside", ticklen=4,
            title=dict(font=dict(size=12, color=TEXTO_2)), tickfont=dict(family=FUENTE, size=11, color=TEXTO_2))
_EJE_3D = dict(backgroundcolor="rgba(0,0,0,0)", showbackground=False, gridcolor=REJILLA, zerolinecolor=BORDE_2,
               linecolor=BORDE, title=dict(font=dict(size=11, color=TEXTO_2)),
               tickfont=dict(family=FUENTE, size=10, color=TEXTO_3))
pio.templates["estacion"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", separators=",.",
    font=dict(family=FUENTE, size=13, color=TEXTO), colorway=[ACENTO, *CATEGORIAS],
    xaxis=_EJE, yaxis=_EJE,
    scene=dict(xaxis=_EJE_3D, yaxis=_EJE_3D, zaxis=_EJE_3D, bgcolor="rgba(0,0,0,0)"),
    polar=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDE_2, font=dict(family=FUENTE, size=12, color=TEXTO)),
    legend=dict(font=dict(size=12, color=TEXTO_2), bgcolor="rgba(0,0,0,0)"),
    annotationdefaults=dict(font=dict(color=TEXTO_2, size=12)),
))
pio.templates.default = "estacion"


def colorbar(**kw) -> dict:
    return dict(thickness=8, outlinewidth=0, tickfont=dict(family=FUENTE, size=10, color=TEXTO_2), **kw)


def estilo(fig: go.Figure, alto: int = 360, titulo: str | None = None, leyenda: bool = True,
           subtitulos: bool = False) -> go.Figure:
    """Plantilla común. Arriba, en filas que no se pisan: título, leyenda y (si hay
    subgráficos con título) los títulos de los subgráficos."""
    arriba = 8
    titulo_dict = None
    if titulo:
        titulo_dict = dict(text=titulo, x=0, xanchor="left", xref="container", yref="container",
                           y=1 - arriba / alto, yanchor="top", pad=dict(l=6),
                           font=dict(size=13, color=TEXTO_2, family=FUENTE))
        arriba += 24
    leyenda_dict = dict(orientation="h", xanchor="left", x=0, yref="container", y=1 - arriba / alto, yanchor="top")
    if leyenda:
        arriba += 26
    if subtitulos:
        arriba += 24
    # Fondos explícitos: si solo están en la plantilla, Streamlit pone los del tema.
    fig.update_layout(height=alto, template="estacion", title=titulo_dict, showlegend=leyenda, legend=leyenda_dict,
                      margin=dict(l=52, r=16, t=max(arriba + 4, 12), b=44),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def mostrar(fig: go.Figure, key: str | None = None, **kw):
    return st.plotly_chart(fig, config=CONFIG, theme=None, key=key, **kw)


def _ejes_imagen(fig: go.Figure, n: int) -> None:
    """Ejes de imagen: fila 0 arriba, píxeles cuadrados y zoom enlazado entre paneles."""
    for i in range(1, n + 1):
        sx = "" if i == 1 else str(i)
        fig.update_layout({
            f"xaxis{sx}": dict(showgrid=False, zeroline=False, showticklabels=False, ticks="", showline=False,
                               matches=None if i == 1 else "x", constrain="domain"),
            f"yaxis{sx}": dict(showgrid=False, zeroline=False, showticklabels=False, ticks="", showline=False,
                               autorange="reversed", scaleanchor=f"x{sx}", constrain="domain",
                               matches=None if i == 1 else "y"),
        })


# --------------------------------------------------------------------------- #
# Un corte
# --------------------------------------------------------------------------- #

def figura_selector_punto(realce: np.ndarray, punto: tuple[int, int], radio: int, paso: int = 4,
                          alto: int = 430) -> go.Figure:
    """Realce con una rejilla invisible de puntos (cada `paso` píxeles) que se pueden
    pulsar, y la ventana elegida marcada en cian."""
    fig = go.Figure(go.Heatmap(z=np.round(realce, 3), colorscale=ESCALA_REALCE_PLOTLY, zmin=-ESCALA_REALCE,
                               zmax=ESCALA_REALCE, hoverinfo="skip", colorbar=colorbar(tickvals=[-0.5, 0, 0.5])))
    filas, columnas = np.mgrid[paso // 2:realce.shape[0]:paso, paso // 2:realce.shape[1]:paso]
    fig.add_trace(go.Scatter(x=columnas.ravel(), y=filas.ravel(), mode="markers",
                             marker=dict(size=paso + 3, color="rgba(0,0,0,0)"),
                             customdata=np.round(realce[filas.ravel(), columnas.ravel()], 3),
                             hovertemplate="fila %{y} · columna %{x}<br>realce %{customdata:+.3f}"
                                           "<br>pulse para elegir<extra></extra>", showlegend=False))
    fila, columna = punto
    fig.add_shape(type="rect", x0=columna - radio - 0.5, x1=columna + radio + 0.5, y0=fila - radio - 0.5,
                  y1=fila + radio + 0.5, line=dict(color=ACENTO, width=2))
    fig.add_trace(go.Scatter(x=[columna], y=[fila], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="cross-thin-open", size=22, color=ACENTO, line=dict(width=2))))
    estilo(fig, alto=alto, leyenda=False)
    _ejes_imagen(fig, 1)
    fig.update_layout(margin=dict(l=2, r=2, t=2, b=2), dragmode=False, plot_bgcolor="#000")
    return fig


def figura_curva_cinetica(curva, referencia=None, alto: int = 330) -> go.Figure:
    fases = ["PRE", "EARLY", "LATE"]
    fig = go.Figure()
    if referencia is not None:
        fig.add_trace(go.Scatter(x=fases, y=referencia, mode="lines+markers", name="Media del tejido",
                                 line=dict(color=REFERENCIA, width=2, dash="dot"), marker=dict(size=8),
                                 hovertemplate="%{x}: %{y:.3f}<extra>media del tejido</extra>"))
    fig.add_trace(go.Scatter(x=fases, y=[curva.pre, curva.early, curva.late], mode="lines+markers+text",
                             name="Zona elegida", line=dict(color=ACENTO, width=3),
                             marker=dict(size=11, color=ACENTO, line=dict(color=FONDO, width=2)),
                             text=[f"{v:.2f}".replace(".", ",") for v in (curva.pre, curva.early, curva.late)],
                             textposition="top center", textfont=dict(family=FUENTE, size=11, color=TEXTO),
                             hovertemplate="%{x}: %{y:.3f}<extra>zona elegida</extra>"))
    estilo(fig, alto=alto)
    fig.update_yaxes(range=[0, 1.05], title_text="intensidad media (0-1)")
    fig.update_xaxes(showgrid=False)
    return fig


def figura_superficie(realce: np.ndarray, alto: int = 540) -> go.Figure:
    filas, columnas = realce.shape
    fig = go.Figure(go.Surface(
        z=realce, surfacecolor=realce, colorscale=ESCALA_REALCE_PLOTLY, cmin=-ESCALA_REALCE, cmax=ESCALA_REALCE,
        colorbar=colorbar(title=dict(text="realce", side="right", font=dict(size=11, color=TEXTO_2)),
                          tickvals=[-0.5, 0, 0.5], len=0.6),
        hovertemplate="fila %{y} · columna %{x}<br>realce %{z:+.3f}<extra></extra>",
        contours=dict(z=dict(show=False)),
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.35, roughness=0.55, fresnel=0.2),
        lightposition=dict(x=-1000, y=-2000, z=3000)))
    fig.update_layout(scene=dict(
        xaxis=dict(title="columna"), yaxis=dict(title="fila", autorange="reversed"),
        zaxis=dict(title="EARLY − PRE", range=[-1, 1]), aspectmode="manual",
        aspectratio=dict(x=1, y=filas / columnas, z=0.5), camera=dict(eye=dict(x=0.95, y=-1.0, z=0.72))))
    estilo(fig, alto=alto, leyenda=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


def figura_atencion(fondo: np.ndarray, mapa: np.ndarray, opacidad: float, titulo_fondo: str,
                    alto: int = 480) -> go.Figure:
    visible = np.where(mapa >= 0.1, np.round(mapa, 3), np.nan)      # lo irrelevante, transparente
    fig = go.Figure(go.Heatmap(z=np.round(fondo, 3), colorscale=ESCALA_GRIS, zmin=0, zmax=1, showscale=False,
                               hovertemplate=f"{titulo_fondo}<br>fila %{{y}} · columna %{{x}}<br>"
                                             "intensidad %{z:.3f}<extra></extra>"))
    fig.add_trace(go.Heatmap(z=visible, colorscale=ESCALA_ATENCION_PLOTLY, zmin=0, zmax=1, opacity=opacidad,
                             colorbar=colorbar(title=dict(text="atención", side="right",
                                                          font=dict(size=11, color=TEXTO_2)), tickvals=[0, 0.5, 1]),
                             hovertemplate="fila %{y} · columna %{x}<br>atención %{z:.2f}<extra></extra>"))
    estilo(fig, alto=alto, leyenda=False)
    _ejes_imagen(fig, 1)
    fig.update_layout(margin=dict(l=2, r=2, t=2, b=2), plot_bgcolor="#000")
    return fig


# --------------------------------------------------------------------------- #
# Rendimiento
# --------------------------------------------------------------------------- #

TRANSICION = dict(duration=450, easing="cubic-in-out")


def _marcadores_umbral(fig: go.Figure, elegido: tuple, actual: tuple | None, etiqueta_x: str, etiqueta_y: str):
    """Estrella = umbral elegido; punto cian con halo = umbral del selector. Siempre
    las mismas trazas, en el mismo orden, para que plotly anime el desplazamiento."""
    fig.add_trace(go.Scatter(x=[elegido[0]], y=[elegido[1]], mode="markers", showlegend=False,
                             marker=dict(symbol="star", size=16, color=TEXTO, line=dict(color=FONDO, width=1.5)),
                             hovertemplate=f"umbral elegido<br>{etiqueta_x} %{{x:.3f}}<br>{etiqueta_y} %{{y:.3f}}"
                                           "<extra></extra>"))
    x, y = actual if actual is not None and actual[1] is not None else elegido
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", showlegend=False,
                             marker=dict(size=13, color=ACENTO, line=dict(color="rgba(34,211,238,.35)", width=7)),
                             hovertemplate=f"selector<br>{etiqueta_x} %{{x:.3f}}<br>{etiqueta_y} %{{y:.3f}}"
                                           "<extra></extra>"))


def figura_roc(m: dict, punto_umbral: tuple[float, float], punto_actual: tuple[float, float] | None,
               nivel: str, alto: int = 430) -> go.Figure:
    """ROC con segmentos rectos entre puntos (con empates, como en «solo cohorte»,
    la recta es la interpolación correcta: la del AUC trapezoidal)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(color=BORDE_2, dash="dot"), hoverinfo="skip"))
    c = m["curva_roc_solo_cohorte"]
    fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines+markers",
                             name=f"Solo cohorte · {m['auc_solo_cohorte']:.3f}".replace(".", ","),
                             line=dict(color=REFERENCIA, width=1.5, dash="dash"), marker=dict(size=6, color=REFERENCIA),
                             hovertemplate="1 − especificidad %{x:.2f}<br>sensibilidad %{y:.2f}<extra>solo cohorte</extra>"))
    c = m["curva_roc"]
    fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"Modelo · AUC {m['auc']:.3f}".replace(".", ","),
                             line=dict(color=ACENTO, width=2.5), fill="tozeroy", fillcolor="rgba(34,211,238,.06)",
                             hovertemplate="1 − especificidad %{x:.2f}<br>sensibilidad %{y:.2f}<extra>modelo</extra>"))
    _marcadores_umbral(fig, punto_umbral, punto_actual, "1 − especificidad", "sensibilidad")
    estilo(fig, alto=alto, titulo=f"Curva ROC · por {nivel}")
    fig.update_xaxes(range=[0, 1], title_text="1 − especificidad", constrain="domain")
    fig.update_yaxes(range=[0, 1.02], title_text="sensibilidad", scaleanchor="x", constrain="domain")
    fig.update_layout(transition=TRANSICION)
    return fig


def figura_pr(m: dict, punto_umbral: tuple[float, float], punto_actual: tuple[float, float] | None,
              nivel: str, alto: int = 430) -> go.Figure:
    """Precisión-sensibilidad en escalón «hv» (el steps-post de sklearn), con la
    prevalencia (el azar) como línea horizontal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[m["tasa_pCR"]] * 2, mode="lines", showlegend=False,
                             line=dict(color=BORDE_2, dash="dot"), hoverinfo="skip"))
    c = m["curva_pr_solo_cohorte"]
    fig.add_trace(go.Scatter(x=c["sensibilidad"], y=c["precision"], mode="lines", line_shape="hv",
                             name=f"Solo cohorte · {m['average_precision_solo_cohorte']:.3f}".replace(".", ","),
                             line=dict(color=REFERENCIA, width=1.5, dash="dash"),
                             hovertemplate="sensibilidad %{x:.2f}<br>precisión %{y:.2f}<extra>solo cohorte</extra>"))
    c = m["curva_pr"]
    fig.add_trace(go.Scatter(x=c["sensibilidad"], y=c["precision"], mode="lines", line_shape="hv",
                             name=f"Modelo · AP {m['average_precision']:.3f}".replace(".", ","),
                             line=dict(color=ACENTO, width=2.5),
                             hovertemplate="sensibilidad %{x:.2f}<br>precisión %{y:.2f}<extra>modelo</extra>"))
    _marcadores_umbral(fig, punto_umbral, punto_actual, "sensibilidad", "precisión")
    estilo(fig, alto=alto, titulo=f"Precisión-sensibilidad · por {nivel}")
    fig.update_xaxes(range=[0, 1], title_text="sensibilidad")
    fig.update_yaxes(range=[0, 1.02], title_text="precisión")
    fig.update_layout(transition=TRANSICION)
    return fig


def figura_calibracion(m: dict, nivel: str, alto: int = 380) -> go.Figure:
    c = m["calibracion"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                             line=dict(color=BORDE_2, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=c["prob_media"], y=c["fraccion_pCR"], mode="lines+markers", name="Modelo",
                             line=dict(color=ACENTO, width=2),
                             marker=dict(size=9, color=ACENTO, line=dict(color=FONDO, width=1.5)),
                             customdata=c["n"],
                             hovertemplate="probabilidad media %{x:.2f}<br>fracción con pCR %{y:.2f}"
                                           f"<br>%{{customdata}} {nivel}s<extra></extra>"))
    fig.add_hline(y=m["tasa_pCR"], line=dict(color=REFERENCIA, dash="dash", width=1),
                  annotation_text="tasa real", annotation_position="bottom right",
                  annotation_font=dict(size=11, color=TEXTO_2))
    estilo(fig, alto=alto, titulo=f"Calibración · 10 grupos · por {nivel}")
    fig.update_xaxes(range=[0, 1], title_text="probabilidad predicha media")
    fig.update_yaxes(range=[0, 1], title_text="fracción real con pCR")
    return fig


def figura_histograma(m: dict, umbral: float, umbral_actual: float | None, nivel: str, agrupar: int = 2,
                      alto: int = 300) -> go.Figure:
    h = m["histograma"]
    bordes = np.array(h["bordes"])
    n0 = np.add.reduceat(np.array(h["pCR_0"]), np.arange(0, len(h["pCR_0"]), agrupar))
    n1 = np.add.reduceat(np.array(h["pCR_1"]), np.arange(0, len(h["pCR_1"]), agrupar))
    centros = bordes[:-1:agrupar] + (bordes[1] - bordes[0]) * agrupar / 2
    ancho = (bordes[1] - bordes[0]) * agrupar
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centros, y=n0, width=ancho * 0.9, name="no pCR", marker_color=NO_PCR,
                         hovertemplate=f"%{{x:.2f}}: %{{y}} {nivel}s<extra>no pCR</extra>"))
    fig.add_trace(go.Bar(x=centros, y=n1, width=ancho * 0.9, name="pCR", marker_color=ACENTO, opacity=0.85,
                         hovertemplate=f"%{{x:.2f}}: %{{y}} {nivel}s<extra>pCR</extra>"))
    fig.add_vline(x=umbral, line=dict(color=TEXTO, width=1.5))
    if umbral_actual is not None and abs(umbral_actual - umbral) > 1e-9:
        fig.add_vline(x=umbral_actual, line=dict(color=ACENTO, width=2, dash="dash"))
    estilo(fig, alto=alto, titulo=f"Probabilidad por {nivel} según su resultado real")
    fig.update_layout(barmode="overlay", bargap=0)
    hay = (n0 + n1) > 0
    lo, hi = float(centros[hay].min()) - 0.05, float(centros[hay].max()) + 0.05
    fig.update_xaxes(range=[max(0.0, lo), min(1.0, hi)], title_text="probabilidad de pCR")
    fig.update_yaxes(title_text=f"{nivel}s")
    return fig


def figura_cv(folds: list[int], auc: list[float], ic: list[tuple[float, float]], cohorte: list[float],
              destacado: int | None, alto: int = 360) -> go.Figure:
    """Un punto por fold con su IC95 y la referencia «solo cohorte» de cada fold."""
    etiquetas = [f"fold {f}" for f in folds]
    fig = go.Figure()
    media = float(np.mean(cohorte))
    fig.add_trace(go.Scatter(x=etiquetas, y=cohorte, mode="markers",
                             name=f"Solo cohorte (media {media:.3f})".replace(".", ","),
                             marker=dict(symbol="line-ew-open", size=26, color=REFERENCIA, line=dict(width=2.5)),
                             hovertemplate="%{x}<br>solo cohorte %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=etiquetas, y=auc, mode="markers+text", name="Modelo (IC95)",
        error_y=dict(type="data", symmetric=False, array=[i[1] - a for a, i in zip(auc, ic)],
                     arrayminus=[a - i[0] for a, i in zip(auc, ic)], color="rgba(34,211,238,.55)", thickness=2, width=6),
        marker=dict(size=12, color=[ACENTO if f == destacado else ACENTO_OSCURO for f in folds],
                    line=dict(color=ACENTO, width=1.5)),
        text=[f"  {a:.3f}".replace(".", ",") for a in auc], textposition="middle right",
        textfont=dict(family=FUENTE, size=11, color=TEXTO),
        customdata=[[i[0], i[1]] for i in ic],
        hovertemplate="%{x}<br>AUC %{y:.3f}<br>IC95 %{customdata[0]:.3f}–%{customdata[1]:.3f}<extra>modelo</extra>"))
    fig.add_hline(y=media, line=dict(color=REFERENCIA, width=1, dash="dash"))
    fig.add_hline(y=0.5, line=dict(color=BORDE_2, width=1), annotation_text="azar", annotation_position="bottom left",
                  annotation_font=dict(size=11, color=TEXTO_3))
    estilo(fig, alto=alto, titulo="AUC de validación por fold · IC95 de Hanley-McNeil")
    fig.update_yaxes(range=[0.38, 0.78], title_text="AUC por paciente")
    fig.update_xaxes(showgrid=False, range=[-0.5, len(folds) - 0.2])
    return fig


def figura_curvas_aprendizaje(c, mejor: int, inicio: int | None, alto: int = 340) -> go.Figure:
    titulos = ("Pérdida (BCE)", "AUC de validación por paciente")
    fig = make_subplots(rows=1, cols=2, subplot_titles=titulos, horizontal_spacing=0.1)
    fig.add_trace(go.Scatter(x=c.epoca, y=c.perdida_entr, name="entrenamiento", line=dict(color=REFERENCIA, width=2),
                             hovertemplate="época %{x}: %{y:.3f}<extra>entrenamiento</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.epoca, y=c.perdida_val, name="validación", line=dict(color=ACENTO, width=2.5),
                             hovertemplate="época %{x}: %{y:.3f}<extra>validación</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.epoca, y=c.auc_val, name="AUC de validación", line=dict(color=ACENTO, width=2.5),
                             showlegend=False, hovertemplate="época %{x}: AUC %{y:.3f}<extra></extra>"), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=mejor, line=dict(color=TEXTO, width=1.5), row=1, col=col)
        if inicio is not None:
            fig.add_vline(x=inicio, line=dict(color=AVISO, width=1.5, dash="dash"), row=1, col=col)
    for xref, yref in (("x", "y domain"), ("x2", "y2 domain")):
        fig.add_annotation(x=mejor, y=0.98, xref=xref, yref=yref, text=" mejor época", showarrow=False,
                           xanchor="left", yanchor="top", font=dict(size=11, color=TEXTO))
        if inicio is not None and inicio != mejor:
            fig.add_annotation(x=inicio, y=0.04, xref=xref, yref=yref, text=" sobreajuste", showarrow=False,
                               xanchor="left", yanchor="bottom", font=dict(size=11, color=AVISO))
    estilo(fig, alto=alto, subtitulos=True)
    fig.update_xaxes(title_text="época", range=[0.5, max(30, int(c.epoca.max())) + 0.5])
    fig.update_yaxes(range=[0.3, 0.9], row=1, col=1)
    fig.update_yaxes(range=[0.4, 0.8], row=1, col=2)
    for titulo in titulos:
        fig.update_annotations(selector=dict(text=titulo), font=dict(size=11, color=TEXTO_2))
    return fig


# --------------------------------------------------------------------------- #
# Modelo final combinado (modelo_app/fusion.json)
# --------------------------------------------------------------------------- #

COLOR_MODELO = {"cohorte": REFERENCIA, "clinico": CATEGORIAS[1], "cnn": CATEGORIAS[2], "combinado": ACENTO}
NOMBRE_MODELO = {"cohorte": "Solo cohorte", "clinico": "Clínico", "cnn": "Solo CNN", "combinado": "Combinado"}
NOMBRE_VARIABLE = {"sub_HRposHER2neg": "Subtipo HR+/HER2−", "sub_HER2pos": "Subtipo HER2+",
                   "sub_TripleNeg": "Subtipo triple negativo", "log_tum_vol": "Volumen tumoral (log)",
                   "age": "Edad", "logit_cnn": "Imagen (logit de la CNN)"}


def _signo(v: float, d: int = 2) -> str:
    return ("+" if v >= 0 else "−") + f"{abs(v):.{d}f}".replace(".", ",")


def figura_contribuciones(res, alto: int = 270) -> go.Figure:
    """Cascada horizontal del logit: paciente media (intercepto) + cada parte = este caso,
    con el umbral del modelo combinado en la misma escala."""
    grupos = list(res.contribuciones)
    nombres = [f"Paciente media {_signo(res.intercepto)}", *grupos, f"<b>Este caso {_signo(res.logit)}</b>"]
    valores = [res.intercepto, *res.contribuciones.values(), res.logit]
    textos = ["", *(_signo(v) for v in res.contribuciones.values()), ""]
    prob = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    acumulado = np.cumsum([res.intercepto, *res.contribuciones.values()])
    ayuda = ([f"logit {_signo(res.intercepto)} · P {prob(res.intercepto):.3f}".replace(".", ",")]
             + [f"aporta {_signo(v)} al logit · queda en P {prob(a):.3f}".replace(".", ",")
                for v, a in zip(res.contribuciones.values(), acumulado[1:])]
             + [f"logit {_signo(res.logit)} · P {res.probabilidad:.3f}".replace(".", ",")])
    fig = go.Figure(go.Waterfall(
        orientation="h", y=nombres, x=valores, measure=["absolute", *["relative"] * len(grupos), "total"],
        increasing=dict(marker=dict(color=ACENTO)), decreasing=dict(marker=dict(color=CATEGORIAS[2])),
        totals=dict(marker=dict(color=TEXTO_3)), connector=dict(line=dict(color=BORDE_2, width=1)),
        text=textos, textposition="outside", textfont=dict(family=FUENTE, size=11, color=TEXTO),
        customdata=ayuda, hovertemplate="%{y}<br>%{customdata}<extra></extra>"))
    u = float(np.log(res.umbral / (1 - res.umbral)))
    fig.add_vline(x=u, line=dict(color=TEXTO, dash="dot", width=1.2),
                  annotation_text=f"umbral {res.umbral:.3f}".replace(".", ","), annotation_position="top",
                  annotation_font=dict(size=10, color=TEXTO_2))
    extremos = np.concatenate([acumulado, [0, u, res.logit]])
    margen = 0.35 * max(np.ptp(extremos), 0.5)
    estilo(fig, alto=alto, titulo="Cuánto aporta cada parte · escala logit", leyenda=False)
    fig.update_xaxes(range=[extremos.min() - margen, extremos.max() + margen], zeroline=False,
                     title_text="← menos pCR · más pCR →")
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="",
                     tickfont=dict(family=FUENTE, size=12, color=TEXTO))
    fig.update_layout(margin=dict(l=150, r=44, b=40))
    fig.update_traces(cliponaxis=False)              # que las cifras de los extremos no se corten
    return fig


def figura_cascada(nivel: dict, alto: int = 340,
                   titulo: str = "Cascada de AUC · solo cohorte → clínico → combinado") -> go.Figure:
    """Solo cohorte -> clínico -> combinado: AUC con IC95 (barras de error) y, entre
    barras, la mejora con su IC95 pareado. `nivel`: por_corte o por_paciente de fusion.json."""
    orden = ["cohorte", "clinico", "combinado"]
    m = nivel["modelos"]
    auc = [m[k]["auc"] for k in orden]
    lo = [m[k]["auc"] - m[k]["auc_ic95"][0] for k in orden]
    hi = [m[k]["auc_ic95"][1] - m[k]["auc"] for k in orden]
    nombres = [NOMBRE_MODELO[k] for k in orden]
    fig = go.Figure(go.Bar(
        x=nombres, y=auc, width=0.52, marker=dict(color=[COLOR_MODELO[k] for k in orden], line=dict(width=0)),
        error_y=dict(type="data", array=hi, arrayminus=lo, color=TEXTO_2, thickness=1.5, width=6),
        text=[f"{a:.3f}".replace(".", ",") for a in auc], textposition="inside", insidetextanchor="end",
        textfont=dict(family=FUENTE, size=13, color=FONDO),
        customdata=[[m[k]["auc_ic95"][0], m[k]["auc_ic95"][1]] for k in orden],
        hovertemplate="%{x}<br>AUC %{y:.3f}<br>IC95 %{customdata[0]:.3f}–%{customdata[1]:.3f}<extra></extra>"))
    for i, (a, b) in enumerate((("clinico", "cohorte"), ("combinado", "clinico"))):
        d = nivel["diferencias"][f"{a}_menos_{b}"]
        y = max(m[a]["auc_ic95"][1], m[b]["auc_ic95"][1]) + 0.025
        fig.add_annotation(x=i + 0.5, y=y, xref="x", yref="y", showarrow=False, align="center",
                           text=f"<b>{_signo(d['valor'], 3)}</b><br><span style='font-size:10px'>"
                                f"{_signo(d['ic95'][0], 3)} a {_signo(d['ic95'][1], 3)}</span>",
                           font=dict(family=FUENTE, size=12, color=TEXTO))
    fig.add_hline(y=0.5, line=dict(color=BORDE_2, width=1, dash="dot"), layer="below")
    estilo(fig, alto=alto, titulo=titulo, leyenda=False)
    fig.update_yaxes(range=[0.4, 0.85], title_text="AUC")
    fig.update_xaxes(showgrid=False, ticks="", tickfont=dict(family=FUENTE, size=12, color=TEXTO))
    return fig


def figura_roc_modelos(nivel: dict, nombre_nivel: str, alto: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=BORDE_2, dash="dot", width=1),
                             name="azar", hoverinfo="skip"))
    for k in ("cohorte", "clinico", "cnn", "combinado"):
        c, m = nivel["curvas_roc"][k], nivel["modelos"][k]
        fig.add_trace(go.Scatter(
            x=c["fpr"], y=c["tpr"], mode="lines", name=f"{NOMBRE_MODELO[k]} {m['auc']:.3f}".replace(".", ","),
            line=dict(color=COLOR_MODELO[k], width=3 if k == "combinado" else 2,
                      dash="dash" if k == "cohorte" else "solid"),
            hovertemplate=f"{NOMBRE_MODELO[k]}<br>1 − especificidad %{{x:.3f}}<br>sensibilidad %{{y:.3f}}<extra></extra>"))
    estilo(fig, alto=alto, titulo=f"ROC por {nombre_nivel} · validación cruzada")
    fig.update_xaxes(range=[0, 1], title_text="1 − especificidad", constrain="domain")
    fig.update_yaxes(range=[0, 1], title_text="sensibilidad", scaleanchor="x", constrain="domain")
    return fig


def figura_receta(antes, despues, nombre_antes: str, nombre_despues: str, escalas: dict,
                  alto: int = 360) -> go.Figure:
    """Curvas de aprendizaje medias de 5 folds, antes y después de la receta: pérdida
    (entrenamiento continua, validación discontinua) y AUC de validación, con escalas fijas."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("Pérdida (BCE) · continua: entrenamiento, discontinua: validación",
                                        "AUC de validación por paciente"))
    for datos, nombre, color in ((antes, nombre_antes, REFERENCIA), (despues, nombre_despues, ACENTO)):
        fig.add_trace(go.Scatter(x=datos.epoca, y=datos.perdida_entr, mode="lines", name=nombre, legendgroup=nombre,
                                 line=dict(color=color, width=2.5),
                                 hovertemplate=f"{nombre}<br>época %{{x}}<br>pérdida de entrenamiento %{{y:.3f}}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.epoca, y=datos.perdida_val, mode="lines", name=nombre, legendgroup=nombre,
                                 showlegend=False, line=dict(color=color, width=2.5, dash="dash"),
                                 hovertemplate=f"{nombre}<br>época %{{x}}<br>pérdida de validación %{{y:.3f}}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.epoca, y=datos.auc_val, mode="lines+markers", name=nombre, legendgroup=nombre,
                                 showlegend=False, line=dict(color=color, width=2.5), marker=dict(size=6),
                                 hovertemplate=f"{nombre}<br>época %{{x}}<br>AUC %{{y:.3f}}<extra></extra>"),
                      row=1, col=2)
    fig.add_hline(y=0.5, line=dict(color=BORDE_2, width=1, dash="dot"), row=1, col=2)
    estilo(fig, alto=alto, subtitulos=True)
    for anotacion in fig.layout.annotations:
        anotacion.font = dict(size=12, color=TEXTO_2, family=FUENTE)
    fig.update_xaxes(range=[escalas["epocas"][0] - 0.5, escalas["epocas"][1] + 0.5], title_text="época", dtick=2)
    fig.update_yaxes(range=list(escalas["perdida"]), row=1, col=1)
    fig.update_yaxes(range=list(escalas["auc"]), row=1, col=2)
    return fig


def figura_coeficientes(coef: dict, alto: int = 300) -> go.Figure:
    nombres = [NOMBRE_VARIABLE.get(k, k) for k in coef]
    valores = list(coef.values())
    fig = go.Figure(go.Bar(
        x=valores, y=nombres, orientation="h", width=0.6,
        marker=dict(color=[ACENTO if v >= 0 else CATEGORIAS[2] for v in valores], line=dict(width=0)),
        text=[_signo(v, 3) for v in valores], textposition="outside", textfont=dict(family=FUENTE, size=11, color=TEXTO),
        hovertemplate="%{y}<br>coeficiente %{x:.3f}<extra></extra>"))
    lim = max(abs(v) for v in valores) * 1.7          # sitio para las cifras a los dos lados
    estilo(fig, alto=alto, titulo="Coeficientes · variables estandarizadas", leyenda=False)
    fig.update_xaxes(range=[-lim, lim], title_text="← menos pCR · más pCR →", zerolinecolor=TEXTO_3)
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", tickfont=dict(family=FUENTE, size=12, color=TEXTO))
    fig.update_layout(margin=dict(l=170, r=16))
    return fig


def figura_cv_test(test: dict, cv: dict, alto: int = 340) -> go.Figure:
    """AUC con IC95 de cada modelo: validación cruzada en train (gris) frente al test
    independiente (color del modelo). `test` y `cv`: modelo -> {"auc", "auc_ic95"}."""
    orden = ["cohorte", "cnn", "clinico", "combinado"]
    fig = go.Figure()
    for nombre, datos, desplazamiento, color, simbolo in (
            ("Validación cruzada (train)", cv, -0.13, [TEXTO_3] * 4, "circle-open"),
            ("Test independiente", test, 0.13, [COLOR_MODELO[k] for k in orden], "circle")):
        auc = [datos[k]["auc"] for k in orden]
        fig.add_trace(go.Scatter(
            x=auc, y=[i + desplazamiento for i in range(len(orden))], mode="markers", name=nombre,
            marker=dict(size=11, color=color, symbol=simbolo, line=dict(width=2, color=color)),
            error_x=dict(type="data", array=[datos[k]["auc_ic95"][1] - datos[k]["auc"] for k in orden],
                         arrayminus=[datos[k]["auc"] - datos[k]["auc_ic95"][0] for k in orden],
                         color=TEXTO_2 if desplazamiento < 0 else TEXTO, thickness=1.5, width=4),
            customdata=[[NOMBRE_MODELO[k], *datos[k]["auc_ic95"]] for k in orden],
            hovertemplate=f"{nombre}<br>%{{customdata[0]}}: AUC %{{x:.3f}}<br>IC95 %{{customdata[1]:.3f}}–"
                          "%{customdata[2]:.3f}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=BORDE_2, width=1, dash="dot"), layer="below")
    estilo(fig, alto=alto, titulo="AUC por paciente · validación cruzada frente a test")
    fig.update_xaxes(range=[0.4, 0.85], title_text="AUC (IC95)")
    fig.update_yaxes(tickvals=list(range(len(orden))), ticktext=[NOMBRE_MODELO[k] for k in orden], showgrid=False,
                     ticks="", range=[-0.6, len(orden) - 0.4], tickfont=dict(family=FUENTE, size=12, color=TEXTO))
    fig.update_layout(margin=dict(l=110, r=16), legend=dict(orientation="h", y=-0.22, x=0))
    return fig
