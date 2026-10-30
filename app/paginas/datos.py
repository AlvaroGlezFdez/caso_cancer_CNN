"""Datos: cohortes, subtipos y pCR (sunburst y tasas), y los hallazgos de la auditoría
visual como tarjetas con una cifra y una miniatura. Solo agregados: ningún dato
individual. Las miniaturas salen de los tres cortes de ejemplo."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nucleo import datos, ejemplos
from nucleo.formato import entero, mayuscula, num, pct
from nucleo.imagenes import a_uint8, colorear_realce
from paginas import comun, graficos as g, visor
from preprocesado import FASES, preprocesar

COHORTES = ["I-SPY1", "I-SPY2", "Duke"]
SUBTIPOS = ["HR+/HER2−", "HER2+", "Triple negativo"]
CIAN = np.array([34, 211, 238], dtype=np.uint8)


def _figura_sunburst(r: dict) -> go.Figure:
    ids, etiquetas, padres, valores, colores = [], [], [], [], []
    for c, color in zip(COHORTES, g.CATEGORIAS):
        sub = r["por_cohorte_subtipo"][c]
        total = sum(v["pCR_0"] + v["pCR_1"] for v in sub.values())
        ids.append(c), etiquetas.append(c), padres.append(""), valores.append(total), colores.append(color)
        for s, v in sub.items():
            ids.append(f"{c}/{s}"), etiquetas.append(s), padres.append(c)
            valores.append(v["pCR_0"] + v["pCR_1"]), colores.append(color)
            for clase, n, col in (("pCR", v["pCR_1"], g.ACENTO), ("no pCR", v["pCR_0"], g.NO_PCR)):
                if n:
                    ids.append(f"{c}/{s}/{clase}"), etiquetas.append(clase), padres.append(f"{c}/{s}")
                    valores.append(n), colores.append(col)
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=etiquetas, parents=padres, values=valores, branchvalues="total",
        marker=dict(colors=colores, line=dict(color=g.FONDO, width=1.5)), insidetextorientation="radial",
        textfont=dict(family=g.FUENTE, size=12), opacity=0.95,
        hovertemplate="<b>%{label}</b><br>%{value} pacientes<br>%{percentParent:.0%} de %{parent}<extra></extra>"))
    g.estilo(fig, alto=470, titulo="Cohorte → subtipo → pCR · pacientes de train", leyenda=False)
    fig.update_layout(margin=dict(l=4, r=4, t=36, b=4))
    return fig


def _figura_tasas(r: dict) -> go.Figure:
    fig = go.Figure()
    for c, color in zip(COHORTES, g.CATEGORIAS):
        sub = r["por_cohorte_subtipo"][c]
        n = [sub.get(s, {"pCR_0": 0, "pCR_1": 0}) for s in SUBTIPOS]
        tasa = [v["pCR_1"] / (v["pCR_0"] + v["pCR_1"]) if v["pCR_0"] + v["pCR_1"] else None for v in n]
        fig.add_trace(go.Bar(
            x=SUBTIPOS, y=tasa, name=c, marker=dict(color=color, line=dict(color=g.FONDO, width=2)),
            text=[pct(t) for t in tasa], textposition="outside", textfont=dict(family=g.FUENTE, size=11, color=g.TEXTO),
            customdata=[[v["pCR_1"], v["pCR_0"] + v["pCR_1"]] for v in n],
            hovertemplate=f"{c} · %{{x}}<br>tasa de pCR %{{y:.0%}}<br>%{{customdata[0]}} de %{{customdata[1]}} "
                          "pacientes<extra></extra>"))
    tasa_global = r["train"]["pCR_1"] / r["train"]["pacientes"]
    fig.add_hline(y=tasa_global, line=dict(color=g.REFERENCIA, dash="dash", width=1),
                  annotation_text=f"train {pct(tasa_global)}", annotation_position="top right",
                  annotation_font=dict(size=11, color=g.TEXTO_2))
    g.estilo(fig, alto=470, titulo="Tasa de pCR por subtipo y cohorte")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 0.62], tickformat=".0%", title_text="pacientes con pCR")
    fig.update_xaxes(showgrid=False)
    return fig


# --------------------------------------------------------------------------- #
# Miniaturas (de los ejemplos)
# --------------------------------------------------------------------------- #

def _rectangulo(rgb: np.ndarray, f0, f1, c0, c1, grosor: int = 2) -> np.ndarray:
    rgb = rgb.copy()
    f0, f1, c0, c1 = (int(round(v)) for v in (f0, f1, c0, c1))
    rgb[f0:f0 + grosor, c0:c1] = rgb[f1 - grosor:f1, c0:c1] = CIAN
    rgb[f0:f1, c0:c0 + grosor] = rgb[f0:f1, c1 - grosor:c1] = CIAN
    return rgb


def _gris_rgb(fase: np.ndarray) -> np.ndarray:
    return np.repeat(a_uint8(fase)[..., None], 3, axis=2)


@st.cache_data(show_spinner=False)
def _miniaturas() -> dict[str, str]:
    indice = {e["clave"]: e for e in ejemplos.indice()["ejemplos"]}
    x = {}
    for clave in indice:
        fich = ejemplos.cargar(clave)
        x[clave] = preprocesar(*(io.BytesIO(fich[f][1]) for f in FASES))
    x1, x2, x3 = x["ejemplo_1"], x["ejemplo_2"], x["ejemplo_3"]
    # Movimiento simulado: la misma fase PRE desplazada 3 píxeles deja bordes dobles rojo/azul.
    movido = colorear_realce(x2[1] - np.roll(x2[0], 3, axis=1))
    f0, f1, c0, c1 = indice["ejemplo_2"]["caja_tumor"]
    m = 28
    lesion = _gris_rgb(x2[1])[max(0, int(f0) - m):int(f1) + m, max(0, int(c0) - m):int(c1) + m]
    caja = _rectangulo(_gris_rgb(x3[1]), *indice["ejemplo_3"]["caja_tumor"])
    recorte = _rectangulo(_gris_rgb(x1[1]), 64, 192, 64, 192)
    return {"corazon": visor.data_uri(colorear_realce(x1[1] - x1[0])), "movimiento": visor.data_uri(movido),
            "lesion": visor.data_uri(lesion), "caja": visor.data_uri(caja), "recorte": visor.data_uri(recorte)}


def _hallazgos(h: dict) -> None:
    mini = _miniaturas()
    co, mo, le, ca = h["corazon"], h["movimiento"], h["lesion_delimitada"], h["caja_tumor"]
    n = co["por_cohorte"] * 3
    recorte = ca["dentro_recorte_central"]["128"]
    comun.tarjetas([
        dict(imagen=mini["corazon"], etiqueta="Corazón o tórax en el campo",
             cifra=f"{sum(co['si'].values())}<small>/{n}</small>", titulo="Una pista de la cohorte",
             texto=f"{co['si']['I-SPY2']}/20 en I-SPY2 y {co['si']['Duke']}/20 en Duke; ninguna en I-SPY1 "
                   f"({co['dudoso']['I-SPY1']} dudosas). Ajeno al tumor."),
        dict(imagen=mini["movimiento"], etiqueta="Movimiento entre fases",
             cifra=f"{sum(mo['si'].values())}<small>/{n}</small>", titulo="Bordes dobles en el realce",
             texto=f"Y {sum(mo['dudoso'].values())} dudosas. Miniatura: simulación con un ejemplo desplazado 3 px."),
        dict(imagen=mini["lesion"], etiqueta="Lesión delimitada visible",
             cifra=f"{sum(le['si'].values())}<small>/{n}</small>", titulo="En el corte central",
             texto=f"Y {sum(le['dudoso'].values())} dudosas: no todas las imágenes muestran la lesión con claridad."),
        dict(imagen=mini["caja"], etiqueta="Caja del tumor de patients.csv",
             cifra=f"{ca['acierta']}<small>/{ca['pacientes_con_lesion_clara']}</small>", titulo="Fiable",
             texto=f"Cae sobre la lesión en todas las pacientes con lesión visible ({ca['errores']} errores)."),
        dict(imagen=mini["recorte"], etiqueta="Caja dentro del recorte central 128",
             cifra=pct(recorte), titulo="Recortar a ciegas pierde el tumor",
             texto=f"Con 192 × 192, el {pct(ca['dentro_recorte_central']['192'])}. En Duke, el "
                   f"{pct(ca['dentro_recorte_128_por_cohorte']['Duke'])} ya cabe en 128."),
    ], columnas_min=190)
    st.caption(mayuscula(h["fuente"]) + ".")


def _contenido() -> None:
    comun.modelo_o_parar()
    r = datos.resumen()
    h = r["hallazgos_auditoria"]
    comun.encabezado("Datos", "BreastDCEDL", "Resonancias DCE de tres estudios con la etiqueta pCR. Solo recuentos "
                     "agregados: ningún dato individual.")
    tr, te = r["train"], r["test"]
    c = st.columns(4, gap="small")
    comun.tile(c[0], "Pacientes de train", entero(tr["pacientes"]))
    comun.tile(c[1], "Cortes de train", entero(tr["cortes"]),
               help="Unos 10 cortes por paciente: más imágenes, no más pacientes independientes.")
    comun.tile(c[2], "Con pCR", f"{tr['pCR_1']} · {pct(tr['pCR_1'] / tr['pacientes'])}",
               help="Desbalance: predecir siempre «no pCR» acertaría el 71 % de las veces.")
    comun.tile(c[3], "Pacientes de test", te["pacientes"], help="Apartadas desde el principio para la evaluación final, hecha una sola vez (ver «Rendimiento»).")

    c1, c2 = st.columns([1, 1.15], gap="medium")
    with c1, st.container(key="panel_sunburst"):
        g.mostrar(_figura_sunburst(r), key="sunburst")
        st.caption("Pulse un sector para ampliarlo.")
    with c2, st.container(key="panel_tasas"):
        g.mostrar(_figura_tasas(r), key="tasas")
        st.caption("HER2+ y triple negativo responden más. La cohorte y el subtipo ya predicen algo sin mirar la imagen.")

    st.html('<div class="antetitulo" style="margin:1rem 0 .5rem">Hallazgos de la auditoría visual</div>')
    _hallazgos(h)

    with comun.mas_detalle("Adquisición por cohorte"):
        pc = r["por_cohorte"]
        st.table(pd.DataFrame([{
            "Cohorte": co,
            "Matriz original (n_xy)": ", ".join(f"{k} ({v})" for k, v in pc[co]["n_xy"].items()),
            "Píxel del PNG (mm)": f"{num(pc[co]['pixel_png_mm']['mediana'], 2)} "
                                  f"[{num(pc[co]['pixel_png_mm']['p5'], 2)}–{num(pc[co]['pixel_png_mm']['p95'], 2)}]",
            "Campo (mm)": f"{num(pc[co]['campo_png_mm']['mediana'], 0)} "
                          f"[{num(pc[co]['campo_png_mm']['p5'], 0)}–{num(pc[co]['campo_png_mm']['p95'], 0)}]",
            "Grosor de corte (mm)": num(pc[co]["grosor_corte_mm"]["mediana"], 1),
        } for co in COHORTES]), hide_index=True)
        st.caption("Mediana [percentiles 5–95]. Cada PNG es una ventana de 256 píxeles del corte original, sin "
                   "reescalar (las dos matrices de 128 se amplían ×2).")
    with comun.mas_detalle("Folds y cortes por paciente"):
        folds = r["folds"]
        st.table(pd.DataFrame([{"Fold": k, "Pacientes": v["pacientes"], "Con pCR": v["pCR_1"],
                                "Tasa de pCR": pct(v["tasa_pCR"])} for k, v in folds.items()]), hide_index=True)
        cortes = r["cortes_por_paciente"]
        st.caption("Cortes por paciente: " + ", ".join(f"{k} cortes en {v}" for k, v in cortes.items())
                   + ". Folds por paciente y estratificados por pCR: ninguna paciente aparece en dos.")
    st.caption("Datos: BreastDCEDL (Fridman et al., Scientific Data, 2026), CC BY-NC 4.0: imágenes desidentificadas "
               "de I-SPY1, I-SPY2 y Duke (TCIA). Uso no comercial y citando la fuente.")


comun.ejecutar(_contenido)
