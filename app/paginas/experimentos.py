"""Experimentos y decisiones: línea temporal de lo probado (resumen del registro del
proyecto; el detalle, en el informe, docs/informe.pdf), las curvas de aprendizaje
antes y después de la «receta regularizada» (medias de 5 folds de
experimentos/*/metricas_epoca.csv, escalas fijas), la escalera con el AUC de cada
variante de validación cruzada y la cascada solo cohorte -> clínico -> combinado del
modelo final (modelo_app/fusion.json). El detalle, en desplegables."""

from __future__ import annotations

import html
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nucleo import experimentos as ex
from nucleo.formato import num, numero_g, tabla_texto
from paginas import comun, graficos as g

T_975_4 = 2.776     # t de Student, 4 grados de libertad: IC95 de la media de 5 folds
# Antes: la línea base (lr 1e-3, sin regularización, parada temprana); después: la receta
# regularizada con la misma entrada (imagen completa), para aislar el efecto de la receta.
ANTES, DESPUES = "cv_baseline_suav", "cv_receta_baseline"
# Escalas fijas (las de informe.py para pérdida y AUC por época; épocas 1-20).
ESCALAS_RECETA = {"perdida": (0.3, 0.9), "auc": (0.40, 0.80), "epocas": (1, 20)}

# Resumen fiel del registro del proyecto (detalle en el informe, docs/informe.pdf).
CRONOLOGIA = [
    dict(fecha=date(2026, 10, 12), tipo="Datos", titulo="Auditoría del dataset",
         esperado="Comprobar que los datos están completos y que ninguna paciente aparece en dos splits ni en "
                  "dos folds.",
         resultado="Las 14 comprobaciones, correctas. Las cohortes difieren en adquisición y en tasa de pCR "
                   "(de 0,21 a 0,32).",
         decision="Evaluar siempre por cohorte y frente a «solo cohorte»."),
    dict(fecha=date(2026, 10, 12), tipo="Entrenamiento", titulo="Línea base, 3 épocas",
         esperado="Comprobar que todo el circuito funciona.",
         resultado="AUC 0,563 y sensibilidad 0 con umbral 0,5: solo aprendió la proporción de clases.",
         decision="Entrenar la línea base completa."),
    dict(fecha=date(2026, 10, 12), tipo="Entrenamiento", titulo="Línea base, 30 épocas",
         esperado="Una primera referencia del rendimiento.",
         resultado="AUC 0,603 (IC95 0,53–0,68) frente a 0,534 de «solo cohorte»; sobreajuste desde la época 8.",
         decision="Probar medidas contra el sobreajuste, una cada vez."),
    dict(fecha=date(2026, 10, 21), tipo="Preparación", titulo="Aumento de datos y weight decay",
         esperado="Herramientas contra el sobreajuste que no alteren la señal.",
         resultado="Las tres fases reciben la misma transformación; AdamW sin weight decay = Adam, bit a bit.",
         decision="Entrenar cada medida por separado."),
    dict(fecha=date(2026, 10, 22), tipo="Entrenamiento", titulo="Cuatro entrenamientos en GPU",
         esperado="Que pos_weight, el aumento o el weight decay mejoraran a la línea base.",
         resultado="AUC de 0,577 a 0,598: diferencias compatibles con el ruido.",
         decision="Comparar con validación cruzada de 5 folds."),
    dict(fecha=date(2026, 10, 22), tipo="Validación cruzada", titulo="Línea base frente a aumento",
         esperado="Saber si el aumento ayuda de verdad.",
         resultado="AUC medio 0,597 frente a 0,582; diferencia −0,015 (IC95 −0,054 a +0,025).",
         decision="Aumento de datos descartado."),
    dict(fecha=date(2026, 10, 22), tipo="Decisión", titulo="Criterio auc_suavizado",
         esperado="Dejar de elegir picos aislados en una curva de AUC muy ruidosa.",
         resultado="La época se juzga por la media de su AUC con la anterior y la siguiente.",
         decision="Criterio por defecto para guardar el modelo."),
    dict(fecha=date(2026, 10, 23), tipo="Datos", titulo="Auditoría visual de 60 pacientes",
         esperado="Ver qué contienen de verdad las imágenes.",
         resultado="Corazón en I-SPY2 y Duke, no en I-SPY1; movimiento en 1 de cada 5; caja del tumor fiable.",
         decision="Probar a recortar alrededor de la caja del tumor."),
    dict(fecha=date(2026, 10, 24), tipo="Validación cruzada", titulo="Quinto bloque, realce y recorte",
         esperado="Que más capacidad, el realce explícito o centrar el tumor ayudaran.",
         resultado="Folds juntos: AUC de 0,538 a 0,567 frente a 0,523 de «solo cohorte»; ningún IC por encima de 0.",
         decision="Ninguna variante supera claramente a la línea base."),
    dict(fecha=date(2026, 10, 27), tipo="Validación cruzada", titulo="Tasas de aprendizaje menores",
         esperado="Que aprender más despacio retrasara el sobreajuste.",
         resultado="AUC medio 0,603 (línea base, lr 3e-4), 0,599 y 0,578 (recorte, lr 3e-4 y 1e-4).",
         decision="Sin mejora clara; siguiente paso: la receta regularizada."),
    dict(fecha=date(2026, 10, 28), tipo="Decisión", titulo="Modelo final y aplicación web",
         esperado="Una herramienta que funcione siempre en la defensa.",
         resultado="La app da la misma probabilidad que train.py, bit a bit, y valida la entrada paso a paso.",
         decision="Modelo de la app: cv_baseline_f2; umbral por corte con especificidad ≥ 0,85."),
    dict(fecha=date(2026, 10, 29), tipo="Decisión", titulo="Modelo final: clínico + CNN",
         esperado="Saber si la imagen añade algo a lo que ya dicen el subtipo, el volumen y la edad.",
         resultado="Validación cruzada por paciente: clínico 0,686; combinado con la CNN recortecaja_lr3e-4, 0,707; "
                   "la CNN sola, ~0,58.",
         decision="Modelo final de la app: regresión logística por corte con las variables clínicas y la CNN."),
    dict(fecha=date(2026, 10, 29), tipo="Validación cruzada", titulo="Receta regularizada y ventana centrada",
         esperado="Frenar el sobreajuste (12 épocas fijas, lr con bajada lineal, más regularización, 3 cortes por "
                  "paciente) y conservar el tamaño del tumor (ventana de 160 px sin reescalar).",
         resultado="Con la CNN receta_centrado160, validación cruzada por paciente: clínico 0,686, combinado 0,691; "
                   "la CNN sola, 0,545.",
         decision="Modelo final cerrado: receta_centrado160 (ejecución del fold 0) + variables clínicas."),
    dict(fecha=date(2026, 10, 30), tipo="Decisión", titulo="Evaluación única en test",
         esperado="Medir el modelo congelado en 176 pacientes apartadas desde el principio.",
         resultado="Por paciente: combinado 0,679 (IC95 0,596–0,760), clínico 0,674, CNN sola 0,572, solo cohorte "
                   "0,543.",
         decision="Resultado final; no se cambia nada a la vista del test."),
]
COLOR_TIPO = {"Datos": g.REFERENCIA, "Preparación": g.CATEGORIAS[1], "Entrenamiento": g.CATEGORIAS[0],
              "Validación cruzada": g.ACENTO, "Decisión": g.CATEGORIAS[2]}
DESCRIPCION_VARIANTE = {
    "cv_baseline": "Línea base",
    "cv_aumento": "Aumento geométrico",
    "cv_5bloques": "Quinto bloque",
    "cv_baseline_suav": "Línea base · auc_suavizado",
    "cv_recortecaja": "Recorte por la caja",
    "cv_realce": "Realce explícito",
    "cv_recortecaja_lr3e-4": "Recorte · lr 3e-4 · 40 ép.",
    "cv_recortecaja_lr1e-4": "Recorte · lr 1e-4 · 40 ép.",
    "cv_baseline_lr3e-4": "Línea base · lr 3e-4 · 40 ép.",
    "cv_receta_baseline": "Receta · imagen completa",
    "cv_receta_centrado160": "Receta · ventana 160 (final)",
    "cv_receta_recortecaja": "Receta · recorte por la caja",
    "cv_receta_centrales4": "Receta · 4 cortes centrales",
    "cv_receta24_baseline": "Receta · 24 ép. · imagen completa",
    "cv_receta24_centrado160": "Receta · 24 ép. · ventana 160",
}


def _linea_temporal() -> None:
    hitos = "".join(
        f'<div class="hito" style="--color:{COLOR_TIPO[h["tipo"]]}"><div class="fecha">{h["fecha"]:%d·%m}</div>'
        f'<div class="tipo">{html.escape(h["tipo"])}</div><h4>{html.escape(h["titulo"])}</h4>'
        f'<p>{html.escape(h["decision"])}</p></div>' for h in CRONOLOGIA)
    st.html(f'<div class="linea">{hitos}</div>')


def _figura_escalera(resumen: pd.DataFrame) -> go.Figure:
    r = resumen.sort_values("auc_media").reset_index(drop=True)
    etiquetas = [DESCRIPCION_VARIANTE.get(v, v) for v in r.variante]
    medio = r.auc_dt * T_975_4 / np.sqrt(r.folds)
    fig = go.Figure()
    cohorte = float(r.solo_cohorte_media.iloc[0])
    fig.add_vrect(x0=0.4, x1=cohorte, fillcolor="rgba(139,151,168,.06)", line_width=0)
    fig.add_trace(go.Scatter(x=r.auc_media, y=etiquetas, mode="lines", line=dict(color=g.BORDE_2, width=1.5,
                                                                                    shape="vh"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=r.auc_media, y=etiquetas, mode="markers+text", name="AUC medio de 5 folds (IC95)",
        error_x=dict(type="data", array=medio, color="rgba(34,211,238,.55)", thickness=2, width=5),
        marker=dict(size=12, color=g.ACENTO, symbol="diamond", line=dict(color=g.FONDO, width=1.5)),
        text=[f"{a:.3f}".replace(".", ",") for a in r.auc_media], textposition="top center",
        textfont=dict(family=g.FUENTE, size=11, color=g.TEXTO),
        customdata=np.stack([r.auc_media - medio, r.auc_media + medio, r.auc_dt], axis=1),
        hovertemplate="%{y}<br>AUC medio %{x:.3f}<br>IC95 %{customdata[0]:.3f}–%{customdata[1]:.3f}"
                      "<br>DT %{customdata[2]:.3f}<extra></extra>"))
    fig.add_vline(x=cohorte, line=dict(color=g.REFERENCIA, dash="dash", width=1.5),
                  annotation_text=f"solo cohorte {num(cohorte)}", annotation_position="bottom right",
                  annotation_font=dict(size=11, color=g.TEXTO_2))
    fig.add_vline(x=0.5, line=dict(color=g.BORDE_2, width=1))
    g.estilo(fig, alto=110 + 46 * len(r), titulo="Escalera de variantes · validación cruzada de 5 folds")
    fig.update_xaxes(range=[0.45, 0.7], title_text="AUC por paciente (mejor época de cada fold)")
    fig.update_yaxes(showgrid=False, ticks="", tickfont=dict(family=g.FUENTE, size=12, color=g.TEXTO))
    fig.update_layout(margin=dict(l=200))
    return fig


def _receta() -> None:
    antes, despues = ex.curvas_medias(ANTES), ex.curvas_medias(DESPUES)
    if antes is None or despues is None:
        return
    st.html('<div class="antetitulo" style="margin:1.4rem 0 .3rem">Antes y después de la receta</div>')
    with st.container(key="panel_receta"):
        g.mostrar(g.figura_receta(antes, despues, "Línea base", "Receta regularizada", ESCALAS_RECETA),
                  key="receta")
        st.caption("Media de 5 folds. Con la receta, la pérdida de entrenamiento deja de desplomarse y la de "
                   "validación no se dispara: menos memorización.",
                   help=f"Línea base ({ANTES}): lr 1e-3, sin regularización, parada temprana; se muestran las "
                        f"{len(antes)} épocas que tienen los 5 folds. Receta ({DESPUES}): 12 épocas fijas, lr con "
                        "bajada lineal, weight decay 1e-2, dropout 0,35 y 3 cortes por paciente y época, con la misma "
                        "imagen completa. El modelo final añade la ventana de 160 px.")


def _contenido() -> None:
    comun.modelo_o_parar()
    comun.encabezado("Experimentos y decisiones", "Registro del proyecto",
                     "Qué se probó y qué se decidió. El test se usó una sola vez, al final.")
    _linea_temporal()
    _receta()

    d = ex.tabla()
    resumen = ex.resumen_variantes(d)
    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1, st.container(key="panel_escalera"):
        g.mostrar(_figura_escalera(resumen), key="escalera")
        st.caption("Ninguna variante se separa con claridad de «solo cohorte»: los intervalos se solapan.",
                   help="IC95 de la media de los 5 folds (t de Student, 4 grados de libertad). Cada AUC es el de la "
                        "mejor época de su fold, elegida con la misma validación: algo optimista. Las variantes no "
                        "usan todas el mismo criterio ni el mismo número de épocas.")
    with c2, st.container(key="panel_cascada"):
        fusion = comun.fusion_o_none()
        if fusion is None:
            st.info("La cascada solo cohorte → clínico → combinado necesita modelo_app/fusion.json "
                    "(exportar_fusion.py).")
        else:
            g.mostrar(g.figura_cascada(fusion.meta["validacion_cruzada"]["por_paciente"],
                                       titulo="AUC por paciente · validación cruzada"), key="cascada")
            st.caption(f"Por paciente, validación cruzada de 5 folds; CNN {fusion.configuracion}"
                       + (" (provisional)." if fusion.provisional else ".")
                       + " La imagen añade poco a las variables clínicas.",
                       help="«Solo cohorte» aquí es una regresión con la cohorte, ajustada en cada fold con los "
                            "folds juntos; en la escalera es la media por fold, por eso difiere un poco.")

    with comun.mas_detalle("Cada hito: qué se esperaba, qué salió y qué se decidió"):
        for i, h in enumerate(CRONOLOGIA, start=1):
            st.markdown(f"**{i}. {h['fecha']:%d-%m-%Y} · {h['titulo']}** · *{h['tipo']}*  \n"
                        f"Se esperaba: {h['esperado']}  \nSalió: {h['resultado']}  \nSe decidió: {h['decision']}")
    with comun.mas_detalle("Tabla de variantes"):
        tabla = pd.DataFrame({
            "Variante": resumen.variante.map(lambda v: DESCRIPCION_VARIANTE.get(v, v)),
            "AUC medio": resumen.auc_media, "DT": resumen.auc_dt, "Mín.": resumen.auc_min, "Máx.": resumen.auc_max,
            "AP media": resumen.ap_media, "Solo cohorte": resumen.solo_cohorte_media,
            "Criterio": resumen.criterio, "lr": resumen.lr.map(numero_g),
            "Épocas máx.": resumen.epocas_max.astype(int), "Mejor época media": resumen.mejor_epoca_media,
        })
        st.table(tabla_texto(tabla, {**{k: 3 for k in ("AUC medio", "DT", "Mín.", "Máx.", "AP media", "Solo cohorte")},
                                     "Mejor época media": 1}), hide_index=True)
        otras = d[d.variante.isna()].sort_values("ejecucion")
        if not otras.empty:
            st.caption("Ejecuciones de un solo fold (fold 0):")
            st.table(tabla_texto(pd.DataFrame({
                "Ejecución": otras.nombre, "Fecha": otras.fecha.dt.strftime("%d-%m-%Y"),
                "Dispositivo": otras.dispositivo, "AUC": otras.auc_val, "Solo cohorte": otras.auc_solo_cohorte,
                "Mejor época": otras.mejor_epoca.astype(int), "Épocas": otras.epocas_ejecutadas.astype(int),
            }), {"AUC": 3, "Solo cohorte": 3}), hide_index=True)


comun.ejecutar(_contenido)
