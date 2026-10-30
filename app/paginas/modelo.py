"""El modelo: primero el modelo final combinado (regresión logística con las variables
clínicas y la probabilidad de la CNN, de modelo_app/fusion.json: coeficientes) y
después la CNN: cifras clave, un esquema plano de la arquitectura (medido sobre el
propio modelo) y la ventana de 160 px que recibe la red sobre un ejemplo. La vista 3D,
los hiperparámetros con su motivo y la tabla capa a capa, en desplegables."""

from __future__ import annotations

import html
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nucleo import arquitectura, datos, ejemplos
from nucleo.formato import entero, mayuscula, num, numero_g
from nucleo.imagenes import a_uint8
from nucleo.inferencia import DatosPaciente, necesita_caja
from paginas import comun, graficos as g, visor
from preprocesado import FASES, preprocesar, ventana_centrada

# Tipo de capa -> (nombre en la leyenda, color). Tres categorías validadas y gris para lo auxiliar.
TIPOS = {
    "entrada": ("Entrada", "#8B97A8"),
    "conv": ("Convolución 3 × 3", g.CATEGORIAS[0]),
    "aux": ("Normalización, ReLU, dropout", "#475569"),
    "reduccion": ("Reducción (MaxPool, GAP)", g.CATEGORIAS[1]),
    "salida": ("Capa lineal", g.CATEGORIAS[2]),
}
CARAS = (  # 12 triángulos de un prisma con los vértices 0-7
    [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3],
    [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6],
    [2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 6, 7],
)


def _tipo(nombre: str) -> str:
    if nombre.startswith("Imagen"):
        return "entrada"
    if nombre.startswith("Conv"):
        return "conv"
    if nombre.startswith(("MaxPool", "Promedio")):
        return "reduccion"
    if nombre.startswith("Lineal"):
        return "salida"
    return "aux"


def _dimensiones(salida: tuple, tipo: str) -> tuple[float, float]:
    """(grosor en x, lado en y-z): el lado crece con la resolución espacial y el
    grosor con el número de canales (escala logarítmica)."""
    if len(salida) == 3:
        canales, lado = salida[0], salida[1]
        alto = 1.2 + 15 * np.sqrt(lado / 256)
    else:
        canales, alto = salida[0], 1.2 + 4 * np.log2(max(salida[0], 1)) / 7
    grosor = 0.35 if tipo == "aux" else 0.45 + 0.42 * np.log2(max(canales, 2))
    return grosor, alto


@st.cache_data(show_spinner=False)
def _figura_3d(filas: tuple, pixel_mm: tuple[float, float]) -> go.Figure:
    fig = go.Figure()
    x, etapa_previa, vistos = 0.0, None, set()
    aristas = {"x": [], "y": [], "z": []}
    rotulos = []
    for etapa, nombre, salida, parametros, rf in filas:
        tipo = _tipo(nombre)
        if etapa != etapa_previa:
            x += 2.2 if etapa_previa else 0
            rotulos.append((x, etapa))
            etapa_previa = etapa
        grosor, alto = _dimensiones(salida, tipo)
        x0, x1, h = x, x + grosor, alto / 2
        vx = [x0, x1, x1, x0, x0, x1, x1, x0]
        vy = [-h, -h, h, h, -h, -h, h, h]
        vz = [-h, -h, -h, -h, h, h, h, h]
        texto_rf = (f"{rf} × {rf} px (≈ {num(rf * pixel_mm[0], 0)}–{num(rf * pixel_mm[1], 0)} mm)" if rf
                    else "toda la imagen")
        leyenda, color = TIPOS[tipo]
        fig.add_trace(go.Mesh3d(
            x=vx, y=vy, z=vz, i=CARAS[0], j=CARAS[1], k=CARAS[2], color=color, flatshading=True,
            opacity=0.55 if tipo == "aux" else 0.95, name=leyenda, legendgroup=tipo, showlegend=tipo not in vistos,
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.25, roughness=0.6),
            hovertemplate=(f"<b>{nombre}</b><br>salida {' × '.join(map(str, salida))}<br>parámetros "
                           f"{entero(parametros)}<br>campo receptivo {texto_rf}<extra>{etapa}</extra>")))
        vistos.add(tipo)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            aristas["x"] += [vx[a], vx[b], None]
            aristas["y"] += [vy[a], vy[b], None]
            aristas["z"] += [vz[a], vz[b], None]
        x = x1 + 0.35
    fig.add_trace(go.Scatter3d(**aristas, mode="lines", line=dict(color="rgba(10,15,26,.9)", width=2),
                               hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[r[0] for r in rotulos], y=[0] * len(rotulos), z=[10.5] * len(rotulos),
                               mode="text", text=[r[1] for r in rotulos], textposition="top right",
                               textfont=dict(family=g.FUENTE, size=11, color=g.TEXTO_2), hoverinfo="skip",
                               showlegend=False))
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data",
        camera=dict(eye=dict(x=-0.3, y=-1.5, z=0.6), center=dict(x=-0.1, y=0, z=-0.12))))
    g.estilo(fig, alto=520)
    fig.update_layout(margin=dict(l=0, r=0, t=36, b=0), legend=dict(y=1, yref="paper", yanchor="bottom"))
    return fig


COLOR_ETAPA = {"Entrada": g.REFERENCIA, "Cabeza": g.CATEGORIAS[2]}


def _flujo(capas) -> str:
    """Esquema plano: entrada -> bloques -> cabeza, con la salida de cada etapa."""
    partes = []
    for f in arquitectura.resumen_bloques(capas):
        espacial = "×".join(map(str, f["salida"][1:]))
        if f["etapa"] == "Entrada":
            detalle = f"3 fases<br>{espacial}"
        elif f["etapa"] == "Cabeza":
            detalle = "GAP, lineal<br>1 logit"
        else:
            detalle = f"{f['salida'][0]} filtros<br>{espacial}"
        color = COLOR_ETAPA.get(f["etapa"], g.ACENTO)
        partes.append(f'<div class="capa" style="--color:{color}"><b>{html.escape(f["etapa"])}</b>'
                      f'<span>{detalle}</span></div>')
    return '<div class="flujo">' + '<div class="paso">›</div>'.join(partes) + "</div>"


@st.cache_data(show_spinner=False)
def _recorte_ejemplo(lado: int) -> str | None:
    """El ejemplo 1: EARLY completa (256 px) con la caja del tumor y la ventana de `lado` px
    que recibe la red, y al lado esa ventana tal cual (preprocesado.preprocesar)."""
    info = next(e for e in ejemplos.indice()["ejemplos"] if e["clave"] == "ejemplo_1")
    v = info.get("variables") or {}
    if any(v.get(c) is None for c in ("n_xy", "sraw", "eraw", "scol", "ecol")):
        return None
    fich = ejemplos.cargar("ejemplo_1")
    fuentes = [io.BytesIO(fich[f][1]) for f in FASES]
    completa = preprocesar(*fuentes)
    caja = DatosPaciente(*(v[c] for c in ("n_xy", "sraw", "eraw", "scol", "ecol"))).caja_png()
    ventana = preprocesar(*(io.BytesIO(fich[f][1]) for f in FASES), caja=caja, lado_centrado=lado)
    r0, k0 = ventana_centrada(caja, lado)
    f0, f1, c0, c1 = caja
    pct = lambda x: f"{100 * x / 256:.2f}%"  # noqa: E731
    marco = f"top:{pct(r0)};left:{pct(k0)};width:{pct(lado)};height:{pct(lado)}"
    caja_css = f"top:{pct(f0)};left:{pct(c0)};width:{pct(c1 - c0)};height:{pct(f1 - f0)}"     # filas en vertical
    caja_ventana = (f"top:{100 * (f0 - r0) / lado:.2f}%;left:{100 * (c0 - k0) / lado:.2f}%;"
                    f"width:{100 * (c1 - c0) / lado:.2f}%;height:{100 * (f1 - f0) / lado:.2f}%")
    return f"""
<div class="recorte">
  <figure><div class="lienzo"><img src="{visor.data_uri(a_uint8(completa[1]))}" alt="Corte completo, fase EARLY">
    <div class="marco" style="{marco}"></div><div class="caja" style="{caja_css}"></div></div>
    <figcaption>Corte completo (256 px, EARLY). Discontinua: caja del tumor; recuadro: la ventana.</figcaption></figure>
  <figure><div class="lienzo"><img src="{visor.data_uri(a_uint8(ventana[1]))}" alt="Ventana que recibe la red">
    <div class="caja" style="{caja_ventana}"></div></div>
    <figcaption>Lo que recibe la red: {lado} × {lado} px, sin reescalar (el tumor conserva su tamaño).</figcaption></figure>
</div>"""


def _motivos(meta: dict) -> pd.DataFrame:
    e, mo = meta["entrenamiento"], meta["modelo"]
    fijas = e["criterio_seleccion"] == "ultima"           # «receta regularizada»: épocas fijas, sin elegir
    lado = meta["preprocesado"].get("recorte_centrado")
    receta = ("Parte de la «receta regularizada» para datos escasos y correlacionados: weight decay 1e-2, dropout "
              "0,35, lr con bajada lineal a 0, 3 cortes al azar por paciente y época y 12 épocas fijas.")
    filas = [
        ("Arquitectura", f"{mo['bloques']} bloques, filtros {'-'.join(map(str, mo['filtros']))}, GAP",
         "CNN pequeña desde cero. Un quinto bloque se probó con validación cruzada y no mejoró (AUC medio 0,545)."),
        ("Entrada", (f"ventana de {lado} × {lado} px centrada en la caja del tumor, sin reescalar; " if lado else "")
         + "PRE, EARLY y LATE, divididos entre 255",
         ("El tumor conserva su tamaño en píxeles (el recorte reescalado a la caja lo borraba). " if lado else "")
         + "Las fases son instantes, no colores: nada de normalización ImageNet. El realce explícito no mejoró."),
        ("Pérdida", f"{e['perdida']}, un logit",
         "Obligatoria en el caso. pos_weight solo desplazaba las probabilidades (AUC 0,579 frente a 0,598)."),
        ("Optimizador", f"{e['optimizador']}, lr {numero_g(e['lr'])}"
         + (" con bajada lineal a 0" if e.get("lr_lineal") else ""),
         "Valores habituales; lr 3e-4 y 1e-4 se probaron después (ver «Experimentos y decisiones»)."),
        ("Weight decay", numero_g(e["weight_decay"]),
         receta if fijas else "Con 1e-4 las curvas apenas cambiaron (AUC 0,586)."),
        ("Aumento de datos", "no" if not e["aumento_datos"] else "sí",
         "Volteo y rotación: −0,015 de AUC en validación cruzada (IC95 −0,054 a +0,025). Descartado."),
        ("Lote y épocas", f"batch {e['batch_size']}, {e['epocas_max']} épocas fijas"
         + (f", {e['cortes_por_paciente']} cortes al azar por paciente en cada época" if e.get("cortes_por_paciente")
            else "") if fijas else
         f"batch {e['batch_size']}, hasta {e['epocas_max']} épocas, paciencia {e['paciencia']}",
         "Todas las variantes sobreajustaban entre las épocas 3 y 8: pocas épocas y sin early stopping." if fijas else
         "La pérdida de validación sube entre las épocas 3 y 12: el early stopping corta a tiempo."),
        ("Criterio de selección", "ninguno: la última época" if fijas else e["criterio_seleccion"],
         "Elegir la mejor época con la validación escogía a veces picos de suerte (épocas 21-24)." if fijas else
         "Desde el 2026-10-22, auc_suavizado (media con las épocas vecinas) para no elegir picos aislados."),
        ("Dropout", f"{num(mo['dropout'], 2)} antes de la capa lineal",
         receta if fijas else "Regularización de la línea base."),
        ("Umbral", num(meta["umbral"]["valor"], 4),
         "Mayor sensibilidad con especificidad ≥ 0,85, elegido por corte: la misma entrada que analiza la app."),
    ]
    return pd.DataFrame(filas, columns=["Elemento", "Valor", "Por qué"])


def _modelo_final(fusion) -> None:
    m = fusion.meta
    mo = m["modelo"]
    st.html('<div class="antetitulo" style="margin:.2rem 0 .2rem">Modelo final · clínico + CNN</div>')
    if fusion.provisional:
        comun.aviso_discreto("Modelo combinado PROVISIONAL: " + m["nota_provisional"])
    c = st.columns(4, gap="small")
    comun.tile(c[0], "Combinación", "Logística", delta="fusión tardía",
               help=f"{mo['tipo']}. Una fila por corte de train ({entero(mo['cortes_entrenamiento'])} cortes de "
                    f"{entero(mo['pacientes_entrenamiento'])} pacientes) con las variables de su paciente.")
    comun.tile(c[1], "Variables", "3 + imagen", delta="subtipo · volumen · edad",
               help="Subtipo HR/HER2 (tres indicadores), log(1 + volumen tumoral) y edad, de clinica.py, más el logit "
                    "de la probabilidad de la CNN para el corte. Todas se conocen antes del tratamiento.")
    comun.tile(c[2], "CNN", "5 folds", delta=m["cnn"]["configuracion"],
               help=f"Probabilidades de validación de {m['cnn']['ejecuciones']}: ninguna paciente se predice con una "
                    "red que la haya visto al entrenar.")
    comun.tile(c[3], "Umbral", num(fusion.umbral, 4), delta="especificidad ≥ 0,85",
               help=f"{mayuscula(m['umbral']['regla'])}. {mayuscula(m['umbral']['nota'])}.")
    izquierda, derecha = st.columns([1.25, 1], gap="medium")
    with izquierda, st.container(key="panel_coeficientes"):
        g.mostrar(g.figura_coeficientes(mo["coeficientes"]), key="coeficientes")
    with derecha:
        st.markdown("**Hacia la derecha, más probabilidad de pCR.** El subtipo es lo que más pesa; la imagen, poco.",
                    help="Cada variable se estandariza (media 0, desviación 1 en train) y su coeficiente dice cuánto "
                         "mueve el logit por cada desviación. Los tres indicadores del subtipo suman siempre 1: se "
                         "leen juntos, como diferencias entre subtipos. Los faltantes se rellenan con la media de "
                         "entrenamiento (z = 0). En el análisis, cada caso muestra cuánto aporta cada parte.")
        with comun.mas_detalle("Escalador y coeficientes exactos"):
            esc = mo["escalador"]
            st.table(pd.DataFrame([{"Variable": g.NOMBRE_VARIABLE.get(k, k), "Media": num(esc["media"][k], 3),
                                    "Desviación": num(esc["desv"][k], 3), "Coeficiente": f"{v:+.4f}"}
                                   for k, v in mo["coeficientes"].items()]), hide_index=True)
            st.caption(f"Intercepto {mo['intercepto']:+.4f}. Exportado con exportar_fusion.py el {m['exportado']}.")
    st.html('<div class="antetitulo" style="margin:1.2rem 0 .2rem">La red · CNN</div>')


def _contenido() -> None:
    paquete = comun.modelo_o_parar()
    fusion = comun.fusion_o_none()
    meta = paquete.meta
    capas = arquitectura.capas(meta)
    total = sum(c.parametros for c in capas)
    resumen = datos.resumen()["por_cohorte"]
    pixeles = sorted(v["pixel_png_mm"]["mediana"] for v in resumen.values())
    rf_final = max(c.campo_receptivo or 0 for c in capas)
    e = meta["entrenamiento"]

    comun.encabezado("El modelo", "Arquitectura",
                     ("Tres variables clínicas y una CNN desde cero, combinadas en una regresión logística." if fusion
                      else "Una CNN 2D construida desde cero en PyTorch, sin pesos preentrenados."))
    if fusion is not None:
        _modelo_final(fusion)
    c = st.columns(4, gap="small")
    comun.tile(c[0], "Parámetros", entero(total), help="Parámetros entrenables, contados sobre el modelo cargado.")
    comun.tile(c[1], "Bloques convolucionales", meta["modelo"]["bloques"],
               help="Cada bloque: convolución 3 × 3, normalización por lotes, ReLU y MaxPool 2 × 2.")
    comun.tile(c[2], "Entrada", "×".join(map(str, meta["modelo"]["entrada"])),
               help="Tres canales: PRE, EARLY y LATE (tres instantes, no tres colores), divididos entre 255.")
    comun.tile(c[3], "Campo receptivo", f"{rf_final} px",
               help=f"Zona de la entrada que ve cada valor del último bloque: ≈ {num(rf_final * pixeles[0], 0)}–"
                    f"{num(rf_final * pixeles[-1], 0)} mm con el píxel mediano de cada cohorte. Tras el promedio "
                    "global, la decisión usa toda la imagen.")

    lado = meta["preprocesado"].get("recorte_centrado")
    izquierda, derecha = st.columns([1.35, 1], gap="large")
    with izquierda:
        st.html('<div class="antetitulo" style="margin:.9rem 0 .6rem">Arquitectura</div>' + _flujo(capas))
        st.caption("Cuatro bloques convolucionales: cada uno duplica los filtros y reduce la imagen a la mitad.",
                   help="Cada bloque: convolución 3 × 3, normalización por lotes, ReLU y MaxPool 2 × 2. El promedio "
                        "global resume cada mapa en un número; la capa lineal da un único logit.")
        c = st.columns(3, gap="small")
        comun.tile(c[0], "Épocas", f"{e['epocas_ejecutadas']} fijas" if e["criterio_seleccion"] == "ultima"
                   else f"{e['mejor_epoca']}/{e['epocas_ejecutadas']}",
                   help="Sin elegir época con la validación: el modelo es el de la última." if
                   e["criterio_seleccion"] == "ultima" else f"Criterio {e['criterio_seleccion']}.")
        comun.tile(c[1], "Optimizador", f"{e['optimizador']}",
                   help=f"lr {numero_g(e['lr'])}" + (" con bajada lineal a 0" if e.get("lr_lineal") else "")
                        + f", weight decay {numero_g(e['weight_decay'])}, lote {e['batch_size']}.")
        comun.tile(c[2], "Dropout", num(meta["modelo"]["dropout"], 2), help="Antes de la capa lineal.")
    with derecha:
        if lado and necesita_caja(meta):
            figura = _recorte_ejemplo(int(lado))
            if figura:
                st.html(f'<div class="antetitulo" style="margin:.9rem 0 .6rem">La ventana de {lado} px</div>' + figura)

    filas = tuple((c_.etapa, c_.nombre, tuple(c_.salida), c_.parametros, c_.campo_receptivo) for c_ in capas)
    with comun.mas_detalle("La arquitectura en 3D (interactiva)"):
        with st.container(key="panel_arquitectura"):
            g.mostrar(_figura_3d(filas, (pixeles[0], pixeles[-1])), key="arquitectura_3d")
            st.caption("Cada bloque es una capa: el lado, la resolución espacial; el grosor, los canales. Pase el "
                       "ratón para ver la salida, los parámetros y el campo receptivo; arrastre para girar.")
    with comun.mas_detalle("Hiperparámetros y por qué"):
        st.table(_motivos(meta), hide_index=True)
    with comun.mas_detalle("Capa a capa"):
        tabla = pd.DataFrame([{"Etapa": c_.etapa, "Capa": c_.nombre, "Salida": " × ".join(map(str, c_.salida)),
                               "Parámetros": entero(c_.parametros),
                               "Campo receptivo (px)": str(c_.campo_receptivo) if c_.campo_receptivo else "—"}
                              for c_ in capas])
        st.dataframe(tabla, hide_index=True, width="stretch", height=(len(tabla) + 1) * 35 + 3)
    st.caption(f"Modelo exportado: {meta['ejecucion']} (fold {e['fold_validacion']}).")


comun.ejecutar(_contenido)
