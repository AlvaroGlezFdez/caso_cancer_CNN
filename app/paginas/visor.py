"""Piezas HTML del análisis de un caso: visor de 4 paneles, anillo de probabilidad
y tira de comprobaciones. Estilos en estaticos/estilo.css.

Visor: PRE, EARLY, LATE y realce sobre fondo negro, dibujados en canvas con los
mismos píxeles que recibe la red (los 0-255 originales; el realce con la escala
fija de -0,5 a 0,5). Cursor, zoom (rueda) y desplazamiento (arrastre)
sincronizados en los cuatro paneles; doble clic restablece. La lectura del
píxel sale de los propios PNG: intensidad = valor / 255, como
preprocesado.preprocesar, y realce = (EARLY - PRE) / 255. Con un modelo de
ventana centrada (--recorte-centrado), la ventana que recibe la red se recuadra
y lo de fuera se atenúa.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math

import numpy as np
import streamlit as st

from nucleo.formato import num
from nucleo.imagenes import a_uint8, colorear_realce, png
from nucleo.validacion import AVISO, ERROR, OK, OMITIDO

ROTULOS = [("PRE", "antes del contraste"), ("EARLY", "temprana"), ("LATE", "tardía"), ("REALCE", "EARLY − PRE")]
ICONO_ESTADO = {OK: "check_circle", AVISO: "warning", ERROR: "cancel", OMITIDO: "remove"}
GRAVEDAD = {OMITIDO: 0, OK: 1, AVISO: 2, ERROR: 3}
ETIQUETA_ESTADO = {OK: "correcto", AVISO: "aviso", ERROR: "error", OMITIDO: "no aplicable"}


def data_uri(arr: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png(arr)).decode("ascii")


@st.cache_data(max_entries=16, show_spinner=False)
def _fuentes(x: np.ndarray) -> list[str]:
    return [data_uri(a_uint8(x[0])), data_uri(a_uint8(x[1])), data_uri(a_uint8(x[2])),
            data_uri(colorear_realce(x[1] - x[0]))]


def _celdas() -> str:
    partes = []
    for k, (nombre, detalle) in enumerate(ROTULOS):
        partes.append(f'<div class="celda"><canvas></canvas><canvas></canvas>'
                      f'<div class="rotulo">{nombre} <span>{detalle}</span></div><div class="valor"></div>'
                      + ('<div class="escala"></div>' if k == 3 else "") + "</div>")
    return "".join(partes)


def visor_vacio() -> None:
    st.html(f'<div class="visor vacio"><div class="rejilla">{_celdas()}</div>'
            '<div class="lectura"><span>Esperando las tres fases</span></div></div>')


def visor(x: np.ndarray, caja=None, ventana=None) -> None:
    """x: (3, 256, 256) en [0, 1], la imagen preprocesada. caja: (f0, f1, c0, c1) o None.
    ventana: (fila0, col0, lado) de la ventana que recibe la red (Analisis.ventana) o None."""
    fuentes = _fuentes(x)
    uid = "v" + hashlib.sha1(np.ascontiguousarray(x).tobytes() + str(caja).encode()
                             + str(ventana).encode()).hexdigest()[:10]
    datos = json.dumps({"fuentes": fuentes, "caja": list(map(float, caja)) if caja is not None else None,
                        "ventana": list(map(int, ventana)) if ventana is not None else None})
    st.html(f"""
<div class="visor" id="{uid}"><div class="rejilla">{_celdas()}</div>
<div class="lectura"><span>fila <b data-l="f">—</b> · col <b data-l="c">—</b></span><span>PRE <b data-l="0">—</b></span>
<span>EARLY <b data-l="1">—</b></span><span>LATE <b data-l="2">—</b></span><span>REALCE <b data-l="3">—</b></span>
<span class="ayuda">rueda: zoom · arrastrar: mover · doble clic: restablecer</span></div></div>
<script>
(function () {{
  const raiz = document.getElementById("{uid}");
  if (!raiz || raiz.dataset.listo) return;
  raiz.dataset.listo = "1";
  const D = {datos}, N = 256;
  const celdas = Array.from(raiz.querySelectorAll(".celda"));
  const imgs = [], grises = [];
  let vista = {{s: 1, x: 0, y: 0}}, cursor = null, arrastre = null, cargadas = 0;
  const fmt = (v, signo) => (signo && v > 0 ? "+" : "") + v.toFixed(3).replace(".", ",");
  D.fuentes.forEach((src, k) => {{
    const im = new Image();
    im.onload = () => {{
      imgs[k] = im;
      if (k < 3) {{
        const c = document.createElement("canvas"); c.width = N; c.height = N;
        const g = c.getContext("2d"); g.drawImage(im, 0, 0); grises[k] = g.getImageData(0, 0, N, N).data;
      }}
      if (++cargadas === 4) dibujar();
    }};
    im.src = src;
  }});
  function limitar() {{
    const m = N - N / vista.s;
    vista.x = Math.min(Math.max(vista.x, 0), m); vista.y = Math.min(Math.max(vista.y, 0), m);
  }}
  function dibujar() {{
    const r = window.devicePixelRatio || 1;
    celdas.forEach((celda, k) => {{
      const [base, sup] = celda.querySelectorAll("canvas");
      const w = Math.max(1, Math.round(celda.clientWidth * r));
      if (base.width !== w) {{ base.width = w; base.height = w; sup.width = w; sup.height = w; }}
      const g = base.getContext("2d");
      g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = "#000"; g.fillRect(0, 0, w, w);
      const f = w / N * vista.s;
      g.imageSmoothingEnabled = vista.s < 1.5;
      g.setTransform(f, 0, 0, f, -vista.x * f, -vista.y * f);
      if (imgs[k]) g.drawImage(imgs[k], 0, 0);
      if (D.ventana) {{
        const [r0, k0, L] = D.ventana;
        g.fillStyle = "rgba(0, 0, 0, .6)";
        g.beginPath(); g.rect(-N, -N, 3 * N, 3 * N); g.rect(k0, r0, L, L); g.fill("evenodd");
        g.strokeStyle = "#E5E9F0"; g.lineWidth = 1.2 * r / f; g.strokeRect(k0, r0, L, L);
      }}
      if (D.caja) {{
        const [f0, f1, c0, c1] = D.caja;
        g.strokeStyle = "#38C6B6"; g.lineWidth = 1.6 * r / f; g.setLineDash([5 * r / f, 4 * r / f]);
        g.strokeRect(c0 - 0.5, f0 - 0.5, c1 - c0, f1 - f0); g.setLineDash([]);
      }}
      const o = sup.getContext("2d"); o.clearRect(0, 0, w, w);
      if (cursor) {{
        const px = (cursor.c + 0.5 - vista.x) * f, py = (cursor.f + 0.5 - vista.y) * f;
        o.strokeStyle = "rgba(56, 198, 182, .85)"; o.lineWidth = r;
        o.beginPath(); o.moveTo(px, 0); o.lineTo(px, w); o.moveTo(0, py); o.lineTo(w, py); o.stroke();
        o.strokeStyle = "#38C6B6"; o.strokeRect(px - Math.max(f, 4 * r) / 2, py - Math.max(f, 4 * r) / 2,
                                                 Math.max(f, 4 * r), Math.max(f, 4 * r));
      }}
    }});
  }}
  function leer() {{
    const campo = (k) => raiz.querySelector('[data-l="' + k + '"]');
    if (!cursor || grises.length < 3) {{
      ["f", "c", 0, 1, 2, 3].forEach((k) => campo(k).textContent = "—");
      celdas.forEach((c) => c.querySelector(".valor").textContent = ""); return;
    }}
    const i = (cursor.f * N + cursor.c) * 4;
    const v = [grises[0][i] / 255, grises[1][i] / 255, grises[2][i] / 255];
    v.push((grises[1][i] - grises[0][i]) / 255);
    campo("f").textContent = cursor.f; campo("c").textContent = cursor.c;
    v.forEach((x, k) => {{
      const t = fmt(x, k === 3); campo(k).textContent = t; celdas[k].querySelector(".valor").textContent = t;
    }});
  }}
  function aPixel(ev, celda) {{
    const b = celda.getBoundingClientRect();
    const u = (ev.clientX - b.left) / b.width, v = (ev.clientY - b.top) / b.height;
    return {{u, v, c: vista.x + u * N / vista.s, f: vista.y + v * N / vista.s, w: b.width}};
  }}
  celdas.forEach((celda) => {{
    celda.addEventListener("mousemove", (ev) => {{
      const p = aPixel(ev, celda);
      if (arrastre) {{
        vista.x = arrastre.vx - (ev.clientX - arrastre.x) * N / (vista.s * p.w);
        vista.y = arrastre.vy - (ev.clientY - arrastre.y) * N / (vista.s * p.w); limitar();
      }}
      const c = Math.floor(p.c), f = Math.floor(p.f);
      cursor = (c >= 0 && f >= 0 && c < N && f < N) ? {{c, f}} : null;
      leer(); dibujar();
    }});
    celda.addEventListener("mouseleave", () => {{ cursor = null; leer(); dibujar(); }});
    celda.addEventListener("wheel", (ev) => {{
      ev.preventDefault();
      const p = aPixel(ev, celda);
      const s = Math.min(10, Math.max(1, vista.s * (ev.deltaY < 0 ? 1.25 : 0.8)));
      vista.x = p.c - p.u * N / s; vista.y = p.f - p.v * N / s; vista.s = s; limitar(); dibujar();
    }}, {{passive: false}});
    celda.addEventListener("mousedown", (ev) => {{
      arrastre = {{x: ev.clientX, y: ev.clientY, vx: vista.x, vy: vista.y}}; ev.preventDefault();
    }});
    celda.addEventListener("dblclick", () => {{ vista = {{s: 1, x: 0, y: 0}}; dibujar(); }});
  }});
  window.addEventListener("mouseup", () => {{ arrastre = null; }});
  new ResizeObserver(dibujar).observe(raiz);
}})();
</script>""", unsafe_allow_javascript=True)


def anillo(probabilidad: float, umbral: float, clase: int, frase: str, etiqueta: str = "Probabilidad de pCR",
           tipo: str = "imagen", datos: dict | None = None) -> str:
    """Anillo de probabilidad con el umbral marcado, insignia de clase y frase. Se
    devuelve como HTML para st.markdown (los datos van también como atributos;
    `tipo`: «combinado» o «imagen»)."""
    r, centro, grosor = 86, 110, 13
    circ = 2 * math.pi * r
    arco = max(0.0, min(1.0, probabilidad)) * circ
    ang = math.radians(umbral * 360 - 90)
    x0, y0 = centro + (r - 13) * math.cos(ang), centro + (r - 13) * math.sin(ang)
    x1, y1 = centro + (r + 13) * math.cos(ang), centro + (r + 13) * math.sin(ang)
    marcas = "".join(
        f'<line x1="{centro + (r + 9) * math.cos(math.radians(t * 36 - 90)):.1f}" '
        f'y1="{centro + (r + 9) * math.sin(math.radians(t * 36 - 90)):.1f}" '
        f'x2="{centro + (r + 12) * math.cos(math.radians(t * 36 - 90)):.1f}" '
        f'y2="{centro + (r + 12) * math.sin(math.radians(t * 36 - 90)):.1f}" stroke="#2C4A6B" stroke-width="1"/>'
        for t in range(10))
    texto_clase = "pCR" if clase else "no pCR"
    extra = "".join(f' data-{k}="{html.escape(str(v))}"' for k, v in (datos or {}).items())
    return f"""
<div class="resultado" data-probabilidad="{num(probabilidad, 3)}" data-clase="{texto_clase}" data-tipo="{tipo}"{extra}>
<svg class="anillo" viewBox="-6 -6 232 232" role="img"
     aria-label="Probabilidad de pCR {num(probabilidad, 3)}, umbral {num(umbral, 3)}">
  {marcas}
  <circle cx="{centro}" cy="{centro}" r="{r}" fill="none" stroke="#1A2E47" stroke-width="{grosor}"/>
  <circle cx="{centro}" cy="{centro}" r="{r}" fill="none" stroke="#38C6B6" stroke-width="{grosor}"
          stroke-dasharray="{arco:.2f} {circ:.2f}" transform="rotate(-90 {centro} {centro})"
          stroke-linecap="butt"/>
  <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#EDF2F7" stroke-width="2.5"/>
  <text x="{centro}" y="{centro - 24}" fill="#A9B7C9" font-size="11" font-weight="500"
        text-anchor="middle">{html.escape(etiqueta)}</text>
  <text x="{centro}" y="{centro + 14}" fill="#EDF2F7" font-size="42" font-weight="600"
        text-anchor="middle">{num(probabilidad, 3)}</text>
  <text x="{centro}" y="{centro + 38}" fill="#7A8DA6" font-size="11" text-anchor="middle">umbral {num(umbral, 3)}</text>
</svg>
<span class="insignia {'pcr' if clase else 'nopcr'}">{'pCR' if clase else 'No pCR'}</span>
<div class="frase">{frase}</div>
<div class="exacta">p = {probabilidad!r}</div>
</div>"""


def _chip(estado: str, texto: str, detalle: str) -> str:
    return (f'<span class="chip {estado}" title="{html.escape(detalle)}"><span class="ms">{ICONO_ESTADO[estado]}</span>'
            f'{html.escape(texto)}</span>')


def tira(pasos, titulo: str = "Validación") -> None:
    """Tira de comprobaciones: un chip por fase (resume sus comprobaciones) y uno por
    cada comprobación del corte. El detalle, al pasar el ratón."""
    chips = []
    for fase in ("PRE", "EARLY", "LATE"):
        propios = [p for p in pasos if p.ambito == fase]
        if not propios:
            continue
        peor = max(propios, key=lambda p: GRAVEDAD[p.estado]).estado
        detalle = "\n".join(f"{p.comprobacion}: {ETIQUETA_ESTADO[p.estado]}. {p.detalle}" for p in propios)
        chips.append(_chip(peor, f"{fase} · {sum(p.estado == OK for p in propios)}/{len(propios)}", detalle))
    for p in pasos:
        if p.ambito not in ("PRE", "EARLY", "LATE"):
            chips.append(_chip(p.estado, p.comprobacion, f"{ETIQUETA_ESTADO[p.estado]}. {p.detalle}"))
    correctas = sum(p.estado == OK for p in pasos)
    st.html(f'<div class="tira"><span class="resumen">{html.escape(titulo)} {correctas}/{len(pasos)}</span>'
            f'{"".join(chips)}</div>')
