"""Informe descargable de un caso: un HTML autocontenido (imágenes en base64 y CSS
en línea) que se abre en cualquier navegador y se puede imprimir o guardar en PDF.

Contiene la entrada (nombres de los ficheros y validación), las imágenes, el
resultado, la curva cinética y el mapa de atención si se han calculado, los
datos técnicos del modelo y el aviso de uso. Todo texto que viene del usuario
(los nombres de los ficheros) se escapa: el informe no puede ejecutar nada.
"""

from __future__ import annotations

import base64
import html
import platform
from datetime import datetime

import numpy as np
import torch

from .explicacion import CurvaCinetica, MapaAtencion
from .formato import entero, num, pct
from .imagenes import a_uint8, colorear_realce, png, superponer_atencion
from .inferencia import Analisis
from .validacion import AVISO, ERROR, OK

ESTADOS = {OK: ("Correcto", "#1f7a4d"), AVISO: ("Aviso", "#a15c07"), ERROR: ("Error", "#b42318")}
AVISO_USO = ("Herramienta docente. Uso exclusivamente educativo: sin validez clínica, no es un dispositivo médico "
             "y no sirve para diagnosticar ni para decidir tratamientos.")

CSS = """
body { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; color: #1f2933; margin: 0; background: #f4f6f9; }
main { max-width: 960px; margin: 24px auto; background: #fff; padding: 32px 40px; border: 1px solid #d9e2ec; }
h1 { font-size: 22px; margin: 0 0 4px; color: #12355b; }
h2 { font-size: 16px; margin: 28px 0 10px; color: #12355b; border-bottom: 1px solid #d9e2ec; padding-bottom: 4px; }
.sub { color: #52606d; font-size: 13px; }
.aviso { background: #fff7e6; border-left: 4px solid #d97706; padding: 10px 14px; font-size: 13px; margin: 16px 0; }
.tarjetas { display: flex; gap: 12px; margin: 8px 0 12px; }
.tarjeta { flex: 1; border: 1px solid #d9e2ec; padding: 10px 14px; }
.tarjeta .et { font-size: 12px; color: #52606d; text-transform: uppercase; letter-spacing: .04em; }
.tarjeta .va { font-size: 26px; font-weight: 600; color: #12355b; }
.imagenes { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.imagenes figure { margin: 0; text-align: center; font-size: 12px; color: #52606d; }
.imagenes img { width: 100%; image-rendering: pixelated; border: 1px solid #d9e2ec; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #e4e7eb; vertical-align: top; }
th { background: #f0f4f8; color: #334e68; font-weight: 600; }
.mono { font-family: Consolas, 'Courier New', monospace; font-size: 12px; word-break: break-all; }
footer { margin-top: 28px; font-size: 11.5px; color: #7b8794; }
@media print { body { background: #fff; } main { border: none; margin: 0; } }
"""


def _img(arr: np.ndarray, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(png(arr)).decode()}">'


def _filas(pares) -> str:
    return "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in pares)


def informe_html(analisis: Analisis, meta: dict, sha256: str, origen: str,
                 cinetica: CurvaCinetica | None = None, atencion: MapaAtencion | None = None,
                 opacidad_atencion: float = 0.6, licencia: str | None = None, combinado=None,
                 fusion_meta: dict | None = None) -> str:
    """`combinado` (nucleo.fusion.ResultadoFusion) y `fusion_meta` (fusion.json): el
    resultado del modelo final; sin ellos, el informe es el de la imagen sola."""
    r, x = analisis.resultado, analisis.imagen
    m, u = meta["metricas_validacion"], meta["umbral"]
    e = html.escape
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M")
    clase = "pCR" if r.clase else "no pCR"
    if combinado is not None:
        vc = fusion_meta["validacion_cruzada"]
        clase_c = "pCR" if combinado.clase else "no pCR"
        resultado_html = f"""<div class="tarjetas">
  <div class="tarjeta"><div class="et">Modelo combinado</div><div class="va">{num(combinado.probabilidad)}</div></div>
  <div class="tarjeta"><div class="et">Umbral</div><div class="va">{num(combinado.umbral)}</div></div>
  <div class="tarjeta"><div class="et">Clase</div><div class="va">{clase_c}</div></div>
  <div class="tarjeta"><div class="et">Solo imagen</div><div class="va">{num(r.probabilidad)}</div></div>
</div>
<p>Probabilidad combinada (variables clínicas + imagen) {'mayor o igual que' if combinado.clase else 'menor que'}
el umbral: se clasifica como <strong>{clase_c}</strong>. Es una estimación estadística de un modelo docente con un
rendimiento modesto (AUC de validación cruzada por corte
{num(vc['por_corte']['modelos']['combinado']['auc'])}); no es un diagnóstico ni una predicción segura de la
respuesta al tratamiento.{' Modelo combinado PROVISIONAL (solo para desarrollo).' if fusion_meta.get('provisional') else ''}</p>
<h3>Cuánto aporta cada parte (logit)</h3><table>""" + _filas(
            [("Paciente media (intercepto)", f"{combinado.intercepto:+.3f}")]
            + [(k, f"{v:+.3f}") for k, v in combinado.contribuciones.items()]
            + [("Este caso", f"{combinado.logit:+.3f} → probabilidad {num(combinado.probabilidad)}"),
               ("Rellenadas con la media", e(", ".join(combinado.imputadas)) if combinado.imputadas else "ninguna")]
        ) + "</table>"
    else:
        resultado_html = f"""<div class="tarjetas">
  <div class="tarjeta"><div class="et">Probabilidad de pCR (solo imagen)</div><div class="va">{num(r.probabilidad)}</div></div>
  <div class="tarjeta"><div class="et">Umbral</div><div class="va">{num(r.umbral)}</div></div>
  <div class="tarjeta"><div class="et">Clase</div><div class="va">{clase}</div></div>
</div>
<p>Probabilidad del corte {'mayor o igual que' if r.clase else 'menor que'} el umbral: se clasifica como
<strong>{clase}</strong>. <strong>Es la probabilidad de la imagen sola, no la del modelo final</strong> (que combina
la imagen con las variables clínicas de la paciente). Estimación estadística de un modelo docente con un rendimiento
modesto (AUC de validación {num(m['auc'])}); no es un diagnóstico ni una predicción segura de la respuesta al
tratamiento.</p>"""

    figuras = [_img(a_uint8(x[0]), "PRE"), _img(a_uint8(x[1]), "EARLY"), _img(a_uint8(x[2]), "LATE"),
               _img(colorear_realce(x[1] - x[0]), "Realce EARLY − PRE")]
    pies = ["PRE · antes del contraste", "EARLY · poscontraste temprana", "LATE · poscontraste tardía",
            "Realce EARLY − PRE (−0,5 a 0,5)"]
    bloque_imagenes = "".join(f"<figure>{f}<figcaption>{p}</figcaption></figure>" for f, p in zip(figuras, pies))
    if atencion is not None:
        fondo = analisis.entrada_red[1]
        bloque_imagenes += (f"<figure>{_img(superponer_atencion(fondo, atencion.mapa, opacidad_atencion), 'Grad-CAM')}"
                            f"<figcaption>Mapa de atención (Grad-CAM, hacia "
                            f"{'pCR' if atencion.hacia_pcr else 'no pCR'}), orientativo</figcaption></figure>")

    validacion = "".join(
        f"<tr><td>{e(p.ambito)}</td><td>{e(p.comprobacion)}</td>"
        f"<td style='color:{ESTADOS.get(p.estado, ('—', '#52606d'))[1]}'>{ESTADOS.get(p.estado, ('No aplicable',))[0]}</td>"
        f"<td>{e(p.detalle)}</td></tr>" for p in analisis.informe.pasos)
    nombres = "".join(f"<tr><th>{e(f)}</th><td class='mono'>{e(n)}</td></tr>" for f, n in analisis.nombres.items())

    cinetica_html = ""
    if cinetica is not None:
        cinetica_html = "<h2>Curva cinética</h2><table>" + _filas([
            ("Zona", f"fila {cinetica.fila}, columna {cinetica.columna}; ventana de "
                     f"{2 * cinetica.radio + 1} × {2 * cinetica.radio + 1} píxeles"),
            ("Intensidad media", f"PRE {num(cinetica.pre)} → EARLY {num(cinetica.early)} → LATE {num(cinetica.late)}"),
            ("Realce inicial (EARLY − PRE) / PRE", "—" if cinetica.realce_inicial is None else pct(cinetica.realce_inicial)),
            ("Cambio tardío (LATE − EARLY) / EARLY", "—" if cinetica.cambio_tardio is None else pct(cinetica.cambio_tardio)),
            ("Tipo de curva (orientativo)", e(cinetica.tipo)),
        ]) + "</table>"

    c = m["con_umbral"]
    modelo = _filas([
        ("Ejecución", e(meta["ejecucion"])),
        ("SHA-256 de modelo.pt", f"<span class='mono'>{e(sha256)}</span>"),
        ("Arquitectura", f"CNN desde cero, {meta['modelo']['bloques']} bloques "
                         f"({'-'.join(map(str, meta['modelo']['filtros']))} filtros), "
                         f"{entero(meta['modelo']['n_parametros'])} parámetros"),
        ("Preprocesado", f"{e(meta['preprocesado']['funcion'])}: {e(meta['preprocesado']['normalizacion'])}"),
        ("Umbral", f"{num(u['valor'], 4)}: {e(u['regla'])}"),
        ("Validación interna", f"fold {meta['entrenamiento']['fold_validacion']}, {m['pacientes']} pacientes. "
                               f"AUC {num(m['auc'])} (IC95 {num(m['auc_ic95'][0])}–{num(m['auc_ic95'][1])}); "
                               f"«solo cohorte» {num(m['auc_solo_cohorte'])}; con el umbral, sensibilidad "
                               f"{num(c['sensibilidad'])}, especificidad {num(c['especificidad'])}, "
                               f"precisión {num(c['precision'])}"),
    ])
    tecnico = _filas([
        ("Probabilidad exacta", f"<span class='mono'>{r.probabilidad!r}</span> (logit {r.logit!r})"),
        ("Dispositivo", f"{e(r.dispositivo)} · PyTorch {e(torch.__version__)} · Python {platform.python_version()}"),
        ("Tiempo de respuesta", f"{num(r.tiempo_inferencia_ms, 1)} ms de inferencia; "
                                f"{num(r.tiempo_total_ms, 1)} ms con validación y preprocesado"),
        ("Nivel del umbral", e(u["nivel"])),
    ])
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe del caso · pCR · DCE-MRI</title><style>{CSS}</style></head>
<body><main>
<h1>Informe del caso</h1>
<div class="sub">pCR · DCE-MRI · herramienta docente · generado el {ahora} · origen: {e(origen)}</div>
<div class="aviso"><strong>{e(AVISO_USO)}</strong></div>
<h2>Resultado</h2>
{resultado_html}
<h2>Imágenes</h2><div class="imagenes" style="grid-template-columns: repeat({5 if atencion is not None else 4}, 1fr)">{bloque_imagenes}</div>
{cinetica_html}
<h2>Entrada y validación</h2><table>{nombres}</table><br><table><tr><th>Ámbito</th><th>Comprobación</th>
<th>Resultado</th><th>Detalle</th></tr>{validacion}</table>
<h2>Modelo</h2><table>{modelo}</table>
<h2>Datos técnicos</h2><table>{tecnico}</table>
<footer>{e(licencia) + '<br>' if licencia else ''}{e(AVISO_USO)}</footer>
</main></body></html>"""
