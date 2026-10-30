"""Validación de los 3 PNG de un corte, con un informe paso a paso en español.

Se comprueba cada fichero antes de pasarlo a preprocesado.preprocesar, la
función única del proyecto (que vuelve a exigir PNG, modo L y 256x256). Nada de
lo subido se ejecuta, se guarda en disco ni se interpreta como ruta: solo se
decodifica en memoria con Pillow, y solo después de comprobar el tamaño del
fichero, la firma PNG y las dimensiones de la cabecera.

Por fase:      fichero recibido, tamaño del fichero, firma PNG, dimensiones
               256x256, un solo fotograma, decodificación completa, escala de
               grises de 8 bits.
Por corte:     tres fases presentes, fases distintas entre sí, nombres
               coherentes con su casilla y con el mismo corte (si siguen el
               patrón <corte>_<FASE>.png).
Tras el preprocesado (inferencia.py): valores finitos en [0, 1] y aviso si PRE
               es más brillante que EARLY en el tejido.

Escala de grises "sin ambigüedad": un PNG en RGB, RGBA, LA o con paleta se
admite solo si sus tres canales son idénticos y no tiene transparencia; se usa
uno de ellos, sin pérdida. 16 bits, 1 bit o coma flotante se rechazan: pasarlos
a 8 bits cambiaría la escala de intensidades.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import rutas  # noqa: F401  (añade la raíz al sys.path)
from .formato import num
from preprocesado import FASES, TAMANO  # noqa: E402

TAMANO_MAXIMO_BYTES = 2 * 1024 * 1024          # un PNG de 256x256 y 8 bits ocupa < 100 KB
FIRMA_PNG = b"\x89PNG\r\n\x1a\n"
FIRMAS_CONOCIDAS = {                            # para decir qué es lo que no es un PNG
    b"\xff\xd8\xff": "JPEG", b"GIF8": "GIF", b"BM": "BMP", b"II*\x00": "TIFF", b"MM\x00*": "TIFF",
    b"%PDF": "PDF", b"PK\x03\x04": "ZIP", b"RIFF": "WebP o RIFF",
}
UMBRAL_TEJIDO = 0.1                             # como en utils_caso: PRE > 0,1 es tejido, no aire
PIXELES_TEJIDO_MINIMOS = 500
PATRON_NOMBRE = re.compile(r"^(?P<corte>.+?)[_\-. ](?P<fase>PRE|EARLY|LATE)\.png$", re.IGNORECASE)

OK, AVISO, ERROR, OMITIDO = "ok", "aviso", "error", "omitido"
CORTE = "Corte"
# Qué hacer ante cada error, por comprobación (también las de nucleo/variables.py e
# inferencia.py). Informe.anadir lo añade al detalle de todo error: el mensaje dice qué
# falla y cómo corregirlo. «Datos de la paciente» ya lo explica en su propio mensaje.
SOLUCION = {
    "Fichero recibido": "suba el PNG de esa fase en su casilla (PRE, EARLY o LATE).",
    "Tamaño del fichero": "suba el PNG original del corte (256 × 256, escala de grises), sin convertirlo ni ampliarlo.",
    "Formato PNG": "exporte la imagen como PNG en escala de grises de 8 bits y 256 × 256, o pruebe con un ejemplo.",
    "Dimensiones": "use el PNG de 256 × 256 del dataset, sin recortarlo ni reescalarlo.",
    "Un solo fotograma": "guarde cada fase como una imagen PNG fija.",
    "Decodificación": "vuelva a descargar o exportar el fichero y súbalo de nuevo.",
    "Escala de grises de 8 bits": "convierta la imagen a escala de grises de 8 bits sin transparencia, o suba el PNG "
                                  "original del dataset.",
    "Nombres de los ficheros": "ponga cada fichero en la casilla de su fase, con los tres PNG del mismo corte.",
    "Tres fases presentes": "suba las tres fases del mismo corte, cada una en su casilla.",
    "Fases distintas": "suba en cada casilla su fase: PRE, EARLY y LATE son imágenes distintas.",
    "Valores finitos": "vuelva a subir los PNG originales del corte.",
    "Preprocesado": "compruebe que las tres imágenes son PNG de 256 × 256 en escala de grises y súbalas de nuevo.",
    "Preprocesado del modelo": "el paquete del modelo no es coherente: vuelva a exportarlo con exportar_modelo.py.",
    "Fichero de variables": "descargue la plantilla CSV, rellene una fila con los datos de la paciente y súbala, o "
                            "use el formulario.",
    "Columnas": "use la plantilla CSV (botón «Descargar plantilla CSV»), que tiene la cabecera exacta.",
    "Subtipo HR/HER2": "escriba HRposHER2neg, HER2pos o TripleNeg, o deje la celda vacía.",
    "Volumen tumoral": "escriba un número (punto o coma decimal) o deje la celda vacía: se rellena con la media de "
                       "entrenamiento.",
    "Edad": "escriba la edad en años o deje la celda vacía: se rellena con la media de entrenamiento.",
    "Variables clínicas": "rellene al menos una de las tres: subtipo, volumen tumoral o edad.",
    "Caja del tumor": "copie n_xy, sraw, eraw, scol, ecol y xy_spacing de la fila de la paciente en patients.csv "
                      "(números, en píxeles del corte original y mm).",
}


@dataclass
class Paso:
    ambito: str          # "PRE", "EARLY", "LATE" o "Corte"
    comprobacion: str
    estado: str          # ok, aviso, error u omitido
    detalle: str


@dataclass
class Informe:
    pasos: list[Paso] = field(default_factory=list)

    def anadir(self, ambito: str, comprobacion: str, estado: str, detalle: str) -> None:
        if estado == ERROR and comprobacion in SOLUCION and "Qué hacer:" not in detalle:
            detalle = detalle.rstrip()
            detalle += ("" if detalle.endswith((".", ")", "»", ":")) else ".") + " Qué hacer: " + SOLUCION[comprobacion]
        self.pasos.append(Paso(ambito, comprobacion, estado, detalle))

    @property
    def errores(self) -> list[Paso]:
        return [p for p in self.pasos if p.estado == ERROR]

    @property
    def avisos(self) -> list[Paso]:
        return [p for p in self.pasos if p.estado == AVISO]

    @property
    def valido(self) -> bool:
        return not self.errores


@dataclass
class FaseValidada:
    nombre_fichero: str
    datos_png: bytes           # PNG en modo L que recibe preprocesado.preprocesar
    pixeles: np.ndarray        # uint8 (256, 256)
    convertida: bool           # True si venía en RGB/RGBA/LA/P con canales idénticos


def _que_es(datos: bytes) -> str | None:
    for firma, tipo in FIRMAS_CONOCIDAS.items():
        if datos.startswith(firma):
            return tipo
    if len(datos) > 132 and datos[128:132] == b"DICM":
        return "DICOM"
    return None


def _png_modo_l(pixeles: np.ndarray) -> bytes:
    salida = io.BytesIO()
    Image.fromarray(pixeles).save(salida, format="PNG")      # uint8 2D -> modo L
    return salida.getvalue()


def validar_fase(fase: str, nombre: str, datos: bytes, informe: Informe) -> FaseValidada | None:
    """Comprueba un fichero; añade sus pasos a `informe` y devuelve la fase lista
    para preprocesar, o None si hay un error (los pasos siguientes se omiten)."""
    if not datos:
        informe.anadir(fase, "Fichero recibido", ERROR, f"«{nombre}» está vacío (0 bytes).")
        return None
    informe.anadir(fase, "Fichero recibido", OK, f"«{nombre}»")

    if len(datos) > TAMANO_MAXIMO_BYTES:
        informe.anadir(fase, "Tamaño del fichero", ERROR,
                       f"{num(len(datos) / 2**20, 1)} MB; el máximo es {TAMANO_MAXIMO_BYTES / 2**20:.0f} MB "
                       "(un PNG de 256 × 256 en 8 bits ocupa menos de 100 KB).")
        return None
    informe.anadir(fase, "Tamaño del fichero", OK, f"{len(datos) / 1024:.0f} KB (máximo "
                                                   f"{TAMANO_MAXIMO_BYTES / 2**20:.0f} MB)")

    if not datos.startswith(FIRMA_PNG):
        tipo = _que_es(datos)
        informe.anadir(fase, "Formato PNG", ERROR,
                       "El contenido no es un PNG" + (f": parece un {tipo}" if tipo else "")
                       + ". Solo se admiten PNG (la extensión del nombre no basta).")
        return None

    try:
        imagen = Image.open(io.BytesIO(datos))
        formato, modo, (ancho, alto) = imagen.format, imagen.mode, imagen.size
        fotogramas = getattr(imagen, "n_frames", 1)
    except Exception:                                            # cabecera ilegible o dañada
        informe.anadir(fase, "Formato PNG", ERROR, "Tiene firma PNG pero la cabecera está dañada o "
                                                   "no se puede leer.")
        return None
    if formato != "PNG":
        informe.anadir(fase, "Formato PNG", ERROR, f"Pillow lo reconoce como {formato}, no como PNG.")
        return None
    informe.anadir(fase, "Formato PNG", OK, "Firma y cabecera PNG correctas")

    if (ancho, alto) != TAMANO:
        informe.anadir(fase, "Dimensiones", ERROR,
                       f"{ancho} × {alto} píxeles; el modelo necesita {TAMANO[0]} × {TAMANO[1]}. "
                       "No se reescala para no alterar la imagen.")
        return None
    informe.anadir(fase, "Dimensiones", OK, f"{ancho} × {alto} píxeles")

    if fotogramas > 1:
        informe.anadir(fase, "Un solo fotograma", ERROR, f"Es un PNG animado de {fotogramas} fotogramas; "
                                                         "cada fase debe ser una sola imagen.")
        return None

    try:
        imagen.load()                                            # decodificación completa, en memoria
    except Exception:
        informe.anadir(fase, "Decodificación", ERROR, "El fichero está dañado o incompleto: "
                                                      "no se puede decodificar la imagen entera.")
        return None
    informe.anadir(fase, "Decodificación", OK, "Imagen íntegra")

    if modo == "L":
        pixeles = np.asarray(imagen, dtype=np.uint8)
        informe.anadir(fase, "Escala de grises de 8 bits", OK, "Un canal, valores de 0 a 255")
        return FaseValidada(nombre, datos, pixeles, convertida=False)
    if modo in ("RGB", "RGBA", "LA", "P", "PA"):
        rgba = np.asarray(imagen.convert("RGBA"))
        if (rgba[..., 3] != 255).any():
            informe.anadir(fase, "Escala de grises de 8 bits", ERROR,
                           f"Tiene transparencia (modo {modo}); el modelo necesita una imagen opaca en grises.")
            return None
        r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
        if not (np.array_equal(r, g) and np.array_equal(g, b)):
            informe.anadir(fase, "Escala de grises de 8 bits", ERROR,
                           f"Es una imagen en color (modo {modo}, los canales R, G y B no coinciden). "
                           "Cada fase es una imagen en escala de grises, no un canal de color.")
            return None
        pixeles = np.ascontiguousarray(r)
        informe.anadir(fase, "Escala de grises de 8 bits", AVISO,
                       f"Guardada en modo {modo} con los tres canales idénticos: se usa uno de ellos, sin pérdida.")
        return FaseValidada(nombre, _png_modo_l(pixeles), pixeles, convertida=True)
    if modo.startswith("I"):
        detalle = ("Tiene 16 bits por píxel (o enteros de 32); el modelo se entrenó con 8 bits (0-255) y "
                   "pasarla a 8 bits cambiaría la escala de intensidades.")
    elif modo == "1":
        detalle = "Es una imagen de 1 bit (solo blanco y negro), no una imagen de resonancia en grises."
    elif modo == "F":
        detalle = "Tiene píxeles en coma flotante; el modelo necesita enteros de 8 bits (0-255)."
    else:
        detalle = f"Modo de imagen {modo} no admitido; se necesita escala de grises de 8 bits."
    informe.anadir(fase, "Escala de grises de 8 bits", ERROR, detalle)
    return None


def _comprobar_nombres(nombres: dict[str, str], informe: Informe) -> None:
    analizados = {f: PATRON_NOMBRE.match(n) for f, n in nombres.items()}
    for fase, m in analizados.items():
        if m and m.group("fase").upper() != fase:
            informe.anadir(CORTE, "Nombres de los ficheros", ERROR,
                           f"La casilla {fase} tiene «{nombres[fase]}», cuyo nombre indica la fase "
                           f"{m.group('fase').upper()}. Compruebe que cada fichero está en su casilla.")
            return
    if len(analizados) == 3 and all(analizados.values()):
        cortes = {m.group("corte") for m in analizados.values()}
        if len(cortes) > 1:
            informe.anadir(CORTE, "Nombres de los ficheros", ERROR,
                           "Según sus nombres, los tres ficheros no son del mismo corte: "
                           + ", ".join(f"«{nombres[f]}»" for f in FASES) + ".")
            return
        informe.anadir(CORTE, "Nombres de los ficheros", OK, "Cada fichero está en la casilla de su fase "
                                                             "y los tres son del mismo corte")
        return
    informe.anadir(CORTE, "Nombres de los ficheros", OMITIDO,
                   "Los nombres no siguen el patrón <corte>_<FASE>.png: no se puede comprobar con ellos "
                   "que cada fichero esté en su casilla.")


def validar_corte(ficheros: dict[str, tuple[str, bytes] | None]) -> tuple[Informe, dict[str, FaseValidada]]:
    """`ficheros`: fase -> (nombre, contenido) o None si falta. Devuelve el
    informe y las fases que han pasado todas sus comprobaciones."""
    informe, validadas = Informe(), {}
    for fase in FASES:
        entrada = ficheros.get(fase)
        if entrada is None:
            informe.anadir(fase, "Fichero recibido", ERROR, f"Falta el fichero de la fase {fase}.")
            continue
        nombre, datos = entrada
        fase_ok = validar_fase(fase, nombre, datos, informe)
        if fase_ok is not None:
            validadas[fase] = fase_ok

    faltan = [f for f in FASES if ficheros.get(f) is None]
    if faltan:
        informe.anadir(CORTE, "Tres fases presentes", ERROR,
                       f"Falta{'n' if len(faltan) > 1 else ''}: {', '.join(faltan)}. "
                       "Un corte son tres PNG: PRE, EARLY y LATE.")
    else:
        informe.anadir(CORTE, "Tres fases presentes", OK, "PRE, EARLY y LATE")

    presentes = [f for f in FASES if f in validadas]
    duplicadas = [(a, b) for i, a in enumerate(presentes) for b in presentes[i + 1:]
                  if np.array_equal(validadas[a].pixeles, validadas[b].pixeles)]
    if duplicadas:
        informe.anadir(CORTE, "Fases distintas", ERROR,
                       "; ".join(f"{a} y {b} son idénticas píxel a píxel" for a, b in duplicadas)
                       + ": probablemente se ha subido el mismo fichero en dos casillas.")
    elif len(presentes) == 3:
        informe.anadir(CORTE, "Fases distintas", OK, "Las tres imágenes son diferentes")

    nombres = {f: ficheros[f][0] for f in FASES if ficheros.get(f) is not None}
    if nombres:
        _comprobar_nombres(nombres, informe)
    return informe, validadas


def comprobar_preprocesado(x: np.ndarray, informe: Informe) -> None:
    """Comprobaciones sobre la salida de preprocesado.preprocesar, (3, 256, 256)."""
    if not np.isfinite(x).all():
        informe.anadir(CORTE, "Valores finitos", ERROR, "La imagen preprocesada tiene valores no finitos "
                                                        "(NaN o infinito).")
        return
    if x.min() < 0 or x.max() > 1:
        informe.anadir(CORTE, "Valores finitos", ERROR, f"Valores fuera de [0, 1] tras dividir entre 255 "
                                                        f"(de {num(x.min())} a {num(x.max())}).")
        return
    informe.anadir(CORTE, "Valores finitos", OK, "Todos finitos, entre 0 y 1 tras dividir entre 255")

    tejido = x[0] > UMBRAL_TEJIDO
    if tejido.sum() < PIXELES_TEJIDO_MINIMOS:
        informe.anadir(CORTE, "PRE más oscura que EARLY en el tejido", AVISO,
                       f"Solo {int(tejido.sum())} píxeles de tejido (PRE > {num(UMBRAL_TEJIDO, 1)}): no se puede "
                       "comprobar el orden de las fases.")
        return
    pre, early = float(x[0][tejido].mean()), float(x[1][tejido].mean())
    if pre >= early:
        informe.anadir(CORTE, "PRE más oscura que EARLY en el tejido", AVISO,
                       f"La media en el tejido es {num(pre)} en PRE y {num(early)} en EARLY. Al inyectar el "
                       "contraste el tejido se ilumina, así que EARLY suele ser más brillante (en el 99,4 % "
                       "de las pacientes del entrenamiento). Compruebe que las fases no están cambiadas.")
    else:
        informe.anadir(CORTE, "PRE más oscura que EARLY en el tejido", OK,
                       f"Media en el tejido: PRE {num(pre)} < EARLY {num(early)}")
