"""Variables de la paciente: lectura (CSV de una fila con las columnas de
patients.csv, o formulario) y validación con mensajes claros, paso a paso.

Solo comprueba que los valores se pueden usar; las variables del modelo las
construye clinica.construir_variables (nucleo/fusion.py), sin duplicar esa lógica.

    Clínicas (clinica.COLUMNAS_ENTRADA)   HR_HER2_STATUS, tum_vol, age
    Caja del tumor (solo si la CNN        n_xy, sraw, eraw, scol, ecol y xy_spacing
    recorta por la caja)

Errores (no se analiza con el modelo combinado): CSV ilegible, más de una fila,
columnas que faltan, valor no numérico, subtipo desconocido, valor imposible,
caja invertida o fuera del corte, o falta la caja con un modelo que la necesita.
Avisos (se analiza): valor clínico vacío (se rellena con la media de
entrenamiento, como en el ajuste del modelo), valor fuera de lo habitual en
entrenamiento, columnas que no se usan (la pCR se ignora) o HR/HER2 que no
cuadran con el subtipo.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import pandas as pd

from . import rutas  # noqa: F401  (añade la raíz al sys.path)
from .formato import mayuscula, num
from .inferencia import DatosPaciente
from .validacion import AVISO, ERROR, OK, Informe
from clinica import COLUMNAS_ENTRADA, SUBTIPOS  # noqa: E402

AMBITO = "Paciente"
COLUMNAS_CAJA = ("n_xy", "sraw", "eraw", "scol", "ecol", "xy_spacing")
TAMANO_MAXIMO_CSV = 64 * 1024
ETIQUETA = {"HR_HER2_STATUS": "Subtipo HR/HER2", "tum_vol": "Volumen tumoral", "age": "Edad",
            "n_xy": "n_xy", "sraw": "sraw", "eraw": "eraw", "scol": "scol", "ecol": "ecol",
            "xy_spacing": "xy_spacing"}
NOMBRE_SUBTIPO = {"HRposHER2neg": "HR+ / HER2−", "HER2pos": "HER2+", "TripleNeg": "Triple negativo"}
# Formas habituales de escribir el subtipo (en minúsculas y sin espacios) -> valor de patients.csv.
SINONIMOS_SUBTIPO = {
    **{s.lower(): s for s in SUBTIPOS},
    "hr+/her2-": "HRposHER2neg", "hr+her2-": "HRposHER2neg", "hr+/her2−": "HRposHER2neg",
    "her2+": "HER2pos", "triplenegativo": "TripleNeg", "triple-negativo": "TripleNeg", "tn": "TripleNeg",
}
# Límites de lo posible (fuera: error) y, para la caja, lo visto en el dataset (fuera: aviso).
POSIBLE = {"tum_vol": (0.0, 2000.0), "age": (18.0, 100.0), "n_xy": (64.0, 1024.0), "xy_spacing": (0.1, 3.0)}
N_XY_DATASET = (128, 256, 312, 320, 384, 448, 512)
XY_SPACING_DATASET = (0.3125, 1.40625)


@dataclass
class VariablesPaciente:
    valores: dict = field(default_factory=dict)     # columna -> valor normalizado (str, float o None)
    informe: Informe = field(default_factory=Informe)
    origen: str = ""

    @property
    def valido(self) -> bool:
        return self.informe.valido

    def tiene_caja(self) -> bool:
        return all(self.valores.get(c) is not None for c in COLUMNAS_CAJA[:5])

    def clave(self) -> tuple:
        """Hashable, para las cachés de la interfaz."""
        return tuple((c, self.valores.get(c)) for c in (*COLUMNAS_ENTRADA, *COLUMNAS_CAJA))

    def caja_cruda(self) -> tuple | None:
        return tuple(self.valores[c] for c in COLUMNAS_CAJA[:5]) if self.tiene_caja() else None

    def datos_paciente(self) -> DatosPaciente:
        return DatosPaciente.desde_fila(dict(self.valores))


# --------------------------------------------------------------------------- #
# Lectura
# --------------------------------------------------------------------------- #

def leer_csv(nombre: str, datos: bytes, necesita_caja: bool) -> tuple[dict | None, Informe]:
    """CSV de una fila con columnas de patients.csv -> (columna -> texto, informe).
    Acepta coma, punto y coma o tabulador como separador, y UTF-8 o Latin-1."""
    informe = Informe()
    if not datos:
        informe.anadir(AMBITO, "Fichero de variables", ERROR, f"«{nombre}» está vacío.")
        return None, informe
    if len(datos) > TAMANO_MAXIMO_CSV:
        informe.anadir(AMBITO, "Fichero de variables", ERROR,
                       f"«{nombre}» ocupa {len(datos) // 1024} KB: una fila de patients.csv ocupa menos de 1 KB.")
        return None, informe
    for codificacion in ("utf-8-sig", "cp1252", "latin-1"):     # latin-1 no falla nunca
        try:
            texto = datos.decode(codificacion)
            break
        except UnicodeDecodeError:
            continue
    try:
        tabla = pd.read_csv(io.StringIO(texto), sep=None, engine="python", dtype=str, keep_default_na=False)
    except Exception as e:                                          # pandas lanza varios tipos
        informe.anadir(AMBITO, "Fichero de variables", ERROR,
                       f"«{nombre}» no se puede leer como CSV ({type(e).__name__}). Debe tener una fila de "
                       "cabecera con los nombres de las columnas de patients.csv y una fila de datos.")
        return None, informe
    tabla.columns = [str(c).strip() for c in tabla.columns]
    tabla = tabla[[c for c in tabla.columns if not c.startswith("Unnamed")]]
    tabla = tabla[~(tabla.apply(lambda f: f.str.strip()).eq("").all(axis=1))]   # filas vacías
    if len(tabla) != 1:
        informe.anadir(AMBITO, "Fichero de variables", ERROR,
                       f"«{nombre}» tiene {len(tabla)} filas de datos: debe tener exactamente una (una paciente).")
        return None, informe
    necesarias = list(COLUMNAS_ENTRADA) + (list(COLUMNAS_CAJA) if necesita_caja else [])
    faltan = [c for c in necesarias if c not in tabla.columns]
    if faltan:
        informe.anadir(AMBITO, "Columnas", ERROR,
                       f"Faltan las columnas {', '.join(faltan)}"
                       + (" (la caja del tumor es necesaria: este modelo recorta la imagen por ella)"
                          if any(c in COLUMNAS_CAJA for c in faltan) else "")
                       + ". Los nombres deben ser los de patients.csv.")
        return None, informe
    usadas = set(COLUMNAS_ENTRADA) | set(COLUMNAS_CAJA) | {"HR", "HER2"}   # HR y HER2: solo para comprobar el subtipo
    ignoradas = [c for c in tabla.columns if c not in usadas]
    detalle = f"{len(usadas & set(tabla.columns))} columnas reconocidas"
    if ignoradas:
        detalle += f"; no se usan otras {len(ignoradas)}"
    informe.anadir(AMBITO, "Columnas", OK, detalle + ".")
    if "pCR" in tabla.columns:
        informe.anadir(AMBITO, "Columna pCR", AVISO,
                       "El fichero trae la respuesta real (pCR): se ignora. El modelo no la usa y la app no la muestra.")
    fila = tabla.iloc[0]
    return {c: fila[c].strip() for c in tabla.columns if c in usadas}, informe


# --------------------------------------------------------------------------- #
# Validación
# --------------------------------------------------------------------------- #

def _numero(texto) -> float | None | str:
    """None si está vacío; float si es un número (admite coma decimal); el texto
    original si no se puede convertir."""
    if texto is None:
        return None
    if isinstance(texto, (int, float)):
        return None if (isinstance(texto, float) and math.isnan(texto)) else float(texto)
    t = str(texto).strip()
    if t == "" or t.lower() in ("nan", "na", "n/a", "none", "null", "-"):
        return None
    try:
        v = float(t.replace(",", ".") if t.count(",") == 1 and "." not in t else t)
    except ValueError:
        return t
    return None if math.isnan(v) else v


def _subtipo(texto) -> str | None | bool:
    if texto is None or (isinstance(texto, float) and math.isnan(texto)) or str(texto).strip() == "":
        return None
    return SINONIMOS_SUBTIPO.get(str(texto).strip().lower().replace(" ", ""), False)


def validar(entrada: dict, necesita_caja: bool, rango: dict | None = None, origen: str = "",
            informe: Informe | None = None) -> VariablesPaciente:
    """`entrada`: columna -> texto o número (del CSV, del formulario o del ejemplo).
    `rango`: percentiles 1 y 99 de entrenamiento (fusion.json, rango_entrenamiento)."""
    inf = informe or Informe()
    valores: dict = {}

    sub = _subtipo(entrada.get("HR_HER2_STATUS"))
    if sub is False:
        inf.anadir(AMBITO, ETIQUETA["HR_HER2_STATUS"], ERROR,
                   f"«{entrada['HR_HER2_STATUS']}» no es un subtipo válido. Valores admitidos: "
                   + ", ".join(f"{s} ({NOMBRE_SUBTIPO[s]})" for s in SUBTIPOS) + ".")
    elif sub is None:
        valores["HR_HER2_STATUS"] = None
        inf.anadir(AMBITO, ETIQUETA["HR_HER2_STATUS"], AVISO,
                   "Sin dato: se rellena con la media de entrenamiento, como en el ajuste del modelo. El resultado "
                   "es menos fiable.")
    else:
        valores["HR_HER2_STATUS"] = sub
        inf.anadir(AMBITO, ETIQUETA["HR_HER2_STATUS"], OK, f"{sub} ({NOMBRE_SUBTIPO[sub]}).")
        hr, her2 = _numero(entrada.get("HR")), _numero(entrada.get("HER2"))
        if isinstance(hr, float) and isinstance(her2, float):
            esperado = "HER2pos" if her2 == 1 else ("HRposHER2neg" if hr == 1 else "TripleNeg")
            if esperado != sub:
                inf.anadir(AMBITO, "HR y HER2", AVISO,
                           f"HR = {hr:g} y HER2 = {her2:g} corresponden a {esperado}, no a {sub}. Se usa "
                           "HR_HER2_STATUS, la columna con la que se entrenó.")

    for col in ("tum_vol", "age"):
        v = _numero(entrada.get(col))
        lo, hi = POSIBLE[col]
        if isinstance(v, str):
            inf.anadir(AMBITO, ETIQUETA[col], ERROR, f"«{v}» no es un número.")
        elif v is None:
            valores[col] = None
            inf.anadir(AMBITO, ETIQUETA[col], AVISO,
                       "Sin dato: se rellena con la media de entrenamiento, como en el ajuste del modelo. El "
                       "resultado es menos fiable.")
        elif not (lo < v <= hi):
            inf.anadir(AMBITO, ETIQUETA[col], ERROR,
                       f"{num(v, 2)} no es plausible: debe estar entre {num(lo, 0)} y {num(hi, 0)}"
                       + (" años." if col == "age" else " (unidades de tum_vol en patients.csv)."))
        else:
            valores[col] = v
            r = (rango or {}).get(col)
            if r and not (r["p01"] <= v <= r["p99"]):
                inf.anadir(AMBITO, ETIQUETA[col], AVISO,
                           f"{num(v, 2)} queda fuera de lo habitual en entrenamiento (percentiles 1-99: "
                           f"{num(r['p01'], 1)}-{num(r['p99'], 1)}): el modelo extrapola.")
            else:
                inf.anadir(AMBITO, ETIQUETA[col], OK, num(v, 2) + (" años" if col == "age" else ""))
    if all(valores.get(c, "x") is None for c in COLUMNAS_ENTRADA):
        inf.anadir(AMBITO, "Variables clínicas", ERROR,
                   "Faltan las tres variables clínicas (subtipo, volumen y edad): el modelo combinado no tiene con "
                   "qué trabajar.")

    _validar_caja(entrada, necesita_caja, valores, inf)
    return VariablesPaciente(valores=valores, informe=inf, origen=origen)


def _validar_caja(entrada: dict, necesita_caja: bool, valores: dict, inf: Informe) -> None:
    crudos = {c: _numero(entrada.get(c)) for c in COLUMNAS_CAJA}
    if not necesita_caja and all(v is None for v in crudos.values()):
        return
    malos = [f"{c} = «{v}»" for c, v in crudos.items() if isinstance(v, str)]
    if malos:
        inf.anadir(AMBITO, "Caja del tumor", ERROR, "No son números: " + ", ".join(malos) + ".")
        return
    faltan = [c for c in COLUMNAS_CAJA if crudos[c] is None]
    if faltan:
        if necesita_caja:
            inf.anadir(AMBITO, "Caja del tumor", ERROR,
                       f"Faltan {', '.join(faltan)}: este modelo recorta la imagen alrededor de la caja del tumor "
                       "y la necesita completa.")
        else:
            inf.anadir(AMBITO, "Caja del tumor", AVISO, f"Incompleta (faltan {', '.join(faltan)}): no se usa.")
        return
    n, s0, s1, c0, c1, esp = (crudos[c] for c in COLUMNAS_CAJA)
    lo, hi = POSIBLE["n_xy"]
    if not (lo <= n <= hi) or n != int(n):
        inf.anadir(AMBITO, "Caja del tumor", ERROR,
                   f"n_xy = {num(n, 1)} no es plausible: un número entero de píxeles entre {lo:.0f} y {hi:.0f}.")
        return
    if not (0 <= s0 < s1 <= n and 0 <= c0 < c1 <= n):
        inf.anadir(AMBITO, "Caja del tumor", ERROR,
                   f"La caja no cabe en el corte o está invertida: deben cumplirse 0 ≤ sraw < eraw ≤ n_xy y "
                   f"0 ≤ scol < ecol ≤ n_xy (filas {s0:g}-{s1:g}, columnas {c0:g}-{c1:g}, n_xy {n:g}).")
        return
    lo, hi = POSIBLE["xy_spacing"]
    if not (lo <= esp <= hi):
        inf.anadir(AMBITO, "Caja del tumor", ERROR,
                   f"xy_spacing = {num(esp, 3)} mm no es plausible (entre {num(lo, 1)} y {num(hi, 1)} mm).")
        return
    for c in COLUMNAS_CAJA:
        valores[c] = crudos[c]
    from caja_tumor import caja_en_png
    f0, f1, k0, k1 = caja_en_png(DatosPaciente(n_xy=n, sraw=s0, eraw=s1, scol=c0, ecol=c1))
    pixel_png = esp if n >= 256 else esp * n / 256
    detalle = (f"filas {s0:g}-{s1:g} y columnas {c0:g}-{c1:g} de un corte de {n:g} px; en el PNG, filas "
               f"{num(f0, 0)}-{num(f1, 0)} y columnas {num(k0, 0)}-{num(k1, 0)} (≈ {num((s1 - s0) * esp, 0)} × "
               f"{num((c1 - c0) * esp, 0)} mm)")
    rara = []
    if int(n) not in N_XY_DATASET:
        rara.append(f"n_xy {n:g} no aparece en el dataset ({', '.join(map(str, N_XY_DATASET))})")
    if not (XY_SPACING_DATASET[0] <= esp <= XY_SPACING_DATASET[1]):
        rara.append(f"xy_spacing {num(esp, 3)} mm queda fuera de lo visto en el dataset")
    if rara:
        inf.anadir(AMBITO, "Caja del tumor", AVISO, detalle + ". " + mayuscula("; ".join(rara)) + ".")
    else:
        inf.anadir(AMBITO, "Caja del tumor", OK, detalle + f"; píxel del PNG {num(pixel_png, 3)} mm.")


# Fila inventada (no es de ninguna paciente; sin id ni etiqueta) que pasa la validación sin avisos.
FILA_PLANTILLA = {"HR_HER2_STATUS": "HER2pos", "tum_vol": 15.0, "age": 50, "n_xy": 256, "sraw": 100, "eraw": 160,
                  "scol": 90, "ecol": 150, "xy_spacing": 0.7}


def plantilla_csv(necesita_caja: bool = True) -> bytes:
    """CSV de una fila con la cabecera exacta que acepta leer_csv: las variables clínicas y, si
    la CNN la necesita, la caja del tumor (con xy_spacing)."""
    columnas = list(COLUMNAS_ENTRADA) + (list(COLUMNAS_CAJA) if necesita_caja else [])
    return (",".join(columnas) + "\n" + ",".join(str(FILA_PLANTILLA[c]) for c in columnas) + "\n").encode("utf-8")


def desde_csv(nombre: str, datos: bytes, necesita_caja: bool, rango: dict | None = None) -> VariablesPaciente:
    fila, informe = leer_csv(nombre, datos, necesita_caja)
    if fila is None:
        return VariablesPaciente(informe=informe, origen="CSV")
    return validar(fila, necesita_caja, rango, origen="CSV", informe=informe)
