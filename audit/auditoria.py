"""Auditoría del dataset BreastDCEDL. No entrena ningún modelo.

Uso, desde la raíz del repositorio:

    python audit/auditoria.py [--hilos 8]

Escribe tablas (CSV), figuras (PNG) y un resumen.md en outputs/auditoria/.

Restricción sobre test: de test solo se obtienen recuentos descriptivos
(pacientes, cortes, clases, cohortes) y comprobaciones estructurales (que las
rutas existan y los PNG tengan el formato esperado). Las intensidades, la
figura de ejemplos y pos_weight usan exclusivamente train.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

RAIZ_REPO = Path(__file__).resolve().parents[1]
RAIZ_DATOS = RAIZ_REPO / "breastdcedl"
SALIDA = RAIZ_REPO / "outputs" / "auditoria"

sys.path.insert(0, str(RAIZ_DATOS))
import utils_caso as uc  # noqa: E402

SEMILLA = 42

# Cohorte deducida del prefijo de patient_id. ACRIN-6698 es el subestudio de
# difusión de I-SPY2, por eso cuenta como I-SPY2.
PREFIJOS_COHORTE = (
    ("ISPY1_", "I-SPY1"),
    ("ISPY2-", "I-SPY2"),
    ("ACRIN-6698-", "I-SPY2"),
    ("Breast_MRI_", "Duke"),
)
COHORTES = ("I-SPY1", "I-SPY2", "Duke")
DATASET_A_COHORTE = {"spy1": "I-SPY1", "spy2": "I-SPY2", "duke": "Duke"}

# Criterio de tejido (ver seccion c del resumen): máscara calculada SOLO sobre PRE
# y aplicada idéntica a las tres fases.
UMBRAL_FIJO = 0.1          # el que sugiere utils_caso.cargar_imagen, en [0,1]
SUELO_OTSU = 0.02          # evita que Otsu corte dentro del ruido del aire

# Colores: paleta categórica por defecto (orden fijo) y par divergente azul-rojo
# con punto medio gris neutro.
COLOR_COHORTE = {"I-SPY1": "#2a78d6", "I-SPY2": "#eb6834", "Duke": "#1baf7a"}
CMAP_REALCE = LinearSegmentedColormap.from_list(
    "realce", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1d1d"]
)
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def cohorte_de(patient_id: str) -> str:
    for prefijo, cohorte in PREFIJOS_COHORTE:
        if patient_id.startswith(prefijo):
            return cohorte
    return "desconocida"


def otsu(hist: np.ndarray) -> int:
    """Umbral de Otsu (0-255) a partir de un histograma de 256 bins."""
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return 0
    niveles = np.arange(256)
    w0 = np.cumsum(p)
    w1 = total - w0
    m0 = np.cumsum(p * niveles)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mt * w0 - m0 * total) ** 2 / (w0 * w1)
    var[~np.isfinite(var)] = -1
    return int(np.argmax(var))           # tejido = valores > umbral


def tabla_md(df: pd.DataFrame, decimales: int = 3, indice: bool = True) -> str:
    """DataFrame -> tabla Markdown (sin depender de tabulate)."""
    d = df.reset_index() if indice else df
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else f"{v:.{decimales}f}"
        return str(v)
    cab = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    filas = ["| " + " | ".join(fmt(v) for v in fila) + " |" for fila in d.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


class Registro:
    """Acumula las comprobaciones OK/FALLO con su detalle."""

    def __init__(self):
        self.filas = []

    def check(self, nombre: str, ok: bool, detalle: str = ""):
        self.filas.append({"comprobación": nombre, "resultado": "OK" if ok else "FALLO", "detalle": detalle})
        print(f"  [{'OK' if ok else 'FALLO'}] {nombre}" + (f" — {detalle}" if detalle else ""))

    def df(self):
        return pd.DataFrame(self.filas)


# --------------------------------------------------------------------------- #
# a) Recuentos
# --------------------------------------------------------------------------- #

def recuentos(samples: pd.DataFrame) -> dict:
    pac = samples.drop_duplicates("patient_id")
    t = {}

    def por(grupos, df_cortes, df_pac):
        c = df_cortes.groupby(grupos).size().rename("cortes")
        p = df_pac.groupby(grupos).size().rename("pacientes")
        return pd.concat([p, c], axis=1)

    t["split_clase"] = por(["split", "pCR"], samples, pac)
    t["split_cohorte_clase"] = por(["split", "cohorte", "pCR"], samples, pac)

    tasa = pac.groupby(["cohorte", "split"]).agg(pacientes=("pCR", "size"), pCR_1=("pCR", "sum"))
    tasa["tasa_pCR"] = tasa.pCR_1 / tasa.pacientes
    tot = pac.groupby("split").agg(pacientes=("pCR", "size"), pCR_1=("pCR", "sum"))
    tot["tasa_pCR"] = tot.pCR_1 / tot.pacientes
    tot.index = pd.MultiIndex.from_product([["Todas"], tot.index], names=["cohorte", "split"])
    t["tasa_pcr"] = pd.concat([tasa.reindex(COHORTES, level=0), tot])

    n = samples.groupby(["split", "cohorte", "patient_id"]).size().rename("n")
    desc = n.groupby(["split", "cohorte"]).describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc_tot = n.groupby("split").describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc_tot.index = pd.MultiIndex.from_product([desc_tot.index, ["Todas"]], names=["split", "cohorte"])
    t["cortes_por_paciente"] = pd.concat([desc, desc_tot]).sort_index()
    t["hist_cortes_por_paciente"] = (
        n.reset_index().groupby(["n", "split"]).size().unstack(fill_value=0)
    )
    return t


# --------------------------------------------------------------------------- #
# b) Integridad
# --------------------------------------------------------------------------- #

def integridad_metadatos(samples: pd.DataFrame, patients: pd.DataFrame, reg: Registro):
    # patient_id en un único split
    splits = samples.groupby("patient_id").split.nunique()
    malos = splits[splits > 1]
    reg.check("Cada patient_id aparece en un único split", malos.empty,
              f"{len(malos)} pacientes en >1 split" if len(malos) else f"{len(splits)} pacientes")

    # dentro de train, un único fold por paciente y folds 0-4; test con fold -1
    tr = samples[samples.split == "train"]
    folds = tr.groupby("patient_id").fold.nunique()
    malos = folds[folds > 1]
    reg.check("En train, cada paciente pertenece a un único fold", malos.empty,
              f"{len(malos)} pacientes en >1 fold" if len(malos) else f"{len(folds)} pacientes")
    reg.check("Folds de train en {0..4} y fold=-1 en test",
              set(tr.fold.unique()) <= set(range(5)) and set(samples[samples.split == "test"].fold.unique()) == {-1},
              f"train: {sorted(int(f) for f in tr.fold.unique())}")

    # etiqueta constante por paciente
    etiquetas = samples.groupby("patient_id").pCR.nunique()
    malos = etiquetas[etiquetas > 1]
    reg.check("La etiqueta es la misma en todos los cortes de una paciente", malos.empty,
              f"{len(malos)} pacientes con etiqueta variable" if len(malos) else "")
    reg.check("Sin nulos en samples.csv", not samples.isna().any().any(),
              f"{int(samples.isna().sum().sum())} nulos")
    reg.check("pCR solo toma valores {0, 1}", set(samples.pCR.unique()) <= {0, 1})

    # coherencia con patients.csv
    pac = samples.groupby("patient_id").agg(pCR=("pCR", "first"), split=("split", "first"))
    uni = pac.join(patients.set_index("pid")[["pCR", "split", "dataset"]], rsuffix="_p", how="left")
    ok = (uni.pCR == uni.pCR_p).all() and (uni.split == uni.split_p).all() \
        and set(pac.index) == set(patients.pid)
    reg.check("samples.csv y patients.csv coinciden en pacientes, pCR y split", bool(ok))
    coh = uni.index.map(cohorte_de)
    ok = (coh == uni.dataset.map(DATASET_A_COHORTE)).all()
    reg.check("La cohorte deducida del patient_id coincide con patients.dataset", bool(ok),
              "prefijos: ISPY1_→I-SPY1, ISPY2-/ACRIN-6698-→I-SPY2, Breast_MRI_→Duke")

    # identificadores y rutas bien formados
    ok_id = (samples.sample_id == samples.patient_id + "_z" + samples.slice_index.map("{:03d}".format)).all()
    reg.check("sample_id = {patient_id}_z{slice_index:03d} y es único",
              bool(ok_id) and samples.sample_id.is_unique)
    esperado = {f: "dataset/" + samples.split + "/" + samples.patient_id + "/" + samples.sample_id + f"_{f}.png"
                for f in uc.FASES}
    ok_rutas = all((samples[f"path_{f.lower()}"] == esperado[f]).all() for f in uc.FASES)
    reg.check("Cada sample apunta a sus 3 fases (PRE, EARLY, LATE) con el nombre esperado", bool(ok_rutas))


def integridad_ficheros(samples: pd.DataFrame, reg: Registro):
    """Existencia de rutas y ausencia de ficheros huérfanos en las carpetas."""
    rutas = pd.concat([samples[f"path_{f.lower()}"] for f in uc.FASES])
    faltan = [r for r in rutas if not (RAIZ_DATOS / r).is_file()]
    reg.check("Todas las rutas de samples.csv existen", not faltan,
              f"{len(faltan)} rutas inexistentes" if faltan else f"{len(rutas)} ficheros")

    esperados = set(rutas)
    presentes = {p.relative_to(RAIZ_DATOS).as_posix() for p in (RAIZ_DATOS / "dataset").rglob("*.png")}
    sobran = presentes - esperados
    reg.check("Cada sample tiene exactamente sus 3 fases (sin PNG huérfanos en disco)",
              not sobran and not faltan and samples.sample_id.is_unique,
              f"{len(sobran)} PNG en disco no listados en samples.csv" if sobran else
              f"{len(presentes)} PNG = 3 × {len(samples)} cortes")


# --------------------------------------------------------------------------- #
# Pasada por los píxeles: formato (todos) + intensidades (solo train)
# --------------------------------------------------------------------------- #

def leer_png(ruta: Path):
    with Image.open(ruta) as png:
        modo, tam = png.mode, png.size
        arr = np.asarray(png)
    return modo, tam, arr


def procesar_corte(fila, con_intensidades: bool) -> dict:
    res = {"sample_id": fila.sample_id, "formato_ok": True, "formato_error": ""}
    planos = []
    for f in uc.FASES:
        modo, tam, arr = leer_png(RAIZ_DATOS / getattr(fila, f"path_{f.lower()}"))
        if not (modo == "L" and tam == (256, 256) and arr.shape == (256, 256) and arr.dtype == np.uint8):
            res["formato_ok"] = False
            res["formato_error"] += f"{f}: mode={modo} size={tam} shape={arr.shape} dtype={arr.dtype}; "
        planos.append(arr)
    if not (con_intensidades and res["formato_ok"]):
        return res

    pre, early, late = planos
    res["hist_pre"] = np.bincount(pre.ravel(), minlength=256)
    t_otsu = max(otsu(res["hist_pre"]), int(round(SUELO_OTSU * 255)))
    mascaras = {"fijo": pre > int(round(UMBRAL_FIJO * 255)), "otsu": pre > t_otsu}
    res["umbral_otsu"] = t_otsu / 255
    for nombre, m in mascaras.items():
        res[f"n_{nombre}"] = int(m.sum())
        for f, a in zip(uc.FASES, planos):
            res[f"suma_{f}_{nombre}"] = float(a[m].sum()) / 255.0
    # histogramas de tejido (criterio principal) para comparar cohortes
    m = mascaras["otsu"]
    for f, a in zip(uc.FASES, planos):
        res[f"hist_{f}"] = np.bincount(a[m], minlength=256)
    res["frac_cero"] = float((pre == 0).mean())
    res["frac_sat"] = float(np.mean([(a == 255).mean() for a in planos]))
    return res


def pasada_pixeles(samples: pd.DataFrame, hilos: int):
    filas = list(samples.itertuples(index=False))
    es_train = [f.split == "train" for f in filas]
    with ThreadPoolExecutor(hilos) as ex:
        res = list(ex.map(procesar_corte, filas, es_train, chunksize=64))
    return res


def integridad_formato(res, reg: Registro):
    malos = [r for r in res if not r["formato_ok"]]
    reg.check("Cada PNG es 256×256, un canal (modo L), uint8", not malos,
              f"{len(malos)} cortes con formato incorrecto" if malos else f"{3 * len(res)} PNG revisados")
    return malos


# --------------------------------------------------------------------------- #
# c) Intensidades (solo train)
# --------------------------------------------------------------------------- #

def intensidades(res, samples: pd.DataFrame) -> dict:
    tr = samples[samples.split == "train"].set_index("sample_id")
    escalares = [{k: v for k, v in r.items() if not k.startswith("hist")} for r in res if "n_otsu" in r]
    cortes = pd.DataFrame(escalares).set_index("sample_id").join(tr[["patient_id", "cohorte", "pCR"]])

    t = {}
    # medias por paciente: media de todos los píxeles de tejido de todos sus cortes
    for crit in ("otsu", "fijo"):
        g = cortes.groupby("patient_id")
        pac = pd.DataFrame({f: g[f"suma_{f}_{crit}"].sum() / g[f"n_{crit}"].sum() for f in uc.FASES})
        pac["cohorte"] = g.cohorte.first()
        pac["pCR"] = g.pCR.first()
        pac["frac_tejido"] = g[f"n_{crit}"].sum() / (g.size() * 256 * 256)
        pac["realce_abs"] = pac.EARLY - pac.PRE
        pac["realce_rel"] = pac.realce_abs / pac.PRE
        pac["lavado"] = pac.LATE - pac.EARLY
        t[f"pac_{crit}"] = pac

        def pct(d):
            return pd.Series({
                "pacientes": len(d),
                "% PRE<EARLY": 100 * (d.PRE < d.EARLY).mean(),
                "% LATE>EARLY": 100 * (d.LATE > d.EARLY).mean(),
                "% LATE<EARLY (washout)": 100 * (d.LATE < d.EARLY).mean(),
            })
        tab = pac.groupby("cohorte").apply(pct).reindex(COHORTES)
        tab.loc["Todas"] = pct(pac)
        tab["pacientes"] = tab.pacientes.astype(int)
        t[f"porcentajes_{crit}"] = tab

    # cortes con máscara vacía (no aportan a la media)
    t["cortes_sin_tejido"] = {c: int((cortes[f"n_{c}"] == 0).sum()) for c in ("otsu", "fijo")}

    # comparación entre cohortes (criterio principal, a nivel de paciente)
    pac = t["pac_otsu"]
    cols = ["PRE", "EARLY", "LATE", "realce_abs", "realce_rel", "lavado", "frac_tejido"]
    t["media_por_fase_cohorte"] = pac.groupby("cohorte")[cols].mean().reindex(COHORTES)
    t["media_por_fase_cohorte"].loc["Todas"] = pac[cols].mean()
    q = pac.groupby("cohorte")[["PRE", "EARLY", "LATE", "realce_abs"]].quantile([0.1, 0.5, 0.9]).unstack()
    q.columns = [f"{v}_p{int(p * 100)}" for v, p in q.columns]
    t["cuantiles_cohorte"] = q.reindex(COHORTES)

    extra = cortes.groupby("cohorte").agg(
        umbral_otsu_mediano=("umbral_otsu", "median"),
        frac_pixeles_cero=("frac_cero", "mean"),
        frac_pixeles_255=("frac_sat", "mean"),
    ).reindex(COHORTES)
    t["fondo_saturacion_cohorte"] = extra

    # Kolmogorov-Smirnov (estadístico D) entre pares de cohortes, por paciente
    def ks(a, b):
        a, b = np.sort(a), np.sort(b)
        x = np.concatenate([a, b])
        return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a)
                                   - np.searchsorted(b, x, "right") / len(b))))
    filas = []
    for i, c1 in enumerate(COHORTES):
        for c2 in COHORTES[i + 1:]:
            fila = {"par": f"{c1} vs {c2}"}
            for v in ("PRE", "EARLY", "LATE", "realce_abs"):
                fila[f"KS_D_{v}"] = ks(pac.loc[pac.cohorte == c1, v].values, pac.loc[pac.cohorte == c2, v].values)
            filas.append(fila)
    t["ks_cohortes"] = pd.DataFrame(filas).set_index("par")

    # histogramas de píxeles de tejido por cohorte y fase
    hist = {c: {f: np.zeros(256, np.int64) for f in uc.FASES} for c in COHORTES}
    coh = tr.cohorte.to_dict()
    for r in res:
        if "hist_PRE" in r:
            for f in uc.FASES:
                hist[coh[r["sample_id"]]][f] += r[f"hist_{f}"]
    t["hist"] = hist
    return t


def figura_distribuciones(t: dict, ruta: Path):
    hist, pac = t["hist"], t["pac_otsu"]
    x = np.arange(256) / 255
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for j, f in enumerate(uc.FASES):
        a = ax[0, j]
        for c in COHORTES:
            h = hist[c][f].astype(float)[:-1]        # sin el 255: saturación, ver tabla
            a.plot(x[:-1], h / h.sum(), color=COLOR_COHORTE[c], lw=2, label=c)
        a.set_title(f"{f}: píxeles de tejido (densidad, sin 255)", color=TINTA, fontsize=11)
        a.set_xlabel("intensidad [0,1]", color=TINTA_2)
        a.set_xlim(0, 1)
    ax[0, 0].legend(frameon=False)
    variables = [("PRE", "media PRE por paciente"), ("EARLY", "media EARLY por paciente"),
                 ("realce_abs", "EARLY − PRE por paciente")]
    for j, (v, titulo) in enumerate(variables):
        a = ax[1, j]
        datos = [pac.loc[pac.cohorte == c, v].values for c in COHORTES]
        bp = a.boxplot(datos, tick_labels=COHORTES, widths=0.5, patch_artist=True,
                       medianprops=dict(color=TINTA, lw=2), flierprops=dict(markersize=3, alpha=0.5))
        for caja, c in zip(bp["boxes"], COHORTES):
            caja.set_facecolor(COLOR_COHORTE[c])
            caja.set_alpha(0.55)
        a.set_title(titulo, color=TINTA, fontsize=11)
    for a in ax.ravel():
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=TINTA_2)
    fig.suptitle("Intensidades en train por cohorte (máscara de tejido Otsu sobre PRE)", color=TINTA)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# d) pos_weight por fold (solo train)
# --------------------------------------------------------------------------- #

def tabla_pos_weight(samples: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for k in range(5):
        entr, val = uc.particion(samples, fold_val=k)
        pac = entr.drop_duplicates("patient_id")
        filas.append({
            "fold_val": k,
            "cortes_N0": int((entr.pCR == 0).sum()), "cortes_N1": int((entr.pCR == 1).sum()),
            "pos_weight_corte": uc.pos_weight(entr),
            "pac_N0": int((pac.pCR == 0).sum()), "pac_N1": int((pac.pCR == 1).sum()),
            "pos_weight_paciente": uc.pos_weight(pac),
        })
    return pd.DataFrame(filas).set_index("fold_val")


# --------------------------------------------------------------------------- #
# e) Figura de ejemplos (solo train)
# --------------------------------------------------------------------------- #

def elegir_pacientes(samples: pd.DataFrame, rng) -> list[str]:
    """2 pCR=1 y 2 pCR=0; dentro de cada clase, cohortes distintas, y entre las
    cuatro, las tres cohortes representadas."""
    pac = samples[samples.split == "train"].drop_duplicates("patient_id")
    plan = [(1, "I-SPY1"), (1, "I-SPY2"), (0, "Duke"), (0, "I-SPY2")]
    elegidas = []
    for etiqueta, cohorte in plan:
        cand = pac[(pac.pCR == etiqueta) & (pac.cohorte == cohorte)].patient_id.values
        elegidas.append(str(rng.choice(cand)))
    return elegidas


def figura_ejemplos(samples: pd.DataFrame, ruta: Path, rng):
    ids = elegir_pacientes(samples, rng)
    filas = []
    for pid in ids:
        cortes = samples[samples.patient_id == pid].sort_values("slice_index")
        filas.append(cortes.iloc[len(cortes) // 2])          # corte central
    imgs = [uc.cargar_imagen(f, RAIZ_DATOS) for f in filas]
    realces = [x[1] - x[0] for x in imgs]
    lim = float(np.percentile(np.abs(np.concatenate([r.ravel() for r in realces])), 99.5))

    fig, ax = plt.subplots(4, 4, figsize=(13, 13.6), constrained_layout=True)
    for i, (fila, x, r) in enumerate(zip(filas, imgs, realces)):
        for j, f in enumerate(uc.FASES):
            ax[i, j].imshow(x[j], cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax[i, j].set_title(f, fontsize=12, color=TINTA)
        im = ax[i, 3].imshow(r, cmap=CMAP_REALCE, vmin=-lim, vmax=lim)
        if i == 0:
            ax[i, 3].set_title("EARLY − PRE", fontsize=12, color=TINTA)
        ax[i, 0].set_ylabel(f"{fila.patient_id}\n{cohorte_de(fila.patient_id)} · pCR={fila.pCR}\n"
                            f"z={fila.slice_index}", fontsize=10, color=TINTA)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(im, ax=ax[:, 3], shrink=0.5, location="right")
    cb.set_label("realce (unidades de intensidad [0,1])", color=TINTA_2)
    fig.suptitle("PRE, EARLY y LATE con escala de grises común [0,1]; realce con escala divergente "
                 f"simétrica ±{lim:.2f}", color=TINTA)
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    return [(f.patient_id, cohorte_de(f.patient_id), int(f.pCR), f.sample_id) for f in filas], lim


# --------------------------------------------------------------------------- #
# f) Resumen
# --------------------------------------------------------------------------- #

def escribir_resumen(ruta, rec, checks, malos_formato, inten, pw, ejemplos, lim):
    L = []
    w = L.append
    w("# Auditoría del dataset BreastDCEDL\n")
    w("Generado por `audit/auditoria.py`. No se ha entrenado ningún modelo. "
      "De **test** solo se reportan recuentos descriptivos y comprobaciones estructurales; "
      "intensidades, figura de ejemplos y `pos_weight` usan **solo train**.\n")
    w("Cohorte deducida del prefijo de `patient_id`: `ISPY1_` → I-SPY1; `ISPY2-` y `ACRIN-6698-` → "
      "I-SPY2 (ACRIN 6698 es un subestudio de I-SPY2); `Breast_MRI_` → Duke. Se verifica contra "
      "`patients.dataset` más abajo.\n")

    w("## b) Comprobaciones de integridad\n")
    w(tabla_md(checks, indice=False) + "\n")
    if malos_formato:
        w("Primeros cortes con formato incorrecto:\n")
        for r in malos_formato[:20]:
            w(f"- `{r['sample_id']}`: {r['formato_error']}")
        w("")

    w("## a) Recuentos\n")
    w("### Pacientes y cortes por split y clase\n")
    w(tabla_md(rec["split_clase"]) + "\n")
    w("### Por split, cohorte y clase\n")
    w(tabla_md(rec["split_cohorte_clase"]) + "\n")
    w("### Tasa de pCR (a nivel de paciente) por cohorte y split\n")
    w(tabla_md(rec["tasa_pcr"]) + "\n")
    w("### Número de cortes por paciente\n")
    w(tabla_md(rec["cortes_por_paciente"], decimales=2) + "\n")
    w("Distribución (número de pacientes con n cortes):\n")
    w(tabla_md(rec["hist_cortes_por_paciente"]) + "\n")

    w("## c) Intensidades (solo train)\n")
    w("**Criterio tejido/fondo.** El fondo es aire, con intensidad casi nula. La máscara se calcula "
      "**solo sobre PRE** y se aplica **idéntica a las tres fases**, por dos razones: (1) las tres medias "
      "se calculan sobre los mismos píxeles, así que son comparables; (2) una máscara que no depende del "
      "contraste no favorece artificialmente a EARLY (si se usara EARLY, entrarían justo los píxeles que "
      "más realzan). Criterio principal: **umbral de Otsu por corte sobre el histograma de PRE**, con un "
      f"suelo de {SUELO_OTSU} para no cortar dentro del ruido del aire. Es adaptativo, lo que importa porque "
      "la ventana de intensidades cambia entre cohortes. Como contraste de sensibilidad se reporta también "
      f"el umbral fijo PRE > {UMBRAL_FIJO} que sugiere `utils_caso.cargar_imagen`. La media por paciente es "
      "la media de todos los píxeles de tejido de todos sus cortes.\n")
    w(f"Cortes con máscara vacía: Otsu = {inten['cortes_sin_tejido']['otsu']}, "
      f"fijo = {inten['cortes_sin_tejido']['fijo']}.\n")
    w("### Porcentaje de pacientes — criterio principal (Otsu sobre PRE)\n")
    w(tabla_md(inten["porcentajes_otsu"], decimales=1) + "\n")
    w(f"### Porcentaje de pacientes — umbral fijo PRE > {UMBRAL_FIJO}\n")
    w(tabla_md(inten["porcentajes_fijo"], decimales=1) + "\n")
    w("### Pacientes con PRE ≥ EARLY (criterio principal)\n")
    pac = inten["pac_otsu"]
    w(tabla_md(pac.loc[pac.PRE >= pac.EARLY, ["cohorte", "pCR", "PRE", "EARLY", "LATE"]]) + "\n")
    w("### Media por fase en tejido (media de las medias por paciente)\n")
    w("`realce_abs` = EARLY − PRE; `realce_rel` = (EARLY − PRE)/PRE; `lavado` = LATE − EARLY; "
      "`frac_tejido` = fracción de la imagen dentro de la máscara.\n")
    w(tabla_md(inten["media_por_fase_cohorte"]) + "\n")
    w("### Cuantiles por cohorte (p10 / p50 / p90 de las medias por paciente)\n")
    w(tabla_md(inten["cuantiles_cohorte"]) + "\n")
    w("### Fondo y saturación por cohorte (media por corte)\n")
    w(tabla_md(inten["fondo_saturacion_cohorte"]) + "\n")
    w("### Distancia entre distribuciones de cohortes (estadístico D de Kolmogorov-Smirnov, por paciente)\n")
    w("D ∈ [0,1]: máxima diferencia entre funciones de distribución acumuladas. 0 = idénticas.\n")
    w(tabla_md(inten["ks_cohortes"]) + "\n")
    w("![Distribuciones por cohorte](distribuciones_intensidad.png)\n")

    w("## d) pos_weight = N0/N1 (solo train, excluyendo el fold de validación)\n")
    w(tabla_md(pw) + "\n")

    w("## e) Ejemplos (train)\n")
    w("Corte central de cada paciente. PRE/EARLY/LATE con escala de grises común [0,1]; "
      f"EARLY − PRE con mapa divergente azul–gris–rojo simétrico en ±{lim:.2f} (percentil 99,5 de |realce|).\n")
    w(tabla_md(pd.DataFrame(ejemplos, columns=["patient_id", "cohorte", "pCR", "corte"]), indice=False) + "\n")
    w("![Ejemplos](ejemplos_fases.png)\n")
    ruta.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hilos", type=int, default=8)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")       # consola de Windows en cp1252
    SALIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)

    samples = uc.cargar_samples(RAIZ_DATOS)
    patients = uc.cargar_patients(RAIZ_DATOS)
    samples["cohorte"] = samples.patient_id.map(cohorte_de)

    print("a) Recuentos")
    rec = recuentos(samples)
    for nombre, df in rec.items():
        df.to_csv(SALIDA / f"a_{nombre}.csv")

    print("b) Integridad")
    reg = Registro()
    integridad_metadatos(samples, patients, reg)
    integridad_ficheros(samples, reg)
    print("   leyendo píxeles…")
    res = pasada_pixeles(samples, args.hilos)
    malos = integridad_formato(res, reg)
    checks = reg.df()
    checks.to_csv(SALIDA / "b_integridad.csv", index=False)

    print("c) Intensidades (train)")
    inten = intensidades(res, samples)
    for crit in ("otsu", "fijo"):
        inten[f"pac_{crit}"].to_csv(SALIDA / f"c_intensidad_por_paciente_{crit}.csv")
        inten[f"porcentajes_{crit}"].to_csv(SALIDA / f"c_porcentajes_{crit}.csv")
    for nombre in ("media_por_fase_cohorte", "cuantiles_cohorte", "fondo_saturacion_cohorte", "ks_cohortes"):
        inten[nombre].to_csv(SALIDA / f"c_{nombre}.csv")
    figura_distribuciones(inten, SALIDA / "distribuciones_intensidad.png")

    print("d) pos_weight")
    pw = tabla_pos_weight(samples)
    pw.to_csv(SALIDA / "d_pos_weight.csv")

    print("e) Figura de ejemplos")
    ejemplos, lim = figura_ejemplos(samples, SALIDA / "ejemplos_fases.png", rng)

    print("f) Resumen")
    escribir_resumen(SALIDA / "resumen.md", rec, checks, malos, inten, pw, ejemplos, lim)
    print(f"Hecho. Resultados en {SALIDA}")


if __name__ == "__main__":
    main()
