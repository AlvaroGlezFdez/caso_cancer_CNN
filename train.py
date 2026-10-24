"""Entrenamiento de la CNN que predice pCR (caso BreastDCEDL). No usa el conjunto test.

Uso, desde la raíz del repositorio:

    python train.py --nombre baseline_3ep_cpu --epocas 3
    python train.py --nombre baseline_pw --pos-weight
    python train.py --nombre aumento --aumento
    python train.py --nombre wd --weight-decay 1e-4
    python train.py --nombre cinco_bloques --bloques 5
    python train.py --nombre realce --realce
    python train.py --nombre recortecaja --recorte-caja
    python train.py --nombre criterio_antiguo --criterio auc
    python train.py --nombre prueba_max5 --max-batches 5 --epocas 2      # prueba
    python train.py --nombre prueba_overfit --overfit-batches 2 --epocas 50
    python train.py --nombre prueba_epoca --epocas 1 --prueba

Los valores por defecto son los de "Decisiones tomadas" en CLAUDE.md: CNNBase de
4 bloques, AdamW con weight decay 0 (equivale a Adam), lr 1e-3, batch 32, 30
épocas como máximo, best.pt y early stopping (paciencia 7) con el criterio
auc_suavizado (seleccion.py), semilla 42, fold 0, sin aumento de datos, sin
pos_weight y sin realce.

--aumento aplica aumento.aumentar (volteo horizontal y rotación de hasta ±10°,
la misma transformación en las tres fases) solo a los lotes de entrenamiento.
--bloques 5 añade un quinto bloque de 128 a 128 filtros antes del GAP y
--realce hace que el modelo use PRE, EARLY-PRE y LATE-EARLY (models/cnn_base.py).
--recorte-caja recorta cada corte alrededor de la caja del tumor de la paciente
(caja_tumor.caja_en_png, preprocesado.recortar) y la red recibe 128x128.
--criterio auc reproduce la selección anterior al 2026-10-22 (AUC de cada época).

Una ejecución normal escribe:
  experimentos/<fecha_hora>_<nombre>/  config.json, train.log, metricas_epoca.csv
                                       y el informe (gráficas y tablas). Va a git.
  outputs/runs/<fecha_hora>_<nombre>/  best.pt, last.pt y predicciones_val.csv. No va a git.
y añade una fila a experimentos/experimentos.csv.

Las pruebas (--prueba, --max-batches, --overfit-batches) escriben todo en
outputs/pruebas/<fecha_hora>_<nombre>/ y no tocan experimentos/. Con
--max-batches u --overfit-batches la validación también se recorta a N lotes.

Un experimento (no una prueba) no arranca si hay código .py sin commit, para
que el commit de config.json sea exactamente el código entrenado. Para saltarse
ese control: --permitir-sin-commit (queda registrado en config.json).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import logging
import os
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    import psutil
except ImportError:                                   # las columnas de CPU y RAM quedan vacías
    psutil = None

RAIZ_REPO = Path(__file__).resolve().parent
RAIZ_DATOS = RAIZ_REPO / "breastdcedl"
sys.path.insert(0, str(RAIZ_DATOS))
import utils_caso as uc  # noqa: E402

import aumento  # noqa: E402
import informe  # noqa: E402
import seleccion  # noqa: E402
from models.cnn_base import FILTROS_POR_BLOQUES, PARAMETROS_ESPERADOS, contar_parametros, crear_modelo  # noqa: E402
from caja_tumor import caja_en_png  # noqa: E402
from preprocesado import cuadrado_recorte, LADO_MINIMO, LADO_RECORTE, MARGEN_RECORTE, preprocesar  # noqa: E402

DIR_EXPERIMENTOS = RAIZ_REPO / "experimentos"
DIR_RUNS = RAIZ_REPO / "outputs" / "runs"
DIR_PRUEBAS = RAIZ_REPO / "outputs" / "pruebas"

UMBRAL = 0.5
METODO = "mean"
DATASET_A_COHORTE = {"spy1": "I-SPY1", "spy2": "I-SPY2", "duke": "Duke"}

log = logging.getLogger("train")


class DatasetCortes(uc.BreastDCEDataset):
    """BreastDCEDataset de utils_caso, pero cargando cada corte con
    preprocesado.preprocesar, la misma función que usará la app. Con `cajas`
    (patient_id -> caja en el PNG) recorta cada corte por la caja de su paciente."""

    def __init__(self, filas, raiz, transform=None, cajas=None):
        super().__init__(filas, raiz, transform)
        self.cajas = cajas

    def __getitem__(self, i: int):
        fila = self.filas.iloc[i]
        caja = None if self.cajas is None else self.cajas[fila.patient_id]
        x = torch.from_numpy(preprocesar(self.raiz / fila.path_pre, self.raiz / fila.path_early,
                                         self.raiz / fila.path_late, caja=caja))
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.tensor(float(fila.pCR), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Entorno, registro y reproducibilidad
# --------------------------------------------------------------------------- #

def argumentos() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", required=True, help="identificador de la ejecución")
    ap.add_argument("--fold", type=int, default=0, help="fold de validación interna (0-4)")
    ap.add_argument("--epocas", type=int, default=30, help="número máximo de épocas")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--paciencia", type=int, default=7,
                    help="épocas sin mejorar el criterio de selección antes de parar")
    ap.add_argument("--criterio", choices=seleccion.CRITERIOS, default="auc_suavizado",
                    help="criterio para best.pt y el early stopping (seleccion.py): auc_suavizado = media "
                         "del AUC de validación por paciente con la época anterior y la siguiente; "
                         "auc = el de la propia época (criterio anterior)")
    ap.add_argument("--bloques", type=int, choices=sorted(FILTROS_POR_BLOQUES), default=4,
                    help="bloques convolucionales; con 5 se añade uno de 128 a 128 filtros antes del GAP")
    ap.add_argument("--realce", action="store_true",
                    help="el modelo transforma su entrada en PRE, EARLY-PRE y LATE-EARLY antes del primer bloque")
    ap.add_argument("--recorte-caja", action="store_true",
                    help="recorta un cuadrado centrado en la caja del tumor (lado mayor + 25 %%, mínimo 64 px) "
                         "y lo reescala a 128x128, igual en las tres fases (preprocesado.recortar)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--pos-weight", action="store_true",
                    help="BCEWithLogitsLoss con pos_weight = N0/N1 de la partición de entrenamiento")
    ap.add_argument("--aumento", action="store_true",
                    help="aumento geométrico solo en entrenamiento (aumento.py): volteo horizontal p=0,5 "
                         "y rotación de hasta ±10°, la misma transformación en las tres fases")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="weight decay de AdamW (por defecto 0: equivale a Adam)")
    ap.add_argument("--max-batches", type=int, metavar="N",
                    help="prueba: cada época se recorta a N lotes de entrenamiento y N de validación")
    ap.add_argument("--overfit-batches", type=int, metavar="N",
                    help="prueba: entrena una y otra vez sobre los mismos N lotes, sin early stopping")
    ap.add_argument("--prueba", action="store_true",
                    help="guarda en outputs/pruebas/ y no añade fila a experimentos.csv")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="procesos de carga de datos (en Windows cada uno se arranca una vez)")
    ap.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--permitir-sin-commit", action="store_true",
                    help="lanza un experimento aunque haya código .py sin commit (queda en config.json)")
    args = ap.parse_args()
    if args.max_batches and args.overfit_batches:
        ap.error("--max-batches y --overfit-batches son excluyentes")
    for nombre in ("max_batches", "overfit_batches"):
        if getattr(args, nombre) is not None and getattr(args, nombre) < 1:
            ap.error(f"--{nombre.replace('_', '-')} debe ser >= 1")
    if args.weight_decay < 0:
        ap.error("--weight-decay debe ser >= 0")
    args.prueba = args.prueba or bool(args.max_batches or args.overfit_batches)
    return args


def configurar_log(ruta: Path):
    sys.stdout.reconfigure(encoding="utf-8")          # consola de Windows en cp1252
    log.setLevel(logging.INFO)
    formato = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for manejador in (logging.StreamHandler(sys.stdout), logging.FileHandler(ruta, encoding="utf-8")):
        manejador.setFormatter(formato)
        log.addHandler(manejador)


def fijar_semilla(semilla: int):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def info_git() -> tuple[str | None, list[str] | None]:
    """Commit actual y ficheros .py modificados o sin commitear (None si git no responde)."""
    try:
        def git(*a):
            return subprocess.run(["git", *a], cwd=RAIZ_REPO, capture_output=True, text=True,
                                  check=True).stdout.strip()
        return git("rev-parse", "HEAD"), git("status", "--porcelain", "--", "*.py").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None, None


def nombre_cpu() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            clave = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, clave) as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                return linea.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def entorno(dispositivo: torch.device, num_workers: int, git: tuple) -> dict:
    commit, cambios = git
    cuda = dispositivo.type == "cuda"
    return {
        "git_commit": commit,
        "git_codigo_sin_commit": None if cambios is None else bool(cambios),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda if cuda else None,
        "dispositivo": str(dispositivo),
        "gpu": torch.cuda.get_device_name(dispositivo) if cuda else None,
        "cpu": nombre_cpu(),
        "nucleos_logicos": os.cpu_count(),
        "hilos_torch": torch.get_num_threads(),
        "num_workers": num_workers,
        "so": platform.platform(),
        "utils_caso_sha256": sha256(RAIZ_DATOS / "utils_caso.py"),
    }


class MonitorRecursos:
    """Muestrea cada segundo, en un hilo aparte, el % de CPU del proceso y sus
    workers (100 % = todos los núcleos lógicos ocupados) y la RAM (RSS) del
    proceso principal."""

    def __init__(self, intervalo: float = 1.0):
        self.intervalo = intervalo
        self.proc = psutil.Process() if psutil else None
        if psutil is None:
            log.warning("psutil no está instalado: no se medirán CPU ni RAM")

    def _cpu(self) -> float:
        total = 0.0
        for p in [self.proc, *self.proc.children(recursive=True)]:
            p = self._vistos.setdefault(p.pid, p)       # cpu_percent mide desde la llamada anterior
            try:
                total += p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / psutil.cpu_count()

    def _bucle(self):
        while not self._parar.wait(self.intervalo):
            self.cpu.append(self._cpu())
            self.ram_max = max(self.ram_max, self.proc.memory_info().rss)

    def iniciar(self):
        if self.proc is None:
            return
        self.cpu, self._vistos, self._parar = [], {}, threading.Event()
        self.ram_max = self.proc.memory_info().rss
        self._cpu()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def detener(self) -> dict:
        if self.proc is None:
            return {"cpu_pct_medio": np.nan, "ram_max_mb": np.nan}
        self._parar.set()
        self._hilo.join()
        self.ram_max = max(self.ram_max, self.proc.memory_info().rss)
        return {"cpu_pct_medio": float(np.mean(self.cpu)) if self.cpu else np.nan,
                "ram_max_mb": self.ram_max / 2**20}


# --------------------------------------------------------------------------- #
# Entrenamiento y evaluación
# --------------------------------------------------------------------------- #

def comprobar_preprocesado(fila):
    """La función compartida debe dar lo mismo que la carga de referencia del caso."""
    x = preprocesar(*(RAIZ_DATOS / r for r in (fila.path_pre, fila.path_early, fila.path_late)))
    if not np.array_equal(x, uc.cargar_imagen(fila, RAIZ_DATOS)):
        raise RuntimeError("preprocesado.preprocesar no coincide con utils_caso.cargar_imagen")


def entrenar_epoca(modelo, lotes, n_lotes: int, criterio, optimizador, dispositivo, transformar=None) -> dict:
    """Una época sobre `lotes` (DataLoader o lista de lotes fijos), separando el
    tiempo esperando datos del tiempo de cálculo. `transformar` es el aumento de
    datos (o None): se aplica a cada lote ya en el dispositivo y solo aquí, nunca
    en predecir()."""
    modelo.train()
    suma, n, t_datos, t_calculo, primera = 0.0, 0, 0.0, 0.0, None
    cada = n_lotes // 10 if n_lotes >= 20 else 0
    t = time.perf_counter()
    for i, (x, y) in enumerate(lotes, 1):
        t_lote = time.perf_counter()
        t_datos += t_lote - t
        x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo, non_blocking=True)
        if transformar is not None:
            x = transformar(x)
        optimizador.zero_grad(set_to_none=True)
        perdida = criterio(modelo(x).squeeze(1), y)
        perdida.backward()
        optimizador.step()
        valor = perdida.item()                        # sincroniza con la GPU si la hay
        if not np.isfinite(valor):
            raise FloatingPointError(f"pérdida no finita en el lote {i}")
        primera = valor if primera is None else primera
        suma += valor * len(y)
        n += len(y)
        t = time.perf_counter()
        t_calculo += t - t_lote
        if cada and i % cada == 0:
            log.info(f"  lote {i}/{n_lotes}  pérdida media {suma / n:.4f}  "
                     f"datos {t_datos:.0f} s  cálculo {t_calculo:.0f} s")
        if i >= n_lotes:
            break
    return {"perdida": suma / n, "imagenes": n, "t_datos": t_datos, "t_calculo": t_calculo, "primera": primera}


@torch.no_grad()
def predecir(modelo, lotes, criterio, dispositivo) -> tuple[np.ndarray, float]:
    """Probabilidad por corte en el orden de `lotes` (shuffle=False) y pérdida media."""
    modelo.eval()
    probs, suma = [], 0.0
    for x, y in lotes:
        logit = modelo(x.to(dispositivo, non_blocking=True)).squeeze(1)
        suma += criterio(logit, y.to(dispositivo, non_blocking=True)).item() * len(y)
        probs.append(torch.sigmoid(logit).cpu().numpy())
    probs = np.concatenate(probs)
    return probs, suma / len(probs)


def guardar_pesos(ruta: Path, pesos: dict, arquitectura: dict, epoca: int, auc: float, optimizador=None):
    """`pesos` es un state_dict; `arquitectura` lo necesario para reconstruir el modelo."""
    estado = {"modelo": pesos, **arquitectura, "epoca": epoca, "auc_val": auc}
    if optimizador is not None:
        estado["optimizador"] = optimizador.state_dict()
    torch.save(estado, ruta)


def anadir_fila(ruta: Path, fila: dict):
    tabla = pd.DataFrame([fila])
    if ruta.exists():
        # round_trip: releer y reescribir no cambia los decimales de las filas anteriores
        tabla = pd.concat([pd.read_csv(ruta, float_precision="round_trip"), tabla], ignore_index=True)
    tabla.to_csv(ruta, index=False)


def relativa(ruta: Path) -> str:
    return ruta.relative_to(RAIZ_REPO).as_posix()


# --------------------------------------------------------------------------- #

def main():
    args = argumentos()
    git = info_git()
    if not args.prueba and git[1] != [] and not args.permitir_sin_commit:
        motivo = ("no se puede consultar git" if git[1] is None
                  else "hay código sin commit:\n" + "\n".join(f"    {c}" for c in git[1]))
        sys.exit(f"ERROR: no se lanza el experimento porque {motivo}\n"
                 "Haz commit antes, para que config.json apunte al código exacto que se entrena,\n"
                 "o añade --permitir-sin-commit (quedará registrado en config.json).")
    inicio = datetime.now()
    ejecucion = f"{inicio:%Y%m%d_%H%M%S}_{args.nombre}"
    if args.prueba:
        dir_exp = dir_out = DIR_PRUEBAS / ejecucion
    else:
        dir_exp, dir_out = DIR_EXPERIMENTOS / ejecucion, DIR_RUNS / ejecucion
    dir_exp.mkdir(parents=True)
    dir_out.mkdir(parents=True, exist_ok=True)
    configurar_log(dir_exp / "train.log")
    log.info(f"Ejecución {ejecucion} ({'PRUEBA' if args.prueba else 'experimento'})")
    log.info("Comando: python " + " ".join(sys.argv))
    if not args.prueba and git[1] != []:
        log.warning("Experimento con código sin commit (--permitir-sin-commit): "
                    + ("git no disponible" if git[1] is None else ", ".join(git[1])))

    fijar_semilla(args.semilla)
    if args.dispositivo == "auto":
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dispositivo = torch.device(args.dispositivo)
    cuda = dispositivo.type == "cuda"

    # Datos: solo la partición de train. El conjunto test no se carga.
    samples = uc.cargar_samples(RAIZ_DATOS)
    patients = uc.cargar_patients(RAIZ_DATOS).set_index("pid")
    cohorte = patients.dataset.map(DATASET_A_COHORTE)
    samples["cohorte"] = samples.patient_id.map(cohorte)
    if samples.cohorte.isna().any():
        raise RuntimeError("Hay pacientes sin cohorte conocida en patients.csv")
    entrenamiento, validacion = uc.particion(samples, fold_val=args.fold)
    comprobar_preprocesado(entrenamiento.iloc[0])
    tasa_cohorte = entrenamiento.drop_duplicates("patient_id").groupby("cohorte").pCR.mean()

    recorte = args.max_batches or args.overfit_batches
    val_eval = validacion.iloc[: recorte * args.batch_size] if recorte else validacion
    log.info(f"Fold de validación {args.fold}: entrenamiento {entrenamiento.patient_id.nunique()} pacientes / "
             f"{len(entrenamiento)} cortes; validación {validacion.patient_id.nunique()} pacientes / "
             f"{len(validacion)} cortes"
             + (f" (recortada a {val_eval.patient_id.nunique()} pacientes / {len(val_eval)} cortes)"
                if recorte else ""))
    log.info("Preprocesado: preprocesado.preprocesar, idéntico a utils_caso.cargar_imagen (comprobado)")
    cajas = None
    if args.recorte_caja:
        # La caja es un dato de la paciente (patients.csv), como en la app.
        cajas = {pid: caja_en_png(patients.loc[pid]) for pid in samples.patient_id.unique()}
        lados = np.array([cuadrado_recorte(cajas[p])[2] for p in entrenamiento.patient_id.unique()])
        log.info(f"Recorte por la caja del tumor a {LADO_RECORTE}x{LADO_RECORTE}: lado del cuadrado en "
                 f"entrenamiento mediana {np.median(lados):.0f} px (mín. {lados.min():.0f}, máx. {lados.max():.0f}; "
                 f"{np.mean(lados == LADO_MINIMO):.0%} con el mínimo de {LADO_MINIMO})")

    workers = 0 if args.overfit_batches else args.num_workers

    def cargador(filas, barajar, generador=None):
        return DataLoader(DatasetCortes(filas, RAIZ_DATOS, cajas=cajas), batch_size=args.batch_size, shuffle=barajar,
                          generator=generador, num_workers=workers, persistent_workers=workers > 0,
                          pin_memory=cuda)

    dl_entr = cargador(entrenamiento, True, torch.Generator().manual_seed(args.semilla))
    dl_val = cargador(val_eval, False)
    if args.overfit_batches:
        lotes_fijos = list(itertools.islice(dl_entr, args.overfit_batches))
        n_lotes = len(lotes_fijos)
        log.info(f"Modo sobreajuste: {n_lotes} lotes fijos ({sum(len(y) for _, y in lotes_fijos)} cortes, "
                 f"{int(sum(y.sum() for _, y in lotes_fijos))} con pCR=1), sin early stopping")
    else:
        n_lotes = min(len(dl_entr), args.max_batches or len(dl_entr))

    # Modelo, pérdida y optimizador
    modelo = crear_modelo(args.bloques, args.realce).to(dispositivo)
    n_parametros = contar_parametros(modelo)
    if n_parametros != PARAMETROS_ESPERADOS[args.bloques]:
        raise RuntimeError(f"{n_parametros} parámetros; se esperaban {PARAMETROS_ESPERADOS[args.bloques]}")
    arquitectura = {"arquitectura": "CNNBase", "bloques": args.bloques, "realce": args.realce,
                    "n_parametros": n_parametros}
    if args.recorte_caja:                       # solo si se usa: los pesos anteriores no cambian
        arquitectura["recorte_caja"] = True
    pos_weight = uc.pos_weight(entrenamiento) if args.pos_weight else None
    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=dispositivo) if pos_weight is not None else None)
    # AdamW con weight_decay 0 hace exactamente lo mismo que Adam (la línea base)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    paciencia = None if args.overfit_batches else args.paciencia

    # Aumento de datos: generador propio, para no tocar el del barajado ni el global (dropout)
    semilla_aumento = args.semilla + 1 if args.aumento else None
    transformar = (functools.partial(aumento.aumentar, generador=torch.Generator().manual_seed(semilla_aumento))
                   if args.aumento else None)
    log.info(f"Modelo CNNBase: {args.bloques} bloques, realce {'sí' if args.realce else 'no'}, "
             f"{n_parametros} parámetros entrenables; dispositivo {dispositivo}; "
             f"criterio {args.criterio}, paciencia {paciencia}; "
             f"pos_weight {f'{pos_weight:.4f}' if pos_weight is not None else 'no'}; "
             f"AdamW weight decay {args.weight_decay:g}; "
             f"aumento {aumento.etiqueta() + ' (solo entrenamiento)' if args.aumento else 'no'}; "
             f"recorte por la caja {'sí' if args.recorte_caja else 'no'}; "
             f"{n_lotes} lotes de {args.batch_size} por época")

    config = {
        "ejecucion": ejecucion,
        "fecha_hora": inicio.isoformat(timespec="seconds"),
        "modo": "prueba" if args.prueba else "experimento",
        "comando": "python " + " ".join(sys.argv),
        "semilla": args.semilla,
        "fold_val": args.fold,
        "pos_weight": pos_weight,
        "n_parametros": n_parametros,
        "hiperparametros": {
            "modelo": "CNNBase",
            "bloques": args.bloques,
            "filtros": list(FILTROS_POR_BLOQUES[args.bloques]),
            "realce": args.realce,
            "entrada_primer_bloque": ("PRE, EARLY-PRE, LATE-EARLY (calculados dentro del modelo)"
                                      if args.realce else "PRE, EARLY, LATE"),
            "dropout": 0.3,
            "inicializacion": "por defecto de PyTorch",
            "optimizador": "AdamW",
            "nota_optimizador": "AdamW con weight_decay 0 equivale a Adam (las ejecuciones "
                                "anteriores a este cambio usaban torch.optim.Adam)",
            "lr": args.lr,
            "betas": list(optimizador.defaults["betas"]),
            "weight_decay": optimizador.defaults["weight_decay"],
            "batch_size": args.batch_size,
            "epocas_max": args.epocas,
            "paciencia": paciencia,
            "criterio_seleccion": args.criterio,
            "criterio_parada": f"{seleccion.DESCRIPCION[args.criterio]} (agregación {METODO}); "
                               "decide best.pt y el early stopping",
            "perdida": "BCEWithLogitsLoss",
            "usar_pos_weight": args.pos_weight,
            "perdida_validacion": "el mismo criterio que en entrenamiento",
            "aumento_datos": ({**aumento.descripcion(), "semilla_generador": semilla_aumento}
                              if args.aumento else None),
            "normalizacion": "dividir entre 255 (preprocesado.preprocesar)",
            "recorte_caja": ({"caja": "caja_tumor.caja_en_png (patients.csv)", "margen": MARGEN_RECORTE,
                              "lado_minimo_px": LADO_MINIMO, "tamano_salida": [LADO_RECORTE, LADO_RECORTE],
                              "interpolacion": "bilineal de PIL (antialiasing al reducir)",
                              "fuera_de_imagen": "ceros"}
                             if args.recorte_caja else None),
            "agregacion_paciente": METODO,
            "umbral": UMBRAL,
            "max_batches": args.max_batches,
            "overfit_batches": args.overfit_batches,
        },
        "datos": {
            "pacientes_entrenamiento": int(entrenamiento.patient_id.nunique()),
            "cortes_entrenamiento": len(entrenamiento),
            "pacientes_validacion": int(val_eval.patient_id.nunique()),
            "cortes_validacion": len(val_eval),
            "validacion_recortada": bool(recorte),
            "tasa_pCR_entrenamiento_por_cohorte": tasa_cohorte.round(4).to_dict(),
        },
        "entorno": {**entorno(dispositivo, workers, git), "permitir_sin_commit": args.permitir_sin_commit},
        "rutas": {"experimento": relativa(dir_exp), "salidas": relativa(dir_out)},
        "resultado": None,
        "sha256": None,
    }
    ruta_config = dir_exp / "config.json"
    informe.guardar_json(ruta_config, config)
    log.info(f"Entorno: {config['entorno']}")

    # Bucle de épocas. El criterio (seleccion.py) puede fijar el valor de una época
    # un paso después (auc_suavizado necesita el AUC de la siguiente): hasta
    # entonces se guardan en `pendientes` sus pesos y sus predicciones.
    monitor = MonitorRecursos()
    filas, primera = [], None
    sel = seleccion.Seleccion(args.criterio, paciencia)
    pendientes, mejor_probs, parada_temprana = {}, None, False

    def fijar(fijadas) -> str:
        """Anota en `filas` el valor del criterio de las épocas fijadas, guarda
        best.pt si alguna mejora y devuelve el texto para el log."""
        nonlocal mejor_probs
        texto = ""
        for e, v, mejora in fijadas:
            filas[e - 1].update(auc_criterio=v, mejor=mejora)
            if mejora:
                pesos, mejor_probs = pendientes[e]
                guardar_pesos(dir_out / "best.pt", pesos, arquitectura, e, filas[e - 1]["auc_val"])
            if args.criterio == "auc":
                texto += " | mejor" if mejora else ""
            else:
                texto += f" | {args.criterio} época {e} {v:.4f}" + (" mejor" if mejora else "")
            del pendientes[e]
        pd.DataFrame(filas).to_csv(dir_exp / "metricas_epoca.csv", index=False)
        return texto

    t_inicio = time.perf_counter()
    for epoca in range(1, args.epocas + 1):
        monitor.iniciar()
        if cuda:
            torch.cuda.reset_peak_memory_stats(dispositivo)
        t0 = time.perf_counter()
        r = entrenar_epoca(modelo, lotes_fijos if args.overfit_batches else dl_entr, n_lotes,
                           criterio, optimizador, dispositivo, transformar)
        if primera is None:
            primera = r["primera"]
            log.info(f"Pérdida del primer lote: {primera:.4f} (referencia ln 2 = 0.6931)")
        t1 = time.perf_counter()
        probs, perdida_val = predecir(modelo, dl_val, criterio, dispositivo)
        ev = uc.evaluar_por_paciente(probs, val_eval, umbral=UMBRAL, metodo=METODO)
        t2 = time.perf_counter()
        recursos = monitor.detener()

        auc = ev["auc"]
        pendientes[epoca] = ({k: v.detach().clone() for k, v in modelo.state_dict().items()}, probs)
        guardar_pesos(dir_out / "last.pt", modelo.state_dict(), arquitectura, epoca, auc, optimizador)

        filas.append({
            "epoca": epoca,
            "perdida_entr": r["perdida"],
            "perdida_val": perdida_val,
            "auc_val": auc,
            "sensibilidad_val": ev["sensibilidad"],
            "especificidad_val": ev["especificidad"],
            "lr": optimizador.param_groups[0]["lr"],
            "tiempo_epoca_s": t2 - t0,
            "tiempo_datos_s": r["t_datos"],
            "tiempo_calculo_s": r["t_calculo"],
            "tiempo_val_s": t2 - t1,
            "imagenes_s": r["imagenes"] / (r["t_datos"] + r["t_calculo"]),
            **recursos,
            "gpu_mem_max_mb": torch.cuda.max_memory_allocated(dispositivo) / 2**20 if cuda else np.nan,
            "auc_criterio": np.nan,                   # valor del criterio, cuando se fija
            "mejor": False,
        })
        texto = fijar(sel.anadir(auc))
        f = filas[-1]
        log.info(f"Época {epoca}/{args.epocas} | pérdida entr {f['perdida_entr']:.4f} val {perdida_val:.4f} | "
                 f"AUC val {auc:.4f} sens {f['sensibilidad_val']:.3f} esp {f['especificidad_val']:.3f} | "
                 f"{f['tiempo_epoca_s']:.0f} s (datos {r['t_datos']:.0f}, cálculo {r['t_calculo']:.0f}, "
                 f"val {f['tiempo_val_s']:.0f}) | {f['imagenes_s']:.1f} img/s | CPU {f['cpu_pct_medio']:.0f} % | "
                 f"RAM {f['ram_max_mb']:.0f} MB" + texto)
        if sel.parar:
            parada_temprana = True
            log.info(f"Early stopping: {paciencia} épocas sin mejorar el criterio {args.criterio}")
            break
    texto = fijar(sel.cerrar())
    if texto:
        log.info("Al terminar, con las épocas disponibles" + texto)
    mejor_epoca = sel.mejor_epoca
    tiempo_total = time.perf_counter() - t_inicio

    if args.overfit_batches:
        _, perdida_eval = predecir(modelo, lotes_fijos, criterio, dispositivo)
        log.info(f"Sobreajuste: pérdida en los lotes fijos con model.eval() = {perdida_eval:.4f}")

    # Informe con el mejor modelo, recargado desde disco
    estado = torch.load(dir_out / "best.pt", map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(estado["modelo"])
    probs, _ = predecir(modelo, dl_val, criterio, dispositivo)
    diferencia = float(np.abs(probs - mejor_probs).max())
    log.info(f"Mejor época {mejor_epoca}; best.pt recargado: diferencia máxima con las predicciones "
             f"de esa época = {diferencia:.2e}")

    pred = val_eval[["sample_id", "patient_id", "cohorte", "slice_index", "pCR"]].reset_index(drop=True)
    pred["prob"] = probs
    por_paciente = uc.agregar_por_paciente(pred.patient_id.values, probs, METODO, UMBRAL).prob
    (pred.rename(columns={"prob": "prob_corte"})
         .assign(prob_paciente=pred.patient_id.map(por_paciente))
         .to_csv(dir_out / "predicciones_val.csv", index=False))

    contexto = {"ejecucion": ejecucion, "fold": args.fold, "modelo": "CNNBase", "n_parametros": n_parametros,
                "bloques": args.bloques, "realce": args.realce, "criterio": args.criterio,
                "dispositivo": f"{dispositivo} ({config['entorno']['gpu'] or config['entorno']['cpu']})",
                "pos_weight": None if pos_weight is None else round(pos_weight, 4),
                "weight_decay": args.weight_decay, "aumento": aumento.etiqueta() if args.aumento else None,
                "epocas_ejecutadas": len(filas), "mejor_epoca": mejor_epoca}
    final = informe.generar_informe(dir_exp, pd.DataFrame(filas), pred, tasa_cohorte, contexto, UMBRAL, METODO)
    log.info(f"Mejor modelo (validación, por paciente): AUC {final['auc']:.4f} | AP {final['average_precision']:.4f} "
             f"| sens {final['sensibilidad']:.3f} | esp {final['especificidad']:.3f} | Brier {final['brier']:.4f} | "
             f"AUC solo cohorte {final['auc_solo_cohorte']:.4f}")

    config["resultado"] = {
        "perdida_primer_lote": primera,
        "epocas_ejecutadas": len(filas),
        "mejor_epoca": mejor_epoca,
        "valor_criterio_mejor_epoca": sel.mejor_valor,
        "parada_temprana": parada_temprana,
        "tiempo_total_s": tiempo_total,
        "tiempo_medio_epoca_s": float(np.mean([f["tiempo_epoca_s"] for f in filas])),
        "auc_val": final["auc"],
        "average_precision_val": final["average_precision"],
        "auc_solo_cohorte": final["auc_solo_cohorte"],
    }
    config["sha256"] = {n: sha256(dir_out / n) for n in ("best.pt", "last.pt")}
    informe.guardar_json(ruta_config, config)

    if not args.prueba:
        anadir_fila(DIR_EXPERIMENTOS / "experimentos.csv", {
            "ejecucion": ejecucion, "fecha_hora": config["fecha_hora"],
            "git_commit": config["entorno"]["git_commit"], "dispositivo": config["entorno"]["dispositivo"],
            "modelo": "CNNBase", "n_parametros": n_parametros, "fold_val": args.fold, "semilla": args.semilla,
            "optimizador": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "epocas_max": args.epocas, "paciencia": paciencia,
            "pos_weight": pos_weight, "aumento_datos": aumento.etiqueta() if args.aumento else None,
            "epocas_ejecutadas": len(filas), "mejor_epoca": mejor_epoca,
            "auc_val": final["auc"], "sensibilidad_val": final["sensibilidad"],
            "especificidad_val": final["especificidad"], "accuracy_val": final["accuracy"],
            "precision_val": final["precision"], "brier_val": final["brier"],
            "auc_solo_cohorte": final["auc_solo_cohorte"],
            **{f"auc_{c}": v for c, v in final["auc_por_cohorte"].items()},
            "tiempo_total_min": tiempo_total / 60,
            # Columnas añadidas el 2026-10-22: en las filas anteriores quedan vacías
            # (equivalen a criterio auc, 4 bloques, sin realce).
            "criterio": args.criterio, "bloques": args.bloques, "realce": args.realce,
            "average_precision_val": final["average_precision"],
            # Añadida el 2026-10-24 (vacía antes: sin recorte).
            "recorte_caja": args.recorte_caja,
        })
        log.info("Fila añadida a experimentos/experimentos.csv")
    log.info(f"Hecho en {tiempo_total / 60:.1f} min. Resultados en {relativa(dir_exp)}"
             + ("" if args.prueba else f" y {relativa(dir_out)}"))


if __name__ == "__main__":                            # obligatorio en Windows con num_workers > 0
    main()
