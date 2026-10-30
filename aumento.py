"""Aumento de datos geométrico para el entrenamiento (train.py --aumento).

Solo se aplica a los lotes de entrenamiento: nunca a validación, test ni la app,
que usan preprocesado.preprocesar sin más. A cada corte del lote se le sortea

    volteo horizontal   con probabilidad 0,5
    rotación            ángulo uniforme entre -10 y +10 grados, alrededor del
                        centro; interpolación bilineal y relleno con 0 (negro)

y esa misma transformación se aplica a sus tres canales (PRE, EARLY, LATE) con
una única rejilla de muestreo, así que las fases siguen alineadas y el realce
(EARLY-PRE, LATE-EARLY) se conserva. Nada de brillo, contraste ni otras
transformaciones de intensidad: los canales son fases temporales y su
intensidad relativa es la señal.

train.py lo aplica al lote ya movido al dispositivo (la GPU si la hay), y los
parámetros se sortean con un torch.Generator propio en CPU. Así el resultado no
depende del dispositivo ni de --num-workers, y no se consumen ni el generador
global de PyTorch (el del dropout) ni el del barajado: con y sin --aumento la
red ve los mismos cortes en el mismo orden.

Comprobación, desde la raíz del repositorio:

    python aumento.py

Guarda en outputs/pruebas/ una figura con un corte de entrenamiento original y
tres versiones aumentadas (PRE, EARLY, LATE y EARLY-PRE de cada una) y comprueba
numéricamente que los tres canales reciben la misma transformación.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

PROB_VOLTEO = 0.5
ROTACION_MAX_GRADOS = 10.0


def descripcion() -> dict:
    """Lo que train.py guarda en config.json."""
    return {
        "funcion": "aumento.aumentar",
        "solo_entrenamiento": True,
        "misma_transformacion_en_las_3_fases": True,
        "volteo_horizontal_prob": PROB_VOLTEO,
        "rotacion_max_grados": ROTACION_MAX_GRADOS,
        "rotacion_distribucion": "uniforme",
        "interpolacion": "bilineal",
        "relleno": 0.0,
        "intensidad": "sin cambios",
    }


def etiqueta() -> str:
    """Resumen corto para experimentos.csv."""
    return f"volteo_h{PROB_VOLTEO:g}_rot{ROTACION_MAX_GRADOS:g}"


def sortear(n: int, generador: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Parámetros para n cortes: volteo (bool) y ángulo en grados."""
    volteo = torch.rand(n, generator=generador) < PROB_VOLTEO
    angulo = (2 * torch.rand(n, generator=generador) - 1) * ROTACION_MAX_GRADOS
    return volteo, angulo


def aplicar(x: torch.Tensor, volteo: torch.Tensor, angulo: torch.Tensor) -> torch.Tensor:
    """x (B, C, H, W) con H == W. Voltea en horizontal los cortes con volteo=True y
    después rota cada corte `angulo` grados. Todos los canales de un corte
    comparten la misma rejilla de muestreo."""
    if x.shape[-1] != x.shape[-2]:
        raise ValueError("la rotación en coordenadas normalizadas supone imágenes cuadradas")
    x = torch.where(volteo.to(x.device).view(-1, 1, 1, 1), x.flip(-1), x)
    rad = torch.deg2rad(angulo.to(device=x.device, dtype=x.dtype))
    cos, sen, cero = torch.cos(rad), torch.sin(rad), torch.zeros_like(rad)
    theta = torch.stack([torch.stack([cos, -sen, cero], dim=1),
                         torch.stack([sen, cos, cero], dim=1)], dim=1)          # (B, 2, 3)
    rejilla = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, rejilla, mode="bilinear", padding_mode="zeros", align_corners=False)


def aumentar(x: torch.Tensor, generador: torch.Generator) -> torch.Tensor:
    """Lo que usa train.py con --aumento: una transformación sorteada por corte."""
    return aplicar(x, *sortear(len(x), generador))


# --------------------------------------------------------------------------- #
# Comprobación (python aumento.py)
# --------------------------------------------------------------------------- #

def comprobar(x: torch.Tensor, volteo: torch.Tensor, angulo: torch.Tensor) -> dict:
    """x (3, H, W). Nombre -> (diferencia máxima, tolerancia).

    La alineación entre canales debe ser exacta. La identidad y el volteo pasan
    por grid_sample, cuya rejilla en float32 se desvía ~1e-5 píxeles: de ahí
    errores de ~1e-5, muy por debajo de un nivel de gris (1/255 = 3,9e-3)."""
    b, n = x.unsqueeze(0), len(volteo)
    no, si, cero = torch.tensor([False]), torch.tensor([True]), torch.tensor([0.0])
    # Las tres fases iguales: si un canal recibiera otra transformación, saldría distinto
    iguales = aplicar(x[:1].expand(3, -1, -1).unsqueeze(0).repeat(n, 1, 1, 1), volteo, angulo)
    # El realce de la versión aumentada es el realce aumentado (la bilineal es lineal)
    aumentado = aplicar(b.repeat(n, 1, 1, 1), volteo, angulo)
    realce = aplicar((x[1] - x[0]).expand(n, 1, -1, -1), volteo, angulo)[:, 0]
    return {
        "identidad (sin volteo, 0 grados)": (float((aplicar(b, no, cero) - b).abs().max()), 1e-4),
        "solo volteo frente a flip": (float((aplicar(b, si, cero) - b.flip(-1)).abs().max()), 1e-4),
        "tres fases iguales -> tres canales iguales": (float((iguales - iguales[:, :1]).abs().max()), 0.0),
        "EARLY-PRE aumentado frente a aumento de EARLY-PRE":
            (float((aumentado[:, 1] - aumentado[:, 0] - realce).abs().max()), 1e-6),
    }


def figura(x: torch.Tensor, volteo: torch.Tensor, angulo: torch.Tensor, sample_id: str,
           comprobaciones: dict, ruta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Ink y diverging de la paleta de informe.py: azul <-> rojo con gris neutro en 0
    tinta, tinta_2, anillo = "#0b0b0b", "#52514e", "#eb6834"
    divergente = LinearSegmentedColormap.from_list(
        "realce", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#a8302f"])

    versiones = torch.cat([x.unsqueeze(0), aplicar(x.unsqueeze(0).repeat(len(volteo), 1, 1, 1), volteo, angulo)])
    realce = versiones[:, 1] - versiones[:, 0]
    vmax = float(torch.quantile(x.flatten(), 0.995))
    lim = float(torch.quantile(realce[0].abs().flatten(), 0.995))
    # Punto de máximo realce (suavizado) de cada versión: se marca igual en las 4 columnas
    suave = F.avg_pool2d(realce.unsqueeze(1), 9, stride=1, padding=4)[:, 0]
    picos = [divmod(int(s.argmax()), s.shape[-1]) for s in suave]

    filas = ["original"] + [
        f"aumento {i}\nvolteo: {'sí' if v else 'no'} · rotación {a:+.1f}°".replace(".", ",")
        for i, (v, a) in enumerate(zip(volteo.tolist(), angulo.tolist()), 1)]
    columnas = ["PRE", "EARLY", "LATE", "EARLY − PRE"]

    fig, ejes = plt.subplots(len(filas), 4, figsize=(12, 3.05 * len(filas) + 1.2), constrained_layout=True)
    for i, fila in enumerate(ejes):
        imagenes = [versiones[i, 0], versiones[i, 1], versiones[i, 2], realce[i]]
        for j, (ax, img) in enumerate(zip(fila, imagenes)):
            if j < 3:
                im_gris = ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=vmax)
            else:
                im_realce = ax.imshow(img.numpy(), cmap=divergente, vmin=-lim, vmax=lim)
            fy, fx = picos[i]
            ax.add_patch(plt.Circle((fx, fy), 12, fill=False, color=anillo, lw=1.6))
            ax.set_xticks([])
            ax.set_yticks([])
            for borde in ax.spines.values():
                borde.set_visible(False)
            if i == 0:
                ax.set_title(columnas[j], color=tinta, fontsize=12)
        fila[0].set_ylabel(filas[i], color=tinta, fontsize=11)

    fig.colorbar(im_gris, ax=ejes[:, :3], location="bottom", shrink=0.5, aspect=40,
                 label="intensidad (0-1, tras dividir entre 255)")
    fig.colorbar(im_realce, ax=ejes[:, 3], location="bottom", shrink=0.9, aspect=14, label="EARLY − PRE")
    canales = comprobaciones["tres fases iguales -> tres canales iguales"][0]
    fig.suptitle(f"Aumento geométrico: la misma transformación en las tres fases · corte {sample_id}\n"
                 f"El anillo marca el punto de máximo realce de cada versión, en la misma posición en las "
                 f"cuatro columnas. Con las tres fases iguales,\nlos tres canales aumentados salen idénticos "
                 f"(diferencia máxima {canales:.1e})",
                 color=tinta, fontsize=12)
    for cb in fig.axes[-2:]:
        cb.xaxis.label.set_color(tinta_2)
        cb.tick_params(colors=tinta_2)
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    import argparse
    import sys
    from datetime import datetime
    from pathlib import Path

    import numpy as np

    raiz = Path(__file__).resolve().parent
    sys.path.insert(0, str(raiz / "breastdcedl"))
    import utils_caso as uc
    from preprocesado import preprocesar

    ap = argparse.ArgumentParser(description="Figura y comprobación del aumento de datos.")
    ap.add_argument("--sample-id", help="corte a mostrar; por defecto, el de mayor realce entre los "
                                        "cortes de las 40 primeras pacientes de entrenamiento (fold 0)")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    entrenamiento, _ = uc.particion(uc.cargar_samples(raiz / "breastdcedl"), fold_val=0)

    def cargar(fila):
        return torch.from_numpy(preprocesar(*(raiz / "breastdcedl" / r
                                              for r in (fila.path_pre, fila.path_early, fila.path_late))))

    if args.sample_id:
        fila = entrenamiento.loc[entrenamiento.sample_id == args.sample_id].iloc[0]
    else:
        pacientes = entrenamiento.patient_id.drop_duplicates().iloc[:40]
        candidatos = entrenamiento[entrenamiento.patient_id.isin(pacientes)]
        realce = [float(np.quantile((x[1] - x[0]).numpy(), 0.99))
                  for x in (cargar(f) for f in candidatos.itertuples())]
        fila = candidatos.iloc[int(np.argmax(realce))]
    x = cargar(fila)

    # Tres sorteos con la función de entrenamiento; se repite hasta que salgan
    # cortes con y sin volteo, para que la figura muestre los dos casos.
    generador = torch.Generator().manual_seed(args.semilla)
    while True:
        volteo, angulo = sortear(3, generador)
        if volteo.any() and not volteo.all():
            break

    comprobaciones = comprobar(x, volteo, angulo)
    fallos = []
    for nombre, (valor, tolerancia) in comprobaciones.items():
        bien = valor <= tolerancia
        print(f"  {'OK   ' if bien else 'FALLO'} {nombre}: diferencia máxima {valor:.2e} "
              f"(tolerancia {tolerancia:g})")
        fallos += [] if bien else [nombre]
    if fallos:
        raise SystemExit("ERROR en la comprobación del aumento: " + "; ".join(fallos))

    ruta = raiz / "outputs" / "pruebas" / f"{datetime.now():%Y%m%d_%H%M%S}_aumento_fases.png"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura(x, volteo, angulo, fila.sample_id, comprobaciones, ruta)
    print(f"Corte {fila.sample_id}; sorteos: " + "; ".join(
        f"volteo {'sí' if v else 'no'}, rotación {a:+.2f}°" for v, a in zip(volteo.tolist(), angulo.tolist())))
    print(f"Figura: {ruta.relative_to(raiz).as_posix()}")


if __name__ == "__main__":
    main()
