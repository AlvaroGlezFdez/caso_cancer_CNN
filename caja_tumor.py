"""Caja del tumor de patients.csv en coordenadas del PNG de 256x256.

patients.csv trae, por paciente, una caja sraw:eraw (filas) y scol:ecol
(columnas) en píxeles del corte original, de n_xy x n_xy. La auditoría visual
(../exploracion/03_caja_tumor y 04_posicion_tumor, fuera del repositorio)
estableció cómo se pasa del corte original al PNG:

    n_xy >= 256   ventana de 256 px del corte original, sin reescalar, centrada
                  en la caja y desplazada lo justo para no salirse de la imagen
                  (con n_xy = 256 es la imagen entera)
    n_xy <  256   el corte ampliado a 256 (con 128, x2 por vecino más próximo)

y después un giro de 90° (como np.rot90): la fila del PNG sale de la columna
original (fila = 256 - columna) y la columna del PNG, de la fila original.

Con esta forma la caja cayó sobre la lesión en las 40 de 60 pacientes que
tenían una lesión visible (0 errores; las otras 20 no tenían una lesión clara
con la que comparar), y en otras 8 con n_xy 128, 312 y 448.

Las coordenadas son continuas: cada eje del PNG va de 0 a 256 y el píxel i
ocupa [i, i+1). Hay una sola caja por paciente, la misma en todos sus cortes.

Comprobación, desde la raíz del repositorio:

    python caja_tumor.py

Compara la caja con la obtenida en la auditoría en 12 pacientes de
entrenamiento (las tres cohortes y todos los n_xy) y guarda en outputs/pruebas/
una figura con la caja sobre el realce EARLY-PRE del corte central de cada una,
y otra con el recorte de preprocesado.recortar (train.py --recorte-caja): el
cuadrado sobre el PNG completo y el recorte de 128x128 (PRE, EARLY y realce).
"""

from __future__ import annotations

LADO = 256

# (fila0, fila1, col0, col1) de la auditoría visual: 03_caja_tumor/cajas.csv y,
# para n_xy 128, 312 y 448, 04_posicion_tumor/cajas_train.csv.
REFERENCIA = {
    "ISPY1_1069": (132.0, 176.0, 128.0, 176.0),          # n_xy 256
    "ISPY1_1038": (101.0, 139.0, 161.0, 198.0),          # 256
    "ISPY1_1235": (63.0, 193.0, 67.0, 189.0),            # 512
    "ISPY2-829491": (38.0, 137.0, 50.0, 172.0),          # 384
    "ISPY2-759418": (69.0, 139.0, 74.0, 153.0),          # 320
    "ACRIN-6698-328160": (141.0, 173.0, 93.0, 123.0),    # 256
    "Breast_MRI_053": (120.0, 154.0, 105.0, 151.0),      # 256
    "Breast_MRI_129": (59.0, 84.0, 115.0, 141.0),        # 256
    "Breast_MRI_512": (152.0, 196.0, 91.0, 165.0),       # 256
    "ISPY2-726091": (102.0, 150.0, 164.0, 220.0),        # 128
    "ISPY2-239061": (99.0, 174.0, 89.5, 166.5),          # 448
    "ACRIN-6698-511851": (71.5, 184.5, 45.0, 155.0),     # 312
}


def _origen_ventana(centro: float, n: float) -> float:
    """Origen de la ventana de LADO px centrada en `centro`, sin salirse de 0..n."""
    return min(max(centro - LADO / 2, 0.0), n - LADO)


def caja_en_png(paciente) -> tuple[float, float, float, float]:
    """Caja del tumor en el PNG: (fila0, fila1, col0, col1), coordenadas de 0 a 256.

    `paciente` es una fila de patients.csv (de .itertuples() o .loc[pid]) con
    n_xy, sraw, eraw, scol y ecol.
    """
    n = float(paciente.n_xy)
    if n >= LADO:
        escala = 1.0
        origen_f = _origen_ventana((paciente.sraw + paciente.eraw) / 2, n)
        origen_c = _origen_ventana((paciente.scol + paciente.ecol) / 2, n)
    else:
        escala, origen_f, origen_c = LADO / n, 0.0, 0.0
    # Caja en la ventana (o en el corte ampliado), todavía sin girar.
    f0, f1 = (paciente.sraw - origen_f) * escala, (paciente.eraw - origen_f) * escala
    c0, c1 = (paciente.scol - origen_c) * escala, (paciente.ecol - origen_c) * escala
    # Giro de 90°: fila del PNG = LADO - columna; columna del PNG = fila.
    return float(LADO - c1), float(LADO - c0), float(f0), float(f1)


def figura(cortes: list, ruta) -> None:
    """cortes: lista de (pid, n_xy, slice_index, imagen (3,256,256), caja)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ejes = plt.subplots(3, 4, figsize=(14, 11.5), layout="constrained")
    for eje, (pid, n_xy, z, imagen, (f0, f1, c0, c1)) in zip(ejes.ravel(), cortes):
        im = eje.imshow(imagen[1] - imagen[0], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        eje.add_patch(Rectangle((c0, f0), c1 - c0, f1 - f0, fill=False, ec="lime", lw=2))
        eje.set_title(f"{pid}  |  n_xy = {n_xy}  |  z = {z}", fontsize=10)
        eje.set_xticks([])
        eje.set_yticks([])
    fig.colorbar(im, ax=ejes, location="bottom", shrink=0.4, aspect=40, extend="both",
                 label="EARLY − PRE")
    fig.suptitle("Caja del tumor (caja_tumor.caja_en_png) sobre el realce del corte central\n"
                 "Debe caer sobre la lesión en todas", fontsize=12)
    fig.savefig(ruta, dpi=90)
    plt.close(fig)


def figura_recorte(cortes: list, ruta) -> None:
    """Para cada corte: PNG completo (EARLY) con la caja y el cuadrado de recorte,
    y el recorte de 128x128 en PRE, EARLY y EARLY-PRE con la caja encima."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from preprocesado import LADO_RECORTE, cuadrado_recorte, recortar

    fig, ejes = plt.subplots(6, 8, figsize=(20, 16), layout="constrained")
    for i, (pid, n_xy, z, imagen, caja) in enumerate(cortes):
        fila_ejes = ejes[i // 2, 4 * (i % 2): 4 * (i % 2) + 4]
        f0, f1, c0, c1 = caja
        fc, cc, lado = cuadrado_recorte(caja)
        rec = recortar(imagen, caja)
        e = fila_ejes[0]
        e.imshow(imagen[1], cmap="gray", vmin=0, vmax=1)
        e.add_patch(Rectangle((c0, f0), c1 - c0, f1 - f0, fill=False, ec="lime", lw=1.5))
        e.add_patch(Rectangle((cc - lado / 2, fc - lado / 2), lado, lado, fill=False, ec="orange", lw=1.5))
        e.set_xlim(-0.5, 255.5)
        e.set_ylim(255.5, -0.5)
        e.set_title(f"{pid}\nn_xy {n_xy}, z {z}, lado {lado:.0f}", fontsize=8)
        # Caja en coordenadas del recorte
        k = LADO_RECORTE / lado
        caja_rec = ((c0 - cc + lado / 2) * k, (f0 - fc + lado / 2) * k, (c1 - c0) * k, (f1 - f0) * k)
        for e, img, titulo, kw in zip(fila_ejes[1:], (rec[0], rec[1], rec[1] - rec[0]),
                                      ("PRE", "EARLY", "EARLY − PRE"),
                                      ({"cmap": "gray", "vmin": 0, "vmax": 1},) * 2
                                      + ({"cmap": "RdBu_r", "vmin": -0.5, "vmax": 0.5},)):
            e.imshow(img, **kw)
            e.add_patch(Rectangle(caja_rec[:2], *caja_rec[2:], fill=False, ec="lime", lw=1, ls="--"))
            e.set_title(f"recorte {titulo}", fontsize=8)
    for e in ejes.ravel():
        e.set_xticks([])
        e.set_yticks([])
    fig.suptitle("Recorte por la caja del tumor (preprocesado.recortar, train.py --recorte-caja)\n"
                 "verde: caja del tumor; naranja: cuadrado recortado (lado mayor + 25 %, mín. 64 px) "
                 "-> 128x128; fuera del PNG, ceros", fontsize=12)
    fig.savefig(ruta, dpi=80)
    plt.close(fig)


def main():
    import sys
    from datetime import datetime
    from pathlib import Path

    raiz = Path(__file__).resolve().parent
    sys.path.insert(0, str(raiz / "breastdcedl"))
    import utils_caso as uc
    from preprocesado import preprocesar

    sys.stdout.reconfigure(encoding="utf-8")
    patients = uc.cargar_patients(raiz / "breastdcedl").set_index("pid")
    samples = uc.cargar_samples(raiz / "breastdcedl")
    samples = samples[samples.split == "train"]

    fallos, cortes = [], []
    for pid, esperada in REFERENCIA.items():
        paciente = patients.loc[pid]
        caja = caja_en_png(paciente)
        diferencia = max(abs(a - b) for a, b in zip(caja, esperada))
        bien = diferencia <= 1e-6
        print(f"  {'OK   ' if bien else 'FALLO'} {pid:18s} n_xy {int(paciente.n_xy):3d}  "
              f"caja {tuple(round(v, 1) for v in caja)}  diferencia máxima {diferencia:.1e}")
        fallos += [] if bien else [pid]

        filas = samples[samples.patient_id == pid].sort_values("slice_index")
        fila = filas.iloc[len(filas) // 2]
        imagen = preprocesar(*(raiz / "breastdcedl" / r
                               for r in (fila.path_pre, fila.path_early, fila.path_late)))
        cortes.append((pid, int(paciente.n_xy), int(fila.slice_index), imagen, caja))
    if fallos:
        raise SystemExit("ERROR: la caja no coincide con la de la auditoría en " + ", ".join(fallos))

    ruta = raiz / "outputs" / "pruebas" / f"{datetime.now():%Y%m%d_%H%M%S}_caja_tumor.png"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura(cortes, ruta)
    ruta_recorte = ruta.with_name(ruta.stem + "_recorte.png")
    figura_recorte(cortes, ruta_recorte)
    print(f"Las {len(REFERENCIA)} cajas coinciden con la auditoría. "
          f"Figuras: {ruta.relative_to(raiz).as_posix()} y {ruta_recorte.relative_to(raiz).as_posix()}")


if __name__ == "__main__":
    main()
