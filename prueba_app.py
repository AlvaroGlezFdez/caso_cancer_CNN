"""Prueba automática de la app (fase 1). Desde la raíz del repositorio, en la
máquina que tenga los pesos y las predicciones de la ejecución exportada:

    python prueba_app.py

1. Paquete: modelo_app/ se carga, su SHA-256 coincide y se detecta si se altera;
   el umbral es el elegido por corte (la entrada de la app) en predicciones_val.csv;
   el recorte (por la caja o ventana centrada) y la época son los de config.json
   (con --criterio ultima, la última). Pesos desde Hugging Face (en una copia del
   paquete sin modelo.pt): con un SHA-256 de modelo.json que no es el suyo, la
   descarga se rechaza con un error claro y no deja fichero; con el bueno, se
   descargan y se cargan los mismos pesos; si Hugging Face no responde, error en
   español. Sin conexión, las dos primeras salen como AVISO.
2. App = train.py, exacto: para cortes de validación de la ejecución exportada,
   la app (nucleo.inferencia.analizar, el mismo código que la interfaz) y
   train.predecir (DatasetCortes -> preprocesado.preprocesar -> modelo de best.pt
   -> sigmoide) dan el mismo logit y la misma probabilidad, bit a bit, con la
   misma muestra.
3. App frente a predicciones_val.csv (lo que train.py guardó al entrenar, en lotes
   de 32): mismo valor salvo, en algunos cortes, 1-2 unidades del último bit del
   float32. Los logits son idénticos; la diferencia está en la sigmoide, que
   PyTorch calcula con una rutina vectorizada para lotes y escalar para un único
   valor. Si el modelo se entrenó en GPU se admite hasta 1e-5 (otro dispositivo).
   Si el modelo recorta (por la caja o con la ventana centrada), la app recibe la
   caja de cada paciente, como train.py.
4. Ejemplos: píxeles idénticos al dataset, resultado igual al de train.predecir
   y reproducible.
5. Validación: 20 entradas incorrectas o dudosas dan el error o el aviso
   esperado, con mensaje en español y sin excepciones.
6. Interfaz (streamlit.testing.v1.AppTest): arranca sin excepciones; cinco
   muestras distintas se cargan una tras otra con «Nuevo caso», sin reiniciar,
   y cada una muestra su probabilidad exacta; un fichero erróneo muestra un
   error claro; los ejemplos funcionan.
7. Secciones: la página por defecto es la portada (titular, cifras de los datos y
   el botón «Analizar un caso», que abre el análisis); las seis páginas restantes
   se cargan sin excepciones; Rendimiento
   muestra primero el test independiente (cifras de experimentos/test/, el mismo
   paquete por SHA-256) y debajo, separada, la validación cruzada; Grad-CAM da
   el mismo logit que la inferencia; la curva cinética clasifica bien un caso
   sintético; el informe HTML escapa los nombres de los ficheros; ?ejemplo= abre
   un ejemplo.
8. Modelo final combinado (modelo_app/fusion.json): sus coeficientes son los de
   la regresión que reajusta exportar_fusion.py con los mismos datos, el umbral
   es el de la validación cruzada por corte, y la probabilidad combinada de la
   app (nucleo.fusion, con las variables del dict o leídas de un CSV) coincide
   con la de exportar_fusion.py (LogisticRegression.predict_proba) para la misma
   muestra y las mismas variables, también con valores que faltan.
9. Variables de la paciente: 20 entradas mal formadas o dudosas (CSV ilegible,
   varias filas, columnas que faltan, subtipo desconocido, valores imposibles,
   caja incoherente…) dan el error o el aviso esperado, con mensaje claro.
10. Recorte por la caja: con un modelo que recorta (--recorte-caja), la caja de
   las variables da la misma entrada y el mismo logit que train.py
   (DatasetCortes con las cajas de caja_tumor.caja_en_png), bit a bit.
11. Defensa simulada (AppTest): cinco muestras con su fila de patients.csv en un
   CSV, una tras otra con «Nuevo caso»: la probabilidad combinada mostrada es la
   de exportar_fusion.py; variables con errores (con la caja válida, si la CNN
   la necesita) muestran un error y solo la imagen; el formulario y los ejemplos
   dan el combinado esperado; y con un paquete que recorta por la caja, sin
   variables hay un error claro y con ellas el combinado esperado. Si la CNN del
   paquete necesita la caja: sin variables o con la caja mal, la app no predice
   y lo explica; con la caja válida y sin las variables clínicas, solo la imagen
   con el aviso de que no es el modelo final.
12. Ventana centrada (--recorte-centrado N): con un paquete de ventana centrada
   (el exportado, o uno simulado de 160 px con los mismos pesos), la entrada de
   la red es preprocesar(..., caja=caja, lado_centrado=N) y el logit el de
   train.py (DatasetCortes con cajas y lado_centrado), bit a bit; la ventana que
   se muestra en el visor es la de preprocesado.ventana_centrada; sin la caja,
   error claro y ninguna predicción. En la interfaz (AppTest): sin variables,
   error; con su CSV, el resultado y el pie de la ventana bajo el visor.

14. Uso sin ayuda: la plantilla CSV descargable se acepta tal cual (sin errores
   ni avisos) y da el combinado; todo error de validación dice qué hacer; y
   cinco casos seguidos quedan en el registro de la sesión con la probabilidad
   que muestra la app.
13. Estilo (estaticos/estilo.css; AppTest no aplica CSS, así que se comprueba
   el texto): la barra lateral empieza abierta; no se oculta stToolbar (en
   Streamlit 1.66 contiene el botón para desplegar la barra) y los botones de
   plegar y desplegar se fuerzan visibles; el CSS no lleva «<» seguido de letra
   (el sanitizador de st.html descartaría la hoja entera); el selector de
   ficheros de las variables dice «Subir CSV» con el límite de nucleo/variables.py
   y los de las fases «Subir PNG» con el de nucleo/validacion.py.

Termina con código 1 si algo falla. Las comprobaciones que dependen de exportar
de nuevo en la GPU (fusion.json provisional o de otra CNN) salen como AVISO.
También las que necesitan best.pt y predicciones_val.csv de la ejecución
exportada (solo están en la máquina que entrenó): sin ellos, el modelo de
referencia de train.py se carga desde modelo_app/modelo.pt (los mismos pesos,
verificados por SHA-256) y los cortes son los de validación de su fold.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import exportar_fusion
import train                                    # añade breastdcedl/ al sys.path
import utils_caso as uc
from analisis_presentable import confusion
from caja_tumor import caja_en_png
from clinica import COLUMNAS_ENTRADA, construir_variables, logit
from models.cnn_base import crear_modelo
from preprocesado import FASES, leer_fase, preprocesar, ventana_centrada

RAIZ_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_REPO / "app"))
from nucleo import ejemplos, variables as var  # noqa: E402
from nucleo.formato import num  # noqa: E402
from nucleo.fusion import ModeloFusion, cargar_fusion  # noqa: E402
from nucleo.inferencia import DatosPaciente, analizar, necesita_caja  # noqa: E402
from nucleo.paquete import ErrorPaquete, PaqueteModelo, cargar_paquete  # noqa: E402
from nucleo.validacion import AVISO, ERROR, OK, comprobar_preprocesado, Informe  # noqa: E402

N_PACIENTES = 25                    # pacientes de validación para la comparación con train.py
ULP_MAXIMOS = 2                     # diferencia admitida con predicciones_val.csv (CPU)
TOLERANCIA_OTRO_DISPOSITIVO = 1e-5
TOLERANCIA_FUSION = 1e-12           # app (numpy) frente a sklearn: mismo cálculo, otro orden de sumas
COLUMNAS_CAJA = ("n_xy", "sraw", "eraw", "scol", "ecol")

fallos: list[str] = []
avisos: list[str] = []


def comprobar(nombre: str, bien: bool, detalle: str = "") -> bool:
    print(f"  {'OK   ' if bien else 'FALLO'} {nombre}" + (f": {detalle}" if detalle else ""))
    if not bien:
        fallos.append(nombre)
    return bien


def avisar(nombre: str, detalle: str = "") -> None:
    """Algo que no es un fallo de la app pero hay que resolver (p. ej., reexportar en la GPU)."""
    print(f"  AVISO {nombre}" + (f": {detalle}" if detalle else ""))
    avisos.append(nombre)


def ficheros_de(fila, nombre: str | None = None) -> dict:
    """Los 3 PNG de un corte del dataset, como los recibe la app."""
    base = nombre or fila.sample_id
    return {f: (f"{base}_{f}.png", (train.RAIZ_DATOS / r).read_bytes())
            for f, r in zip(FASES, (fila.path_pre, fila.path_early, fila.path_late))}


def recorta(paquete: PaqueteModelo) -> bool:
    """¿Necesita la caja de la paciente (recorte por la caja o ventana centrada)?"""
    return necesita_caja(paquete.meta)


def con_recorte_caja(meta: dict) -> dict:
    """Copia de modelo.json simulando un modelo con --recorte-caja (y sin ventana centrada)."""
    meta = copy.deepcopy(meta)
    meta["preprocesado"].update(recorte_caja={"margen": 1.25, "tamano_salida": [128, 128]}, recorte_centrado=None)
    meta["modelo"]["entrada"] = [3, 128, 128]
    return meta


def datos_paciente(paquete: PaqueteModelo, pid: str) -> DatosPaciente | None:
    """La caja de la paciente (patients.csv) si el modelo recorta por ella; si no, None."""
    if not recorta(paquete):
        return None
    fila = patients().loc[pid]
    return DatosPaciente(*(float(fila[c]) for c in COLUMNAS_CAJA))


def datos_ejemplo(paquete: PaqueteModelo, clave: str) -> DatosPaciente | None:
    if not recorta(paquete):
        return None
    v = next(e for e in ejemplos.indice()["ejemplos"] if e["clave"] == clave)["variables"]
    return DatosPaciente(*(v[c] for c in COLUMNAS_CAJA))


def via_train(modelo, filas: pd.DataFrame, paquete: PaqueteModelo | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidad (train.predecir) y logit de cada corte, de uno en uno (con la caja
    de cada paciente si el modelo recorta, como train.py --recorte-caja o
    --recorte-centrado)."""
    cajas = ({pid: caja_en_png(patients().loc[pid]) for pid in filas.patient_id.unique()}
             if paquete is not None and recorta(paquete) else None)
    lado = paquete.meta["preprocesado"].get("recorte_centrado") if paquete is not None else None
    dl = DataLoader(train.DatasetCortes(filas.reset_index(drop=True), train.RAIZ_DATOS, cajas=cajas,
                                        lado_centrado=lado), batch_size=1, shuffle=False, num_workers=0)
    probs, _ = train.predecir(modelo, dl, nn.BCEWithLogitsLoss(), torch.device("cpu"))
    with torch.no_grad():
        logits = np.array([modelo(x).item() for x, _ in dl], dtype=np.float32)
    return probs, logits


def dir_run(paquete: PaqueteModelo) -> Path:
    return RAIZ_REPO / "outputs" / "runs" / paquete.meta["ejecucion"]


def hay_salidas(paquete: PaqueteModelo) -> bool:
    """¿Están best.pt y predicciones_val.csv de la ejecución (solo en la máquina que entrenó)?"""
    return all((dir_run(paquete) / n).is_file() for n in ("best.pt", "predicciones_val.csv"))


def modelo_referencia(paquete: PaqueteModelo):
    """El modelo de train.py: crear_modelo + best.pt de la ejecución o, si no está aquí,
    modelo_app/modelo.pt (el state_dict de best.pt, verificado por SHA-256)."""
    mo = paquete.meta["modelo"]
    modelo = crear_modelo(mo["bloques"], mo["realce"], float(mo.get("dropout", 0.3)))
    if hay_salidas(paquete):
        estado = torch.load(dir_run(paquete) / "best.pt", map_location="cpu", weights_only=True)["modelo"]
    else:
        estado = torch.load(RAIZ_REPO / "modelo_app" / mo["fichero"], map_location="cpu", weights_only=True)
    modelo.load_state_dict(estado)
    return modelo.eval()


def png(arr: np.ndarray, **kw) -> bytes:
    salida = io.BytesIO()
    Image.fromarray(arr).save(salida, format="PNG", **kw)
    return salida.getvalue()


# --------------------------------------------------------------------------- #

def prueba_paquete() -> PaqueteModelo:
    print("1. Paquete del modelo")
    paquete = cargar_paquete()
    comprobar("modelo_app/ se carga y el SHA-256 de modelo.pt coincide con modelo.json", True,
              f"{paquete.sha256[:16]}…, ejecución {paquete.meta['ejecucion']}")
    comprobar("modelo en eval() y sin gradientes", not paquete.modelo.training
              and not any(p.requires_grad for p in paquete.modelo.parameters()))
    texto = (RAIZ_REPO / "modelo_app" / "modelo.json").read_text(encoding="utf-8")
    comprobar("modelo.json sin ids de pacientes ni de cortes",
              not any(s in texto for s in ("ISPY1_", "ISPY2-", "ACRIN-6698", "Breast_MRI_", "patient_id")))
    with tempfile.TemporaryDirectory() as tmp:
        copia = Path(tmp) / "modelo_app"
        shutil.copytree(RAIZ_REPO / "modelo_app", copia)
        with open(copia / "modelo.pt", "ab") as f:
            f.write(b"\0")                                    # un byte de más
        try:
            cargar_paquete(copia)
            alterado = False
        except ErrorPaquete:
            alterado = True
    comprobar("un modelo.pt alterado se rechaza (SHA-256 distinto)", alterado)

    # Pesos desde Hugging Face, en copias del paquete sin modelo.pt (lo de disco no se toca).
    from nucleo import paquete as nucleo_paquete
    temporal_hf = nucleo_paquete.DIR_TEMPORAL / "modelo.pt"
    previo_hf = temporal_hf.exists()
    with tempfile.TemporaryDirectory() as tmp:
        copia = Path(tmp) / "modelo_app"
        copia.mkdir()
        meta = json.loads((RAIZ_REPO / "modelo_app" / "modelo.json").read_text(encoding="utf-8"))
        meta["modelo"]["sha256"] = "0" * 64                     # un SHA-256 que no es el de los pesos
        (copia / "modelo.json").write_text(json.dumps(meta), encoding="utf-8")
        try:
            cargar_paquete(copia)
            mensaje = ""
        except ErrorPaquete as e:
            mensaje = str(e)
        if "no se pudieron descargar" in mensaje.lower() or "respondió con un error" in mensaje:
            avisar("descarga desde Hugging Face con SHA-256 incorrecto", f"sin conexión: {mensaje[:70]}")
        else:
            comprobar("descarga desde Hugging Face con SHA-256 incorrecto -> error claro, sin modelo ni fichero",
                      "no coinciden" in mensaje and "no se carga el modelo" in mensaje
                      and not (copia / "modelo.pt").exists() and temporal_hf.exists() == previo_hf, mensaje[:90])
        (copia / "modelo.json").write_text((RAIZ_REPO / "modelo_app" / "modelo.json").read_text(encoding="utf-8"),
                                           encoding="utf-8")
        try:
            descargado = cargar_paquete(copia)
            comprobar("sin modelo.pt, los pesos se descargan de Hugging Face (revisión fijada) y son los mismos",
                      (copia / "modelo.pt").exists() and descargado.sha256 == paquete.sha256
                      and all(torch.equal(a, b) for a, b in zip(descargado.modelo.state_dict().values(),
                                                                 paquete.modelo.state_dict().values())),
                      f"revisión {nucleo_paquete.REVISION_HF[:12]}, SHA-256 {descargado.sha256[:12]}")
        except ErrorPaquete as e:
            avisar("descarga desde Hugging Face", f"sin conexión: {str(e)[:70]}")
    try:
        nucleo_paquete.descargar_pesos(Path(tempfile.gettempdir()) / "no_se_guarda" / "modelo.pt", paquete.sha256,
                                       url="http://127.0.0.1:9/modelo.pt", espera=3)
        caido = ""
    except ErrorPaquete as e:
        caido = str(e)
    comprobar("si Hugging Face no responde, error en español y nada guardado",
              "No se pudieron descargar los pesos" in caido
              and not (Path(tempfile.gettempdir()) / "no_se_guarda" / "modelo.pt").exists(), caido[:80])

    # El umbral es el de la entrada de la app (un corte): el de mayor sensibilidad con
    # especificidad >= 0,85 sobre la probabilidad de cada corte de validación.
    from analisis_presentable import confusion
    m = paquete.meta["metricas_validacion"]
    if hay_salidas(paquete):
        pred = pd.read_csv(dir_run(paquete) / "predicciones_val.csv")
        elegido = confusion(pred.pCR.to_numpy().astype(int), pred.prob_corte.to_numpy().astype(float))
        mc = m["con_umbral"]["matriz_confusion"]
        comprobar("umbral elegido por corte (especificidad >= 0,85) y métricas por corte con ese umbral",
                  paquete.umbral == elegido["umbral"] and paquete.meta["umbral"]["nivel"].startswith("corte")
                  and m["cortes"] == len(pred) and (mc["VP"], mc["FN"], mc["FP"], mc["VN"])
                  == (elegido["vp"], elegido["fn"], elegido["fp"], elegido["vn"]) and elegido["especificidad"] >= 0.85,
                  f"umbral {paquete.umbral:.4f}, especificidad {elegido['especificidad']:.3f}, "
                  f"sensibilidad {elegido['sensibilidad']:.3f}")
        comprobar("las métricas por paciente se mantienen como información adicional",
                  m["por_paciente"]["pacientes"] == pred.patient_id.nunique() and "umbral" in m["por_paciente"])
    else:
        avisar("umbral por corte frente a predicciones_val.csv",
               "sin las salidas de la ejecución en esta máquina; se comprueba en la GPU")
        comprobar("umbral por corte y métricas por paciente en modelo.json",
                  paquete.meta["umbral"]["nivel"].startswith("corte") and "umbral" in m["por_paciente"],
                  f"umbral {paquete.umbral:.4f}")

    # Recorte y época, los de la ejecución (config.json).
    config = json.loads((RAIZ_REPO / "experimentos" / paquete.meta["ejecucion"] / "config.json")
                        .read_text(encoding="utf-8"))
    hiper, resultado = config["hiperparametros"], config.get("resultado") or {}
    lado_config = (hiper.get("recorte_centrado") or {}).get("lado_px")
    preproc = paquete.meta["preprocesado"]
    entrada_config = ([lado_config] * 2 if lado_config else
                      (hiper.get("recorte_caja") or {}).get("tamano_salida") or [256, 256])
    comprobar("recorte de modelo.json = el de config.json (recorte_caja y recorte_centrado)",
              bool(preproc.get("recorte_caja")) == bool(hiper.get("recorte_caja"))
              and preproc.get("recorte_centrado") == lado_config
              and list(paquete.meta["modelo"]["entrada"][1:]) == list(entrada_config),
              f"recorte por la caja {'sí' if preproc.get('recorte_caja') else 'no'}, ventana centrada "
              f"{preproc.get('recorte_centrado') or 'no'}, entrada {paquete.meta['modelo']['entrada']}")
    criterio = hiper.get("criterio_seleccion", "auc")
    sel = paquete.meta["modelo"].get("seleccion") or {}
    comprobar("época y criterio en modelo.json" + (" (criterio ultima: la última época)" if criterio == "ultima" else ""),
              sel.get("criterio") == criterio and sel.get("epoca_usada") == paquete.meta["modelo"]["epoca"]
              and (criterio != "ultima" or paquete.meta["modelo"]["epoca"] == resultado.get("epocas_ejecutadas")),
              f"época {paquete.meta['modelo']['epoca']} de {resultado.get('epocas_ejecutadas')}, criterio {criterio}")
    return paquete


def prueba_equivalencia(paquete: PaqueteModelo) -> pd.DataFrame:
    print("2-3. La app frente a train.py")
    ejecucion = paquete.meta["ejecucion"]
    samples = uc.cargar_samples(train.RAIZ_DATOS)
    if hay_salidas(paquete):
        pred = pd.read_csv(dir_run(paquete) / "predicciones_val.csv")
    else:                                       # los cortes de validación de su fold, en el orden de samples.csv
        fold = paquete.meta["entrenamiento"]["fold_validacion"]
        pred = samples[(samples.split == "train") & (samples.fold == fold)][["sample_id", "patient_id"]]
    pacientes = pred.patient_id.drop_duplicates().to_numpy()
    elegidas = pacientes[np.linspace(0, len(pacientes) - 1, N_PACIENTES).astype(int)]
    pred = pred[pred.patient_id.isin(elegidas)].reset_index(drop=True)
    filas = samples.set_index("sample_id").loc[pred.sample_id].reset_index()

    p_train, l_train = via_train(modelo_referencia(paquete), filas, paquete)

    resultados = [analizar(paquete, ficheros_de(f), datos_paciente(paquete, f.patient_id)).resultado
                  for f in filas.itertuples()]
    comprobar("la app analiza todos los cortes", all(r is not None for r in resultados), f"{len(filas)} cortes de "
              f"{N_PACIENTES} pacientes de validación")
    p_app = np.array([r.probabilidad for r in resultados], dtype=np.float32)
    l_app = np.array([r.logit for r in resultados], dtype=np.float32)
    comprobar("logit: app = train.py, bit a bit", np.array_equal(l_app, l_train),
              f"diferencia máxima {np.abs(l_app.astype(float) - l_train).max():.1e}")
    comprobar("probabilidad: app = train.predecir con la misma muestra, bit a bit", np.array_equal(p_app, p_train),
              f"diferencia máxima {np.abs(p_app.astype(float) - p_train).max():.1e}")

    if not hay_salidas(paquete):
        avisar("probabilidad: app frente a predicciones_val.csv del entrenamiento",
               "sin las salidas de la ejecución en esta máquina; se comprueba en la GPU (pasó allí con 1e-5)")
        return filas.assign(prob=p_app, logit=l_app)
    p_csv = pred.prob_corte.to_numpy(np.float64).astype(np.float32)
    diferencia = np.abs(p_app.astype(np.float64) - p_csv)
    iguales = int((p_app == p_csv).sum())
    misma_maquina = (json.loads((RAIZ_REPO / "experimentos" / ejecucion / "config.json")
                                .read_text(encoding="utf-8"))["entorno"]["dispositivo"] == "cpu")
    if misma_maquina:
        ulp = np.spacing(np.maximum(p_app, p_csv)).astype(np.float64)
        bien = bool(np.all(diferencia <= ULP_MAXIMOS * ulp))
        texto = (f"{iguales} de {len(p_csv)} idénticos; el resto difiere en {int(np.max(diferencia / ulp))} "
                 f"unidad(es) del último bit (máx. {diferencia.max():.1e}): sigmoide vectorizada en lotes de 32 "
                 "frente a escalar en un corte")
    else:
        bien = bool(diferencia.max() <= TOLERANCIA_OTRO_DISPOSITIVO)
        texto = f"entrenado en otro dispositivo; diferencia máxima {diferencia.max():.1e}"
    comprobar("probabilidad: app frente a predicciones_val.csv del entrenamiento", bien, texto)
    return filas.assign(prob=p_app, logit=l_app)


def prueba_ejemplos(paquete: PaqueteModelo, modelo_train) -> None:
    print("4. Ejemplos")
    from preparar_ejemplos import EJEMPLOS
    samples = uc.cargar_samples(train.RAIZ_DATOS)
    lista = ejemplos.indice()["ejemplos"]
    comprobar("indice.json con 3 ejemplos y la licencia CC BY-NC 4.0",
              len(lista) == 3 and "CC BY-NC 4.0" in ejemplos.indice()["licencia"])
    for (pid, _), info in zip(EJEMPLOS, lista):
        cortes = samples[(samples.patient_id == pid) & (samples.split == "train")].sort_values("slice_index")
        fila = cortes.iloc[len(cortes) // 2]
        fich = ejemplos.cargar(info["clave"])
        identicos = all(np.array_equal(leer_fase(io.BytesIO(fich[f][1])), leer_fase(train.RAIZ_DATOS / r))
                        for f, r in zip(FASES, (fila.path_pre, fila.path_early, fila.path_late)))
        datos = datos_ejemplo(paquete, info["clave"])
        a, b = analizar(paquete, fich, datos), analizar(paquete, fich, datos)
        p_train, _ = via_train(modelo_train, pd.DataFrame([fila]), paquete)
        comprobar(f"{info['titulo']}: píxeles del dataset, = train.py y reproducible",
                  identicos and a.resultado.probabilidad == b.resultado.probabilidad
                  and np.float32(a.resultado.probabilidad) == p_train[0],
                  f"probabilidad {a.resultado.probabilidad:.6f}")
    texto = json.dumps(ejemplos.indice(), ensure_ascii=False)
    comprobar("los ejemplos no llevan id ni etiqueta",
              not any(s in texto for s in ("ISPY", "ACRIN", "Breast_MRI", "pCR=", '"pCR"')))
    try:
        ejemplos.cargar("../modelo_app")
        comprobar("una clave de ejemplo que no está en el índice se rechaza", False)
    except ValueError:
        comprobar("una clave de ejemplo que no está en el índice se rechaza", True)


def prueba_validacion(paquete: PaqueteModelo) -> None:
    print("5. Validación de la entrada")
    base = ejemplos.cargar("ejemplo_1")
    neutros = {f: (f"fase_{i}.png", base[f][1]) for i, f in enumerate(FASES)}
    datos = datos_ejemplo(paquete, "ejemplo_1")
    referencia = analizar(paquete, base, datos).resultado.probabilidad
    pre = leer_fase(io.BytesIO(base["PRE"][1]))

    def con(fase: str, nombre: str, datos: bytes | None, partida: dict = base) -> dict:
        """Cambia una fase de `partida`. Un nombre "x_<FASE>.png" pasa a ser el del
        ejemplo, para que la comprobación de nombres no interfiera."""
        d = dict(partida)
        if nombre.startswith("x_"):
            nombre = nombre.replace("x_", "ejemplo_1_", 1)
        d[fase] = None if datos is None else (nombre, datos)
        return d

    rgb = np.stack([pre] * 3, axis=-1)
    color = rgb.copy()
    color[..., 0] = np.clip(color[..., 0].astype(int) + 1, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, np.full(pre.shape, 255, np.uint8)])
    rgba[0, 0, 3] = 0
    jpeg = io.BytesIO()
    Image.fromarray(pre).save(jpeg, format="JPEG")
    paleta = Image.fromarray(pre).convert("P")
    paleta.putpalette([v for i in range(256) for v in (i, i, i)])
    paleta.putdata(pre.flatten().tolist())
    salida_paleta = io.BytesIO()
    paleta.save(salida_paleta, format="PNG")
    animado = io.BytesIO()
    Image.fromarray(pre).save(animado, format="PNG", save_all=True, append_images=[Image.fromarray(255 - pre)])
    casi_negra = np.zeros((256, 256), np.uint8)
    casi_negra[:10, :10] = 200                      # 100 píxeles por encima del umbral de tejido

    # (nombre, ficheros, (ámbito, comprobación, estado esperado), debe haber resultado)
    casos = [
        ("fichero vacío", con("PRE", "vacio_PRE.png", b""), ("PRE", "Fichero recibido", ERROR), False),
        ("JPEG con extensión .png", con("PRE", "x_PRE.png", jpeg.getvalue()), ("PRE", "Formato PNG", ERROR), False),
        ("texto con extensión .png", con("PRE", "x_PRE.png", "hola".encode()), ("PRE", "Formato PNG", ERROR), False),
        ("firma PNG y contenido basura", con("EARLY", "x_EARLY.png", b"\x89PNG\r\n\x1a\n" + bytes(100)),
         ("EARLY", "Formato PNG", ERROR), False),
        ("PNG truncado", con("LATE", "x_LATE.png", base["LATE"][1][: len(base["LATE"][1]) // 2]),
         ("LATE", "Decodificación", ERROR), False),
        ("más de 2 MB", con("PRE", "x_PRE.png", base["PRE"][1] + bytes(3 * 2**20)),
         ("PRE", "Tamaño del fichero", ERROR), False),
        ("255 x 256", con("PRE", "x_PRE.png", png(pre[:255])), ("PRE", "Dimensiones", ERROR), False),
        ("PNG animado", con("PRE", "x_PRE.png", animado.getvalue()), ("PRE", "Un solo fotograma", ERROR), False),
        ("16 bits", con("PRE", "x_PRE.png", png(pre.astype(np.uint16) * 257)),
         ("PRE", "Escala de grises de 8 bits", ERROR), False),
        ("1 bit", con("PRE", "x_PRE.png", png((pre > 128))), ("PRE", "Escala de grises de 8 bits", ERROR), False),
        ("RGB en color", con("PRE", "x_PRE.png", png(color)), ("PRE", "Escala de grises de 8 bits", ERROR), False),
        ("RGBA con transparencia", con("PRE", "x_PRE.png", png(rgba)),
         ("PRE", "Escala de grises de 8 bits", ERROR), False),
        ("falta EARLY", con("EARLY", "", None), ("Corte", "Tres fases presentes", ERROR), False),
        ("la misma imagen en PRE y EARLY", con("EARLY", "fase_1.png", base["PRE"][1], neutros),
         ("Corte", "Fases distintas", ERROR), False),
        ("fichero en la casilla de otra fase", con("PRE", "ejemplo_1_EARLY.png", base["PRE"][1]),
         ("Corte", "Nombres de los ficheros", ERROR), False),
        ("ficheros de cortes distintos", {"PRE": ("A_z001_PRE.png", base["PRE"][1]),
                                          "EARLY": ("B_z002_EARLY.png", base["EARLY"][1]),
                                          "LATE": ("A_z001_LATE.png", base["LATE"][1])},
         ("Corte", "Nombres de los ficheros", ERROR), False),
        ("PRE y EARLY intercambiadas", {"PRE": ("a.png", base["EARLY"][1]), "EARLY": ("b.png", base["PRE"][1]),
                                        "LATE": ("c.png", base["LATE"][1])},
         ("Corte", "PRE más oscura que EARLY en el tejido", AVISO), True),
        ("RGB con canales idénticos", con("PRE", "x_PRE.png", png(rgb)),
         ("PRE", "Escala de grises de 8 bits", AVISO), True),
        ("paleta de grises", con("PRE", "x_PRE.png", salida_paleta.getvalue()),
         ("PRE", "Escala de grises de 8 bits", AVISO), True),
        ("PRE casi negra (sin tejido para comprobar el orden)", con("PRE", "x_PRE.png", png(casi_negra)),
         ("Corte", "PRE más oscura que EARLY en el tejido", AVISO), True),
    ]
    for nombre, fich, (ambito, comprobacion, estado), con_resultado in casos:
        try:
            a = analizar(paquete, fich, datos)
        except Exception as e:                                  # no debería ocurrir nunca
            comprobar(nombre, False, f"excepción {type(e).__name__}: {e}")
            continue
        paso = next((p for p in a.informe.pasos if p.ambito == ambito and p.comprobacion == comprobacion
                     and p.estado == estado), None)
        bien = paso is not None and bool(paso.detalle) and ((a.resultado is not None) == con_resultado)
        comprobar(f"{nombre} -> {estado} en «{comprobacion}»", bien, paso.detalle[:90] if paso else "no aparece")

    sin_perdida = [analizar(paquete, con("PRE", "x_PRE.png", d), datos).resultado.probabilidad
                   for d in (png(rgb), salida_paleta.getvalue())]
    comprobar("RGB con canales idénticos y paleta de grises dan la probabilidad del PNG original",
              all(p == referencia for p in sin_perdida))
    informe = Informe()
    comprobar_preprocesado(np.full((3, 256, 256), np.nan, np.float32), informe)
    comprobar("valores no finitos tras el preprocesado -> error",
              [p.estado for p in informe.pasos] == [ERROR])
    meta = con_recorte_caja(paquete.meta)
    sin_caja = analizar(PaqueteModelo(paquete.modelo, meta, paquete.sha256, paquete.dispositivo), base)
    comprobar("un modelo con recorte por la caja sin datos de la paciente -> error claro",
              sin_caja.resultado is None and any(p.comprobacion == "Datos de la paciente" for p in sin_caja.informe.errores))
    meta = copy.deepcopy(paquete.meta)
    meta["preprocesado"].update(recorte_caja=None, recorte_centrado=160)
    sin_caja = analizar(PaqueteModelo(paquete.modelo, meta, paquete.sha256, paquete.dispositivo), base)
    comprobar("un modelo de ventana centrada sin datos de la paciente -> error claro, sin centrar en otro sitio",
              sin_caja.resultado is None and sin_caja.entrada_red is None and any(
                  p.comprobacion == "Datos de la paciente" and "160 × 160" in p.detalle for p in sin_caja.informe.errores))
    meta["preprocesado"]["recorte_caja"] = {"margen": 1.25}
    ambos = analizar(PaqueteModelo(paquete.modelo, meta, paquete.sha256, paquete.dispositivo), base,
                     datos_ejemplo(PaqueteModelo(paquete.modelo, meta, paquete.sha256, paquete.dispositivo), "ejemplo_1"))
    comprobar("modelo.json con los dos recortes a la vez -> error, sin predicción",
              ambos.resultado is None and any(p.comprobacion == "Preprocesado del modelo" for p in ambos.informe.errores))


def app_en_analisis():
    """La app en la página de análisis (la de por defecto es la portada)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(RAIZ_REPO / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    at.switch_page("paginas/analisis.py").run()
    return at


def resultado_de(at, clave: str) -> str | None:
    """El anillo de resultado (paginas/visor.anillo) lleva la probabilidad redondeada
    y la clase como atributos data-probabilidad y data-clase."""
    for m in at.markdown:
        hallado = re.search(rf'data-{clave}="([^"]+)"', m.value)
        if hallado:
            return hallado.group(1)
    return None


def esperada_ejemplo(paquete: PaqueteModelo, clave: str) -> float:
    """Lo que debe mostrar la app para un ejemplo con sus variables: el combinado si
    hay fusion.json; si no, la imagen sola."""
    info = next(e for e in ejemplos.indice()["ejemplos"] if e["clave"] == clave)
    v = info["variables"]
    caja = tuple(v[c] for c in COLUMNAS_CAJA) if recorta(paquete) else None
    prob = analizar(paquete, ejemplos.cargar(clave), None if caja is None else DatosPaciente(*caja)).resultado.probabilidad
    fusion = cargar_fusion()
    return prob if fusion is None else fusion.combinar(prob, v).probabilidad


def exacta_de(at) -> float | None:
    """La probabilidad exacta del anillo («p = …»)."""
    for m in at.markdown:
        hallado = re.search(r'class="exacta">p = ([^<]+)<', m.value)
        if hallado:
            return float(hallado.group(1))
    return None


def prueba_interfaz(filas: pd.DataFrame, paquete: PaqueteModelo) -> None:
    print("6. Interfaz (AppTest)")
    from streamlit.testing.v1 import AppTest

    at = app_en_analisis()
    comprobar("la app arranca sin excepciones", not at.exception, f"{len(at.file_uploader)} casillas de subida")

    def texto_completo() -> str:
        return "\n".join(m.value for m in at.markdown)

    def resultado(clave: str) -> str | None:
        """Probabilidad («probabilidad») o clase («clase») del anillo de resultado."""
        return resultado_de(at, clave)

    # Cinco muestras de cinco pacientes, una tras otra, sin reiniciar.
    una_por_paciente = filas.groupby("patient_id").head(1).head(5)
    for i, fila in enumerate(una_por_paciente.itertuples(), start=1):
        fich = ficheros_de(fila, nombre=f"muestra{i}")
        for subida, fase in zip(at.file_uploader, FASES):
            subida.set_value((fich[fase][0], fich[fase][1], "image/png"))
        if recorta(paquete):            # sin la caja no hay análisis: se sube también su fila de patients.csv
            at.file_uploader[3].set_value((f"muestra{i}.csv", csv_de(patients().loc[fila.patient_id]), "text/csv"))
        at.run()
        esperada = float(np.float32(fila.prob))
        if recorta(paquete):            # la imagen sola va en los atributos del anillo del combinado
            mostrada, clase = resultado("imagen"), resultado("clase")
            exacta = resultado("imagen-exacta") == repr(esperada)
        else:
            mostrada, clase = resultado("probabilidad"), resultado("clase")
            exacta = repr(esperada) in texto_completo()
        comprobar(f"muestra {i}: probabilidad mostrada = train.py", not at.exception and mostrada == num(esperada, 3)
                  and exacta and clase in ("pCR", "no pCR"), f"{mostrada} (exacta {esperada!r} en el resultado)")
        nuevo = next(b for b in at.button if b.label == "Nuevo caso")
        nuevo.click()
        at.run()
        comprobar(f"muestra {i}: «Nuevo caso» vacía las casillas",
                  not at.exception and all(s.value is None for s in at.file_uploader) and resultado("clase") is None)

    # Un fichero que no es PNG en una casilla.
    fich = ficheros_de(una_por_paciente.iloc[0], nombre="error")
    for subida, fase in zip(at.file_uploader, FASES):
        datos = b"esto no es una imagen" if fase == "EARLY" else fich[fase][1]
        subida.set_value((fich[fase][0], datos, "image/png"))
    at.run()
    errores = " ".join(e.value for e in at.error)
    comprobar("un fichero erróneo muestra un error en español, sin excepción y sin resultado",
              not at.exception and "no es un PNG" in errores and resultado("clase") is None, errores[:90])

    # Ejemplos (con sus variables: el modelo combinado, si hay fusion.json).
    at.segmented_control(key="origen").set_value("Ejemplo").run()
    for clave in ("ejemplo_1", "ejemplo_2", "ejemplo_3"):
        at.radio[0].set_value(clave).run()
        esperada = esperada_ejemplo(cargar_paquete(), clave)
        comprobar(f"{clave} desde la interfaz", not at.exception and resultado("probabilidad") == num(esperada, 3),
                  resultado("probabilidad") or "sin resultado")


def prueba_secciones(paquete: PaqueteModelo) -> None:
    print("7. Secciones y ayudas de interpretación")
    from streamlit.testing.v1 import AppTest

    from nucleo.explicacion import curva_cinetica, grad_cam
    from nucleo.informe_caso import informe_html

    # La portada es la página por defecto: titular, cifras leídas de los datos y el botón al análisis.
    from nucleo import test_independiente
    at = AppTest.from_file(str(RAIZ_REPO / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    portada = " ".join(h.value for h in at.html)
    test = test_independiente.cargar()
    cifras = [f"{paquete.meta['modelo']['n_parametros']:,}".replace(",", ".")]       # 98.049
    if test is not None:
        cifras += [num(test["niveles"]["paciente"]["modelos"]["combinado"]["auc"], 2),
                   str(test["niveles"]["paciente"]["n"])]
    boton = next((b for b in at.button if b.label == "Analizar un caso"), None)
    comprobar("la página por defecto es la portada: titular, cifras clave y botón «Analizar un caso»",
              not at.exception
              and 'Predicción de la respuesta patológica completa con <span class="sin-corte">DCE-MRI</span>' in portada
              and all(f'<div class="valor">{c}</div> <div class="rotulo">' in portada for c in cifras)
              and boton is not None and not at.file_uploader,
              " · ".join(cifras))
    if boton is not None:
        boton.click().run()
        comprobar("el botón «Analizar un caso» abre el análisis",
                  not at.exception and len(at.file_uploader) >= 3, f"{len(at.file_uploader)} casillas de subida")

    at = app_en_analisis()
    generico = "Se ha producido un error inesperado"
    for pagina in ("inicio", "modelo", "rendimiento", "datos", "experimentos", "limitaciones"):
        at.switch_page(f"paginas/{pagina}.py").run()
        errores = [e.value for e in at.error if generico in e.value]
        comprobar(f"la página «{pagina}» se carga sin excepciones ni errores", not at.exception and not errores,
                  (at.title[0].value if at.title else "sin título"))
        if pagina == "rendimiento":
            from nucleo import test_independiente
            test = test_independiente.cargar()
            if test is None:
                avisar("no hay experimentos/test/: la página de rendimiento solo muestra la validación cruzada")
                continue
            textos = [h.value for h in at.html] + [m.value for m in at.markdown] + [c.value for c in at.caption]
            todo = " ".join(textos) + " " + " ".join(str(m.value) for m in at.metric)
            i_test = next((i for i, x in enumerate(textos) if "Test independiente (" in x), None)
            i_cv = next((i for i, x in enumerate(textos) if "Validación cruzada en train (selección del modelo)" in x),
                        None)
            auc = num(test["niveles"]["paciente"]["modelos"]["combinado"]["auc"])
            comprobar("rendimiento: primero «Test independiente (176 pacientes, evaluado una sola vez)» y debajo, "
                      "separada, la validación cruzada; AUC del test leído de experimentos/test/",
                      i_test is not None and i_cv is not None and i_test < i_cv
                      and f"({test['niveles']['paciente']['n']} pacientes, evaluado una sola vez)" in textos[i_test]
                      and any(m.value == auc for m in at.metric) and len(at.table) >= 1
                      and "externa" not in todo.lower(), f"AUC combinado en test {auc}")
            comprobar("el test es del paquete que carga la app (SHA-256 de modelo.pt y fusion.json)",
                      test_independiente.es_el_paquete(test, paquete.sha256) and not at.warning)

    base = ejemplos.cargar("ejemplo_1")
    analisis = analizar(paquete, base, datos_ejemplo(paquete, "ejemplo_1"))
    for hacia in (True, False):
        mapa = grad_cam(paquete, analisis.entrada_red, hacia)
        comprobar(f"Grad-CAM hacia {'pCR' if hacia else 'no pCR'}: mismo logit que la inferencia y mapa en [0, 1]",
                  mapa.logit == analisis.resultado.logit and float(mapa.mapa.min()) >= 0
                  and abs(float(mapa.mapa.max()) - 1) < 1e-6, f"logit {mapa.logit:.6f}")
    plano = np.zeros((3, 256, 256), np.float32)
    plano[0], plano[1], plano[2] = 0.2, 0.6, 0.4
    curva = curva_cinetica(plano, 128, 128, 2)
    comprobar("curva cinética sintética PRE 0,2 → EARLY 0,6 → LATE 0,4 = lavado",
              curva.tipo == "lavado" and abs(curva.cambio_tardio + 1 / 3) < 1e-6)
    a = analizar(paquete, base, datos_ejemplo(paquete, "ejemplo_1"))
    a.nombres = {"PRE": "<img src=x onerror=alert(1)>_PRE.png", "EARLY": "e.png", "LATE": "l.png"}
    html = informe_html(a, paquete.meta, paquete.sha256, "prueba", cinetica=curva)
    comprobar("el informe HTML escapa los nombres de los ficheros y lleva el aviso de uso",
              "<img src=x" not in html and "&lt;img src=x" in html and "no es un dispositivo médico" in html)

    # El análisis desde la URL (?ejemplo=) y la pestaña por defecto.
    at = AppTest.from_file(str(RAIZ_REPO / "app" / "streamlit_app.py"), default_timeout=180)
    at.query_params["ejemplo"] = "ejemplo_3"
    at.run()
    esperada = esperada_ejemplo(paquete, "ejemplo_3")
    valor = resultado_de(at, "probabilidad")
    comprobar("?ejemplo=ejemplo_3 abre ese ejemplo en el análisis", not at.exception and valor == num(esperada, 3),
              valor or "sin resultado")


# --------------------------------------------------------------------------- #
# Modelo final combinado
# --------------------------------------------------------------------------- #

def patients() -> pd.DataFrame:
    return pd.read_csv(train.RAIZ_DATOS / "metadata" / "patients.csv").set_index("pid")


def valores_de(fila: pd.Series) -> dict:
    """Las variables clínicas de una fila de patients.csv, como las usa la app (None = sin dato)."""
    return {c: (None if pd.isna(fila[c]) else fila[c]) for c in (*COLUMNAS_ENTRADA, *COLUMNAS_CAJA, "xy_spacing")}


def csv_de(fila: pd.Series, con_pcr: bool = False, cambios: dict | None = None) -> bytes:
    """Un CSV de una fila con las columnas de patients.csv (id incluido; pCR solo si se pide)."""
    d = fila.to_frame().T.infer_objects()
    d.insert(0, "pid", fila.name)
    if not con_pcr:
        d = d.drop(columns="pCR")
    for k, v in (cambios or {}).items():
        d[k] = v
    return d.to_csv(index=False).encode("utf-8")


def prob_referencia(esc, lr, valores: dict, prob_cnn: float) -> float:
    """exportar_fusion.py: clinica.construir_variables + logit de la CNN -> predict_proba."""
    X = construir_variables(pd.DataFrame([{c: valores.get(c) for c in COLUMNAS_ENTRADA}]))
    X["logit_cnn"] = logit(np.float64(prob_cnn))
    return float(lr.predict_proba(esc.transformar(X))[0, 1])


def prueba_fusion(paquete: PaqueteModelo, filas: pd.DataFrame):
    print("8. Modelo final combinado (fusion.json) frente a exportar_fusion.py")
    fusion = cargar_fusion()
    if fusion is None:
        comprobar("modelo_app/fusion.json existe (python exportar_fusion.py <configuración>)", False)
        return None
    texto = (RAIZ_REPO / "modelo_app" / "fusion.json").read_text(encoding="utf-8")
    comprobar("fusion.json se carga y no lleva ids ni etiquetas individuales",
              not exportar_fusion.PATRON_IDS.search(texto) and '"pCR":' not in texto,
              f"CNN {fusion.configuracion}{' (provisional)' if fusion.provisional else ''}")
    fuente_oof = ((exportar_fusion.OOF if fusion.provisional else exportar_fusion.OOF_CORTES)
                  / f"{fusion.configuracion}.csv")
    if not fuente_oof.is_file():
        # experimentos/oof*/ no se versiona (lleva ids de pacientes): sin él no se puede
        # reajustar la regresión. La referencia es entonces la regresión de fusion.json
        # evaluada por sklearn (predict_proba), otro camino de cálculo que el de la app.
        avisar("coeficientes y umbral de fusion.json frente a exportar_fusion.py",
               f"falta {fuente_oof.relative_to(RAIZ_REPO).as_posix()}: ejecuta exportar_oof.py en la máquina con los "
               "pesos; la referencia es la regresión de fusion.json vía sklearn")
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.classes_ = np.array([0, 1])
        lr.coef_ = fusion.coef.reshape(1, -1)
        lr.intercept_ = np.array([fusion.intercepto])
        lr.n_features_in_ = len(fusion.variables)
        lr.feature_names_in_ = np.array(fusion.variables, dtype=object)
        if not fusion.coincide_cnn(paquete.meta["ejecucion"]):
            comprobar("la CNN del paquete es de la configuración de la fusión", False,
                      f"paquete {paquete.meta['ejecucion']}, fusión {fusion.configuracion}")
        return _comparar_fusion(fusion, filas, fusion.escalador, lr)
    datos, ruta = exportar_fusion.cargar_filas(fusion.configuracion, fusion.provisional)
    X = exportar_fusion.matriz(datos, "combinado")
    esc, lr = exportar_fusion.ajustar(datos, X)
    # Bit a bit en la máquina que exportó; entre máquinas, la aritmética de cada una
    # cambia el último bit (~1e-16). El SHA-256 de la fuente, con los finales de línea
    # normalizados (git los convierte a CRLF en Windows).
    def iguales(a, b) -> bool:
        return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=TOLERANCIA_FUSION))
    fuente = ruta.read_bytes().replace(b"\r\n", b"\n")
    comprobar("coeficientes, intercepto y escalador = la regresión que reajusta exportar_fusion.py",
              iguales(fusion.coef, lr.coef_[0]) and iguales(fusion.intercepto, lr.intercept_[0])
              and iguales(fusion.escalador.media[esc.media.index], esc.media)
              and iguales(fusion.escalador.desv[esc.desv.index], esc.desv)
              and hashlib.sha256(fuente).hexdigest() == fusion.meta["cnn"]["sha256_fuente"],
              f"{len(datos)} cortes de {datos.patient_id.nunique()} pacientes; fuente {ruta.name}; diferencia "
              f"máxima de coeficientes {np.abs(fusion.coef - lr.coef_[0]).max():.0e}")
    oof = exportar_fusion.validacion_cruzada(datos, X)
    elegido = confusion(datos.pCR.to_numpy().astype(int), oof)
    comprobar("umbral = el de mayor sensibilidad con especificidad >= 0,85 por corte (validación cruzada)",
              elegido["umbral"] == fusion.umbral and elegido["especificidad"] >= 0.85,
              f"umbral {fusion.umbral:.4f}, sensibilidad {elegido['sensibilidad']:.3f}")
    if fusion.provisional:
        avisar("fusion.json es PROVISIONAL (probabilidad por paciente en cada corte)",
               "en la GPU: exportar_oof.py y exportar_fusion.py sin --provisional")
    if not fusion.coincide_cnn(paquete.meta["ejecucion"]):
        (avisar if fusion.provisional else lambda n, d: comprobar(n, False, d))(
            "la CNN del paquete es de la configuración de la fusión",
            f"paquete {paquete.meta['ejecucion']}, fusión {fusion.configuracion}")
    return _comparar_fusion(fusion, filas, esc, lr)


def _comparar_fusion(fusion: ModeloFusion, filas: pd.DataFrame, esc, lr):
    """Probabilidad combinada de la app (nucleo.fusion) frente a la referencia (esc, lr)."""
    pac = patients()
    dif_max, clases_iguales, csv_iguales = 0.0, True, True
    for i, f in enumerate(filas.itertuples()):
        valores = valores_de(pac.loc[f.patient_id])
        if i % 9 == 0:
            valores["age"] = None                         # valores que faltan: media de entrenamiento
        if i % 13 == 0:
            valores["HR_HER2_STATUS"] = None
        app = fusion.combinar(f.prob, valores)
        ref = prob_referencia(esc, lr, valores, f.prob)
        dif_max = max(dif_max, abs(app.probabilidad - ref))
        clases_iguales &= app.clase == int(ref >= fusion.umbral) or abs(ref - fusion.umbral) < TOLERANCIA_FUSION
        if i % 9 and i % 13:                              # por el CSV, con la fila completa (y la pCR)
            leidas = var.desde_csv("fila.csv", csv_de(pac.loc[f.patient_id], con_pcr=True), False,
                                   fusion.meta.get("rango_entrenamiento"))
            csv_iguales &= leidas.valido and fusion.combinar(f.prob, leidas.valores).probabilidad == app.probabilidad
    comprobar("probabilidad combinada: app = exportar_fusion.py con la misma muestra y variables",
              dif_max <= TOLERANCIA_FUSION and clases_iguales,
              f"{len(filas)} cortes (con faltantes en algunos); diferencia máxima {dif_max:.1e}")
    comprobar("las variables leídas de un CSV de patients.csv dan la misma probabilidad", csv_iguales)
    suma = all(abs(fusion.combinar(f.prob, valores_de(pac.loc[f.patient_id])).intercepto
                   + sum(fusion.combinar(f.prob, valores_de(pac.loc[f.patient_id])).contribuciones.values())
                   - fusion.combinar(f.prob, valores_de(pac.loc[f.patient_id])).logit) < 1e-12
               for f in filas.head(20).itertuples())
    comprobar("las contribuciones mostradas suman el logit (intercepto + aportes)", suma)
    return fusion, esc, lr


def prueba_variables(referencia) -> None:
    print("9. Variables de la paciente mal formadas o dudosas")
    rango = referencia[0].meta.get("rango_entrenamiento") if referencia else None
    pac = patients()
    base = pac[(pac.split == "train") & pac[["HR_HER2_STATUS", "tum_vol", "age"]].notna().all(axis=1)].iloc[0]

    def con(**cambios) -> bytes:
        return csv_de(base, cambios=cambios)

    sin_age = base.drop("age").to_frame().T.drop(columns="pCR").to_csv(index=False).encode()
    dos_filas = pd.concat([base.to_frame().T] * 2).drop(columns="pCR").to_csv(index=False).encode()
    punto_coma = (base.to_frame().T.infer_objects().drop(columns="pCR").assign(nota="señal")
                  .to_csv(index=False, sep=";", decimal=",").encode("cp1252"))
    # (nombre, CSV, necesita caja, (comprobación, estado esperado), ¿válidas?)
    casos = [
        ("CSV vacío", b"", False, ("Fichero de variables", ERROR), False),
        ("fichero binario", bytes(range(256)) * 4, False, ("Fichero de variables", ERROR), False),
        ("dos filas", dos_filas, False, ("Fichero de variables", ERROR), False),
        ("falta la columna age", sin_age, False, ("Columnas", ERROR), False),
        ("subtipo desconocido", con(HR_HER2_STATUS="Luminal A"), False, ("Subtipo HR/HER2", ERROR), False),
        ("edad no numérica", con(age="abc"), False, ("Edad", ERROR), False),
        ("edad imposible", con(age=150), False, ("Edad", ERROR), False),
        ("volumen negativo", con(tum_vol=-3), False, ("Volumen tumoral", ERROR), False),
        ("las tres clínicas vacías", con(HR_HER2_STATUS="", tum_vol="", age=""), False,
         ("Variables clínicas", ERROR), False),
        ("caja invertida", con(sraw=200, eraw=100), True, ("Caja del tumor", ERROR), False),
        ("caja fuera del corte", con(ecol=999), True, ("Caja del tumor", ERROR), False),
        ("falta xy_spacing con recorte", con(xy_spacing=""), True, ("Caja del tumor", ERROR), False),
        ("n_xy no entero", con(n_xy=100.5), True, ("Caja del tumor", ERROR), False),
        ("falta la columna sraw con recorte", base.drop("sraw").to_frame().T.drop(columns="pCR")
         .to_csv(index=False).encode(), True, ("Columnas", ERROR), False),
        ("volumen vacío (se rellena)", con(tum_vol=""), False, ("Volumen tumoral", AVISO), True),
        ("subtipo vacío (se rellena)", con(HR_HER2_STATUS=""), False, ("Subtipo HR/HER2", AVISO), True),
        ("edad fuera de lo habitual", con(age=90), False, ("Edad", AVISO), True),
        ("HR/HER2 que no cuadran con el subtipo", con(HR_HER2_STATUS="TripleNeg", HR=1, HER2=0), False, ("HR y HER2", AVISO), True),
        ("con la pCR (se ignora)", csv_de(base, con_pcr=True), False, ("Columna pCR", AVISO), True),
        ("punto y coma, coma decimal y Latin-1", punto_coma, True, ("Caja del tumor", OK), True),
    ]
    referencia_valores = var.desde_csv("base.csv", csv_de(base), True, rango).valores
    for nombre, datos, caja, (comprobacion, estado), validas in casos:
        try:
            v = var.desde_csv("variables.csv", datos, caja, rango)
        except Exception as e:                                  # no debería ocurrir nunca
            comprobar(nombre, False, f"excepción {type(e).__name__}: {e}")
            continue
        paso = next((p for p in v.informe.pasos if p.comprobacion == comprobacion and p.estado == estado), None)
        bien = paso is not None and bool(paso.detalle) and v.valido == validas
        comprobar(f"{nombre} -> {estado} en «{comprobacion}»", bien, paso.detalle[:90] if paso else "no aparece")
        if nombre.startswith("punto y coma"):
            comprobar("el CSV con punto y coma y coma decimal da los mismos valores", v.valores == referencia_valores)


def prueba_recorte(paquete: PaqueteModelo, filas: pd.DataFrame, modelo_train) -> None:
    print("10. Recorte por la caja: app = train.py")
    recorta = PaqueteModelo(paquete.modelo, con_recorte_caja(paquete.meta), paquete.sha256, paquete.dispositivo)
    pac = patients()
    elegidas = filas.groupby("patient_id").head(2).head(12).reset_index(drop=True)
    cajas = {pid: caja_en_png(pac.loc[pid]) for pid in elegidas.patient_id.unique()}
    _, logits_train = via_train(modelo_train, elegidas, recorta)      # DatasetCortes con las cajas, en eval()
    iguales_entrada, logits_app, csv_iguales = True, [], True
    for f in elegidas.itertuples():
        v = var.desde_csv("fila.csv", csv_de(pac.loc[f.patient_id]), True)
        a = analizar(recorta, ficheros_de(f), v.datos_paciente())
        esperada = preprocesar(*(train.RAIZ_DATOS / r for r in (f.path_pre, f.path_early, f.path_late)),
                               caja=cajas[f.patient_id])
        iguales_entrada &= a.entrada_red is not None and np.array_equal(a.entrada_red, esperada)
        logits_app.append(a.resultado.logit)
        b = analizar(recorta, ficheros_de(f), DatosPaciente(*(pac.loc[f.patient_id][c] for c in COLUMNAS_CAJA)))
        csv_iguales &= b.resultado.logit == a.resultado.logit
    comprobar("entrada de la red con la caja de las variables = preprocesar(..., caja=caja_en_png)", iguales_entrada,
              f"{len(elegidas)} cortes, entrada {tuple(a.entrada_red.shape)}")
    comprobar("logit con recorte: app = train.py (DatasetCortes con cajas), bit a bit",
              np.array_equal(np.array(logits_app, dtype=np.float32), logits_train))
    comprobar("la caja leída del CSV = la de patients.csv", csv_iguales)


def prueba_defensa(paquete: PaqueteModelo, filas: pd.DataFrame, referencia) -> None:
    print("11. Defensa simulada (AppTest): cinco muestras con sus variables")
    if referencia is None:
        comprobar("hay modelo combinado para la defensa simulada", False)
        return
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    fusion, esc, lr = referencia
    pac = patients()
    at = app_en_analisis()

    def tipo() -> str | None:
        return resultado_de(at, "tipo")

    una_por_paciente = filas.groupby("patient_id").head(1).iloc[5:10]
    for i, f in enumerate(una_por_paciente.itertuples(), start=1):
        fila = pac.loc[f.patient_id]
        fich = ficheros_de(f, nombre=f"defensa{i}")
        subidas = at.file_uploader
        for subida, fase in zip(subidas[:3], FASES):
            subida.set_value((fich[fase][0], fich[fase][1], "image/png"))
        subidas[3].set_value((f"paciente{i}.csv", csv_de(fila, con_pcr=(i == 1)), "text/csv"))
        at.run()
        esperada = prob_referencia(esc, lr, valores_de(fila), f.prob)
        exacta = exacta_de(at)
        bien = (not at.exception and tipo() == "combinado" and resultado_de(at, "probabilidad") == num(esperada, 3)
                and exacta is not None and abs(exacta - esperada) <= TOLERANCIA_FUSION
                and resultado_de(at, "imagen") == num(float(np.float32(f.prob)), 3))
        comprobar(f"defensa {i}: probabilidad combinada mostrada = exportar_fusion.py", bien,
                  f"{resultado_de(at, 'probabilidad')} (imagen {resultado_de(at, 'imagen')}, clase "
                  f"{resultado_de(at, 'clase')})")
        next(b for b in at.button if b.label == "Nuevo caso").click()
        at.run()
        comprobar(f"defensa {i}: «Nuevo caso» vacía imágenes y variables",
                  not at.exception and all(s.value is None for s in at.file_uploader) and tipo() is None)

    # Variables con errores: error claro y solo la imagen.
    f = una_por_paciente.iloc[0]
    fich = ficheros_de(f, nombre="errores")
    for subida, fase in zip(at.file_uploader[:3], FASES):
        subida.set_value((fich[fase][0], fich[fase][1], "image/png"))
    at.file_uploader[3].set_value(("mal.csv", csv_de(pac.loc[f.patient_id], cambios={"age": "abc"}), "text/csv"))
    at.run()
    errores = " ".join(e.value for e in at.error)
    comprobar("variables con errores" + (" (caja válida)" if recorta(paquete) else "")
              + ": error en español y solo la probabilidad de la imagen",
              not at.exception and "Edad" in errores and "no es un número" in errores and tipo() == "imagen", errores[:90])
    if recorta(paquete):
        # Caja válida y sin variables clínicas: la imagen sola, con el aviso de que no es el modelo final.
        at.file_uploader[3].set_value(("sin_clinicas.csv", csv_de(pac.loc[f.patient_id], cambios={
            "HR_HER2_STATUS": "", "tum_vol": "", "age": ""}), "text/csv"))
        at.run()
        comprobar("caja válida sin variables clínicas: solo la imagen, con el aviso de que no es el modelo final",
                  not at.exception and tipo() == "imagen" and any("Solo imagen" in m.value for m in at.markdown),
                  " ".join(e.value for e in at.error)[:80])
        # Caja mal escrita: esta CNN no puede mirar la imagen; no predice y lo explica.
        at.file_uploader[3].set_value(("caja_mal.csv", csv_de(pac.loc[f.patient_id], cambios={"sraw": "abc"}),
                                       "text/csv"))
        at.run()
        errores = " ".join(e.value for e in at.error)
        comprobar("caja mal escrita: no predice y lo explica en español",
                  not at.exception and tipo() is None and "No se puede analizar" in errores and "caja" in errores,
                  errores[:90])
        at.file_uploader[3].set_value(("mal.csv", csv_de(pac.loc[f.patient_id], cambios={"age": "abc"}), "text/csv"))
        at.run()

    # Formulario, con la misma muestra.
    at.segmented_control(key="fuente_variables").set_value("Formulario").run()
    caso = at.session_state["caso"]
    fila = pac.loc[f.patient_id]
    at.selectbox(key=f"f_subtipo_subida_{caso}").set_value(fila.HR_HER2_STATUS)
    at.number_input(key=f"f_tum_vol_subida_{caso}").set_value(float(fila.tum_vol))
    at.number_input(key=f"f_age_subida_{caso}").set_value(float(fila.age))
    if recorta(paquete):                        # esta CNN necesita también la caja del tumor
        for c in (*COLUMNAS_CAJA, "xy_spacing"):
            at.number_input(key=f"f_{c}_subida_{caso}").set_value(float(fila[c]))
    next(b for b in at.button if b.label == "Aplicar").click()
    at.run()
    esperada = prob_referencia(esc, lr, valores_de(fila), f.prob)
    comprobar("formulario: la misma probabilidad combinada que el CSV", not at.exception and tipo() == "combinado"
              and abs((exacta_de(at) or -1) - esperada) <= TOLERANCIA_FUSION, resultado_de(at, "probabilidad") or "—")
    at.segmented_control(key="fuente_variables").set_value("Ninguna").run()
    if recorta(paquete):
        errores = " ".join(e.value for e in at.error)
        comprobar("sin variables: esta CNN necesita la caja; no predice y lo explica en español",
                  not at.exception and tipo() is None and "No se puede analizar" in errores and "caja" in errores,
                  errores[:90])
    else:
        comprobar("sin variables: solo la imagen, con el aviso de que no es el modelo final",
                  not at.exception and tipo() == "imagen" and any("Solo imagen" in m.value for m in at.markdown))

    # Un paquete que recorta por la caja (copia de modelo_app/ con recorte_caja en modelo.json).
    from nucleo import fusion as nucleo_fusion, paquete as nucleo_paquete
    with tempfile.TemporaryDirectory() as tmp:
        copia = Path(tmp) / "modelo_app"
        shutil.copytree(RAIZ_REPO / "modelo_app", copia)
        meta = con_recorte_caja(json.loads((copia / "modelo.json").read_text(encoding="utf-8")))
        (copia / "modelo.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        con_caja = PaqueteModelo(paquete.modelo, meta, paquete.sha256, paquete.dispositivo)
        previos = nucleo_paquete.cargar_paquete.__defaults__, nucleo_fusion.cargar_fusion.__defaults__
        nucleo_paquete.cargar_paquete.__defaults__ = (copia, "cpu")
        nucleo_fusion.cargar_fusion.__defaults__ = (copia,)
        st.cache_resource.clear()
        st.cache_data.clear()
        try:
            at = app_en_analisis()
            for i, f in enumerate(una_por_paciente.head(3).itertuples(), start=1):
                fila = pac.loc[f.patient_id]
                fich = ficheros_de(f, nombre=f"recorte{i}")
                for subida, fase in zip(at.file_uploader[:3], FASES):
                    subida.set_value((fich[fase][0], fich[fase][1], "image/png"))
                at.run()
                errores = " ".join(e.value for e in at.error)
                comprobar(f"con recorte {i}: sin variables, error claro (necesita la caja)",
                          not at.exception and "caja del tumor" in errores and tipo() is None, errores[:80])
                at.file_uploader[3].set_value((f"p{i}.csv", csv_de(fila), "text/csv"))
                at.run()
                cnn = analizar(con_caja, fich, DatosPaciente(*(fila[c] for c in COLUMNAS_CAJA))).resultado.probabilidad
                esperada = prob_referencia(esc, lr, valores_de(fila), cnn)
                comprobar(f"con recorte {i}: con sus variables, el combinado esperado",
                          not at.exception and tipo() == "combinado"
                          and abs((exacta_de(at) or -1) - esperada) <= TOLERANCIA_FUSION,
                          resultado_de(at, "probabilidad") or "—")
                next(b for b in at.button if b.label == "Nuevo caso").click()
                at.run()
        finally:
            nucleo_paquete.cargar_paquete.__defaults__, nucleo_fusion.cargar_fusion.__defaults__ = previos
            st.cache_resource.clear()
            st.cache_data.clear()


def prueba_ventana(paquete: PaqueteModelo, filas: pd.DataFrame, modelo_train) -> None:
    print("12. Ventana centrada (--recorte-centrado): app = train.py")
    lado = paquete.meta["preprocesado"].get("recorte_centrado")
    if lado:
        centrado = paquete
        print(f"  (paquete exportado: ventana de {lado} px)")
    else:                                       # la CNN acaba en GAP: los mismos pesos admiten 160 x 160
        lado = 160
        meta = copy.deepcopy(paquete.meta)
        meta["preprocesado"].update(recorte_caja=None, recorte_centrado=lado)
        meta["modelo"]["entrada"] = [3, lado, lado]
        centrado = PaqueteModelo(paquete.modelo, meta, paquete.sha256, paquete.dispositivo)
        print(f"  (paquete simulado: los pesos de {paquete.meta['ejecucion']} con una ventana de {lado} px)")
    pac = patients()
    elegidas = filas.groupby("patient_id").head(2).head(12).reset_index(drop=True)
    _, logits_train = via_train(modelo_train, elegidas, centrado)   # DatasetCortes con cajas y lado_centrado
    iguales_entrada, ventanas, logits_app, dentro = True, True, [], 0
    for f in elegidas.itertuples():
        v = var.desde_csv("fila.csv", csv_de(pac.loc[f.patient_id]), True)
        a = analizar(centrado, ficheros_de(f), v.datos_paciente())
        caja = caja_en_png(pac.loc[f.patient_id])
        esperada = preprocesar(*(train.RAIZ_DATOS / r for r in (f.path_pre, f.path_early, f.path_late)),
                               caja=caja, lado_centrado=lado)
        iguales_entrada &= (a.entrada_red is not None and a.entrada_red.shape == (3, lado, lado)
                            and np.array_equal(a.entrada_red, esperada))
        r0, k0 = ventana_centrada(caja, lado)
        ventanas &= a.ventana == (r0, k0, lado)
        dentro += bool(r0 <= caja[0] and caja[1] <= r0 + lado and k0 <= caja[2] and caja[3] <= k0 + lado)
        logits_app.append(a.resultado.logit)
    comprobar(f"entrada de la red = preprocesar(..., caja=caja, lado_centrado={lado}), {lado} x {lado}", iguales_entrada,
              f"{len(elegidas)} cortes de {elegidas.patient_id.nunique()} pacientes; la caja cabe entera en la "
              f"ventana en {dentro}")
    comprobar("ventana del visor = preprocesado.ventana_centrada", ventanas)
    comprobar("logit con ventana centrada: app = train.py (DatasetCortes con cajas y lado_centrado), bit a bit",
              np.array_equal(np.array(logits_app, dtype=np.float32), logits_train),
              f"diferencia máxima {np.abs(np.array(logits_app, dtype=np.float64) - logits_train).max():.1e}")
    sin = analizar(centrado, ficheros_de(elegidas.iloc[0]))
    comprobar("sin la caja: error claro y ninguna predicción", sin.resultado is None and sin.entrada_red is None
              and any(p.comprobacion == "Datos de la paciente" for p in sin.informe.errores))

    # Interfaz con ese paquete (copia de modelo_app/ con su modelo.json): sin variables,
    # error; con su CSV, resultado y el pie de la ventana bajo el visor.
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from nucleo import fusion as nucleo_fusion, paquete as nucleo_paquete
    with tempfile.TemporaryDirectory() as tmp:
        copia = Path(tmp) / "modelo_app"
        shutil.copytree(RAIZ_REPO / "modelo_app", copia)
        (copia / "modelo.json").write_text(json.dumps(centrado.meta, ensure_ascii=False), encoding="utf-8")
        previos = nucleo_paquete.cargar_paquete.__defaults__, nucleo_fusion.cargar_fusion.__defaults__
        nucleo_paquete.cargar_paquete.__defaults__ = (copia, "cpu")
        nucleo_fusion.cargar_fusion.__defaults__ = (copia,)
        st.cache_resource.clear()
        st.cache_data.clear()
        try:
            at = app_en_analisis()
            f = elegidas.iloc[0]
            fich = ficheros_de(f, nombre="ventana")
            for subida, fase in zip(at.file_uploader[:3], FASES):
                subida.set_value((fich[fase][0], fich[fase][1], "image/png"))
            at.run()
            errores = " ".join(e.value for e in at.error)
            comprobar("interfaz, sin variables: error claro (la ventana necesita la caja)",
                      not at.exception and f"{lado} × {lado}" in errores and resultado_de(at, "tipo") is None,
                      errores[:80])
            at.file_uploader[3].set_value(("ventana.csv", csv_de(pac.loc[f.patient_id]), "text/csv"))
            at.run()
            esperada = analizar(centrado, fich, var.desde_csv("v.csv", csv_de(pac.loc[f.patient_id]), True)
                                .datos_paciente()).resultado.probabilidad
            pie = " ".join(c.value for c in at.caption)
            comprobar("interfaz, con su CSV: resultado y pie de la ventana que recibe la red bajo el visor",
                      not at.exception and resultado_de(at, "tipo") is not None
                      and resultado_de(at, "imagen" if resultado_de(at, "tipo") == "combinado" else "probabilidad")
                      == num(float(np.float32(esperada)), 3) and f"ventana de {lado} × {lado} px que recibe la red" in pie,
                      f"imagen {num(esperada, 3)}")
        finally:
            nucleo_paquete.cargar_paquete.__defaults__, nucleo_fusion.cargar_fusion.__defaults__ = previos
            st.cache_resource.clear()
            st.cache_data.clear()


def referencias_rotas() -> list[str]:
    """Nombres que las páginas usan de nucleo/ y paginas/ y que no existen (p. ej., una función
    renombrada en un módulo pero no en la página que la llama)."""
    import ast
    import importlib
    rotas = []
    for fichero in sorted((RAIZ_REPO / "app" / "paginas").glob("*.py")) + [RAIZ_REPO / "app" / "streamlit_app.py"]:
        arbol = ast.parse(fichero.read_text(encoding="utf-8"))
        alias = {}
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.module.split(".")[0] in ("nucleo", "paginas"):
                for n in nodo.names:
                    try:
                        alias[n.asname or n.name] = importlib.import_module(f"{nodo.module}.{n.name}")
                    except ModuleNotFoundError:
                        if not hasattr(importlib.import_module(nodo.module), n.name):
                            rotas.append(f"{fichero.name}: {nodo.module}.{n.name}")
        for nodo in ast.walk(arbol):
            if (isinstance(nodo, ast.Attribute) and isinstance(nodo.value, ast.Name) and nodo.value.id in alias
                    and not hasattr(alias[nodo.value.id], nodo.attr)):
                rotas.append(f"{fichero.name}: {nodo.value.id}.{nodo.attr}")
    return rotas


def prueba_estilo() -> None:
    print("13. Estilo: barra lateral y selectores de ficheros; referencias de las páginas")
    rotas = referencias_rotas()
    comprobar("las páginas solo usan nombres que existen en nucleo/ y paginas/", not rotas, "; ".join(rotas[:3]))
    app_py = (RAIZ_REPO / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    comprobar("al arrancar, la app vuelve a importar los módulos del proyecto que cambiaron en disco",
              "_modulos_al_dia()" in app_py and app_py.index("_modulos_al_dia()\n") < app_py.index("from paginas import comun"))
    from nucleo.validacion import TAMANO_MAXIMO_BYTES
    from nucleo.variables import TAMANO_MAXIMO_CSV
    css = (RAIZ_REPO / "app" / "estaticos" / "estilo.css").read_text(encoding="utf-8")
    sin_comentarios = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    reglas = [(s.strip(), c) for s, c in re.findall(r"([^{}]+)\{([^{}]*)\}", sin_comentarios)]

    def cuerpo(selector_contiene: str, *otros: str) -> str:
        return " ".join(c for s, c in reglas if selector_contiene in s and all(o in s for o in otros))

    app = (RAIZ_REPO / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    comprobar("la barra lateral empieza abierta (initial_sidebar_state=\"expanded\")",
              'initial_sidebar_state="expanded"' in app)
    ocultan_toolbar = [s for s, c in reglas if re.search(r"display:\s*none", c)
                       and any(p.strip() == '[data-testid="stToolbar"]' for p in s.split(","))]
    comprobar("no se oculta stToolbar (contiene el botón para desplegar la barra)", not ocultan_toolbar,
              "; ".join(ocultan_toolbar)[:80])
    comprobar("botones de plegar y desplegar la barra lateral visibles siempre",
              "visibility: visible" in cuerpo('[data-testid="stExpandSidebarButton"]')
              and "display: flex" in cuerpo('[data-testid="stExpandSidebarButton"]')
              and "visibility: visible" in cuerpo('[data-testid="stSidebarCollapseButton"]'))
    peligrosos = re.findall(r"<[/\w]", css)
    comprobar("estilo.css sin «<» seguido de letra (st.html descartaría la hoja entera)", not peligrosos,
              ", ".join(peligrosos))
    analisis = (RAIZ_REPO / "app" / "paginas" / "analisis.py").read_text(encoding="utf-8")
    csv_kb = f"máx. {TAMANO_MAXIMO_CSV / 1024:.0f} KB"
    png_mb = f"máx. {TAMANO_MAXIMO_BYTES / 2**20:.0f} MB"
    comprobar("selector de las variables: «Subir CSV» y «una fila · " + csv_kb + "» (nucleo/variables.py)",
              'key=f"subida_csv_{caso}"' in analisis
              and '"Subir CSV"' in cuerpo('[class*="st-key-subida_csv_"]', "stMarkdownContainer")
              and f'"una fila · {csv_kb}"' in cuerpo('[class*="st-key-subida_csv_"]', "DropzoneInstructions"))
    generales = [c for s, c in reglas if "st-key-" not in s]
    comprobar("selectores de las fases: «Subir PNG» y «256 × 256 · " + png_mb + "» (nucleo/validacion.py)",
              any('"Subir PNG"' in c for c in generales) and any(f'"256 × 256 · {png_mb}"' in c for c in generales))


def prueba_uso_sin_ayuda(paquete: PaqueteModelo, filas: pd.DataFrame) -> None:
    print("14. Uso sin ayuda: plantilla CSV, errores con solución y registro de la sesión")
    from nucleo.validacion import SOLUCION
    fusion = cargar_fusion()
    rango = fusion.meta.get("rango_entrenamiento") if fusion else None
    plantilla = var.plantilla_csv(recorta(paquete))
    v = var.desde_csv("plantilla_variables.csv", plantilla, recorta(paquete), rango)
    comprobar("la plantilla CSV se acepta tal cual, sin errores ni avisos",
              v.valido and not v.informe.avisos and "pid" not in plantilla.decode() and "pCR" not in plantilla.decode(),
              plantilla.decode().splitlines()[0])
    f = filas.iloc[0]
    a = analizar(paquete, ficheros_de(f), v.datos_paciente() if recorta(paquete) else None)
    comprobar("la plantilla con un corte da el resultado (combinado si hay fusion.json)",
              a.resultado is not None and (fusion is None or fusion.combinar(a.resultado.probabilidad, v.valores)
                                           .probabilidad is not None))

    # Todo error dice qué hacer: entradas erróneas de imágenes y de variables.
    base = ejemplos.cargar("ejemplo_1")
    malos = [dict(base, PRE=None), dict(base, PRE=("x_PRE.png", b"")), dict(base, EARLY=("x_EARLY.png", b"no es png")),
             dict(base, LATE=base["PRE"])]
    pasos = [p for fich in malos for p in analizar(paquete, fich, datos_ejemplo(paquete, "ejemplo_1")).informe.errores]
    fila = patients().loc[f.patient_id]
    for cambios in ({"age": "abc"}, {"HR_HER2_STATUS": "Luminal"}, {"sraw": "abc"}, {"eraw": "1"}, {"tum_vol": "-3"}):
        pasos += var.desde_csv("v.csv", csv_de(fila, cambios=cambios), True, rango).informe.errores
    pasos += var.desde_csv("v.csv", b"", True, rango).informe.errores
    sin = [f"{p.comprobacion}: {p.detalle[:50]}" for p in pasos
           if "Qué hacer:" not in p.detalle and p.comprobacion != "Datos de la paciente"]
    comprobar("todo error de validación dice qué hacer para corregirlo", bool(pasos) and not sin,
              f"{len(pasos)} errores de {len({p.comprobacion for p in pasos})} tipos; sin solución: {sin[:2]}")
    comprobar("cada solución de SOLUCION es una indicación completa (termina en punto)",
              all(isinstance(k, str) and v_.endswith(".") for k, v_ in SOLUCION.items()))

    # Cinco casos seguidos en el registro de la sesión, con la probabilidad que se muestra.
    at = app_en_analisis()
    pac = patients()
    mostradas = []
    for i, f in enumerate(filas.groupby("patient_id").head(1).iloc[10:15].itertuples(), start=1):
        fich = ficheros_de(f, nombre=f"registro{i}")
        for subida, fase in zip(at.file_uploader[:3], FASES):
            subida.set_value((fich[fase][0], fich[fase][1], "image/png"))
        at.file_uploader[3].set_value((f"r{i}.csv", csv_de(pac.loc[f.patient_id]), "text/csv"))
        at.run()
        mostradas.append((resultado_de(at, "probabilidad"), resultado_de(at, "clase")))
        next(b for b in at.button if b.label == "Nuevo caso").click()
        at.run()
    registro = at.session_state["registro_sesion"] if "registro_sesion" in at.session_state else []
    en_registro = [(r["Probabilidad"], "pCR" if r["Clase"] == "pCR" else "no pCR") for r in registro]
    comprobar("cinco casos seguidos quedan en el registro con la probabilidad y la clase mostradas",
              not at.exception and len(registro) == 5 and en_registro == mostradas
              and all("/" not in r["Ficheros"] and "\\" not in r["Ficheros"] for r in registro),
              " · ".join(p for p, _ in en_registro))
    comprobar("el registro no lleva etiquetas reales",
              all(set(r) == {"Hora", "Ficheros", "Modelo", "Probabilidad", "Umbral", "Clase"} for r in registro))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)
    paquete = prueba_paquete()
    filas = prueba_equivalencia(paquete)
    modelo_train = modelo_referencia(paquete)
    prueba_ejemplos(paquete, modelo_train)
    prueba_validacion(paquete)
    prueba_interfaz(filas, paquete)
    prueba_secciones(paquete)
    referencia = prueba_fusion(paquete, filas)
    prueba_variables(referencia)
    prueba_recorte(paquete, filas, modelo_train)
    prueba_defensa(paquete, filas, referencia)
    prueba_ventana(paquete, filas, modelo_train)
    prueba_estilo()
    prueba_uso_sin_ayuda(paquete, filas)
    print()
    if avisos:
        print(f"{len(avisos)} aviso(s): " + "; ".join(avisos))
    if fallos:
        print(f"FALLAN {len(fallos)} comprobaciones: " + "; ".join(fallos))
        sys.exit(1)
    print("Todas las comprobaciones OK.")


if __name__ == "__main__":
    main()
