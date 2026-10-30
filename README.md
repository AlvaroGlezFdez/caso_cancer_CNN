# Predicción de pCR con una CNN desde cero (BreastDCEDL)

Autor: Álvaro González Fernández.

Estimación de la respuesta patológica completa (pCR) al tratamiento neoadyuvante del cáncer de mama a partir
de un corte de resonancia magnética DCE (fases PRE, EARLY y LATE). Una red convolucional 2D entrenada desde
cero en PyTorch, sin modelos preentrenados, analiza el corte, y su probabilidad se combina con tres variables
clínicas (subtipo HR/HER2, volumen tumoral y edad) en una regresión logística. Incluye una aplicación web
(Streamlit) que reproduce exactamente el preprocesado y la inferencia del entrenamiento.

> **Uso exclusivamente educativo.** Sin validez clínica: no es un dispositivo médico y no sirve para
> diagnosticar ni para decidir tratamientos.

## Resultados

El modelo, sus coeficientes y sus umbrales se fijaron solo con train. Después se evaluaron **una sola vez** en el
conjunto de test independiente: 176 pacientes, 53 de ellas con pCR, apartadas desde el principio y de las mismas
cohortes que train (I-SPY1, I-SPY2 y Duke).

Métrica principal: el AUC **por paciente**, calculado sobre la media de las probabilidades de sus cortes. Los
IC95 son de un bootstrap por paciente con 2.000 remuestreos.

| Modelo | Test independiente, AUC (IC95) | Validación cruzada en train, AUC (IC95) |
|---|---|---|
| Solo cohorte (referencia) | 0,543 (0,461–0,622) | 0,523 (0,484–0,560) |
| Solo CNN | 0,572 (0,477–0,661) | 0,545 (0,508–0,584) |
| Clínico (subtipo, volumen, edad) | 0,674 (0,590–0,758) | 0,686 (0,653–0,720) |
| **Combinado (clínico + CNN)** | **0,679 (0,596–0,760)** | **0,691 (0,659–0,725)** |

La validación cruzada es de 5 folds por paciente en train: 1.097 pacientes y 10.945 cortes.

**El combinado con su umbral congelado** (0,431 por paciente, el de mayor sensibilidad con especificidad
≥ 0,85 en train) da esto en test:

| | Predicho pCR | Predicho no pCR |
|---|---|---|
| **pCR** | 15 (VP) | 38 (FN) |
| **No pCR** | 18 (FP) | 105 (VN) |

| Sensibilidad | Especificidad | Precisión | VPN |
|---|---|---|---|
| 0,283 (IC95 0,161–0,411) | 0,854 (IC95 0,794–0,913) | 0,455 | 0,734 |

**Lectura:**
- La imagen apenas añade información a las variables clínicas. La diferencia combinado − clínico es +0,005
  (IC95 −0,021 a +0,029) en test y +0,006 (IC95 −0,005 a +0,015) en validación cruzada.
- La CNN sola queda cerca de la referencia «solo cohorte», que solo conoce la tasa de pCR de cada cohorte.

El detalle está en [`experimentos/test/resumen_test.md`](experimentos/test/resumen_test.md): nivel por corte,
AUC por cohorte, AP, Brier, curvas ROC y calibración. Las cifras de validación cruzada están en
[`modelo_app/fusion.json`](modelo_app/fusion.json).

## Enlaces

- **Aplicación web:** <https://casocancercnn-wxbzjp2szt25nxqmidpwfy.streamlit.app/>. Si la aplicación aparece
  dormida, pulse el botón para reactivarla; tarda alrededor de un minuto.
- **Pesos del modelo en Hugging Face:** <https://huggingface.co/AlvaroGlez/pcr-cnn-breastdcedl> (`modelo.pt`, con
  `modelo.json` y `fusion.json`). No están en este repositorio: ver [Pesos del modelo](#pesos-del-modelo).
- **Informe:** [`docs/informe.pdf`](docs/informe.pdf).
- **Presentación:** [`docs/presentacion_pcr.pptx`](docs/presentacion_pcr.pptx).

## Modelo final en breve

- **CNN** ([`models/cnn_base.py`](models/cnn_base.py)), con 98.049 parámetros entrenables:
  - 4 bloques Conv 3×3 → BatchNorm → ReLU → MaxPool, con filtros 16-32-64-128;
  - después, pooling global medio → Dropout (0,35) → Linear;
  - un único logit y pérdida `BCEWithLogitsLoss`.
- **Entrada:** las tres fases DCE apiladas como canales (PRE, EARLY, LATE) y divididas entre 255, sin
  normalización ImageNet.
  - Una ventana fija de **160 × 160 px centrada en la caja del tumor**, sin reescalar, para que el tumor
    conserve su tamaño.
  - La caja es un dato de la paciente (`patients.csv`). Sin ella, la app no predice.
- **Receta de entrenamiento**, contra el sobreajuste, que aparecía entre las épocas 3 y 8:
  - AdamW, lr 1e-3 con bajada lineal hasta 0, weight decay 1e-2, batch 32;
  - 12 épocas fijas, sin elegir época con la validación;
  - 3 cortes al azar por paciente en cada época.
- **Modelo final:** la ejecución del fold 0 (`cv_receta_centrado160_f0`, regla fijada de antemano).
- **Fusión tardía** ([`exportar_fusion.py`](exportar_fusion.py)): regresión logística sobre:
  - el subtipo HR/HER2 (indicadores);
  - log(1 + volumen tumoral);
  - la edad;
  - el logit de la CNN.

  Se ajusta por corte con todo train.
- **Variables no usadas:** raza (tratarla como causal sería un error metodológico), menopausia (12 % de valores
  faltantes) y variables de adquisición.
- **Valores faltantes:** un valor clínico que falte se rellena con la media de train, como en el ajuste, y la
  app lo avisa.

## Pesos del modelo

Los pesos de la CNN (`modelo.pt`, 0,4 MB) no se versionan: están en Hugging Face, en
[AlvaroGlez/pcr-cnn-breastdcedl](https://huggingface.co/AlvaroGlez/pcr-cnn-breastdcedl). En el repositorio solo
quedan [`modelo_app/modelo.json`](modelo_app/modelo.json), con la configuración y el SHA-256 de los pesos, y
[`modelo_app/fusion.json`](modelo_app/fusion.json), con el modelo combinado. Ninguno de los dos lleva pesos.

- **La app los descarga sola.** Al arrancar, si falta `modelo_app/modelo.pt`, lo descarga de Hugging Face en una
  revisión fija (commit `fcf55eeca9ac`, en [`app/nucleo/paquete.py`](app/nucleo/paquete.py)).
- **Y los verifica.** Comprueba su SHA-256 contra el de `modelo.json` antes de usarlos. Si no coincide, o si Hugging
  Face no responde, no carga el modelo y lo explica.
- **Para reproducir en local**, descárgalos a `modelo_app/`. Los necesitan `prueba_app.py` y `evaluar_test.py`:

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/AlvaroGlez/pcr-cnn-breastdcedl/resolve/fcf55eeca9ac944575fc8623da279efa7482a826/modelo.pt', 'modelo_app/modelo.pt')"
python -c "import hashlib; print(hashlib.sha256(open('modelo_app/modelo.pt', 'rb').read()).hexdigest())"
```

El segundo comando debe imprimir el mismo SHA-256 que figura en `modelo_app/modelo.json` (`e86a36a9…`).

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `train.py` | Entrenamiento de la CNN; guarda configuración, métricas y gráficas en `experimentos/` y pesos en `outputs/` |
| `preprocesado.py` | Función única de preprocesado (entrenamiento, evaluación y app), con el recorte por la caja |
| `models/cnn_base.py` | Arquitectura de la CNN |
| `seleccion.py`, `muestreo.py`, `aumento.py` | Criterio de época, muestreo de cortes por paciente y aumento geométrico |
| `caja_tumor.py` | Caja del tumor de `patients.csv` llevada al PNG (y figuras de comprobación) |
| `clinica.py` | Construcción de las variables clínicas (única fuente, también para la app) |
| `informe.py`, `regenerar_informes.py` | Informe y gráficas de cada ejecución |
| `cv_resumen.py`, `analisis_presentable.py`, `analisis_ultimo.py` | Resúmenes y análisis de la validación cruzada |
| `exportar_oof.py` | Probabilidades out-of-fold de la CNN (se quedan en local: llevan ids) |
| `fusion_clinica.py`, `exportar_fusion.py` | Modelos clínico y combinado; `modelo_app/fusion.json` |
| `exportar_modelo.py` | Paquete de la CNN para la app: `modelo_app/modelo.pt` (no se versiona) y `modelo.json` |
| `evaluar_test.py` | Evaluación única en test del modelo congelado → `experimentos/test/` |
| `prueba_app.py` | Prueba automática de la app (equivalencia con `train.py`, validación de entradas, interfaz) |
| `descargar_datos.py` | Descarga del dataset a `breastdcedl/` |
| `preparar_ejemplos.py`, `preparar_datos_app.py` | Ejemplos y resumen agregado del dataset para la app |
| `audit/auditoria.py` | Auditoría de integridad del dataset |
| `app/` | Aplicación Streamlit: `nucleo/` (lógica, sin Streamlit), `paginas/` (interfaz), `ejemplos/`, `datos/` |
| `modelo_app/` | Configuración del modelo final (`modelo.json`) y modelo combinado (`fusion.json`); los pesos, en Hugging Face |
| `experimentos/` | Resultados agregados de cada ejecución (`experimentos.csv` y una carpeta por ejecución) y `test/` |
| `docs/` | Informe y diapositivas |

## Cómo reproducir

### 1. Instalación (Python 3.11)

`requirements.txt` no incluye PyTorch, que se instala aparte según la máquina.

**CPU (Windows, PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**GPU (Linux, bash):** instala PyTorch 2.14.1 con el comando de <https://pytorch.org/get-started/locally/>
para tu GPU. Los resultados se obtuvieron con ROCm 7.14 en una AMD Radeon RX 6700 XT. Después:

```bash
python -m pip install -r requirements.txt
```

### 2. Datos

```bash
python descargar_datos.py
```

Descarga BreastDCEDL en `breastdcedl/`: PNG de 256 × 256 y `metadata/samples.csv` y `patients.csv`. Ese
directorio no se versiona (ver [Datos que no están en el repositorio](#datos-que-no-están-en-el-repositorio)).

### 3. Entrenamiento de la configuración final (5 folds)

```bash
for f in 0 1 2 3 4; do
  python train.py --epocas 12 --criterio ultima --lr 1e-3 --lr-lineal --weight-decay 1e-2 --dropout 0.35 \
    --cortes-por-paciente 3 --recorte-centrado 160 --fold $f --nombre cv_receta_centrado160_f$f
done
```

`train.py` no lanza un experimento si hay ficheros `.py` sin commit; `--permitir-sin-commit` lo salta y queda
registrado. En la GPU tarda alrededor de 1 minuto por fold.

### 4. Modelo combinado y paquete de la app

```bash
python exportar_oof.py                                # probabilidades out-of-fold de los 5 folds
python exportar_fusion.py receta_centrado160          # modelo combinado -> modelo_app/fusion.json
python exportar_modelo.py cv_receta_centrado160_f0    # CNN del fold 0 -> modelo_app/modelo.pt y modelo.json
python prueba_app.py                                  # la app da la misma probabilidad que train.py
```

### 5. Evaluación única en test

```bash
python evaluar_test.py                                # ensayo en seco: valida todo sin predecir en test
python evaluar_test.py --confirmo-evaluacion-unica    # una sola vez -> experimentos/test/
```

`evaluar_test.py` se niega a ejecutarse en estos casos:
- si `modelo_app/` o el código tienen cambios sin commit;
- si el SHA-256 de `modelo.pt` no coincide con el de `modelo.json`;
- si ya existe `experimentos/test/`. Este repositorio ya incluye la evaluación hecha.

### 6. Aplicación web en local

```bash
streamlit run app/streamlit_app.py
```

Se abre en <http://localhost:8501>.

Para desplegarla en Streamlit Community Cloud:
- archivo principal `app/streamlit_app.py`;
- Python 3.11;
- dependencias de [`app/requirements.txt`](app/requirements.txt), con torch CPU.

## Datos que no están en el repositorio

Aquí solo hay código y resultados agregados. El `.gitignore` deja fuera:

- **El dataset (`breastdcedl/`).** Contiene las etiquetas de pCR de cada paciente, también las de test, y no se
  publican. Se descarga con `descargar_datos.py`.
- **`outputs/`.** Contiene los pesos de cada ejecución (`best.pt`, `last.pt` y los de cada época) y los CSV de
  predicciones por corte y por paciente, que llevan las etiquetas.
- **Los pesos del modelo final (`modelo_app/modelo.pt`).** Ningún `.pt` se versiona. Están en Hugging Face, y la
  app los descarga y los verifica (ver [Pesos del modelo](#pesos-del-modelo)).
- **`experimentos/oof/` y `experimentos/oof_cortes/`.** Contienen las probabilidades out-of-fold de la CNN
  (`exportar_oof.py`). No llevan la etiqueta, pero sí los ids de pacientes y cortes. Se regeneran en la máquina
  que tenga los pesos. Sin ellos, `exportar_fusion.py` para con un mensaje que lo explica.

## Nota sobre el historial

Los commits que citan los `config.json` de `experimentos/*/` (campo `git_commit`) corresponden al historial de
desarrollo. Ese historial se conserva en un repositorio privado: este repositorio público empieza con el estado
final.

## Datos y licencia

- **Datos:** BreastDCEDL (Fridman et al., *Scientific Data*, 2026,
  [doi:10.1038/s41597-026-06589-6](https://doi.org/10.1038/s41597-026-06589-6)). Se construyó a partir de las
  colecciones I-SPY1, I-SPY2 y Duke de The Cancer Imaging Archive (TCIA). Son imágenes desidentificadas. Si
  usas los datos, cita el artículo de BreastDCEDL y las colecciones originales de TCIA.
- **Imágenes de ejemplo** ([`app/ejemplos/`](app/ejemplos/)): tres cortes de BreastDCEDL, bajo
  CC BY-NC 4.0, la licencia del dataset.
- **Código:** licencia MIT (ver [`LICENSE`](LICENSE)), © 2026 Álvaro González Fernández.
