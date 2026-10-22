# Proyecto: predicción de pCR con una CNN desde cero (caso BreastDCEDL)

Contexto: lee breastdcedl/GUIA.md y breastdcedl/documentation/caso_breastdcedl.pdf
antes de cualquier tarea. Este archivo resume el estado del proyecto y las
decisiones ya tomadas: respétalas salvo que el usuario diga lo contrario.

## Cómo trabajamos: sesiones sin memoria
- El proyecto se desarrolla en dos máquinas: un portátil Windows sin GPU (solo CPU)
  y un ordenador de laboratorio Linux con GPU AMD (ROCm). El código se sincroniza con git: pull al empezar,
  commit y push al terminar.
- Cada sesión de Claude Code empieza sin ningún contexto del proyecto. Este archivo
  es la única memoria compartida. Al empezar una sesión: lee este archivo entero,
  revisa los últimos commits (git log) y el estado del repositorio, comprueba en qué
  máquina estás (torch.cuda.is_available()) y haz un resumen breve al usuario de
  dónde está el proyecto antes de hacer nada.
- Las decisiones de diseño se toman fuera de Claude Code, con el usuario. Tu función
  es leer el contexto, implementar lo que se te pida, ejecutarlo e informar de los
  resultados. No rediseñes por tu cuenta.
- Al terminar cada tarea, actualiza "Estado actual" (qué se hizo, en qué máquina y
  qué queda pendiente) para que la siguiente sesión pueda continuar en cualquiera
  de las dos máquinas, y recuerda al usuario hacer commit y push.

## Reglas obligatorias
- CNN 2D construida desde cero en PyTorch. Prohibido usar modelos preentrenados,
  torchvision.models o transfer learning.
- Salida de la red: un único logit. Pérdida: BCEWithLogitsLoss.
- La unidad estadística es la paciente: ningún patient_id puede aparecer en más de
  un split ni en más de un fold. Toda partición interna se hace por paciente.
- El conjunto test solo se usa para la evaluación final. Nunca para elegir
  hiperparámetros, época, umbral ni método de agregación.
- Los 3 canales son fases temporales (PRE, EARLY, LATE, en ese orden), no colores:
  nada de normalización ImageNet, aumento de color ni ImageFolder. Cualquier
  transformación geométrica se aplica idéntica a las tres fases.
- Preprocesado: 3 PNG uint8 de 256x256 apilados y divididos entre 255. Debe existir
  una única función de preprocesado compartida por entrenamiento, evaluación y app.
- Reutiliza breastdcedl/utils_caso.py en lugar de reescribir su lógica.
- Reproducibilidad: semilla fija; configuración, métricas y pesos guardados en disco.
- Nunca modifiques el dataset original. breastdcedl/ no se sube a git: contiene
  etiquetas que no pueden publicarse.
- Código en archivos .py, no en notebooks.

## Dónde se guardan los resultados
- experimentos/ (SÍ se sube a git, para compartir resultados entre máquinas):
  experimentos/experimentos.csv con una fila por ejecución, y una carpeta
  experimentos/<fecha_hora>_<nombre>/ por ejecución con config.json, train.log,
  metricas_epoca.csv, metricas por cohorte y todas las gráficas.
  Aquí solo van resultados agregados: nunca etiquetas ni predicciones por paciente.
- outputs/ (NO se sube a git): pesos de los modelos (*.pt) y CSV de predicciones
  por corte o por paciente, porque contienen etiquetas. Misma estructura:
  outputs/runs/<fecha_hora>_<nombre>/.
- Si una tarea necesita pesos o predicciones generados en la otra máquina, avisa:
  no están en el repositorio.

## Forma de trabajar
- Las decisiones de diseño (arquitectura, hiperparámetros, umbral) las toma el usuario.
  Si algo no está especificado, pregunta en lugar de decidir.
- Pregunta antes de instalar paquetes.
- Al terminar cada tarea, explica brevemente qué has hecho y por qué, y actualiza
  la sección "Estado actual" de este archivo.
- Los entrenamientos los lanza el usuario desde la terminal: prepara el código, dale
  el comando exacto (PowerShell en el portátil, bash en el laboratorio; desde la
  raíz del repo) y no los ejecutes tú.
  Pruebas rápidas (--prueba) solo si las pide.

## Puesta en marcha en un ordenador nuevo
requirements.txt NO incluye torch: torch se instala aparte según la máquina y
después se instala requirements.txt (versiones exactas del .venv del portátil).
- Portátil (Windows, solo CPU, PowerShell):
  1. Entorno virtual .venv con Python 3.11.
  2. torch 2.14.1 versión CPU con el comando de https://pytorch.org/get-started/locally/.
  3. .venv\Scripts\python.exe -m pip install -r requirements.txt
  4. .venv\Scripts\python.exe descargar_datos.py (queda en breastdcedl/).
- Laboratorio (Linux, bash, GPU AMD Radeon RX 6700 XT):
  1. Entorno de conda "cancer" (Python 3.11), que ya trae PyTorch 2.14.1+rocm7.14:
     no hay que instalar torch ni crear .venv. Para ejecutar, sin activar el entorno:
     /home/aulas/miniconda3/envs/cancer/bin/python (o conda activate cancer).
  2. python -m pip install -r requirements.txt (hecho el 2026-10-22; torch siguió
     en +rocm y pip check sin conflictos).
  3. python descargar_datos.py (hecho el 2026-10-22).
  4. ROCm se usa a través de la API cuda de PyTorch: torch.cuda.is_available()
     devuelve True y train.py registra "dispositivo cuda". No hace falta ninguna
     variable de entorno (ni HSA_OVERRIDE_GFX_VERSION ni otras). Los avisos de
     MIOpen y libnuma que salen al ejecutar son normales y se ignoran.
  Los comandos que se den para esta máquina son de bash, no de PowerShell.

## Estructura del dataset (verificada en la auditoría)
- breastdcedl/metadata/samples.csv: una fila por corte (sample_id, patient_id,
  split, path_pre, path_early, path_late, slice_index, pCR, fold).
  fold 0-4 en train, -1 en test.
- breastdcedl/metadata/patients.csv: una fila por paciente, incluida la cohorte.
- Cohortes por prefijo de patient_id: ISPY1_ = I-SPY1; ISPY2- y ACRIN-6698- = I-SPY2;
  Breast_MRI_ = Duke.
- utils_caso.py ofrece: cargar_imagen (devuelve 3x256x256 float32 en [0,1]),
  particion(samples, fold_val), conjunto_test, BreastDCEDataset, pos_weight,
  agregar_por_paciente (media, máximo, mediana, voto) y evaluar_por_paciente.

## Resultados de la auditoría (audit/auditoria.py -> outputs/auditoria/)
- Integridad: las 14 comprobaciones OK (sin solape de pacientes entre splits ni folds).
- Train: 1.097 pacientes (775 sin pCR, 322 con pCR). Test: 176 (123 / 53).
  Casi todas las pacientes tienen 10 cortes.
- Tasa de pCR en train por cohorte: I-SPY2 0,32; I-SPY1 0,25; Duke 0,21.
- Diferencias de adquisición entre cohortes: Duke tiene menor realce EARLY-PRE;
  I-SPY1 tiene PRE más brillante. Riesgo de que la red aprenda la cohorte en lugar
  de la biología: evaluar siempre por cohorte y contra la referencia "solo cohorte".
- PRE < EARLY en el 99,4 % de pacientes (7 excepciones, se mantienen y se documentan).
  Lavado (LATE < EARLY) en el 18-20 %.
- pos_weight = N0/N1 varía entre 2,33 y 2,55 según el fold: calcularlo siempre
  con la función de utils_caso.py sobre la partición de entrenamiento.

## Decisiones tomadas
- Línea base (98.049 parámetros entrenables):
  4 bloques Conv2d 3x3 (padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2),
  filtros 3->16->32->64->128; AdaptiveAvgPool2d(1) -> Dropout(0.3) -> Linear(128, 1).
- Entrenamiento por defecto: AdamW con weight decay 0 (equivale a Adam: comprobado
  bit a bit), lr 1e-3, batch 32, máximo 30 épocas, early stopping con paciencia 7
  vigilando el AUC de validación por paciente (agregación media), semilla 42, sin
  aumento de datos.
- Contra el sobreajuste (decidido el 2026-10-21), cada uno por separado:
  - Aumento geométrico (--aumento): volteo horizontal p=0,5 y rotación uniforme de
    hasta ±10°, la misma transformación en las 3 fases, solo en entrenamiento.
    Nada de brillo ni contraste.
  - Weight decay 1e-4 (--weight-decay 1e-4) con AdamW.
- Validación interna: fold 0 fijo para comparar todas las variantes.
- Normalización: solo dividir entre 255.
- Experimentos previstos, cambiando una sola cosa cada vez: con y sin pos_weight
  (obligatorio); versión estrecha 8->64 filtros; quinto bloque; realce explícito
  (EARLY-PRE y LATE-EARLY como canales internos).
- Evaluación: además de las métricas globales, métricas por cohorte y AUC de la
  referencia "solo cohorte".

## Estado actual
- Proyecto retomado el 2026-10-21 (estuvo en pausa desde el 2026-10-14).
- Hecho (portátil sin GPU, 2026-10-12):
  - Descarga del dataset, entorno, auditoría completa.
  - Infraestructura de entrenamiento: models/cnn_base.py (CNNBase), preprocesado.py
    (función única de preprocesado, también para la app), train.py e informe.py.
    psutil instalado en el .venv del portátil.
  - Circuito validado en CPU: --max-batches 5 sin errores (pérdida del primer lote
    0,6234 y no 0,69 porque con la inicialización por defecto el logit inicial medio
    es -0,52; con logit 0 daría 0,6931); --overfit-batches 2 baja la pérdida de 0,61
    a ~0,05 en 100 iteraciones.
  - Línea base de 3 épocas sin pos_weight (experimentos/20261012_134406_baseline_3ep_cpu):
    AUC val 0,563, sensibilidad 0 con umbral 0,5: solo aprendió la proporción de
    clases. Su config.json apunta a d6f7286 (código sin commit); el código entrenado
    es el del commit b402ff4 salvo el control de commit de train.py.
  - Línea base completa sin pos_weight (experimentos/20261012_142807_baseline_30ep_cpu,
    commit c7e5967): early stopping en la época 17, mejor época 10; 177,6 min en CPU
    (~10,4 min/época: ~565 s cálculo, ~60 s validación, ~3 s espera de datos).
    - AUC de validación por paciente 0,603 (IC95 de Hanley-McNeil 0,52-0,69;
      bootstrap por paciente 0,53-0,68) frente a 0,534 de "solo cohorte", que cae
      dentro del intervalo. Por cohorte: I-SPY1 0,66 (21 pac.), I-SPY2 0,61 (156),
      Duke 0,56 (42).
    - Umbral 0,5: sensibilidad 0,45 (29/64), especificidad 0,70 (108/155).
    - Sobreajuste: la pérdida de validación es mínima en la época 8 y luego sube.
      El AUC por época oscila entre 0,52 y 0,60; el 0,603 es un pico aislado y algo
      optimista (época elegida con la misma validación).
    - Mal calibrada: probabilidad media 0,44 frente a tasa real 0,29; Brier 0,228
      (predecir siempre la tasa de pCR da 0,207). Sensibilidad y especificidad con
      umbral 0,5 cambian mucho de una época a otra.
    - Probabilidades por paciente entre 0,12 y 0,76; media 0,43 en no pCR y 0,48
      en pCR (poca separación).
    - Sus 3 primeras épocas son idénticas a las de baseline_3ep_cpu (la semilla es
      reproducible).
- Hecho (portátil sin GPU, 2026-10-21, commit d793f98): preparación de los
  experimentos contra el sobreajuste.
  - requirements.txt con las versiones exactas del .venv del portátil.
  - aumento.py + train.py --aumento: volteo horizontal p=0,5 y rotación uniforme
    ±10° (bilineal, relleno 0), la misma transformación en las 3 fases (una única
    rejilla de muestreo por corte), aplicada solo a los lotes de entrenamiento ya en
    el dispositivo. Usa un generador propio (semilla+1): no altera ni el barajado ni
    el dropout, y no depende de --num-workers. `python aumento.py` comprueba que con
    las 3 fases iguales salen 3 canales idénticos (diferencia 0) y guarda una figura
    en outputs/pruebas/ (revisada: fases alineadas, EARLY-PRE nítido).
  - train.py --weight-decay (por defecto 0) y optimizador AdamW. AdamW con weight
    decay 0 da pesos idénticos bit a bit a Adam (20 pasos de CNNBase), así que la
    línea base no cambia. config.json e informe.md registran optimizador, weight
    decay y aumento; experimentos.csv tiene una columna weight_decay nueva (0 en las
    dos filas anteriores, que usaron Adam).
  - anadir_fila relee experimentos.csv con float_precision="round_trip" (antes
    reescribía el último decimal de algunos valores de filas anteriores).
  - Pruebas --prueba --max-batches 2 --epocas 2 con --aumento y con
    --weight-decay 1e-4: sin errores. Con weight decay, la pérdida del primer lote
    es 0,6234292984008789, idéntica a la línea base (misma inicialización y barajado).
- Hecho (laboratorio, Linux con GPU AMD RX 6700 XT, 2026-10-22, commit b47e6e7):
  puesta en marcha.
  - requirements.txt sin torch (torch se instala aparte según la máquina) e
    instalado en el entorno de conda "cancer": torch sigue en 2.14.1+rocm7.14,
    torch.cuda.is_available() True, pip check sin conflictos.
  - Dataset descargado y completo: samples.csv 12.703 filas (10.945 train, 1.758
    test), patients.csv 1.273, 38.109 PNG (ninguno falta ni vacío), sha256 de
    utils_caso.py 1d6bf858... (coincide con el del portátil).
  - Prueba `--prueba --max-batches 20 --aumento` (outputs/pruebas/
    20261022_035040_prueba_gpu): "dispositivo cuda", sin errores; ~314 img/s en
    entrenamiento con aumento y espera de datos 0 con --num-workers 2. La primera
    época tarda ~10 s más (calentamiento de MIOpen). Estimación: ~40 s por época
    completa (8.759 cortes de entrenamiento + 2.186 de validación), ~20 min por
    ejecución de 30 épocas, ~1 h 20 min las cuatro (frente a ~10 min por época
    en la CPU del portátil).
- Hecho (laboratorio, GPU AMD, 2026-10-22, commit 843fd52): los cuatro
  entrenamientos de 30 épocas máx. y paciencia 7, fold 0 (~29 s/época, ~8,5 min
  por ejecución). Pesos y predicciones en outputs/runs/ del laboratorio.
  Todos paran en la época 17 con mejor época 10, igual que baseline_30ep_cpu: no
  es un error, la misma semilla da el mismo orden de lotes y las curvas de AUC de
  validación están correlacionadas entre ejecuciones (Spearman 0,71-0,86 entre
  las probabilidades por paciente). El pico de la época 10 es en parte suerte del
  orden de datos. AUC por paciente con IC95 por bootstrap (2.000 remuestreos por
  paciente; script fuera del repo):
  | ejecución | AUC (IC95) | sens / esp (0,5) | prob media | Brier |
  | baseline_30ep_gpu | 0,598 (0,52-0,68) | 0,06 / 0,93 | 0,30 | 0,209 |
  | baseline_pw_30ep_gpu | 0,579 (0,50-0,66) | 0,84 / 0,26 | 0,62 | 0,325 |
  | aumento_30ep_gpu | 0,577 (0,50-0,65) | 0,00 / 0,99 | 0,25 | 0,208 |
  | wd_30ep_gpu | 0,586 (0,51-0,67) | 0,20 / 0,85 | 0,38 | 0,215 |
  (baseline_30ep_cpu: 0,603, 0,45 / 0,70, 0,44, 0,228; "solo cohorte" 0,534;
  tasa real de pCR 0,29; predecir siempre la tasa da Brier 0,207.)
  - Diferencias pareadas frente a baseline_30ep_gpu (bootstrap): pos_weight
    -0,019 (IC -0,09 a 0,05), aumento -0,021 (-0,08 a 0,04), weight decay -0,012
    (-0,06 a 0,04). Ninguna variante mejora; todas las diferencias son ruido.
  - Los AUC por época oscilan entre ~0,48 y 0,60 (media ~0,53-0,55, cerca de
    "solo cohorte"); el mejor AUC es un máximo elegido sobre una curva ruidosa.
  - Sobreajuste: la pérdida de entrenamiento baja sin parar (0,61 -> 0,46 en la
    línea base) y la de validación sube desde la época ~3-10. El aumento frena
    el descenso de la pérdida de entrenamiento (0,55 en la época 17) pero no
    mejora la validación. Weight decay 1e-4 con AdamW casi no cambia nada
    (curvas casi iguales a la línea base): con lr 1e-3 es demasiado pequeño.
  - pos_weight desplaza las probabilidades hacia arriba (media 0,62) sin cambiar
    el AUC; su pérdida de validación es ponderada y no se compara con las demás.
  - Calibración de la línea base en GPU: media 0,30 frente a 0,29, pero Brier
    0,209, prácticamente el de predecir la tasa constante: casi no hay
    información útil más allá de la prevalencia. La "mala calibración" de
    baseline_30ep_cpu (0,44) era propia de la época elegida.
  - Sensibilidad y especificidad con umbral 0,5 saltan de una época a otra
    (las probabilidades se concentran cerca del umbral): no sirven para comparar.
  - Por cohorte el AUC no es interpretable con fold 0: I-SPY1 tiene 4 pCR (AUC
    0,41-0,75 según la ejecución) y Duke 11.
  - GPU frente a CPU en la línea base: 0,598 frente a 0,603, compatible.
- Siguiente paso: lo decide el usuario a la vista de estos resultados (ver
  decisiones pendientes). Sin entrenamientos pendientes.
- 2026-10-22: en el laboratorio apareció un cambio no intencionado en .gitignore
  ("experimentos/" en lugar de "!experimentos/**/train.log"), que hacía que git
  ignorase las carpetas nuevas de experimentos/. Se deshizo con git restore. Si
  las carpetas nuevas no salen en git status, revisar .gitignore.
- Decisiones pendientes (las toma el usuario; no decidir por él):
  - Tras los resultados: si se usa pos_weight, qué regularización se adopta
    (aumento, weight decay, ambos o ninguno) y si hacen falta más pruebas (otros
    valores de weight decay, combinaciones).
  - Qué hacer con la mala calibración (probabilidad media 0,44 frente a 0,29).
  - Umbral (ahora 0,5) y método de agregación por paciente (ahora media): con la
    validación interna, nunca con test.
  - Cuándo y en qué orden lanzar los experimentos de arquitectura previstos
    (estrecha 8->64, quinto bloque, realce explícito).
  - Si el modelo final se valida solo con el fold 0 o con validación cruzada de
    5 folds.
  - Qué modelo se cierra como final y cuándo se hace la evaluación única en test.
  - App web: framework (Streamlit, Gradio...) y dónde desplegarla (debe ser
    accesible por URL; requisitos en GUIA.md C6).
  - Detalles del aumento que eligió Claude y el usuario dejó "de momento así"
    (2026-10-21): interpolación bilineal, relleno 0, ángulo uniforme y semilla del
    aumento = semilla + 1. Revisables.
- Entregables aún sin empezar (GUIA.md C7): informe, cinco diapositivas, app
  desplegada y pesos del modelo final.
- Notas:
  - train.py no lanza un experimento (no prueba) si hay archivos .py sin commit;
    --permitir-sin-commit lo salta y queda registrado en config.json.
  - psutil ya está instalado en las dos máquinas (está en requirements.txt).
  - Pesos y predicciones de las ejecuciones del 2026-10-12, y la figura del aumento
    (outputs/pruebas/20261021_144701_aumento_fases.png; se regenera con
    `python aumento.py`), solo están en outputs/ del portátil (no van a git).
    Mejor modelo hasta ahora:
    outputs/runs/20261012_142807_baseline_30ep_cpu/best.pt (sha256 b53dfe83..., en
    su config.json).
  - El histograma de probabilidades por clase está en el panel derecho de
    calibracion.png (no hay archivo aparte).
  - En el portátil las épocas pasaron de ~9,3 a ~10,5 min a partir de la 3 (posible
    bajada de frecuencia por temperatura; sin confirmar). En baseline_3ep_cpu la
    época 3 tardó ~526 s, así que no depende del número de época.
  - En CUDA el backward de AdaptiveAvgPool2d no es determinista: dos ejecuciones
    iguales en la GPU pueden diferir ligeramente aunque cudnn.deterministic=True.