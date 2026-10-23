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

## Hallazgos de la auditoría visual (portátil, 2026-10-22 y 2026-10-23)
Hecha fuera del repositorio, en ../exploracion/ (01_cortes a 04_posicion_tumor,
cada una con su observaciones.md). Solo existe en el portátil y no va a git (sus
CSV llevan etiquetas). Solo pacientes de train. Las anotaciones son visuales, de
un único anotador (Claude) y sobre el corte central.
- El PNG es una ventana de 256 px del corte original, sin reescalar y centrada en
  la caja del tumor (desplazada lo justo para no salirse de la imagen). Un píxel
  del PNG mide xy_spacing. Coincide con el PDF: "La versión MinCrop recorta el
  entorno tumoral a 256 × 256".
  - Con n_xy = 256 (912 de las 1.097 pacientes de train) la ventana es la imagen
    entera. En I-SPY el tumor puede estar en cualquier sitio (distancia mediana
    al centro 31-36 px, hasta 113). En Duke suele estar centrado (el 36 % a menos
    de 2 px del centro), lo que apunta a que Duke ya viene recortado; no está
    documentado.
  - Con n_xy = 128 (2 pacientes) el corte está ampliado x2 por vecino más
    próximo, y el píxel mide xy_spacing / 2.
- La caja del tumor de patients.csv es fiable: sraw:eraw son las filas y
  scol:ecol las columnas, en píxeles del corte original. caja_tumor.caja_en_png la
  lleva al PNG en dos pasos:
  1. aplica la ventana anterior;
  2. gira 90° (fila del PNG = 256 - columna original, columna del PNG = fila
     original).

  Comprobación de que cae bien:
  - Cayó sobre la lesión en las 40 de 60 pacientes con una lesión visible: 0
    errores. Las otras 20 no tenían una lesión clara con la que comparar.
  - Cayó también sobre la lesión en 8 pacientes más con n_xy 128, 312 y 448.
  - Hay una sola caja por paciente, la misma en todos sus cortes.

  Caja entera dentro de un recorte central:
  - 128 px: 39 % de las pacientes (I-SPY1 37 %, I-SPY2 27 %, Duke 83 %);
  - 160 px: 61 %;
  - 192 px: 83 %.
- Corazón/tórax: una estructura grande y muy realzada detrás de la pared torácica
  aparece en 13 de 20 imágenes de I-SPY2, en 13 de 20 de Duke y en ninguna de
  I-SPY1. En I-SPY1 hubo 9 dudosas: una estructura de realce moderado cortada por
  el borde. Es una pista ligada a la cohorte.
- Movimiento entre fases: hay bordes dobles rojo/azul en el realce EARLY-PRE
  (posible movimiento entre PRE y EARLY) en 13 de 60 imágenes de forma clara y en
  14 dudosas, repartidas por las tres cohortes.
- Relación con la pCR (AUC como predictor único, en train, IC95 por bootstrap):
  - El tamaño del encuadre en mm (xy_spacing) no se relaciona: 0,50 (0,46-0,54).
  - El tamaño de la caja sí: 0,44 (0,41-0,48) en columnas y 0,47 (0,43-0,50) en
    filas. Dentro de I-SPY2 baja a 0,41; en I-SPY1 y Duke el intervalo incluye
    0,5. Es decir: tumores grandes, menos pCR.
- mask_start/mask_end no comparten el eje z de slice_index en I-SPY:
  - En Duke el corte central siempre cae entre ellos.
  - En I-SPY solo cae en el 52 % (I-SPY1) y el 29 % (I-SPY2), aunque la lesión se
    vea.
- En la defensa el profesor dará todas las variables de patients.csv, incluida la
  caja del tumor (lo dijo el usuario el 2026-10-23).

## Decisiones tomadas
- Línea base (98.049 parámetros entrenables):
  4 bloques Conv2d 3x3 (padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2),
  filtros 3->16->32->64->128; AdaptiveAvgPool2d(1) -> Dropout(0.3) -> Linear(128, 1).
- Entrenamiento por defecto: AdamW con weight decay 0 (equivale a Adam: comprobado
  bit a bit), lr 1e-3, batch 32, máximo 30 épocas, early stopping con paciencia 7
  (en los experimentos de validación cruzada se usa --paciencia 10), semilla 42,
  sin aumento de datos.
- Criterio de selección (decidido el 2026-10-22; train.py --criterio, seleccion.py):
  auc_suavizado, la media del AUC de validación por paciente (agregación media) de
  cada época con la anterior y la siguiente (en los extremos, las disponibles).
  Decide best.pt y el early stopping. --criterio auc reproduce el anterior (AUC de
  la propia época; comprobado idéntico al código antiguo).
- Contra el sobreajuste (decidido el 2026-10-21), cada uno por separado:
  - Aumento geométrico (--aumento): volteo horizontal p=0,5 y rotación uniforme de
    hasta ±10°, la misma transformación en las 3 fases, solo en entrenamiento.
    Nada de brillo ni contraste.
  - Weight decay 1e-4 (--weight-decay 1e-4) con AdamW.
  - 2026-10-22: el aumento de datos queda DESCARTADO por la validación cruzada de
    5 folds (no mejora el AUC: -0,015 de media, IC95 -0,054 a +0,025). Los
    experimentos siguientes van sin aumento.
- Validación interna: hasta el 2026-10-21, fold 0 fijo. Desde el 2026-10-22 las
  variantes se comparan con validación cruzada de 5 folds (train.py --fold,
  cv_resumen.py).
- Normalización: solo dividir entre 255.
- Experimentos previstos, cambiando una sola cosa cada vez: con y sin pos_weight
  (obligatorio); versión estrecha 8->64 filtros; quinto bloque; realce explícito
  (EARLY-PRE y LATE-EARLY como canales internos).
  - Quinto bloque (--bloques 5): un bloque idéntico de 128 a 128 filtros antes del
    GAP; 245.889 parámetros.
  - Realce (--realce): dentro del modelo, antes del primer bloque, la entrada pasa
    a PRE, EARLY-PRE, LATE-EARLY (en ese orden); 98.049 parámetros; la entrada del
    modelo y el preprocesado no cambian.
  - lr 3e-4 (--lr 3e-4).
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
- Decidido el 2026-10-22 (usuario), a la vista de lo anterior: comparar línea
  base y aumento con validación cruzada de 5 folds, sin cambiar nada más del
  diseño ni el criterio de parada. train.py ya tenía --fold (0-4); comprobado que
  uc.particion da en cada fold ~219 pacientes de validación (59-75 con pCR), sin
  solape con entrenamiento ni con test, y que las validaciones de los 5 folds
  son disjuntas.
- Preparado (laboratorio, 2026-10-22): cv_resumen.py lee de experimentos.csv
  las 10 ejecuciones cv_* (si un nombre se repite, usa la más reciente y avisa)
  y escribe experimentos/cv_resumen.csv y cv_resumen.md: AUC por fold de cada
  configuración, media y desviación típica (ddof=1), diferencia aumento - línea
  base por fold y AUC "solo cohorte" por fold y su media. Comprueba que solo
  cambian el fold y el aumento. Solo se ha comprobado la sintaxis; aún no se ha
  ejecutado.
- Hecho (laboratorio, 2026-10-22, commit 7dd2e39): validación cruzada de 5 folds,
  línea base y aumento (cv_baseline_f0-f4, cv_aumento_f0-f4; ~29 s/época, 1 h 40
  min en total). OJO: se lanzaron con --paciencia 10, no 7 como estaba decidido
  (queda en config.json y experimentos.csv). Como el entrenamiento es
  determinista (ver abajo), el resultado con paciencia 7 se reconstruye exacto
  desde metricas_epoca.csv: solo cambia cv_aumento_f1 (0,569 -> 0,540, mejor
  época 4). experimentos/cv_resumen.md (cv_resumen.py; un fallo con los NaN de
  pos_weight se corrigió antes de ejecutarlo):
  | fold | línea base | aumento | dif. | solo cohorte |
  | 0 | 0,598 | 0,577 | -0,021 | 0,534 |
  | 1 | 0,626 | 0,569 | -0,057 | 0,526 |
  | 2 | 0,608 | 0,600 | -0,008 | 0,538 |
  | 3 | 0,540 | 0,549 | +0,009 | 0,593 |
  | 4 | 0,611 | 0,614 | +0,004 | 0,554 |
  | media (DT) | 0,597 (0,033) | 0,582 (0,026) | -0,015 (0,026) | 0,549 |
  Bootstrap por paciente estratificado por fold (2.000 remuestreos; solo mide el
  ruido de la validación, no el del entrenamiento; script fuera del repo):
  - aumento - línea base: -0,015 (IC95 -0,054 a +0,025). Con paciencia 7: -0,021.
    El aumento no mejora el AUC en ningún sentido demostrable.
  - línea base - solo cohorte: +0,048 (IC95 +0,005 a +0,090); aumento - solo
    cohorte: +0,033 (-0,008 a +0,075). Pero son AUC de la mejor época elegida con
    la misma validación: la media del AUC por época (épocas 2 en adelante) es
    0,539 en la línea base y 0,535 con aumento, por debajo de "solo cohorte"
    (0,549). El margen sobre "solo cohorte" depende de escoger el pico.
  - Curvas: en la línea base la pérdida de validación es mínima en la época 2-3
    en los 5 folds y la de entrenamiento baja hasta 0,30-0,49 (memoriza). Con
    aumento la de entrenamiento se queda en 0,51-0,59, pero el AUC no mejora; en
    los folds 2 y 4 la mejor época es la 2 y la 1 (modelo casi sin entrenar).
  - Fold 3: la línea base tiene AUC por época < 0,5 casi siempre (mejor 0,540)
    y "solo cohorte" da 0,593: en ese fold la tasa de pCR de I-SPY2 es 0,40
    (0,14 en I-SPY1). El modelo no separa cohortes (probabilidad media parecida
    en las tres), así que no parece aprender la cohorte, pero tampoco la usa.
  - AUC dentro de cohorte, con las 1.097 pacientes juntas (línea base / aumento):
    I-SPY2 (784, 252 pCR) 0,61 / 0,56; I-SPY1 (104, 26) 0,51 / 0,59; Duke (209,
    44) 0,53 / 0,52. La señal más clara está dentro de I-SPY2.
  - Calibración: en la línea base la probabilidad media varía por fold entre
    0,30 y 0,55 (real 0,27-0,34) y el Brier nunca es mejor que predecir la tasa
    constante. Con aumento está más cerca (0,25-0,41) y en los folds 2 y 4 el
    Brier queda algo por debajo del constante.
  - Spearman entre las probabilidades por paciente de línea base y aumento:
    0,17-0,77 según el fold (ordenan a las pacientes de forma bastante distinta).
  - Determinismo: cv_baseline_f0 es idéntica a baseline_30ep_gpu (diferencia 0
    en pérdidas y AUC de las 17 épocas comunes). En esta GPU AMD con ROCm el
    entrenamiento es reproducible; para medir la variabilidad del entrenamiento
    habría que cambiar la semilla.
- Hecho (laboratorio, 2026-10-22, commit en el que se añade esta línea): a
  petición del usuario, tras la validación cruzada.
  - Criterio auc_suavizado (seleccion.py, train.py --criterio, por defecto). Con
    él la decisión sobre la época e se toma al terminar la e+1; train.py guarda
    los pesos de la época anterior hasta entonces. metricas_epoca.csv tiene una
    columna auc_criterio con el valor del criterio de cada época.
  - Reconstrucción del criterio en las 10 ejecuciones cv_baseline y cv_aumento
    (experimentos/cv_resumen.md y cv_criterio_suavizado.csv): solo es exacta en
    2 de 10 (cv_baseline_f2 y cv_aumento_f2). En las demás, el valor suavizado
    de la última época entrenada necesita el AUC de una época que no se entrenó.
    En 3 la elección depende solo de esa época (se mantiene salvo AUC > 0,563
    en baseline f0, > 0,669 en f1 y > 0,543 en aumento f0); en 5 harían falta
    varias épocas más. Media provisional: línea base 0,586, aumento 0,564.
    seleccion.simular reproduce exactamente la época elegida y la de parada de
    las 16 ejecuciones anteriores con el criterio auc, y coincide con train.py en
    las pruebas con auc_suavizado.
  - Curvas de precisión-sensibilidad: curva_pr.png y average_precision en el
    informe de cada ejecución (y en experimentos.csv como average_precision_val);
    en cv_resumen, AP por fold, media y DT, y cv_curvas_pr.png. AP de la
    validación cruzada: línea base 0,396 (DT 0,045), aumento 0,363 (DT 0,033),
    azar 0,294.
  - Escalas fijas en todas las gráficas, en un único sitio (ESCALA_* al principio
    de informe.py): pérdida 0,3-0,9 (con pos_weight 0,6-1,4 y título
    "ponderada"), AUC por época 0,40-0,80, épocas 1-30, ROC/PR/calibración 0-1,
    histograma X 0-1 e Y 0-40 pacientes por barra (40 lo eligió Claude: cubre
    todas las ejecuciones menos cv_aumento_f2 y f4, modelos casi sin entrenar).
    Lo que se sale se marca con ▲/▼ en el borde.
  - regenerar_informes.py rehace los informes de ejecuciones ya terminadas: las
    14 de la GPU completas (comprueba que las métricas anteriores salen
    idénticas; Brier y probabilidad media cambian < 1e-9 por el redondeo de
    predicciones_val.csv) y las 2 de CPU solo curvas_aprendizaje.png (sus
    predicciones están en el portátil; si se ejecuta allí, se rehacen enteras).
  - --bloques (4 o 5) y --realce en models/cnn_base.py (crear_modelo) y
    train.py; config.json guarda bloques, filtros, realce y criterio_seleccion;
    experimentos.csv tiene columnas nuevas criterio, bloques, realce y
    average_precision_val (vacías en las filas anteriores). La inicialización y
    los nombres de los pesos de la línea base no cambian (comprobado con hash).
  - Pruebas --prueba --max-batches 5 de cada novedad sin errores, y
    --criterio auc con --max-batches 20 --aumento idéntico (diferencia 0) a la
    prueba hecha con el código anterior.
- Hecho (laboratorio, 2026-10-22, commit 52163ff): cv_5bloques_f0-f4 con criterio
  auc_suavizado (sin aumento, --epocas 30 --paciencia 10).
- Hecho (portátil sin GPU, 2026-10-23): auditoría visual en ../exploracion/ (ver
  "Hallazgos de la auditoría visual") y caja_tumor.py.
  - caja_en_png(paciente) sitúa la caja de patients.csv en el PNG.
  - `python caja_tumor.py` la compara con la auditoría en 12 pacientes de train
    (las tres cohortes y todos los n_xy): las 12 OK, con diferencia 0.
  - También guarda una figura en outputs/pruebas/. Revisada: la caja cae sobre la
    lesión en las 12.
  - Todavía no se usa en train.py ni en ninguna otra parte.
- Siguiente paso (laboratorio; el usuario lanza los entrenamientos, Claude no).
  Estado a 2026-10-23, según el usuario:
  1. Hecho con criterio auc_suavizado: cv_5bloques (5 folds) y cv_realce_f0.
     OJO: en el repositorio solo hay una copia parcial de cv_realce_f0
     (experimentos/20261022_073855_cv_realce_f0). Su train.log llega a la época
     10 y no tiene informe ni fila en experimentos.csv, porque se subió con la
     ejecución en marcha. Hay que subir desde el laboratorio la carpeta completa
     y la fila de experimentos.csv.
  2. Faltan dos tandas, sin aumento, criterio auc_suavizado (el de por defecto),
     --epocas 30 --paciencia 10; ~10 min por ejecución, ~1 h 30 min las 9:
     - cv_baseline_suav, 5 folds: la línea base relanzada con auc_suavizado,
       para compararla con las variantes con el mismo criterio;
     - cv_realce, folds 1-4.

     Desde la raíz, en bash:
     `P=/home/aulas/miniconda3/envs/cancer/bin/python; for f in 0 1 2 3 4; do $P train.py --fold $f --nombre cv_baseline_suav_f$f --epocas 30 --paciencia 10; done; for f in 1 2 3 4; do $P train.py --realce --fold $f --nombre cv_realce_f$f --epocas 30 --paciencia 10; done`
  3. cv_lr3e-4: el usuario no lo mencionó el 2026-10-23; confirmar si sigue
     previsto.
  4. cv_resumen.py está escrito para línea base frente a aumento: para resumir
     las variantes nuevas hay que adaptarlo (pedírselo a Claude).
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
  - Cuándo lanzar la versión estrecha 8->64 (quinto bloque, realce y lr 3e-4
    están en curso desde el 2026-10-22).
  - Cómo se elige y se entrena el modelo final a partir de la validación
    cruzada (por ejemplo, reentrenar con todo train o combinar los 5 folds).
  - Qué modelo se cierra como final y cuándo se hace la evaluación única en test.
  - App web: framework (Streamlit, Gradio...) y dónde desplegarla (debe ser
    accesible por URL; requisitos en GUIA.md C6).
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
    En la GPU AMD del laboratorio (ROCm) no se ha observado: dos ejecuciones
    iguales dieron resultados idénticos (2026-10-22).