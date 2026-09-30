# Proyecto: predicción de pCR con una CNN desde cero (caso BreastDCEDL)

Contexto: lee breastdcedl/GUIA.md y breastdcedl/documentation/caso_breastdcedl.pdf
antes de cualquier tarea. Este archivo resume el estado del proyecto y las
decisiones ya tomadas: respétalas salvo que el usuario diga lo contrario.

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

## Forma de trabajar
- Las decisiones de diseño (arquitectura, hiperparámetros, umbral) las toma el usuario.
  Si algo no está especificado, pregunta en lugar de decidir.
- Pregunta antes de instalar paquetes.
- Al terminar cada tarea, explica brevemente qué has hecho y por qué, y actualiza
  la sección "Estado actual" de este archivo.

## Puesta en marcha en un ordenador nuevo
1. Crear el entorno virtual .venv con Python 3.11.
2. Si hay GPU NVIDIA (comprobar con nvidia-smi), instalar PyTorch con CUDA desde
   las instrucciones oficiales de pytorch.org en lugar de la versión +cpu que
   figura en requirements.txt. El resto de dependencias, desde requirements.txt.
3. Descargar el dataset con: python descargar_datos.py (queda en breastdcedl/).
4. Comprobar que torch.cuda.is_available() devuelve True si hay GPU NVIDIA.

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
- Entrenamiento por defecto: Adam, lr 1e-3, batch 32, máximo 30 épocas,
  early stopping con paciencia 7 vigilando el AUC de validación por paciente
  (agregación media), semilla 42, sin aumento de datos.
- Validación interna: fold 0 fijo para comparar todas las variantes.
- Normalización: solo dividir entre 255.
- Experimentos previstos, cambiando una sola cosa cada vez: con y sin pos_weight
  (obligatorio); versión estrecha 8->64 filtros; quinto bloque; realce explícito
  (EARLY-PRE y LATE-EARLY como canales internos).
- Evaluación: además de las métricas globales, métricas por cohorte y AUC de la
  referencia "solo cohorte".

## Estado actual
- Hecho: descarga del dataset, entorno, auditoría completa.
- Desarrollo hecho en un portátil sin GPU NVIDIA (solo CPU).
- Siguiente tarea: infraestructura de entrenamiento (models/cnn_base.py, train.py,
  registro de cada ejecución en outputs/runs/, outputs/experimentos.csv) y
  ejecuciones de prueba: --max-batches 5, --overfit-batches 2 y una época completa.
  Todavía no se ha entrenado ningún modelo.