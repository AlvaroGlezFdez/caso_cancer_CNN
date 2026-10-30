"""Limitaciones y ética: una idea por tarjeta, en frases cortas (resultados, qué ve el modelo,
datos y uso); las cifras y el detalle, en un desplegable."""

from __future__ import annotations

import streamlit as st

from nucleo import datos, test_independiente as ti
from nucleo.formato import num, pct
from paginas import comun


def _grupo(titulo: str, tarjetas: list[dict]) -> None:
    st.html(f'<div class="antetitulo" style="margin:1.1rem 0 .55rem">{titulo}</div>')
    comun.tarjetas(tarjetas, columnas_min=190)


def _contenido() -> None:
    comun.modelo_o_parar()
    r = datos.resumen()
    h = r["hallazgos_auditoria"]
    pc = r["por_cohorte"]
    test = ti.cargar()
    comun.encabezado("Limitaciones y ética", "Uso responsable", "Lo que esta herramienta no puede decir.")
    comun.tarjetas([dict(icono="block", aviso=True, titulo="No es un dispositivo médico",
                         texto="Uso docente. No diagnostica ni recomienda tratamientos.")], columnas_min=600)

    if test is not None:
        tp = test["niveles"]["paciente"]
        auc, dif = tp["modelos"]["combinado"]["auc"], tp["diferencias"]["combinado_menos_clinico"]["valor"]
        rendimiento = f"AUC en test {num(auc, 2)}: discrimina, pero poco."
        clinica = f"La imagen suma {num(dif, 3)} de AUC a las variables clínicas."
    else:
        rendimiento, clinica = "Discrimina poco.", "La imagen añade poco a las variables clínicas."
    _grupo("Resultados", [
        dict(icono="trending_flat", titulo="Rendimiento modesto", texto=rendimiento),
        dict(icono="clinical_notes", titulo="La clínica pesa más", texto=clinica),
        dict(icono="percent", titulo="No es un riesgo individual", texto="La probabilidad ordena casos."),
        dict(icono="science", titulo="Test pequeño", texto="176 pacientes: intervalos anchos."),
    ])
    _grupo("Qué ve el modelo", [
        dict(icono="layers", titulo="Un corte, no la paciente", texto="Otro corte puede dar otra probabilidad."),
        dict(icono="hub", titulo="La imagen delata la cohorte", texto="Una red puede acertar sin mirar el tumor."),
        dict(icono="visibility", titulo="Atención orientativa", texto="Grad-CAM no señala la causa."),
        dict(icono="motion_blur", titulo="Movimiento entre fases", texto="Altera el realce en algunas imágenes."),
    ])
    _grupo("Datos y uso", [
        dict(icono="history", titulo="Datos retrospectivos", texto="Sin validación en otros hospitales."),
        dict(icono="groups", titulo="Una cohorte domina", texto="I-SPY2 aporta la mayoría de pacientes."),
        dict(icono="copyright", titulo="CC BY-NC 4.0", texto="Uso no comercial, citando BreastDCEDL."),
        dict(icono="lock", titulo="Privacidad", texto="Lo subido se procesa en memoria y no se guarda."),
    ])

    with comun.mas_detalle("El detalle de cada limitación"):
        lineas = []
        if test is not None:
            tm, td = tp["modelos"], tp["diferencias"]["combinado_menos_clinico"]
            lineas += [
                f"**Rendimiento.** En test ({tp['n']} pacientes, evaluadas una sola vez), AUC del modelo final "
                f"{num(tm['combinado']['auc'])} (IC95 {num(tm['combinado']['auc_ic95'][0])}–"
                f"{num(tm['combinado']['auc_ic95'][1])}); 0,5 es el azar.",
                f"**La clínica pesa más.** El clínico solo da {num(tm['clinico']['auc'])}; la diferencia del combinado "
                f"es {num(td['valor'])} (IC95 {num(td['ic95'][0])} a {num(td['ic95'][1])}). La CNN sola, "
                f"{num(tm['cnn']['auc'])}."]
        lineas += [
            "**Probabilidad.** Ordena casos de menos a más probable; no es la probabilidad real de una paciente.",
            "**Un corte.** La app analiza un corte suelto con un umbral elegido por corte, pero la pCR es de la "
            "paciente.",
            f"**Cohorte.** La tasa de pCR va de {pct(pc['Duke']['tasa_pCR'])} a {pct(pc['I-SPY2']['tasa_pCR'])} según "
            f"la cohorte, y el corazón aparece en {h['corazon']['si']['I-SPY2']}/20 imágenes auditadas de I-SPY2 y en "
            "0/20 de I-SPY1: pistas que no son el tumor.",
            "**Atención.** Grad-CAM muestra qué regiones influyen en la salida, no dónde está el tumor ni por qué "
            "responde.",
            f"**Movimiento.** Bordes dobles en el realce en {sum(h['movimiento']['si'].values())} de 60 imágenes "
            "auditadas.",
            "**Datos.** Dos ensayos clínicos y un centro, ya recogidos; otros hospitales usan otros protocolos.",
            f"**Cohorte dominante.** I-SPY2 aporta el {pct(pc['I-SPY2']['pacientes_train'] / r['train']['pacientes'])} "
            "de las pacientes de entrenamiento.",
            "**Licencia.** BreastDCEDL (Fridman et al., 2026), CC BY-NC 4.0: uso no comercial, citando la fuente; "
            "imágenes desidentificadas, sin intentar reidentificar a nadie.",
            "**Privacidad.** Sin etiquetas individuales ni ids; lo subido no se escribe en disco ni se envía a "
            "terceros."]
        st.markdown("\n\n".join(lineas))


comun.ejecutar(_contenido)
