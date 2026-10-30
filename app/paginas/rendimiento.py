"""Rendimiento, en dos bloques separados:

1. Test independiente (experimentos/test/, de evaluar_test.py): las 176 pacientes
   de test, apartadas desde el principio y evaluadas una sola vez con el modelo
   congelado (mismas cohortes que train). AUC con IC95, comparación con clínico,
   solo CNN y solo cohorte, ROC, matriz de confusión con el umbral congelado,
   calibración y la tabla «CV vs test». Se leen los agregados ya calculados: nada
   se recalcula ni se cargan etiquetas.
2. Validación cruzada en train (selección del modelo): el modelo final combinado
   (modelo_app/fusion.json: cascada solo cohorte -> clínico -> combinado y ROC de
   los cuatro modelos) y la CNN sola (modelo_app/modelo.json y experimentos/).

Dos niveles: por corte (la entrada de la app; los umbrales de la app se eligieron
así) y por paciente (media de sus cortes; la métrica principal del test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nucleo import datos, experimentos as ex, test_independiente as ti
from nucleo.formato import entero, mayuscula, num, tabla_texto
from paginas import comun, graficos as g

NIVEL_CORTE, NIVEL_PACIENTE = "Por corte · el de la app", "Por paciente"
NIVEL_TEST_PACIENTE, NIVEL_TEST_CORTE = "Por paciente · principal", "Por corte"
NOMBRE = {"cohorte": "Solo cohorte", "clinico": "Clínico", "cnn": "Solo CNN", "combinado": "Combinado"}
AYUDA = {
    "auc": "Área bajo la curva ROC: probabilidad de que un caso con pCR reciba una probabilidad mayor que uno sin "
           "pCR. 0,5 es el azar; 1, perfecto.",
    "cohorte": "Referencia «solo cohorte»: a cada caso se le asigna la tasa de pCR de su cohorte en entrenamiento. "
               "Si el modelo no la supera, podría estar reconociendo la cohorte y no la biología.",
    "ap": "Average precision: área bajo la curva precisión-sensibilidad. El azar es la prevalencia de pCR.",
    "brier": "Error cuadrático medio de la probabilidad (0 es perfecto). Predecir siempre la prevalencia da "
             "prevalencia × (1 − prevalencia).",
    "sens": "De los casos con pCR, fracción clasificada como pCR.",
    "esp": "De los casos sin pCR, fracción clasificada como no pCR.",
    "prec": "Valor predictivo positivo: de los clasificados como pCR, fracción que la alcanzó.",
    "vpn": "Valor predictivo negativo: de los clasificados como no pCR, fracción que no la alcanzó.",
}


def _con_umbral(m: dict, t: float) -> dict:
    """Matriz y métricas con el umbral t (positivo si prob >= t), desde el histograma
    de probabilidades por clase (intervalos de 0,01)."""
    h = m["histograma"]
    positivo = np.array(h["bordes"][:-1]) >= t - 1e-9
    n1, n0 = np.array(h["pCR_1"]), np.array(h["pCR_0"])
    vp, fn = int(n1[positivo].sum()), int(n1[~positivo].sum())
    fp, vn = int(n0[positivo].sum()), int(n0[~positivo].sum())
    return {"vp": vp, "fn": fn, "fp": fp, "vn": vn, "sens": vp / (vp + fn) if vp + fn else np.nan,
            "esp": vn / (vn + fp) if vn + fp else np.nan, "prec": vp / (vp + fp) if vp + fp else None,
            "vpn": vn / (vn + fn) if vn + fn else None}


def _exactas(c: dict) -> dict:
    mc = c["matriz_confusion"]
    return {"vp": mc["VP"], "fn": mc["FN"], "fp": mc["FP"], "vn": mc["VN"], "sens": c["sensibilidad"],
            "esp": c["especificidad"], "prec": c["precision"],
            "vpn": mc["VN"] / (mc["VN"] + mc["FN"]) if mc["VN"] + mc["FN"] else None}


def _hanley_mcneil(auc: float, n1: int, n0: int) -> tuple[float, float]:
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    return auc - 1.96 * se, auc + 1.96 * se


def _tiles(m: dict) -> None:
    c = st.columns(4, gap="small")
    comun.tile(c[0], "AUC", num(m["auc"]), help=AYUDA["auc"] + " " + mayuscula(m["ic95_metodo"]) + ".",
               delta=f"IC95 {num(m['auc_ic95'][0])}–{num(m['auc_ic95'][1])}")
    dif = m["auc"] - m["auc_solo_cohorte"]
    comun.tile(c[1], "Solo cohorte", num(m["auc_solo_cohorte"]),
               help=AYUDA["cohorte"] + f" Diferencia modelo − cohorte: IC95 {num(m['diferencia_ic95'][0])} a "
                    f"{num(m['diferencia_ic95'][1])}; si incluye el cero, no es concluyente.",
               delta=f"modelo {'+' if dif >= 0 else ''}{num(dif)}")
    comun.tile(c[2], "Average precision", num(m["average_precision"]), help=AYUDA["ap"],
               delta=f"azar {num(m['tasa_pCR'])}")
    comun.tile(c[3], "Brier", num(m["brier"]), help=AYUDA["brier"],
               delta=f"constante {num(m['tasa_pCR'] * (1 - m['tasa_pCR']))}")


def _seccion_umbral(m: dict, elegido: float, nombre_nivel: str, es_app: bool) -> None:
    exacto = _exactas(m["con_umbral"])
    opciones = sorted({round(k / 100, 2) for k in range(101)} | {elegido})
    t = st.select_slider(
        "Umbral", options=opciones, value=elegido, key=f"umbral_{nombre_nivel}",
        format_func=lambda v: f"{num(v, 4)} · {'el de la app' if es_app else 'elegido'}" if v == elegido else num(v, 2),
        help=f"Positivo (pCR) si la probabilidad ≥ umbral. Empieza en el elegido: mayor sensibilidad con "
             f"especificidad ≥ 0,85 sobre las probabilidades por {nombre_nivel}.")
    r = exacto if t == elegido else _con_umbral(m, t)
    actual = None if t == elegido else (1 - r["esp"], r["sens"])
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(key="panel_roc"):
        g.mostrar(g.figura_roc(m, (1 - exacto["esp"], exacto["sens"]), actual, nombre_nivel), key="roc")
    with c2, st.container(key="panel_pr"):
        g.mostrar(g.figura_pr(m, (exacto["sens"], exacto["prec"]),
                              None if t == elegido else (r["sens"], r["prec"]), nombre_nivel), key="pr")
    st.caption("Estrella: umbral elegido · punto cian: el del selector · discontinua: «solo cohorte» · punteada: azar.")

    izquierda, derecha = st.columns([1, 1.25], gap="medium")
    with izquierda, st.container(key="grande_matriz"):
        a, b = st.columns(2, gap="small")
        comun.tile(a, "VP · pCR acertada", entero(r["vp"]), help="Verdaderos positivos: con pCR y clasificados como pCR.")
        comun.tile(b, "FN · pCR no detectada", entero(r["fn"]), help="Falsos negativos: con pCR, clasificados como no pCR.")
        a, b = st.columns(2, gap="small")
        comun.tile(a, "FP · falsa pCR", entero(r["fp"]), help="Falsos positivos: sin pCR, clasificados como pCR.")
        comun.tile(b, "VN · no pCR acertada", entero(r["vn"]), help="Verdaderos negativos: sin pCR y clasificados así.")
    with derecha:
        g.mostrar(g.figura_histograma(m, elegido, t, nombre_nivel), key="histograma")
    c = st.columns(4, gap="small")
    ic = m["con_umbral"]["sensibilidad_ic95"]
    comun.tile(c[0], "Sensibilidad", num(r["sens"]), help=AYUDA["sens"],
               delta=f"IC95 {num(ic[0])}–{num(ic[1])}" if t == elegido
               else f"{'+' if r['sens'] >= exacto['sens'] else ''}{num(r['sens'] - exacto['sens'])} frente al elegido")
    comun.tile(c[1], "Especificidad", num(r["esp"]), help=AYUDA["esp"], delta="objetivo ≥ 0,85")
    comun.tile(c[2], "Precisión", "—" if r["prec"] is None else num(r["prec"]), help=AYUDA["prec"],
               delta=f"azar {num(m['tasa_pCR'])}")
    comun.tile(c[3], "VPN", "—" if r["vpn"] is None else num(r["vpn"]), help=AYUDA["vpn"],
               delta=f"azar {num(1 - m['tasa_pCR'])}")


def _seccion_final(fusion, es_app: bool) -> None:
    """Modelo final combinado: tiles, cascada con IC95, ROC de los cuatro modelos y,
    por corte, las métricas con su umbral."""
    vc = fusion.meta["validacion_cruzada"]
    n = vc["por_corte" if es_app else "por_paciente"]
    nombre_nivel = "corte" if es_app else "paciente"
    mo, d = n["modelos"], n["diferencias"]
    if fusion.provisional:
        comun.aviso_discreto("Modelo combinado PROVISIONAL: " + fusion.meta["nota_provisional"])
    c = st.columns(4, gap="small")
    ic = lambda k: f"IC95 {num(mo[k]['auc_ic95'][0])}–{num(mo[k]['auc_ic95'][1])}"  # noqa: E731
    comun.tile(c[0], "AUC combinado", num(mo["combinado"]["auc"]), delta=ic("combinado"),
               help=f"Modelo final: regresión logística con el subtipo HR/HER2, el volumen tumoral, la edad y la "
                    f"probabilidad de la CNN ({fusion.configuracion}). {AYUDA['auc']} {mayuscula(vc['ic95_metodo'])}.")
    dif = d["combinado_menos_clinico"]
    comun.tile(c[1], "Clínico", num(mo["clinico"]["auc"]),
               delta=f"combinado {'+' if dif['valor'] >= 0 else ''}{num(dif['valor'])}",
               help=f"Solo las variables clínicas, sin la imagen. Lo que añade la CNN: IC95 {num(dif['ic95'][0])} a "
                    f"{num(dif['ic95'][1])} (pareado); si incluye el cero, no es concluyente.")
    comun.tile(c[2], "Solo CNN", num(mo["cnn"]["auc"]), delta=ic("cnn"),
               help="La probabilidad out-of-fold de la CNN sola, recalibrada con una regresión logística en cada fold.")
    comun.tile(c[3], "Solo cohorte", num(mo["cohorte"]["auc"]), delta=ic("cohorte"), help=AYUDA["cohorte"])

    izquierda, derecha = st.columns([1, 1], gap="medium")
    with izquierda, st.container(key="panel_cascada_final"):
        g.mostrar(g.figura_cascada(n), key="cascada_final")
        st.caption("Cada barra, el AUC con su IC95; entre barras, la mejora y debajo su IC95 pareado (mismos "
                   "remuestreos).")
    with derecha, st.container(key="panel_roc_modelos"):
        g.mostrar(g.figura_roc_modelos(n, nombre_nivel), key="roc_modelos")
    if es_app:
        u = n["con_umbral"]
        mc = u["matriz_confusion"]
        c = st.columns(4, gap="small")
        comun.tile(c[0], "Umbral combinado", num(u["umbral"], 4), delta="especificidad ≥ 0,85",
                   help=f"{mayuscula(fusion.meta['umbral']['regla'])}. {mayuscula(fusion.meta['umbral']['nota'])}.")
        comun.tile(c[1], "Sensibilidad", num(u["sensibilidad"]),
                   delta=f"IC95 {num(u['sensibilidad_ic95'][0])}–{num(u['sensibilidad_ic95'][1])}",
                   help=AYUDA["sens"] + f" Con la misma especificidad, el azar daría {num(u['sensibilidad_azar'])}.")
        comun.tile(c[2], "Especificidad", num(u["especificidad"]), delta=f"{entero(mc['VN'])} VN · {entero(mc['FP'])} FP",
                   help=AYUDA["esp"])
        comun.tile(c[3], "Precisión", num(u["precision"]), delta=f"azar {num(u['precision_azar'])}",
                   help=AYUDA["prec"] + f" {entero(mc['VP'])} VP y {entero(mc['FN'])} FN.")
    with comun.mas_detalle("El modelo combinado por cohorte"):
        filas = [{"Cohorte": k, "Pacientes": v["pacientes"], "Cortes": v["cortes"], "AUC combinado": v["auc_combinado"],
                  "AUC clínico": v["auc_clinico"]} for k, v in vc["por_cohorte_corte"].items()]
        st.table(tabla_texto(pd.DataFrame(filas), {"AUC combinado": 3, "AUC clínico": 3}), hide_index=True)
        st.caption("Por corte, validación cruzada de 5 folds. Con pocas pCR por cohorte, son inestables.")


def _seccion_validacion_cruzada(meta: dict) -> None:
    d = ex.tabla()
    variante, fold = ex.variante_de(meta["ejecucion"])
    filas = d[d.variante == variante].sort_values("fold")
    if len(filas) != 5:
        st.info("No hay validación cruzada completa de esta configuración en experimentos/.")
        return
    folds = datos.resumen()["folds"]
    ic = [_hanley_mcneil(a, folds[str(int(f))]["pCR_1"], folds[str(int(f))]["pacientes"] - folds[str(int(f))]["pCR_1"])
          for f, a in zip(filas.fold, filas.auc_val)]
    g.mostrar(g.figura_cv(filas.fold.astype(int).tolist(), filas.auc_val.tolist(), ic,
                          filas.auc_solo_cohorte.tolist(), fold), key="cv")
    st.caption(f"Configuración {variante}: cinco modelos, cada uno validado en un quinto distinto de las pacientes. "
               f"AUC medio {num(filas.auc_val.mean())} (DT {num(filas.auc_val.std(ddof=1))}).",
               help="Cada AUC es el de la mejor época de su fold, elegida con la misma validación: algo optimista. "
                    "IC95 aproximado de Hanley-McNeil con las pacientes y pCR de cada fold."
                    + ("" if fold is not None else " El modelo exportado es esta configuración entrenada aparte."))


def _seccion_curvas(meta: dict) -> None:
    c = ex.curvas(meta["ejecucion"])
    if c is None:
        st.info("No están las curvas de esta ejecución en experimentos/.")
        return
    mejor = int(meta["entrenamiento"]["mejor_epoca"])
    inicio = ex.inicio_sobreajuste(c.perdida_val.to_numpy())
    g.mostrar(g.figura_curvas_aprendizaje(c, mejor, inicio), key="curvas")
    st.caption("La pérdida de entrenamiento baja sin parar y la de validación sube: el modelo memoriza. "
               + (f"Sobreajuste desde la época {inicio} (pérdida de validación > mínimo + 0,03 dos épocas seguidas)."
                  if inicio else ""))


def _sin_cero_negativo(v: float) -> str:
    return num(0.0 if abs(v) < 5e-4 else v)


def _ic(par) -> str:
    return "—" if par is None or par[0] is None else f"{num(par[0])}–{num(par[1])}"


def _cv_vs_test(test_nivel: dict, cv_nivel: dict) -> pd.DataFrame:
    filas = []
    for k in ("combinado", "clinico", "cnn", "cohorte"):
        t, c = test_nivel["modelos"][k], cv_nivel["modelos"].get(k, {})
        cu = c.get("con_umbral") or {}
        par = lambda a, b: f"{'—' if a is None else num(a)} / {'—' if b is None else num(b)}"  # noqa: E731
        filas.append({
            "Modelo": NOMBRE[k],
            "AUC CV (IC95)": f"{num(c['auc'])} ({_ic(c.get('auc_ic95'))})" if "auc" in c else "—",
            "AUC test (IC95)": f"{num(t['auc'])} ({_ic(t['auc_ic95'])})",
            "Test − CV": f"{'+' if t['auc'] >= c.get('auc', 0) else ''}{num(t['auc'] - c['auc'])}" if "auc" in c else "—",
            "AP CV / test": par(c.get("average_precision"), t["average_precision"]),
            "Brier CV / test": par(c.get("brier"), t["brier"]),
            "Sensibilidad CV / test": par(cu.get("sensibilidad"), t["sensibilidad"]),
            "Especificidad CV / test": par(cu.get("especificidad"), t["especificidad"]),
        })
    return pd.DataFrame(filas)


def _seccion_test(test: dict, fusion, paquete) -> None:
    """Test independiente: lo leído de experimentos/test/, sin recalcular nada."""
    p = test["niveles"]["paciente"]
    st.html('<div class="antetitulo" style="margin:.2rem 0 .2rem">Test independiente '
            f'({p["n"]} pacientes, evaluado una sola vez)</div>')
    fecha = (test.get("fecha") or "").replace("T", " ")[:16]
    st.caption(f"{p['n']} pacientes ({p['pCR_1']} con pCR) y {entero(test['niveles']['corte']['n'])} cortes de las "
               "mismas cohortes que train (I-SPY1, I-SPY2 y Duke), apartadas desde el principio. Modelos, coeficientes "
               f"y umbrales congelados con train; evaluado una sola vez el {fecha} (evaluar_test.py).")
    if not ti.es_el_paquete(test, paquete.sha256):
        st.warning("Estos resultados de test son de otro paquete del modelo: el SHA-256 de modelo.pt o de fusion.json "
                   "no coincide con el que carga la app.", icon=":material/warning:")
    nivel = st.segmented_control("Nivel del test", [NIVEL_TEST_PACIENTE, NIVEL_TEST_CORTE], default=NIVEL_TEST_PACIENTE,
                                 required=True, key="nivel_test", persist_state="page",
                                 help="Por paciente (métrica principal): la media de las probabilidades de sus cortes, "
                                      "con el umbral por paciente elegido en train. Por corte: cada corte, con el "
                                      "umbral por corte (el de la app). IC95 por bootstrap por paciente (2.000 "
                                      "remuestreos, los mismos para todos los modelos).")
    clave = "paciente" if nivel == NIVEL_TEST_PACIENTE else "corte"
    n = test["niveles"][clave]
    mo, d = n["modelos"], n["diferencias"]
    cv = (fusion.meta.get("validacion_cruzada") or {}).get(f"por_{clave}", {"modelos": {}}) if fusion else {"modelos": {}}

    c = st.columns(4, gap="small")
    comun.tile(c[0], "AUC combinado", num(mo["combinado"]["auc"]), delta=f"IC95 {_ic(mo['combinado']['auc_ic95'])}",
               help="Modelo final (clínico + CNN) en test. " + AYUDA["auc"])
    dif = d["combinado_menos_clinico"]
    comun.tile(c[1], "Clínico", num(mo["clinico"]["auc"]),
               delta=f"combinado {'+' if dif['valor'] >= 0 else ''}{num(dif['valor'])}",
               help=f"Solo subtipo, volumen y edad. Lo que añade la CNN en test: IC95 {num(dif['ic95'][0])} a "
                    f"{num(dif['ic95'][1])} (pareado); si incluye el cero, no es concluyente.")
    comun.tile(c[2], "Solo CNN", num(mo["cnn"]["auc"]), delta=f"IC95 {_ic(mo['cnn']['auc_ic95'])}",
               help="La probabilidad de la CNN sola, recalibrada con una regresión logística ajustada en train.")
    comun.tile(c[3], "Solo cohorte", num(mo["cohorte"]["auc"]), delta=f"IC95 {_ic(mo['cohorte']['auc_ic95'])}",
               help=AYUDA["cohorte"])

    izquierda, derecha = st.columns([1, 1], gap="medium")
    with izquierda, st.container(key="panel_cascada_test"):
        g.mostrar(g.figura_cascada(n, titulo=f"AUC en test por {clave}"),
                  key="cascada_test")
        st.caption("Cada barra, el AUC con su IC95; entre barras, la mejora y su IC95 pareado.")
    with derecha, st.container(key="panel_cv_test"):
        cv_auc = {k: v for k, v in cv["modelos"].items() if "auc" in v}
        if len(cv_auc) == 4:
            g.mostrar(g.figura_cv_test(mo, cv_auc).update_layout(
                title_text=f"AUC por {clave} · validación cruzada frente a test"), key="cv_test")
            st.caption("Hueco: validación cruzada de 5 folds en train. Relleno: test independiente.")

    izquierda, derecha = st.columns([1.25, 1], gap="medium")
    with izquierda, st.container(key="test_matriz"):
        u = mo["combinado"]
        st.markdown(f"**Combinado con el umbral congelado** ({num(u['umbral'], 4)}, elegido en train: especificidad "
                    "≥ 0,85)")
        a, b = st.columns(2, gap="small")
        comun.tile(a, "VP · pCR acertada", entero(u["VP"]), help="Con pCR y clasificados como pCR.")
        comun.tile(b, "FN · pCR no detectada", entero(u["FN"]), help="Con pCR, clasificados como no pCR.")
        a, b = st.columns(2, gap="small")
        comun.tile(a, "FP · falsa pCR", entero(u["FP"]), help="Sin pCR, clasificados como pCR.")
        comun.tile(b, "VN · no pCR acertada", entero(u["VN"]), help="Sin pCR y clasificados así.")
        c = st.columns(4, gap="small")
        for col, (k, et, ayuda) in zip(c, (("sensibilidad", "Sensib.", "sens"), ("especificidad", "Especif.", "esp"),
                                          ("precision", "Precisión", "prec"), ("vpn", "VPN", "vpn"))):
            comun.tile(col, et, "—" if u[k] is None else num(u[k]), delta=_ic(u[f"{k}_ic95"]),
                       help=AYUDA[ayuda] + " Debajo, su IC95 (bootstrap por paciente).")
    with derecha:
        st.image(str(test["figuras"]["roc"]), width="stretch",
                 caption="Curvas ROC en test de los cuatro modelos (izquierda, por paciente; derecha, por corte).")
    izquierda, derecha = st.columns([1, 1.25], gap="medium")
    with izquierda:
        st.image(str(test["figuras"]["calibracion"]), width="stretch",
                 caption="Calibración en test (10 cuantiles de la probabilidad). La horizontal es la tasa de pCR.")
    with derecha:
        nombres = {"clinico_menos_cohorte": "Clínico − solo cohorte", "combinado_menos_clinico": "Combinado − clínico",
                   "combinado_menos_cohorte": "Combinado − solo cohorte", "combinado_menos_cnn": "Combinado − solo CNN"}
        st.markdown("**Diferencias de AUC en test** (pareadas: los mismos remuestreos para todos los modelos)")
        st.table(pd.DataFrame([{"Diferencia": nombres.get(k, k), "Valor": f"{'+' if v['valor'] >= 0 else ''}"
                                f"{num(v['valor'])}", "IC95": f"{_sin_cero_negativo(v['ic95'][0])} a "
                                f"{_sin_cero_negativo(v['ic95'][1])}"}
                               for k, v in d.items()]), hide_index=True)
        st.caption("Si el IC95 incluye el cero, la diferencia no es concluyente.")
    st.markdown("**CV vs test**")
    st.table(_cv_vs_test(n, cv), hide_index=True)
    st.caption("CV: validación cruzada de 5 folds en train (fusion.json); su sensibilidad es optimista porque el "
               "umbral se eligió con esas mismas predicciones. Test: con los umbrales congelados.")

    with comun.mas_detalle("Matrices de confusión de los cuatro modelos en test"):
        st.image(str(test["figuras"]["confusion"]), width="stretch")
        filas = [{"Modelo": NOMBRE[k], "Umbral": num(m["umbral"], 4), "VP": m["VP"], "FN": m["FN"], "FP": m["FP"],
                  "VN": m["VN"], **{et: ("—" if m[k2] is None else f"{num(m[k2])} ({_ic(m[f'{k2}_ic95'])})")
                                    for k2, et in (("sensibilidad", "Sensibilidad"), ("especificidad", "Especificidad"),
                                                   ("precision", "Precisión"), ("vpn", "VPN"))}}
                 for k, m in mo.items()]
        st.table(pd.DataFrame(filas), hide_index=True)
    with comun.mas_detalle("AUC por cohorte en test"):
        filas = [{"Cohorte": c, "Pacientes": v["n"] if clave == "paciente" else "", "Con pCR": v["pacientes_pCR"],
                  **{NOMBRE[k]: f"{num(v[k]['auc'])} ({_ic(v[k]['auc_ic95'])})" for k in ("clinico", "cnn", "combinado")
                     if k in v}} for c, v in n["por_cohorte"].items()]
        tabla = pd.DataFrame(filas)
        if clave == "corte":
            tabla = tabla.drop(columns="Pacientes")
        st.table(tabla, hide_index=True)
        st.caption("Con pocas pCR por cohorte, los intervalos son muy anchos. «Solo cohorte» es "
                   "constante dentro de cada cohorte.")


def _contenido() -> None:
    paquete = comun.modelo_o_parar()
    fusion = comun.fusion_o_none()
    meta = paquete.meta
    mc = meta["metricas_validacion"]
    if fusion is not None:
        vc = fusion.meta["validacion_cruzada"]
        sub = (f"Modelo final (clínico + CNN): validación cruzada de 5 folds, {entero(vc['por_paciente']['n'])} "
               f"pacientes y {entero(vc['por_corte']['n'])} cortes de train"
               + (". Con estas cifras se eligió el modelo; el test no intervino." if ti.cargar() is not None
                  else ". El conjunto test no se ha usado."))
    else:
        sub = (f"Fold {meta['entrenamiento']['fold_validacion']} · {mc['pacientes']} pacientes ({mc['pCR_1']} con "
               f"pCR) · {entero(mc['cortes'])} cortes"
               + (". El test no intervino." if ti.cargar() is not None else ". El conjunto test no se ha usado."))
    test = ti.cargar()
    if test is not None:
        comun.encabezado("Rendimiento", "Test independiente y validación cruzada",
                         "Arriba, el test independiente: la estimación del rendimiento del modelo final. Debajo, la "
                         "validación cruzada en train con la que se eligió el modelo.")
        _seccion_test(test, fusion, paquete)
        st.html('<div class="antetitulo" style="margin:1.6rem 0 .2rem;border-top:1px solid #1F2A3C;padding-top:1rem">'
                'Validación cruzada en train (selección del modelo)</div>')
        st.caption(sub)
    else:
        comun.encabezado("Rendimiento", "Validación interna", sub)
    nivel = st.segmented_control("Nivel", [NIVEL_CORTE, NIVEL_PACIENTE], default=NIVEL_CORTE, required=True,
                                 key="nivel", persist_state="page",
                                 help="Por corte: cada corte de validación por separado, como lo analiza la app (el "
                                      "umbral de la app se eligió así). Por paciente: la media de sus cortes, con su "
                                      "propio umbral por el mismo criterio. IC95 por bootstrap por paciente.")
    es_app = nivel == NIVEL_CORTE
    m = mc if es_app else mc["por_paciente"]
    elegido = paquete.umbral if es_app else m["umbral"]
    nombre_nivel = "corte" if es_app else "paciente"
    if fusion is not None:
        st.html('<div class="antetitulo" style="margin:.2rem 0 .2rem">Modelo final · clínico + CNN</div>')
        _seccion_final(fusion, es_app)
        st.html('<div class="antetitulo" style="margin:1.2rem 0 .2rem">La red sola · CNN</div>')
        st.caption(f"{meta['ejecucion']}: fold {meta['entrenamiento']['fold_validacion']}, {mc['pacientes']} pacientes "
                   f"({mc['pCR_1']} con pCR), {entero(mc['cortes'])} cortes de validación.")
    _tiles(m)
    _seccion_umbral(m, elegido, nombre_nivel, es_app)

    c1, c2 = st.columns([1.25, 1], gap="medium")
    with c1, st.container(key="panel_cv"):
        _seccion_validacion_cruzada(meta)
    with c2, st.container(key="panel_calibracion"):
        g.mostrar(g.figura_calibracion(m, nombre_nivel), key="calibracion")
        st.caption(f"Probabilidad media {num(m['prob_media'])} frente a una tasa real de {num(m['tasa_pCR'])}: la "
                   "probabilidad ordena casos, no es un riesgo individual.")

    with comun.mas_detalle("Curvas de aprendizaje del modelo exportado"):
        _seccion_curvas(meta)
    with comun.mas_detalle("AUC por cohorte"):
        filas = [{"Cohorte": c, "Pacientes": v["pacientes"], "Con pCR": v["pCR_1"],
                  **({"Cortes": v["cortes"]} if "cortes" in v else {}), "AUC": v["auc"]}
                 for c, v in m["por_cohorte"].items()]
        st.table(tabla_texto(pd.DataFrame(filas), {"AUC": 3}), hide_index=True)
        st.caption("Con tan pocas pCR por cohorte (sobre todo en I-SPY1) estos AUC son muy inestables.")


comun.ejecutar(_contenido)
