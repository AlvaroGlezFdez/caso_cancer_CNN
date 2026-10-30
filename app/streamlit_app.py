"""App web del proyecto (Streamlit). Desde la raíz del repositorio:

    streamlit run app/streamlit_app.py

La lógica (validación, preprocesado con preprocesado.py, inferencia, Grad-CAM…)
está en nucleo/; aquí y en paginas/ solo está la interfaz (estilo propio en
estaticos/estilo.css, plantilla de plotly en paginas/graficos.py). El modelo es el
paquete de modelo_app/ (exportar_modelo.py). La página por defecto es la portada
(Inicio). Con ?ejemplo=ejemplo_2 en la URL, la página por defecto pasa a ser el
análisis, que abre ese ejemplo; el análisis vive también en /analisis.
"""

import os
import sys
from pathlib import Path

import streamlit as st

from nucleo import rutas  # noqa: F401  (añade la raíz del repositorio al sys.path)

RAIZ_REPO = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner=False)
def _fechas_modulos() -> dict:
    """Módulo del proyecto -> fecha de su fichero cuando se importó (dura lo que el proceso)."""
    return {}


def _modulos_al_dia() -> None:
    """Tras un despliegue, el proceso de Streamlit puede seguir con módulos del proyecto
    importados en su versión anterior (nucleo/, paginas/, preprocesado.py…), mientras las
    páginas ya ejecutan el código nuevo: una función nueva no existe y falla la sección. Si
    algún fichero cambió desde que se importó, se descartan todos los módulos del proyecto
    (se importan de nuevo, juntos) y se vacían las cachés (también el estilo)."""
    fechas = _fechas_modulos()
    entorno = RAIZ_REPO / ".venv"
    propios = {}
    for nombre, modulo in list(sys.modules.items()):
        fichero = getattr(modulo, "__file__", None)
        if not fichero:
            continue
        ruta = Path(fichero).resolve()
        if ruta.is_relative_to(RAIZ_REPO) and not ruta.is_relative_to(entorno) and ruta.exists():
            propios[nombre] = os.path.getmtime(ruta)
    if any(fechas.setdefault(n, t) != t for n, t in propios.items()):
        for nombre in propios:          # también los paquetes: si no, «from nucleo import x» daría el viejo
            sys.modules.pop(nombre, None)
        st.cache_data.clear()
        st.cache_resource.clear()          # también vacía las fechas: se apuntan de nuevo


_modulos_al_dia()
from paginas import comun  # noqa: E402  (después de comprobar que los módulos están al día)

st.set_page_config(page_title="Estación pCR · DCE-MRI", page_icon=":material/radiology:", layout="wide",
                   initial_sidebar_state="expanded")

# ?ejemplo= en la URL abre directamente el análisis con ese ejemplo (enlaces de la documentación).
con_ejemplo = "ejemplo" in st.query_params
navegacion = st.navigation({
    "Herramienta": [
        st.Page("paginas/inicio.py", title="Inicio", icon=":material/home:", url_path="inicio",
                default=not con_ejemplo),
        st.Page("paginas/analisis.py", title="Análisis de un caso", icon=":material/biotech:", url_path="analisis",
                default=con_ejemplo),
    ],
    "Modelo y evidencia": [
        st.Page("paginas/modelo.py", title="El modelo", icon=":material/account_tree:", url_path="modelo"),
        st.Page("paginas/rendimiento.py", title="Rendimiento", icon=":material/monitoring:", url_path="rendimiento"),
        st.Page("paginas/datos.py", title="Datos", icon=":material/database:", url_path="datos"),
        st.Page("paginas/experimentos.py", title="Experimentos y decisiones", icon=":material/timeline:",
                url_path="experimentos"),
    ],
    "Uso responsable": [
        st.Page("paginas/limitaciones.py", title="Limitaciones y ética", icon=":material/gavel:",
                url_path="limitaciones"),
    ],
})
comun.marco()
navegacion.run()
comun.pie()
