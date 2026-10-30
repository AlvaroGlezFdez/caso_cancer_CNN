"""Núcleo de la app: todo lo que no depende de Streamlit.

    rutas       carpetas del repositorio y acceso a preprocesado.py y models/
    paquete     carga y verificación del paquete del modelo (modelo_app/)
    validacion  comprobación de los 3 PNG de un corte, con informe paso a paso
    inferencia  validación -> preprocesado.preprocesar -> modelo -> probabilidad
    variables   variables de la paciente (CSV o formulario), validadas paso a paso
    fusion      modelo final: variables clínicas (clinica.py) + CNN -> probabilidad combinada
    ejemplos    los 3 cortes de ejemplo (app/ejemplos/)

La interfaz (streamlit_app.py y paginas/) solo llama a estas funciones, y la
prueba automática (prueba_app.py, en la raíz) usa exactamente las mismas.
"""
