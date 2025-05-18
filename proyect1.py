import streamlit as st
import textwrap
from docx import Document
from io import BytesIO

"""---------------------------------------------------------
Generador de Introducciones de Tesis (modo simple, sin LLM)
-----------------------------------------------------------
Este script crea:
1. Introducción ≥ 9 páginas A4 siguiendo la estructura solicitada.
2. Bases teóricas vinculadas a la pregunta de investigación.
3. Hipótesis de investigación y estadísticas.

No usa API externos; el texto se construye con plantillas para demostración.
"""

# --------- Utilidades ---------

def wrap(text: str) -> str:
    """Ajusta a 100 caracteres para evitar líneas largas en Streamlit."""
    return textwrap.fill(text, width=100)

# --------- Generadores simples ---------

def generate_introduction(title: str, objective: str, word_target: int = 4500) -> str:
    sections = [
        # 1-2 párrafos sobre importancia y plausibilidad
        "La investigación abordará un fenómeno de gran relevancia para la salud pública y la economía, cuyo análisis resultará imprescindible para comprender las implicancias sanitarias y sociales derivadas. Con ello se demostrará que la pregunta científica planteada será verosímil y necesaria de resolver en el escenario contemporáneo.",
        "Asimismo, se evidenciará que la temática elegida responderá a una necesidad real de conocimiento, al mismo tiempo que permitirá proyectar intervenciones efectivas orientadas a la mejora continua de la calidad de vida de la población.",
        # 1-2 párrafos impacto global-regional-nacional
        "Se observará que la frecuencia del problema y la morbimortalidad asociada superarán los márgenes aceptables en numerosos países. A escala mundial, las cifras indicarán una tendencia ascendente que ameritará atención prioritaria por los organismos internacionales.",
        "En América Latina y, de forma particular, en el Perú, se documentarán indicadores que confirmarán la presencia de brechas significativas en la atención y el control del fenómeno, lo cual repercutirá negativamente en los sistemas sanitarios y en la economía de los hogares.",
        # 1-2 párrafos vacíos de literatura
        "Los hallazgos bibliográficos mostrarán ausencia de estudios rigurosos en contextos comparables, además de limitaciones metodológicas que reducirán la validez externa de los resultados previos.",
        "Con frecuencia, se identificarán investigaciones focalizadas en poblaciones distintas o con diseños no generalizables; esta situación justificará la necesidad de la presente propuesta académica.",
        # 1 párrafo ODS
        "La indagación se alineará con el Objetivo de Desarrollo Sostenible 3, meta 3.8, orientada a garantizar una vida sana y promover el bienestar para todos en todas las edades, aportando evidencia para políticas públicas inclusivas y sostenibles.",
        # 1 párrafo pregunta investig.
        f"¿Hasta qué punto el fenómeno descrito se relacionará con las variables seleccionadas según el objetivo general planteado?",
        # 1 párrafo justificación teórica
        "Teóricamente, la investigación ampliará el marco conceptual vigente, integrando modelos interdisciplinarios que explicarán la interacción compleja entre los determinantes biológicos, económicos y sociales del problema.",
        # 1 párrafo justificación práctica
        "En el aspecto práctico, los resultados facilitarán la toma de decisiones basadas en evidencia, optimizando intervenciones y programas que puedan implementarse en entornos clínicos y comunitarios.",
        # 1 párrafo justificación metodológica
        "Metodológicamente, se empleará un diseño robusto, con técnicas analíticas avanzadas que garantizarán la confiabilidad y validez interna de los hallazgos, lo que redundará en su aplicabilidad científica.",
        # 1 párrafo justificación social
        "Desde la perspectiva social, la generación de conocimiento contribuirá a reducir inequidades y a mejorar la calidad de vida de los grupos poblacionales afectados, favoreciendo la cohesión y el desarrollo sostenible de la comunidad."  
    ]
    intro = "\n\n".join(wrap(p) for p in sections)
    # Relleno aproximado para alcanzar ~4500 palabras (opcional)
    placeholder_paragraph = wrap("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40)
    while len(intro.split()) < word_target:
        intro += "\n\n" + placeholder_paragraph
    return intro


def generate_theoretical_bases(title: str, objective: str) -> str:
    text = (
        "Se establecerá un andamiaje teórico que articulará los conceptos clave inherentes al objetivo general: variable independiente, variable dependiente y factores de confusión. "
        "Se revisarán teorías como el Modelo Socio-Ecológico y la Economía de la Salud, enfatizando la forma en que dichas perspectivas explicarán la causalidad entre los determinantes estudiados y los resultados observados. "
        "Esta síntesis conceptual permitirá construir hipótesis consistentes y fundamentar la elección de los indicadores operativos que habrán de medirse en el estudio."  
    )
    return wrap(text)


def generate_hypotheses(objective: str) -> str:
    hip_inv = (
        "Hipótesis de investigación: Se postulará que la variable independiente ejercerá un efecto significativo y positivo sobre la variable dependiente, luego de controlar los factores de confusión predefinidos."  
    )
    hip_est = (
        "Hipótesis nula (H0): β1 = 0 — no existirá asociación entre la variable independiente y la dependiente.\n"
        "Hipótesis alternativa (H1): β1 ≠ 0 — existirá una asociación estadísticamente significativa entre ambas variables."  
    )
    return wrap(hip_inv) + "\n\n" + wrap(hip_est)


def build_docx(intro: str, bases: str, hyps: str) -> bytes:
    doc = Document()
    doc.add_heading("Proyecto de Tesis – Secciones Generadas", level=1)
    doc.add_heading("Introducción", level=2)
    for p in intro.split("\n"):
        doc.add_paragraph(p)
    doc.add_page_break()

    doc.add_heading("Bases Teóricas", level=2)
    for p in bases.split("\n"):
        doc.add_paragraph(p)
    doc.add_page_break()

    doc.add_heading("Hipótesis", level=2)
    for p in hyps.split("\n"):
        doc.add_paragraph(p)

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

# --------- Interfaz Streamlit ---------

st.set_page_config(page_title="Generador de Introducciones de Tesis", layout="wide")
st.title("📝 Generador Automático de Introducciones de Tesis (modo simple)")

with st.sidebar:
    st.header("Datos básicos")
    title_input = st.text_input("Título de la Investigación")
    objective_input = st.text_area("Objetivo General")
    generar = st.button("Generar Secciones")

if generar:
    st.info("Modo simple activado — se generan textos de demostración sin LLM.")

    intro_text = generate_introduction(title_input, objective_input)
    st.subheader("Introducción")
    st.markdown(intro_text)

    bases_text = generate_theoretical_bases(title_input, objective_input)
    st.subheader("Bases Teóricas")
    st.markdown(bases_text)

    hyps_text = generate_hypotheses(objective_input)
    st.subheader("Hipótesis")
    st.markdown(hyps_text)

    docx_bytes = build_docx(intro_text, bases_text, hyps_text)
    st.download_button("Descargar DOCX", data=docx_bytes, file_name="secciones_tesis.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
