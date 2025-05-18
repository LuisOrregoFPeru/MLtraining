import streamlit as st
import datetime
import requests
import xml.etree.ElementTree as ET
import textwrap
import openai
import os

st.set_page_config(page_title="Generador de Introducción de Tesis", layout="wide")
st.title("📘 Generador de Introducción de Proyecto de Tesis")

st.markdown("""
Esta aplicación te ayudará a generar automáticamente la sección de **Introducción** de tu tesis, con una estructura académica completa y organizada a partir de un **título** u **objetivo general**. Los textos se redactan en **prosa**, en **tercera persona**, en **tiempo futuro del modo indicativo**, con párrafos de máximo 10 líneas, y con una extensión aproximada a **9 páginas A4**.
""")

# Entradas del usuario
titulo = st.text_input("🎓 Título del proyecto de tesis")
objetivo_general = st.text_area("🎯 Objetivo general de la investigación")
consulta_pubmed = st.text_input("🔎 Palabra clave para búsqueda de antecedentes en PubMed")

# Configurar clave de API de OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# Función para generar texto con ChatGPT
def generar_introduccion_con_gpt(titulo, objetivo):
    prompt = f"""
Redacta en prosa, sin subtítulos, en tercera persona y tiempo futuro del modo indicativo, una introducción para un proyecto de tesis titulado: "{titulo}". 
Incluye los siguientes elementos:

1. La importancia del tema a investigar y la plausibilidad de la pregunta.
2. El impacto del problema: frecuencia, morbimortalidad a nivel mundial, América Latina y Perú.
3. Los vacíos en la literatura: falta de estudios, deficiencias metodológicas, contextos no generalizables.
4. Vinculación con al menos un Objetivo de Desarrollo Sostenible (ODS).
5. Formulación del problema en modo interrogativo.
6. Justificación teórica, práctica, metodológica y social.
7. Mención al objetivo general: "{objetivo}".
8. Redacción en párrafos de máximo 10 líneas cada uno.

Extensión aproximada: 1200 palabras.
"""

    respuesta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000
    )
    return respuesta.choices[0].message["content"]

# Función para parafrasear abstract
@st.cache_data
def obtener_antecedentes(termino):
    anio_inicio = datetime.date.today().year - 5
    url_search = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={termino}&retmax=10&retmode=json&mindate={anio_inicio}&datetype=pdat"
    )
    ids = requests.get(url_search).json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    url_fetch = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&id={','.join(ids)}&retmode=xml"
    )
    root = ET.fromstring(requests.get(url_fetch).content)
    antecedentes = []

    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="")
        abstract_elem = article.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else ""
        author = article.findtext(".//Author/LastName", default="Autor")
        initial = article.findtext(".//Author/Initials", default="N")
        year = article.findtext(".//PubDate/Year") or article.findtext(".//DateCreated/Year") or "s.f."
        resumen = textwrap.shorten(abstract.replace("\n", " ").strip(), width=850, placeholder="...")
        if resumen:
            antecedentes.append(f"{author}, {initial}. et al. ({year}) señala que {resumen.lower()}")
    return antecedentes

# Botón para generar la introducción
generar = st.button("📝 Generar Introducción")

if generar:
    if not titulo or not objetivo_general:
        st.warning("Por favor completa el título y el objetivo general antes de generar la introducción.")
    else:
        with st.spinner("✍️ Generando introducción con ChatGPT..."):
            introduccion = generar_introduccion_con_gpt(titulo, objetivo_general)
            st.subheader("🧾 Introducción Generada")
            st.markdown(introduccion)

        st.subheader("📚 Antecedentes (PubMed, últimos 5 años)")
        antecedentes = obtener_antecedentes(consulta_pubmed)
        if antecedentes:
            for ant in antecedentes:
                st.markdown(f"- {ant}")
        else:
            st.warning("No se encontraron antecedentes relevantes en PubMed para esa búsqueda.")

        st.subheader("🧠 Bases teóricas y enfoques conceptuales")
        st.markdown("Se adoptarán teorías relevantes para comprender la dinámica de las variables involucradas en el fenómeno investigado. Los enfoques conceptuales servirán como marco interpretativo para guiar la operacionalización de conceptos clave, categorías analíticas y relaciones hipotéticas.")

        st.subheader("🔬 Hipótesis de investigación")
        st.markdown(f"- **Hipótesis de investigación:** Se planteará que el objetivo general propuesto (\"{objetivo_general}\") tendrá una relación significativa con las variables de estudio, bajo condiciones previamente establecidas.")
        st.markdown("- **Hipótesis estadísticas:** Se formularán contrastes de hipótesis nula (H0) y alternativa (H1) según el diseño metodológico, nivel de medición y tipo de análisis previsto.")
