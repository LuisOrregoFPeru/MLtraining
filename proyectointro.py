import streamlit as st
import datetime
import requests
import xml.etree.ElementTree as ET
import textwrap
import os

# ────────────────────────────────────────────────────────────────────────────────
# Intentamos importar openai; si no está, mostramos un mensaje claro y detenemos
# la app. En Streamlit Cloud basta con añadir "openai" a requirements.txt.
# ────────────────────────────────────────────────────────────────────────────────
try:
    import openai
except ModuleNotFoundError:
    st.error(
        "❌ El paquete `openai` no está instalado.\n\n"
        "Agrega una línea `openai` (sin comillas) en tu `requirements.txt` y vuelve a desplegar."
    )
    st.stop()

# Configuración de página
st.set_page_config(page_title="Generador de Introducción de Tesis", layout="wide")
st.title("📘 Generador de Introducción de Proyecto de Tesis")

st.markdown(
    """
Esta aplicación genera automáticamente la **Introducción** de una tesis a partir de un título u objetivo general, 
utilizando la API de ChatGPT, antecedentes de PubMed y una estructura académica prediseñada.
"""
)

# Entradas del usuario
titulo = st.text_input("🎓 Título del proyecto de tesis")
objetivo_general = st.text_area("🎯 Objetivo general de la investigación")
consulta_pubmed = st.text_input("🔎 Palabra clave para búsqueda de antecedentes en PubMed")

# Configurar clave de API
openai.api_key = (
    st.secrets.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

if not openai.api_key:
    st.warning(
        "⚠️ No se ha encontrado `OPENAI_API_KEY`. Añádela en `.streamlit/secrets.toml` "
        "o en *Settings → Secrets* de Streamlit Cloud."
    )

# Función para crear introducción con ChatGPT
def generar_introduccion(titulo: str, objetivo: str) -> str:
    prompt = f"""Redacta en prosa, sin subtítulos, en tercera persona y tiempo futuro del modo indicativo, \
una introducción de tesis titulada \"{titulo}\". Incorpora: importancia, plausibilidad, impacto global/LatAm/Perú, vacíos, ODS,\nproblema interrogativo, justificaciones teórica-práctica-metodológica-social y referencia al objetivo \"{objetivo}\".\nMáximo 10 líneas por párrafo. Extensión ≈ 1200 palabras."""
    respuesta = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000,
    )
    return respuesta.choices[0].message.content

# Función para extraer y parafrasear antecedentes
def obtener_antecedentes(termino: str):
    anio_inicio = datetime.date.today().year - 5
    url_search = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={termino}&retmax=10&retmode=json&mindate={anio_inicio}&datetype=pdat"
    )
    ids = requests.get(url_search).json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    url_fetch = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&id={','.join(ids)}&retmode=xml"
    )
    root = ET.fromstring(requests.get(url_fetch).content)
    antecedentes = []

    for art in root.findall(".//PubmedArticle"):
        title = art.findtext(".//ArticleTitle", default="")
        abstract = art.findtext(".//Abstract/AbstractText", default="")
        autor = art.findtext(".//Author/LastName", default="Autor")
        inicial = art.findtext(".//Author/Initials", default="N")
        year = (
            art.findtext(".//PubDate/Year")
            or art.findtext(".//DateCreated/Year")
            or "s.f."
        )
        resumen = textwrap.shorten(
            abstract.replace("\n", " ").strip(), width=850, placeholder="..."
        )
        if resumen:
            antecedentes.append(
                f"{autor}, {inicial}. et al. ({year}) señala que {resumen.lower()}"
            )
    return antecedentes

# Botón para generar
generar = st.button("📝 Generar Introducción")

if generar:
    if not (titulo and objetivo_general):
        st.warning("Completa título y objetivo general.")
        st.stop()

    with st.spinner("Generando introducción con ChatGPT..."):
        try:
            intro = generar_introduccion(titulo, objetivo_general)
        except Exception as e:
            st.error(f"Error al llamar a OpenAI: {e}")
            st.stop()

    st.subheader("🧾 Introducción Generada")
    st.markdown(intro)

    st.subheader("📚 Antecedentes (PubMed, últimos 5 años)")
    antecedentes = obtener_antecedentes(consulta_pubmed) if consulta_pubmed else []
    if antecedentes:
        for ant in antecedentes:
            st.markdown(f"- {ant}")
    else:
        st.info("Introduce una palabra clave para obtener antecedentes.")

    st.subheader("🧠 Bases teóricas y enfoques conceptuales")
    st.markdown(
        "Se adoptarán teorías pertinentes para explicar la interacción de las variables, "
        "y se delimitarán enfoques conceptuales que orientarán la operacionalización y el análisis."
    )

    st.subheader("🔬 Hipótesis")
    st.markdown(
        f"- **Hipótesis de investigación:** El objetivo general \"{objetivo_general}\" mostrará "
        "una relación significativa con las variables exploradas."
    )
    st.markdown(
        "- **Hipótesis estadísticas:** Se contrastará H0 (no hay efecto) frente a H1 (existe efecto) según el diseño."
    )
