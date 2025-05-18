import streamlit as st
import datetime
import requests
import xml.etree.ElementTree as ET

st.set_page_config(page_title="Asistente de Tesis Automático", layout="wide")
st.title("🤖 Generador Automático de Introducción de Proyecto de Tesis")

st.markdown("""
Esta aplicación web genera automáticamente la sección de **Introducción** de un proyecto de tesis a partir de un **título** o **objetivo general** proporcionado por el usuario. La redacción se realiza en **prosa**, en **tercera persona** y en **tiempo futuro del modo indicativo**, siguiendo una estructura académica rigurosa.
""")

# Entradas del usuario
titulo = st.text_input("🎓 Ingresa el título del proyecto de tesis (opcional)")
objetivo_general = st.text_area("🎯 Ingresa el objetivo general de la investigación")
consulta_pubmed = st.text_input("🔎 Palabra clave para búsqueda de antecedentes en PubMed")

# Buscar artículos recientes en PubMed
@st.cache_data
def buscar_antecedentes_pubmed(termino):
    fecha_actual = datetime.date.today()
    anio_actual = fecha_actual.year
    anio_inicio = anio_actual - 5
    url_search = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={termino}&retmax=5&retmode=json&mindate={anio_inicio}&datetype=pdat"
    )
    r = requests.get(url_search)
    if r.status_code != 200:
        return []
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    url_fetch = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&id={','.join(ids)}&retmode=xml"
    )
    r = requests.get(url_fetch)
    root = ET.fromstring(r.content)
    abstracts = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="Título no disponible")
        abstract_elem = article.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else "Resumen no disponible"
        abstracts.append(f"**{title}**: {abstract}")
    return abstracts

if st.button("📝 Generar Introducción"):
    st.subheader("🧾 Introducción Generada")
    st.markdown(f"A partir del análisis del objetivo general '{objetivo_general}', se desarrollará una investigación que abordará una problemática actual de relevancia científica y social.")

    st.markdown("En primer lugar, se expondrá de manera general la realidad problemática. Se describirá su impacto en términos de frecuencia y morbimortalidad asociada, considerando datos relevantes a nivel mundial, regional (América Latina) y nacional (Perú).")

    st.markdown("A continuación, se presentarán antecedentes científicos obtenidos de estudios primarios recientes, preferentemente publicados en PubMed durante los últimos cinco años. Estos hallazgos sustentarán la plausibilidad científica del problema abordado.")

    antecedentes = buscar_antecedentes_pubmed(consulta_pubmed)
    if antecedentes:
        st.markdown("**Antecedentes relevantes:**")
        for a in antecedentes:
            st.markdown(f"- {a}")
    else:
        st.warning("No se encontraron antecedentes recientes con esa palabra clave.")

    st.markdown("Posteriormente, se identificarán vacíos en la literatura existente, como la escasez de investigaciones pertinentes, estudios limitados a contextos no generalizables o deficiencias metodológicas.")

    st.markdown("El problema de investigación se formulará en modo interrogativo, considerando la relación directa con los objetivos propuestos y el marco conceptual de la investigación.")

    st.markdown("La justificación del estudio se estructurará en cuatro dimensiones: teórica, práctica, metodológica y social, lo que permitirá sustentar la relevancia y viabilidad del estudio.")

    st.markdown("Asimismo, se discutirá la utilidad potencial de responder a la pregunta de investigación, subrayando la importancia del tema en el contexto académico y su contribución a los Objetivos de Desarrollo Sostenible (ODS).")

    st.markdown(f"Se definirá formalmente el siguiente objetivo general: {objetivo_general}. Además, se derivarán objetivos específicos que orientarán el desarrollo metodológico del estudio.")

    st.markdown("Finalmente, se desarrollará un análisis teórico-conceptual que incluirá las principales teorías relacionadas con la problemática abordada, así como los enfoques conceptuales vinculados a las variables o categorías en estudio. También se indicará de qué manera este análisis contribuirá a superar las limitaciones identificadas en la literatura.")
