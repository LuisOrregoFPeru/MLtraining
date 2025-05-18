import streamlit as st
import datetime
import requests
import xml.etree.ElementTree as ET
import textwrap

st.set_page_config(page_title="Generador de Introducción de Tesis", layout="wide")
st.title("📘 Generador de Introducción de Proyecto de Tesis")

st.markdown("""
Esta aplicación te ayudará a generar automáticamente la sección de **Introducción** de tu tesis, con una estructura académica completa y organizada a partir de un **título** u **objetivo general**. Los textos se redactan en **prosa**, en **tercera persona**, en **tiempo futuro del modo indicativo**, con párrafos de máximo 10 líneas, y con una extensión aproximada a **9 páginas A4**.
""")

# Entradas del usuario
titulo = st.text_input("🎓 Título del proyecto de tesis")
objetivo_general = st.text_area("🎯 Objetivo general de la investigación")
consulta_pubmed = st.text_input("🔎 Palabra clave para búsqueda de antecedentes en PubMed")

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
    st.subheader("🧾 Introducción Generada")
    st.markdown(f"**Título del proyecto:** {titulo}")
    
    st.markdown("**Realidad problemática e importancia del tema**")
    st.markdown("El presente estudio abordará una problemática relevante en el contexto actual, cuya importancia se fundamentará en la necesidad de generar evidencia científica sobre un fenómeno con implicancias en la salud pública, el bienestar social o el desarrollo económico. La pregunta de investigación será formulada considerando la plausibilidad científica, el vacío empírico existente y el interés académico sobre el tema.")

    st.markdown("**Impacto del problema en cifras**")
    st.markdown("Este fenómeno presentará un impacto significativo en términos de frecuencia, carga de enfermedad y morbimortalidad. A nivel global, se observarán tendencias que justifican su estudio. En América Latina y particularmente en el Perú, los indicadores revelarán un crecimiento sostenido del problema, acompañado de brechas en acceso a servicios, políticas de intervención o respuesta institucional.")

    st.markdown("**Vacíos en la literatura científica**")
    st.markdown("Los estudios previos disponibles presentarán limitaciones metodológicas, cobertura geográfica restringida o ausencia de enfoques integrales. Se identificará una necesidad crítica de investigaciones que ofrezcan datos contextualizados, robustos y replicables, con el fin de aportar soluciones basadas en evidencia.")

    st.markdown("**Vinculación con los ODS**")
    st.markdown("La presente investigación contribuirá directamente al logro de uno o más Objetivos de Desarrollo Sostenible (ODS), promoviendo el cumplimiento de metas asociadas a la salud, equidad, educación de calidad, innovación, sostenibilidad o reducción de desigualdades.")

    st.markdown("**Pregunta de investigación**")
    st.markdown("¿Cuál será la relación, influencia o efecto de las variables identificadas dentro del contexto de estudio propuesto?")

    st.markdown("**Justificación del estudio**")
    st.markdown("- **Teórica:** Permitirá profundizar en modelos conceptuales existentes y enriquecer el marco teórico del fenómeno.")
    st.markdown("- **Práctica:** Generará insumos aplicables para la mejora de intervenciones, servicios o programas.")
    st.markdown("- **Metodológica:** Introducirá enfoques innovadores, herramientas o estrategias analíticas que enriquecerán futuras investigaciones.")
    st.markdown("- **Social:** Tendrá un impacto potencial sobre la calidad de vida de la población beneficiaria o el entorno social involucrado.")

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
