# ==============================================================================
#  Streamlit Tesis App  ·  Requiere la librería  openai
# ------------------------------------------------------------------------------
#  requirements.txt (mínimo):
#      streamlit
#      openai>=1.3.0
#      requests
# ------------------------------------------------------------------------------
#  Instalar localmente:
#      pip install streamlit openai requests
# ==============================================================================

import streamlit as st
import datetime
import textwrap
import xml.etree.ElementTree as ET
import requests
import os
import openai

# Configuración de página
st.set_page_config(page_title="Asistente de Tesis", layout="wide")
st.title("📘 Asistente Automático para Introducciones de Tesis")

st.markdown(
    """
Genera automáticamente la **Introducción** (≥ 9 páginas A4) de un proyecto de tesis a partir de un título u objetivo general. 
Redacción en **prosa**, **tercera persona**, **tiempo futuro del indicativo**, párrafos ≤ 10 líneas.  Incluye antecedentes parafraseados desde PubMed, bases teóricas, enfoques conceptuales e hipótesis.
"""
)

# ──────────────────────────────────────────────────────────────
# Entradas de usuario
# ──────────────────────────────────────────────────────────────

titulo = st.text_input("🎓 Título del proyecto de tesis")
objetivo_general = st.text_area("🎯 Objetivo general de la investigación")
keyword_pubmed = st.text_input("🔎 Palabra clave para antecedentes (PubMed)")
num_articulos = st.slider("# de artículos PubMed", 5, 10, 10)

# Clave OpenAI
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("⚠️ Falta la variable `OPENAI_API_KEY`. Agrégala en `.streamlit/secrets.toml` o Settings → Secrets.")
    st.stop()

# ──────────────────────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────────────────────

def chat_gpt(prompt: str, max_tokens: int = 1500, temperature: float = 0.7) -> str:
    """Envia un prompt a ChatGPT y devuelve el texto generado"""
    r = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()


def generar_introduccion(titulo: str, objetivo: str) -> str:
    prompt = f"""
Redacta la introducción de una tesis titulada \"{titulo}\" en prosa, tercera persona y tiempo futuro del indicativo.
Estructura solicitada:
1. 1–2 párrafos: realidad problemática (importancia y plausibilidad).
2. 1–2 párrafos: impacto (frecuencia y morbimortalidad global, Latinoamérica y Perú).
3. 1–2 párrafos: vacíos en la literatura.
4. 1 párrafo: vínculo con el ODS correspondiente.
5. 1 párrafo: pregunta de investigación en modo interrogativo.
6. 1 párrafo: justificación teórica.
7. 1 párrafo: justificación práctica.
8. 1 párrafo: justificación metodológica.
9. 1 párrafo: justificación social.
Cada párrafo ≤10 líneas. Longitud total ≈9 páginas A4 (4500-5000 palabras). Debe mencionar el objetivo general: \"{objetivo}\".
Devuelve solo el texto final, sin títulos ni enumeraciones.
"""
    return chat_gpt(prompt, max_tokens=3900)


def buscar_articulos_pubmed(query: str, n: int):
    anio_inicio = datetime.date.today().year - 5
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={query}&retmax={n}&retmode=json&mindate={anio_inicio}&datetype=pdat"
    )
    ids = requests.get(url).json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    url_fetch = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&id={','.join(ids)}&retmode=xml"
    )
    root = ET.fromstring(requests.get(url_fetch).content)
    res = []
    for art in root.findall(".//PubmedArticle"):
        autor = art.findtext(".//Author/LastName", default="Autor")
        inicial = art.findtext(".//Author/Initials", default="N")
        year = art.findtext(".//PubDate/Year") or art.findtext(".//DateCreated/Year") or "s.f."
        abstract = art.findtext(".//Abstract/AbstractText", default="")
        res.append((autor, inicial, year, abstract))
    return res


def parafrasear_abstract(autor, inicial, año, abstracto):
    prompt = (
        f"Parafrasea el siguiente abstract en ~130 palabras, sin plagio. "
        f"Inicia el párrafo con '{autor}, {inicial}. et al. ({año})'.\n\n"
        f"Abstract original:\n{abstracto}"
    )
    return chat_gpt(prompt, max_tokens=220, temperature=0.3)

# ──────────────────────────────────────────────────────────────
# Ejecución de la app
# ──────────────────────────────────────────────────────────────

if st.button("🚀 Generar Introducción Completa"):
    if not (titulo and objetivo_general):
        st.warning("Ingresa tanto el título como el objetivo general.")
        st.stop()

    # 1. Introducción
    with st.spinner("Generando introducción…"):
        intro = generar_introduccion(titulo, objetivo_general)
    st.subheader("🧾 Introducción Generada")
    st.write(intro)

    # 2. Antecedentes
    if keyword_pubmed:
        with st.spinner("Extrayendo y parafraseando antecedentes…"):
            articulos = buscar_articulos_pubmed(keyword_pubmed, num_articulos)
            antecedentes = []
            for a, i, y, abs_txt in articulos:
                if abs_txt.strip():
                    antecedentes.append(parafrasear_abstract(a, i, y, abs_txt))
        st.subheader("📚 Antecedentes")
        if antecedentes:
            for parrafo in antecedentes:
                st.markdown(parrafo)
        else:
            st.info("No se obtuvieron antecedentes para la palabra clave indicada.")
    else:
        st.info("Ingresa una palabra clave (PubMed) para generar antecedentes.")

    # 3. Bases teóricas y enfoques conceptuales
    st.subheader("🧠 Bases teóricas y enfoques conceptuales")
    bases_prompt = (
        "Redacta en tercera persona y tiempo futuro, párrafos ≤10 líneas, las bases teóricas y enfoques conceptuales "
        f"relacionados con el estudio titulado '{titulo}'."
    )
    st.write(chat_gpt(bases_prompt, max_tokens=650))

    # 4. Hipótesis
    st.subheader("🔬 Hipótesis")
    hip_prompt = (
        f"Formula hipótesis de investigación y las correspondientes hipótesis estadísticas basadas en el objetivo general: '{objetivo_general}'. "
        "Redacta en tercera persona, tiempo futuro."
    )
    st.write(chat_gpt(hip_prompt, max_tokens=300))

    st.success("✅ Documento generado.")
