# ==============================================================================
#  Streamlit Tesis App  ·  Versión sin dependencia obligatoria de OpenAI
# ------------------------------------------------------------------------------
#  requirements.txt (mínimo):
#      streamlit
#      requests
# ------------------------------------------------------------------------------
#  Si deseas la generación automática con GPT, instala además:
#      openai>=1.3.0  ·  y agrega tu OPENAI_API_KEY
#  En ausencia de `openai`, la app funciona, pero solicitará la introducción
#  manualmente y mostrará los abstracts originales (acortados) de PubMed.
# ==============================================================================

import streamlit as st
import datetime
import textwrap
import xml.etree.ElementTree as ET
import requests
import os

# ──────────────────────────────────────────────────────────────
# ¿OpenAI disponible?
# ──────────────────────────────────────────────────────────────
openai_available = False
try:
    import openai
    openai_available = True
except ModuleNotFoundError:
    pass  # La app continuará en modo manual

# Configuración de página
st.set_page_config(page_title="Asistente de Tesis (modo manual)", layout="wide")
st.title("📘 Asistente de Tesis – Generación de Introducción y Antecedentes")

# ──────────────────────────────────────────────────────────────
# Entradas de usuario
# ──────────────────────────────────────────────────────────────

titulo = st.text_input("🎓 Título del proyecto de tesis")
objetivo_general = st.text_area("🎯 Objetivo general de la investigación")
keyword_pubmed = st.text_input("🔎 Palabra clave para antecedentes (PubMed)")
num_articulos = st.slider("Número de artículos PubMed", 5, 10, 10)

# Si openai está disponible y hay clave, permitir modo automático
modo_auto = False
if openai_available:
    openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if openai.api_key:
        modo_auto = st.checkbox("Usar ChatGPT para generar automáticamente la introducción", value=True)
    else:
        st.info("Se detectó la librería `openai`, pero falta `OPENAI_API_KEY`. La introducción será manual.")

# ──────────────────────────────────────────────────────────────
# Funciones auxiliares sin GPT
# ──────────────────────────────────────────────────────────────

def acortar_texto(texto: str, max_palabras: int = 130) -> str:
    palabras = texto.split()
    return " ".join(palabras[:max_palabras]) + (" …" if len(palabras) > max_palabras else "")


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
        year = art.findtex
