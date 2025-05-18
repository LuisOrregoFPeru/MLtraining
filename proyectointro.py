import streamlit as st
from huggingface_hub import InferenceClient
import requests
from docx import Document
from datetime import date
from typing import List

"""
Aplicación Streamlit para generar introducciones de tesis SIN exponer tu clave:
* Introducción ≥ 9 páginas A4 (prosa, 3.ª persona, futuro indicativo)
* 10 antecedentes de PubMed parafraseados
* Bases teóricas y hipótesis

🚩 **La clave se obtiene de los Secrets de Streamlit Cloud o del cuadro lateral.** No hay token hard-codeado.
"""

# ---------------- Token y modelo por defecto ---------------- #

DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Intenta leer la clave desde los Secrets del deploy
DEFAULT_HF_TOKEN = st.secrets.get("HF_TOKEN", "")  # ← vacío si no se ha configurado

# ---------------- Configuración del cliente HF ---------------- #

def get_client(token: str | None = None, model_id: str = DEFAULT_MODEL_ID) -> InferenceClient:
    token = token or DEFAULT_HF_TOKEN
    if not token:
        st.error("⚠️ Debes proporcionar un Hugging Face API Token en Secrets o en el panel lateral.")
        st.stop()
    return InferenceClient(model=model_id, token=token)


def hf_generate(client: InferenceClient, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Genera texto con el modelo HF (formato instrucción)."""
    return client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
    )

# ---------------- Funciones de negocio ---------------- #

def generate_introduction(client: InferenceClient, title: str, objective: str, word_target: int = 4500) -> str:
    prompt = f"""
Redacta una introducción de tesis académica con las siguientes características:
• Extensión mínima: {word_target} palabras (≈ 9 páginas A4).
• Prosa, sin subtítulos, tercera persona, tiempo futuro del modo indicativo.
• Máximo 10 líneas por párrafo.
• Secuencia de párrafos exacta:
  1-2 p Importancia + plausibilidad del problema.
  1-2 p Impacto (mundo, Latinoamérica, Perú).
  1-2 p Vacíos de literatura.
  1 p Contribución a ODS.
  1 p Pregunta de investigación (interrogativa).
  1 p Justificación teórica.
  1 p Justificación práctica.
  1 p Justificación metodológica.
  1 p Justificación social.
Al final escribe la línea EXACTA "===ANTECEDENTES===".
Título: {title}
Objetivo general: {objective}
"""
    return hf_generate(client, prompt, max_tokens=4096)


def paraphrase_abstract(client: InferenceClient, abstract: str, word_count: int = 130) -> str:
    prompt = (
        f"Parafrasea en español académico el siguiente resumen en aproximadamente {word_count} palabras, evitando plagio:\n{abstract}"
    )
    return hf_generate(client, prompt, max_tokens=256, temperature=0.4)


def search_pubmed(query: str, n: int = 10) -> List[dict]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    ids = requests.get(base + "esearch.fcgi", params={"db": "pubmed", "term": query, "retmax": n, "retmode": "json"}).json()["esearchresult"]["idlist"]
    items = []
    for pmid in ids:
        xml = requests.get(base + "efetch.fcgi", params={"db": "pubmed", "id": pmid, "retmode": "xml"}).text
        import re, html
        abstract = re.search(r"<AbstractText.*?>(.*?)</AbstractText>", xml, re.S)
        if not abstract:
            continue
        authors = re.findall(r"<LastName>(.*?)</LastName>.*?<Initials>(.*?)</Initials>", xml, re.S)
        year = re.search(r"<PubDate>.*?<Year>(\\d{4})</Year>", xml, re.S)
        if authors:
            first = f"{authors[0][0]}, {authors[0][1][0]}."
            cite = first + (" et al." if len(authors) > 3 else "")
        else:
            cite = "Autor desconocido"
        items.append({
            "cite": f"{cite} ({year.group(1) if year else 's.f.'})",
            "abstract": html.unescape(abstract.group(1))
        })
    return items[:n]


def build_antecedents(client: InferenceClient, pubs: List[dict]) -> str:
    return "\n\n".join(
        f"{p['cite']} {paraphrase_abstract(client, p['abstract'])}" for p in pubs
    )


def generate_theoretical_bases(client: InferenceClient, title: str, objective: str) -> str:
    prompt = (
        "Redacta las bases teóricas pertinentes a la pregunta de investigación, incluyendo enfoques conceptuales y variables clave. "
        "Prosa académica, tercera persona, futuro indicativo.\n"
        f"Título: {title}\nObjetivo general: {objective}"
    )
    return hf_generate(client, prompt, max_tokens=1024)


def generate_hypotheses(client: InferenceClient, objective: str) -> str:
    prompt = (
        "Formula la hipótesis de investigación y las hipótesis estadísticas (nula y alternativa) basadas en el siguiente objetivo general.\n"
        f"{objective}"
    )
    return hf_generate(client, prompt, max_tokens=256)


def build_docx(intro: str, antecedentes: str, bases: str, hyps: str) -> bytes:
    doc = Document()
    doc.add_heading("Proyecto de Tesis – Secciones Generadas", 1)
    doc.add_heading("Introducción", 2)
    for p in intro.split("\n"):
        doc.add_paragraph(p)
    doc.add_page_break()

    doc.add_heading("Antecedentes", 2)
    for p in antecedentes.split("\n"):
        doc.add_paragraph(p)
    doc.add_page_break()

    doc.add_heading("Bases Teóricas", 2)
    for p in bases.split("\n"):
        doc.add_paragraph(p)
    doc.add_page_break()

    doc.add_heading("Hipótesis", 2)
    for p in hyps.split("\n"):
        doc.add_paragraph(p)

    from io import BytesIO
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

# ---------------- Interfaz Streamlit ---------------- #

st.set_page_config(page_title="Generador de Introducciones de Tesis", layout="wide")
st.title("📝 Generador Automático de Introducciones de Tesis (Hugging Face)")

with st.sidebar:
    st.header("Configuración")
    hf_token = st.text_input("Hugging Face API Token", type="passwo
