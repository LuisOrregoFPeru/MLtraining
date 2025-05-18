import streamlit as st
from huggingface_hub import InferenceClient
import requests
from docx import Document
from datetime import date
from typing import List

"""
Aplicación Streamlit que genera automáticamente:
* Introducción (≥ 9 páginas A4, prosa, tercera persona, futuro indicativo, ≤ 10 líneas por párrafo)
* 10 antecedentes de PubMed parafraseados (130 palabras c/u)
* Bases teóricas
* Hipótesis de investigación y estadísticas

👉 Ya **NO** depende de `openai`; emplea modelos de Hugging Face vía `InferenceClient`.
   - Modelo por defecto: *mistralai/Mistral‑7B‑Instruct‑v0.2* (puedes cambiarlo).
   - Se requiere un **HUGGINGFACE API TOKEN** (guárdalo en *Secrets* o ingrésalo en la barra lateral).
"""

# ---------------- Configuración del modelo ---------------- #

def get_client(token: str, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2") -> InferenceClient:
    if not token:
        st.error("Se necesita un token de Hugging Face válido.")
        st.stop()
    return InferenceClient(model=model_id, token=token)


def hf_generate(client: InferenceClient, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Genera texto con el modelo HF (formato instrucción)."""
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
  1‑2 p Importancia + plausibilidad del problema.
  1‑2 p Impacto (mundo, Latinoamérica, Perú).
  1‑2 p Vacíos de literatura.
  1 p Contribución a ODS.
  1 p Pregunta de investigación (interrogativa).
  1 p Justificación teórica.
  1 p Justificación práctica.
  1 p Justificación metodológica.
  1 p Justificación social.
Al final escribe la línea EXACTA "===ANTECEDENTES===" para indicar dónde se insertarán los antecedentes.
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
    paras = []
    for p in pubs:
        para = paraphrase_abstract(client, p["abstract"])
        paras.append(f"{p['cite']} {para}")
    return "\n\n".join(paras)


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
st.title("📝 Generador Automático de Introducciones de Tesis (sin OpenAI)")

with st.sidebar:
    st.header("Configuración")
    hf_token = st.text_input("Hugging Face API Token", type="password")
    model_id = st.text_input("ID del modelo (HF Hub)", value="mistralai/Mistral-7B-Instruct-v0.2")
    title_input = st.text_input("Título de la Investigación")
    objective_input = st.text_area("Objetivo General")
    generate_btn = st.button("Generar Introducción")

if generate_btn:
    client = get_client(hf_token, model_id)

    with st.spinner("Generando introducción…"):
        intro = generate_introduction(client, title_input, objective_input)
    st.subheader("Introducción")
    st.markdown(intro)

    with st.spinner("Buscando antecedentes en PubMed…"):
        pubs = search_pubmed(title_input or objective_input)
        antecedents = build_antecedents(client, pubs)
    st.subheader("Antecedentes")
    st.markdown(antecedents)

    with st.spinner("Generando bases teóricas…"):
        bases = generate_theoretical_bases(client, title_input, objective_input)
    st.subheader("Bases Teóricas")
    st.markdown(bases)

    with st.spinner("Generando hipótesis…"):
        hyps = generate_hypotheses(client, objective_input)
    st.subheader("Hipótesis")
    st.markdown(hyps)

    docx_file = build_docx(intro, antecedents, bases, hyps)
    st.download_button(
        "Descargar DOCX",
        data=docx_file,
        file_name=f"Tesis_{date.today()}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

st.markdown("""
---
### ¿Cómo usar esta versión?
1. Obtén un **Hugging Face API TOKEN** gratis en <https://huggingface.co/settings/tokens>.
2. Agrégalo en *Secrets* de Streamlit Cloud o en la barra lateral.
3. Ingresa el título u objetivo general y pulsa *Generar Introducción*.
4. Descarga el documento `.docx` para editarlo.

> Puedes cambiar `model_id` por cualquier modelo instructivo en el Hub que admita tarea *text‑generation* (p. ej. `meta-llama/Meta-Llama-3-8B-Instruct`).
""")

