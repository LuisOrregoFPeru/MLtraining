# ────────────────────────────────────────────────────────────────────────────────
# app.py – Suite cualitativa robusta (Streamlit, spaCy con descarga tolerante)
# ────────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import io, itertools, logging, subprocess, sys, traceback
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# ---------- spaCy: descarga en caliente con tolerancia a fallos -----------------
import spacy
from spacy.util import is_package

MODEL = "es_core_news_sm"

try:
    nlp = spacy.load(MODEL)  # ¿ya está instalado?
except OSError:
    st.warning("Descargando modelo spaCy… (solo la primera vez, ~10 s)")
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", MODEL],
            check=False,               # ← no detiene el proceso si falla
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        nlp = spacy.load(MODEL)        # intento cargar de nuevo
    except Exception as e:
        logging.warning("No se pudo descargar spaCy: %s", e)
        traceback.print_exc()
        st.warning("⚠️ El modelo spaCy no se pudo descargar; se usará uno en blanco.")
        nlp = spacy.blank("es")        # modelo vacío (sin lematización)

NLP = nlp  # alias global
# --------------------------------------------------------------------------------

from rake_nltk import Rake
import gensim
from gensim import corpora
from sklearn.decomposition import LatentDirichletAllocation

# Lectores opcionales de PDF
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except ImportError:
    HAVE_PYMUPDF = False

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    HAVE_PDFMINER = True
except ImportError:
    HAVE_PDFMINER = False

import docx2txt

# ───────────────────────── CONFIG Streamlit ────────────────────────────
st.set_page_config(page_title="Suite Cualitativa (Robusta)", layout="wide")
st.title("📝 Suite de Análisis Cualitativo (Robusta)")

nltk.download("stopwords", quiet=True)
STOPWORDS_ES = set(stopwords.words("spanish"))

# ───────────────────── Funciones de lectura seguras ────────────────────
def leer_txt(buf: bytes) -> str:
    return buf.decode("utf-8", errors="ignore")

def leer_pdf(file) -> str:
    if HAVE_PYMUPDF:
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif HAVE_PDFMINER:
        return pdf_extract_text(file)
    else:
        st.warning("❗️ Sin soporte PDF (instala pymupdf o pdfminer.six).")
        return ""

def leer_docx(file) -> str:
    tmp = Path("_temp.docx")
    tmp.write_bytes(file.read())
    text = docx2txt.process(tmp)
    tmp.unlink(missing_ok=True)
    return text

def leer_csv(file) -> str:
    df = pd.read_csv(file)
    cols = [c for c in df.columns if df[c].dtype == "object"]
    if not cols:
        return ""
    col = st.selectbox("Columna de texto", cols, key=file.name)
    return "\n".join(df[col].dropna().astype(str))

def leer_archivo(subido) -> str:
    ext = Path(subido.name).suffix.lower()
    if ext == ".txt":  return leer_txt(subido.read())
    if ext == ".pdf":  return leer_pdf(subido)
    if ext == ".docx": return leer_docx(subido)
    if ext == ".csv":  return leer_csv(subido)
    st.warning(f"{ext} no soportado")
    return ""

# ───────────────────── NLP utilidades ────────────────────────────────
@st.cache_data(show_spinner=False)
def limpiar(texto: str) -> List[str]:
    doc = NLP(texto.lower())
    return [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in STOPWORDS_ES]

def preprocesar(corpus: List[str]) -> List[List[str]]:
    return [limpiar(t) for t in corpus]

def freq_global(tokens: List[List[str]]) -> Counter:
    return Counter(itertools.chain.from_iterable(tokens))

def coocurrencias(docs_tokens, win=2, top=30):
    pares = Counter()
    for toks in docs_tokens:
        for i in range(len(toks)):
            for j in range(1, win + 1):
                if i + j < len(toks):
                    pares[(toks[i], toks[i + j])] += 1
    return pares.most_common(top)

@st.cache_data(show_spinner=False)
def rake_frases(texto, n=5):
    r = Rake(language="spanish", stopwords=list(STOPWORDS_ES))
    r.extract_keywords_from_text(texto)
    return r.get_ranked_phrases()[:n]

@st.cache_data(show_spinner=False)
def lda_temas(docs_tokens, k=5):
    dic = corpora.Dictionary(docs_tokens)
    corp = [dic.doc2bow(t) for t in docs_tokens]
    lda = gensim.models.LdaModel(corp, id2word=dic, num_topics=k, passes=10, random_state=42)
    temas = [[w for w, _ in lda.show_topic(t)] for t in range(k)]
    asign = [max(lda.get_document_topics(b), key=lambda x: x[1])[0] for b in corp]
    return temas, asign

POS = {"excelente","bueno","positivo","feliz","fantástico"}
NEG = {"malo","terrible","negativo","triste","pésimo"}

def sentimiento(txt):
    toks = limpiar(txt)
    sc = sum(t in POS for t in toks) - sum(t in NEG for t in toks)
    return "Positivo" if sc>0 else "Negativo" if sc<0 else "Neutral"

def nube(freq):
    return WordCloud(width=800,height=400,background_color="white",colormap="Dark2").generate_from_frequencies(freq)

# ───────────────────── Interfaz principal ─────────────────────────────
uploads = st.file_uploader("Sube archivos (.txt, .pdf, .docx, .csv)", type=["txt","pdf","docx","csv"], accept_multiple_files=True)

if not uploads:
    st.stop()

docs = [leer_archivo(f) for f in uploads]
docs = [d for d in docs if d]
if not docs:
    st.error("No se pudo leer ningún documento.")
    st.stop()

if st.button("🚀 Ejecutar análisis"):
    with st.spinner("Procesando…"):
        toks  = preprocesar(docs)
        freq = freq_global(toks)
        cooc = coocurrencias(toks)
        frases = [rake_frases(t) for t in docs]
        k = st.sidebar.slider("Temas LDA", 2, 10, 5, 1)
        temas, asign = lda_temas(toks, k)
        sentis = [sentimiento(t) for t in docs]

    tab1, tab2, tab3, tab4 = st.tabs(["Resumen","Temas","Léxico","Sentimiento"])

    with tab1:
        st.subheader("Ejemplo")
        st.text_area("Texto original", docs[0][:800])
        st.text_area("Tokens limpios", " ".join(toks[0][:60]))

    with tab2:
        st.subheader("Temas detectados")
        for i, p in enumerate(temas, 1):
            st.markdown(f"**Tema {i}:** " + ", ".join(p))
        df_asig = pd.DataFrame({"Doc": range(1,len(docs)+1), "Tema": [a+1 for a in asign], "Sent": sentis})
        st.dataframe(df_asig)

    with tab3:
        st.subheader("Nube de palabras global")
        fig, ax = plt.subplots(figsize=(10,4)); ax.imshow(nube(freq)); ax.axis("off"); st.pyplot(fig)
        st.subheader("Co‐ocurrencias"); st.dataframe(pd.DataFrame(cooc, columns=["A","B","f"]))
        st.subheader("Frases clave (RAKE)")
        for i, fr in enumerate(frases, 1):
            st.markdown(f"**Doc {i}:** " + "; ".join(fr) if fr else f"Doc {i}: (sin frases)")

    with tab4:
        st.subheader("Distribución de Sentimientos")
        st.bar_chart(pd.Series(sentis).value_counts())

# ──────────────────────────────────────────────────────────────────────
