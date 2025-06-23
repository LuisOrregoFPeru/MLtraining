# ────────────────────────────────────────────────────────────────────────────────
# app.py – Suite experta de análisis cualitativo en español (Streamlit)
# Descarga automática el modelo spaCy 'es_core_news_sm' si no está presente
# Autor: Jarvis – 2025 | Licencia: MIT
# ────────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import io
import itertools
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# ---------------  BLOQUE DE DESCARGA EN CALIENTE DE spaCy -----------------------
import spacy, subprocess, sys
from spacy.util import is_package

MODEL = "es_core_news_sm"
if not is_package(MODEL):
    st.warning(
        "Descargando modelo spaCy… (solo ocurre la primera vez, puede tardar 10–15 s)"
    )
    subprocess.run([sys.executable, "-m", "spacy", "download", MODEL], check=True)

nlp = spacy.load(MODEL)  # objeto de uso global
NLP = nlp  # alias para compatibilidad con funciones más abajo
# -------------------------------------------------------------------------------

from rake_nltk import Rake
import gensim
from gensim import corpora
import fitz  # PyMuPDF
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN INICIAL
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Suite de Análisis Cualitativo", layout="wide")
st.title("📝 Suite de Análisis Cualitativo de Textos (ES)")

# Descargar recursos NLTK si es primera vez
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOPWORDS_ES = set(stopwords.words("spanish"))

# ────────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE ENTRADA
# ────────────────────────────────────────────────────────────────────────────────


def leer_txt(bytestr: bytes) -> str:
    return bytestr.decode("utf-8", errors="ignore")


def leer_pdf(file) -> str:
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def leer_docx(file) -> str:
    temp_path = Path("_temp_docx.docx")
    temp_path.write_bytes(file.read())
    text = docx2txt.process(temp_path)
    temp_path.unlink(missing_ok=True)
    return text


def leer_csv(file) -> str:
    df = pd.read_csv(file)
    st.session_state["csv_df"] = df
    texcols = [c for c in df.columns if df[c].dtype == "object"]
    if not texcols:
        st.warning("No se detectaron columnas con texto.")
        return ""
    col_selec = st.selectbox("Selecciona la columna de texto a analizar:", texcols)
    return "\n".join(df[col_selec].dropna().astype(str).tolist())


def leer_archivo(subido) -> str:
    suffix = Path(subido.name).suffix.lower()
    if suffix == ".txt":
        return leer_txt(subido.read())
    if suffix == ".pdf":
        return leer_pdf(subido)
    if suffix == ".docx":
        return leer_docx(subido)
    if suffix == ".csv":
        return leer_csv(subido)
    st.warning(f"Formato {suffix} no soportado.")
    return ""


# ────────────────────────────────────────────────────────────────────────────────
# PREPROCESAMIENTO
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def limpiar_texto(texto: str) -> List[str]:
    doc = NLP(texto.lower())
    return [
        tok.lemma_
        for tok in doc
        if tok.is_alpha and tok.lemma_ not in STOPWORDS_ES and len(tok) > 2
    ]


def preprocesar_docs(docs: List[str]) -> List[List[str]]:
    return [limpiar_texto(t) for t in docs]


# ────────────────────────────────────────────────────────────────────────────────
# ANÁLISIS LÉXICO
# ────────────────────────────────────────────────────────────────────────────────
def frecuencias_global(tokens_limpios: List[List[str]]) -> Counter:
    return Counter(itertools.chain.from_iterable(tokens_limpios))


def calcular_coocurrencias(
    docs_tokens: List[List[str]], ventana: int = 2, top_k: int = 30
) -> List[Tuple[str, str, int]]:
    pares = Counter()
    for tokens in docs_tokens:
        for i in range(len(tokens)):
            for j in range(1, ventana + 1):
                if i + j < len(tokens):
                    pares[(tokens[i], tokens[i + j])] += 1
    return pares.most_common(top_k)


# ────────────────────────────────────────────────────────────────────────────────
# FRASES CLAVE (RAKE)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extraer_rake(texto: str, top_n: int = 5) -> List[str]:
    rake = Rake(language="spanish", stopwords=list(STOPWORDS_ES))
    rake.extract_keywords_from_text(texto)
    return rake.get_ranked_phrases()[:top_n]


# ────────────────────────────────────────────────────────────────────────────────
# MODELADO DE TEMAS
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def lda_gensim(docs_tokens: List[List[str]], n_temas: int = 5):
    dictionary = corpora.Dictionary(docs_tokens)
    corpus = [dictionary.doc2bow(text) for text in docs_tokens]
    lda_model = gensim.models.LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=n_temas, passes=10, random_state=42
    )
    temas = [
        [palabra for palabra, _ in lda_model.show_topic(t, topn=10)]
        for t in range(n_temas)
    ]
    asignaciones = [
        max(lda_model.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus
    ]
    return temas, asignaciones, lda_model


# ────────────────────────────────────────────────────────────────────────────────
# SENTIMIENTO SIMPLE
# ────────────────────────────────────────────────────────────────────────────────
POS = {
    "excelente",
    "bueno",
    "agradable",
    "positivo",
    "maravilloso",
    "feliz",
    "fantástico",
    "profesional",
    "amable",
}
NEG = {
    "malo",
    "terrible",
    "horrible",
    "negativo",
    "triste",
    "desagradable",
    "incompetente",
    "lento",
    "pésimo",
}


def sentimiento(texto: str) -> str:
    tokens = limpiar_texto(texto)
    score = sum(1 for t in tokens if t in POS) - sum(1 for t in tokens if t in NEG)
    if score > 0:
        return "Positivo"
    if score < 0:
        return "Negativo"
    return "Neutral"


# ────────────────────────────────────────────────────────────────────────────────
# VISUALIZACIONES
# ────────────────────────────────────────────────────────────────────────────────
def nube_palabras(freq: Counter, max_words: int = 150) -> WordCloud:
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=max_words,
        colormap="Dark2",
    )
    return wc.generate_from_frequencies(freq)


def fig_to_download(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ────────────────────────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────────
def main():
    archivos = st.file_uploader(
        "📂 Sube archivos (.txt, .pdf, .docx, .csv)", type=["txt", "pdf", "docx", "csv"], accept_multiple_files=True
    )

    if not archivos:
        st.info("Carga uno o más archivos para comenzar.")
        st.stop()

    docs_raw = [leer_archivo(f) for f in archivos if leer_archivo(f)]
    if not docs_raw:
        st.error("No se pudo leer ningún documento válido.")
        st.stop()

    if st.button("🚀 Ejecutar análisis"):
        with st.spinner("Procesando textos…"):
            docs_tokens = preprocesar_docs(docs_raw)
            freq = frecuencias_global(docs_tokens)
            coocs = calcular_coocurrencias(docs_tokens)
            frases_rake = [extraer_rake(t) for t in docs_raw]
            num_temas = st.sidebar.slider("Número de temas LDA", 2, 10, 5, 1)
            temas, asignaciones, _ = lda_gensim(docs_tokens, num_temas)
            sentis = [sentimiento(t) for t in docs_raw]

        tabs = st.tabs(["Resumen", "Temas", "Léxico", "Sentimiento"])

        with tabs[0]:
            st.subheader("Vista previa")
            st.write(f"Documentos analizados: {len(docs_raw)}")
            st.text_area("Texto original", docs_raw[0][:1000], height=200)
            st.text_area(
                "Tokens limpios (ejemplo)", " ".join(docs_tokens[0][:50]), height=100
            )

        with tabs[1]:
            st.subheader("Temas detectados (LDA)")
            for i, palabras in enumerate(temas, 1):
                st.markdown(f"**Tema {i}:** " + ", ".join(palabras))
            df_temas = pd.DataFrame(
                {
                    "Documento": range(1, len(docs_raw) + 1),
                    "Tema_asignado": [t + 1 for t in asignaciones],
                    "Sentimiento": sentis,
                }
            )
            st.dataframe(df_temas)
            st.download_button(
                "Descargar asignaciones CSV",
                df_temas.to_csv(index=False).encode("utf-8"),
                "asignaciones_temas.csv",
                "text/csv",
            )

        with tabs[2]:
            st.subheader("Nube de palabras global")
            wc = nube_palabras(freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            st.download_button(
                "Descargar nube (PNG)",
                fig_to_download(fig),
                "nube_palabras.png",
                "image/png",
            )

            st.markdown("---")
            st.subheader("Co-ocurrencias principales")
            cooc_df = pd.DataFrame(coocs, columns=["Palabra_A", "Palabra_B", "Frecuencia"])
            st.dataframe(cooc_df)
            st.download_button(
                "Descargar co-ocurrencias CSV",
                cooc_df.to_csv(index=False).encode("utf-8"),
                "coocurrencias.csv",
                "text/csv",
            )

            st.markdown("---")
            st.subheader("Frases clave (RAKE)")
            for idx, frases in enumerate(frases_rake, 1):
                st.markdown(f"**Doc {idx}:** " + "; ".join(frases) if frases else f"Doc {idx}: (sin frases)")

        with tabs[3]:
            st.subheader("Distribución de Sentimientos")
            senti_counts = Counter(sentis)
            plt.figure(figsize=(4, 3))
            sns.barplot(
                x=list(senti_counts.keys()),
                y=list(senti_counts.values()),
                palette=["green", "gray", "red"],
            )
            plt.ylabel("Nº documentos")
            st.pyplot(plt.gcf())

            st.markdown("---")
            st.subheader("Nubes por sentimiento")
            for cat in ["Positivo", "Neutral", "Negativo"]:
                subset = list(
                    itertools.chain.from_iterable(
                        [tkn for tkn, s in zip(docs_tokens, sentis) if s == cat]
                    )
                )
                if subset:
                    st.markdown(f"**{cat}**")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    wc_cat = nube_palabras(Counter(subset), max_words=80)
                    ax.imshow(wc_cat, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.write(f"Sin textos {cat.lower()}s.")

    else:
        st.info("Cuando termines de subir, presiona **Ejecutar análisis**.")


if __name__ == "__main__":
    main()
