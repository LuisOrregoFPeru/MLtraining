# app.py ─ Análisis cualitativo ligero con carga de .txt (Streamlit)
# ─────────────────────────────────────────────────────────────────
import re, itertools, collections, io, logging, sys, subprocess
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt; plt.rcParams["font.size"] = 9
import seaborn as sns, networkx as nx
from wordcloud import WordCloud
from pyvis.network import Network

# -------- Stop-words y conectores ---------------------------------
STOP = set("""
de la que el en y a los las del se un con por no una su para es al lo como
más o pero sus le ya entre cuando todo esta ser son fue así bien sin embargo entonces
además asimismo luego aunque también mientras osea pues porque porqué por otro lado
por lo tanto
""".split())

def limpiar(texto:str):
    tokens = re.findall(r"\b\w+\b", texto.lower())
    return [t for t in tokens if t not in STOP and len(t)>2 and not t.isdigit()]

# -------- UI Streamlit --------------------------------------------
st.set_page_config(page_title="Vis QDA", layout="wide")
st.title("📝 Visualizador cualitativo exprés")

subidos = st.file_uploader(
    "Sube uno o más archivos .txt con tus entrevistas",
    type=["txt"], accept_multiple_files=True
)

if subidos:
    # Leer y limpiar
    docs_raw  = [s.read().decode("utf-8", errors="ignore") for s in subidos]
    nombres   = [Path(s.name).stem for s in subidos]
    tokens    = [limpiar(t) for t in docs_raw]
    freq      = collections.Counter(itertools.chain.from_iterable(tokens))
    
    # Tabs de resultados
    tabs = st.tabs(["WordCloud", "Top-20", "Heat-map", "Red"])

    # ---- Tab 1: nube ---------------------------------------------
    with tabs[0]:
        st.header("Nube de palabras depurada")
        wc = WordCloud(width=1000, height=500, background_color="white", colormap="Dark2")
        wc_img = wc.generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(10,5)); ax.imshow(wc_img); ax.axis("off")
        st.pyplot(fig)
        buf = io.BytesIO(); wc_img.to_image().save(buf, format="PNG")
        st.download_button("Descargar PNG", buf.getvalue(),
                           file_name="nube_palabras.png", mime="image/png")

    # ---- Tab 2: barras -------------------------------------------
    with tabs[1]:
        st.header("Top-20 términos")
        top20 = freq.most_common(20)
        if top20:
            term, cnt = zip(*top20)
            plt.figure(figsize=(8,4))
            sns.barplot(x=list(cnt), y=list(term), orient="h", color="#4c72b0")
            plt.xlabel("Frecuencia"); plt.tight_layout()
            st.pyplot(plt.gcf())
            buf = io.BytesIO(); plt.savefig(buf, format="PNG", dpi=300)
            st.download_button("Descargar PNG", buf.getvalue(),
                               "top20.png", "image/png")
            plt.close()
        else:
            st.info("No hay términos después del filtrado.")

    # ---- Tab 3: Heat-map -----------------------------------------
    with tabs[2]:
        st.header("Matriz Código × Documento")
        codigos = [t for t,_ in freq.most_common(30)]
        mat = pd.DataFrame(0, index=codigos, columns=nombres)
        for n, toks in zip(nombres, tokens):
            c = collections.Counter(toks)
            for term in codigos: mat.loc[term, n] = c[term]
        sns.heatmap(mat, cmap="viridis", linewidths=.3)
        plt.title("Densidad de códigos"); plt.tight_layout()
        st.pyplot(plt.gcf())
        buf = io.BytesIO(); plt.savefig(buf, format="PNG", dpi=300)
        st.download_button("Descargar PNG", buf.getvalue(),
                           "heatmap_codigos.png", "image/png")
        plt.close()

    # ---- Tab 4: Red co-ocurrencias -------------------------------
    with tabs[3]:
        st.header("Red de co-ocurrencias (≥ 3)")
        # Construir red básica
        G = nx.Graph(); win = 2
        for toks in tokens:
            for i in range(len(toks)-win):
                for j in range(1,win+1):
                    u,v = toks[i], toks[i+j]
                    if u==v: continue
                    G.add_edge(u,v, weight=G[u][v]['weight']+1 if G.has_edge(u,v) else 1)
        H = nx.Graph([(u,v,d) for u,v,d in G.edges(data=True) if d["weight"]>=3])
        if H.number_of_edges()==0:
            st.info("No hay suficientes co-ocurrencias fuertes.")
        else:
            net = Network(height="600px", width="100%", bgcolor="#ffffff")
            net.from_nx(H); net.repulsion(node_distance=120, central_gravity=0.2)
            html_path = "coocurrencias.html"; net.save_graph(html_path)
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=600, scrolling=True)
            with open(html_path, "rb") as f:
                st.download_button("Descargar HTML", f, "coocurrencias.html",
                                   "text/html")
else:
    st.info("• Sube archivos para comenzar …")


print(f"\n✅ Gráficos guardados en: {OUT_DIR}\n")
for f in OUT_DIR.iterdir():
    print("  •", f.name)

