# qual_vis.py  · versión ligera con rutas absolutas personalizadas
# -------------------------------------------------------------------
import re, pathlib, itertools, collections
import matplotlib.pyplot as plt; plt.rcParams["font.size"] = 9
import seaborn as sns, networkx as nx, pandas as pd
from wordcloud import WordCloud
from docx import Document
from pyvis.network import Network

# --- 1. Carpetas ----------------------------------------------------
DATA_DIR = pathlib.Path(r"C:\Users\luiso\Documents\data_quali")   # ← coloca aquí tus textos
OUT_DIR  = pathlib.Path(r"C:\Users\luiso\Documents\output_quali") # ← salen los gráficos
OUT_DIR.mkdir(exist_ok=True)                                     # la crea si no existe

# --- 2. Stop-words + conectores ------------------------------------
STOP_ES = set("""
de la que el en y a los las del se un con por no una su para es al lo como
más o pero sus le ya entre cuando todo esta ser son fue había así si bien
""".strip().split())

CONECTORES = {
    "además","asimismo","entonces","pues","osea","o","sea","sin","embargo",
    "luego","aunque","también","mientras","por","otro","lado","porqué",
    "porque","por", "lo", "tanto","así","que"
}

STOP = STOP_ES | CONECTORES

def clean_tokens(text:str):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in STOP and len(t) > 2 and not t.isdigit()]

# --- 3. Leer documentos ---------------------------------------------
paths = list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.docx"))
if not paths:
    raise SystemExit("👉 No se encontraron .txt o .docx en "+str(DATA_DIR))

docs_raw, names = [], []
for p in paths:
    if p.suffix.lower() == ".txt":
        docs_raw.append(p.read_text(encoding="utf-8", errors="ignore"))
    else:  # .docx
        docs_raw.append("\n".join([para.text for para in Document(p).paragraphs]))
    names.append(p.stem)

docs_tokens = [clean_tokens(t) for t in docs_raw]

# --- 4. Word cloud + barras -----------------------------------------
freq = collections.Counter(itertools.chain.from_iterable(docs_tokens))
WordCloud(width=1200,height=600,background_color="white",colormap="Dark2")\
    .generate_from_frequencies(freq)\
    .to_file(OUT_DIR/"nube_palabras.png")

top20 = freq.most_common(20)
terms, counts = zip(*top20)
plt.figure(figsize=(8,4))
sns.barplot(x=list(counts), y=list(terms), orient="h")
plt.xlabel("Frecuencia"); plt.title("Top 20 términos depurados")
plt.tight_layout(); plt.savefig(OUT_DIR/"top20.png", dpi=300); plt.close()

# --- 5. Heat-map Código × Documento ----------------------------------
codes = [t for t,_ in freq.most_common(30)]
mat = pd.DataFrame(0, index=codes, columns=names)
for name,toks in zip(names, docs_tokens):
    c = collections.Counter(toks)
    for term in codes:
        mat.loc[term, name] = c[term]
sns.heatmap(mat, cmap="viridis", linewidths=.3)
plt.title("Matriz Código × Documento"); plt.tight_layout()
plt.savefig(OUT_DIR/"heatmap_codigos.png", dpi=300); plt.close()

# --- 6. Red de co-ocurrencias ----------------------------------------
G = nx.Graph(); window = 2
for toks in docs_tokens:
    for i in range(len(toks)-window):
        for j in range(1, window+1):
            u,v = toks[i], toks[i+j]
            if u==v: continue
            G.add_edge(u,v, weight=G[u][v]['weight']+1 if G.has_edge(u,v) else 1)
H = nx.Graph([(u,v,d) for u,v,d in G.edges(data=True) if d["weight"]>=3])
net = Network(height="700px", width="100%", bgcolor="#ffffff")
net.from_nx(H); net.repulsion(node_distance=120, central_gravity=0.2)
net.save_graph(str(OUT_DIR/"coocurrencias.html"))

print(f"\n✅ Gráficos guardados en: {OUT_DIR}\n")
for f in OUT_DIR.iterdir():
    print("  •", f.name)

