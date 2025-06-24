# panel.py ─ Discourse Analytics Panel
# ─────────────────────────────────────
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Discourse Analytics Panel",
                   page_icon="🎛️",
                   layout="centered")

# ------------------- ESTILO -----------------------
st.markdown(
    """
    <style>
    /* panel container */
    .panel {
        background:#222; padding:1rem 1.2rem; border-radius:8px;
        max-width:650px; margin:auto; color:#eee; font-family:Arial, sans-serif;
    }
    .panel h2 {margin-top:0; font-size:1.4rem; letter-spacing:.5px;}
    /* botones */
    .btn-grid {display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:.8rem;}
    .btn-grid button {
        width:100%; padding:.6rem .5rem; border:none; border-radius:6px;
        background:#e0e0e0; color:#111; font-weight:600; cursor:pointer;
        transition: all .15s ease;
    }
    .btn-grid button:hover {background:#f4b400; color:#000;}
    /* close */
    .close {position:absolute; top:.7rem; right:.8rem;
            width:28px; height:28px; border:2px solid #ccc; border-radius:50%;
            font-size:18px; line-height:22px; text-align:center; cursor:pointer;}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- LÓGICA -----------------------
def show_panel():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="close" onclick="window.location.reload()">×</div>',
                unsafe_allow_html=True)
    st.markdown("<h2>Discourse Analytics Panel</h2>", unsafe_allow_html=True)

    # Grid de botones
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        if st.button("1. AI Insights"):
            ai_insights()
    with col2:
        if st.button("2. Main Ideas"):
            main_ideas()
    with col3:
        if st.button("3. Content Gaps"):
            content_gaps()

    col4, col5, col6 = st.columns(3, gap="medium")
    with col4:
        if st.button("4. Relations"):
            relations()
    with col5:
        if st.button("5. Sentiment"):
            sentiment()
    with col6:
        if st.button("6. Stats"):
            stats()

    st.markdown('</div>', unsafe_allow_html=True)

# --------------- SECCIONES DE ANÁLISIS -------------
def ai_insights():
    st.subheader("AI Insights")
    st.info("Aquí insertas la síntesis automática: tendencias, alertas, etc.")

def main_ideas():
    st.subheader("Main Ideas")
    st.write("⮞ Idea 1: ...\n\n⮞ Idea 2: ...")

def content_gaps():
    st.subheader("Content Gaps")
    st.warning("Sinapsis de vacíos temáticos detectados...")

def relations():
    st.subheader("Relations")
    st.write("Ejemplo de gráfico Plotly (dummy):")
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_length", y="sepal_width",
                     color="species", height=350)
    st.plotly_chart(fig, use_container_width=True)

def sentiment():
    st.subheader("Sentiment")
    st.success("💬 Positivo 45 %   😐 Neutral 40 %   😞 Negativo 15 %")

def stats():
    st.subheader("Stats")
    st.metric("Total palabras", "12 543")
    st.metric("Códigos", "87")
    st.metric("Entrevistas", "8")

# ------------------- Render -----------------------
show_panel()
