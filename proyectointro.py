import streamlit as st
import datetime

st.set_page_config(page_title="Asistente de Tesis", layout="wide")
st.title("📝 Asistente para la Elaboración de Proyectos de Tesis")

st.markdown("""
Esta aplicación te ayudará a estructurar el capítulo de **Introducción** de tu tesis. A partir de un título propuesto, podrás redactar los elementos fundamentales del primer capítulo según las buenas prácticas académicas. 

Asegúrate de escribir en **prosa**, en **tercera persona** y en **tiempo pasado del modo indicativo**.
""")

# Entrada del título del proyecto
titulo = st.text_input("🎓 Ingresa el título del proyecto de tesis")

st.subheader("1. Introducción")
st.markdown("Redacta tu introducción en el cuadro a continuación. Puedes guiarte con los elementos propuestos.")

intro = st.text_area("✍️ Introducción (mínimo 9 páginas A4 en redacción final):", height=400)

with st.expander("🧩 Guía para redactar la introducción"):
    st.markdown("""
    - Exponer la **realidad problemática** de forma general, incluyendo datos sobre frecuencia y morbimortalidad global, en América Latina y Perú.
    - Presentar **antecedentes científicos relevantes** que fundamenten la pregunta de investigación.
    - Identificar **vacíos o limitaciones en la literatura existente**.
    - Explicar cómo el estudio **abordará esas limitaciones**.
    - Discutir la **utilidad de resolver la pregunta de investigación**.
    - Relacionar el tema con los **Objetivos de Desarrollo Sostenible (ODS)** pertinentes.
    """)

st.subheader("2. Problema de investigación")
problema = st.text_area("❓ Formula el problema de investigación en modo interrogativo")

st.subheader("3. Justificación")
justificacion_teorica = st.text_area("📚 Justificación teórica")
justificacion_practica = st.text_area("🛠 Justificación práctica")
justificacion_metodologica = st.text_area("🔬 Justificación metodológica")
justificacion_social = st.text_area("👥 Justificación social")

st.subheader("4. Objetivos")
objetivo_general = st.text_area("🎯 Objetivo general")
objetivos_especificos = st.text_area("📌 Objetivos específicos (uno por línea)")

st.subheader("5. Revisión de Literatura")
revision_literatura = st.text_area("🔍 Síntesis de antecedentes nacionales e internacionales (PubMed, Scopus, WoS, SciELO)", height=300)

st.subheader("6. Fundamentación teórica y conceptual")
teorias_relacionadas = st.text_area("📖 Teorías relacionadas al tema")
enfoques_conceptuales = st.text_area("🧠 Enfoques conceptuales vinculados a las variables")

st.subheader("7. Hipótesis (si aplica)")
hipotesis = st.text_area("🔬 Hipótesis de investigación")

# Exportación o resumen
if st.button("📄 Generar vista previa del capítulo"):
    st.markdown("---")
    st.subheader("🧾 Vista Previa del Capítulo 1: Introducción")
    st.markdown(f"**Título del proyecto:** {titulo}")
    st.markdown(f"**Introducción:**\n{intro}")
    st.markdown(f"**Problema de investigación:**\n{problema}")
    st.markdown("**Justificación:**")
    st.markdown(f"- Teórica: {justificacion_teorica}")
    st.markdown(f"- Práctica: {justificacion_practica}")
    st.markdown(f"- Metodológica: {justificacion_metodologica}")
    st.markdown(f"- Social: {justificacion_social}")
    st.markdown(f"**Objetivo general:**\n{objetivo_general}")
    st.markdown(f"**Objetivos específicos:**\n{objetivos_especificos}")
    st.markdown(f"**Revisión de literatura:**\n{revision_literatura}")
    st.markdown(f"**Teorías relacionadas:**\n{teorias_relacionadas}")
    st.markdown(f"**Enfoques conceptuales:**\n{enfoques_conceptuales}")
    st.markdown(f"**Hipótesis:**\n{hipotesis if hipotesis else 'No aplica.'}")
