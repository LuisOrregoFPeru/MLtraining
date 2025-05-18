import streamlit as st
import requests
import xml.etree.ElementTree as ET
import datetime

st.set_page_config(page_title="Buscador PubMed", page_icon="🧬")
st.title("🔬 Buscador de artículos científicos en PubMed")
st.markdown("Busca artículos publicados en los últimos 5 años a partir de una palabra clave.")

# Cálculo del rango de fechas
fecha_actual = datetime.date.today()
anio_actual = fecha_actual.year
anio_limite = anio_actual - 5
fecha_limite = f"{anio_limite}/01/01"

# Entrada del usuario
consulta = st.text_input("🔑 Palabra clave para búsqueda")

# Botón de búsqueda
if st.button("Buscar"):
    if not consulta:
        st.warning("Por favor, ingresa una palabra clave.")
    else:
        st.info("Buscando artículos, por favor espera...")

        # Paso 1: Buscar IDs en PubMed
        url_search = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term={consulta}&retmax=10&retmode=json&mindate={fecha_limite}&datetype=pdat"
        )
        respuesta = requests.get(url_search)
        if respuesta.status_code == 200:
            ids = respuesta.json()["esearchresult"]["idlist"]
            if not ids:
                st.warning("No se encontraron artículos con esa búsqueda.")
            else:
                # Paso 2: Obtener detalles de los artículos
                id_string = ",".join(ids)
                url_summary = (
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
                    f"db=pubmed&id={id_string}&retmode=json"
                )
                resumenes = requests.get(url_summary).json()["result"]

                for uid in ids:
                    art = resumenes[uid]
                    titulo = art.get("title", "Sin título")
                    autores = ", ".join([a["name"] for a in art.get("authors", [])]) or "Sin autores"
                    revista = art.get("source", "Revista desconocida")
                    anio = art.get("pubdate", "Fecha desconocida")
                    enlace = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"

                    st.markdown(f"""
                    **Título**: {titulo}  
                    **Autores**: {autores}  
                    **Revista**: *{revista}*  
                    **Año**: {anio}  
                    [🔗 Ver en PubMed]({enlace})  
                    ---
                    """)
        else:
            st.error("Ocurrió un error en la búsqueda. Intenta nuevamente.")
