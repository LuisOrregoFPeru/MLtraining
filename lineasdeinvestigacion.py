import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Líneas Nacionales de Investigación en Salud al 2030", layout="wide")
st.title("🧬 Líneas Nacionales de Investigación en Salud al 2030")
st.markdown("""
Esta aplicación permite explorar de forma interactiva las líneas de investigación en salud propuestas para el Perú al 2030.
Puedes filtrar por **problema sanitario** y ver los **objetivos estratégicos (OE)** relacionados.
""")

# Cargar texto plano del documento transcrito
with open("lineas_investigacion_salud_2030.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Extraer bloques por problema sanitario y objetivos
pattern = r"\((ED|SS|CVT)\) Problema sanitario: (.*?)\n(.*?)(?=\(ED|\(SS|\(CVT|\Z)"
matches = re.findall(pattern, raw_text, re.DOTALL)

# Estructurar en DataFrame
data = []
for tipo, problema, contenido in matches:
    objetivos = re.findall(r"(OE \d+): (.*?)\n(?=(OE \d+:|\Z))", contenido, re.DOTALL)
    for i, (oe_id, oe_desc, _) in enumerate(objetivos):
        data.append({
            "Tipo": tipo,
            "Problema Sanitario": problema.strip(),
            "OE": oe_id,
            "Descripción del OE": oe_desc.strip()
        })

df = pd.DataFrame(data)

# Sidebar para selección de problema sanitario
problemas = df["Problema Sanitario"].unique()
seleccion = st.sidebar.selectbox("Selecciona un problema sanitario:", sorted(problemas))

# Filtrar y mostrar resultados
st.subheader(f"🎯 Objetivos Estratégicos para: {seleccion}")
filtrado = df[df["Problema Sanitario"] == seleccion]
st.dataframe(filtrado.drop(columns=["Tipo"]).reset_index(drop=True), use_container_width=True)

# Opción para descargar CSV
csv = filtrado.to_csv(index=False).encode('utf-8')
st.download_button("📥 Descargar objetivos en CSV", data=csv, file_name="objetivos_salud2030.csv", mime='text/csv')
