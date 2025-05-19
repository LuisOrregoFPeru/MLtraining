import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ---------------------------------------------------------
# SUITE COMPLETA DE EVALUACIONES ECONÓMICAS EN SALUD – Versión 1.2
# Autor: Jarvis (ChatGPT)
# Mayo 2025 (fix: completar módulos CCA, CEA, CUA, CBA)
# ---------------------------------------------------------

st.set_page_config(page_title="Evaluaciones Económicas", layout="wide")
st.title("🩺💲 Suite de Evaluaciones Económicas en Salud")

TIPOS = [
    "1️⃣ COI • Costo de la Enfermedad",
    "2️⃣ BIA • Impacto Presupuestario",
    "3️⃣ ROI • Retorno sobre la Inversión",
    "4️⃣ CC  • Comparación de Costos",
    "5️⃣ CMA • Minimización de Costos",
    "6️⃣ CCA • Costo‑Consecuencia",
    "7️⃣ CEA • Costo‑Efectividad",
    "8️⃣ CUA • Costo‑Utilidad",
    "9️⃣ CBA • Costo‑Beneficio",
]
analisis = st.sidebar.radio("Selecciona el tipo de análisis", TIPOS)

# Función descarga CSV

def descarga_csv(df: pd.DataFrame, nombre: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Descargar CSV", csv, file_name=f"{nombre}.csv", mime="text/csv")

# 1) COI – Costo de la enfermedad 
elif analisis.startswith("1️⃣"):
    st.header("1️⃣ Costo de la Enfermedad (COI)")
    # Incluimos columna de variación porcentual editable
    coi_df = st.data_editor(
        pd.DataFrame({
            "Categoría": [
                "Directo médico", "Directo no médico", 
                "Indirecto (productividad)", "Intangible"
            ],
            "Costo anual": [0.0, 0.0, 0.0, 0.0],
            "Variación (%)": [20.0, 20.0, 20.0, 20.0]
        }),
        num_rows="dynamic",
        key="coi_tabla"
    )

    # Validación de valores negativos
    if (coi_df["Costo anual"] < 0).any() or (coi_df["Variación (%)"] < 0).any():
        st.error("Valores negativos no permitidos en costos o variaciones.")
    else:
        total = coi_df["Costo anual"].sum()
        st.success(f"Costo total anual: US$ {total:,.2f}")

        if total > 0:
            # Cálculo tornado: impacto univariado ±Variación (%) en cada categoría
            sens = []
            for _, row in coi_df.iterrows():
                cat = row["Categoría"]
                c   = row["Costo anual"]
                v   = row["Variación (%)"] / 100
                sens.append({
                    "Categoría": cat,
                    "Menos": -c * v,
                    "Más":  c * v
                })
            sens_df = pd.DataFrame(sens).set_index("Categoría")
            # Ordenar por magnitud de efecto
            sens_df = sens_df.reindex(sens_df["Más"].abs().sort_values().index)

            # Dibujar tornado
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(sens_df.index, sens_df["Menos"], color="skyblue")
            ax.barh(sens_df.index, sens_df["Más"],  color="orange")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Cambio en costo anual (US$)")
            ax.set_title("Análisis Tornado – COI")
            fig.tight_layout()
            st.pyplot(fig)

            # Botón para descargar tornado
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                "📥 Descargar gráfico Tornado",
                buf,
                file_name="COI_tornado.png",
                mime="image/png"
            )

        else:
            st.info("Introduce valores > 0 para el costo anual.")

    descarga_csv(coi_df.drop(columns="Variación (%)"), "COI_resultados")
    
# 2) BIA – Impacto Presupuestario
elif analisis.startswith("2️⃣"):
    st.header("2️⃣ Impacto Presupuestario (BIA)")
    delta = st.number_input("Δ Costo por paciente (US$)",1000.0)
    pop   = st.number_input("Población objetivo",10000)
    yrs   = st.number_input("Horizonte (años)",3)
    pag   = st.number_input("N pagadores/asegurados",500000)
    anual = delta*pop
    df   = pd.DataFrame({"Año":[f"Año {i+1}" for i in range(int(yrs))],"Costo incremental":[anual]*int(yrs)})
    df['Acumulado']=df['Costo incremental'].cumsum()
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.success(f"Acumulado en {yrs} años: US$ {df['Acumulado'].iloc[-1]:,.0f}")
    if pag>0: st.info(f"Impacto por pagador: US$ {anual/pag:,.2f}")
    fig,ax=plt.subplots(); ax.bar(df['Año'],df['Costo incremental']); st.pyplot(fig)
    descarga_csv(df,"BIA_resultados")

# 3) ROI – Retorno sobre la Inversión
elif analisis.startswith("3️⃣"):
    st.header("3️⃣ Retorno sobre la Inversión (ROI)")
    inv=st.number_input("Costo de inversión (US$)",50000.0)
    ben=st.number_input("Beneficio monetario (US$)",70000.0)
    roi = ((ben-inv)/inv*100) if inv else np.nan
    st.success(f"ROI: {roi:,.2f}%")
    fig,ax=plt.subplots(); ax.bar(['Inversión','Beneficio'],[inv,ben]); st.pyplot(fig)

# 4) CC – Comparación de Costos
elif analisis.startswith("4️⃣"):
    st.header("4️⃣ Comparación de Costos (CC)")
    df=st.data_editor(pd.DataFrame({'Alternativa':['A','B'],'Costo':[1000.0,1200.0]}),num_rows='dynamic',key='cc')
    if not df.empty:
        base=df['Costo'].iloc[0]
        df['Δ vs Base']=df['Costo']-base
        st.dataframe(df,hide_index=True)
        descarga_csv(df,'CC')

# 5) CMA – Minimización de Costos
elif analisis.startswith("5️⃣"):
    st.header("5️⃣ Minimización de Costos (CMA)")
    df=st.data_editor(pd.DataFrame({'Alt':['A','B'],'Costo':[1000.0,1200.0]}),num_rows='dynamic',key='cma')
    if not df.empty:
        m=df.loc[df['Costo'].idxmin()]
        st.success(f"Opción mínima: {m['Alt']} US$ {m['Costo']:,.2f}")
        descarga_csv(df,'CMA')

# 6) CCA – Costo‑Consecuencia
elif analisis.startswith("6️⃣"):
    st.header("6️⃣ Costo-Consecuencia (CCA)")
    # 1. Parámetros de entrada
    n_alt = st.number_input(
        "Número de alternativas", 
        value=2, min_value=2, step=1
    )
    vars_txt = st.text_input(
        "Variables de consecuencia (sep. por comas)", 
        value="QALYs, Hospitalizaciones"
    )
    vlist = [v.strip() for v in vars_txt.split(",") if v.strip()]

    # 2. Inicializar DataFrame con n_alt filas y columnas para cada variable
    data = {"Alternativa": [f"A{i+1}" for i in range(n_alt)]}
    for v in vlist:
        data[v] = [0.0] * n_alt
    df_cca = pd.DataFrame(data)

    # 3. Editor interactivo
    df_cca = st.data_editor(
        df_cca, 
        num_rows="dynamic", 
        key="cca"
    )

    # 4. Validación y salida
    if df_cca.empty:
        st.info("Agrega al menos una alternativa y una variable de consecuencia.")
    else:
        st.subheader("Tabla de Costo-Consecuencia")
        st.dataframe(df_cca, hide_index=True, use_container_width=True)
        descarga_csv(df_cca, "CCA_resultados")

# 7+8+9) CEA, CUA, CBA
else:
    # Definir tabla de tratamientos
    st.header(f"{analisis}")
    tx0=pd.DataFrame({'Tratamiento':['A','B','C'],'Costo total':[0,10000,22000],'Efectividad':[0,0.4,0.55]})
    tx=st.data_editor(tx0,num_rows='dynamic',key='tx')
    if tx.shape[0]>=2:
        df=tx.copy().reset_index(drop=True)
        df=df.sort_values('Costo total').reset_index(drop=True)
        df['ΔCosto']=df['Costo total'].diff()
        df['ΔEfect']=df['Efectividad'].diff()
        df['ICER']=df.apply(lambda r: r['ΔCosto']/r['ΔEfect'] if r['ΔEfect']>0 else np.nan,axis=1)
        st.subheader("Tabla incremental")
        st.dataframe(df,hide_index=True,use_container_width=True)
        # Gráfico CE plane
        fig,ax=plt.subplots(); ax.scatter(df['Efectividad'],df['Costo total']);
        for i,r in df.iterrows(): ax.annotate(r['Tratamiento'],(r['Efectividad'],r['Costo total']))
        st.pyplot(fig)
        descarga_csv(df,'CEA_CUA')
    else:
        st.info("Agregue al menos 2 tratamientos.")
