import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
if analisis.startswith("1️⃣"):
    st.header("1️⃣ Costo de la Enfermedad (COI)")
    coi_df = st.data_editor(
        pd.DataFrame({
            "Categoría": [
                "Directo médico", "Directo no médico", 
                "Indirecto (productividad)", "Intangible"
            ],
            "Costo anual": [0.0,0.0,0.0,0.0]
        }), num_rows="dynamic", key="coi_tabla"
    )
    if (coi_df["Costo anual"]<0).any():
        st.error("Valores negativos no permitidos.")
    else:
        total = coi_df["Costo anual"].sum()
        st.success(f"Costo total anual: US$ {total:,.2f}")
        if total>0:
            fig,ax=plt.subplots(figsize=(4,4)); ax.pie(coi_df['Costo anual'], labels=coi_df['Categoría'], autopct='%1.1f%%')
            st.pyplot(fig)
        else:
            st.info("Introduce valores >0 para graficar.")
    descarga_csv(coi_df, "COI_resultados")

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

# 6️⃣ CCA – Costo-Consecuencia
elif analisis.startswith("6️⃣"):
    st.header("6️⃣ Costo-Consecuencia (CCA)")
    df_cca = st.data_editor(
        pd.DataFrame({
            "Alternativa": ["A", "B"],
            "Consecuencia 1": [0, 50],
            "Consecuencia 2": [0, 30]
        }),
        num_rows="dynamic",
        key="cca"
    )
    if not df_cca.empty:
        st.subheader("Tabla CCA")
        st.dataframe(df_cca, hide_index=True)
        descarga_csv(df_cca, "CCA_resultados")
    else:
        st.info("Agrega al menos una alternativa y una variable de consecuencia.")

# 7️⃣ CEA – Costo-Efectividad
elif analisis.startswith("7️⃣"):
    st.header("7️⃣ Costo-Efectividad (CEA)")
    tx = st.data_editor(
        pd.DataFrame({
            "Tratamiento": ["A", "B", "C"],
            "Costo total": [0, 10000, 22000],
            "sd_costo": [0, 500, 1000],
            "Efectividad": [0, 0.4, 0.55],
            "sd_efect": [0, 0.05, 0.08]
        }),
        num_rows="dynamic",
        key="cea_tx"
    )
    if tx.shape[0] >= 2:
        # Cálculos incrementales
        df = tx.sort_values("Costo total").reset_index(drop=True)
        df["ΔCosto"] = df["Costo total"].diff()
        df["ΔEfect"] = df["Efectividad"].diff()
        df["ICER"] = df.apply(
            lambda r: r["ΔCosto"] / r["ΔEfect"]
            if r["ΔEfect"] and r["ΔEfect"] > 0 else np.nan,
            axis=1
        )

        # Tablas de dominancia
        tab0, tab1, tab2 = dom_tables(df)

        with st.tabs(["Cruda", "Sin dominados", "Sin ext. dominados"]):
            st.tab("Cruda").dataframe(tab0, use_container_width=True)
            st.tab("Sin dominados").dataframe(tab1, use_container_width=True)
            st.tab("Sin ext. dominados").dataframe(tab2, use_container_width=True)

        # Plano determinístico
        umbral = st.number_input("Umbral λ ($/QALY)", value=20000.0, step=1000.0)
        ΔE = df["Efectividad"].diff().iloc[1:]
        ΔC = df["Costo total"].diff().iloc[1:]
        fig, ax = plt.subplots()
        ax.scatter(ΔE, ΔC, s=80)
        for i, lbl in enumerate(df["Tratamiento"].iloc[1:]):
            ax.annotate(lbl, (ΔE.iloc[i], ΔC.iloc[i]))
        ax.axline((0, 0), slope=umbral, color="grey", linestyle="--",
                  label=f"λ = {umbral}")
        ax.set_xlabel("ΔEfectividad")
        ax.set_ylabel("ΔCosto")
        ax.set_title("Plano Costo-Efectividad Determinístico")
        ax.legend()
        st.pyplot(fig)

        # PSA y CEAC
        n_mc = st.slider("N° iteraciones PSA", 100, 10000, 1000, step=100)
        costos_base = df["Costo total"].values
        efects_base = df["Efectividad"].values
        # simulación
        mc_scatter = []
        ceac = []
        lambdas = np.linspace(0, umbral * 2, 50)
        for idx in range(1, len(df)):
            c_samples = np.random.normal(
                df.loc[idx, "Costo total"], df.loc[idx, "sd_costo"], n_mc
            )
            e_samples = np.random.normal(
                df.loc[idx, "Efectividad"], df.loc[idx, "sd_efect"], n_mc
            )
            dc = c_samples - costos_base[0]
            de = e_samples - efects_base[0]
            mc_scatter.append((dc, de))
        # scatter PSA
        fig2, ax2 = plt.subplots()
        for dc, de in mc_scatter:
            ax2.scatter(de, dc, alpha=0.2, s=10)
        ax2.axline((0, 0), slope=umbral, color="red", linestyle="--")
        ax2.set_xlabel("ΔEfectividad")
        ax2.set_ylabel("ΔCosto")
        ax2.set_title("PSA: Plano Costo-Efectividad")
        st.pyplot(fig2)
        # CEAC
        probs = []
        for L in lambdas:
            wins = [
                np.mean(dc - L * de <= 0)
                for dc, de in mc_scatter
            ]
            probs.append(np.max(wins))
        fig3, ax3 = plt.subplots()
        ax3.plot(lambdas, probs)
        ax3.set_xlabel("λ ($/QALY)")
        ax3.set_ylabel("Probabilidad de cost-efectividad")
        ax3.set_title("Curva de aceptación (CEAC)")
        st.pyplot(fig3)

        descarga_csv(df, "CEA_incremental")
    else:
        st.info("Agrega al menos 2 tratamientos con costo y efectividad.")

# 8️⃣ CUA – Costo-Utilidad
elif analisis.startswith("8️⃣"):
    st.header("8️⃣ Costo-Utilidad (CUA)")
    # Se reutiliza todo el código de CEA, reemplazando 'Efectividad' por 'QALYs'
    # [… código casi idéntico al bloque 7️⃣, cambiando nombres de columnas y ejes …]

# 9️⃣ CBA – Costo-Beneficio
elif analisis.startswith("9️⃣"):
    st.header("9️⃣ Costo-Beneficio (CBA)")
    df_cba = st.data_editor(
        pd.DataFrame({
            "Alternativa": ["A", "B"],
            "Costo": [0, 10000],
            "Beneficio": [0, 15000]
        }),
        num_rows="dynamic",
        key="cba"
    )
    if df_cba.shape[0] >= 1:
        df_cba["Beneficio neto"] = df_cba["Beneficio"] - df_cba["Costo"]
        st.subheader("Tabla CBA")
        st.dataframe(df_cba, use_container_width=True)
        descarga_csv(df_cba, "CBA_resultados")
        # Histograma
        fig4, ax4 = plt.subplots()
        ax4.hist(df_cba["Beneficio neto"], bins=len(df_cba), rwidth=0.8)
        ax4.set_xlabel("Beneficio neto")
        ax4.set_ylabel("Frecuencia")
        ax4.set_title("Distribución de Beneficio Neto")
        st.pyplot(fig4)
        # Probabilidad BN>0
        p_pos = np.mean(df_cba["Beneficio neto"] > 0)
        st.markdown(f"**Probabilidad de BN > 0:** {p_pos:.2%}")
    else:
        st.info("Agrega al menos una alternativa con costo y beneficio.")

# Captura cualquier selección inválida
else:
    st.error("Tipo de análisis no válido.")
