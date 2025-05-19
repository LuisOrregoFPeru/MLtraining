import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SUITE COMPLETA DE EVALUACIONES ECONÓMICAS EN SALUD
# Autor: Jarvis (ChatGPT)
# Versión: 1.1 – mayo 2025 (fix gráfico COI y validaciones adicionales)
# ---------------------------------------------------------
# Cobertura de análisis
#   1. COI   – Costo de la enfermedad
#   2. BIA   – Impacto presupuestario
#   3. ROI   – Retorno sobre la inversión
#   4. CC    – Comparación de costos
#   5. CMA   – Minimización de costos
#   6. CCA   – Costo‑consecuencia
#   7. CEA   – Costo‑efectividad
#   8. CUA   – Costo‑utilidad
#   9. CBA   – Costo‑beneficio
# Funcionalidades generales
#   • Tabla editable interactiva en cada módulo.
#   • Cálculos automáticos y validaciones básicas.
#   • Gráficos apropiados (pie, barras, plano CE, tornado, etc.).
#   • Botón de descarga CSV del resultado resumido.
# ---------------------------------------------------------

st.set_page_config(page_title="Evaluaciones Económicas", layout="wide")

st.title("🩺💲 Suite de Evaluaciones Económicas en Salud")

TIPOS = [
    "1️⃣ COI • Costo de la Enfermedad",
    "2️⃣ BIA • Impacto Presupuestario",
    "3️⃣ ROI • Retorno sobre la Inversión",
    "4️⃣ CC  • Comparación de Costos",
    "5️⃣ CMA • Minimización de Costos",
    "6️⃣ CCA • Costo‑Consecuencia",
    "7️⃣ CEA • Costo‑Efectividad",
    "8️⃣ CUA • Costo‑Utilidad",
    "9️⃣ CBA • Costo‑Beneficio",
]

analisis = st.sidebar.radio("Selecciona el tipo de análisis", TIPOS)

# Función utilitaria para descarga

def descarga_csv(df: pd.DataFrame, nombre: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Descargar CSV", csv, file_name=f"{nombre}.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────
# 1) COI – Costo de la enfermedad
# ──────────────────────────────────────────────────────────
if analisis.startswith("1️⃣"):
    st.header("1️⃣ Costo de la Enfermedad (COI)")
    st.write("Desglosa los costos anuales por categoría para estimar la carga económica de la enfermedad.")

    coi_df = st.data_editor(
        pd.DataFrame(
            {
                "Categoría": [
                    "Directo médico",
                    "Directo no médico",
                    "Indirecto (productividad)",
                    "Intangible (dolor/ansiedad)",
                ],
                "Costo anual": [0.0, 0.0, 0.0, 0.0],
            }
        ),
        num_rows="dynamic",
        key="coi_tabla",
        use_container_width=True,
    )

    # Validación de valores negativos
    if (coi_df["Costo anual"] < 0).any():
        st.error("Existen valores de costo negativos. Corríjalos para continuar.")
    else:
        total = coi_df["Costo anual"].sum()
        st.success(f"**Costo total anual:** US$ {total:,.2f}")

        # Mostrar gráfico solo si hay valores positivos
        if total > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(
                coi_df["Costo anual"],
                labels=coi_df["Categoría"],
                autopct="%1.1f%%",
            )
            ax.set_title("Distribución de costos")
            st.pyplot(fig)
        else:
            st.info("Introduzca valores mayores que cero para visualizar la distribución de costos.")

    descarga_csv(coi_df, "COI_resultados")

# ──────────────────────────────────────────────────────────
# 2) BIA – Impacto presupuestario
# ──────────────────────────────────────────────────────────
elif analisis.startswith("2️⃣"):
    st.header("2️⃣ Impacto Presupuestario (BIA)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_cost = st.number_input("Δ Costo por paciente (US$)", value=1000.0, step=100.0)
    with col2:
        pop = st.number_input("Población objetivo", value=10000, step=1000)
    with col3:
        años = st.number_input("Años de horizonte", value=3, step=1)
    with col4:
        pagadores = st.number_input("N.º de pagadores/asegurados", value=500000, step=10000)

    anual = delta_cost * pop
    tabla = pd.DataFrame({
        "Año": [f"Año {i+1}" for i in range(int(años))],
        "Costo incremental": [anual] * int(años),
    })
    tabla["Acumulado"] = tabla["Costo incremental"].cumsum()

    st.dataframe(tabla, use_container_width=True, hide_index=True)
    st.success(f"**Costo acumulado ({años} años):** US$ {tabla['Acumulado'].iloc[-1]:,.0f}")
    impacto = anual / pagadores if pagadores else 0
    st.info(f"**Impacto anual por pagador:** US$ {impacto:,.2f}")

    fig, ax = plt.subplots()
    ax.bar(tabla["Año"], tabla["Costo incremental"])
    ax.set_ylabel("Costo (US$)")
    st.pyplot(fig)

    descarga_csv(tabla, "BIA_resultados")

# ──────────────────────────────────────────────────────────
# 3) ROI – Retorno sobre la inversión
# ──────────────────────────────────────────────────────────
elif analisis.startswith("3️⃣"):
    st.header("3️⃣ Retorno sobre la Inversión (ROI)")
    col1, col2 = st.columns(2)
    with col1:
        costo_inv = st.number_input("Costo de la inversión (US$)", value=50000.0, step=1000.0)
    with col2:
        beneficio = st.number_input("Beneficio monetario (US$)", value=70000.0, step=1000.0)

    roi_pct = ((beneficio - costo_inv) / costo_inv) * 100 if costo_inv else 0
    st.success(f"**ROI:** {roi_pct:,.2f}%")

    fig, ax = plt.subplots()
    ax.bar(["Costo", "Beneficio"], [costo_inv, beneficio], color=["#d62728", "#2ca02c"])
    ax.set_ylabel("US$")
    st.pyplot(fig)

# ──────────────────────────────────────────────────────────
# 4) CC – Comparación de Costos
# ──────────────────────────────────────────────────────────
elif analisis.startswith("4️⃣"):
    st.header("4️⃣ Comparación de Costos (CC)")
    cc_df = st.data_editor(
        pd.DataFrame({"Alternativa": ["A", "B"], "Costo": [1000.0, 1200.0]}),
        num_rows="dynamic",
        key="cc_tabla",
        use_container_width=True,
    )
    base = cc_df.iloc[0]["Costo"] if not cc_df.empty else 0
    cc_df["ΔCosto vs. Base"] = cc_df["Costo"] - base
    st.dataframe(cc_df, use_container_width=True, hide_index=True)
    descarga_csv(cc_df, "CC_resultados")

# ──────────────────────────────────────────────────────────
# 5) CMA – Minimización de Costos
# ──────────────────────────────────────────────────────────
elif analisis.startswith("5️⃣"):
    st.header("5️⃣ Minimización de Costos (CMA)")
    cma_df = st.data_editor(
        pd.DataFrame({"Alternativa": ["A", "B"], "Costo": [1000.0, 1200.0]}),
        num_rows="dynamic",
        key="cma_tabla",
        use_container_width=True,
    )
    if not cma_df.empty:
        min_row = cma_df.loc[cma_df["Costo"].idxmin()]
        st.success(f"**Opción más económica:** {min_row['Alternativa']} – US$ {min_row['Costo']:,.2f}")
    descarga_csv(cma_df, "CMA_resultados")

# ──────────────────────────────────────────────────────────
# 6) CCA – Costo‑consecuencia
# ──────────────────────────────────────────────────────────
elif analisis.startswith("6️⃣"):
    st.header("6️⃣ Costo‑Consecuencia (CCA)")
    st.write("Añade las distintas consecuencias en columnas independientes.")

    cca
