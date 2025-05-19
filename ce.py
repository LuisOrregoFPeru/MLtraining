import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SUITE COMPLETA DE EVALUACIONES ECONÓMICAS EN SALUD
# Autor: Jarvis (ChatGPT)
# Versión: 1.0 – mayo 2025
# ---------------------------------------------------------
# Cobertura
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

    total = coi_df["Costo anual"].sum()
    st.success(f"**Costo total anual:** US$ {total:,.2f}")

    # Pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(coi_df["Costo anual"], labels=coi_df["Categoría"], autopct="%1.1f%%")
    ax.set_title("Distribución de costos")
    st.pyplot(fig)

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
        "Costo incremental": [anual]*(int(años)),
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

    cca_df = st.data_editor(
        pd.DataFrame({"Alternativa": ["A"], "Costo": [1000.0], "Curaciones": [50]}),
        num_rows="dynamic",
        key="cca_tabla",
        use_container_width=True,
    )
    st.dataframe(cca_df, use_container_width=True, hide_index=True)
    descarga_csv(cca_df, "CCA_resultados")

# ──────────────────────────────────────────────────────────
# 7‑8) CEA / CUA – Costo‑efectividad y Costo‑utilidad
# ──────────────────────────────────────────────────────────
elif analisis.startswith("7️⃣") or analisis.startswith("8️⃣"):
    tipo = "CEA" if analisis.startswith("7️⃣") else "CUA"
    unidad = "Efect" if tipo == "CEA" else "QALY"
    st.header(f"{analisis}")

    # ----------------------- Entrada tratamientos ----------------------
    st.sidebar.subheader("Tratamientos & parámetros determinísticos")
    default_df = pd.DataFrame(
        {
            "Tratamiento": ["A (Base)", "B", "C", "D", "E"],
            "Costo total": [0.0, 10000.0, 22000.0, 25000.0, 40000.0],
            f"{unidad}s": [0.0, 0.40, 0.55, 0.50, 1.0],
        }
    )
    tx_df = st.sidebar.data_editor(default_df, num_rows="dynamic", key="cea_tabla", use_container_width=True)

    # ----------------------- Dominancia & tablas ----------------------
    def dom_tables(df: pd.DataFrame):
        df_sorted = df.sort_values(["Costo total", f"{unidad}s"], ascending=[True, False]).reset_index(drop=True)
        df_sorted["Dominancia"] = "Ninguna"

        # Fuerte dominancia
        for i in range(len(df_sorted)):
            for j in range(len(df_sorted)):
                if i == j:
                    continue
                cond1 = df_sorted.loc[j, "Costo total"] <= df_sorted.loc[i, "Costo total"]
                cond2 = df_sorted.loc[j, f"{unidad}s"] >= df_sorted.loc[i, f"{unidad}s"]
                better = df_sorted.loc[j, ["Costo total", f"{unidad}s"]].ne(df_sorted.loc[i, ["Costo total", f"{unidad}s"]]).any()
                if cond1 and cond2 and better:
                    df_sorted.loc[i, "Dominancia"] = "Fuerte"
                    break

        no_strong = df_sorted[df_sorted["Dominancia"] == "Ninguna"].copy()
        no_strong["ΔCosto"] = no_strong["Costo total"].diff()
        no_strong["ΔEfect"] = no_strong[f"{unidad}s"].diff()
        no_strong["ICER"] = no_strong["ΔCosto"] / no_strong["ΔEfect"]
        no_strong["ExtDominancia"] = "Ninguna"
        for i in range(1, len(no_strong) - 1):
            if no_strong.loc[i, "ICER"] > no_strong.loc[i + 1, "ICER"]:
                no_strong.loc[i, "ExtDominancia"] = "Extendida"
        final = no_strong[no_strong["ExtDominancia"] == "Ninguna"].copy()
        return df_sorted, no_strong, final

    tab0, tab1, tab2 = dom_tables(tx_df)

    tabs = st.tabs(["Cruda", "Sin dominados", "Sin ext. dominados"])
    tablas = [tab0, tab1, tab2]
    for t, df in zip(tabs, tablas):
        with t:
            df_show = df.copy()
            df_show.columns = [col.replace(f"{unidad}s", f"Total {unidad}s") for col in df_show.columns]
            st.dataframe(df_show, use_container_width=True, hide_index=True)
            descarga_csv(df_show, f"{tipo}_tabla_{t.title()}")

    # ----------------------- Gráfico incremental ----------------------
    st.subheader("Frente de eficiencia incremental")
    plot_df = tab0.copy()
    plot_df["Clase"] = plot_df.apply(lambda r: "Dominado" if r["Dominancia"] == "Fuerte" else ("ExtDominado" if r.get("ExtDominancia", "Ninguna") == "Extendida" else "Eficiente"), axis=1)
    styles = {"Eficiente": "o", "Dominado": "x", "ExtDominado": "^"}
    colors = {"Eficiente": "black", "Dominado": "gray", "ExtDominado": "darkgray"}

    figF, axF = plt.subplots()
    axF.set_xlabel(f"{unidad}s")
    axF.set_ylabel("Costo (US$)")

    for _, row in plot_df.iterrows():
        axF.scatter(row[f"{unidad}s"], row["Costo total"], marker=styles[row["Clase"]], color=colors[row["Clase"]])
        axF.annotate(row["Tratamiento"], (row[f"{unidad}s"], row["Costo total"]), textcoords="offset points", xytext=(5, -7))

    eff_pts = plot_df[plot_df["Clase"] == "Eficiente"].sort_values(f"{unidad}s")
    axF.plot(eff_pts[f"{unidad}s"], eff_pts["Costo total"], linestyle="-", color="black")

    for i in range(1, len(eff_pts)):
        x0, y0 = eff_pts.iloc[i - 1][[f"{unidad}s", "Costo total"]]
        x1, y1 = eff_pts.iloc[i][[f"{unidad}s", "Costo total"]]
        icer = (y1 - y0) / (x1 - x0)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        axF.plot([x0, x1], [y0, y1], linestyle="--", color="gray")
        axF.annotate(f"{icer:,.0f}/{unidad}", (xm, ym), textcoords="offset points", xytext=(0, -15), ha="center")

    st.pyplot(figF)

    # ----------------------- Plano clásico ----------------------
    st.subheader("Plano de Costo‑Efectividad vs. Umbral λ")
    umbral = st.sidebar.number_input("Umbral λ", value=30000.0, step=1000.0)

    figP, axP = plt.subplots()
    axP.axhline(0, color="gray", linewidth=0.5)
    axP.axvline(0, color="gray", linewidth=0.5)
    x_line = np.linspace(-1, 1, 100)
    axP.plot(x_line, umbral * x_line, linestyle="--", label="Umbral λ")
    axP.set_xlabel(f"Δ{unidad}")
    axP.set_ylabel("ΔCosto (US$)")

    base = tx_df.iloc[0]
    for _, row in tx_df.iterrows():
        if row.name == 0:
            continue
        d_c = row["Costo total"] - base["Costo total"]
        d_e = row[f"{unidad}s"] - base[f"{unidad}s"]
        axP.scatter(d_e, d_c)
        axP.annotate(row["Tratamiento"], (d_e, d_c))
    axP.legend()
    st.pyplot(figP)

# ──────────────────────────────────────────────────────────
# 9) CBA – Costo‑beneficio
# ──────────────────────────────────────────────────────────
else:
    st.header("9️⃣ Costo‑Beneficio (CBA)")
    cba_df = st.data_editor(
        pd.DataFrame({"Alternativa": ["A"], "Costo": [10000.0], "Beneficio": [15000.0]}),
        num_rows="dynamic",
        key="cba_tabla",
        use_container_width=True,
    )
    cba_df["Beneficio neto"] = cba_df["Beneficio"] - cba_df["Costo"]
    cba_df["Ratio B/C"] = cba_df["Beneficio"] / cba_df["Costo"]
    st.dataframe(cba_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots()
    ax.bar(cba_df["Alternativa"], cba_df["Ratio B/C"])
    ax.axhline(1, color="gray", linestyle="--")
    ax.set_ylabel("Beneficio/Costo")
    st.pyplot(fig)
    descarga_csv(cba_df, "CBA_resultados")

