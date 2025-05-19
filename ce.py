import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Suite de Evaluaciones Económicas en Salud – Streamlit
# Autor: Jarvis (ChatGPT)
# Basado en Drabo et al. 2023 y Gray & Briggs 2019
# ---------------------------------------------------------
# Permite elegir entre 9 tipos de análisis económicos y
# despliega una interfaz específica para cada uno:
#   • COI  – Análisis del costo de la enfermedad
#   • BIA  – Análisis de impacto presupuestario
#   • ROI  – Retorno sobre la inversión
#   • CC   – Comparación de costos
#   • CMA  – Minimización de costos
#   • CCA  – Costo‑consecuencia
#   • CEA  – Costo‑efectividad
#   • CUA  – Costo‑utilidad (extiende CEA)
#   • CBA  – Costo‑beneficio
# Para CEA/CUA incorpora PSA y tornado con desglose de costos.
# ---------------------------------------------------------

st.set_page_config(page_title="Evaluaciones Económicas", layout="wide")

st.title("Suite de Evaluaciones Económicas en Salud")

TIPOS = [
    "Análisis del Costo de la Enfermedad (COI)",
    "Análisis de Impacto Presupuestario (BIA)",
    "Retorno sobre la Inversión (ROI)",
    "Comparación de Costos (CC)",
    "Minimización de Costos (CMA)",
    "Costo‑Consecuencia (CCA)",
    "Costo‑Efectividad (CEA)",
    "Costo‑Utilidad (CUA)",
    "Costo‑Beneficio (CBA)",
]

analisis = st.sidebar.selectbox("Selecciona el tipo de análisis", TIPOS)

# ──────────────────────────────────────────────────────────
# 1) COI – Costo de la enfermedad
# ──────────────────────────────────────────────────────────
if analisis == TIPOS[0]:
    st.header("Análisis del Costo de la Enfermedad (COI)")
    st.write("Ingresa las categorías de costo y su valor anual.")

    coi_df = st.data_editor(
        pd.DataFrame(
            {
                "Categoría": [
                    "Directo médico",
                    "Directo no médico",
                    "Indirecto (productividad)",
                    "Intangible (dolor/ansiedad)",
                ],
                "Costo_anual": [0.0, 0.0, 0.0, 0.0],
            }
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="coi_tabla",
    )

    total = coi_df["Costo_anual"].sum()
    st.success(f"**Costo total anual de la enfermedad:** US$ {total:,.2f}")

# ──────────────────────────────────────────────────────────
# 2) BIA – Impacto presupuestario
# ──────────────────────────────────────────────────────────
elif analisis == TIPOS[1]:
    st.header("Análisis de Impacto Presupuestario (BIA)")

    col1, col2, col3 = st.columns(3)
    with col1:
        delta_cost = st.number_input("Δ Costo por paciente (US$)", value=1000.0, step=100.0)
    with col2:
        pop = st.number_input("Población objetivo", value=10000, step=1000)
    with col3:
        pagadores = st.number_input("N.º de pagadores/asegurados", value=500000, step=10000)

    total_cost = delta_cost * pop
    impacto = total_cost / pagadores if pagadores else 0

    st.success(f"**Costo total del programa:** US$ {total_cost:,.0f}")
    st.info(f"**Impacto por pagador:** US$ {impacto:,.2f}")
    st.caption("El horizonte temporal debe corresponder al periodo presupuestario (1‑3 años) y los costos no se descuentan.")

# ──────────────────────────────────────────────────────────
# 3) ROI – Retorno sobre la inversión
# ──────────────────────────────────────────────────────────
elif analisis == TIPOS[2]:
    st.header("Retorno sobre la Inversión (ROI)")
    col1, col2 = st.columns(2)
    with col1:
        costo_inv = st.number_input("Costo de la inversión (US$)", value=50000.0, step=1000.0)
    with col2:
        beneficio = st.number_input("Beneficio monetario (US$)", value=70000.0, step=1000.0)

    roi = ((beneficio - costo_inv) / costo_inv) * 100 if costo_inv else 0
    st.success(f"**ROI:** {roi:,.2f}%")
    st.caption("Un ROI positivo indica una inversión rentable.")

# ──────────────────────────────────────────────────────────
# 4) Comparación de Costos (CC)
# ──────────────────────────────────────────────────────────
elif analisis == TIPOS[3]:
    st.header("Comparación de Costos (CC)")

    cc_df = st.data_editor(
        pd.DataFrame({"Alternativa": ["A", "B"], "Costo": [1000.0, 1200.0]}),
        num_rows="dynamic",
        use_container_width=True,
        key="cc_tabla",
    )

    comparador = cc_df.loc[0, "Costo"] if not cc_df.empty else 0
    diffs = cc_df["Costo"] - comparador
    cc_df["ΔCosto vs. Alt.0"] = diffs
    st.dataframe(cc_df, use_container_width=True)

# ──────────────────────────────────────────────────────────
# 5) CMA – Minimización de costos
# ──────────────────────────────────────────────────────────
elif analisis == TIPOS[4]:
    st.header("Minimización de Costos (CMA)")

    cma_df = st.data_editor(
        pd.DataFrame({"Alternativa": ["A", "B"], "Costo": [1000.0, 1200.0]}),
        num_rows="dynamic",
        use_container_width=True,
        key="cma_tabla",
    )

    min_row = cma_df.loc[cma_df["Costo"].idxmin()] if not cma_df.empty else None
    if min_row is not None:
        st.success(f"**Alternativa de menor costo:** {min_row['Alternativa']} (US$ {min_row['Costo']:,.2f})")

# ──────────────────────────────────────────────────────────
# 6) CCA – Costo‑consecuencia
# ──────────────────────────────────────────────────────────
elif analisis == TIPOS[5]:
    st.header("Costo‑Consecuencia (CCA)")
    st.write("Lista cada alternativa, su costo y los resultados clínicos/naturales relevantes.")

    cca_df = st.data_editor(
        pd.DataFrame({"Alternativa": ["A"], "Costo": [1000.0], "Resultado_1": [50]}),
        num_rows="dynamic",
        use_container_width=True,
        key="cca_tabla",
    )
    st.dataframe(cca_df, use_container_width=True)
    st.caption("El CCA no pondera los resultados; la interpretación queda al decisor.")

# ──────────────────────────────────────────────────────────
# 7‑8) CEA / CUA – Cost‑effectiveness & Cost‑utility
#     Incluye PSA y Tornado con desglose de costos
# ──────────────────────────────────────────────────────────
elif analisis in [TIPOS[6], TIPOS[7]]:
    st.header(analisis)

    # ----------------------- Tabla de tratamientos ----------------------
    st.sidebar.subheader("Datos de tratamientos")
    default_df = pd.DataFrame(
        {
            "Tratamiento": ["Comparador", "Intervención"],
            "Costo_total": [1000.0, 1500.0],
            "Efectividad": [0.70, 0.85],
            "SD_Costo": [100.0, 120.0],
            "SD_Efect": [0.07, 0.08],
        }
    )

    tx_df = st.sidebar.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        key="cea_tabla",
    )

    umbral = st.sidebar.number_input("Umbral λ (US$/ΔEfect)", value=5000.0, step=100.0)
    n_iter = st.sidebar.slider("Iteraciones PSA", 100, 10000, 1000, 100)

    base = tx_df.iloc[0]
    resultados = []
    for idx, row in tx_df.iterrows():
        if idx == 0:
            continue
        d_c = row["Costo_total"] - base["Costo_total"]
        d_e = row["Efectividad"] - base["Efectividad"]
        icer = d_c / d_e if d_e != 0 else np.nan
        resultados.append((row["Tratamiento"], d_c, d_e, icer))

    res_df = pd.DataFrame(resultados, columns=["Tratamiento", "ΔCosto", "ΔEfectividad", "ICER"])
    st.subheader("Resultados determinísticos")
    st.dataframe(res_df, use_container_width=True)

    # ----------------------- Plano determinístico ----------------------
    fig1, ax1 = plt.subplots()
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.set_xlabel("ΔEfectividad")
    ax1.set_ylabel("ΔCosto (US$)")

    x_line = np.linspace(-0.3, 0.3, 100)
    ax1.plot(x_line, umbral * x_line, linestyle="--", label=f"Umbral = {umbral:,.0f}")

    for _, r in res_df.iterrows():
        ax1.scatter(r["ΔEfectividad"], r["ΔCosto"])
        ax1.annotate(r["Tratamiento"], (r["ΔEfectividad"], r["ΔCosto"]))

    ax1.legend()
    st.subheader("Plano de Costo‑Efectividad (determinístico)")
    st.pyplot(fig1)

    # ----------------------- PSA ----------------------
    psa_points = {}
    for _, row in tx_df.iterrows():
        cost_sim = np.random.normal(row["Costo_total"], row["SD_Costo"], n_iter)
        eff_sim = np.random.normal(row["Efectividad"], row["SD_Efect"], n_iter)
        psa_points[row["Tratamiento"]] = (cost_sim, eff_sim)

    base_c_sim, base_e_sim = psa_points[base["Tratamiento"]]

    fig2, ax2 = plt.subplots()
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.axvline(0, color="gray", linewidth=0.5)
    ax2.set_xlabel("ΔEfectividad")
    ax2.set_ylabel("ΔCosto (US$)")
    ax2.plot(x_line, umbral * x_line, linestyle="--", label="Umbral")

    for name, (c_sim, e_sim) in psa_points.items():
        if name == base["Tratamiento"]:
            continue
        ax2.scatter(e_sim - base_e_sim, c_sim - base_c_sim, alpha=0.05, s=10, label=name)

    ax2.legend()
    st.subheader("Plano de PSA")
    st.pyplot(fig2)

    # ----------------------- Tornado ----------------------
    st.subheader("Diagrama Tornado – Desglose de Costos")
    sel_tx = st.selectbox("Tratamiento a analizar", tx_df["Tratamiento"].tolist()[1:])

    def make_cost_df(total):
        return pd.DataFrame(
            {
                "Componente": [
                    "Fijo directo",
                    "Variable directo",
                    "Fijo indirecto",
                    "Variable indirecto",
                    "Tangible",
                    "Intangible",
                ],
                "Costo": [round(total / 6, 2)] * 6,
            }
        )

    cost_key = f"costos_{sel_tx}"
    if cost_key not in st.session_state:
        total_tx = tx_df[tx_df["Tratamiento"] == sel_tx]["Costo_total"].iloc[0]
        st.session_state[cost_key] = make_cost_df(total_tx)

    desglose_df = st.data_editor(
        st.session_state[cost_key],
        num_rows="dynamic",
        use_container_width=True,
        key=cost_key,
    )

    total_desg = desglose_df["Costo"].sum()
    if total_desg <= 0:
        st.warning("Introduce valores positivos de costo para generar el tornado.")
    else:
        base_eff = base["Efectividad"]
        eff_tx = tx_df[tx_df["Tratamiento"] == sel_tx]["Efectividad"].iloc[0]
        sens = []
        for _, comp in desglose_df.iterrows():
            for pct in [-0.2, 0.2]:
                nuevo_costo = total_desg + comp["Costo"] * pct
                d_c = nuevo_costo - base["Costo_total"]
                d_e = eff_tx - base_eff
                icer_mod = d_c / d_e if d_e != 0 else np.nan
                sens.append({
                    "Variable": f"{comp['Componente']} {'-20%' if pct<0 else '+20%'}",
                    "ICER": icer_mod,
                })

        torn_df = pd.DataFrame(sens)
        torn_df["abs"] = torn_df["ICER"].abs()
        torn_df = torn_df.sort_values("abs", ascending=True)

        fig3, ax3 = plt.subplots()
        ax3.barh(torn_df["Variable"], torn_df["ICER"], edgecolor="black")
        ax3.set_xlabel("ICER modificado (US$/ΔEfect)")
        st.pyplot(fig3)

# ──────────────────────────────────────────────────────────
# 9) CBA – Costo‑beneficio
# ──────────────────────────────────────────────────────────
else:
    st.header("Costo‑Beneficio (CBA)")
    col1, col2 = st.columns(2)
    with col1:
        costo_prog = st.number_input("Costo de la intervención (US$)", value=100000.0, step=1000.0)
    with col2:
        beneficio_prog = st.number_input("Beneficio monetario (US$)", value=150000.0, step=1000.0)

    beneficio_neto = beneficio_prog - costo_prog
    ratio = beneficio_prog / costo_prog if costo_prog else np.nan

    st.success(f"**Beneficio monetario neto (BMN):** US$ {beneficio_neto:,.0f}")
    st.info(f"**Ratio beneficio/costo:** {ratio:,.2f}")

# ---------------------------------------------------------
# Pie de página común
# ---------------------------------------------------------

st.caption("Aplicación educativa – versiones simplificadas de cada método. Consulte las guías NICE, Drummond 2005 y Gray & Briggs 2019 para implementaciones completas.")
