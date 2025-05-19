import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Análisis de Costo‑Efectividad – App Streamlit
# Autor: Jarvis (ChatGPT)
# ---------------------------------------------------------
# Entradas: costos, efectividades, desviaciones estándar y umbral.
# Salidas: tabla determinística, plano de costo‑efectividad,
#          PSA (plano) y diagrama de tornado.
# ---------------------------------------------------------

st.set_page_config(page_title="CEA App", layout="wide")

st.title("Análisis de Costo‑Efectividad (ACE)")

st.sidebar.header("Datos de entrada")

# --- tabla editable ----------------------------------------------------------
default_df = pd.DataFrame(
    {
        "Tratamiento": ["Comparador", "Intervención"],
        "Costo": [1000.0, 1500.0],
        "Efectividad": [0.70, 0.85],
        "SD_Costo": [100.0, 120.0],
        "SD_Efect": [0.07, 0.08],
    }
)

st.sidebar.write(
    "**Editar tabla:** la primera fila será el comparador base. ")

# `st.data_editor` es la versión estable de la antigua experimental_data_editor
entrada_df = st.sidebar.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
    key="tabla_datos",
)

# --- umbral y PSA -----------------------------------------------------------
umbral = st.sidebar.number_input(
    "Umbral (λ) – disposición a pagar por ΔEfectividad",
    value=5000.0,
    step=100.0,
)

n_iter = st.sidebar.slider(
    "Iteraciones PSA (Monte Carlo)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
)

# ---------------------------------------------------------------------------
# 1) Cálculo determinístico
# ---------------------------------------------------------------------------
base = entrada_df.iloc[0]
resultados = []
for idx, row in entrada_df.iterrows():
    if idx == 0:
        continue  # comparador
    d_cost = row["Costo"] - base["Costo"]
    d_eff = row["Efectividad"] - base["Efectividad"]
    icer = d_cost / d_eff if d_eff != 0 else np.nan
    resultados.append((row["Tratamiento"], d_cost, d_eff, icer))

res_df = pd.DataFrame(
    resultados, columns=["Tratamiento", "ΔCosto", "ΔEfectividad", "ICER"]
)

st.subheader("Resultados determinísticos")
st.dataframe(res_df, use_container_width=True)

# ---------------------------------------------------------------------------
# 2) Plano de Costo‑Efectividad determinístico
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots()
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.axvline(0, color="gray", linewidth=0.5)
ax1.set_xlabel("ΔEfectividad")
ax1.set_ylabel("ΔCosto (US$)")

# línea del umbral
x_vals = np.linspace(min(res_df["ΔEfectividad"].min(), -0.1),
                     max(res_df["ΔEfectividad"].max(), 0.1), 100)
ax1.plot(x_vals, umbral * x_vals, linestyle="--", label=f"Umbral = {umbral:.0f}")

for _, r in res_df.iterrows():
    ax1.scatter(r["ΔEfectividad"], r["ΔCosto"], label=r["Tratamiento"])
    ax1.annotate(r["Tratamiento"], (r["ΔEfectividad"], r["ΔCosto"]))

ax1.legend()

st.subheader("Plano de Costo‑Efectividad (determinístico)")
st.pyplot(fig1)

# ---------------------------------------------------------------------------
# 3) PSA (Monte Carlo)
# ---------------------------------------------------------------------------
psa_points = {}
for _, row in entrada_df.iterrows():
    cost_sim = np.random.normal(row["Costo"], row["SD_Costo"], size=n_iter)
    eff_sim = np.random.normal(row["Efectividad"], row["SD_Efect"], size=n_iter)
    psa_points[row["Tratamiento"]] = (cost_sim, eff_sim)

base_cost, base_eff = psa_points[base["Tratamiento"]]

fig2, ax2 = plt.subplots()
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.axvline(0, color="gray", linewidth=0.5)
ax2.set_xlabel("ΔEfectividad")
ax2.set_ylabel("ΔCosto (US$)")

# línea del umbral
x_vals2 = np.linspace(-0.3, 0.3, 100)
ax2.plot(x_vals2, umbral * x_vals2, linestyle="--", label=f"Umbral = {umbral:.0f}")

for name, (c_sim, e_sim) in psa_points.items():
    if name == base["Tratamiento"]:
        continue
    d_cost_sim = c_sim - base_cost
    d_eff_sim = e_sim - base_eff
    ax2.scatter(d_eff_sim, d_cost_sim, alpha=0.05, s=10, label=name)

ax2.legend()

st.subheader("Plano de Costo‑Efectividad (PSA)")
st.pyplot(fig2)

# ---------------------------------------------------------------------------
# 4) Tornado – análisis univariado ±20 %
# ---------------------------------------------------------------------------
selected_tx = st.selectbox(
    "Selecciona tratamiento para diagrama tornado", entrada_df["Tratamiento"].tolist()[1:]
)

row_tx = entrada_df[entrada_df["Tratamiento"] == selected_tx].iloc[0]

sens_rows = []
for var in ["Costo", "Efectividad"]:
    for change in [-0.2, 0.2]:
        mod_row = row_tx.copy()
        mod_row[var] = mod_row[var] * (1 + change)
        d_cost = mod_row["Costo"] - base["Costo"]
        d_eff = mod_row["Efectividad"] - base["Efectividad"]
        icer = d_cost / d_eff if d_eff != 0 else np.nan
        sens_rows.append({"Variable": f"{var} {'-20%' if change<0 else '+20%'}", "ICER": icer})

tornado_df = pd.DataFrame(sens_rows)

# ordenar por ICER absoluto
tornado_df["abs"] = tornado_df["ICER"].abs()

fig3, ax3 = plt.subplots()
ax3.barh(tornado_df["Variable"], tornado_df["ICER"], edgecolor="black")
ax3.set_xlabel("ICER modificado (US$/ΔEfectividad)")
ax3.set_title(f"Diagrama Tornado – {selected_tx}")

st.subheader("Análisis de Sensibilidad Univariado (Tornado)")
st.pyplot(fig3)

