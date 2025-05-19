import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Análisis de Costo‑Efectividad – App Streamlit
# Autor: Jarvis (ChatGPT)
# ---------------------------------------------------------
# • Entradas: costos, efectividades, desviaciones estándar y umbral
# • Salidas: tabla determinística, plano de costo‑efectividad,
#   PSA (plano) y tornado diagram.
# ---------------------------------------------------------

st.set_page_config(page_title="Análisis de Costo‑Efectividad", layout="wide")

st.title("Aplicación de Análisis de Costo‑Efectividad")

st.sidebar.header("Datos de entrada")

# --- Datos iniciales ---
default_df = pd.DataFrame({
    'Tratamiento': ['A', 'B'],
    'Costo': [10000, 15000],
    'Efectividad': [1.5, 2.0],
    'Desv_Costo': [1000, 1500],
    'Desv_Efectividad': [0.1, 0.15]
})

# Editor interactivo
help_text = (
    "Ingrese los parámetros para cada alternativa. "
    "Puede añadir/eliminar filas y modificar los valores. "
    "El primer tratamiento se usará como comparador base."
)

df = st.sidebar.experimental_data_editor(
    default_df,
    num_rows="dynamic",
    key="tabla_datos",
    help=help_text
)

threshold = st.sidebar.number_input(
    "Umbral de disposición a pagar (moneda / unidad de efecto)",
    value=30000.0,
    step=1000.0,
    format="%.0f"
)

n_sim = st.sidebar.number_input(
    "Iteraciones PSA",
    min_value=100,
    max_value=20000,
    value=1000,
    step=100,
    format="%i"
)

if st.sidebar.button("Ejecutar análisis"):
    if len(df) < 2:
        st.error("Ingrese al menos dos tratamientos para comparar.")
        st.stop()

    base = df.iloc[0]

    # --- Cálculo determinístico ---
    incr_cost = df['Costo'] - base['Costo']
    incr_eff = df['Efectividad'] - base['Efectividad']
    icer = incr_cost / incr_eff

    results = df.copy()
    results['ΔCosto'] = incr_cost
    results['ΔEfectividad'] = incr_eff
    results['ICER'] = icer

    st.subheader("Resultados determinísticos")
    st.dataframe(
        results.style.format({
            'Costo': "{:.0f}",
            'Efectividad': "{:.2f}",
            'Desv_Costo': "{:.0f}",
            'Desv_Efectividad': "{:.2f}",
            'ΔCosto': "{:.0f}",
            'ΔEfectividad': "{:.2f}",
            'ICER': "{:.0f}"
        })
    )

    # --- Plano de costo‑efectividad ---
    fig_ce, ax_ce = plt.subplots()
    ax_ce.axhline(0, color='grey', linewidth=0.5)
    ax_ce.axvline(0, color='grey', linewidth=0.5)
    ax_ce.set_xlabel("Δ Efectividad")
    ax_ce.set_ylabel("Δ Costo")
    ax_ce.set_title("Plano de Costo‑Efectividad (Determinístico)")

    x_line = np.linspace(min(incr_eff.min(), -1), max(incr_eff.max(), 1), 100)
    ax_ce.plot(x_line, threshold * x_line, linestyle='--', label='Umbral')

    for idx, row in results.iterrows():
        if idx == 0:
            continue  # omit base
        ax_ce.scatter(row['ΔEfectividad'], row['ΔCosto'])
        ax_ce.text(row['ΔEfectividad'], row['ΔCosto'], row['Tratamiento'])

    ax_ce.legend()
    st.pyplot(fig_ce)

    # --- Análisis de sensibilidad probabilístico (PSA) ---
    st.subheader("Análisis de Sensibilidad Probabilístico (PSA)")

    sims = []
    for _, row in df.iterrows():
        cost_sim = np.random.normal(row['Costo'], row['Desv_Costo'], n_sim)
        eff_sim = np.random.normal(row['Efectividad'], row['Desv_Efectividad'], n_sim)
        sims.append(pd.DataFrame({
            'Tratamiento': row['Tratamiento'],
            'Costo': cost_sim,
            'Efectividad': eff_sim
        }))
    sim_df = pd.concat(sims, ignore_index=True)

    base_cost = sim_df[sim_df['Tratamiento'] == base['Tratamiento']]['Costo'].values
    base_eff = sim_df[sim_df['Tratamiento'] == base['Tratamiento']]['Efectividad'].values

    psa_records = []
    for treat in df['Tratamiento'][1:]:
        treat_cost = sim_df[sim_df['Tratamiento'] == treat]['Costo'].values
        treat_eff = sim_df[sim_df['Tratamiento'] == treat]['Efectividad'].values
        psa_records.append(pd.DataFrame({
            'Tratamiento': treat,
            'ΔCosto': treat_cost - base_cost,
            'ΔEfectividad': treat_eff - base_eff
        }))
    psa_df = pd.concat(psa_records, ignore_index=True)

    fig_psa, ax_psa = plt.subplots()
    ax_psa.axhline(0, color='grey', linewidth=0.5)
    ax_psa.axvline(0, color='grey', linewidth=0.5)
    ax_psa.set_xlabel("Δ Efectividad")
    ax_psa.set_ylabel("Δ Costo")
    ax_psa.set_title(f"PSA – {n_sim} iteraciones")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, treat in enumerate(df['Tratamiento'][1:]):
        data = psa_df[psa_df['Tratamiento'] == treat]
        ax_psa.scatter(data['ΔEfectividad'], data['ΔCosto'], alpha=0.25, s=10, label=treat, color=colors[i % len(colors)])

    ax_psa.plot(x_line, threshold * x_line, linestyle='--', color='black', label='Umbral')
    ax_psa.legend()
    st.pyplot(fig_psa)

    # --- Análisis Tornado ---
    st.subheader("Análisis Tornado")

    tornado_tr = st.selectbox(
        "Seleccione el tratamiento (distinto al base) para el tornado:",
        options=list(df['Tratamiento'][1:])
    )

    if tornado_tr:
        sel_row = df[df['Tratamiento'] == tornado_tr].iloc[0]
        params = {
            'Costo': (sel_row['Costo'] * 0.8, sel_row['Costo'] * 1.2),
            'Efectividad': (sel_row['Efectividad'] * 0.8, sel_row['Efectividad'] * 1.2)
        }

        base_row = base
        base_icer = (sel_row['Costo'] - base_row['Costo']) / (sel_row['Efectividad'] - base_row['Efectividad'])

        names, lows, highs = [], [], []
        for p, (low_v, high_v) in params.items():
            low_icer = ((low_v if p == 'Costo' else sel_row['Costo']) - base_row['Costo']) / ((low_v if p == 'Efectividad' else sel_row['Efectividad']) - base_row['Efectividad'])
            high_icer = ((high_v if p == 'Costo' else sel_row['Costo']) - base_row['Costo']) / ((high_v if p == 'Efectividad' else sel_row['Efectividad']) - base_row['Efectividad'])
            names.append(p)
            lows.append(low_icer)
            highs.append(high_icer)

        fig_t, ax_t = plt.subplots()
        y_pos = np.arange(len(names))
        ax_t.barh(y_pos, np.array(highs) - base_icer, left=base_icer, height=0.4, label='High')
        ax_t.barh(y_pos, np.array(lows) - base_icer, left=base_icer, height=0.4, label='Low')
        ax_t.set_yticks(y_pos)
        ax_t.set_yticklabels(names)
        ax_t.set_xlabel("ICER")
        ax_t.set_title("Diagrama de Tornado (±20%)")
        ax_t.legend()
        st.pyplot(fig_t)

    st.caption("\nNota: se asumen distribuciones normales para costos y efectividades en PSA. "
               "Puede editar desviaciones estándar para reflejar incertidumbre.")
else:
    st.info("⇦ Complete los datos en la barra lateral y haga clic en **Ejecutar análisis** para comenzar.")

# --- Requisitos ---
# requirements.txt recomendado:
# streamlit
# pandas
# numpy
# matplotlib
