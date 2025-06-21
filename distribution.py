# dist_app.py – Streamlit app to identify the best‑fitting distribution for pasted or uploaded data
# ---------------------------------------------------------------------------------------------
# How to run locally
# ------------------
# 1. Create a virtual environment (recommended)
#    python -m venv venv && source venv/bin/activate  # Linux/Mac
#    venv\Scripts\activate                            # Windows
# 2. Install dependencies
#    pip install streamlit pandas numpy scipy matplotlib
# 3. Launch the app
#    streamlit run dist_app.py
# ---------------------------------------------------------------------------------------------
# Author: ChatGPT (Jarvis)
# License: MIT

import io
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Detector de distribuciones", layout="centered")

# -----------------------
# Helper functions
# -----------------------

def parse_text_input(text: str) -> np.ndarray:
    """Parse a raw text input of numbers separated by spaces, commas, semicolons or newlines."""
    if not text:
        return np.array([])
    # Replace common delimiters by spaces and split
    for sep in [",", "\n", "\t", ";"]:
        text = text.replace(sep, " ")
    # Split and convert to float, ignoring non‑numeric tokens gracefully
    tokens = [t for t in text.strip().split(" ") if t]
    numbers = []
    for tk in tokens:
        try:
            numbers.append(float(tk))
        except ValueError:
            continue
    return np.array(numbers, dtype=float)


def get_candidate_distributions(data: np.ndarray) -> List[str]:
    """Return plausible distribution names given basic properties of the data."""
    candidates = []

    # Universal candidate
    candidates.append("norm")

    # All non‑negative continuous
    if np.all(data >= 0):
        candidates += ["expon", "gamma", "lognorm", "weibull_min"]

    # Data ranged 0–1 (could be proportions)
    if np.all((data >= 0) & (data <= 1)):
        candidates.append("beta")

    # Integer non‑negative counts with moderately small mean
    if np.all(np.equal(np.mod(data, 1), 0)) and np.max(data) < 1e6:
        candidates.append("poisson")

    # Remove duplicates while preserving order
    seen = set()
    ordered = []
    for d in candidates:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def fit_distribution(dist_name: str, data: np.ndarray) -> Dict[str, object]:
    """Fit a SciPy distribution and return parameters and information criteria."""
    n = len(data)

    if dist_name == "poisson":
        # Poisson MLE λ = mean
        lam = data.mean()
        loglik = np.sum(stats.poisson.logpmf(data, lam))
        k = 1
        params = (lam,)
    else:
        dist = getattr(stats, dist_name)
        params = dist.fit(data)
        loglik = np.sum(dist.logpdf(data, *params))
        k = len(params)

    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return {
        "distribution": dist_name,
        "params": params,
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
    }


def summarize_results(results: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df_sorted = df.sort_values("aic").reset_index(drop=True)
    return df_sorted


def show_aic_plot(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.bar(df["distribution"], df["aic"])
    ax.set_ylabel("AIC (menor es mejor)")
    ax.set_xlabel("Distribución")
    ax.set_title("Comparación de AIC entre distribuciones candidatas")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


# -----------------------
# Streamlit UI
# -----------------------

st.title("🔍 Detector automático de distribuciones")

st.markdown(
    """
Pegue sus datos en el cuadro de texto, o cargue un archivo **CSV/Excel**. La aplicación ajustará varias

distribuciones candidatas y mostrará cuál describe mejor a los datos, usando el **AIC** y el **BIC**.

**Nota**: si los datos son proporciones (0–1) o conteos, la app lo detectará automáticamente y probará
distribuciones apropiadas (beta, Poisson, etc.).
    """
)

# --- Sidebar for settings ---
st.sidebar.header("⚙️ Opciones")
default_alpha = st.sidebar.slider(
    "Nivel de significancia para pruebas KS (solo informativo)", 0.01, 0.20, 0.05, 0.01
)

# --- Data input ---

input_method = st.radio("Seleccione el método de entrada de datos:", ["Pegar texto", "Subir archivo"])

if input_method == "Pegar texto":
    raw_text = st.text_area("Pegue aquí los valores numéricos (separados por espacios, comas o saltos de línea)")
    data = parse_text_input(raw_text)
else:
    uploaded_file = st.file_uploader("Cargue un archivo .csv o .xlsx", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            numeric_cols = df_upload.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No se encontraron columnas numéricas en el archivo.")
                data = np.array([])
            else:
                col_selected = st.selectbox("Seleccione la columna numérica a analizar", numeric_cols)
                data = df_upload[col_selected].dropna().values
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            data = np.array([])
    else:
        data = np.array([])

# --- Analysis ---

if data.size == 0:
    st.info("Ingrese datos para comenzar el análisis.")
    st.stop()

if data.size < 8:
    st.warning("Para resultados fiables se recomienda al menos 8 observaciones.")

st.write(f"**Observaciones válidas:** {data.size}")

# Candidate distributions
cands = get_candidate_distributions(data)

st.write("Distribuciones candidatas detectadas:", ", ".join(cands))

results = []
for dist_name in cands:
    try:
        fit_res = fit_distribution(dist_name, data)
        results.append(fit_res)
    except Exception as ex:
        st.warning(f"{dist_name}: Error durante el ajuste → {ex}")

if not results:
    st.error("Ninguna distribución pudo ajustarse a los datos.")
    st.stop()

summary_df = summarize_results(results)

# --- Output ---

st.subheader("🏆 Distribución mejor ajustada (AIC mínimo)")

best_row = summary_df.iloc[0]

st.markdown(
    f"**{best_row['distribution'].upper()}** con parámetros: `{np.round(best_row['params'], 4).tolist()}`\n\n"
    f"AIC = {best_row['aic']:.2f},  BIC = {best_row['bic']:.2f}"
)

# Show table of all fits
st.subheader("Tabla completa de resultados")

st.dataframe(summary_df[["distribution", "aic", "bic", "params"]])

# Show AIC plot
st.subheader("Gráfico de comparación de AIC")
show_aic_plot(summary_df)

# Optionally perform KS goodness‑of‑fit for the best distribution
st.subheader("📊 Prueba de bondad de ajuste KS (solo referencial)")
dist_name = best_row["distribution"]
alpha = default_alpha

try:
    if dist_name == "poisson":
        lam = best_row["params"][0]
        d_stat, p_val = stats.kstest(data, "poisson", args=(lam,))
    else:
        dist = getattr(stats, dist_name)
        d_stat, p_val = stats.kstest(data, dist_name, args=best_row["params"])
    st.write(f"KS D = {d_stat:.3f}, p = {p_val:.4f}")
    if p_val < alpha:
        st.warning("La prueba rechaza la hipótesis nula: la distribución elegida puede no ajustarse bien al nivel de significancia seleccionado.")
    else:
        st.success("No se rechaza la hipótesis nula: la distribución elegida es compatible con los datos.")
except Exception as ex:
    st.info(f"No se pudo ejecutar KS para {dist_name}: {ex}")

# --- Footer ---

st.markdown("---")
st.markdown("Aplicación generada por **Jarvis** – Inserte esta carpeta en GitHub y despliegue en Streamlit Cloud para compartirla.")
