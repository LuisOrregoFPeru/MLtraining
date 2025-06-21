# dist_app.py – Streamlit app to identify the best‑fitting distribution for pasted or uploaded data
# ---------------------------------------------------------------------------------------------
# Versión extendida: incluye distribución piramidal (triangular) y otras distribuciones comunes
# ---------------------------------------------------------------------------------------------
# Cómo ejecutar
#   pip install streamlit pandas numpy scipy matplotlib
#   streamlit run dist_app.py
# ---------------------------------------------------------------------------------------------

import io
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Detector de distribuciones", layout="centered")

# Diccionarios auxiliares ------------------------------------------------------
DIST_FULL_NAMES = {
    "norm": "Normal (Gaussiana)",
    "expon": "Exponencial",
    "gamma": "Gamma",
    "lognorm": "Log-normal",
    "weibull_min": "Weibull (mínimo)",
    "beta": "Beta",
    "poisson": "Poisson",
    "triang": "Triangular (Piramidal)",
    "uniform": "Uniforme",
    "nbinom": "Binomial negativa",
    "geom": "Geométrica",
    "pareto": "Pareto",
}

REG_RECOMMENDED = {
    "norm": "Regresión lineal (OLS)",
    "expon": "GLM Exponencial (link log)",
    "gamma": "GLM Gamma (link log)",
    "lognorm": "Regresión lineal sobre log(Y)",
    "weibull_min": "Regresión de supervivencia Weibull",
    "beta": "Regresión Beta (proporciones)",
    "poisson": "Regresión Poisson (GLM link log)",
    "triang": "Ajuste triangular (método mínimos cuadrados)",
    "uniform": "Modelos no paramétricos / rango",
    "nbinom": "Regresión Binomial Negativa (GLM link log)",
    "geom": "Regresión Geométrica (caso especial de NB)",
    "pareto": "Modelos de colas Pareto / POT",
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def parse_text_input(text: str) -> np.ndarray:
    """Parsea números separados por espacios, comas, punto y coma o saltos de línea."""
    if not text:
        return np.array([])
    for sep in [",", "\n", "\t", ";"]:
        text = text.replace(sep, " ")
    tokens = [t for t in text.strip().split(" ") if t]
    vals = []
    for tk in tokens:
        try:
            vals.append(float(tk))
        except ValueError:
            continue
    return np.array(vals, dtype=float)


def get_candidate_distributions(data: np.ndarray) -> List[str]:
    """Devuelve distribuciones candidatas según propiedades básicas."""
    cands = ["norm"]  # universal

    # Continuas no negativas (incluye piramidal/triangular)
    if np.all(data >= 0):
        cands += [
            "expon",
            "gamma",
            "lognorm",
            "weibull_min",
            "triang",
            "uniform",
            "pareto",
        ]
    # Proporciones 0‑1
    if np.all((0 <= data) & (data <= 1)):
        cands.append("beta")
    # Conteos
    if np.all(np.mod(data, 1) == 0):
        cands += ["poisson", "nbinom", "geom"]

    # Quitar duplicados conservando orden
    seen, ordered = set(), []
    for d in cands:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def fit_distribution(dist_name: str, data: np.ndarray) -> Dict[str, object]:
    """Ajusta una distribución de SciPy y devuelve parámetros + AIC/BIC."""
    n = len(data)

    # Casos especiales con fórmulas cerradas
    if dist_name == "poisson":
        lam = data.mean()
        loglik = np.sum(stats.poisson.logpmf(data, lam))
        params, k = (lam,), 1
    elif dist_name == "nbinom":
        # Ajuste rápido usando método de momentos (aprox) como inicial
        mean, var = data.mean(), data.var()
        if var > mean:
            p = mean / var
            r = mean * p / (1 - p)
            params0 = (r, p)
        else:
            params0 = (1, 0.5)
        dist = stats.nbinom
        params = dist.fit(data, *params0)
        loglik = np.sum(dist.logpmf(data, *params))
        k = len(params)
    else:
        dist = getattr(stats, dist_name)
        params = dist.fit(data)
        try:
            loglik = np.sum(dist.logpdf(data, *params))
        except Exception:
            loglik = -np.inf  # si falla la logpdf, para penalizar
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


def summarize_results(res: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(res)
    return df.sort_values("aic").reset_index(drop=True)


def show_aic_plot(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.bar(df["distribution"], df["aic"])
    ax.set_ylabel("AIC (menor es mejor)")
    ax.set_xlabel("Distribución")
    ax.set_title("Comparación de AIC entre distribuciones candidatas")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# Interfaz Streamlit
# -----------------------------------------------------------------------------

st.title("🔍 Detector automático de distribuciones – versión extendida")

st.markdown(
    """
Pegue sus datos en el cuadro de texto o cargue un archivo **CSV/Excel**. La aplicación probará una
variedad más amplia de **distribuciones candidatas** (incluye la piramidal/triangular, uniforme, binomial
negativa, etc.) y mostrará la mejor según **AIC/BIC** junto con la **regresión recomendada**.
    """
)

# Sidebar opciones
st.sidebar.header("⚙️ Opciones")
alpha = st.sidebar.slider("Nivel de significancia para KS (opcional)", 0.01, 0.20, 0.05, 0.01)

# Entrada de datos
method = st.radio("Método de entrada de datos", ["Pegar texto", "Subir archivo"])

if method == "Pegar texto":
    raw = st.text_area("Pegue los valores numéricos")
    data = parse_text_input(raw)
else:
    up = st.file_uploader("Archivo .csv o .xlsx", type=["csv", "xlsx"])
    if up is not None:
        try:
            df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            num_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.error("No hay columnas numéricas.")
                data = np.array([])
            else:
                sel = st.selectbox("Columna a analizar", num_cols)
                data = df_up[sel].dropna().to_numpy()
        except Exception as e:
            st.error(f"Error de lectura: {e}")
            data = np.array([])
    else:
        data = np.array([])

# Chequeos
if data.size == 0:
    st.info("Ingrese datos para continuar.")
    st.stop()
if data.size < 8:
    st.warning("Se recomiendan al menos 8 observaciones para una estimación estable.")

st.write(f"**Observaciones válidas:** {data.size}")

# Ajuste de distribuciones
cands = get_candidate_distributions(data)
st.write("Distribuciones candidatas detectadas:", ", ".join(cands))

results = []
for d in cands:
    try:
        results.append(fit_distribution(d, data))
    except Exception as e:
        st.warning(f"{d}: error de ajuste → {e}")

if not results:
    st.error("No se pudo ajustar ninguna distribución.")
    st.stop()

summary = summarize_results(results)

# Salida principal
st.subheader("🏆 Mejor distribución (AIC mínimo)")
best = summary.iloc[0]
st.markdown(
    f"**{DIST_FULL_NAMES.get(best['distribution'], best['distribution']).upper()}** "
    f"con parámetros `{np.round(best['params'], 4).tolist()}`  \
    AIC = {best['aic']:.2f}, BIC = {best['bic']:.2f}  \
    **Regresión sugerida:** {REG_RECOMMENDED.get(best['distribution'], '–')}"
)

# Tabla con todos los modelos
st.subheader("Tabla de resultados")
summary_disp = summary.copy()
summary_disp["Distribución completa"] = summary_disp["distribution"].map(DIST_FULL_NAMES)
summary_disp["Regresión recomendada"] = summary_disp["distribution"].map(REG_RECOMMENDED)
st.dataframe(
    summary_disp[["distribution", "Distribución completa", "


# --- Footer ---

st.markdown("---")
st.markdown("Aplicación creada por Orrego-Ferreyros, LA.")
