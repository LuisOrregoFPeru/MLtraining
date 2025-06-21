# dist_app.py – Streamlit app to identify the best‑fitting distribution for pasted or uploaded data
# ---------------------------------------------------------------------------------------------
# Versión extendida: incluye distribución piramidal (triangular) y otras distribuciones comunes
# ---------------------------------------------------------------------------------------------
# Ejecución local
#   pip install streamlit pandas numpy scipy matplotlib
#   streamlit run dist_app.py
# ---------------------------------------------------------------------------------------------

from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Detector de distribuciones", layout="centered")

# ──────────────────────────────────────────────────────────────────────────────
# Diccionarios auxiliares
# ──────────────────────────────────────────────────────────────────────────────

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
    "triang": "Ajuste triangular (mín–moda–máx)",
    "uniform": "Modelos no paramétricos / rango",
    "nbinom": "Regresión Binomial Negativa (GLM link log)",
    "geom": "Regresión Geométrica (caso NB, r=1)",
    "pareto": "Modelos de colas Pareto / POT",
}

# ──────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────────────────────────────────────

def parse_text_input(text: str) -> np.ndarray:
    """Convierte texto con números separados por delimitadores en un array NumPy."""
    if not text:
        return np.array([])
    for sep in (",", "\n", "\t", ";"):
        text = text.replace(sep, " ")
    tokens = [t for t in text.strip().split(" ") if t]
    values = []
    for tk in tokens:
        try:
            values.append(float(tk))
        except ValueError:
            continue
    return np.array(values, dtype=float)


def get_candidate_distributions(data: np.ndarray) -> List[str]:
    """Genera lista de distribuciones candidatas basadas en propiedades básicas."""
    cands = ["norm"]  # universal

    if np.all(data >= 0):  # continuas positivas
        cands += [
            "expon",
            "gamma",
            "lognorm",
            "weibull_min",
            "triang",
            "uniform",
            "pareto",
        ]
    if np.all((0 <= data) & (data <= 1)):
        cands.append("beta")
    if np.all(np.mod(data, 1) == 0):  # enteros
        cands += ["poisson", "nbinom", "geom"]

    # eliminar duplicados preservando orden
    seen, ordered = set(), []
    for d in cands:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def fit_distribution(dist_name: str, data: np.ndarray) -> Dict[str, object]:
    """Ajusta distribución SciPy y devuelve log‑verosimilitud, AIC y BIC."""
    n = len(data)

    if dist_name == "poisson":
        lam = data.mean()
        loglik = np.sum(stats.poisson.logpmf(data, lam))
        params, k = (lam,), 1
    elif dist_name == "nbinom":
        mean, var = data.mean(), data.var()
        p_init = mean / var if var > mean else 0.5
        r_init = mean * p_init / (1 - p_init) if p_init < 1 else 1
        dist = stats.nbinom
        params = dist.fit(data, r_init, p_init)
        loglik = np.sum(dist.logpmf(data, *params))
        k = len(params)
    else:
        dist = getattr(stats, dist_name)
        params = dist.fit(data)
        try:
            loglik = np.sum(dist.logpdf(data, *params))
        except Exception:
            loglik = -np.inf
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
    return pd.DataFrame(results).sort_values("aic").reset_index(drop=True)


def show_aic_plot(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.bar(df["distribution"], df["aic"])
    ax.set_ylabel("AIC (menor es mejor)")
    ax.set_xlabel("Distribución")
    ax.set_title("Comparación de AIC entre distribuciones candidatas")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Interfaz Streamlit
# ──────────────────────────────────────────────────────────────────────────────

st.title("🔍 Detector automático de distribuciones – versión extendida")

st.markdown(
    """
Pegue sus datos en el cuadro o cargue un archivo **CSV/Excel**. El sistema ajustará una amplia gama de
**distribuciones candidatas** (incluye triangular, uniforme, NB, Pareto) y mostrará la mejor según **AIC/BIC**.
    """
)

# Sidebar
st.sidebar.header("⚙️ Opciones")
alpha = st.sidebar.slider("Nivel de significancia KS (opcional)", 0.01, 0.20, 0.05, 0.01)

# Entrada de datos
method = st.radio("Método de entrada", ["Pegar texto", "Subir archivo"])

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
                st.error("No se encontraron columnas numéricas en el archivo.")
                data = np.array([])
            else:
                sel = st.selectbox("Columna a analizar", num_cols)
                data = df_up[sel].dropna().to_numpy()
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
            data = np.array([])
    else:
        data = np.array([])

# Validaciones
if data.size == 0:
    st.info("Ingrese datos para continuar.")
    st.stop()
if data.size < 8:
    st.warning("Para mayor robustez se recomiendan al menos 8 observaciones.")

st.write(f"**Observaciones válidas:** {data.size}")

# Ajuste distribuciones
cands = get_candidate_distributions(data)
st.write("Distribuciones candidatas:", ", ".join(cands))

results = []
for d in cands:
    try:
        results.append(fit_distribution(d, data))
    except Exception as err:
        st.warning(f"{d}: error de ajuste → {err}")

if not results:
    st.error("No se pudo ajustar ninguna distribución.")
    st.stop()

summary = summarize_results(results)

# Mejora
st.subheader("🏆 Mejor distribución (AIC mínimo)")
best = summary.iloc[0]

st.markdown(
    f"**{DIST_FULL_NAMES.get(best['distribution'], best['distribution']).upper()}**  \
    Parámetros: `{np.round(best['params'], 4).tolist()}`  \
    AIC = {best['aic']:.2f} | BIC = {best['bic']:.2f}  \
    **Regresión sugerida:** {REG_RECOMMENDED.get(best['distribution'], 'No disponible')}"
)

# Tabla completa
st.subheader("Tabla completa de resultados")
summary_disp = summary.copy()
summary_disp["Distribución completa"] = summary_disp["distribution"].map(DIST_FULL_NAMES)
summary_disp["Regresión recomendada"] = summary_disp["distribution"].map(REG_RECOMMENDED)

st.dataframe(
    summary_disp[[
        "distribution",
        "Distribución completa",
        "aic",
        "bic",
        "Regresión recomendada",
        "params",
    ]]
)

# Gráfico AIC
st.subheader("Gráfico de comparación de AIC")
show_aic_plot(summary)

# KS para la mejor distribución
st.subheader("📊 Prueba de bondad de ajuste KS")
try:
    if best["distribution"] == "poisson":
        lam = best["params"][0]
        D, p = stats.kstest(data, "poisson", args=(lam,))
    else:
        D, p = stats.kstest(data, best["distribution"], args=best["params"])
    st.write(f"D = {D:.3f}, p = {p:.4f}")
    if p < alpha:
        st.warning("Se rechaza H0: la distribución podría no ajustar bien.")
    else:
        st.success("No se rechaza H0: ajuste compatible con los datos.")
except Exception as e:
    st.info(f"No se pudo ejecutar KS: {e}")

# Footer
st.markdown("---")
st.markdown("Aplicación creada por Orrego‑Ferreyros, versión extendida con piramidal y otras distribuciones.")

