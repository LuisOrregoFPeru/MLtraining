# dist_app.py – Streamlit app to identify the best‑fitting distribution for pasted or uploaded data
# ---------------------------------------------------------------------------------------------
# Versión extendida: incluye distribución piramidal (triangular) y otras distribuciones comunes
# + NUEVO: menú interactivo de **opciones GLM** para cada distribución ganadora
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
    "lognorm": "Log‑normal",
    "weibull_min": "Weibull (mínimo)",
    "beta": "Beta",
    "poisson": "Poisson",
    "triang": "Triangular (Piramidal)",
    "uniform": "Uniforme",
    "nbinom": "Binomial negativa",
    "geom": "Geométrica",
    "pareto": "Pareto",
}

# GLM sugerido por defecto (mostrar en tabla y encabezado)
REG_RECOMMENDED = {
    "norm": "GLM Gaussian (identidad)",
    "expon": "GLM Exponencial (log)",
    "gamma": "GLM Gamma (log)",
    "lognorm": "GLM Gaussian sobre log(Y)",
    "weibull_min": "Modelo Weibull AFT",
    "beta": "Regresión Beta (logit)",
    "poisson": "GLM Poisson (log)",
    "triang": "Mínimos cuadrados triangular",
    "uniform": "Modelo no paramétrico",
    "nbinom": "GLM Binomial Negativa (log)",
    "geom": "Regresión Geométrica",
    "pareto": "Modelo Pareto POT (log)",
}

# NUEVO: listado de opciones GLM detalladas por distribución
GLM_OPTIONS = {
    "norm": [
        "GLM Gaussian (identidad)",
        "GLM Gaussian (inverse) — varianza ∝ μ²",
        "GLM Gaussian (log)"
    ],
    "expon": [
        "GLM Exponencial (log)",
        "GLM Exponencial (identity)"
    ],
    "gamma": [
        "GLM Gamma (log)",
        "GLM Gamma (inverse)",
        "GLM Gamma (identity)"
    ],
    "beta": [
        "Regresión Beta (logit)",
        "Regresión Beta (probit)",
        "Regresión Beta (cloglog)"
    ],
    "poisson": [
        "GLM Poisson (log)",
        "GLM Poisson (sqrt)",
        "GLM Quasi‑Poisson (log) — sobredispersión"
    ],
    "nbinom": [
        "GLM NB (log)",
        "GLM NB (identity)"
    ],
    "weibull_min": [
        "Modelo AFT Weibull",
        "Modelo de riesgos proporcionales Weibull"
    ],
    "triang": ["No estándar — mínimos cuadrados"],
    "uniform": ["No estándar"],
    "geom": ["Regresión Geométrica (log)"]
}

# ──────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────────────────────────────────────

def parse_text_input(text: str) -> np.ndarray:
    if not text:
        return np.array([])
    for sep in (",", "\n", "\t", ";"):
        text = text.replace(sep, " ")
    tokens = [t for t in text.strip().split() if t]
    vals = []
    for tk in tokens:
        try:
            vals.append(float(tk))
        except ValueError:
            continue
    return np.array(vals, dtype=float)


def get_candidate_distributions(data: np.ndarray) -> List[str]:
    cand = ["norm"]
    if np.all(data >= 0):
        cand += ["expon", "gamma", "lognorm", "weibull_min", "triang", "uniform", "pareto"]
    if np.all((0 <= data) & (data <= 1)):
        cand.append("beta")
    if np.all(np.mod(data, 1) == 0):
        cand += ["poisson", "nbinom", "geom"]
    seen, ordered = set(), []
    for d in cand:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def fit_distribution(dist_name: str, data: np.ndarray) -> Dict[str, object]:
    n = len(data)
    if dist_name == "poisson":
        lam = data.mean()
        loglik = np.sum(stats.poisson.logpmf(data, lam))
        params, k = (lam,), 1
    elif dist_name == "nbinom":
        mean, var = data.mean(), data.var()
        p0 = mean / var if var > mean else 0.5
        r0 = mean * p0 / (1 - p0) if p0 < 1 else 1
        dist = stats.nbinom
        params = dist.fit(data, r0, p0)
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
    return {"distribution": dist_name, "params": params, "loglik": loglik, "aic": aic, "bic": bic}


def summarize_results(res):
    return pd.DataFrame(res).sort_values("aic").reset_index(drop=True)


def show_aic_plot(df):
    fig, ax = plt.subplots()
    ax.bar(df["distribution"], df["aic"])
    ax.set_ylabel("AIC (menor es mejor)")
    ax.set_xlabel("Distribución")
    ax.set_title("Comparación de AIC entre distribuciones candidatas")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Interfaz
# ──────────────────────────────────────────────────────────────────────────────

st.title("🔍 Detector automático de distribuciones + opciones GLM")

st.markdown("Pegue sus datos o cargue un **CSV/Excel** y obtenga la mejor distribución junto con un menú de **modelos GLM** disponibles.")

st.sidebar.header("⚙️ Opciones")
alpha = st.sidebar.slider("Nivel de significancia KS", 0.01, 0.20, 0.05, 0.01)

method = st.radio("Método de entrada", ["Pegar texto", "Subir archivo"])
if method == "Pegar texto":
    raw = st.text_area("Pegue los valores numéricos")
    data = parse_text_input(raw)
else:
    file = st.file_uploader("Archivo .csv o .xlsx", type=["csv", "xlsx"])
    if file is not None:
        try:
            df_up = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            num_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                col = st.selectbox("Columna a analizar", num_cols)
                data = df_up[col].dropna().to_numpy()
            else:
                st.error("No hay columnas numéricas.")
                data = np.array([])
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
            data = np.array([])
    else:
        data = np.array([])

if data.size == 0:
    st.info("Ingrese datos para continuar.")
    st.stop()
if data.size < 8:
    st.warning("Se recomienda n ≥ 8 para robustez.")

st.write(f"**Observaciones válidas:** {data.size}")

cands = get_candidate_distributions(data)
st.write("Distribuciones candidatas:", ", ".join(cands))

results = []
for d in cands:
    try:
        results.append(fit_distribution(d, data))
    except Exception as err:
        st.warning(f"{d}: error en ajuste → {err}")

if not results:
    st.error("No se pudo ajustar ninguna distribución.")
    st.stop()

summary = summarize_results(results)

best = summary.iloc[0]

st.subheader("🏆 Mejor distribución (AIC minimo)")
st.markdown(
    f"**{DIST_FULL_NAMES.get(best['distribution'], best['distribution']).upper()}**  \
    Parámetros: {np.round(best['params'], 4).tolist()}  \
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
st.markdown("Aplicación creada por Orrego‑Ferreyros, LA.")


**************

