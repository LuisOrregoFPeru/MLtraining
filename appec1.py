import streamlit as st
import random

st.set_page_config(page_title="Ecuaciones de Primer Grado", page_icon="🧮")

st.title("🧮 Ecuaciones de Primer Grado")
st.markdown("Resuelve la ecuación de la forma `ax + b = c`")

# Inicializa el estado si es la primera vez
if 'a' not in st.session_state:
    st.session_state.a = random.randint(1, 10)
    st.session_state.b = random.randint(-10, 10)
    st.session_state.c = random.randint(-10, 10)

# Botón para nueva ecuación
if st.button("🔄 Generar nueva ecuación"):
    st.session_state.a = random.randint(1, 10)
    st.session_state.b = random.randint(-10, 10)
    st.session_state.c = random.randint(-10, 10)

# Mostrar ecuación
a = st.session_state.a
b = st.session_state.b
c = st.session_state.c

st.latex(f"{a}x + {b} = {c}")

# Entrada del usuario
respuesta_usuario = st.number_input("Introduce el valor de x", step=0.1, format="%.2f")

# Botón para verificar la respuesta
if st.button("✅ Verificar respuesta"):
    try:
        x_correcto = (c - b) / a
        if abs(respuesta_usuario - x_correcto) < 0.01:
            st.success(f"¡Correcto! 🎉 x = {x_correcto:.2f}")
        else:
            st.error(f"Incorrecto. 😞 La solución correcta es x = {x_correcto:.2f}")
    except ZeroDivisionError:
        st.error("Error: División por cero")

