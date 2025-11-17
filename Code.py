import streamlit as st
# Título principal
st.title("Mi primera app con Streamlit 🎈")

# Texto
st.write("Hola, soy Estefanía y esta es mi primera app desplegada con Streamlit y GitHub.")

# Un slider de ejemplo
valor = st.slider("Selecciona un número", 0, 100, 50)
st.write("El valor seleccionado es:", valor)

# Entrada de texto
nombre = st.text_input("Escribe tu nombre:")
if nombre:
    st.write(f"¡Hola, {nombre}! 👋")
