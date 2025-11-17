import streamlit as st

st.set_page_config(page_title="N. incerta demo", page_icon="🧪")

st.title("My first Streamlit app 🎈")

st.write("Hello, I'm Estefanía and this is my first app deployed with Streamlit and GitHub.")

value = st.slider("Select a number", 0, 100, 50)
st.write("The selected value is:", value)

name = st.text_input("Write your name:")
if name:
    st.write(f"Hello, {name}! 👋")
