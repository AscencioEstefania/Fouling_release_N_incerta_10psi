import streamlit as st

# Main title
st.title("My first Streamlit app 🎈")

# Text
st.write("Hello, I'm Estefanía and this is my first app deployed with Streamlit and GitHub.")

# Example slider
value = st.slider("Select a number", 0, 100, 50)
st.write("The selected value is:", value)

# Text input
name = st.text_input("Write your name:")
if name:
    st.write(f"Hello, {name}! 👋")
