# -*- coding: utf-8 -*-
"""
Minimal Streamlit app – Step 1: just to check everything runs.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


# Basic page configuration
st.set_page_config(
    page_title="Fouling-release demo",
    page_icon="🧪",
    layout="wide",
)

st.title("Demo Streamlit – Step 1")
st.write("If you can see this text, Streamlit is working correctly.")

# --- User inputs for SBMA, PDMS and percentage ---
st.markdown("---")
st.subheader("SBMA–PDMS additive definition")

col1, col2, col3 = st.columns(3)

sbma_mw = col1.number_input(
    "SBMA molecular weight",
    min_value=0.0,
    max_value=5000.0,
    value=280.41,
    step=1.0,
)

pdms_mw = col2.number_input(
    "PDMS molecular weight",
    min_value=0.0,
    max_value=10000.0,
    value=92.12,
    step=1.0,
)

percentage = col3.number_input(
    "Additive percent to add to the coating",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

# Simple validation messages
if sbma_mw < 280.41 or sbma_mw > 5000.0:
    st.error("SBMA molecular weight must be between 280.41 and 5000.0.")

if pdms_mw < 92.12 or pdms_mw > 10000.0:
    st.error("PDMS molecular weight must be between 92.12 and 10000.0.")

if percentage < 0.2 or percentage > 5.0:
    st.error("Additive percentage must be between 0.2 and 5.0.")

st.markdown(
    f"""
**Summary of inputs**  
- SBMA molecular weight: **{sbma_mw}**  
- PDMS molecular weight: **{pdms_mw}**  
- Additive percentage: **{percentage}%**
"""
)

st.markdown(
    f"""
**Summary of inputs**  
- SBMA molecular weight: **{sbma_mw}**  
- PDMS molecular weight: **{pdms_mw}**  
- Additive percentage: **{percentage}%**
"""
)

# --- Step 3: basic derived quantities (fractions and ID) ---
st.markdown("---")
st.subheader("Derived quantities")

SBMA_UNIT_MW = 280.41
PDMS_UNIT_MW = 92.12

fraction_sbma = sbma_mw / SBMA_UNIT_MW if SBMA_UNIT_MW > 0 else 0.0
fraction_pdms = pdms_mw / PDMS_UNIT_MW if PDMS_UNIT_MW > 0 else 0.0

# Build a simple mixture ID (we can refine the formatting later)
mixture_id = f"SBMA{sbma_mw:.2f}_PDMS{pdms_mw:.2f}_at_{percentage:.2f}"

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("SBMA unit MW", f"{SBMA_UNIT_MW}")
    st.metric("SBMA fraction (n units)", f"{fraction_sbma:.2f}")

with col_b:
    st.metric("PDMS unit MW", f"{PDMS_UNIT_MW}")
    st.metric("PDMS fraction (n units)", f"{fraction_pdms:.2f}")

with col_c:
    st.write("Mixture ID")
    st.code(mixture_id, language="text")
