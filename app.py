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
