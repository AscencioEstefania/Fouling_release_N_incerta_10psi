# -*- coding: utf-8 -*-
"""
Full Streamlit application:
- Step 1: SBMA–PDMS input form
- Step 2: Ensemble prediction (N. incerta at 10 psi)
- Step 3: Global SHAP importance (Ensemble)
"""

from pathlib import Path
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ============================================================
#                GLOBAL CONFIG / CONSTANTS
# ============================================================

# Paths (relative to repo root)
MODELS_DIR = "Models"                     # Folder in GitHub with .pkl files
MODEL_FILE = "Ensemble_GBR_Model.pkl"    # Ensemble artifact
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILE)

DATA_PATH = "ml_FR_Predictor.csv"        # Dataset used to drive predictions

# Default feature list (used as fallback if not present in the pickle)
FEATURE_COLS_DEFAULT = [
    "HATS5p",
    "ESpm01d",
    "RDF130p",
    "RDF015e",
    "BELe7",
    "ALOGP2",
    "RDF035p",
]

LABEL_CANDIDATES = ["NAME", "Name", "Nombre", "Coating", "ID"]


# ------------------------------------------------------------------
# Helper EnsembleModel class so that pickle can load the saved object
# (in case the .pkl was created as an instance of this class)
# ------------------------------------------------------------------
class EnsembleModel:
    """
    Simple wrapper for an ensemble of two scikit-learn models.
    It is here mainly so that pickle can find the same class
    that was used when the model was saved.
    """

    def __init__(self, model_A=None, model_B=None,
                 scaler=None, feature_cols=None,
                 wA=0.73, wB=0.27):
        self.model_A = model_A
        self.model_B = model_B
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.wA = wA
        self.wB = wB

    def predict(self, X):
        """
        Ensemble prediction:
        y = wA * model_A(X) + wB * model_B(X)
        """
        pred_A = self.model_A.predict(X)
        pred_B = self.model_B.predict(X)
        return self.wA * pred_A + self.wB * pred_B


# ============================================================
#                ENSEMBLE LOADING FUNCTION
# ============================================================
@st.cache_resource
def load_ensemble(path: str):
    """
    Load the trained Ensemble model / artifact.

    The .pkl can be:
    - A dict with keys: model_A, model_B, scaler, feature_cols, weights
      (or older names: mA, mB, feature_names, wA, wB)
    - An EnsembleModel instance saved in the past.
    This function normalizes everything to a dict with keys:
    {model_A, model_B, scaler, feature_cols, weights}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    # ---------- Case 1: object is an EnsembleModel ----------
    if isinstance(obj, EnsembleModel):
        attrs = obj.__dict__
        st.info(
            "DEBUG: loaded an EnsembleModel instance from pickle. "
            f"Attributes found: {list(attrs.keys())}"
        )

        # Old naming (mA/mB/feature_names) or new naming (model_A/model_B/feature_cols)
        model_A = attrs.get("model_A") or attrs.get("mA")
        model_B = attrs.get("model_B") or attrs.get("mB")
        scaler = attrs.get("scaler")
        feature_cols = (
            attrs.get("feature_cols")
            or attrs.get("feature_names")
            or FEATURE_COLS_DEFAULT
        )
        wA = attrs.get("wA")
        wB = attrs.get("wB")

        if any(x is None for x in [model_A, model_B, scaler, feature_cols, wA, wB]):
            raise ValueError(
                "The loaded EnsembleModel is missing internal models, "
                "scaler, feature names or weights."
            )

        return {
            "model_A": model_A,
            "model_B": model_B,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "weights": (wA, wB),
        }

    # ---------- Case 2: object is a dict ----------
    if isinstance(obj, dict):
        model_A = obj.get("model_A") or obj.get("mA")
        model_B = obj.get("model_B") or obj.get("mB")
        scaler = obj.get("scaler")
        feature_cols = (
            obj.get("feature_cols")
            or obj.get("feature_names")
            or FEATURE_COLS_DEFAULT
        )
        weights = obj.get("weights") or (obj.get("wA"), obj.get("wB"))

        if any(x is None for x in [model_A, model_B, scaler, feature_cols]) or weights is None:
            raise ValueError(
                "Dictionary ensemble pickle is missing required pieces "
                "(models, scaler, feature_cols or weights)."
            )

        return {
            "model_A": model_A,
            "model_B": model_B,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "weights": weights,
        }

    # ---------- Anything else ----------
    raise TypeError(f"Unexpected object type in ensemble pickle: {type(obj)}")


# Try importing SHAP + matplotlib
try:
    import shap
    import matplotlib.pyplot as plt
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ============================================================
#                  BASIC STREAMLIT CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fouling-release predictor",
    page_icon="🧪",
    layout="wide",
)

st.title("Fouling-Release Predictor")
st.write(
    "Minimal demonstration of the SBMA–PDMS additive module "
    "+ Ensemble model prediction for *Navicula incerta* at 10 psi."
)


# ============================================================
#            PART 1 — SBMA–PDMS USER INPUT SECTION
# ============================================================
st.markdown("---")
st.header("1. SBMA–PDMS additive definition")

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
    "Additive percent added to the coating",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

# Validation messages
if sbma_mw < 280.41 or sbma_mw > 5000.0:
    st.error("SBMA molecular weight must be between 280.41 and 5000.0.")

if pdms_mw < 92.12 or pdms_mw > 10000.0:
    st.error("PDMS molecular weight must be between 92.12 and 10000.0.")

if percentage < 0.2 or percentage > 5.0:
    st.error("Additive percentage must be between 0.2 and 5.0.")

# Summary of inputs
st.markdown(
    f"""
### Input summary
- **SBMA molecular weight:** {sbma_mw}  
- **PDMS molecular weight:** {pdms_mw}  
- **Additive percentage:** {percentage}%
"""
)

# Derived quantities
st.markdown("---")
st.subheader("Derived quantities")

SBMA_UNIT_MW = 280.41
PDMS_UNIT_MW = 92.12

fraction_sbma = sbma_mw / SBMA_UNIT_MW if SBMA_UNIT_MW > 0 else 0.0
fraction_pdms = pdms_mw / PDMS_UNIT_MW if PDMS_UNIT_MW > 0 else 0.0

mixture_id = f"SBMA{sbma_mw:.2f}_PDMS{pdms_mw:.2f}_at_{percentage:.2f}"

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("SBMA base unit MW", f"{SBMA_UNIT_MW}")
    st.metric("SBMA fraction (n units)", f"{fraction_sbma:.2f}")

with col_b:
    st.metric("PDMS base unit MW", f"{PDMS_UNIT_MW}")
    st.metric("PDMS fraction (n units)", f"{fraction_pdms:.2f}")

with col_c:
    st.write("Mixture ID")
    st.code(mixture_id, language="text")


# ============================================================
#        PART 2 — ENSEMBLE MODEL (N. INCERTA 10 PSI)
# ============================================================
st.markdown("---")
st.header("2. Ensemble model — *N. incerta* (10 psi)")

# ---- Load ensemble artifact ----
try:
    artifact = load_ensemble(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading Ensemble model:\n\n{e}")
    st.stop()

model_A = artifact["model_A"]
model_B = artifact["model_B"]
scaler = artifact["scaler"]
wA, wB = artifact["weights"]
feature_cols = artifact["feature_cols"]

# Safety check
if (model_A is None) or (model_B is None) or (scaler is None):
    st.error(
        "The loaded Ensemble model is missing internal models or scaler.\n\n"
        "Please re-save `Ensemble_GBR_Model.pkl` as a dictionary with keys:\n"
        "`model_A`, `model_B`, `scaler`, `feature_cols`, `weights`."
    )
    st.stop()


@st.cache_data
def load_predictor_csv(path: str):
    """
    Load ml_FR_Predictor.csv and detect a label column.
    - Try default delimiter ','.
    - If it looks like a single-column file, retry with ';'.
    - Strip spaces from column names.
    """
    # 1) First attempt: default comma delimiter
    df = pd.read_csv(path, encoding="UTF-8")

    # 2) If the file appears to have only one column, retry with ';'
    if df.shape[1] == 1:
        df = pd.read_csv(path, encoding="UTF-8", delimiter=";")

    # 3) Clean column names: remove all surrounding spaces
    df.columns = df.columns.astype(str).str.strip()

    # 4) Detect a label/name column
    label_col = None
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            label_col = c
            break

    # If nothing is found, create a synthetic coating ID
    if label_col is None:
        df["Coating_ID"] = np.arange(len(df))
        label_col = "Coating_ID"

    return df, label_col



# ---- Load dataset for coating selection ----
try:
    df_all, label_col = load_predictor_csv(DATA_PATH)
except Exception as e:
    st.error(f"Error loading predictor CSV:\n\n{e}")
    st.stop()

# Check required columns
missing = [c for c in feature_cols if c not in df_all.columns]
if missing:
    st.error("Missing required feature columns: " + ", ".join(missing))
    st.stop()

df_valid = df_all.dropna(subset=feature_cols).copy()

# ---- Layout: coating selection + prediction ----
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Select an existing coating")

    if df_valid.empty:
        st.error("No valid rows in the dataset. Please check ml_FR_Predictor.csv.")
        row = None
    else:
        options = df_valid[label_col].astype(str).tolist()
        selected = st.selectbox("Coating:", options)

        row = df_valid[df_valid[label_col].astype(str) == selected].iloc[0]
        st.markdown("**Descriptor values used by the model:**")
        st.dataframe(row[feature_cols].to_frame(name="value"))

with col_right:
    st.subheader("Ensemble prediction")

    if df_valid.empty or row is None:
        st.info("No prediction available (dataset is empty).")
    else:
        # Prepare input for the model
        X = row[feature_cols].to_numpy(float).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Individual models
        pred_A = model_A.predict(X_scaled)[0]
        pred_B = model_B.predict(X_scaled)[0]

        # Ensemble
        pred_ens = wA * pred_A + wB * pred_B

        st.metric("Predicted % removal (Ensemble)", f"{pred_ens:.2f} %")

        with st.expander("Model A and B predictions (details)"):
            st.write(f"Model A: **{pred_A:.2f} %**")
            st.write(f"Model B: **{pred_B:.2f} %**")
            st.write(f"Weights → wA = {wA:.2f}, wB = {wB:.2f}")


# ============================================================
#               PART 3 — GLOBAL SHAP IMPORTANCE
# ============================================================
st.markdown("---")
st.header("3. Global feature importance (SHAP – Ensemble)")

if not HAS_SHAP:
    st.info("SHAP is not installed. Install it with: `pip install shap`")
else:

    @st.cache_resource
    def compute_shap_global(artifact, df):
        """Compute SHAP values for the Ensemble (A+B)."""
        model_A = artifact["model_A"]
        model_B = artifact["model_B"]
        scaler = artifact["scaler"]
        wA, wB = artifact["weights"]
        feature_cols = artifact["feature_cols"]

        X = df[feature_cols].to_numpy(float)
        X_scaled = scaler.transform(X)

        explA = shap.TreeExplainer(model_A)
        explB = shap.TreeExplainer(model_B)

        shap_A = explA.shap_values(X_scaled)
        shap_B = explB.shap_values(X_scaled)

        shap_ensemble = wA * shap_A + wB * shap_B

        return shap_ensemble, X_scaled, feature_cols

    if df_valid.empty:
        st.info("No data available to compute SHAP values.")
    else:
        try:
            shap_values, X_scaled_all, feat_names = compute_shap_global(
                artifact, df_valid
            )

            shap.summary_plot(
                shap_values,
                X_scaled_all,
                feature_names=feat_names,
                plot_type="bar",
                show=False,
            )
            fig = plt.gcf()
            st.pyplot(fig, clear_figure=True)

        except Exception as e:
            st.error(f"Error computing SHAP values:\n\n{e}")
