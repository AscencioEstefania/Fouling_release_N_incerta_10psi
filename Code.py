# -*- coding: utf-8 -*-
"""
Created on Fri March 14 17:41:37 2024

@author: Adaptado por Estefanía Ascencio
"""

#%% Importing libraries
from pathlib import Path
import pandas as pd
import pickle
from molvs import Standardizer
from rdkit import Chem
from openbabel import openbabel
from mordred import Calculator, descriptors
from multiprocessing import freeze_support
import numpy as np
from rdkit.Chem import AllChem
import plotly.graph_objects as go
import networkx as nx
import math
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# Packages for Streamlit
import streamlit as st
from PIL import Image
import io
import base64
import tqdm

#%% PAGE CONFIG

st.set_page_config(page_title='ML Fouling Release Property Predictor', page_icon=":ocean:", layout='wide')

# Function to put a picture as header
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

image = Image.open('cropped-header.png')
st.image(image)

st.write("[![Website](https://img.shields.io/badge/website-RasulevGroup-blue)](http://www.rasulev.org)")
st.subheader("About Us")
st.markdown("""
The group of Prof. Rasulev focuses on developing artificial intelligence (AI)-based predictive models to design novel polymeric and nanomaterials, predicting their properties such as toxicity, solubility, fouling release, elasticity, degradation rate, and biodegradation.  
We apply computational chemistry, machine learning, and cheminformatics methods for modeling, data analysis, and predictive structure-property relationship development to identify structural factors responsible for the activity of investigated materials.
""")

#%% INTRODUCTION
st.title(':ocean: ML Fouling Release Property Predictor')

st.write("""
This is a free web application for predicting the fouling release (FR) performance of polymeric coatings.

Marine biofouling is the unwanted adhesion of microorganisms, diatoms, and macroorganisms on submerged surfaces.
Fouling-release (FR) coatings do not kill organisms; instead, they minimize adhesion forces so that hydrodynamic flow can remove them easily.

This predictor uses machine learning models based on molecular descriptors and, when applicable, additive fractions in the formulation (e.g., PDMS, PEG, SBMA, PMHS).  
The goal is to support the design of cleaner, environmentally friendly coatings.
""")

#%% SECTIONS (TABS)
tab1, tab2, tab3, tab4 = st.tabs(["How to Use", "Required Data", "Metrics", "Best Practices"])

with tab1:
    st.markdown("""
1. Upload a CSV file with at least NAME, SMILES, and additive fraction columns (if applicable).  
2. The app standardizes SMILES, calculates descriptors (Mordred, RDKit), and constructs the feature vector.  
3. The model predicts FR performance (e.g., percentage of removal or adhesion strength).  
4. Correlation plots, applicability domain (Williams), and PCA chemical domain plots are generated.
""")

with tab2:
    st.markdown("""
**Minimum CSV requirements:**  
- NAME: sample or formulation identifier  
- SMILES: molecular representation (monomer or mixed)  
- w_PDMS, w_PEG, w_SBMA, w_PMHS: weight fractions if applicable  

**Notes:**  
- SMILES are standardized with MOLVS.  
- Protonation state corrections are applied at pH 7.4.  
- Descriptor mixing combines additive features proportionally to their fractions.
""")

with tab3:
    st.markdown("""
**Main performance metrics:**  
- R2 (train/test): training and external validation scores  
- LOO R2: Leave-One-Out validation for robustness with small datasets  
- KFold (5 and 10): cross-validation to assess variance stability  

**Applicability Domain (AD):**  
- Williams plot: standardized residuals vs leverage with threshold h*  
- Outliers (by residual or leverage) are reported with flags and tables.
""")

with tab4:
    st.markdown("""
**Model Capabilities:**  
- Captures trends across FR coatings and amphiphilic additives  
- Highlights out-of-domain formulations  

**Model Limitations:**  
- Not a replacement for experimental FR tests  
- Predictions can vary for unseen chemical families  

**Good Practices:**  
- Keep additive ratios within trained ranges  
- Check Applicability Domain and PCA plots before interpreting results
""")

#%% SIDEBAR: TARGET ORGANISM
st.sidebar.subheader("Biofouling Target")
organism = st.sidebar.selectbox(
    "Select the target organism:",
    ["Navicula incerta (diatom)", "Ulva linza (green spore)", "General mix"]
)

if organism.startswith("Navicula"):
    st.info("Predictions are interpreted based on Navicula incerta protocol metrics.")
elif organism.startswith("Ulva"):
    st.info("Predictions are interpreted based on Ulva linza adhesion/release data.")
else:
    st.info("General mode: interpret FR as a relative trend without specific bioassay calibration.")

#%% UPLOAD DATASET
st.sidebar.header('Upload your CSV file')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/gmaikelc/ML_water_perm_coef/main/example_file1.csv)
""")

uploaded_file_1 = st.sidebar.file_uploader("Upload a CSV file with SMILES and fractions", type=["csv"])

#%% STANDARDIZATION BY MOLVS
def standardizer(df, pos):
    s = Standardizer()
    molecules = df[pos].tolist()
    standardized_molecules = []
    i = 1
    t = st.empty()

    for molecule in molecules:
        try:
            smiles = molecule.strip()
            mol = Chem.MolFromSmiles(smiles)
            standarized_mol = s.super_parent(mol)
            standardized_smiles = Chem.MolToSmiles(standarized_mol)
            standardized_molecules.append(standardized_smiles)
            t.markdown("Processing monomers: " + str(i) + " / " + str(len(molecules)))
            i += 1
        except:
            standardized_molecules.append(molecule)
    df['Standardized_SMILES'] = standardized_molecules
    return df

#%% PROTONATION AT PH 7.4
def charges_ph(molecule, ph=7.4):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "smi")
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, molecule)
    mol.AddHydrogens()
    mol.CorrectForPH(ph)
    mol.AddHydrogens()
    optimized = obConversion.WriteString(mol)
    return optimized

def smile_obabel_corrector(smiles_ionized):
    mol1 = Chem.MolFromSmiles(smiles_ionized, sanitize=False)
    pattern1 = Chem.MolFromSmarts('[#6]-[#8-]-[#6]')
    if mol1.HasSubstructMatch(pattern1):
        at_matches = mol1.GetSubstructMatches(pattern1)
        at_matches_list = [y[1] for y in at_matches]
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()
    return Chem.MolToSmiles(mol1)

#%% RESULT CARD
def narrar_resultado(fr_pred, loo_r2=None, cv5=None, cv10=None, ad_flags=None):
    texto = []
    texto.append(f"Estimated FR performance: {fr_pred:.2f} (internal scale).")
    if loo_r2 is not None:
        texto.append(f"Model LOO R2: {loo_r2:.2f}")
    if cv5 is not None and cv10 is not None:
        texto.append(f"KFold5 R2: {cv5:.2f} | KFold10 R2: {cv10:.2f}")
    if ad_flags:
        texto.append("Warning: points detected outside the applicability domain.")
    else:
        texto.append("Formulation is within the model's applicability domain.")
    st.success(" ".join(texto))

#%% FOOTER
st.markdown("---")
st.caption("Note: This FR prediction demo is for research purposes only. Experimental validation is required for design decisions.")
st.caption("Base references: Mordred and RDKit descriptors; ML methods from scikit-learn; standardization via MOLVS; parsing via OpenBabel.")
