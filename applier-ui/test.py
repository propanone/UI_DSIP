import os
import pandas as pd
import streamlit as st
import numpy as np
import pickle
from requests.auth import HTTPBasicAuth



# Path to models
MODELS_PATH = 'C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\applier-ui\\data\\models\\'

# Set page config
st.set_page_config(layout="wide", page_title="Risk Client Prediction")

# Load model
def load_model(model_path):
    return pickle.load(open(model_path, "rb"))

# Predefined mappings
MAPPINGS = {
    "usage": {
        "VP": 0, "u1": 1, "moto": 2, "taxi": 3, "U2": 4, "engin": 5, "autre": 6,
        "louage": 7, "transport_rural": 8, "taxi_collectif": 9
    },
    "civilite": {
        "Mr": 0, "Mme": 1, "Entreprise": 2, "mult_CT": 3, "Org": 4, "Couple": 5,
        "Etablissement": 6
    },
    "activite": {
        "EDUCATION_FORMATION": 0, "PROFESSIONS_MEDICALES": 1, "EMPLOYE": 2, "RETRAITE": 3,
        "ACTIVITES_COMMERCIALES": 4, "AGRICULTURE": 5, "RESIDENT_A_L'ETRANGER": 6
        # Add other activities as needed
    },
    "classe": {
        "1.0": 0, "3.0": 1, "4.0": 2, "2.0": 3, "8.0": 4, "5.0": 5, "6.0": 6,
        "9.0": 7, "7.0": 8, "10.0": 9, "11.0": 10, "0.0": 11
    }
}

# Range definitions
RANGES = {
    "puissance": [
        {"min": 0, "max": 3, "level": 1},
        {"min": 3, "max": 4, "level": 2},
        {"min": 4, "max": 6, "level": 3},
        {"min": 6, "max": 9, "level": 4},
        {"min": 9, "max": 14, "level": 5},
        {"min": 14, "max": 49, "level": 6},
        {"min": 49, "max": 999, "level": 7}
    ],
    "age_client": [
        {"min": 18, "max": 24, "level": 1},
        {"min": 24, "max": 29, "level": 2},
        {"min": 29, "max": 39, "level": 3},
        {"min": 39, "max": 49, "level": 4},
        {"min": 49, "max": 59, "level": 5},
        {"min": 59, "max": 120, "level": 6}
    ]
    # Add other ranges as needed
}

def get_level(value, ranges):
    for range_ in ranges:
        if range_["min"] <= value < range_["max"]:
            return range_["level"]
    return ranges[-1]["level"]  # Return highest level if above max

# Load available models
#model_file = st.selectbox(label="Model", options=models, format_func=get_model_name)

model = load_model(os.path.join(MODELS_PATH, "UC5-SKEYEPREDICT.json"))

# Create layout
col1, col2 = st.columns(2, gap='medium')

with col1:
    st.subheader("Object Information")
    puissance = st.number_input("Puissance", min_value=0, step=1)
    age_objet = st.number_input("Age de l'objet assuré", min_value=0, step=1)
    valeur_venale = st.number_input("Valeur vénale", min_value=0, step=1000)
    valeur_neuve = st.number_input("Valeur neuve", min_value=0, step=1000)
    charge_utile = st.number_input("Charge utile", min_value=0.0, step=0.1)
    usage = st.selectbox("Usage", options=list(MAPPINGS["usage"].keys()))

with col2:
    st.subheader("Client Information")
    age_client = st.number_input("Âge du client", min_value=18, step=1)
    civilite = st.selectbox("Civilite", options=list(MAPPINGS["civilite"].keys()))
    activite = st.selectbox("Activité", options=list(MAPPINGS["activite"].keys()))
    anciennete = st.number_input("Ancienneté", min_value=0, step=1)
    classe = st.selectbox("Classe", options=list(MAPPINGS["classe"].keys()))

predict = st.button(label="Predict Risk Level")

if predict:
    st.write("Predicting...")
    
    # Transform inputs
    features = np.array([
        get_level(puissance, RANGES["puissance"]),
        age_objet,
        valeur_venale,
        valeur_neuve,
        charge_utile,
        MAPPINGS["usage"][usage],
        anciennete,
        MAPPINGS["activite"][activite],
        MAPPINGS["classe"][classe],
        get_level(age_client, RANGES["age_client"]),
        MAPPINGS["civilite"][civilite]
    ]).reshape(1, -1)
    
    try:
        prediction = model["model"].predict(features)[0]
        confidence = model["model"].predict_proba(features)[0].max()
        
        st.subheader("Prediction Results:")
        st.write(f"Risk Level: {prediction}")
        st.info(f"Confidence Level: {confidence * 100:.2f}%")
        
        # Display feature importance if available
        if hasattr(model["model"], "feature_importances_"):
            st.warning("Feature Importance Analysis")
            importance_df = pd.DataFrame({
                'Feature': ['Puissance', 'Age Objet', 'Valeur Venale', 'Valeur Neuve', 
                           'Charge Utile', 'Usage', 'Anciennete', 'Activite', 'Classe',
                           'Age Client', 'Civilite'],
                'Importance': model["model"].feature_importances_
            })
            st.write(importance_df.sort_values('Importance', ascending=False))
            
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")