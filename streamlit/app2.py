import streamlit as st
import numpy as np
import pickle

# Configure the page
st.set_page_config(page_title="Client Risky Prediction", layout="centered")

# Load the XGBoost model
#model = pickle.load(open("model.pkl", "rb"))

# Predefined mappings
predefined_mappings = {
    "usage": {
        "VP": 0, "u1": 1, "moto": 2, "taxi": 3, "U2": 4, "engin": 5, "autre": 6,
        "louage": 7, "transport_rural": 8, "taxi_collectif": 9
    },
    "activite": {
        "EDUCATION_FORMATION": 0, "PROFESSIONS_MEDICALES": 1, "EMPLOYE": 2, "RETRAITE": 3,
        "ACTIVITES_COMMERCIALES": 4, "AGRICULTURE": 5, "RESIDENT_A_L'ETRANGER": 6, "ARTISAN": 7,
        "CORPS_ACTIFS": 8, "INGENIEUR": 9, "CHAUFFEUR": 10, "PARAMEDICAL": 11, "OUVRIER": 12,
        "TAXI_LOUAGE_TRASPORT_RURAL": 13, "ARCHITECTURE_BTP_IMMOBILIER": 14, "TECHNICIEN": 15,
        "GERANT_DIRIGEANT": 16, "PROFESSIONNEL_CONSULTANT_EXPERT": 17, "METIERS_LEGAUX": 18,
        "INFORMATIQUE": 19, "DIRECTEUR": 20, "TOURISME": 21, "AUTO_ECOLE": 22,
        "ACTIVITES_SPORTIVES": 23, "ACTIVITES_ARTISTIQUES": 24, "TRANSPORT_AEREEN": 25, "ETAT": 26,
        "TRANSPORT": 27, "ACTIVITES_FINACIAIRES_ET_BANCAIRES": 28, "JOURNALISME": 29, "DIPLOMATIE": 30,
        "ASSOCIATIONS_ONG": 31, "SANS_PROFESSION": 32, "ACTIVITES_INDUSTRIELLES": 33
    },
    "classe": {
        "1.0": 0, "3.0": 1, "4.0": 2, "2.0": 3, "8.0": 4, "5.0": 5, "6.0": 6,
        "9.0": 7, "7.0": 8, "10.0": 9, "11.0": 10, "0.0": 11
    }
}

def get_input_level(value, ranges):
    """Find the level corresponding to the value from a range."""
    for range_obj in ranges:
        if range_obj["min"] <= value <= range_obj["max"]:
            return range_obj["level"]
    return None

def main():
    st.title('Client Risky Prediction')

    # Form for user input
    with st.form('prediction_form'):
        col1, col2 = st.columns(2)

        with col1:
            puissance = st.selectbox('Puissance', [3.0, 2.0, 4.0, 5.0, 6.0, 1.0, 7.0])
            age_objet_assuree = st.number_input('Âge de l\'objet assuré (années)', min_value=0, max_value=50, step=1)
            valeur_venale = st.number_input('Valeur vénale (€)', min_value=0, step=1000)
            valeur_neuve = st.number_input('Valeur neuve (€)', min_value=0, step=1000)
            charge_utile = st.number_input('Charge utile (tonnes)', min_value=0.0, step=0.1)
            usage = st.selectbox('Usage', options=list(predefined_mappings["usage"].keys()))

        with col2:
            anciennete = st.number_input('Ancienneté (années)', min_value=0, max_value=50, step=1)
            activite = st.selectbox('Activité', options=list(predefined_mappings["activite"].keys()))
            classe = st.selectbox('Classe', options=list(predefined_mappings["classe"].keys()))
            age_client = st.number_input('Âge du client (années)', min_value=18, max_value=100, step=1)
            delegation = st.number_input('Délégation', min_value=0, step=1)
            civilite = st.selectbox('Civilité', ['Monsieur', 'Madame', 'Mademoiselle'])

        predict_button = st.form_submit_button('Predict')

    # Prediction process
    if predict_button:
        try:
            features = [
                puissance,
                age_objet_assuree,
                valeur_venale,
                valeur_neuve,
                charge_utile,
                predefined_mappings["usage"][usage],
                anciennete,
                predefined_mappings["activite"][activite],
                predefined_mappings["classe"][classe],
                age_client,
                delegation,
                len(civilite)  # Civilité length as a feature
            ]

            features_array = np.array([features])

            # Make prediction
            prediction = model.predict(features_array)

            # Display prediction
            risk_label = "Risky" if prediction[0] == 1 else "Not Risky"
            st.markdown(f'<div style="text-align: center; font-size: 24px; padding: 15px; background-color: #f0f0f0; border-radius: 5px;">Predicted Risk: <strong>{risk_label}</strong></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f'Error during prediction: {str(e)}')

if __name__ == '__main__':
    main()
