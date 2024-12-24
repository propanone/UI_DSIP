import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="ML API - Client Risk Prediction")

st.markdown("""
<style>
    .stApp {
        font-family: 'Arial', sans-serif;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    h1 {
        color: #2e6c80;
        text-align: center;
        padding: 20px;
    }
    .stButton > button {
        width: 100%;
        background-color: #1abc9c;
        padding: 10px;
        font-size: 18px;
        margin-top: 20px;
    }
    .prediction-result {
        text-align: center;
        margin-top: 20px;
        padding: 15px;
        background-color: #f0f0f0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

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

def get_level(value, ranges):
    for range_info in ranges:
        if range_info["min"] <= value <= range_info["max"]:
            return range_info["level"]
    return 1  # Default level if no range matches

def main():
    st.title('Client Risk Prediction')
    
    with st.form('prediction_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            puissance = st.number_input('Puissance', min_value=0.0, max_value=999.0)
            age_objet = st.number_input("Age de l'objet assuré", min_value=0)
            valeur_venale = st.number_input('Valeur vénale', min_value=0)
            valeur_neuve = st.number_input('Valeur neuve', min_value=0)
            charge_utile = st.number_input('Charge utile', min_value=0.0)
            energie = st.selectbox('Energie', ["ES", "DI"])
            usage = st.selectbox('Usage', options=list(predefined_mappings["usage"].keys()))
            place = st.number_input('Nombre de places', min_value=1)
            
        with col2:
            activite = st.selectbox('Activité', options=list(predefined_mappings["activite"].keys()))
            classe = st.selectbox('Classe', options=list(predefined_mappings["classe"].keys()))
            age_client = st.number_input('Âge du client', min_value=18, max_value=120)
            anciennete = st.number_input('Ancienneté', min_value=0)
            civilite = st.selectbox('Civilité', ['Mr', 'Mme', 'Entreprise', 'mult_CT', 'Org', 'Couple', 'Etablissement'])
            sexe = st.selectbox('Sexe', ['M', 'F'])
        
        predict_button = st.form_submit_button('Predict')
    
    if predict_button:
        try:
            # Convert inputs to feature vector
            features = {
                'puissance': get_level(puissance, predefined_mappings['puissance']),
                'age_objet': get_level(age_objet, predefined_mappings['age_objet_assuree']),
                'valeur_venale': get_level(valeur_venale, predefined_mappings['valeur_ranges']),
                'valeur_neuve': get_level(valeur_neuve, predefined_mappings['valeur_ranges']),
                'charge_utile': get_level(charge_utile, predefined_mappings['charge_utile']),
                'energie': 1 if energie == "ES" else 0,
                'usage': predefined_mappings['usage'][usage],
                'place': get_level(place, predefined_mappings['place']),
                'activite': predefined_mappings['activite'][activite],
                'classe': predefined_mappings['classe'][classe],
                'age_client': get_level(age_client, predefined_mappings['age_client']),
                'anciennete': get_level(anciennete, predefined_mappings['anciennete']),
                'civilite': 1 if civilite in ['Mr', 'Mme'] else 0,
                'sexe': 1 if sexe == 'M' else 0
            }
            
            # Convert features to array for prediction
            feature_array = np.array([[
                features['puissance'], features['age_objet'], features['valeur_venale'],
                features['valeur_neuve'], features['charge_utile'], features['energie'],
                features['usage'], features['place'], features['activite'],
                features['classe'], features['age_client'], features['anciennete'],
                features['civilite'], features['sexe']
            ]])
            
            # Make prediction
            prediction = model.predict(feature_array)
            probability = model.predict_proba(feature_array)[0]
            
            # Display results
            risk_status = "Risky" if prediction[0] == 1 else "Not Risky"
            risk_probability = probability[1] if prediction[0] == 1 else probability[0]
            
            st.markdown(f"""
            <div class="prediction-result">
                <h3>Prediction Results</h3>
                <p>Risk Status: {risk_status}</p>
                <p>Confidence: {risk_probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f'Error during prediction: {str(e)}')

if __name__ == '__main__':
    main()