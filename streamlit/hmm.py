# This is for UI improvements purposes only



import streamlit as st
import numpy as np
import pickle
from map import predefined_mappings

# Page Configuration
st.set_page_config(
    page_title='Risky Client Prediction',
    layout='wide',
    page_icon="risk.png",
    initial_sidebar_state='auto'
)

# Custom CSS for Styling
st.markdown("""
<style>
    .stApp {
        font-family: 'Arial', sans-serif;
        padding: 20px;
    }
    h1 {
        color: #2e6c80;
        text-align: center;
        font-family: 'Pacifico', cursive;
    }
    .stButton > button {
        width: 100%;
        background-color: #1abc9c;
        padding: 10px;
        font-size: 18px;
        border-radius: 5px;
    }
    .prediction-result {
        text-align: center;
        margin-top: 20px;
        padding: 15px;
        background-color: #f0f0f0;
        border-radius: 5px;
    }
    .error-message {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Paths to Models
path0 = "C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\models_out\\xgboost_model.pkl"

# Load the Model
with open(path0, 'rb') as file:
    loaded_artifacts = pickle.load(file)

# Extract Model Components
model = loaded_artifacts['model']
label_encoders = loaded_artifacts['label_encoders']
scaler = loaded_artifacts['scaler']

# Helper Function for Level Assignment
def get_level(value, ranges):
    for range_ in ranges:
        if range_["min"] <= value < range_["max"]:
            return range_["level"]
    raise ValueError(f"Value {value} is out of the defined range!")

# Main Application
def main():
    st.title('Client Risky Prediction')
    st.image("icon.png", width=200)

    # Input Form
    with st.form('prediction_form'):
        st.markdown("### Enter Client and Vehicle Details")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Client Information")
            sexe = st.selectbox('Gender', options=list(predefined_mappings["sexe"].keys()), help="Select the client's gender")
            age_client = st.number_input('Age', min_value=18, help="Enter the client's age")
            civilite = st.selectbox('Civil Status (Civilite)', options=list(predefined_mappings["civilite"].keys()), help="Select the client's civil status")
            delegation = st.selectbox('Main Residence (Delegation)', options=list(predefined_mappings["delegation"].keys()), help="Select the client's residence")
            activite = st.selectbox('Activity/Vocation', options=list(predefined_mappings["activite"].keys()), help="Select the client's occupation")

        with col2:
            st.subheader("Vehicle Information")
            marque = st.selectbox('Vehicle Brand', options=list(predefined_mappings["marque"].keys()), help="Select the brand of the vehicle")
            carrosserie = st.selectbox('Body Type', options=list(predefined_mappings["carrosserie"].keys()), help="Select the vehicle's body type")
            usage = st.selectbox('Vehicle Usage', options=list(predefined_mappings["usage"].keys()), help="Select the usage of the vehicle")
            classe = st.selectbox('Risk Class Assignment', options=list(predefined_mappings["classe"]), help="Assign a risk class")
            anciennete = st.number_input('Client Tenure (years)', min_value=0, max_value=999, help="Enter the number of years the client has been with the company")
            age_objet_assuree = st.number_input('Vehicle Age', min_value=0, max_value=90, help="Enter the vehicle's age")
            puissance = st.number_input('Horsepower', min_value=0, max_value=999, help="Enter the vehicle's horsepower")
            energie = st.selectbox('Fuel Type', options=list(predefined_mappings["energie"].keys()), help="Select the type of fuel the vehicle uses")
            place = st.number_input('Number of Seats', min_value=1, max_value=99, help="Enter the number of seats in the vehicle")
            charge_utile = st.number_input('Payload Capacity (tons)', min_value=0, max_value=999, help="Enter the vehicle's payload capacity")
            valeur_venale = st.number_input('Estimated Value (€)', min_value=0, max_value=9999999, help="Enter the estimated value of the vehicle")
            valeur_neuve = st.number_input('New Price (€)', min_value=0, max_value=9999999, help="Enter the vehicle's new price")

        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                # Map Inputs
                MRQ = predefined_mappings["marque"][marque]
                CRS = predefined_mappings["carrosserie"][carrosserie]
                USG = predefined_mappings["usage"][usage]
                PSS = get_level(puissance, predefined_mappings["puissance"])
                CIV = predefined_mappings["civilite"][civilite]
                AGO = get_level(age_objet_assuree, predefined_mappings["age_objet_assuree"])
                VV = get_level(valeur_venale, predefined_mappings["valeur_ranges"])
                VN = get_level(valeur_neuve, predefined_mappings["valeur_ranges"])
                CU = get_level(charge_utile, predefined_mappings["charge_utile"])
                ANC = get_level(anciennete, predefined_mappings["anciennete"])
                PLA = get_level(place, predefined_mappings["place"])
                AGE = get_level(age_client, predefined_mappings["age_client"])
                DLG = predefined_mappings["delegation"][delegation]
                ACT = predefined_mappings["activite"][activite]
                CLS = float(classe)
                SX = predefined_mappings["sexe"][sexe]
                EN = predefined_mappings["energie"][energie]

                # Feature Array
                features = np.array([
                    USG, ACT, DLG, CIV, MRQ, CRS, EN, SX,
                    PSS, AGO, VV, VN, CU, ANC, CLS, AGE, PLA
                ]).reshape(1, -1)
                features[:, 8:] = scaler.transform(features[:, 8:].reshape(1, -1))

                # Prediction
                prediction = model.predict(features)[0]
                st.markdown(f"<div class='prediction-result'>Prediction: <b>{'Risky' if prediction == 1 else 'Not Risky'}</b></div>", unsafe_allow_html=True)

            except ValueError as e:
                st.markdown(f"<div class='error-message'>Error: {e}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
