# For this to work change the path to the model line 41 , 42
import streamlit as st
import numpy as np
import pickle
from map import predefined_mappings



st.set_page_config(page_title='Risky Client Prediction', layout = 'wide', page_icon ="risk.png", initial_sidebar_state = 'auto')
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
        font-family: 'Pacifico', cursive;
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
path0 = "C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\models_out\\xgboost_model.pkl"
path1 = "C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\models_out\\random_forest.pkl"
#model = pickle.load(open("C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\xgboost_model.pkl", "rb"))
#model = pickle.load(open("C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\random_forest.pkl", "rb"))

# Load the model
with open(path0, 'rb') as file:
    loaded_artifacts = pickle.load(file)

# Get the model and preprocessors
model = loaded_artifacts['model']
label_encoders = loaded_artifacts['label_encoders']
scaler = loaded_artifacts['scaler']

def get_level(value, ranges):
    for range_ in ranges:
        if range_["min"] <= value < range_["max"]:
            return range_["level"]
    raise ValueError(f"Value {value} is out of the defined range!")

def main():
    st.title('Client Risky Prediction')
    st.image("icon.png", width = 300)

    with st.form('prediction_form'):
        col1, col2 = st.columns(2)

        with col1:
      
            st.subheader("Client Info") 
            sexe = st.selectbox('Gender', options=list(predefined_mappings["sexe"].keys()))
            age_client = st.number_input('Age', min_value=18)
            civilite = st.selectbox('Civil Status (Civilite)',  options=list(predefined_mappings["civilite"].keys()))
            delegation = st.selectbox('Main Residence (Delegation)',  options=list(predefined_mappings["delegation"].keys()))
            activite = st.selectbox('Activity/Vocation', options=list(predefined_mappings["activite"].keys()))
           

        with col2:
            
            st.subheader("Vehicule Info")
            marque = st.selectbox('Vehicule Brand', options=list(predefined_mappings["marque"].keys()))
            carrosserie = st.selectbox('Body Type', options=list(predefined_mappings["carrosserie"].keys()))

            usage = st.selectbox('Usage of Vehicule', options=list(predefined_mappings["usage"].keys()))
            classe = st.selectbox('Risk Classe Assginement', options=list(predefined_mappings["classe"]))
            anciennete = st.number_input('Anciente of client', min_value=0, max_value=999)

            age_objet_assuree = st.number_input('Age of vehicule', min_value=0, max_value=90)
            puissance = st.number_input('Horsepower produced', min_value=0,max_value=999)
            energie = st.selectbox('Fuel Type', options=list(predefined_mappings["energie"].keys()))
            place = st.number_input('Number of Seats', min_value= 1, max_value= 99)
            charge_utile = st.number_input('Payload Capacity (CU) tons', min_value=0,max_value=999)

            valeur_venale = st.number_input('Estimated Value (VV)', min_value=0, max_value=9999999)
            valeur_neuve = st.number_input('Its New price was (VN)', min_value=0, max_value=9999999)


        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                MRQ = predefined_mappings["marque"][marque]
                CRS = predefined_mappings["carrosserie"][carrosserie]
                USG = predefined_mappings["usage"][usage] # [usage] is the input value from the user
                PSS = get_level(puissance, predefined_mappings["puissance"])
                CIV =  predefined_mappings["civilite"][civilite]
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
                # 17 feature
                features = np.array([
                     USG, ACT, DLG, CIV, MRQ, CRS, EN, SX,  # Categorical features first
                     PSS, AGO, VV, VN, CU, ANC, CLS, AGE, PLA  # Numeric features second

                ]).reshape(1, -1)
                features[:, 8:] = scaler.transform(features[:, 8:].reshape(1, -1))


                #input_df = pd.DataFrame([features])
                print(f"Input data shape: {features.shape}")
                print(f"Input data content: {features}")

                # Make prediction
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]

                st.success(f"Prediction: {'Risky' if prediction == 1 else 'Not Risky'}")
                print("Probability:", probability)

            except ValueError as e:
                st.error(f"Erreur: {e}")


if __name__ == "__main__":
    main()
