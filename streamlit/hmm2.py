import streamlit as st
import numpy as np
import pickle
from map import predefined_mappings

# Configure the app
st.set_page_config(
    page_title='Risky Client Prediction', 
    layout='wide', 
    page_icon='assets/risk.png', 
    initial_sidebar_state='expanded'
)

# Custom styling for the app
st.markdown(
    """
    <style>
        .stApp {
            font-family: 'Arial', sans-serif;
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            padding: 10px;
            font-family: 'Helvetica', sans-serif;
        }
        .stButton > button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            margin-top: 20px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #2980b9;
        }
        .prediction-result {
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 10px;
            font-size: 18px;
        }
        .info-icon {
            margin-left: 5px;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Paths to the model files
path_to_model = "C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\models_out\\xgboost_model.pkl"

# Load the model
with open(path_to_model, 'rb') as file:
    loaded_artifacts = pickle.load(file)

model = loaded_artifacts['model']
label_encoders = loaded_artifacts['label_encoders']
scaler = loaded_artifacts['scaler']

def get_level(value, ranges):
    for range_ in ranges:
        if range_["min"] <= value < range_["max"]:
            return range_["level"]
    raise ValueError(f"Value {value} is out of the defined range!")

def main():
    st.title('Risky Client Prediction')

    # Sidebar Information
    st.sidebar.title("Navigation")
    st.sidebar.image("assets/icon.png", use_container_width=True)
    st.sidebar.markdown(
        """
        ### About This App
        This app predicts whether a client is risky or not based on their profile and vehicle details.
        """
    )

    # Create three containers
    container_client = st.container()
    container_vehicle = st.container()
    container_results = st.container()

    with container_client:
        st.header("Client Information")
        with st.form("client_form"):
            col1, col2 = st.columns(2)

            with col1:
                sexe = st.selectbox("Gender", options=list(predefined_mappings["sexe"].keys()))
                age_client = st.number_input(
                    "Age", min_value=18, help="Client's age in years."
                )
                civilite = st.selectbox(
                    "Civil Status", options=list(predefined_mappings["civilite"].keys())
                )
                delegation = st.selectbox(
                    "Main Residence", options=list(predefined_mappings["delegation"].keys())
                )

            with col2:
                activite = st.selectbox(
                    "Activity", options=list(predefined_mappings["activite"].keys())
                )

            submitted_client = st.form_submit_button("Next")

    with container_vehicle:
        st.header("Vehicle Information")
        with st.form("vehicle_form"):
            col1, col2 = st.columns(2)

            with col1:
                marque = st.selectbox("Vehicle Brand", options=list(predefined_mappings["marque"].keys()))
                carrosserie = st.selectbox("Body Type", options=list(predefined_mappings["carrosserie"].keys()))
                usage = st.selectbox("Usage", options=list(predefined_mappings["usage"].keys()))
                energie = st.selectbox("Fuel Type", options=list(predefined_mappings["energie"].keys()))
                classe = st.selectbox(
                    "Risk Class", options=list(predefined_mappings["classe"])
                )

            with col2:
                age_objet_assuree = st.number_input(
                    "Vehicle Age (Years)", min_value=0, max_value=90
                )
                puissance = st.number_input("Horsepower", min_value=0, max_value=999)
                anciennete = st.number_input(
                    "Client Seniority (Years)", min_value=0, max_value=999
                )
                valeur_venale = st.number_input(
                    "Estimated Value", min_value=0, max_value=9999999
                )
                valeur_neuve = st.number_input(
                    "New Vehicle Price", min_value=0, max_value=9999999
                )

            submitted_vehicle = st.form_submit_button("Submit")

    if submitted_vehicle:
        try:
            # Process input values
            MRQ = predefined_mappings["marque"][marque]
            CRS = predefined_mappings["carrosserie"][carrosserie]
            USG = predefined_mappings["usage"][usage]
            PSS = get_level(puissance, predefined_mappings["puissance"])
            CIV = predefined_mappings["civilite"][civilite]
            AGO = get_level(age_objet_assuree, predefined_mappings["age_objet_assuree"])
            VV = get_level(valeur_venale, predefined_mappings["valeur_ranges"])
            VN = get_level(valeur_neuve, predefined_mappings["valeur_ranges"])
            CU = get_level(anciennete, predefined_mappings["anciennete"])
            AGE = get_level(age_client, predefined_mappings["age_client"])
            DLG = predefined_mappings["delegation"][delegation]
            ACT = predefined_mappings["activite"][activite]
            CLS = float(classe)
            SX = predefined_mappings["sexe"][sexe]
            EN = predefined_mappings["energie"][energie]

            # Create feature array
            features = np.array([
                USG, ACT, DLG, CIV, MRQ, CRS, EN, SX,  # Categorical
                PSS, AGO, VV, VN, CU, CLS, AGE  # Numerical
            ]).reshape(1, -1)

            # Scale numerical features
            features[:, 8:] = scaler.transform(features[:, 8:].reshape(1, -1))

            # Prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]

            with container_results:
                st.header("Prediction Results")

                result_text = "Risky" if prediction == 1 else "Not Risky"
                st.markdown(
                    f"<div class='prediction-result'>Prediction: <strong>{result_text}</strong></div>", 
                    unsafe_allow_html=True
                )

                st.subheader("Probability Distribution")
                st.bar_chart(
                    {
                        "Risky": probability[1],
                        "Not Risky": probability[0]
                    },
                    use_container_width=True
                )

                st.markdown(
                    f"""<p>Here's what I can tell you about this prospect:<br>
                    This prospect belongs to a persona of insurees that are likely to commit accidents.<br>
                    Expected Loss Rate (S/P): <strong>{probability[1] * 100:.2f}%</strong><br>
                    Suggested Premium Level: <strong>{probability[1] * 500:.2f} TND</strong></p>""",
                    unsafe_allow_html=True
                )

        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
