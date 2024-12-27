# for testing
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
            font-weight: bold;
        }
        .result-container {
            margin-top: 30px;
        }
        .stSidebar {
            background-color: #429ef5;
            color: white;
            padding: 10px;
        }
        .input-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .input-section h3 {
            color: #3498db;
            margin-bottom: 15px;
        }
        .feature-description {
            font-size: 14px;
            color: #7f8c8d;
            margin-top: -10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Paths to the model files
path_to_model_1 = "C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\models_out\\xgboost_model.pkl"
path_to_model_2 = "C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\models_out\\random_forest.pkl"

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as file:
        loaded_artifacts = pickle.load(file)
    return loaded_artifacts


def get_level(value, ranges):
    for range_ in ranges:
        if range_["min"] <= value < range_["max"]:
            return range_["level"]
    raise ValueError(f"Value {value} is out of the defined range!")

def main():
    st.title('Risky Client Prediction')

    # Add a sidebar for navigation and additional options
    st.sidebar.title("Navigation")
    st.sidebar.markdown(
        """
        ### About This App
        This app predicts whether a client is risky or not based on their profile and vehicle details.
        """
    )

    # Main content
    with st.form("prediction_form"):
        st.header("Enter Client and Vehicle Information")

        # Split form into two columns for better organization
        # Sidebar for model selection
        st.sidebar.title("Settings")
        model_choice = st.sidebar.radio("Choose a Model:", ["XGBoost", "Random Forest"])
        selected_path = path_to_model_1 if model_choice == "XGBoost" else path_to_model_2

        loaded_artifacts = load_model(selected_path)
        model = loaded_artifacts['model']
        label_encoders = loaded_artifacts['label_encoders']
        scaler = loaded_artifacts['scaler']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Client Information")
            with st.expander("Details about the client", expanded=True):
                sexe = st.selectbox("Gender", options=list(predefined_mappings["sexe"].keys()))
                st.markdown("<div class='feature-description'>Select the client's gender.</div>", unsafe_allow_html=True)

                age_client = st.number_input("Age", min_value=18)
                st.markdown("<div class='feature-description'>Enter the client's age (18+).</div>", unsafe_allow_html=True)

                civilite = st.selectbox("Civil Status", options=list(predefined_mappings["civilite"].keys()))
                st.markdown("<div class='feature-description'>Select the client's marital status.</div>", unsafe_allow_html=True)

                delegation = st.selectbox("Main Residence", options=list(predefined_mappings["delegation"].keys()))
                st.markdown("<div class='feature-description'>Choose the client's primary residence region.</div>", unsafe_allow_html=True)

                activite = st.selectbox("Activity", options=list(predefined_mappings["activite"].keys()))
                st.markdown("<div class='feature-description'>Choose the client's primary occupation or activity.</div>", unsafe_allow_html=True)

        with col2:
            st.subheader("Vehicle Information")
            with st.expander("Details about the vehicle", expanded=True):
                marque = st.selectbox("Vehicle Brand", options=list(predefined_mappings["marque"].keys()))
                st.markdown("<div class='feature-description'>Select the brand of the vehicle.</div>", unsafe_allow_html=True)

                carrosserie = st.selectbox("Body Type", options=list(predefined_mappings["carrosserie"].keys()))
                st.markdown("<div class='feature-description'>Choose the type of body for the vehicle.</div>", unsafe_allow_html=True)

                usage = st.selectbox("Usage", options=list(predefined_mappings["usage"].keys()))
                st.markdown("<div class='feature-description'>Indicate the vehicle's usage (e.g., personal, commercial).</div>", unsafe_allow_html=True)

                classe = st.selectbox("Risk Class", options=list(predefined_mappings["classe"]))
                st.markdown("<div class='feature-description'>Select the risk class of the vehicle.</div>", unsafe_allow_html=True)

                anciennete = st.number_input("Client Seniority (Years)", min_value=0, max_value=999)
                st.markdown("<div class='feature-description'>Enter the client's seniority in years.</div>", unsafe_allow_html=True)

                age_objet_assuree = st.number_input("Vehicle Age (Years)", min_value=0, max_value=90)
                st.markdown("<div class='feature-description'>Enter the vehicle's age in years.</div>", unsafe_allow_html=True)

                puissance = st.number_input("Horsepower", min_value=0, max_value=999)
                st.markdown("<div class='feature-description'>Specify the horsepower of the vehicle.</div>", unsafe_allow_html=True)

                energie = st.selectbox("Fuel Type", options=list(predefined_mappings["energie"].keys()))
                st.markdown("<div class='feature-description'>Select the vehicle's fuel type.</div>", unsafe_allow_html=True)

                place = st.number_input("Number of Seats", min_value=1, max_value=99)
                st.markdown("<div class='feature-description'>Indicate the number of seats in the vehicle.</div>", unsafe_allow_html=True)

                charge_utile = st.number_input("Payload Capacity (Tons)", min_value=0.0, max_value=999.0, step=0.1)
                st.markdown("<div class='feature-description'>Specify the vehicle's payload capacity in tons.</div>", unsafe_allow_html=True)

                valeur_venale = st.number_input("Estimated Value", min_value=0, max_value=9999999)
                st.markdown("<div class='feature-description'>Enter the estimated market value of the vehicle.</div>", unsafe_allow_html=True)

                valeur_neuve = st.number_input("New Vehicle Price", min_value=0, max_value=9999999)
                st.markdown("<div class='feature-description'>Enter the vehicle's price when new.</div>", unsafe_allow_html=True)

        # Submit button
        submitted = st.form_submit_button("Predict")

        if submitted:
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
                CU = get_level(charge_utile, predefined_mappings["charge_utile"])
                ANC = get_level(anciennete, predefined_mappings["anciennete"])
                PLA = get_level(place, predefined_mappings["place"])
                AGE = get_level(age_client, predefined_mappings["age_client"])
                DLG = predefined_mappings["delegation"][delegation]
                ACT = predefined_mappings["activite"][activite]
                CLS = float(classe)
                SX = predefined_mappings["sexe"][sexe]
                EN = predefined_mappings["energie"][energie]

                # Create feature array
                features = np.array([
                    USG, ACT, DLG, CIV, MRQ, CRS, EN, SX,  # Categorical
                    PSS, AGO, VV, VN, CU, ANC, CLS, AGE, PLA  # Numerical
                ]).reshape(1, -1)

                # Scale numerical features
                features[:, 8:] = scaler.transform(features[:, 8:].reshape(1, -1))

                # Prediction
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]

                # Display results
                result_text = "Risky" if prediction == 1 else "Not Risky"
                st.markdown(f"<div class='prediction-result'>Prediction: <strong>{result_text}</strong></div>", unsafe_allow_html=True)

                st.markdown("### Detailed Insights")

                st.markdown(
                    f"""
                    Here's what I can tell you about this prospect:

                    This client belongs to a persona likely to ... accidents.
                    - **Predictive Loss Rate (S/P)**: 
                    - **Recommended Premium**: 
                    """,
                    unsafe_allow_html=True
                )

                st.bar_chart(
                    {
                        "Risky": probability[1],
                        "Not Risky": probability[0]
                    },
                    use_container_width=True
                )

            except ValueError as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
