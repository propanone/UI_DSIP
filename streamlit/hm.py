import streamlit as st
import numpy as np
import pickle
from map import predefined_mappings
import plotly.graph_objects as go

# Configuration and Setup
st.set_page_config(
    page_title='Risk Assessment Dashboard',
    layout='wide',
    page_icon="🎯",
    initial_sidebar_state='expanded'
)

# Custom CSS with modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Form styling */
    .stForm {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Result container */
    .result-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def get_level(value, ranges):
    for range_ in ranges:
        if range_["min"] <= value < range_["max"]:
            return range_["level"]
    raise ValueError(f"Value {value} is out of the defined range!")

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1e3c72"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        },
        title={'text': "Risk Probability"}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        font={'color': "#2a5298", 'family': "Arial"}
    )
    return fig

def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <h1>Risk Assessment Dashboard</h1>
            <p>Client Risk Prediction System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model Selection in Sidebar
    with st.sidebar:
        st.markdown("### Model Settings")
        model_choice = st.selectbox(
            "Select Model",
            ["XGBoost", "Random Forest"]
        )
        
        st.markdown("### About")
        st.info("""
            This dashboard predicts client risk based on various factors including
            client information and vehicle details.
        """)

    # Load appropriate model
    path0 = "C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\models_out\\xgboost_model.pkl"
    path1 = "C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\model_codes\\models_out\\random_forest.pkl"
    path = path0 if model_choice == "XGBoost" else path1  # Using your original paths
    with open(path, 'rb') as file:
        loaded_artifacts = pickle.load(file)

    model = loaded_artifacts['model']
    label_encoders = loaded_artifacts['label_encoders']
    scaler = loaded_artifacts['scaler']

    # Main Form
    with st.form('prediction_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Client Information")
            sexe = st.selectbox('Gender', options=list(predefined_mappings["sexe"].keys()))
            age_client = st.number_input('Age', min_value=18)
            civilite = st.selectbox('Civil Status', options=list(predefined_mappings["civilite"].keys()))
            delegation = st.selectbox('Main Residence', options=list(predefined_mappings["delegation"].keys()))
            activite = st.selectbox('Activity/Vocation', options=list(predefined_mappings["activite"].keys()))

        with col2:
            st.markdown("### 🚗 Vehicle Information")
            marque = st.selectbox('Vehicle Brand', options=list(predefined_mappings["marque"].keys()))
            carrosserie = st.selectbox('Body Type', options=list(predefined_mappings["carrosserie"].keys()))
            usage = st.selectbox('Vehicle Usage', options=list(predefined_mappings["usage"].keys()))
            classe = st.selectbox('Risk Class Assignment', options=list(predefined_mappings["classe"]))
            energie = st.selectbox('Fuel Type', options=list(predefined_mappings["energie"].keys()))

        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 📊 Technical Details")
            anciennete = st.number_input('Client Tenure (years)', min_value=0, max_value=999)
            age_objet_assuree = st.number_input('Vehicle Age (years)', min_value=0, max_value=90)
            puissance = st.number_input('Horsepower', min_value=0, max_value=999)
            place = st.number_input('Number of Seats', min_value=1, max_value=99)

        with col4:
            st.markdown("### 💰 Financial Information")
            charge_utile = st.number_input('Payload Capacity (tons)', min_value=0.0, max_value=999.0, step=0.1)
            valeur_venale = st.number_input('Current Market Value', min_value=0, max_value=9999999)
            valeur_neuve = st.number_input('Original Price', min_value=0, max_value=9999999)

        submitted = st.form_submit_button("Calculate Risk Score")

        if submitted:
            try:
                # Using your original feature processing logic
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

                features = np.array([
                    USG, ACT, DLG, CIV, MRQ, CRS, EN, SX,  # Categorical features
                    PSS, AGO, VV, VN, CU, ANC, CLS, AGE, PLA  # Numeric features
                ]).reshape(1, -1)
                
                features[:, 8:] = scaler.transform(features[:, 8:].reshape(1, -1))

                # Make prediction
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][1]

                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_gauge_chart(probability))

                with col2:
                    st.markdown(f"""
                        <div class="result-container">
                            <h2 style="color: {'#dc3545' if prediction == 1 else '#28a745'}">
                                {"High Risk" if prediction == 1 else "Low Risk"}
                            </h2>
                            <p style="font-size: 1.2rem;">Confidence: {probability * 100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

            except ValueError as e:
                st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()