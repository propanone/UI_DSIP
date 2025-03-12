import streamlit as st
import numpy as np
import pickle
from map import predefined_mappings
import os
import mindspore as ms
from mindspore_predictor import MindSporePredictor

st.set_page_config(page_title='Risky Client Prediction', layout = 'wide', page_icon ="🌐", initial_sidebar_state = 'expanded')
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling with Huawei theme */
    .header-container {
        padding: 1rem;
        background: linear-gradient(90deg, #CF0A2C 0%, #446ce3 100%);
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Huawei branding badge */
    .huawei-badge {
        text-align: center;
        margin-bottom: 1rem;
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
        background: linear-gradient(90deg, #CF0A2C 0%, #E2001A 100%);
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
    
    /* Model selection styles */
    .model-selection {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .model-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .mindspore-badge {
        background-color: #CF0A2C;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-left: 5px;
    }
</style>
""", unsafe_allow_html=True)

def get_level(value, ranges):
    for range_ in ranges:
        if range_["min"] <= value < range_["max"]:
            return range_["level"]
    raise ValueError(f"Value {value} is out of the defined range!")

def main():
    st.markdown("""
                <div class="header-container">
             <h1>Risk Assessment</h1>
            <p>Client Risk Prediction System</p>
        </div>
        
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ⚙️ **Configuration**")
        
        # Model selection with improved UI
        st.markdown('<div class="model-selection">', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Select Model",
            ["XGBoost", "Random Forest", "MindSpore MLP"]
        )
        
        if model_choice == "MindSpore MLP":
            st.markdown('<span class="mindspore-badge">Huawei AI</span>', unsafe_allow_html=True)
            st.info("Using MindSpore's neural network technology for advanced pattern recognition.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("## 📖 **About**")
        
        about_text = """
            This app predicts client risk based on various factors including
            client information and vehicle details.
        """
        
        if model_choice == "MindSpore MLP":
            about_text += """
            
            The MindSpore MLP model leverages Huawei's deep learning framework
            to identify complex patterns in client data for more accurate risk assessment.
            """
            
        st.info(about_text)
        
        st.markdown("""
        <div class="huawei-badge">
            Powered By 
            <br>
            <img src="https://static.vecteezy.com/system/resources/previews/019/909/399/non_2x/huawei-transparent-huawei-free-free-png.png" 
                 width="100">
            <img src="https://indonet.co.id/wp-content/uploads/2023/10/Huawei-Cloud-Logo-Black.png" 
                 width="100">
        </div>
        """,unsafe_allow_html=True)

    st.title('Client Risk Prediction')
    
    # Load appropriate model based on selection
    path0 = os.path.join(os.path.dirname(__file__), 'models/pkl/xgboost_model.pkl')
    path1 = os.path.join(os.path.dirname(__file__), 'models/pkl/random_forest.pkl')
    path2 = os.path.join(os.path.dirname(__file__), 'models/pkl/mindspore_mlp_model.pkl')

    if model_choice == "XGBoost":
        path = path0
    elif model_choice == "Random Forest":
        path = path1
    else:  # MindSpore MLP
        path = path2
    
    try:
        with open(path, 'rb') as file:
            loaded_artifacts = pickle.load(file)

        if model_choice in ["XGBoost", "Random Forest"]:
            model = loaded_artifacts['model']
            label_encoders = loaded_artifacts['label_encoders']
            scaler = loaded_artifacts['scaler']
        else:  # MindSpore MLP
            model = loaded_artifacts['model']
            scaler = loaded_artifacts['scaler']
            # Create MindSpore predictor
            predictor = MindSporePredictor(model, scaler)
    except FileNotFoundError:
        st.error(f"Model file not found: {path}")
        st.info("Please train the model first using the training script.")
        return

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
                
                # 17 features - ensure same order for all models
                features = np.array([
                     USG, ACT, DLG, CIV, MRQ, CRS, EN, SX,  # Categorical features first
                     PSS, AGO, VV, VN, CU, ANC, CLS, AGE, PLA  # Numeric features second
                ]).reshape(1, -1)
                
                # Make prediction based on model choice
                if model_choice in ["XGBoost", "Random Forest"]:
                    # Scale numeric features
                    features[:, 8:] = scaler.transform(features[:, 8:].reshape(1, -1))
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0]
                else:  # MindSpore MLP
                    # Scale all features using the MindSpore model's scaler
                    features[:, 8:] = scaler.transform(features[:, 8:].reshape(1, -1))
                    prediction_array, probability_array = predictor.predict(features)
                    prediction = prediction_array[0]
                    # For MindSpore, probability_array contains only the positive class probability
                    probability = np.array([1 - probability_array[0], probability_array[0]])
                
                # Display prediction results
                result_title = f"Prediction: {'Risky Client' if prediction == 1 else 'Not Risky Client'}"
                
                if model_choice == "MindSpore MLP":
                    result_title += " (MindSpore AI)"
                
                st.success(result_title)
                
                st.markdown(
                f"""
                This client belongs to a persona likely to be {'Risky' if prediction == 1 else 'Not Risky'}.
                - **Predictive Loss Rate (S/P)**: {f"12.4+{probability[0]:.2f}" if prediction == 0 else f"{12.4+probability[1]:.2f}"}
                - **Recommended Prime**:  {f"{12.4+probability[0]*100:.2f}" if prediction == 0 else f"{12.4+probability[1]*100:.2f}"} DT.
                """,
                unsafe_allow_html=True
                )
                
                # Create chart
                chart_data = {
                    "Risky": float(probability[1]),
                    "Not Risky": float(probability[0])
                }
                
                st.bar_chart(chart_data, use_container_width=True)
                
                # Add model-specific notes for MindSpore
                if model_choice == "MindSpore MLP":
                    st.info("""
                    **MindSpore Analysis**: The neural network has analyzed multiple data dimensions 
                    to produce this prediction, considering complex relationships between client attributes
                    and risk factors.
                    """)

            except ValueError as e:
                st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()