# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessors
with open('random_forest.pkl', 'rb') as file:
    artifacts = pickle.load(file)

model = artifacts['model']
label_encoders = artifacts['label_encoders']
scaler = artifacts['scaler']
numeric_features = artifacts['numeric_features']
categorical_features = artifacts['categorical_features']

# Get unique values for categorical features from the label encoders
categorical_options = {
    'usage': list(label_encoders['usage'].classes_),
    'activite': list(label_encoders['activite'].classes_),
    'delegation': list(label_encoders['delegation'].classes_),
    'civilite': list(label_encoders['civilite'].classes_)
}

@app.route('/')
def home():
    return render_template('index.html',
                         usage_options=categorical_options['usage'],
                         activite_options=categorical_options['activite'],
                         delegation_options=categorical_options['delegation'],
                         civilite_options=categorical_options['civilite'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = {}
        
        # Get numeric features
        for feature in numeric_features:
            features[feature] = float(request.form[feature])
        
        # Get categorical features
        for feature in categorical_features:
            features[feature] = request.form[feature]
        
        # Create a DataFrame with the features
        df = pd.DataFrame([features])
        
        # Process features
        # Encode categorical features
        for col in categorical_features:
            df[col] = label_encoders[col].transform(df[col].astype(str))
        
        # Scale numeric features
        df[numeric_features] = scaler.transform(df[numeric_features])
        
        # Make prediction
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)
        
        # Format the result
        result = "Risky" if prediction[0] == 1 else "Not Risky"
        probability = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
        
        return render_template('index.html',
                             prediction_text=f'Prediction: {result} (Probability: {probability:.2%})',
                             usage_options=categorical_options['usage'],
                             activite_options=categorical_options['activite'],
                             delegation_options=categorical_options['delegation'],
                             civilite_options=categorical_options['civilite'])
                             
    except Exception as e:
        return render_template('index.html',
                             prediction_text=f'Error: {str(e)}',
                             usage_options=categorical_options['usage'],
                             activite_options=categorical_options['activite'],
                             delegation_options=categorical_options['delegation'],
                             civilite_options=categorical_options['civilite'])

if __name__ == '__main_myenv\Scripts\activate_':
    app.run(debug=True)