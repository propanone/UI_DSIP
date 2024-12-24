import numpy as np
from flask import Flask, request, render_template
import pickle

# Create Flask app
flask_app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))  # Ensure model.pkl matches your use case

# Define predefined mappings for categorical columns (same as in your model)
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
    },
    "civilite": {
        "Mr": 0, "Mme": 1, "Entreprise": 2, "mult_CT": 3, "Org": 4, "Couple": 5,
        "Etablissement": 6
    }
}

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # List of form input fields
    input_fields = [
        "puissance", "age_objet_assuree", "valeur_venal", "valeur_neuve", "charge_utile",
        "usage", "anciennete", "activite", "classe", "age_client", "delegation", "civilite"
    ]

    # Extract features from the form and map categorical variables using predefined mappings
    try:
        float_features = []
        for field in input_fields:
            value = request.form.get(field)
            if field in predefined_mappings:  # Check if the field is categorical and apply mapping
                # Map the categorical value to its corresponding integer
                value = predefined_mappings[field].get(value, -1)  # -1 for unknown values
            float_features.append(float(value) if isinstance(value, str) and value.isdigit() else value)
    except ValueError:
        return render_template("index.html", prediction_text="Error: Please enter valid numeric inputs.")

    # Convert the input list to a numpy array (model input format)
    features = np.array([float_features])

    # Make prediction using the model
    try:
        prediction = model.predict(features)
        output = prediction[0]
    except Exception as e:
        output = f"Prediction error: {e}"

    # Return the result to the HTML page
    return render_template("index.html", prediction_text=f"The predicted risk is: {output}")

if __name__ == "__main__":
    flask_app.run(debug=True)
