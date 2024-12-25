import os
import pandas as pd
import streamlit as st
import numpy as np
import requests
from requests.auth import HTTPBasicAuth

from src.models import load_models, get_model_name, load_model_to_dict
from src.discretizer import discretize, reverse

# Load env variables

# Path to models
MODELS_PATH = 'C:\\Users\\Sushi\\Documents\\GitHub\\DSIP\\applier-ui\\data\\models\\'

# Page config
st.set_page_config(layout="wide", page_title="Risk Client Prediction")

# Load available models
models = load_models(MODELS_PATH)

# Model selection
model_file = st.selectbox(label="Model", options=models, format_func=get_model_name)

# Constants
RANGES = {

    "age_objet_assuree": [
        {"min": 0, "max": 1, "level": 1},
        {"min": 1, "max": 4, "level": 2},
        {"min": 4, "max": 9, "level": 3},
        {"min": 9, "max": 14, "level": 4},
        {"min": 14, "max": 19, "level": 5},
        {"min": 19, "max": 50, "level": 6}
    ],
    "age_client": [
        {"min": 18, "max": 24, "level": 1},
        {"min": 24, "max": 29, "level": 2},
        {"min": 29, "max": 39, "level": 3},
        {"min": 39, "max": 49, "level": 4},
        {"min": 49, "max": 59, "level": 5},
        {"min": 59, "max": 69, "level": 6},
        {"min": 69, "max": 79, "level": 7},
        {"min": 79, "max": 120, "level": 8}
    ],
    "place": [
        {"min": 1, "max": 5, "level": 1},
        {"min": 5, "max": 9, "level": 2},
        {"min": 9, "max": 29, "level": 3},
        {"min": 29, "max": 59, "level": 4},
        {"min": 59, "max": 999, "level": 5}
    ],
    "anciennete": [
        {"min": 0, "max": 2, "level": 1},
        {"min": 2, "max": 6, "level": 2},
        {"min": 6, "max": 14, "level": 3},
        {"min": 14, "max": 19, "level": 4},
        {"min": 19, "max": 99, "level": 5},
        {"min": 99, "max": 999, "level": 6}
    ],
    "puissance": [
        {"min": 0, "max": 3, "level": 1},
        {"min": 3, "max": 4, "level": 2},
        {"min": 4, "max": 6, "level": 3},
        {"min": 6, "max": 9, "level": 4},
        {"min": 9, "max": 14, "level": 5},
        {"min": 14, "max": 49, "level": 6},
        {"min": 49, "max": 999, "level": 7}
    ],
    "valeur_ranges": [
        {"min": 0, "max": 9999, "level": 1},
        {"min": 9999, "max": 19999, "level": 2},
        {"min": 19999, "max": 29999, "level": 3},
        {"min": 29999, "max": 49999, "level": 4},
        {"min": 49999, "max": 99999, "level": 5},
        {"min": 99999, "max": 499999, "level": 6},
        {"min": 499999, "max": 999999, "level": 7},
        {"min": 999999, "max": 9999999, "level": 8}
    ],

    "charge_utile": [
        {"min": 0, "max": 1, "level": 1},
        {"min": 1, "max": 1.6, "level": 2},
        {"min": 1.6, "max": 3, "level": 3},
        {"min": 3, "max": 10, "level": 4},
        {"min": 10, "max": 999, "level": 5}
    ],
}

MAPPINGS = {
    "usage": {
        "VP": 0, "u1": 1, "moto": 2, "taxi": 3, "U2": 4, "engin": 5, "autre": 6,
        "louage": 7, "transport_rural": 8, "taxi_collectif": 9
    },
    "civilite": {
        "Mr": 0, "Mme": 1, "Entreprise": 2, "mult_CT": 3, "Org": 4, "Couple": 5,
        "Etablissement": 6
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
    "delegation": [
        "Ariana Ville","Sfax Ville","Monastir","El Menzah","Le Bardo","Mannouba","El Mourouj","Hammamet","Sousse Ville","Sakiet Ezzit","Sousse Jaouhara","La Marsa","La Soukra","Nabeul","Ben Arous", "Msaken", "Raoued",  "Sousse Riadh",   "Kairouan Sud",  "Moknine",   "Bizerte Nord",  "Sakiet Eddaier",  "Rades",  "El Omrane Superieur",  "Ezzahra",  "Hammam Sousse",  "Le Kef Est",  "Nouvelle Medina",   "Sfax Sud",   "El Kabbaria",  "Megrine",  "Bou Mhel El Bassatine",  "Hammam Lif",  "Mahdia",  "El Ouerdia",  "La Goulette",  "Gafsa Sud",  "Jendouba Nord",  "Ksibet El Mediouni",  "Beja Nord",  "Carthage",  "Houmet Essouk",  "Korba",   "Fouchana",   "Hammam Chatt",   "Bab Bhar",   "Kalaa El Kebira",   "Zarzis",   "Ettahrir",   "Ksar Helal",   "Ezzouhour (Tunis)",  "Siliana Sud",  "Kalaa Essghira",  "Kelibia",  "Oued Ellil",  "Akouda",  "Dar Chaabane Elfehri",  "Kasserine Nord",  "El Hrairia",  "Gabes Medina",  "Mornag",  "Mnihla",  "Sayada Lamta Bou Hajar",  "Midoun",  "Sidi El Bechir",  "Cite El Khadra",  "Grombalia",  "Mohamadia",  "Zaghouan",  "Sfax Est", "Beni Khiar",   "Sidi Hassine",  "Ettadhamen",  "La Medina",  "Teboulba",  "Feriana",  "Soliman",  "Jemmal",  "La Chebba",  "Mejez El Bab",  "Sidi Bouzid Ouest",   "Sahline",   "Bembla",   "El Kram",  "Gabes Sud",  "Menzel Bourguiba",  "Menzel Temime", "Medenine Sud",   "El Omrane"    "Bou Merdes",   "El Ksar",   "Ras Jebel",   "Ajim",   "Mornaguia",   "Le Kef Ouest",   "Tozeur",   "Beni Khalled",   "Kebili Sud",   "Douar Hicher",   "Menzel Jemil",   "Testour",   "Ghardimaou",  "Tajerouine", "Enfidha", "Gabes Ouest", "Essijoumi", "Ksour Essaf", "Douz", "Menzel Bouzelfa", "Tataouine Sud", "Ouerdanine", "Jedaida", "Souassi",  "El Hamma",  "El Jem",  "Bou Argoub", "Zeramdine",  "Tinja",  "Jebel Jelloud",  "Sidi Thabet",  "Dahmani",  "Mahras",  "Bekalta",  "Jebeniana",  "Kairouan Nord",  "Makthar",   "Ouled Chamakh",  "Agareb",  "Bou Salem",  "Gaafour",  "Bir Ali Ben Khelifa",  "Jarzouna",  "El Haouaria",  "Sakiet Sidi Youssef",  "Bou Hajla",  "Teboursouk",  "Ben Guerdane",  "El Guettar",  "Ain Draham",   "Sned",  "Chorbane",  "Le Sers",  "Ezzouhour (Kasserine)",  "El Amra",  "Nebeur",   "Hammam El Ghezaz",   "Sbikha",   "Bou Ficha",   "Fernana",  "Beni Hassen",   "El Ksour",  "Foussana",  "El Hencha",   "Sidi Bou Ali",  "Degueche",   "Kalaat Sinane",  "Sidi Alouene",  "Hammam Zriba",  "Kerkenah",  "Metlaoui",  "Oueslatia",  "Borj El Amri",  "Bou Arada",  "Tebourba",   "Bizerte Sud",  "El Mida",  "Hergla",  "Thala", "El Mdhilla",   "Sbeitla", "Tabarka",  "Nasrallah",  "El Fahs",  "Bir Mcherga",  "Souk El Ahad",  "Jendouba", "Cherarda", "Mareth", "Mateur", "Hajeb El Ayoun", "Le Krib", "Ennadhour", "Moulares",  "Nefza", "Mejel Bel Abbes",  "El Metouia",  "Haffouz",  "Oued Mliz", "Chebika", "Ghar El Melh", "Bab Souika", "El Alia", "El Ala", "Tataouine Nord", "Menzel Chaker", "Kalaat Landlous", "Esskhira", "Rohia",  "Regueb",   "Bargou", "Sidi El Heni",  "Redeyef",  "Kesra",  "Hassi El Frid",  "Sidi Aich",  "Nefta",  "Beni Khedache",  "Jerissa",  "Nouvelle Matmata",  "Kebili Nord",  "Ghomrassen",  "Melloulech",  "Utique",  "Kalaa El Khasba",  "El Battan",  "Thibar",  "Maknassy",  "Amdoun",  "Takelsa",  "Ghannouche",  "Sidi Bouzid Est",  "Goubellat",   "El Aroussa",  "Saouef",  "Sidi Bou Rouis",  "Sejnane",  "Kasserine Sud",  "Smar",  "Bir El Haffey",  "Ouled Haffouz",  "Ben Oun",  "Kondar",   "Mezzouna",  "Jilma", "Sbiba",  "Ghraiba",  "Bir Lahmar", "Beja Sud",  "Joumine",  "Dhehiba",  "Haidra",  "Hbira",  "Menzel Bouzaiene",  "Gafsa Nord", "Belkhir", "Cebbala", "Sidi Makhlouf", "Jediliane", "Touiref", "Balta Bou Aouene", "Menzel Habib", "Matmata", "Souk Jedid", "Tameghza","Remada","Medenine Nord","Hezoua","Ghezala","El Faouar","El Ayoun"
    ]
}

# Layout
col1, col2 = st.columns(2, gap='medium')

with col1:
    st.subheader("Object Information")
    puissance = st.number_input("Puissance", min_value=0, step=1)
    age_objet = st.number_input("Age de l'objet assuré", min_value=0, step=1)
    valeur_venale = st.number_input("Valeur vénale", min_value=0, step=1000)
    charge_utile = st.number_input("Charge utile", min_value=0.0, step=0.1)
    usage = st.selectbox("Usage", options=MAPPINGS["usage"])

with col2:
    st.subheader("Client Information")
    age_client = st.number_input("Âge du client", min_value=18, step=1)
    civilite = st.selectbox("Civilite", options=MAPPINGS["civilite"])
    activite = st.selectbox("Activité", options=MAPPINGS["activite"])
    classe = st.selectbox("Classe", options=MAPPINGS["classe"])

predict = st.button(label="Predict Risk Level")

if predict:
    model = load_model_to_dict(model_file)
    st.write("Predicting...")
    
    # Discretize values
    instance_parts = [
        f"PM:PUISSANCE_{discretize(puissance, RANGES['puissance']['avg'], RANGES['puissance']['stddev'])}",
        f"PM:AGE_OBJET_{discretize(age_objet, RANGES['age_objet']['avg'], RANGES['age_objet']['stddev'])}",
        f"PM:VALEUR_{discretize(valeur_venale, RANGES['valeur_venale']['avg'], RANGES['valeur_venale']['stddev'])}",
        f"PM:CHARGE_{discretize(charge_utile, RANGES['charge_utile']['avg'], RANGES['charge_utile']['stddev'])}",
        f"PM:AGE_CLIENT_{discretize(age_client, RANGES['age_client']['avg'], RANGES['age_client']['stddev'])}",
        f"PM:USAGE_{usage}",
        f"PM:CIVILITE_{civilite}",
        f"PM:ACTIVITE_{activite}",
        f"PM:CLASSE_{classe}",
        #f"PM:"charge_utile,
        #valeur_venale,
        #puissance
    ]
    
    instance = ",".join(instance_parts)
    
    payload = {
        'instance': instance,
        'model': model
    }
    
    try:
        applier_url = os.environ['APPLIER_URL']
        username = os.environ['SKEYEPREDICT_USERNAME']
        password = os.environ['SKEYEPREDICT_PASSWORD']
        
        headers = {'Content-type': 'application/json'}
        basic = HTTPBasicAuth(username=username, password=password)
        
        response = requests.post(applier_url, json=payload, auth=basic, headers=headers)
        result = response.json()
        
        if result.get('errors'):
            st.error(f"API Error: {result['errors']}")
        else:
            prediction_df = pd.DataFrame(data=result['predictions'])
            
            st.subheader("Prediction Results:")
            for _, row in prediction_df.iterrows():
                annotation = row['annotation']
                risk_level = annotation.split(':')[1]
                confidence = row['confidence']
                
                st.write(f"Risk Level: {risk_level}")
                st.info(f"Confidence: {confidence * 100:.2f}%")
            
            st.warning("Detailed Predictions")
            st.write(prediction_df)
            
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")