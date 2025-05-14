# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from io import StringIO  # To read the string data

# --- MindSpore Imports ---
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import NumpySlicesDataset
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV # Although not vision, might be used for type casting if needed
from mindspore.common.dtype import Stype # Used for type casting if needed
from mindspore.train import Model, Accuracy, LossMonitor, CheckpointConfig, ModelCheckpoint, BinaryCrossEntropy # Correct loss for binary

# --- Scikit-learn for Preprocessing and Splitting ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # For handling missing values
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# --- Set MindSpore Context ---
# Choose device: "CPU", "GPU", "Ascend"
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU") # GRAPH_MODE is generally faster for training
#ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU") # Use PYNATIVE_MODE for easier debugging
ms.set_seed(42) # for reproducibility
np.random.seed(42)

print(f"MindSpore version: {ms.__version__}")
print(f"Running on device: {ms.get_context('device_target')}")
print(f"MindSpore mode: {ms.get_context('mode')}")

# --- 1. Load Data ---
# Use StringIO to simulate reading from a file
csv_data = read_csv('data.csv')  # Replace with your actual CSV data

# Load data using pandas
try:
    df = pd.read_csv(StringIO(csv_data))
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Display initial info
print("--- Initial Data Info ---")
print(df.head())
print("\nData Shape:", df.shape)
print("\nData Types:\n", df.dtypes)

# --- 2. Preprocessing ---

# Replace empty strings and specific placeholders with NaN for consistent handling
df = df.replace(['', ' ', '?', '#', 'NA', 'N/A', 'None', 'none'], np.nan)

print("\nMissing Values (Before Imputation):\n", df.isnull().sum())

# Predefined Mappings (Copied from prompt) - Ensure consistency
predefined_mappings = {
    "usage": {"VP": 0, "u1": 1, "moto": 2, "taxi": 3, "U2": 4, "engin": 5, "autre": 6, "louage": 7, "transport_rural": 8, "taxi_collectif": 9},
    "civilite": {"Mr": 0, "Mme": 1, "Entreprise": 2, "mult_CT": 3, "Org": 4, "Couple": 5, "Etablissement": 6},
    "activite": {"EDUCATION_FORMATION": 0, "PROFESSIONS_MEDICALES": 1, "EMPLOYE": 2, "RETRAITE": 3, "ACTIVITES_COMMERCIALES": 4, "AGRICULTURE": 5, "RESIDENT_A_L'ETRANGER": 6, "ARTISAN": 7, "CORPS_ACTIFS": 8, "INGENIEUR": 9, "CHAUFFEUR": 10, "PARAMEDICAL": 11, "OUVRIER": 12, "TAXI_LOUAGE_TRASPORT_RURAL": 13, "ARCHITECTURE_BTP_IMMOBILIER": 14, "TECHNICIEN": 15, "GERANT_DIRIGEANT": 16, "PROFESSIONNEL_CONSULTANT_EXPERT": 17, "METIERS_LEGAUX": 18, "INFORMATIQUE": 19, "DIRECTEUR": 20, "TOURISME": 21, "AUTO_ECOLE": 22, "ACTIVITES_SPORTIVES": 23, "ACTIVITES_ARTISTIQUES": 24, "TRANSPORT_AEREEN": 25, "ETAT": 26, "TRANSPORT": 27, "ACTIVITES_FINACIAIRES_ET_BANCAIRES": 28, "JOURNALISME": 29, "DIPLOMATIE": 30, "ASSOCIATIONS_ONG": 31, "SANS_PROFESSION": 32, "ACTIVITES_INDUSTRIELLES": 33},
    # CLS seems numeric or ordinal, treat as float first.
    "age_objet_assuree": [{"min": 0, "max": 1, "level": 1}, {"min": 1, "max": 4, "level": 2}, {"min": 4, "max": 9, "level": 3}, {"min": 9, "max": 14, "level": 4}, {"min": 14, "max": 19, "level": 5}, {"min": 19, "max": 90, "level": 6}], # Map AGO
    "age_client": [{"min": 18, "max": 24, "level": 1}, {"min": 24, "max": 29, "level": 2}, {"min": 29, "max": 39, "level": 3}, {"min": 39, "max": 49, "level": 4}, {"min": 49, "max": 59, "level": 5}, {"min": 59, "max": 69, "level": 6}, {"min": 69, "max": 79, "level": 7}, {"min": 79, "max": 120, "level": 8}], # Map AGE
    "place": [{"min": 1, "max": 5, "level": 1}, {"min": 5, "max": 9, "level": 2}, {"min": 9, "max": 29, "level": 3}, {"min": 29, "max": 59, "level": 4}, {"min": 59, "max": 999, "level": 5}], # Map PLA
    "anciennete": [{"min": 0, "max": 2, "level": 1}, {"min": 2, "max": 6, "level": 2}, {"min": 6, "max": 14, "level": 3}, {"min": 14, "max": 19, "level": 4}, {"min": 19, "max": 99, "level": 5}, {"min": 99, "max": 999, "level": 6}], # Map ANC
    "puissance": [{"min": 0, "max": 3, "level": 1}, {"min": 3, "max": 4, "level": 2}, {"min": 4, "max": 6, "level": 3}, {"min": 6, "max": 9, "level": 4}, {"min": 9, "max": 14, "level": 5}, {"min": 14, "max": 49, "level": 6}, {"min": 49, "max": 999, "level": 7}], # Map PSS
    "valeur_ranges": [{"min": 0, "max": 9999, "level": 1}, {"min": 9999, "max": 19999, "level": 2}, {"min": 19999, "max": 29999, "level": 3}, {"min": 29999, "max": 49999, "level": 4}, {"min": 49999, "max": 99999, "level": 5}, {"min": 99999, "max": 499999, "level": 6}, {"min": 499999, "max": 999999, "level": 7}, {"min": 999999, "max": 9999999, "level": 8}], # Map VV
    "charge_utile": [{"min": 0, "max": 1, "level": 1}, {"min": 1, "max": 1.6, "level": 2}, {"min": 1.6, "max": 3, "level": 3}, {"min": 3, "max": 10, "level": 4}, {"min": 10, "max": 999, "level": 5}], # Map CU
    "marque": {"RENAULT": 0,"VOLKSWAGEN": 1,"PEUGEOT": 2,"FIAT": 3,"CITROEN": 4,"KIA": 5,"FORD": 6,"OPEL": 7,"ISUZU": 8,"MERCEDES-BENZ": 9,"TOYOTA": 10,"HYUNDAI": 11,"NISSAN": 12,"SEAT": 13,"B.M.W.": 14,"CHEVROLET": 15,"AUDI": 16,"MITSUBISHI": 17,"DACIA": 18,"SUZUKI": 19,"MAZDA": 20,"IVECO": 21,"CHERY": 22,"MAHINDRA": 23,"SSANGYONG": 24,"SKODA": 25,"GREATWALL": 26,"MBK": 27,"CHRYSLER": 28,"PIAGGIO": 29,"MINI": 30,"JEEP": 31,"VOLVO": 32,"YAMAHA": 33,"JAGUAR_LAND_ROVER": 34,"HONDA": 35,"TATA": 36,"PO": 37,"ALFA_ROMEO": 38,"MG": 39,"UNISCOOT": 40,"DAEWOO": 41,"JIALING": 42,"LANCIA": 43,"DAIMLER": 44,"BERLIET": 45,"SCANIA": 46,"DONG_FENG_": 47,"DFSK": 48,"ROVER": 49,"APRILIA": 50,"TUNICOM": 51,"COMET": 52,"LANDINI": 53,"WALLYSCAR": 54,"ZIMOTA": 55,"MALAGUTI": 56,"MAN": 57,"SMART": 58,"DAIHATSU": 59,"JEDAA": 60,"MISTRAL": 61,"AVIA": 62,"MASSEY_FERGUSON": 63,"PORSCHE": 64,"FTM": 65,"CATERPILLAR": 66,"FOTON": 67,"KYMCO": 68,"BAIC": 69,"DEUTZ": 70,"KUBOTA": 71,"DAF": 72,"HUARD-TUNISIE": 73,"VESPA": 74,"GILERA": 75,"COMECAB": 76,"SAME_DEUTZ_FAHR": 77,"SAMSUNG": 78,"HIDROMEK": 79,"KINGLONG": 80,"AUTOBIANCHI": 81,"DS": 82,"BENTLEY": 83,"MASERATI": 84,"AIMA": 85,"INFINITI": 86,"BENZHOU": 87,"BOBCAT": 88,"DOOSAN": 89,"SIMATRA": 90,"SYM": 91,"CASE": 92,"BAOLI": 93,"DODGE": 94,"HAVAL": 95,"MAGIRUS": 96,"LADA": 97,"LAMBORGHINI": 98,"GEELY": 99}, # Map MRQ
    "delegation": {'Ariana Ville': 0,'Sfax Ville': 1,'Monastir': 2,'El Menzah': 3,'Le Bardo': 4,'Mannouba': 5,'El Mourouj': 6,'Hammamet': 7,'Sousse Ville': 8,'Sakiet Ezzit': 9,'Sousse Jaouhara': 10,'La Marsa': 11,'La Soukra': 12,'Nabeul': 13,'Ben Arous': 14,'Msaken': 15,'Raoued': 16,'Sousse Riadh': 17,'Kairouan Sud': 18,'Moknine': 19,'Bizerte Nord': 20,'Sakiet Eddaier': 21,'Rades': 22,'El Omrane Superieur': 23,'Ezzahra': 24,'Hammam Sousse': 25,'Le Kef Est': 26,'Nouvelle Medina': 27,'Sfax Sud': 28,'El Kabbaria': 29,'Megrine': 30,'Bou Mhel El Bassatine': 31,'Hammam Lif': 32,'Mahdia': 33,'El Ouerdia': 34,'La Goulette': 35,'Gafsa Sud': 36,'Jendouba Nord': 37,'Ksibet El Mediouni': 38,'Beja Nord': 39,'Carthage': 40,'Houmet Essouk': 41,'Korba': 42,'Fouchana': 43,'Hammam Chatt': 44,'Bab Bhar': 45,'Kalaa El Kebira': 46,'Zarzis': 47,'Ettahrir': 48,'Ksar Helal': 49,'Ezzouhour (Tunis)': 50,'Siliana Sud': 51,'Kalaa Essghira': 52,'Kelibia': 53,'Oued Ellil': 54,'Akouda': 55,'Dar Chaabane Elfehri': 56,'Kasserine Nord': 57,'El Hrairia': 58,'Gabes Medina': 59,'Mornag': 60,'Mnihla': 61,'Sayada Lamta Bou Hajar': 62,'Midoun': 63,'Sidi El Bechir': 64,'Cite El Khadra': 65,'Grombalia': 66,'Mohamadia': 67,'Zaghouan': 68,'Sfax Est': 69,'Beni Khiar': 70,'Sidi Hassine': 71,'Ettadhamen': 72,'La Medina': 73,'Teboulba': 74,'Feriana': 75,'Soliman': 76,'Jemmal': 77,'La Chebba': 78,'Mejez El Bab': 79,'Sidi Bouzid Ouest': 80,'Sahline': 81,'Bembla': 82,'El Kram': 83,'Gabes Sud': 84,'Menzel Bourguiba': 85,'Menzel Temime': 86,'Medenine Sud': 87,'El Omrane': 88,'Bou Merdes': 89,'El Ksar': 90,'Ras Jebel': 91,'Ajim': 92,'Mornaguia': 93,'Le Kef Ouest': 94,'Tozeur': 95,'Beni Khalled': 96,'Kebili Sud': 97,'Douar Hicher': 98,'Menzel Jemil': 99,'Testour': 100,'Ghardimaou': 101,'Tajerouine': 102,'Enfidha': 103,'Gabes Ouest': 104,'Essijoumi': 105,'Ksour Essaf': 106,'Douz': 107,'Menzel Bouzelfa': 108,'Tataouine Sud': 109,'Ouerdanine': 110,'Jedaida': 111,'Souassi': 112,'El Hamma': 113,'El Jem': 114,'Bou Argoub': 115,'Zeramdine': 116,'Tinja': 117,'Jebel Jelloud': 118,'Sidi Thabet': 119,'Dahmani': 120,'Mahras': 121,'Bekalta': 122,'Jebeniana': 123,'Kairouan Nord': 124,'Makthar': 125,'Ouled Chamakh': 126,'Agareb': 127,'Bou Salem': 128,'Gaafour': 129,'Bir Ali Ben Khelifa': 130,'Jarzouna': 131,'El Haouaria': 132,'Sakiet Sidi Youssef': 133,'Bou Hajla': 134,'Teboursouk': 135,'Ben Guerdane': 136,'El Guettar': 137,'Ain Draham': 138,'Sned': 139,'Chorbane': 140,'Le Sers': 141,'Ezzouhour (Kasserine)': 142,'El Amra': 143,'Nebeur': 144,'Hammam El Ghezaz': 145,'Sbikha': 146,'Bou Ficha': 147,'Fernana': 148,'Beni Hassen': 149,'El Ksour': 150,'Foussana': 151,'El Hencha': 152,'Sidi Bou Ali': 153,'Degueche': 154,'Kalaat Sinane': 155,'Sidi Alouene': 156,'Hammam Zriba': 157,'Kerkenah': 158,'Metlaoui': 159,'Oueslatia': 160,'Borj El Amri': 161,'Bou Arada': 162,'Tebourba': 163,'Bizerte Sud': 164,'El Mida': 165,'Hergla': 166,'Thala': 167,'El Mdhilla': 168,'Sbeitla': 169,'Tabarka': 170,'Nasrallah': 171,'El Fahs': 172,'Bir Mcherga': 173,'Souk El Ahad': 174,'Jendouba': 175,'Cherarda': 176,'Mareth': 177,'Mateur': 178,'Hajeb El Ayoun': 179,'Le Krib': 180,'Ennadhour': 181,'Moulares': 182,'Nefza': 183,'Mejel Bel Abbes': 184,'El Metouia': 185,'Haffouz': 186,'Oued Mliz': 187,'Chebika': 188,'Ghar El Melh': 189,'Bab Souika': 190,'El Alia': 191,'El Ala': 192,'Tataouine Nord': 193,'Menzel Chaker': 194,'Kalaat Landlous': 195,'Esskhira': 196,'Rohia': 197,'Regueb': 198,'Bargou': 199,'Sidi El Heni': 200,'Redeyef': 201,'Kesra': 202,'Hassi El Frid': 203,'Sidi Aich': 204,'Nefta': 205,'Beni Khedache': 206,'Jerissa': 207,'Nouvelle Matmata': 208,'Kebili Nord': 209,'Ghomrassen': 210,'Melloulech': 211,'Utique': 212,'Kalaa El Khasba': 213,'El Battan': 214,'Thibar': 215,'Maknassy': 216,'Amdoun': 217,'Takelsa': 218,'Ghannouche': 219,'Sidi Bouzid Est': 220,'Goubellat': 221,'El Aroussa': 222,'Saouef': 223,'Sidi Bou Rouis': 224,'Sejnane': 225,'Kasserine Sud': 226,'Smar': 227,'Bir El Haffey': 228,'Ouled Haffouz': 229,'Ben Oun': 230,'Kondar': 231,'Mezzouna': 232,'Jilma': 233,'Sbiba': 234,'Ghraiba': 235,'Bir Lahmar': 236,'Beja Sud': 237,'Joumine': 238,'Dhehiba': 239,'Haidra': 240,'Hbira': 241,'Menzel Bouzaiene': 242,'Gafsa Nord': 243,'Belkhir': 244,'Cebbala': 245,'Sidi Makhlouf': 246,'Jediliane': 247,'Touiref': 248,'Balta Bou Aouene': 249,'Menzel Habib': 250,'Matmata': 251,'Souk Jedid': 252,'Tameghza': 253,'Remada': 254,'Medenine Nord': 255,'Hezoua': 256,'Ghezala': 257,'El Faouar': 258,'El Ayoun': 259}, # Map DLG
    "carrosserie": {"CI-4P": 0,"BREAK": 1,"CAMIONNETTE": 2,"CI-2P": 3,"CABRIOLET": 4,"CI-5P": 5,"PLATEAU": 6,"FOURGON": 7,"MIXTE": 8,"SOLO": 9,"CI-3P": 10,"BENNE": 11,"CAMION": 12,"ENGIN": 13,"mult_CAROSSERIE": 14,"BUS": 15,"PR REM": 16,"ENGIN_AGRICOLE": 17,"REMORQUAGE": 18,"BD": 19,"BACHE": 20,"PR SREM": 21}, # Map CRS
    "energie": {"ES" : 0,"DI" : 1}, # Map EN
    "sexe" : {"M":0, "F":1,"JP":2,"C":3}, # Map SX
}

# --- Apply Mappings ---

# Direct categorical mappings
df['ACT_Mapped'] = df['ACT'].map(predefined_mappings['activite'])
df['CIV_Mapped'] = df['CIV'].map(predefined_mappings['civilite'])
df['CRS_Mapped'] = df['CRS'].map(predefined_mappings['carrosserie'])
df['DLG_Mapped'] = df['DLG'].map(predefined_mappings['delegation'])
df['EN_Mapped'] = df['EN'].map(predefined_mappings['energie'])
df['MRQ_Mapped'] = df['MRQ'].map(predefined_mappings['marque'])
df['SX_Mapped'] = df['SX'].map(predefined_mappings['sexe'])
df['USG_Mapped'] = df['USG'].map(predefined_mappings['usage'])

# Convert potential numeric columns to numeric types, coercing errors
numeric_cols_for_binning = ['AGE', 'AGO', 'ANC', 'PLA', 'PSS', 'VV', 'CU', 'CLS'] # Added CLS
for col in numeric_cols_for_binning:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Helper function for range-based mapping
def map_to_level(value, ranges):
    if pd.isna(value):
        return np.nan # Keep NaN for imputation later
    try:
        val = float(value)
        for r in ranges:
            # Logic: if it's the last range, it includes the max; otherwise, min <= val < max
            is_last_level = (r == ranges[-1])
            if r['min'] <= val < r['max'] or (is_last_level and val >= r['min'] and val <=r['max']): # <= max for last level
                 return r['level']
            # Special case for the very first range to include the minimum if value is exactly min
            if r == ranges[0] and val == r['min']:
                return r['level']
    except (ValueError, TypeError):
         print(f"Warning: Could not convert value '{value}' to float for range mapping.")
         return np.nan # Cannot convert or invalid type
    # print(f"Warning: Value {value} did not fall into any defined range.") # Be cautious with this, might be noisy
    return np.nan # Value outside all defined ranges

# Apply range-based mappings
df['AGE_Level'] = df['AGE'].apply(lambda x: map_to_level(x, predefined_mappings['age_client']))
df['AGO_Level'] = df['AGO'].apply(lambda x: map_to_level(x, predefined_mappings['age_objet_assuree']))
df['ANC_Level'] = df['ANC'].apply(lambda x: map_to_level(x, predefined_mappings['anciennete']))
df['PLA_Level'] = df['PLA'].apply(lambda x: map_to_level(x, predefined_mappings['place']))
df['PSS_Level'] = df['PSS'].apply(lambda x: map_to_level(x, predefined_mappings['puissance']))
df['VV_Level']  = df['VV'].apply(lambda x: map_to_level(x, predefined_mappings['valeur_ranges']))
df['CU_Level']  = df['CU'].apply(lambda x: map_to_level(x, predefined_mappings['charge_utile']))

# Map target variable 'RISKY'
df['RISKY_Target'] = df['RISKY'].map({'Y': 1, 'N': 0})
df['RISKY_Target'] = pd.to_numeric(df['RISKY_Target'], errors='coerce') # Ensure numeric

# --- Define Features and Target ---
# Features to use in the model (mapped and potentially some original numeric ones if not binned)
# Make sure to include all relevant columns that have been processed.
# Exclude original categorical columns, identifiers (like ID, TV, DG, GOV, C, CEN), and the original target.
# Also excluding FRC, NFC, VN for now as their meaning/mapping wasn't fully clear. Add them if relevant.
feature_columns = [
    'CLS', # Keeping original CLS as numeric for now
    'ACT_Mapped',
    'CIV_Mapped',
    'CRS_Mapped',
    'DLG_Mapped',
    'EN_Mapped',
    'MRQ_Mapped',
    'SX_Mapped',
    'USG_Mapped',
    'AGE_Level',
    'AGO_Level',
    'ANC_Level',
    'PLA_Level',
    'PSS_Level',
    'VV_Level',
    'CU_Level',
    # Add other numeric columns if they are useful and not binned, e.g.,
    'NFC', # Assuming NFC (Number of Family Children?) is numeric
    'FRC', # Assuming FRC is numeric
    'VN',  # Assuming VN is numeric
]
target_column = 'RISKY_Target'

# Select only the relevant columns + target
df_model = df[feature_columns + [target_column]].copy()

# --- Handle Missing Values (Imputation) ---
# Check for missing values again after mapping
print("\nMissing Values (After Mapping, Before Imputation):\n", df_model.isnull().sum())

# Check if target has missing values (critical)
if df_model[target_column].isnull().any():
    print(f"Warning: Target column '{target_column}' contains missing values. Dropping these rows.")
    df_model.dropna(subset=[target_column], inplace=True)
    print(f"Data shape after dropping rows with missing target: {df_model.shape}")

# Impute features - use median for numeric/ordinal, mode for purely categorical (mapped)
# For simplicity here, using median for all feature columns as they are now numeric representations
# A more nuanced approach might use mode for originally categorical features
imputer_median = SimpleImputer(strategy='median')

# Fit and transform the feature columns
df_model[feature_columns] = imputer_median.fit_transform(df_model[feature_columns])

print("\nMissing Values (After Imputation):\n", df_model.isnull().sum())

# --- Convert to Appropriate Data Types ---
# Ensure all features and target are float32 for MindSpore
for col in feature_columns:
    df_model[col] = df_model[col].astype(np.float32)
df_model[target_column] = df_model[target_column].astype(np.float32)

print("\nData Types Before Scaling:\n", df_model.dtypes)

# --- Split Data ---
X = df_model[feature_columns].values
y = df_model[target_column].values

# Check class distribution
print("\nTarget Variable Distribution:")
print(pd.Series(y).value_counts(normalize=True))

if len(np.unique(y)) < 2:
     print("Error: The target variable has only one class after preprocessing. Cannot train a classifier.")
     # Handle this case: maybe collect more data, check preprocessing steps, etc.
     exit()
if len(X) == 0:
     print("Error: No data left after preprocessing.")
     exit()


# Stratified split is good practice for classification, especially with imbalanced data
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
except ValueError as e:
     print(f"Could not perform stratified split (likely due to small sample size for a class): {e}. Performing regular split.")
     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
     )


print("\nData Shapes after Splitting:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# --- Scale Numerical Features ---
# Scale features *after* splitting to prevent data leakage from the test set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use transform only on test data

# --- 3. MindSpore Dataset Creation ---
BATCH_SIZE = 16 # Adjust as needed, smaller for small datasets

def create_dataset(features, labels, batch_size, shuffle=True):
    """Creates a MindSpore dataset."""
    dataset = NumpySlicesDataset(data=(features, labels), column_names=["features", "labels"], shuffle=shuffle)

    # Define transformations - ensure labels are also Float32 if needed by loss fn
    type_cast_op_features = C.TypeCast(ms.float32)
    type_cast_op_labels = C.TypeCast(ms.float32) # BCEWithLogitsLoss expects float labels

    dataset = dataset.map(operations=type_cast_op_features, input_columns="features")
    dataset = dataset.map(operations=type_cast_op_labels, input_columns="labels")

    # Batch the dataset
    dataset = dataset.batch(batch_size)
    return dataset

# Create MindSpore datasets
ds_train = create_dataset(X_train_scaled, y_train, BATCH_SIZE, shuffle=True)
ds_eval = create_dataset(X_test_scaled, y_test, BATCH_SIZE, shuffle=False) # No need to shuffle eval data

print(f"\nTrain dataset size: {ds_train.get_dataset_size()}")
print(f"Eval dataset size: {ds_eval.get_dataset_size()}")

# --- 4. Model Definition ---
class RiskPredictionNet(nn.Cell):
    """Simple MLP for Risk Prediction."""
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=1):
        super(RiskPredictionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Dense(input_dim, hidden_dim1, activation=nn.ReLU())
        self.dropout1 = nn.Dropout(0.3) # Add dropout for regularization
        self.dense2 = nn.Dense(hidden_dim1, hidden_dim2, activation=nn.ReLU())
        self.dropout2 = nn.Dropout(0.3)
        self.dense3 = nn.Dense(hidden_dim2, output_dim) # No activation here, will use BCEWithLogitsLoss
        # self.sigmoid = nn.Sigmoid() # Use Sigmoid if using BinaryCrossEntropy loss

    def construct(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        logits = self.dense3(x)
        # output = self.sigmoid(logits) # Only needed if loss requires probabilities (e.g., BinaryCrossEntropy)
        return logits # Return logits for BCEWithLogitsLoss

input_dimension = X_train_scaled.shape[1]
network = RiskPredictionNet(input_dim=input_dimension)
print("\nModel Architecture:")
print(network)

# --- 5. Training Setup ---
# Loss Function: BCEWithLogitsLoss is numerically stable and expects raw logits
loss_fn = nn.BCEWithLogitsLoss()

# Optimizer
learning_rate = 0.001
optimizer = nn.Adam(network.trainable_params(), learning_rate=learning_rate)

# Metrics: Accuracy for binary classification
# Note: Accuracy expects logits and labels. It applies argmax internally for multi-class,
# but for binary with BCEWithLogitsLoss, we often evaluate based on a threshold (e.g., 0.5) on the *probabilities*.
# MindSpore's Accuracy might need adjustment or use a custom metric if it doesn't handle binary logits correctly.
# Let's try the standard Accuracy first. It might interpret positive logit as class 1, negative as class 0.
# For more robust evaluation, especially with imbalanced data, consider F1-score, Precision, Recall, AUC.
metrics = {"Accuracy": Accuracy()}

# --- 6. Training ---
EPOCHS = 50 # Increase for real datasets, keep low for demonstration
model = Model(network, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

print("\n--- Starting Training ---")
# Callbacks
loss_cb = LossMonitor(per_print_times=ds_train.get_dataset_size()) # Print loss once per epoch
# Checkpoint saving (optional but good practice)
config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size() * 5, # Save every 5 epochs
                             keep_checkpoint_max=3)
ckpt_cb = ModelCheckpoint(prefix="risk_pred", directory="./checkpoints", config=config_ck)

# Check if dataset is empty before training
if ds_train.get_dataset_size() == 0:
    print("Error: Training dataset is empty. Cannot start training.")
else:
    model.train(EPOCHS, ds_train, callbacks=[loss_cb, ckpt_cb], dataset_sink_mode=False) # Sink mode False for CPU often, check performance
    print("--- Training Finished ---")

# --- 7. Evaluation ---
print("\n--- Starting Evaluation ---")

if ds_eval.get_dataset_size() == 0:
    print("Warning: Evaluation dataset is empty. Skipping evaluation.")
    final_acc = None
else:
    eval_results = model.eval(ds_eval, dataset_sink_mode=False)
    print("Evaluation Results:", eval_results)
    final_acc = eval_results.get('Accuracy') # Get accuracy if metric name is 'Accuracy'

# More detailed evaluation using sklearn
print("\n--- Detailed Evaluation (using sklearn) ---")

# Get predictions (logits) from the model
# Note: model.predict expects a dataset or numpy array.
# If using predict on the dataset, iterate through it. Easier to use the numpy array directly.
logits_test = model.predict(ms.Tensor(X_test_scaled, ms.float32))

# Convert logits to probabilities using Sigmoid
sigmoid = ops.Sigmoid()
probabilities_test = sigmoid(logits_test).asnumpy()

# Convert probabilities to class predictions (0 or 1) using a 0.5 threshold
y_pred_test = (probabilities_test > 0.5).astype(int).flatten() # Flatten if output is shape (N, 1)

# Ensure y_test is also integer type for comparison
y_test_int = y_test.astype(int)

# Calculate metrics
accuracy_sk = accuracy_score(y_test_int, y_pred_test)
conf_matrix = confusion_matrix(y_test_int, y_pred_test)
class_report = classification_report(y_test_int, y_pred_test)

print(f"Sklearn Accuracy: {accuracy_sk:.4f}")
if final_acc:
     print(f"MindSpore Model Accuracy: {final_acc:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Calculate AUC
try:
    auc_score = roc_auc_score(y_test_int, probabilities_test)
    print(f"\nAUC Score: {auc_score:.4f}")
except ValueError as e:
    # This can happen if only one class is present in y_test or y_pred
    print(f"\nCould not calculate AUC: {e}")

print("\n--- Script Finished ---")