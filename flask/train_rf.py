# train_rf.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Import from map.py
from map import predefined_mappings, get_level

# --- Configuration ---
TARGET_COLUMN = 'RISKY'
INPUT_FILE_PATH = 'data.csv' # Ensure your CSV is named this or change the path
SAVE_DIR = 'models/pkl'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'random_forest_model.pkl')

os.makedirs(SAVE_DIR, exist_ok=True)

# Define raw CSV column names and their mapping configurations
# These names *must* match the column headers in your data.csv
RAW_FEATURES_CONFIG = {
    # Features to be binned using get_level
    'puissance': {'type': 'binned', 'map_key': 'puissance', 'final_name': 'PSS'},
    'age_objet_assuree': {'type': 'binned', 'map_key': 'age_objet_assuree', 'final_name': 'AGO'},
    'valeur_venale': {'type': 'binned', 'map_key': 'valeur_ranges', 'final_name': 'VV'},
    'valeur_neuve': {'type': 'binned', 'map_key': 'valeur_ranges', 'final_name': 'VN'},
    'charge_utile': {'type': 'binned', 'map_key': 'charge_utile', 'final_name': 'CU'},
    'anciennete': {'type': 'binned', 'map_key': 'anciennete', 'final_name': 'ANC'},
    'place': {'type': 'binned', 'map_key': 'place', 'final_name': 'PLA'},
    'age_client': {'type': 'binned', 'map_key': 'age_client', 'final_name': 'AGE'},
    # Feature to be taken as float
    'classe': {'type': 'float', 'final_name': 'CLS'},
    # Categorical features for direct mapping
    'usage': {'type': 'categorical', 'map_key': 'usage', 'final_name': 'USG'},
    'activite': {'type': 'categorical', 'map_key': 'activite', 'final_name': 'ACT'},
    'delegation': {'type': 'categorical', 'map_key': 'delegation', 'final_name': 'DLG'},
    'civilite': {'type': 'categorical', 'map_key': 'civilite', 'final_name': 'CIV'},
    'marque': {'type': 'categorical', 'map_key': 'marque', 'final_name': 'MRQ'},
    'carrosserie': {'type': 'categorical', 'map_key': 'carrosserie', 'final_name': 'CRS'},
    'energie': {'type': 'categorical', 'map_key': 'energie', 'final_name': 'EN'},
    'sexe': {'type': 'categorical', 'map_key': 'sexe', 'final_name': 'SX'},
}

# Define the exact order of features for the model (must match Streamlit app)
FINAL_FEATURE_ORDER = [
    'USG', 'ACT', 'DLG', 'CIV', 'MRQ', 'CRS', 'EN', 'SX',  # Categorical
    'PSS', 'AGO', 'VV', 'VN', 'CU', 'ANC', 'CLS', 'AGE', 'PLA'  # Numerical
]

MISSING_PLACEHOLDERS = ['', ' ', '?', '#', 'NA', 'N/A', 'None', 'none', 'MISSING', np.nan]
TARGET_MAP = {'Y': 1, 'N': 0}
TARGET_NAMES = ["No (Non-Risky)", "Yes (Risky)"]

# --- Load Data ---
print("--- Loading Data ---")
try:
    data = pd.read_csv(INPUT_FILE_PATH, delimiter=',', na_values=MISSING_PLACEHOLDERS)
    print(f"Successfully loaded data from {INPUT_FILE_PATH}. Shape: {data.shape}")
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

raw_feature_cols_needed = list(RAW_FEATURES_CONFIG.keys()) + [TARGET_COLUMN]
missing_cols = [col for col in raw_feature_cols_needed if col not in data.columns]
if missing_cols:
    print(f"Error: Missing required columns in CSV: {missing_cols}")
    exit()

# --- Preprocess Data ---
print("--- Preprocessing Data ---")
data['target_encoded'] = data[TARGET_COLUMN].map(TARGET_MAP)
if data['target_encoded'].isnull().any():
    data.dropna(subset=['target_encoded'], inplace=True)
data['target_encoded'] = data['target_encoded'].astype(int)
y = data['target_encoded']

processed_features = pd.DataFrame(index=data.index)
for raw_col, config in RAW_FEATURES_CONFIG.items():
    final_name = config['final_name']
    if config['type'] == 'binned':
        def safe_get_level(value, ranges):
            try: return get_level(value, ranges)
            except ValueError: return np.nan
            except TypeError: return np.nan
        processed_features[final_name] = data[raw_col].apply(lambda x: safe_get_level(x, predefined_mappings[config['map_key']]))
    elif config['type'] == 'categorical':
        processed_features[final_name] = data[raw_col].map(predefined_mappings[config['map_key']])
    elif config['type'] == 'float':
        processed_features[final_name] = pd.to_numeric(data[raw_col], errors='coerce')

X = processed_features[FINAL_FEATURE_ORDER]
print("Initial missing values in X (after processing, before imputation):")
print(X.isnull().sum()[X.isnull().sum() > 0])

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1400, stratify=y)

# --- Impute and Scale ---
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X.columns, index=X_train.index)
X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=X.columns, index=X_test.index)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed_df)
X_test_scaled = scaler.transform(X_test_imputed_df)
print("Features imputed and scaled.")

# --- Train Random Forest Model ---
print("--- Training Random Forest Model ---")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=14, random_state=1400,
    class_weight='balanced', n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
print("Model training completed!")

# --- Evaluate Model ---
y_pred = rf_model.predict(X_test_scaled)
y_pred_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
try: roc_auc = roc_auc_score(y_test, y_pred_prob)
except ValueError: roc_auc = np.nan
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f if not np.isnan(roc_auc) else 'N/A'}")
print("\nClassification Report:"); print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))
cm = confusion_matrix(y_test, y_pred); print("\nConfusion Matrix:"); print(cm)

# --- Save Model Artifacts ---
model_artifacts = {
    'model': rf_model, 'scaler': scaler, 'imputer': imputer,
    'feature_order': FINAL_FEATURE_ORDER,
    'raw_features_config': RAW_FEATURES_CONFIG,
    'target_map': TARGET_MAP, 'target_names': TARGET_NAMES
}
try:
    with open(MODEL_SAVE_PATH, 'wb') as file: pickle.dump(model_artifacts, file)
    print(f"Model artifacts successfully saved to {MODEL_SAVE_PATH}")
except Exception as e: print(f"Error saving model artifacts: {e}")

# --- Visualizations (Optional) ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
plt.title('Confusion Matrix (Random Forest)'); plt.xlabel('Predicted Label'); plt.ylabel('Actual Label'); plt.show()

try:
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': FINAL_FEATURE_ORDER, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis')
    plt.title('Top Feature Importances (Random Forest)'); plt.tight_layout(); plt.show()
except Exception as e: print(f"Could not generate feature importance plot: {e}")

print("--- Random Forest Training Script Finished ---")