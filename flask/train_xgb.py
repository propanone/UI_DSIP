# train_xgb.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

# Import from map.py
from map import predefined_mappings, get_level

# --- Configuration ---
TARGET_COLUMN = 'RISKY'
INPUT_FILE_PATH = 'data.csv' # Ensure your CSV is named this or change the path
SAVE_DIR = 'models/pkl'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'xgboost_model.pkl')

os.makedirs(SAVE_DIR, exist_ok=True)

# Define raw CSV column names and their mapping configurations
# These names *must* match the column headers in your data.csv
RAW_FEATURES_CONFIG = {
    # Features to be binned using get_level
    'puissance': {'type': 'binned', 'map_key': 'puissance', 'final_name': 'PSS'},
    'age_objet_assuree': {'type': 'binned', 'map_key': 'age_objet_assuree', 'final_name': 'AGO'},
    'valeur_venale': {'type': 'binned', 'map_key': 'valeur_ranges', 'final_name': 'VV'}, # Uses 'valeur_ranges'
    'valeur_neuve': {'type': 'binned', 'map_key': 'valeur_ranges', 'final_name': 'VN'},   # Uses 'valeur_ranges'
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

# Check for required raw feature columns and target
raw_feature_cols_needed = list(RAW_FEATURES_CONFIG.keys()) + [TARGET_COLUMN]
missing_cols = [col for col in raw_feature_cols_needed if col not in data.columns]
if missing_cols:
    print(f"Error: Missing required columns in CSV: {missing_cols}")
    exit()

# --- Preprocess Data ---
print("--- Preprocessing Data ---")

# 1. Target Encoding
print(f"Encoding target variable '{TARGET_COLUMN}'...")
data['target_encoded'] = data[TARGET_COLUMN].map(TARGET_MAP)
if data['target_encoded'].isnull().any():
    print(f"Warning: Rows with unmappable target values found. Dropping them.")
    data.dropna(subset=['target_encoded'], inplace=True)
data['target_encoded'] = data['target_encoded'].astype(int)
y = data['target_encoded']
print(f"Target encoded. Class distribution:\n{y.value_counts(normalize=True)}")

# 2. Feature Processing
processed_features = pd.DataFrame(index=data.index)

print("Processing features according to RAW_FEATURES_CONFIG...")
for raw_col, config in RAW_FEATURES_CONFIG.items():
    final_name = config['final_name']
    print(f"  Processing raw column '{raw_col}' to final feature '{final_name}' (type: {config['type']})")
    if config['type'] == 'binned':
        # Handle potential errors from get_level (e.g., value out of range)
        def safe_get_level(value, ranges):
            try:
                return get_level(value, ranges)
            except ValueError as e_level: # Catches if value is out of defined range in get_level
                # print(f"Warning for column {raw_col}, value {value}: {e_level}. Assigning NaN.")
                return np.nan # Assign NaN, to be imputed
            except TypeError as e_type: # Catches if value is not suitable for get_level (e.g. non-numeric string)
                # print(f"Warning for column {raw_col}, value {value}: {e_type}. Assigning NaN.")
                return np.nan

        processed_features[final_name] = data[raw_col].apply(
            lambda x: safe_get_level(x, predefined_mappings[config['map_key']])
        )
    elif config['type'] == 'categorical':
        processed_features[final_name] = data[raw_col].map(predefined_mappings[config['map_key']])
        # Values not in map become NaN, will be imputed
    elif config['type'] == 'float':
        processed_features[final_name] = pd.to_numeric(data[raw_col], errors='coerce')
        # Values that can't be coerced become NaN, will be imputed

# Reorder columns to match FINAL_FEATURE_ORDER for consistency
X = processed_features[FINAL_FEATURE_ORDER]

print("Initial missing values in X (after processing, before imputation):")
print(X.isnull().sum()[X.isnull().sum() > 0])

# --- Split Data ---
print("--- Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1400, stratify=y
)
print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

# --- Impute and Scale ---
print("--- Imputing and Scaling ---")
imputer = SimpleImputer(strategy='median') # Median is robust to outliers
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert back to DataFrame to pass to scaler (maintains column order for scaler)
X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X.columns, index=X_train.index)
X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=X.columns, index=X_test.index)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed_df) # Fit on imputed training data
X_test_scaled = scaler.transform(X_test_imputed_df)   # Transform imputed test data

print("Features imputed and scaled.")

# --- Train XGBoost Model ---
print("--- Training XGBoost Model ---")
count_class_0 = np.sum(y_train == 0)
count_class_1 = np.sum(y_train == 1)
scale_pos_weight = count_class_0 / count_class_1 if count_class_1 > 0 else 1
print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

xgb_model = XGBClassifier(
    n_estimators=200, max_depth=14, learning_rate=0.1,
    objective='binary:logistic', random_state=1400,
    tree_method='hist', enable_categorical=False, # Features are all numeric now
    scale_pos_weight=scale_pos_weight, n_jobs=-1,
    use_label_encoder=False, eval_metric='logloss' # Explicitly set use_label_encoder
)
xgb_model.fit(X_train_scaled, y_train)
print("Model training completed!")

# --- Evaluate Model ---
print("--- Evaluating Model ---")
y_pred = xgb_model.predict(X_test_scaled)
y_pred_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
try:
    roc_auc = roc_auc_score(y_test, y_pred_prob)
except ValueError as e:
    roc_auc = np.nan
    print(f"Could not calculate ROC AUC: {e}")

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f if not np.isnan(roc_auc) else 'N/A'}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- Save Model Artifacts ---
print("--- Saving Model Artifacts ---")
model_artifacts = {
    'model': xgb_model, 'scaler': scaler, 'imputer': imputer,
    'feature_order': FINAL_FEATURE_ORDER, # Critical for ensuring correct input order
    'raw_features_config': RAW_FEATURES_CONFIG, # For reference on how features were derived
    'target_map': TARGET_MAP, 'target_names': TARGET_NAMES
}
try:
    with open(MODEL_SAVE_PATH, 'wb') as file:
        pickle.dump(model_artifacts, file)
    print(f"Model artifacts successfully saved to {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Error saving model artifacts: {e}")

# --- Visualizations (Optional) ---
# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
plt.title('Confusion Matrix (XGBoost)')
plt.xlabel('Predicted Label'); plt.ylabel('Actual Label')
plt.show()

# Feature Importance Plot
try:
    importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': FINAL_FEATURE_ORDER, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis')
    plt.title('Top Feature Importances (XGBoost)'); plt.tight_layout(); plt.show()
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")

print("--- XGBoost Training Script Finished ---")