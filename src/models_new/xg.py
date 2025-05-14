# Main Python Script for XGBoost (e.g., train_xgb.py)

# Cell 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier  # Import XGBoost
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings # To suppress potential division by zero warnings if needed

# Attempt to import mappings from map.py
try:
    from map import predefined_mappings
    print("Successfully imported predefined_mappings from map.py")
except ImportError:
    print("Error: Could not find or import map.py.")
    print("Please ensure map.py containing the predefined_mappings dictionary is in the same directory.")
    exit() # Exit if mappings are essential and not found


# Cell 2: Configuration
# Define features, target, and file paths
TARGET_COLUMN = 'RISKY'

# Features used for training - ensure these match columns in your CSV and mappings
NUMERIC_FEATURES = [
    "PSS", "AGO", "VV", "VN",
    "CU", "ANC", "CLS", "AGE", "PLA"
    # Add other directly numeric features if applicable
]

CATEGORICAL_FEATURES = [
    "USG", "ACT", "DLG", "CIV", "MRQ", "CRS", "EN", "SX"
    # Add other categorical features if applicable
]

# Check if all defined features exist in the mappings (where applicable)
print("\n--- Checking Mappings ---")
all_mappings_found = True
for feature in CATEGORICAL_FEATURES:
    map_key = feature if feature in predefined_mappings else feature.lower()
    if map_key not in predefined_mappings:
        print(f"Warning: Mapping dictionary not found for categorical feature: '{feature}' in predefined_mappings.")
        all_mappings_found = False
if all_mappings_found:
    print("All categorical features have corresponding mapping dictionaries.")

# Define required columns (features + target)
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]

# File paths
INPUT_FILE_PATH = 'data.csv'
SAVE_DIR = 'pkl' # Directory to save artifacts
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'xgboost_risk_model.pkl') # Updated model name

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save directory '{SAVE_DIR}' ensured.")

# Constants for Missing Value Handling
MISSING_PLACEHOLDERS = ['', ' ', '?', '#', 'NA', 'N/A', 'None', 'none', 'MISSING']
UNKNOWN_CATEGORY_VALUE = -1 # Placeholder if needed, though median imputation is used here

# Target mapping
TARGET_MAP = {'Y': 1, 'N': 0} # 1 = Risky, 0 = Non-Risky
TARGET_NAMES = ["No (Non-Risky)", "Yes (Risky)"] # Corresponds to 0 and 1


# Cell 3: Load and Initial Clean Data
print("\n--- Loading Data ---")
try:
    # Use na_values to handle various missing representations during load
    data = pd.read_csv(INPUT_FILE_PATH, delimiter=',', na_values=MISSING_PLACEHOLDERS)
    print(f"Successfully loaded data from {INPUT_FILE_PATH}")
    print("Initial data shape:", data.shape)
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Basic check for required columns
missing_req_cols = [col for col in REQUIRED_COLUMNS if col not in data.columns]
if missing_req_cols:
    print(f"Error: The following required columns are missing in the CSV: {missing_req_cols}")
    exit()

# Keep only necessary columns
data = data[REQUIRED_COLUMNS].copy()
print("Data shape after selecting required columns:", data.shape)


# Cell 4: Preprocess Data (Mapping, Target Encoding, Type Conversion)
print("\n--- Preprocessing Data ---")

# 1. Target Encoding
print(f"Encoding target variable '{TARGET_COLUMN}'...")
initial_target_nan_count = data[TARGET_COLUMN].isnull().sum()
data['target_encoded'] = data[TARGET_COLUMN].map(TARGET_MAP)
encoded_target_nan_count = data['target_encoded'].isnull().sum()

if encoded_target_nan_count > initial_target_nan_count:
    print(f"Warning: {encoded_target_nan_count - initial_target_nan_count} rows had unmappable target values (not 'Y' or 'N').")

# Handle missing target values -> Drop rows
if encoded_target_nan_count > 0:
    print(f"Dropping {encoded_target_nan_count} rows with missing or unmappable target values.")
    data.dropna(subset=['target_encoded'], inplace=True)
    print("Data shape after dropping rows with missing target:", data.shape)

if data.empty:
    print("Error: No data remaining after handling missing target values.")
    exit()

data['target_encoded'] = data['target_encoded'].astype(int)
y = data['target_encoded']
print(f"Target variable '{TARGET_COLUMN}' encoded. Class distribution:")
print(y.value_counts(normalize=True))


# 2. Feature Preparation
X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

# Apply predefined mappings to categorical features
print("Applying predefined mappings to categorical features...")
for col in CATEGORICAL_FEATURES:
    map_key = col if col in predefined_mappings else col.lower()
    if map_key in predefined_mappings:
        print(f"  - Mapping feature: {col} (using key: {map_key})")
        X[col] = X[col].map(predefined_mappings[map_key])
    else:
        print(f"  - Warning: No mapping found for {col}, attempting to coerce to numeric.")
        X[col] = pd.to_numeric(X[col], errors='coerce')


# Convert numeric features to numeric type, coercing errors
print("Ensuring numeric features are of numeric type...")
for col in NUMERIC_FEATURES:
    if X[col].dtype == 'object':
       print(f"  - Converting feature to numeric: {col}")
       X[col] = pd.to_numeric(X[col], errors='coerce')

# Check for non-numeric columns after processing
non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
if non_numeric_cols:
     print(f"Warning: The following feature columns remain non-numeric: {non_numeric_cols}")
     print("Attempting final coercion. This might introduce NaNs.")
     for col in non_numeric_cols:
         X[col] = pd.to_numeric(X[col], errors='coerce')

print("Initial Missing values in features (before imputation):")
print(X.isnull().sum())


# Cell 5: Split Data (Train/Test)
print("\n--- Splitting Data ---")
# Split data *before* imputation and scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,       # Using 25% for testing
    random_state=1400,   # For reproducibility
    stratify=y           # Important for classification
)

print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")


# Cell 6: Impute Missing Values and Scale Features
print("\n--- Imputing and Scaling ---")

# Impute missing values using Median strategy
imputer = SimpleImputer(strategy='median')
print("Fitting imputer on training data...")
imputer.fit(X_train)

print("Transforming training and test data with imputer...")
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert back to DataFrames to keep column names (optional)
X_train = pd.DataFrame(X_train_imputed, columns=X.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_imputed, columns=X.columns, index=X_test.index)

print("Missing values after imputation (Train):", X_train.isnull().sum().sum())
print("Missing values after imputation (Test):", X_test.isnull().sum().sum())


# Scale features using StandardScaler
# Although XGBoost is less sensitive, scaling doesn't hurt and ensures consistency
scaler = StandardScaler()
print("Fitting scaler on training data...")
scaler.fit(X_train) # Fit only on training data

print("Transforming training and test data with scaler...")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep as numpy arrays for XGBoost training
print("Features are now imputed and scaled.")


# Cell 7: Train XGBoost Model
print("\n--- Training XGBoost Model ---")

# Calculate scale_pos_weight for handling class imbalance (important for XGBoost)
count_class_0 = np.sum(y_train == 0)
count_class_1 = np.sum(y_train == 1)

if count_class_1 == 0:
    print("Warning: No samples of the positive class (1) found in the training set. Setting scale_pos_weight to 1.")
    scale_pos_weight = 1
else:
    scale_pos_weight = count_class_0 / count_class_1
    print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.4f}")


# Initialize and train the XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,          # Number of boosting rounds (trees)
    max_depth=14,              # Max depth of each tree
    learning_rate=0.1,         # Step size shrinkage (typical default)
    objective='binary:logistic',# Objective function for binary classification
    random_state=1400,         # For reproducibility
    tree_method='hist',        # Use histogram-based method (often faster)
    enable_categorical=False,  # Set to False as we manually mapped categories
    scale_pos_weight=scale_pos_weight, # Handle class imbalance
    n_jobs=-1                  # Use all available CPU cores
    # Add other hyperparameters like 'subsample', 'colsample_bytree' etc. for tuning
    # use_label_encoder=False, # Deprecated in newer XGBoost, avoid setting unless needed
    # eval_metric='logloss'    # Or 'auc', can be specified
)

# Optional: Use eval_set for early stopping (requires separate validation set or use test set cautiously)
# eval_set = [(X_test_scaled, y_test)]
# xgb_model.fit(X_train_scaled, y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=False)

xgb_model.fit(X_train_scaled, y_train) # Fit without early stopping for simplicity here

print("Model training completed!")


# Cell 8: Evaluate Model
print("\n--- Evaluating Model ---")
# Make predictions on the test set
y_pred = xgb_model.predict(X_test_scaled)
# Get probabilities for the positive class (class 1)
y_pred_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

# --- Print evaluation metrics ---
accuracy = accuracy_score(y_test, y_pred)
# Handle cases where only one class might be present in y_test or predicted probabilities
try:
    roc_auc = roc_auc_score(y_test, y_pred_prob)
except ValueError as e:
    roc_auc = np.nan # Assign NaN if AUC cannot be calculated
    print(f"Could not calculate ROC AUC: {e}")

f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print("-------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"F1 Score (Risky): {f1:.4f}")
print(f"Precision (Risky): {precision:.4f}")
print(f"Recall (Risky): {recall:.4f}")

print("\nClassification Report:")
# Ensure target_names matches the order [0, 1] -> ["No (Non-Risky)", "Yes (Risky)"]
print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))


# Cell 9: Save Model Artifacts
print("\n--- Saving Model Artifacts ---")
# Save the trained model, scaler, imputer, feature lists, and target map
model_artifacts = {
    'model': xgb_model,         # The trained XGBoost model
    'scaler': scaler,           # The fitted scaler instance
    'imputer': imputer,         # The fitted imputer instance
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'feature_order': X.columns.tolist(), # Save the exact order of features
    'target_map': TARGET_MAP,
    'target_names': TARGET_NAMES
}

try:
    with open(MODEL_SAVE_PATH, 'wb') as file:
        pickle.dump(model_artifacts, file)
    print(f"Model artifacts successfully saved to {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Error saving model artifacts: {e}")


# Cell 10 (Optional): Test Loading Model Artifacts
print("\n--- Testing Loading Artifacts ---")
try:
    with open(MODEL_SAVE_PATH, 'rb') as file:
        loaded_artifacts = pickle.load(file)
    print("Model artifacts loaded successfully.")
    print("Available artifacts:", list(loaded_artifacts.keys()))

    # Example: Check the loaded model's type
    if isinstance(loaded_artifacts.get('model'), XGBClassifier):
        print("Loaded model type verified as XGBClassifier.")
    else:
        print("Warning: Loaded model type mismatch.")
except FileNotFoundError:
     print(f"Error: Saved model file not found at {MODEL_SAVE_PATH}")
except Exception as e:
     print(f"Error loading model artifacts: {e}")


# Cell 11: Visualization - Confusion Matrix
print("\n--- Generating Confusion Matrix Plot ---")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=TARGET_NAMES, # Use correct labels
            yticklabels=TARGET_NAMES) # Use correct labels
plt.title('Confusion Matrix (XGBoost)')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# Print interpretation based on TARGET_NAMES order [No, Yes]
print("\nConfusion Matrix Interpretation:")
print(f"True Non-Risky (TN): {cm[0][0]}")
print(f"False Risky (FP): {cm[0][1]}")
print(f"False Non-Risky (FN): {cm[1][0]}")
print(f"True Risky (TP): {cm[1][1]}")


# Cell 12: Visualization - Feature Importances (XGBoost)
print("\n--- Generating Feature Importance Plot ---")
try:
    importances = xgb_model.feature_importances_
    feature_names = X.columns.tolist() # Get feature names
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot top N features
    top_n = 20
    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Feature Importances (XGBoost)')
    plt.xlabel('Importance Score (Gini or Gain based, check XGBoost docs)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    print("\nTop Features:")
    print(feature_importance_df.head(top_n))
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")


# Cell 13: Visualization - ROC Curve
print("\n--- Generating ROC Curve Plot ---")
if not np.isnan(roc_auc): # Only plot if AUC was calculated
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    # roc_auc_value = auc(fpr, tpr) # Already calculated as roc_auc

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity/Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    print(f"ROC AUC Score: {roc_auc:.4f}")
else:
    print("Skipping ROC curve plot as AUC could not be calculated.")


# Cell 14: Visualization - Precision-Recall Curve
print("\n--- Generating Precision-Recall Curve Plot ---")
try:
    precision_points, recall_points, _ = precision_recall_curve(y_test, y_pred_prob)
    avg_precision = average_precision_score(y_test, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_points, precision_points, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - XGBoost')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.show()

    print(f"Average Precision (AP): {avg_precision:.4f}")
except ValueError as e:
    print(f"Could not calculate Precision-Recall curve: {e}")


print("\n--- Analysis Complete ---")