
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score
import matplotlib.pyplot as plt
import pickle


# Define features and file paths
NUMERIC_FEATURES = [
    "PSS", "AGO", "VV", "VN",
    "CU", "ANC", "CLS", "AGE"
]

CATEGORICAL_FEATURES = ["USG", "ACT", "DLG", "CIV"]

# File paths
INPUT_FILE_PATH = 'output.csv'  # Update this path
MODEL_SAVE_PATH = 'xgboost_final_model.pkl'


# Load the labeled dataset
labeled_data =data = pd.read_csv(INPUT_FILE_PATH, delimiter=',')

print(data.columns)

# Display the first few rows to confirm
print(labeled_data.head())

###################
# Import required libraries



# Features to retain
features_to_use = [
    "USG", "ACT", "DLG", "CIV","PSS", "AGO", "VV", "VN",
    "CU", "ANC", "CLS", "AGE","RISKY"
]

# Filter only the required features
labeled_data = labeled_data[features_to_use]

# Define categorical and numerical columns
CATEGORICAL_FEATURES = ["USG", "ACT", "DLG", "CIV"]
NUMERIC_FEATURES = ["PSS", "AGO", "VV", "VN",
    "CU", "ANC", "CLS", "AGE"]

#categorical_cols =>CATEGORICAL_FEATURES
#numerical_cols =>NUMERIC_FEATURES


# Ensure all numerical columns are floats or integers
for col in NUMERIC_FEATURES:
    labeled_data[col] = pd.to_numeric(labeled_data[col], errors='coerce')

# Ensure categorical columns are properly set to 'category' dtype
for col in CATEGORICAL_FEATURES:
    labeled_data[col] = labeled_data[col].astype('category')

# Handle missing values: Drop rows with missing values in any critical column
labeled_data = labeled_data.dropna(subset=CATEGORICAL_FEATURES + NUMERIC_FEATURES)

# Define Features (X) and Target (y)
X = labeled_data[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y = (labeled_data["RISKY"] == "No").astype(int)  # Binary: 1 for 'No', 0 for 'Yes'

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Calculate scale_pos_weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Initialize XGBoost Model
xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    tree_method='hist',
    enable_categorical=True,
    scale_pos_weight=scale_pos_weight
)

# Train the model
xgb.fit(X_train, y_train)

# Predictions
y_pred = xgb.predict(X_test)
y_pred_prob = xgb.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Yes (Risky)", "No (Non-Risky)"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")


# Adjust threshold
threshold = 0.4
y_pred_adjusted = (y_pred_prob >= threshold).astype(int)
print("\nClassification Report with Adjusted Threshold:")
print(classification_report(y_test, y_pred_adjusted, target_names=["Yes (Risky)", "No (Non-Risky)"]))
f2_score = fbeta_score(y_test, y_pred_adjusted, beta=2)
print(f"F2-Score: {f2_score:.4f}")


############################################Final###########################
##################run,is kept
#trying class weights
# Define class weights
class_weights = {0: 1, 1: len(y) / (2 * sum(y))}  # Inverse class frequency as weight

# Train the model with class weights
xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    tree_method='hist',
    enable_categorical=True,
    scale_pos_weight=class_weights[1]  # Weight for the minority class
)
xgb.fit(X_train, y_train)

# Evaluate
y_pred = xgb.predict(X_test)
y_pred_prob = xgb.predict_proba(X_test)[:, 1]
print("Classification Report with Cost-Sensitive Learning:")
print(classification_report(y_test, y_pred, target_names=["Yes (Risky)", "No (Non-Risky)"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")

# Save the model using pickle
with open("xgboost_final_model.pkl", "wb") as file:
    pickle.dump(xgb, file)
print("Model saved as 'file:///team5/data/xgboost_final_model.pkl'") #replace with your path
