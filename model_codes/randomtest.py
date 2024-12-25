# Cell 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, fbeta_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from map import predefined_mappings


# Cell 2: Configuration
# Define features and file paths
NUMERIC_FEATURES = [
    "PSS", "AGO", "VV", "VN",
    "CU", "ANC", "CLS", "AGE", "PLA"
]

CATEGORICAL_FEATURES = ["USG", "ACT", "DLG", "CIV", "MRQ", "CRS", "EN",  "SX"]

# File paths
INPUT_FILE_PATH = 'output.csv'  # Update this path
MODEL_SAVE_PATH = 'models_out/random_forest.pkl'


# Cell 3: Load and Clean Data
# Load the dataset
data = pd.read_csv(INPUT_FILE_PATH, delimiter=',')
print(data.columns)

# Handle missing values
#data = data.dropna(subset=CATEGORICAL_FEATURES + NUMERIC_FEATURES)


# Cell 4: Prepare Features
# Initialize preprocessors
scaler = StandardScaler()
label_encoders = {}

# Prepare features (X)
X = data[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()

# Encode categorical features
#for col in CATEGORICAL_FEATURES:
#    label_encoders[col] = LabelEncoder()
#    X[col] = label_encoders[col].fit_transform(X[col].astype(str))

for col in CATEGORICAL_FEATURES:
    X[col] = X[col].map(predefined_mappings[col])      
# Scale numeric features
X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])

# Prepare target (y)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(data['RISKY'])

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# Cell 5: Train Model
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1400, stratify=y
)

# Initialize and train the model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=14,
    random_state=1400
)
rf.fit(X_train, y_train)

print("Model training completed!")


# Cell 6: Evaluate Model
# Make predictions
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]

# Print evaluation metrics
print("Model Evaluation Metrics:")
print("-----------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Yes (Risky)", "No (Non-Risky)"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")


# Cell 7: Save Model
# Ensure the directory exists
os.makedirs('models', exist_ok=True)

# Save the model and preprocessors
model_artifacts = {
    'model': rf,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'target_encoder': target_encoder,
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES
}

with open(MODEL_SAVE_PATH, 'wb') as file:
    pickle.dump(model_artifacts, file)

print(f"Model and preprocessors saved to {MODEL_SAVE_PATH}")


# Cell 8 (Optional): Test Loading Model
# Verify the saved model
with open(MODEL_SAVE_PATH, 'rb') as file:
    loaded_artifacts = pickle.load(file)

print("Available artifacts:", loaded_artifacts.keys())
print("\nModel loaded successfully!")


# Cell 9: Calculate and Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Create confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Yes (Risky)", "No (Non-Risky)"],
            yticklabels=["Yes (Risky)", "No (Non-Risky)"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print confusion matrix interpretation
print("\nConfusion Matrix Interpretation:")
print(f"True Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives: {cm[1][1]}")


# Cell 10: Calculate and Plot F1 Score
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)

# Create F1 score plot
plt.figure(figsize=(8, 4))
plt.barh(['F1 Score'], [f1], color='skyblue')
plt.xlim(0, 1)
plt.title('F1 Score')
plt.xlabel('Score')

# Add value annotation to the bar
plt.text(f1, 0, f'{f1:.3f}', 
         verticalalignment='center',
         fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nF1 Score: {f1:.3f}")


# Cell 11: Detailed Metrics Comparison Plot
from sklearn.metrics import precision_score, recall_score

metrics = {
    'F1 Score': f1,
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'Accuracy': accuracy_score(y_test, y_pred)
}

# Create comparison plot
plt.figure(figsize=(10, 5))
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
bars = plt.barh(list(metrics.keys()), list(metrics.values()), color=colors)
plt.xlim(0, 1)
plt.title('Model Performance Metrics')
plt.xlabel('Score')

# Add value annotations to the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}',
             ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print(f"ROC AUC Score: {roc_auc:.4f}")
