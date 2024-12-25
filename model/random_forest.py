# Cell 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


# Cell 2: Configuration
# Define features and file paths
NUMERIC_FEATURES = [
    "PSS", "AGO", "VV", "VN",
    "CU", "ANC", "CLS", "AGE"
]

CATEGORICAL_FEATURES = ["USG", "ACT", "DLG", "CIV"]

# File paths
INPUT_FILE_PATH = 'output.csv'  # Update this path
MODEL_SAVE_PATH = 'random_forest.pkl'


# Cell 3: Load and Clean Data
# Load the data
data = pd.read_csv(INPUT_FILE_PATH, delimiter=',')

print(data.columns)

# Cell 4: Prepare Features
# Initialize preprocessors
scaler = StandardScaler()
label_encoders = {}

# Prepare features (X)
X = data[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()

# Encode categorical features
for col in CATEGORICAL_FEATURES:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col].astype(str))

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
    X, y, test_size=0.2, random_state=1400
)

# Initialize and train the model
rf = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=1400)
rf.fit(X_train, y_train)

print("Model training completed!")

# Cell 6: Evaluate Model
# Make predictions
y_pred = rf.predict(X_test)

# Print evaluation metrics
print("Model Evaluation Metrics:")
print("-----------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))


# Cell 7: Save Model
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


import os

# Ensure the directory exists
os.makedirs('modelss', exist_ok=True)

# Update the save path
MODEL_SAVE_PATH = 'modelss/random_forest.pkl'

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

# Cell 9: Import Additional Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

# Cell 10: Calculate and Plot Confusion Matrix
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
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


# Cell 11: Calculate and Plot F1 Score
# Calculate F1 score
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

# Cell 12 (Optional): Detailed Metrics Comparison Plot
metrics = {
    'F1 Score': f1_score(y_test, y_pred),
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


# Loop through each column and perform the group by, count, and sort
for col in numeric_cols:
    print(f"Processing column: {col}")
    result = (
        labeled_data.groupby(col)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )
    
    # Display top 10 results for the current column
    print(result.head(260))
    print("\n" + "="*50 + "\n")  # Separator for better readability


    # Add this code at the end of your notebook after training the model
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# Ensure the directory exists
os.makedirs('modelsss', exist_ok=True)

# Create a new model with the same parameters
new_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=14,
    random_state=1400
)

# Fit the model with your training data
new_rf.fit(X_train, y_train)

# Create the artifacts dictionary
artifacts = {
    'model': new_rf,
    'label_encoders': label_encoders,  # Assuming `label_encoders` is defined earlier in your code
    'scaler': scaler,                 # Assuming `scaler` is defined earlier in your code
    'numeric_features': NUMERIC_FEATURES,  # Replace with your actual numeric features list
    'categorical_features': CATEGORICAL_FEATURES  # Replace with your actual categorical features list
}

# Save using joblib (more reliable for scikit-learn objects)
joblib.dump(artifacts, 'modelsss/random_forest.joblib')

print("Model saved successfully!")
