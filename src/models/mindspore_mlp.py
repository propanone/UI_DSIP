# Import required libraries
import pandas as pd
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.common.initializer import Normal
from mindspore import context, Tensor, Model, save_checkpoint
from mindspore.train.serialization import export
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import pickle
import os
from map import predefined_mappings

# Set MindSpore context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Define features
NUMERIC_FEATURES = [
   "PSS", "AGO", "VV", "VN",
    "CU", "ANC", "CLS", "AGE", "PLA"
]

CATEGORICAL_FEATURES = ["USG", "ACT", "DLG", "CIV", "MRQ", "CRS", "EN", "SX"]

# File paths
INPUT_FILE_PATH = 'data.csv'  # Update this path
MODEL_SAVE_PATH = 'models/pkl/mindspore_mlp_model.pkl'
ONNX_MODEL_PATH = 'models/mindspore_mlp.onnx'

# Ensure directories exist
os.makedirs('models/pkl', exist_ok=True)

# Define MLP network
class MLPClassifier(nn.Cell):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, output_dim=2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Dense(input_dim, hidden_dim1, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(hidden_dim1, hidden_dim2, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(hidden_dim2, output_dim, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax()
        
    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def predict_proba(self, x):
        logits = self.construct(x)
        return self.softmax(logits)
    
    def predict(self, x):
        probs = self.predict_proba(x)
        return ops.argmax(probs, axis=1)

# Define custom dataset
def create_mindspore_dataset(features, labels, batch_size=32, shuffle=True):
    feature_tensor = Tensor(features.astype(np.float32))
    label_tensor = Tensor(labels.astype(np.int32))
    
    data = {'features': feature_tensor, 'labels': label_tensor}
    dataset = ds.NumpySlicesDataset(data, column_names=['features', 'labels'], shuffle=shuffle)
    dataset = dataset.batch(batch_size)
    return dataset

# Load and preprocess data
def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv(INPUT_FILE_PATH, delimiter=',')
    print("Data loaded. Shape:", data.shape)
    
    # Extract features
    X = data[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    
    # Map categorical features using predefined mappings
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].map(predefined_mappings[col])
    
    # Initialize and fit scaler
    scaler = StandardScaler()
    X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])
    
    # Prepare target (y)
    y = (data['RISKY'] == 'Y').astype(int)
    
    return X, y, scaler

# Main training function
def train_mindspore_model():
    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1400, stratify=y
    )
    
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    # Create MindSpore datasets
    train_dataset = create_mindspore_dataset(X_train.values, y_train.values)
    test_dataset = create_mindspore_dataset(X_test.values, y_test.values, shuffle=False)
    
    # Create MLP model
    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim=input_dim)
    
    # Define loss function and optimizer
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # Calculate class weights for imbalanced dataset
    class_weights = np.array([1.0, len(y_train[y_train == 0]) / len(y_train[y_train == 1])])
    print(f"Class weights: {class_weights}")
    
    # Define optimizer with weight decay
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001, weight_decay=1e-4)
    
    # Wrap model with loss function
    train_model = Model(model, loss_fn, optimizer, metrics={'accuracy'})
    
    # Define callbacks
    config_ck = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix="mlp_classifier", directory='./checkpoints', config=config_ck)
    loss_cb = LossMonitor()
    
    # Train model
    print("Starting model training...")
    train_model.train(epochs=50, train_dataset=train_dataset, callbacks=[ckpoint_cb, loss_cb])
    print("Model training completed!")
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = []
    y_prob = []
    
    for data in test_dataset.create_dict_iterator():
        features = data['features']
        pred = model.predict(features).asnumpy()
        prob = model.predict_proba(features).asnumpy()[:, 1]
        y_pred.extend(pred)
        y_prob.extend(prob)
    
    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    y_test_np = y_test.values
    
    # Print evaluation metrics
    accuracy = accuracy_score(y_test_np, y_pred)
    roc_auc = roc_auc_score(y_test_np, y_prob)
    cm = confusion_matrix(y_test_np, y_pred)
    
    print("Model Evaluation Metrics:")
    print("-----------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred, target_names=["Not Risky", "Risky"]))
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save the model and preprocessors
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES
    }
    
    with open(MODEL_SAVE_PATH, 'wb') as file:
        pickle.dump(model_artifacts, file)
    
    print(f"Model and preprocessors saved to {MODEL_SAVE_PATH}")
    
    # Export to ONNX for better compatibility
    input_data = Tensor(np.ones((1, input_dim)).astype(np.float32))
    export(model, input_data, file_name=ONNX_MODEL_PATH, file_format='ONNX')
    print(f"Model exported to ONNX at {ONNX_MODEL_PATH}")
    
    return model, scaler

# Simple prediction function to test the model
def predict_with_model(model, scaler, sample_data):
    # Preprocess sample data
    for col in CATEGORICAL_FEATURES:
        sample_data[col] = sample_data[col].map(predefined_mappings[col])
    
    # Scale numeric features
    sample_data[NUMERIC_FEATURES] = scaler.transform(sample_data[NUMERIC_FEATURES])
    
    # Convert to MindSpore tensor
    features = Tensor(sample_data.values.astype(np.float32))
    
    # Make prediction
    pred = model.predict(features).asnumpy()
    prob = model.predict_proba(features).asnumpy()[:, 1]
    
    return pred, prob

if __name__ == "__main__":
    model, scaler = train_mindspore_model()
    
    # Test prediction with sample data
    print("\nTesting model with a sample...")
    sample = pd.DataFrame({
        # Fill with sample data matching your features
        "USG": [0], "ACT": [0], "DLG": [0], "CIV": [0], 
        "MRQ": [0], "CRS": [0], "EN": [0], "SX": [0],
        "PSS": [3], "AGO": [2], "VV": [3], "VN": [3],
        "CU": [1], "ANC": [1], "CLS": [5], "AGE": [3], "PLA": [1]
    })
    
    pred, prob = predict_with_model(model, scaler, sample)
    print(f"Prediction: {'Risky' if pred[0] == 1 else 'Not Risky'}")
    print(f"Probability of being risky: {prob[0]:.4f}")