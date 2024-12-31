import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import NumpySlicesDataset
from mindspore import Model, context, load_checkpoint, load_param_into_net
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os
import json
from map import predefined_mappings

# Set MindSpore context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class RiskClassificationNet(nn.Cell):
    def __init__(self, input_dim):
        super(RiskClassificationNet, self).__init__()
        
        self.layer1 = nn.Dense(input_dim, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Dense(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Dense(64, 32)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.output = nn.Dense(32, 2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def construct(self, x):
        x = self.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.layer3(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x

class RiskModel:
    def __init__(self):
        self.NUMERIC_FEATURES = [
            "PSS", "AGO", "VV", "VN",
            "CU", "ANC", "CLS", "AGE", "PLA"
        ]
        
        self.CATEGORICAL_FEATURES = [
            "USG", "ACT", "DLG", "CIV", "MRQ", "CRS", "EN", "SX"
        ]
        
        self.MODEL_DIR = 'models_out'
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, 'mindspore_model.ckpt')
        self.SCALER_PATH = os.path.join(self.MODEL_DIR, 'scaler.json')
        
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.network = None
        
    def train(self, data_path, epochs=50, batch_size=32):
        """Train the model with the given data"""
        print("Loading and preprocessing data...")
        
        # Load data
        data = pd.read_csv(data_path, delimiter=',')
        
        # Split features into numeric and categorical
        X_num = data[self.NUMERIC_FEATURES].copy()
        X_cat = data[self.CATEGORICAL_FEATURES].copy()
        
        # Handle missing values for numeric features
        for col in self.NUMERIC_FEATURES:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
            X_num[col] = X_num[col].fillna(X_num[col].mean())
        
        # Handle missing values and map categorical features
        for col in self.CATEGORICAL_FEATURES:
            X_cat[col] = X_cat[col].fillna('UNKNOWN')
            X_cat[col] = X_cat[col].astype(str)  # Convert to string
            X_cat[col] = X_cat[col].map(predefined_mappings[col])
            X_cat[col] = X_cat[col].fillna(0)  # Handle any unmapped categories
        
        # Scale numeric features
        X_num_scaled = self.scaler.fit_transform(X_num)
        
        # Save scaler parameters
        scaler_params = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist()
        }
        with open(self.SCALER_PATH, 'w') as f:
            json.dump(scaler_params, f)
        
        # Combine scaled numeric and categorical features
        X = np.hstack((X_num_scaled, X_cat.values))
        
        # Prepare target (y)
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(data['RISKY'])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1400, stratify=y
        )
        
        # Convert to correct datatypes
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        
        # Create dataset
        dataset = NumpySlicesDataset(
            data=(X_train, y_train),  # Use tuple here
            column_names=['features', 'label'],
            shuffle=True
        ).batch(batch_size)
        
        # Initialize network
        input_dim = X_train.shape[1]
        self.network = RiskClassificationNet(input_dim)
        
        # Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = nn.Adam(self.network.trainable_params(), learning_rate=0.001)
        
        # Create model
        model = Model(self.network, loss_fn, optimizer, metrics={'accuracy'})
        
        # Train the model
        print("Starting model training...")
        #model.train(epochs=epochs, train_dataset=dataset)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            model.train(epoch=1, train_dataset=dataset)
        
        # Save model
        ms.save_checkpoint(self.network, self.MODEL_PATH)
        print(f"Model saved to {self.MODEL_PATH}")
        
        # Convert test data to tensor for evaluation
        X_test_tensor = ms.Tensor(X_test)
        y_test_tensor = ms.Tensor(y_test)
        
        # Evaluate model
        self._evaluate_model(model, X_test_tensor, y_test_tensor)
        
        return model
    
    def _evaluate_model(self, model, X_test, y_test):
        """Evaluate the model performance"""
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred.asnumpy(), axis=1)
        y_test_numpy = y_test.asnumpy()
        
        accuracy = accuracy_score(y_test_numpy, y_pred_labels)
        report = classification_report(y_test_numpy, y_pred_labels)
        
        print("\nModel Evaluation Metrics:")
        print("-" * 30)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
    
    def load_model(self):
        """Load a trained model"""
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError("Model file not found. Please train the model first.")
        
        if not os.path.exists(self.SCALER_PATH):
            raise FileNotFoundError("Scaler file not found. Please train the model first.")
        
        # Load scaler parameters
        with open(self.SCALER_PATH, 'r') as f:
            scaler_params = json.load(f)
            self.scaler.mean_ = np.array(scaler_params['mean'])
            self.scaler.scale_ = np.array(scaler_params['scale'])
        
        # Load model
        input_dim = len(self.NUMERIC_FEATURES) + len(self.CATEGORICAL_FEATURES)
        self.network = RiskClassificationNet(input_dim)
        param_dict = load_checkpoint(self.MODEL_PATH)
        load_param_into_net(self.network, param_dict)
        self.network.set_train(False)
        
        print("Model loaded successfully!")
    
    def predict(self, input_data):
        """Make prediction for input data"""
        if self.network is None:
            self.load_model()

        # Validate input data
        missing_keys = [key for key in self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES if key not in input_data]
        if missing_keys:
            raise ValueError(f"Missing keys in input data: {missing_keys}")

        # Split input into numeric and categorical
        X_num = pd.DataFrame([{k: input_data.get(k, 0) for k in self.NUMERIC_FEATURES}])
        X_cat = pd.DataFrame([{k: input_data.get(k, 'UNKNOWN') for k in self.CATEGORICAL_FEATURES}])

        # Process numeric features
        for col in self.NUMERIC_FEATURES:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
            X_num[col] = X_num[col].fillna(0)

        # Process categorical features
        for col in self.CATEGORICAL_FEATURES:
            X_cat[col] = X_cat.fillna('UNKNOWN')
            X_cat[col] = X_cat[col].astype(str)
            X_cat[col] = X_cat[col].map(predefined_mappings[col])
            X_cat[col] = X_cat[col].fillna(0)

        # Scale numeric features
        X_num_scaled = self.scaler.transform(X_num)

        # Combine features
        X = np.hstack((X_num_scaled, X_cat.values)).astype(np.float32)

        # Convert to MindSpore tensor
        preprocessed_input = ms.Tensor(X)

        # Make prediction
        output = self.network(preprocessed_input)
        probabilities = nn.Softmax()(output)
        probabilities = probabilities.asnumpy()[0]

        # Create response
        prediction = {
            'risk_probability': float(probabilities[1]),
            'is_risky': bool(probabilities[1] > 0.5),
            'confidence': float(max(probabilities)),
            'probabilities': {
                'not_risky': float(probabilities[0]),
                'risky': float(probabilities[1])
            }
        }

        return prediction


# Example usage
if __name__ == "__main__":
    # Initialize model
    risk_model = RiskModel()
    
    # Train model
    risk_model.train(data_path='output.csv')

    # Make prediction
    #result = risk_model.predict(sample_input)
    #print("Prediction Result:", result)