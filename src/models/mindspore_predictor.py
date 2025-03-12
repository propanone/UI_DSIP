# mindspore_predictor.py
import numpy as np
import mindspore as ms
from mindspore import Tensor, context

class MindSporePredictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        # Set inference context
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        
    def predict(self, features):
        """
        Make predictions with the MindSpore model
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            
        Returns:
            predictions: binary classification result (0 or 1)
            probabilities: probability of positive class
        """
        # Convert to MindSpore tensor
        features_tensor = Tensor(features.astype(np.float32))
        
        # Make prediction
        predictions = self.model.predict(features_tensor).asnumpy()
        probabilities = self.model.predict_proba(features_tensor).asnumpy()[:, 1]
        
        return predictions, probabilities