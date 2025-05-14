# services/prediction_service.py
import numpy as np
# No need for pickle here if model object is passed in

def get_model_prediction(processed_features_np, loaded_model_object):
    """
    Gets a prediction using the already loaded model object.
    """
    if processed_features_np.shape[1] != loaded_model_object.n_features_in_:
         raise ValueError(f"Model expects {loaded_model_object.n_features_in_} input features, "
                          f"but received {processed_features_np.shape[1]}. Check data preprocessing.")
    try:
        prediction = loaded_model_object.predict(processed_features_np)
        probability = loaded_model_object.predict_proba(processed_features_np)
        # prediction[0] and probability[0] assume single sample prediction
        return prediction[0], probability[0]
    except Exception as e:
        raise Exception(f"Error during prediction: {e}")