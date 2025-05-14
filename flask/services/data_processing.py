# services/data_processing.py
import numpy as np
import pandas as pd # Useful for organizing data before converting to numpy

# get_level_insurance might not be needed here for model features if they are direct numerics
# from utils.mappings import get_level_insurance # Only if used for display or non-model features

def preprocess_form_data(form_data_dict, model_artifacts_dict):
    """
    Processes raw form data into a NumPy array for model prediction.
    Aligns with the preprocessing steps of the new training scripts.
    """
    # Extract components from the loaded model artifacts
    imputer = model_artifacts_dict['imputer']
    scaler = model_artifacts_dict['scaler']
    feature_order = model_artifacts_dict['feature_order'] # CRITICAL
    # The actual string-to-int maps are in `model_artifacts_dict['predefined_mappings']` if needed,
    # but form values for SelectFields are already the strings we need to map.
    # We need the map.py (loaded as form_select_mappings in app.py) to do the mapping.
    from utils.mappings import predefined_mappings as all_mappings_for_app

    raw_input_dict_for_display = form_data_dict.copy()

    # --- Prepare data in a dictionary matching expected feature names ---
    # This helps manage inputs before converting to the ordered NumPy array.
    processed_data = {}

    # Training script's CATEGORICAL_FEATURES:
    # ["USG", "ACT", "DLG", "CIV", "MRQ", "CRS", "EN", "SX"]
    # Map form field names to these training feature names
    # (form_field_name, training_feature_name, mapping_key_in_all_mappings)
    categorical_mapping_info = [
        ('usage', 'USG', 'usg'),
        ('activite', 'ACT', 'act'),
        ('delegation', 'DLG', 'dlg'),
        ('civilite', 'CIV', 'civ'),
        ('marque', 'MRQ', 'mrq'),
        ('carrosserie', 'CRS', 'crs'),
        ('energie', 'EN', 'en'),
        ('sexe', 'SX', 'sx')
    ]

    for form_key, model_feature_name, map_key_in_all_mappings in categorical_mapping_info:
        form_value_str = form_data_dict.get(form_key)
        if form_value_str is not None and form_value_str != '':
            # Use the mappings from all_mappings_for_app (which is a copy of training map.py)
            mapped_value = all_mappings_for_app[map_key_in_all_mappings].get(form_value_str)
            if mapped_value is None:
                # Value from form not in map, will become NaN
                processed_data[model_feature_name] = np.nan
                print(f"Warning: Value '{form_value_str}' for '{form_key}' (feature {model_feature_name}) not found in mapping. Will be imputed.")
            else:
                processed_data[model_feature_name] = mapped_value
        else:
            processed_data[model_feature_name] = np.nan # Empty or None from form becomes NaN for imputer

    # Training script's NUMERIC_FEATURES:
    # ["PSS", "AGO", "VV", "VN", "CU", "ANC", "CLS", "AGE", "PLA"]
    # Map form field names to these training feature names
    # (form_field_name, training_feature_name, data_type_converter)
    numeric_mapping_info = [
        ('puissance', 'PSS', int),
        ('age_objet_assuree', 'AGO', int),
        ('valeur_venale', 'VV', int),   # Assuming this maps to VV
        ('valeur_neuve', 'VN', int),    # Assuming this maps to VN
        ('charge_utile', 'CU', float),
        ('anciennete', 'ANC', int),
        ('classe', 'CLS', float), # 'classe' from form is numeric-like string e.g. "8.0"
        ('age_client', 'AGE', int),
        ('place', 'PLA', int)
    ]

    for form_key, model_feature_name, converter_func in numeric_mapping_info:
        form_value = form_data_dict.get(form_key)
        if form_value is not None and str(form_value) != '': # Check for empty strings too
            try:
                processed_data[model_feature_name] = converter_func(form_value)
            except (ValueError, TypeError):
                processed_data[model_feature_name] = np.nan # Coercion error, becomes NaN
                print(f"Warning: Value '{form_value}' for '{form_key}' (feature {model_feature_name}) could not be converted to numeric. Will be imputed.")
        else:
            processed_data[model_feature_name] = np.nan # Empty or None from form

    # --- Create DataFrame in the correct feature order ---
    # This ensures that when we convert to NumPy, it's in the order the imputer/scaler expect
    # and that the model was trained on.
    try:
        # Create a temporary DataFrame from processed_data, then reorder
        temp_df = pd.DataFrame([processed_data])
        # Ensure all columns from feature_order are present, fill missing with NaN
        for col in feature_order:
            if col not in temp_df.columns:
                temp_df[col] = np.nan
        features_df_ordered = temp_df[feature_order]
    except KeyError as e:
        raise ValueError(f"Missing expected feature in processed data: {e}. Check form field names and mapping logic.")

    # Convert to NumPy array (should now have NaNs where appropriate)
    features_np_raw_ordered = features_df_ordered.to_numpy()

    if features_np_raw_ordered.shape[1] != len(feature_order):
        raise ValueError(f"Feature array shape mismatch before imputation. Expected {len(feature_order)} features, got {features_np_raw_ordered.shape[1]}. Check feature_order and processing.")

    # --- Apply Imputer ---
    try:
        features_imputed_np = imputer.transform(features_np_raw_ordered)
    except Exception as e:
        raise ValueError(f"Error during imputation: {e}. Input shape: {features_np_raw_ordered.shape}")


    # --- Apply Scaler ---
    try:
        features_scaled_np = scaler.transform(features_imputed_np)
    except Exception as e:
        raise ValueError(f"Error during scaling: {e}. Input shape: {features_imputed_np.shape}")


    # Final check on feature count
    expected_feature_count = len(feature_order)
    if features_scaled_np.shape[1] != expected_feature_count:
         raise ValueError(f"Processed feature array has incorrect shape: {features_scaled_np.shape}. Expected (1, {expected_feature_count}).")

    return features_scaled_np, raw_input_dict_for_display