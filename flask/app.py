from flask import Flask, render_template, request, flash, redirect, url_for, current_app
from config import Config
from forms import CarInsuranceAssessmentForm
from utils.mappings import predefined_mappings as form_select_mappings # For form choices
from services.data_processing import preprocess_form_data
from services.prediction_service import get_model_prediction
from services.llm_service import get_pangu_llm_summary_mock # This now calls Gemini
# Removed configure_gemini import from here, it's called from run.py or llm_service
import os
import pickle
import numpy as np
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config.from_object(Config)

# --- Load Model Artifacts ONCE ---
SELECTED_MODEL_ARTIFACT_PATH = app.config["XGBOOST_MODEL_ARTIFACT_PATH"] # Or RF path

MODEL_ARTIFACTS = None
LOADED_MODEL = None
LOADED_SCALER = None
LOADED_IMPUTER = None
FEATURE_ORDER = None

try:
    with open(SELECTED_MODEL_ARTIFACT_PATH, 'rb') as f:
        MODEL_ARTIFACTS = pickle.load(f)
    # app.logger is not available yet, use print for initial loading feedback
    print(f"INFO: Successfully loaded model artifacts from {SELECTED_MODEL_ARTIFACT_PATH}")
    LOADED_MODEL = MODEL_ARTIFACTS['model']
    LOADED_SCALER = MODEL_ARTIFACTS['scaler']
    LOADED_IMPUTER = MODEL_ARTIFACTS['imputer']
    FEATURE_ORDER = MODEL_ARTIFACTS['feature_order']
except FileNotFoundError:
    print(f"ERROR: FATAL: Model artifact not found at {SELECTED_MODEL_ARTIFACT_PATH}. App may not function correctly.")
    MODEL_ARTIFACTS = None
except Exception as e:
    print(f"ERROR: FATAL: Error loading model artifacts from {SELECTED_MODEL_ARTIFACT_PATH}: {e}")
    MODEL_ARTIFACTS = None

# Setup logging after app object is created
if not app.debug: # Production logging
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/risknet_app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

if MODEL_ARTIFACTS:
    app.logger.info('RiskNet AI Car Insurance App startup - ML Model Loaded')
else:
    app.logger.error('RiskNet AI Car Insurance App startup - FAILED TO LOAD ML MODEL ARTIFACTS')


@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

@app.route('/')
def home_route():
    return render_template('home.html', title="Welcome to RiskNet AI")

@app.route('/car-assessment', methods=['GET', 'POST'])
def car_assessment_route():
    if not MODEL_ARTIFACTS:
        flash("Critical Error: The prediction model is not loaded. Please contact support.", "danger")
        # form_select_mappings should be globally available now
        return render_template('car_assessment_form.html', form=CarInsuranceAssessmentForm(form_select_mappings), title="Car Insurance Client Risk Assessment - System Error")

    form = CarInsuranceAssessmentForm(form_select_mappings)

    if form.validate_on_submit():
        app.logger.info(f"Form submitted with data: {form.data}")
        try:
            form_data_for_processing = form.data.copy()
            form_data_for_processing.pop('csrf_token', None)
            form_data_for_processing.pop('submit', None)

            processed_features_np, raw_input_dict_for_display = preprocess_form_data(
                form_data_for_processing,
                MODEL_ARTIFACTS
            )
            app.logger.info(f"Processed features shape: {processed_features_np.shape}")

            prediction_val, probability_val = get_model_prediction(
                processed_features_np,
                LOADED_MODEL
            )
            app.logger.info(f"Prediction: {prediction_val}, Probability: {probability_val}")

            llm_input_data = {
                "model_prediction": "Risky" if prediction_val == 1 else "Not Risky",
                "confidence": float(probability_val[1] if prediction_val == 1 else probability_val[0]),
                "client_age": form.age_client.data,
                "vehicle_age": form.age_objet_assuree.data,
                "horsepower": form.puissance.data,
                 # Pass display versions of some categorical data for richer LLM prompt
                "vehicle_usage_display": raw_input_dict_for_display.get('usage'),
                "client_activity_display": raw_input_dict_for_display.get('activite'),
                "vehicle_brand_display": raw_input_dict_for_display.get('marque')
            }
            llm_summary = get_pangu_llm_summary_mock(llm_input_data) # Calls Gemini
            app.logger.info(f"LLM Summary: {llm_summary}")

            if not isinstance(probability_val, np.ndarray):
                 probability_val = np.array(probability_val)

            if prediction_val == 0:
                s_p_value = 12.4 + probability_val[0]
                recommended_premium = 12.4 + probability_val[0] * 100
            else:
                s_p_value = 12.4 + probability_val[1]
                recommended_premium = 12.4 + probability_val[1] * 100

            return render_template('car_assessment_results.html',
                                   form_data_display=raw_input_dict_for_display,
                                   prediction=int(prediction_val),
                                   probability=probability_val.tolist(),
                                   model_used=SELECTED_MODEL_ARTIFACT_PATH.split(os.sep)[-1].replace('_risk_model.pkl','').replace('_',' ').title(),
                                   llm_summary=llm_summary,
                                   s_p_value=s_p_value,
                                   recommended_premium=recommended_premium,
                                   title_prefix="Car Assessment Results",
                                   form=form)

        except ValueError as ve:
            flash(f"Error processing input: {str(ve)}", "danger")
            app.logger.error(f"ValueError during processing: {ve}", exc_info=True)
        except FileNotFoundError as fe: # Should be caught at startup for model artifact
            flash(f"A required model or data file was not found: {str(fe)}. Please contact support.", "danger")
            app.logger.error(f"FileNotFoundError: {fe}", exc_info=True)
        except Exception as e:
            flash(f"An unexpected error occurred during risk assessment. Details: {str(e)}", "danger")
            app.logger.error(f"General Exception: {e}", exc_info=True)

        return render_template('car_assessment_form.html', form=form, title="Car Insurance Client Risk Assessment")

    elif request.method == 'POST':
        app.logger.warning(f"Form validation failed: {form.errors}")
        flash("Please correct the errors below and resubmit.", "warning")

    return render_template('car_assessment_form.html', form=form, title="Car Insurance Client Risk Assessment")

# Removed other insurance type routes