# config.py
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env')) # This loads your .env file

class Config:
    SECRET_KEY = os.environ.get('APP_SECRET_KEY') or 'a_very_secret_risknet_ai_key'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    PORT = int(os.environ.get('APP_PORT', 5003))

    BASE_MODEL_DIR = os.path.join(basedir, "models", "pkl")
    XGBOOST_MODEL_ARTIFACT_PATH = os.path.join(BASE_MODEL_DIR, "xgboost_risk_model.pkl")
    RF_MODEL_ARTIFACT_PATH = os.path.join(BASE_MODEL_DIR, "random_forest_risk_model.pkl")

    # Gemini API Key
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Reads from .env

    # PANGU variables can be removed or repurposed if only using Gemini
    # PANGU_API_ENDPOINT = os.environ.get("LLM_API_ENDPOINT", "YOUR_LLM_ENDPOINT_HERE")
    # PANGU_APP_CODE = os.environ.get("LLM_APP_CODE", "YOUR_LLM_APP_CODE_HERE")

    @staticmethod
    def init_app(app):
        pass