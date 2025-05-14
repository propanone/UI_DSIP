# services/llm_service.py
import google.generativeai as genai
from flask import current_app # To access app.config

# This will be configured when the app starts or on first call
GEMINI_MODEL = None

def configure_gemini():
    """Configures the Gemini API key and model."""
    global GEMINI_MODEL
    api_key = current_app.config.get("GEMINI_API_KEY")
    if not api_key:
        current_app.logger.error("GEMINI_API_KEY not found in configuration. LLM service will not work.")
        return False
    try:
        genai.configure(api_key=api_key)
        # Use gemini-1.5-flash for speed and cost-effectiveness.
        # Verify the exact model name from Google's documentation if this changes.
        GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash-latest') # Or 'gemini-1.5-flash'
        current_app.logger.info("Gemini API configured successfully with model 'gemini-1.5-flash-latest'.")
        return True
    except Exception as e:
        current_app.logger.error(f"Error configuring Gemini API: {e}", exc_info=True)
        GEMINI_MODEL = None # Ensure model is None if config fails
        return False

def generate_risk_summary_with_gemini(risk_data_dict):
    """
    Generates a risk summary using the Gemini API.
    risk_data_dict should contain:
    - model_prediction: "Risky" or "Not Risky"
    - confidence: float (probability of the predicted class)
    - client_age: int
    - vehicle_age: int
    - horsepower: int
    - (Optionally add other key features from form_data_display if relevant for summary)
    """
    if GEMINI_MODEL is None:
        # Attempt to configure if not already done (e.g., after app restart or first call)
        if not configure_gemini():
            return "LLM Service (Gemini) is not configured. Please check API key and server logs."

    prediction = risk_data_dict.get("model_prediction", "N/A")
    confidence = risk_data_dict.get("confidence", 0) * 100 # As percentage
    client_age = risk_data_dict.get("client_age", "N/A")
    vehicle_age = risk_data_dict.get("vehicle_age", "N/A")
    horsepower = risk_data_dict.get("horsepower", "N/A")
    # You can add more features here if they are useful for the LLM prompt
    # For example:
    vehicle_usage = risk_data_dict.get("vehicle_usage_display", "N/A")
    client_activity = risk_data_dict.get("client_activity_display", "N/A")

    prompt = f"""
    Analyze the following car insurance risk assessment and provide a concise, helpful summary for an underwriter.
    The client has been predicted as **{prediction}**.
    The confidence in this prediction is {confidence:.1f}%.

    Key client and vehicle details:
    - Client Age: {client_age} years
    - Vehicle Age: {vehicle_age} years
    - Vehicle Horsepower (Fiscal): {horsepower}

    Consider these factors and provide:
    1. A brief reiteration of the risk level.
    2. Potential reasons or contributing factors (be general if specific feature importance isn't available).
    3. A suggested next step or consideration for the underwriter.

    Keep the summary to 2-3 sentences. Be professional and direct.
    Example for Risky: "The model indicates a {prediction.lower()} profile with {confidence:.1f}% confidence. Factors such as younger client age or older vehicle age might contribute. Recommend reviewing the application details closely and consider premium adjustment or further checks."
    Example for Not Risky: "The model suggests a {prediction.lower()} profile with {confidence:.1f}% confidence. The client's details (Age: {client_age}, Vehicle Age: {vehicle_age}) align with lower-risk indicators. Standard underwriting procedures are likely appropriate."
    """

    try:
        current_app.logger.info(f"Sending prompt to Gemini: {prompt[:200]}...") # Log snippet
        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                # candidate_count=1, # Default is 1
                # stop_sequences=['.'], # Optional: if you want to force shorter sentences
                max_output_tokens=150, # Adjust as needed
                temperature=0.3 # Lower for more factual, higher for more creative
            )
        )
        current_app.logger.info(f"Gemini response received. Safety: {response.prompt_feedback}")
        if response.parts:
            summary = response.text
            current_app.logger.info(f"Gemini summary: {summary}")
            return summary
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            safety_ratings_str = ", ".join([f"{sr.category}: {sr.probability}" for sr in response.prompt_feedback.safety_ratings])
            current_app.logger.warning(f"Gemini content generation blocked. Reason: {block_reason}. Safety Ratings: [{safety_ratings_str}]")
            return f"LLM content generation was blocked due to safety settings (Reason: {block_reason}). Please try rephrasing or check content policies."
        else:
            current_app.logger.warning("Gemini response has no parts and no block reason. Full response: %s", response)
            return "LLM (Gemini) returned an empty or unexpected response."

    except Exception as e:
        current_app.logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return f"Error generating summary with LLM (Gemini): {str(e)}"

# You can rename this if you prefer, or keep the old name for compatibility in app.py
# This makes the change transparent to app.py for now.
def get_pangu_llm_summary_mock(risk_data_dict): # Keep old name for now
    return generate_risk_summary_with_gemini(risk_data_dict)