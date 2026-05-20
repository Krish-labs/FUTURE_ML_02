import joblib
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Define input schema based on dataset features
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str  # Kept as str to handle empty spaces like in raw data

# Global variable for the model pipeline
model_pipeline = None

@app.on_event("startup")
def load_model():
    """Loads the production model on startup."""
    global model_pipeline
    model_path = "models/churn_model.joblib"
    try:
        model_pipeline = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model at {model_path}")

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    """
    Accepts customer data and returns churn probability and risk level.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic model to DataFrame for the pipeline
        input_data = pd.DataFrame([customer.dict()])
        
        # Ensure numeric conversion for TotalCharges (matching training/inference logic)
        input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce')

        # Generate Prediction
        probability = float(model_pipeline.predict_proba(input_data)[:, 1][0])
        prediction = int(model_pipeline.predict(input_data)[0])
        
        risk_level = "High Risk" if probability > 0.5 else "Low Risk"
        
        logger.info(f"Prediction successful. Probability: {probability:.4f}, Risk: {risk_level}")

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk_level
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
