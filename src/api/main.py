import mlflow
import numpy as np
from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk Prediction API")
# Load model from MLflow Registry
MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Production"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    X = np.array(request.features).reshape(1, -1)

    probability = model.predict(X)[0]

    return PredictionResponse(risk_probability=float(probability))
