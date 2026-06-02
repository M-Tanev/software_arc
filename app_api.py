from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Prediction API")

try:
    model = joblib.load("model.pkl")
except Exception as e:
    raise RuntimeError(f"Cannot load model: {e}")


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "ML service is running"}


@app.post("/predict")
def predict(data: PredictionRequest):

    if len(data.features) != 4:
        raise HTTPException(
            status_code=400,
            detail="Exactly 4 features are required."
        )

    try:
        X = np.array(data.features).reshape(1, -1)

        prediction = model.predict(X)[0]

        return {
            "prediction": int(prediction)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    
@app.get("/predict")
def predict_example():
    return {
        "message": "Use POST /predict with JSON body"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "features": 4
    }