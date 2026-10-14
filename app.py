import joblib
from download_model import download_model
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Wine Quality Prediction API")

MODEL_PATH = "wine_quality.pkl"

download_model()

model = joblib.load(MODEL_PATH)


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "wine_quality"
    }


@app.post("/predict")
def predict(features: WineFeatures):

    data = pd.DataFrame([{
        "fixed acidity": features.fixed_acidity,
        "volatile acidity": features.volatile_acidity,
        "citric acid": features.citric_acid,
        "residual sugar": features.residual_sugar,
        "chlorides": features.chlorides,
        "free sulfur dioxide": features.free_sulfur_dioxide,
        "total sulfur dioxide": features.total_sulfur_dioxide,
        "density": features.density,
        "pH": features.pH,
        "sulphates": features.sulphates,
        "alcohol": features.alcohol
    }])

    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]

    confidence = float(max(probabilities))

    return {
        "prediction": prediction,
        "confidence": confidence
    }
