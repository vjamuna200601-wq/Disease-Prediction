# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

ENSEMBLE_PATH = "ensemble_model.joblib"
ENCODER_PATH = "label_encoder.joblib"
COLUMNS_PATH = "feature_columns.joblib"

app = FastAPI(title="Disease Predictor API")

class Symptoms(BaseModel):
    symptoms: list  # list of symptom names (strings) that match feature column names

# load artifacts at startup
ensemble = joblib.load(ENSEMBLE_PATH)
label_enc = joblib.load(ENCODER_PATH)
feature_columns = joblib.load(COLUMNS_PATH)

@app.post("/predict")
def predict(payload: Symptoms):
    # Construct input vector in the required feature order
    user_vec = np.array([1 if c in payload.symptoms else 0 for c in feature_columns]).reshape(1, -1)
    probs = ensemble.predict_proba(user_vec)[0]
    top_idx = int(np.argmax(probs))
    prediction = label_enc.inverse_transform([top_idx])[0]
    return {"prediction": prediction, "probability": float(probs[top_idx])}
