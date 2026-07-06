import pickle
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Income Classifier API")

session = ort.InferenceSession("models/model_quantized.onnx")
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

Instrumentator().instrument(app).expose(app)

class InputData(BaseModel):
    age: float
    workclass: float
    fnlwgt: float
    education: float
    education_num: float
    marital_status: float
    occupation: float
    relationship: float
    race: float
    sex: float
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[
        data.age, data.workclass, data.fnlwgt, data.education,
        data.education_num, data.marital_status, data.occupation,
        data.relationship, data.race, data.sex, data.capital_gain,
        data.capital_loss, data.hours_per_week, data.native_country
    ]], dtype=np.float32)

    features_scaled = scaler.transform(features).astype(np.float32)
    result = session.run(None, {"input": features_scaled})
    probability = float(result[0][0][0])
    prediction = 1 if probability >= 0.5 else 0

    return {
        "prediction": prediction,
        "label": ">50K" if prediction == 1 else "<=50K",
        "probability": round(probability, 4)
        "model_version": "v1.0"
    }