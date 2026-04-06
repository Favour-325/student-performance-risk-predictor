import io
import boto3
from botocore.exceptions import NoCredentialsError
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .schema import StudentInput


S3_BUCKET = "student-performance-risk-predictor"
S3_KEY    = "gradient_boosting"
CLASS_LABELS = ["At Risk", "Satisfactory", "Excellent"]
AT_RISK_THRESHOLD = 0.41


app = FastAPI(title="Student Performance Risk Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model from S3 at startup when running in the EC2 instance and load from ml_models directory when running locally
try:
    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, S3_KEY, buffer)
    buffer.seek(0)
    model = joblib.load(buffer)

except NoCredentialsError:
    model = joblib.load('app/ml_models/gradient_boosting')


MODEL_METRICS = {
    "model":           "Gradient Boosting + GridSearchCV",
    "dataset":         "Merged student-por + student-mat",
    "feature_selection": "6 low-importance features dropped",
    "threshold":       AT_RISK_THRESHOLD,
    "accuracy":        0.91,
    "macro_f1":        0.90,
    "macro_precision": 0.89,
    "macro_recall":    0.91,
    "per_class": {
        "At Risk": {
            "precision": 0.79,
            "recall":    0.89,
            "f1":        0.84
        },
        "Satisfactory": {
            "precision": 0.96,
            "recall":    0.92,
            "f1":        0.94
        },
        "Excellent": {
            "precision": 0.93,
            "recall":    0.93,
            "f1":        0.93
        }
    }
}

def engineer_features(data: StudentInput) -> pd.DataFrame:
    """Compute the 4 engineered features and return a single-row DataFrame."""
    g1, g2 = data.G1, data.G2
    return pd.DataFrame([{
        "G1":                        g1,
        "G2":                        g2,
        "studytime":                 data.studytime,
        "absences":                  data.absences,
        "age":                       data.age,
        "schoolsup":                 data.schoolsup,
        "Medu":                      data.Medu,
        "Fedu":                      data.Fedu,
        "traveltime":                data.traveltime,
        "Walc":                      data.Walc,
        "Dalc":                      data.Dalc,
        "failures":                  data.failures,
        "grade_trajectory":          g2 - g1,
        "weighted_grade":            0.4 * g1 + 0.6 * g2,
        "study_efficiency":          g2 / (data.studytime + 1),
        "absence_grade_interaction": data.absences * (g1 + g2),
    }])


@app.get("/metrics")
def get_metrics():
    return MODEL_METRICS

@app.post("/predict")
def predict(data: StudentInput):
    try:
        # Build feature row
        X = engineer_features(data)

        # Get class probabilities from the model
        probas = model.predict_proba(X)[0]  # shape: (3,)

        # Apply threshold for At Risk class
        at_risk_idx = CLASS_LABELS.index("At Risk")
        
        if probas[at_risk_idx] >= AT_RISK_THRESHOLD:
            predicted_label = "At Risk"

        else:
            # For the remaining classes, pick the one with the highest probability
            probas_copy = probas.copy()
            probas_copy[at_risk_idx] = 0    # Zero-out the At Risk probability to remove it out of contention (Defensive code, almost useless 😋😂)
            predicted_label = CLASS_LABELS[np.argmax(probas_copy)]

        return {
            "prediction": predicted_label,
            "probabilities": {
                label: round(float(prob), 4)
                for label, prob in zip(CLASS_LABELS, probas)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/static", StaticFiles(directory="app/static"), name="static")
@app.get("/")
def home():
    return FileResponse("app/static/index.html")