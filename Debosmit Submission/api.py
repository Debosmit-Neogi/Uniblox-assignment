import joblib
import pandas as pd

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "Debosmit models"
    / "best_model.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at:\n{MODEL_PATH}\n\n"
        "Please run train.py first."
    )

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Employee Insurance Enrollment API",
    description=(
        "Predicts whether an employee is likely "
        "to enroll in a voluntary insurance product."
    ),
    version="1.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class Employee(BaseModel):

    age: int

    gender: str

    marital_status: str

    salary: float

    employment_type: str

    region: str

    has_dependents: str

    tenure_years: float


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Employee Enrollment Prediction API",
        "status": "running",
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(employee: Employee):

    # Convert request into DataFrame
    # with the same feature names used during training.

    data = pd.DataFrame(
        [
            {
                "age": employee.age,
                "gender": employee.gender,
                "marital_status": employee.marital_status,
                "salary": employee.salary,
                "employment_type": employee.employment_type,
                "region": employee.region,
                "has_dependents": employee.has_dependents,
                "tenure_years": employee.tenure_years,
            }
        ]
    )


    # Make prediction

    prediction = model.predict(
        data
    )[0]


    # Get probability of enrollment

    probability = model.predict_proba(
        data
    )[0][1]


    # Return prediction

    return {
        "enrollment_probability": round(
            float(probability),
            4,
        ),

        "predicted_enrollment": int(
            prediction
        ),

        "enrolled": (
            "Yes"
            if prediction == 1
            else "No"
        ),
    }
