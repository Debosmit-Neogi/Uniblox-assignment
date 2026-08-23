import joblib

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = joblib.load(
    "C:\\Users\\debos\\Downloads\\Code\\Uniblox-assignment\\Debosmit models\\best_model.joblib"
)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Employee Insurance Enrollment API",
    description=(
        "Predicts whether an employee is likely "
        "to enroll in a voluntary insurance product."
    ),
    version="1.0",
)


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------

class Employee(BaseModel):

    age: int
    gender: str
    marital_status: str
    salary: float
    employment_type: str
    region: str
    has_dependents: str
    tenure_years: float


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Employee Enrollment Prediction API",
        "status": "running",
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(employee: Employee):

    data = pd.DataFrame(
        [{
            "age": employee.age,
            "gender": employee.gender,
            "marital_status": employee.marital_status,
            "salary": employee.salary,
            "employment_type": employee.employment_type,
            "region": employee.region,
            "has_dependents": employee.has_dependents,
            "tenure_years": employee.tenure_years,
        }]
    )

    prediction = model.predict(data)[0]

    probability = model.predict_proba(
        data
    )[0][1]

    return {
        "enrollment_probability": round(
            float(probability),
            4
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