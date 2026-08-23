import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Trained model
MODEL_PATH = (
    BASE_DIR
    / "Debosmit models"
    / "best_model.joblib"
)

# Prediction log
REPORT_DIR = BASE_DIR / "Debosmit reports"

LOG_PATH = (
    REPORT_DIR
    / "predictions.csv"
)


# Make sure report directory exists
REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("=" * 50)
print("EMPLOYEE ENROLLMENT PREDICTION")
print("=" * 50)

print("\nLoading model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at:\n{MODEL_PATH}\n\n"
        "Please run train.py first."
    )

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 2. GET EMPLOYEE INFORMATION
# ============================================================

print("\nEnter employee information")
print("-" * 40)


age = int(
    input("Age: ")
)


gender = input(
    "Gender (Male/Female): "
).strip()


marital_status = input(
    "Marital status (Single/Married/Divorced): "
).strip()


salary = float(
    input("Salary: ")
)


employment_type = input(
    "Employment type (Full-time/Part-time): "
).strip()


region = input(
    "Region (West/Midwest/Northeast/South): "
).strip()


has_dependents = input(
    "Has dependents (Yes/No): "
).strip()


tenure_years = float(
    input("Tenure years: ")
)


# ============================================================
# 3. CREATE INPUT DATAFRAME
# ============================================================

employee = pd.DataFrame(
    [
        {
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "salary": salary,
            "employment_type": employment_type,
            "region": region,
            "has_dependents": has_dependents,
            "tenure_years": tenure_years,
        }
    ]
)


# ============================================================
# 4. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    employee
)[0]


probability = model.predict_proba(
    employee
)[0][1]


# ============================================================
# 5. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 50)
print("PREDICTION")
print("=" * 50)

print(
    f"\nEnrollment probability: "
    f"{probability:.2%}"
)

print(
    "Predicted enrollment:   ",
    "Yes" if prediction == 1 else "No"
)


# ============================================================
# 6. LOG PREDICTION
# ============================================================

log_row = employee.copy()

log_row["enrollment_probability"] = probability

log_row["predicted_enrollment"] = int(
    prediction
)

log_row["prediction_time"] = (
    datetime.now().isoformat()
)


# Append to existing log or create a new file.

if LOG_PATH.exists():

    log_row.to_csv(
        LOG_PATH,
        mode="a",
        header=False,
        index=False,
    )

else:

    log_row.to_csv(
        LOG_PATH,
        index=False,
    )


print(
    f"\nPrediction logged to:\n"
    f"{LOG_PATH}"
)

print("\nPrediction completed successfully.")