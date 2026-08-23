import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime


MODEL_PATH = Path("models/best_model.joblib")
LOG_PATH = Path("reports/predictions.csv")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ---------------------------------------------------------
# Get employee information
# ---------------------------------------------------------

print("\nEnter employee information")
print("-" * 40)

age = int(input("Age: "))

gender = input(
    "Gender (Male/Female): "
)

marital_status = input(
    "Marital status (Single/Married/Divorced): "
)

salary = float(
    input("Salary: ")
)

employment_type = input(
    "Employment type (Full-time/Part-time): "
)

region = input(
    "Region (West/Midwest/Northeast/South): "
)

has_dependents = input(
    "Has dependents (Yes/No): "
)

tenure_years = float(
    input("Tenure years: ")
)


# ---------------------------------------------------------
# Create input DataFrame
# ---------------------------------------------------------

employee = pd.DataFrame(
    [{
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "salary": salary,
        "employment_type": employment_type,
        "region": region,
        "has_dependents": has_dependents,
        "tenure_years": tenure_years,
    }]
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

prediction = model.predict(employee)[0]

probability = model.predict_proba(
    employee
)[0][1]


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("PREDICTION")
print("=" * 50)

print(
    f"Enrollment probability: "
    f"{probability:.2%}"
)

print(
    "Predicted enrollment:",
    "Yes" if prediction == 1 else "No"
)


# ---------------------------------------------------------
# Log prediction
# ---------------------------------------------------------

LOG_PATH.parent.mkdir(
    exist_ok=True
)

log_row = employee.copy()

log_row["enrollment_probability"] = probability
log_row["predicted_enrollment"] = prediction
log_row["prediction_time"] = datetime.now().isoformat()


if LOG_PATH.exists():

    log_row.to_csv(
        LOG_PATH,
        mode="a",
        header=False,
        index=False
    )

else:

    log_row.to_csv(
        LOG_PATH,
        index=False
    )


print(
    f"\nPrediction logged to: {LOG_PATH}"
)