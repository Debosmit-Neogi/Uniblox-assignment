# Employee Insurance Enrollment Prediction

This project implements an end-to-end machine learning pipeline to predict whether an employee will enroll in a voluntary insurance product.

The project includes:

- Data preprocessing
- Random Forest classification
- Hyperparameter tuning
- Model evaluation
- Best model saving
- Local inference
- Prediction logging
- Experiment logging
- FastAPI REST API for serving predictions

---

## Project Structure

```text
Uniblox-assignment/
│
├── employee_data.csv
│
├── Debosmit Submission/
│   ├── train.py
│   ├── inference.py
│   ├── api.py
│   └── requirements.txt
│
├── Debosmit models/
│   └── best_model.joblib
│
└── Debosmit reports/
    ├── experiments.csv
    ├── metrics.json
    ├── predictions.csv
    └── report.pdf
```

### File Description

| File | Description |
|---|---|
| `employee_data.csv` | Input employee dataset |
| `train.py` | Trains, tunes, and evaluates the ML model |
| `inference.py` | Runs predictions for individual employees |
| `api.py` | FastAPI REST API for model predictions |
| `requirements.txt` | Python dependencies |
| `best_model.joblib` | Saved best trained model |
| `metrics.json` | Model evaluation metrics |
| `experiments.csv` | Experiment results |
| `predictions.csv` | Logged inference results |
| `report.pdf` | Detailed project report |

---

# Requirements

- Python 3.14.7
- pip

The project uses the following Python libraries:

- NumPy
- pandas
- scikit-learn
- joblib
- FastAPI
- Uvicorn
- Pydantic

The exact package versions are available in:

```text
Debosmit Submission/requirements.txt
```

---

# Setup

## 1. Clone the repository

```powershell
git clone https://github.com/Debosmit-Neogi/Uniblox-assignment.git
cd Uniblox-assignment
```
---

## 2. Install dependencies

From the project root directory:

```powershell
python -m pip install -r "Debosmit Submission\requirements.txt"
```

---

# Dataset

The input dataset is:

```text
employee_data.csv
```

The dataset contains approximately 10,000 synthetic employee records.

The columns are:

```text
employee_id
age
gender
marital_status
salary
employment_type
region
has_dependents
tenure_years
enrolled
```

The target variable is:

```text
enrolled
```

where:

```text
1 = enrolled
0 = not enrolled
```

The `employee_id` column is excluded from model training because it is an identifier rather than a meaningful predictive feature.

---

# 1. Train the Model

The training script is:

```text
Debosmit Submission/train.py
```

From the project root, run:

```powershell
cd "Debosmit Submission"
python train.py
```

The training script performs the following steps:

1. Loads `employee_data.csv`
2. Checks the dataset
3. Separates features and target
4. Removes `employee_id`
5. Identifies numerical and categorical features
6. Handles missing values
7. Encodes categorical features
8. Splits the dataset into training and testing sets
9. Trains a Random Forest classifier
10. Performs hyperparameter tuning
11. Evaluates the model
12. Saves the best model
13. Saves evaluation metrics
14. Saves experiment results

---

# Training/Test Split

The dataset is split into:

```text
Training samples: 8000
Testing samples: 2000
```

A stratified train/test split is used to preserve the target class distribution.

A fixed random state is used for reproducibility.

---

# Model

The main model used is:

```text
Random Forest Classifier
```

Random Forest was selected because:

- The dataset is tabular.
- It contains both numerical and categorical features.
- It can model non-linear relationships.
- It does not require feature scaling.
- It provides probability estimates.
- It is relatively robust for this type of classification problem.

---

# Data Preprocessing

### Numerical Features

```text
age
salary
tenure_years
```

Numerical missing values are handled using median imputation.

### Categorical Features

```text
gender
marital_status
employment_type
region
has_dependents
```

Categorical missing values are handled using the most frequent value.

Categorical features are converted into numerical features using one-hot encoding.

The preprocessing and model are kept together in a scikit-learn pipeline so that the same preprocessing is automatically applied during inference.

---

# Hyperparameter Tuning

Hyperparameter tuning is performed using:

```text
RandomizedSearchCV
```

The search explores parameters such as:

- Number of trees
- Maximum tree depth
- Minimum samples required to split a node
- Minimum samples required at a leaf
- Maximum features considered at each split

ROC-AUC is used as the optimization metric.

Three-fold cross-validation is used during the hyperparameter search.

---

# Saved Model

After training, the best model is saved to:

```text
Debosmit models/best_model.joblib
```

The saved model contains the preprocessing pipeline and trained classifier.

This allows the model to be loaded later without retraining.

---

# Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

The current evaluation produced:

```text
Accuracy : 1.0000
Precision: 1.0000
Recall   : 1.0000
F1 Score : 1.0000
ROC-AUC  : 1.0000
```

Confusion matrix:

```text
[[765    0]
 [   0 1235]]
```

Classification report:

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       765
           1       1.00      1.00      1.00      1235

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000
```

The dataset is synthetic, so the perfect performance should not be interpreted as expected performance on real-world employee data.

Detailed analysis is available in:

```text
Debosmit reports/report.pdf
```

---

# Evaluation Output Files

The detailed evaluation metrics are saved to:

```text
Debosmit reports/metrics.json
```

Experiment results are saved to:

```text
Debosmit reports/experiments.csv
```

---

# 2. Run Local Inference

The inference script is:

```text
Debosmit Submission/inference.py
```

After training the model, run:

```powershell
python inference.py
```

The script accepts employee information and returns:

- Enrollment probability
- Predicted enrollment

Example input:

```text
Age: 35
Gender: Male
Marital status: Married
Salary: 75000
Employment type: Full-time
Region: West
Has dependents: Yes
Tenure years: 8
```

Example output:

```text
Enrollment probability: 97.32%
Predicted enrollment: Yes
```

The exact probability depends on the trained model.

---

# Prediction Logging

Inference results are logged to:

```text
Debosmit reports/predictions.csv
```

This provides a record of predictions generated using the trained model.

---

# 3. Run the REST API

The REST API is implemented using FastAPI.

The API code is:

```text
Debosmit Submission/api.py
```

The API loads the saved model:

```text
Debosmit models/best_model.joblib
```

Make sure the model has been trained before starting the API.

From the `Debosmit Submission` directory, run:

```powershell
python -m uvicorn api:app --reload
```

The API will start at:

```text
http://127.0.0.1:8000
```

Expected server output:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

# API Health Check

Open the following URL in a browser:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
    "message": "Employee Enrollment Prediction API",
    "status": "running"
}
```

This confirms that the API is running.

---

# API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

The documentation provides an interface to test the prediction endpoint.

---

# Prediction Endpoint

The prediction endpoint is:

```text
POST /predict
```

In the Swagger UI:

1. Open `POST /predict`
2. Click **Try it out**
3. Enter the employee information
4. Click **Execute**

Example request:

```json
{
    "age": 35,
    "gender": "Male",
    "marital_status": "Married",
    "salary": 75000,
    "employment_type": "Full-time",
    "region": "West",
    "has_dependents": "Yes",
    "tenure_years": 8
}
```

Example response:

```json
{
    "enrollment_probability": 0.9732,
    "predicted_enrollment": 1,
    "enrolled": "Yes"
}
```

The probability represents the model's estimated likelihood of enrollment.

---

# Complete Run Instructions

For a fresh setup, follow these steps in order.

## Step 1: Clone the repository

```powershell
git clone https://github.com/Debosmit-Neogi/Uniblox-assignment.git
cd Uniblox-assignment
```

## Step 2: Install dependencies

```powershell
python -m pip install -r "Debosmit Submission\requirements.txt"
```

## Step 3: Train the model

```powershell
cd "Debosmit Submission"
python train.py
```

This generates/updates:

```text
../Debosmit models/best_model.joblib
../Debosmit reports/metrics.json
../Debosmit reports/experiments.csv
```

## Step 4: Run local inference

```powershell
python inference.py
```

Predictions are logged to:

```text
../Debosmit reports/predictions.csv
```

## Step 5: Start the REST API

```powershell
python -m uvicorn api:app --reload
```

## Step 6: Test the API

Open:

```text
http://127.0.0.1:8000/docs
```

Then use:

```text
POST /predict
```

to test an employee prediction.

---

# Complete Pipeline

The complete workflow is:

```text
employee_data.csv
       |
       v
Data Loading
       |
       v
Data Preprocessing
       |
       v
Train/Test Split
       |
       v
Random Forest
       |
       v
Hyperparameter Tuning
       |
       v
Model Evaluation
       |
       v
best_model.joblib
       |
       +-----------------------+
       |                       |
       v                       v
inference.py               FastAPI
       |                       |
       v                       v
predictions.csv           POST /predict
```

---

# Output Files

After running the training and inference scripts:

```text
Debosmit models/
└── best_model.joblib
```

```text
Debosmit reports/
├── experiments.csv
├── metrics.json
├── predictions.csv
└── report.pdf
```

---

# Experiment Tracking

A lightweight experiment tracking approach is included.

Model experiment results are stored in:

```text
Debosmit reports/experiments.csv
```

Detailed evaluation metrics are stored in:

```text
Debosmit reports/metrics.json
```

This keeps track of the model experiments without requiring an external experiment tracking platform.

---

# Key Takeaways

- The project implements a complete machine learning pipeline.
- Both numerical and categorical features are processed.
- `employee_id` is excluded from model training.
- Random Forest is used as the primary classifier.
- Hyperparameter tuning is performed using `RandomizedSearchCV`.
- The best model is saved using `joblib`.
- A separate inference script is provided.
- Prediction results are logged.
- A FastAPI REST API is provided for serving predictions.
- Interactive API documentation is available through `/docs`.

The supplied synthetic dataset produced perfect performance on the held-out test set. Independent validation would be required before using the model for real-world insurance enrollment decisions.

---

# Detailed Report

The detailed project report covering:

- Data observations
- Data processing
- Model selection
- Hyperparameter tuning
- Evaluation
- Results
- Limitations
- Future improvements

is available at:

```text
Debosmit reports/report.pdf
```