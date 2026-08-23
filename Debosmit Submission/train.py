import json
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
DATA_FILE = BASE_DIR / "employee_data.csv"

# Model directory
MODEL_DIR = BASE_DIR / "Debosmit models"

# Reports directory
REPORT_DIR = BASE_DIR / "Debosmit reports"

# Saved model
MODEL_PATH = MODEL_DIR / "best_model.joblib"

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("EMPLOYEE INSURANCE ENROLLMENT PREDICTION")
print("=" * 60)

print("\nLoading dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. DATA CHECKS
# ============================================================

print("\n" + "=" * 60)
print("DATA CHECKS")
print("=" * 60)

print("\nMissing values:")

missing_values = df.isnull().sum()

print(missing_values)


print("\nTarget distribution:")

target_distribution = df["enrolled"].value_counts()

print(target_distribution)


# ============================================================
# 3. FEATURES / TARGET
# ============================================================

# employee_id is an identifier and should not be used
# as a predictive feature.

X = df.drop(
    columns=[
        "enrolled",
        "employee_id",
    ]
)

y = df["enrolled"]


# ============================================================
# 4. COLUMN TYPES
# ============================================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\n" + "=" * 60)
print("FEATURES")
print("=" * 60)

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 5. PREPROCESSING
# ============================================================

# Numerical preprocessing:
# Missing numerical values are replaced with the median.

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]
)


# Categorical preprocessing:
# 1. Missing values are replaced with the most frequent value.
# 2. Categorical values are converted to numerical features
#    using one-hot encoding.

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


# Combine numerical and categorical preprocessing.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 7. BASE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)


# Combine preprocessing and model into one pipeline.

pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor,
        ),
        (
            "model",
            model,
        ),
    ]
)


# ============================================================
# 8. TRAIN BASE MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING BASE RANDOM FOREST")
print("=" * 60)

start_time = time.time()

pipeline.fit(
    X_train,
    y_train,
)

training_time = time.time() - start_time

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds"
)


# ============================================================
# 9. BASE MODEL EVALUATION
# ============================================================

y_pred_base = pipeline.predict(X_test)

y_prob_base = pipeline.predict_proba(
    X_test
)[:, 1]


base_metrics = {
    "accuracy": accuracy_score(
        y_test,
        y_pred_base,
    ),
    "precision": precision_score(
        y_test,
        y_pred_base,
        zero_division=0,
    ),
    "recall": recall_score(
        y_test,
        y_pred_base,
        zero_division=0,
    ),
    "f1": f1_score(
        y_test,
        y_pred_base,
        zero_division=0,
    ),
    "roc_auc": roc_auc_score(
        y_test,
        y_prob_base,
    ),
}


print("\n" + "=" * 60)
print("BASE MODEL EVALUATION")
print("=" * 60)

for metric, value in base_metrics.items():
    print(
        f"{metric.upper():10}: "
        f"{value:.4f}"
    )


# ============================================================
# 10. HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

print("\nStarting hyperparameter tuning...")


param_grid = {
    "model__n_estimators": [
        100,
        200,
        300,
        500,
    ],
    "model__max_depth": [
        None,
        5,
        10,
        20,
    ],
    "model__min_samples_split": [
        2,
        5,
        10,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        4,
    ],
    "model__max_features": [
        "sqrt",
        "log2",
    ],
}


search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)


start_time = time.time()

search.fit(
    X_train,
    y_train,
)

tuning_time = time.time() - start_time


print(
    f"\nHyperparameter tuning completed in "
    f"{tuning_time:.2f} seconds"
)


print("\nBest parameters:")

for parameter, value in search.best_params_.items():
    print(
        f"{parameter}: {value}"
    )


print(
    f"\nBest CV ROC-AUC: "
    f"{search.best_score_:.4f}"
)


# ============================================================
# 11. BEST MODEL
# ============================================================

best_model = search.best_estimator_


# ============================================================
# 12. FINAL MODEL EVALUATION
# ============================================================

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(
    X_test
)[:, 1]


metrics = {
    "accuracy": accuracy_score(
        y_test,
        y_pred,
    ),
    "precision": precision_score(
        y_test,
        y_pred,
        zero_division=0,
    ),
    "recall": recall_score(
        y_test,
        y_pred,
        zero_division=0,
    ),
    "f1": f1_score(
        y_test,
        y_pred,
        zero_division=0,
    ),
    "roc_auc": roc_auc_score(
        y_test,
        y_prob,
    ),
}


# ============================================================
# 13. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

for metric, value in metrics.items():
    print(
        f"{metric.upper():10}: "
        f"{value:.4f}"
    )


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)


print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
)

print(cm)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(
    best_model,
    MODEL_PATH,
)

print(
    f"\nModel saved to:\n"
    f"{MODEL_PATH}"
)


# ============================================================
# 15. SAVE METRICS
# ============================================================

results = {
    "dataset": {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
    },

    "features": {
        "numerical": numerical_columns,
        "categorical": categorical_columns,
    },

    "base_model": {
        "model": "RandomForestClassifier",
        "metrics": base_metrics,
        "training_time_seconds": round(
            training_time,
            4,
        ),
    },

    "tuned_model": {
        "model": "RandomForestClassifier",
        "metrics": metrics,
        "best_parameters": search.best_params_,
        "best_cv_roc_auc": float(
            search.best_score_
        ),
        "tuning_time_seconds": round(
            tuning_time,
            4,
        ),
    },

    "confusion_matrix": cm.tolist(),

    "classification_report": classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    ),
}


metrics_path = REPORT_DIR / "metrics.json"

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )


print(
    f"Metrics saved to:\n"
    f"{metrics_path}"
)


# ============================================================
# 16. SAVE EXPERIMENT LOG
# ============================================================

experiment = pd.DataFrame(
    [
        {
            "experiment": "Random Forest - Base",
            "accuracy": base_metrics["accuracy"],
            "precision": base_metrics["precision"],
            "recall": base_metrics["recall"],
            "f1": base_metrics["f1"],
            "roc_auc": base_metrics["roc_auc"],
            "training_time_seconds": training_time,
        },
        {
            "experiment": "Random Forest - Tuned",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "training_time_seconds": tuning_time,
        },
    ]
)


experiments_path = REPORT_DIR / "experiments.csv"

experiment.to_csv(
    experiments_path,
    index=False,
)


print(
    f"Experiment log saved to:\n"
    f"{experiments_path}"
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nBest model: RandomForestClassifier"
)

print(
    f"Test Accuracy : {metrics['accuracy']:.4f}"
)

print(
    f"Test Precision: {metrics['precision']:.4f}"
)

print(
    f"Test Recall   : {metrics['recall']:.4f}"
)

print(
    f"Test F1       : {metrics['f1']:.4f}"
)

print(
    f"Test ROC-AUC  : {metrics['roc_auc']:.4f}"
)

print(
    f"\nModel:"
    f"\n{MODEL_PATH}"
)

print(
    f"\nMetrics:"
    f"\n{metrics_path}"
)

print(
    f"\nExperiments:"
    f"\n{experiments_path}"
)