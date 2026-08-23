import json
import time
from pathlib import Path

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

DATA_PATH = r"C:\Users\debos\Downloads\Code\Uniblox-assignment\employee_data.csv"

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.joblib"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DATA CHECKS
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["enrolled"].value_counts())


# ============================================================
# 3. FEATURES / TARGET
# ============================================================

X = df.drop(
    columns=["enrolled", "employee_id"]
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

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. BASE MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ]
)


# ============================================================
# 8. TRAIN BASE MODEL
# ============================================================

print("\nTraining base Random Forest...")

start_time = time.time()

pipeline.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print(
    f"Training completed in "
    f"{training_time:.2f} seconds"
)


# ============================================================
# 9. BASE MODEL EVALUATION
# ============================================================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

base_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test,
        y_pred,
        zero_division=0
    ),
    "recall": recall_score(
        y_test,
        y_pred,
        zero_division=0
    ),
    "f1": f1_score(
        y_test,
        y_pred,
        zero_division=0
    ),
    "roc_auc": roc_auc_score(
        y_test,
        y_prob
    ),
}

print("\n" + "=" * 50)
print("BASE MODEL")
print("=" * 50)

for metric, value in base_metrics.items():
    print(f"{metric.upper():10}: {value:.4f}")


# ============================================================
# 10. HYPERPARAMETER TUNING
# ============================================================

print("\nStarting hyperparameter tuning...")

param_grid = {
    "model__n_estimators": [
        100,
        200,
        300,
        500
    ],
    "model__max_depth": [
        None,
        5,
        10,
        20
    ],
    "model__min_samples_split": [
        2,
        5,
        10
    ],
    "model__min_samples_leaf": [
        1,
        2,
        4
    ],
    "model__max_features": [
        "sqrt",
        "log2"
    ],
}


search = RandomizedSearchCV(
    pipeline,
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
    y_train
)

tuning_time = time.time() - start_time

print(
    f"\nHyperparameter tuning completed in "
    f"{tuning_time:.2f} seconds"
)

print("\nBest parameters:")
print(search.best_params_)

print(
    f"\nBest CV ROC-AUC: "
    f"{search.best_score_:.4f}"
)


# ============================================================
# 11. BEST MODEL
# ============================================================

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]


# ============================================================
# 12. FINAL METRICS
# ============================================================

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test,
        y_pred,
        zero_division=0
    ),
    "recall": recall_score(
        y_test,
        y_pred,
        zero_division=0
    ),
    "f1": f1_score(
        y_test,
        y_pred,
        zero_division=0
    ),
    "roc_auc": roc_auc_score(
        y_test,
        y_prob
    ),
}

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

for metric, value in metrics.items():
    print(f"{metric.upper():10}: {value:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    best_model,
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)


# ============================================================
# 14. SAVE METRICS
# ============================================================

results = {
    "dataset": {
        "rows": len(df),
        "columns": len(df.columns),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
    },

    "features": {
        "numerical": numerical_columns,
        "categorical": categorical_columns,
    },

    "base_model": {
        "model": "RandomForestClassifier",
        "metrics": base_metrics,
        "training_time_seconds": training_time,
    },

    "tuned_model": {
        "model": "RandomForestClassifier",
        "metrics": metrics,
        "best_parameters": search.best_params_,
        "best_cv_roc_auc": search.best_score_,
        "tuning_time_seconds": tuning_time,
    },
}


with open(
    REPORT_DIR / "metrics.json",
    "w"
) as f:
    json.dump(
        results,
        f,
        indent=4
    )

print(
    f"Metrics saved to: "
    f"{REPORT_DIR / 'metrics.json'}"
)


# ============================================================
# 15. SAVE EXPERIMENT LOG
# ============================================================

experiment = pd.DataFrame([
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
    }
])

experiment.to_csv(
    REPORT_DIR / "experiments.csv",
    index=False
)

print(
    f"Experiment log saved to: "
    f"{REPORT_DIR / 'experiments.csv'}"
)

print("\nTraining completed successfully.")