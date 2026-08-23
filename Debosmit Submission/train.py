## training script for the model
import pandas as pd

from sklearn.model_selection import train_test_split
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


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

DATA_PATH = r"C:\Users\debos\Downloads\Code\Uniblox-assignment\employee_data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ---------------------------------------------------------
# 2. Basic data checks
# ---------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["enrolled"].value_counts())
print(df["enrolled"].value_counts(normalize=True))


# ---------------------------------------------------------
# 3. Features and target
# ---------------------------------------------------------

# employee_id is just an identifier, so don't use it
# as a predictive feature.
X = df.drop(columns=["enrolled", "employee_id"])

y = df["enrolled"]


# ---------------------------------------------------------
# 4. Identify numerical and categorical columns
# ---------------------------------------------------------

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

print("\nNumerical columns:")
print(list(numerical_columns))

print("\nCategorical columns:")
print(list(categorical_columns))


# ---------------------------------------------------------
# 5. Preprocessing
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
        ),
    ]
)

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


# ---------------------------------------------------------
# 6. Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# 7. Random Forest model
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------
# 8. Complete ML pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ]
)


# ---------------------------------------------------------
# 9. Train
# ---------------------------------------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training complete.")


# ---------------------------------------------------------
# 10. Predictions
# ---------------------------------------------------------

y_pred = pipeline.predict(X_test)

# Probability of class 1
y_prob = pipeline.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# 11. Evaluation
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    y_prob,
)


print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ---------------------------------------------------------
# 12. Classification report
# ---------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# 13. Confusion matrix
# ---------------------------------------------------------

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))