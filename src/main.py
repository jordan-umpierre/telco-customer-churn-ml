from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass(frozen=True)
class PreprocessResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names_out: np.ndarray


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic validation
    required_cols = {"customerID", "Churn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> PreprocessResult:
    # Drop identifier (not predictive)
    df = df.copy()
    df = df.drop(columns=["customerID"])

    # Target
    y = df["Churn"].astype(str).str.strip()
    X = df.drop(columns=["Churn"])

    # Clean known "semi-numeric" column (TotalCharges sometimes has blanks)
    if "TotalCharges" in X.columns:
        # Convert blanks to NaN, then numeric
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce")

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Build preprocessing transformer. It is fitted only on training data below.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # Split before fitting imputers, encoders, and scalers to reduce leakage risk.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=y.to_numpy()  # churn class balance maintained
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names_out = preprocessor.get_feature_names_out()

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names_out=feature_names_out,
    )

def train_logistic_model(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train a baseline Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(x_train, y_train)
    return model

def evaluate_model(model: LogisticRegression,
                   x_test: np.ndarray,
                   y_test: np.ndarray) -> None:
    """
    Evaluate model performance using common classification metrics.
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="Yes")
    recall = recall_score(y_test, y_pred, pos_label="Yes")
    f1 = f1_score(y_test, y_pred, pos_label="Yes")

    print("\nModel Evaluation (Logistic Regression)")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

def cross_validate_model(model, X_train, y_train) -> None:
    """
    Apply stratified k-fold cross-validation to evaluate model robustness.
    """
    print("\nCross-Validation (5-Fold) Results:")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring="accuracy"
    )

    print("Fold Accuracies:", scores)
    print("Mean CV Accuracy:", round(scores.mean(), 4))

def tune_logistic_model(X_train, y_train):
    print("\nHyperparameter Tuning (GridSearchCV):")

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs"],
        "class_weight": [None, "balanced"]
    }

    scorer = make_scorer(f1_score, pos_label="Yes")

    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=5000),
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best CV Score (F1):", round(grid.best_score_, 4))

    return grid.best_estimator_

def export_preprocessed_dataset(result: PreprocessResult) -> None:
    """
    Export the fully preprocessed dataset (train + test combined)
    to a CSV file for repository upload.
    """
    X_combined = np.vstack([result.X_train, result.X_test])
    y_combined = np.concatenate([result.y_train, result.y_test])

    df_processed = pd.DataFrame(X_combined, columns=result.feature_names_out)
    df_processed["Churn"] = y_combined

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "preprocessed_telco_churn.csv")
    df_processed.to_csv(output_path, index=False)

    print("\nPreprocessed dataset exported to:", output_path)

def main() -> None:
    csv_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = load_dataset(csv_path)

    result = preprocess(df)
    model = tune_logistic_model(result.X_train, result.y_train)
    cross_validate_model(model, result.X_train, result.y_train)
    evaluate_model(model, result.X_test, result.y_test)
    export_preprocessed_dataset(result)

    # Quick sanity prints
    print("Loaded rows, cols:", df.shape)
    print("X_train shape:", result.X_train.shape)
    print("X_test shape:", result.X_test.shape)
    print("Target distribution (train):", pd.Series(result.y_train).value_counts().to_dict())
    print("Target distribution (test):", pd.Series(result.y_test).value_counts().to_dict())


if __name__ == "__main__":
    main()
