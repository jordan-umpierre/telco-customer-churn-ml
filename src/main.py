from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_KAGGLE_DATASET = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEMO_DATASET = DATA_DIR / "demo_telco_churn.csv"
DEFAULT_OUTPUT = OUTPUTS_DIR / "preprocessed_telco_churn.csv"
DEFAULT_DEMO_OUTPUT = OUTPUTS_DIR / "preprocessed_telco_churn_demo.csv"
REQUIRED_COLUMNS = {"customerID", "Churn"}


@dataclass(frozen=True)
class DatasetSource:
    path: Path
    label: str
    using_demo: bool
    fallback_reason: str | None = None


@dataclass(frozen=True)
class PreprocessResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names_out: np.ndarray


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class PipelineSummary:
    dataset_source: DatasetSource
    output_path: Path
    best_params: dict[str, Any]
    best_cv_f1: float
    cv_accuracy_scores: np.ndarray
    metrics: EvaluationMetrics
    input_shape: tuple[int, int]
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    train_distribution: dict[str, int]
    test_distribution: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Telco churn modeling pipeline on the Kaggle dataset or the bundled demo dataset."
    )
    parser.add_argument(
        "--dataset",
        help="Optional path to a CSV dataset. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--demo-data",
        action="store_true",
        help="Use the bundled synthetic demo dataset even if the Kaggle dataset is available.",
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV path. Relative paths are resolved from the repository root.",
    )
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_dataset_source(dataset_arg: str | None, use_demo_data: bool) -> DatasetSource:
    if dataset_arg:
        dataset_path = resolve_repo_path(dataset_arg)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

        return DatasetSource(
            path=dataset_path,
            label="custom dataset",
            using_demo=dataset_path == DEMO_DATASET,
        )

    if use_demo_data:
        return DatasetSource(
            path=DEMO_DATASET,
            label="bundled demo dataset",
            using_demo=True,
        )

    if DEFAULT_KAGGLE_DATASET.exists():
        return DatasetSource(
            path=DEFAULT_KAGGLE_DATASET,
            label="Kaggle dataset",
            using_demo=False,
        )

    return DatasetSource(
        path=DEMO_DATASET,
        label="bundled demo dataset",
        using_demo=True,
        fallback_reason=(
            "Kaggle dataset not found at "
            f"{DEFAULT_KAGGLE_DATASET}. Falling back to the bundled demo dataset so the project can still run."
        ),
    )


def resolve_output_path(output_arg: str | None, using_demo: bool) -> Path:
    if output_arg:
        return resolve_repo_path(output_arg)

    return DEFAULT_DEMO_OUTPUT if using_demo else DEFAULT_OUTPUT


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> PreprocessResult:
    df = df.copy()
    df = df.drop(columns=["customerID"])

    y = df["Churn"].astype(str).str.strip()
    X = df.drop(columns=["Churn"])

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"].astype(str).str.strip().replace("", np.nan),
            errors="coerce",
        )

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [column for column in X.columns if column not in numeric_cols]

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=y.to_numpy(),
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names_out=preprocessor.get_feature_names_out(),
    )


def get_cv_splits(labels: np.ndarray, requested_splits: int = 5) -> int:
    class_counts = Counter(labels.tolist())
    minority_class_count = min(class_counts.values())
    return max(2, min(requested_splits, minority_class_count))


def tune_logistic_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple[LogisticRegression, dict[str, Any], float]:
    cv_splits = get_cv_splits(y_train)
    print(f"\nHyperparameter Tuning (GridSearchCV, {cv_splits}-Fold):")

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs"],
        "class_weight": [None, "balanced"],
    }

    scorer = make_scorer(f1_score, pos_label="Yes", zero_division=0)

    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=5000),
        param_grid=param_grid,
        scoring=scorer,
        cv=cv_splits,
        n_jobs=1,
    )
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best CV Score (F1):", round(grid.best_score_, 4))

    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


def cross_validate_model(model: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    cv_splits = get_cv_splits(y_train)
    print(f"\nCross-Validation ({cv_splits}-Fold) Results:")

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring="accuracy",
    )

    print("Fold Accuracies:", scores)
    print("Mean CV Accuracy:", round(scores.mean(), 4))

    return scores


def evaluate_model(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationMetrics:
    y_pred = model.predict(X_test)

    metrics = EvaluationMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, pos_label="Yes", zero_division=0)),
        recall=float(recall_score(y_test, y_pred, pos_label="Yes", zero_division=0)),
        f1=float(f1_score(y_test, y_pred, pos_label="Yes", zero_division=0)),
    )

    print("\nModel Evaluation (Logistic Regression)")
    print("Accuracy:", round(metrics.accuracy, 4))
    print("Precision:", round(metrics.precision, 4))
    print("Recall:", round(metrics.recall, 4))
    print("F1 Score:", round(metrics.f1, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def export_preprocessed_dataset(result: PreprocessResult, output_path: Path) -> Path:
    X_combined = np.vstack([result.X_train, result.X_test])
    y_combined = np.concatenate([result.y_train, result.y_test])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed = pd.DataFrame(X_combined, columns=result.feature_names_out)
    df_processed["Churn"] = y_combined
    df_processed.to_csv(output_path, index=False)

    print("\nPreprocessed dataset exported to:", output_path)
    return output_path


def run_pipeline(dataset_source: DatasetSource, output_path: Path) -> PipelineSummary:
    df = load_dataset(dataset_source.path)
    result = preprocess(df)
    model, best_params, best_cv_f1 = tune_logistic_model(result.X_train, result.y_train)
    cv_accuracy_scores = cross_validate_model(model, result.X_train, result.y_train)
    metrics = evaluate_model(model, result.X_test, result.y_test)
    export_preprocessed_dataset(result, output_path)

    train_distribution = pd.Series(result.y_train).value_counts().to_dict()
    test_distribution = pd.Series(result.y_test).value_counts().to_dict()

    print("\nRun Summary")
    print("Dataset source:", dataset_source.label)
    print("Loaded rows, cols:", df.shape)
    print("X_train shape:", result.X_train.shape)
    print("X_test shape:", result.X_test.shape)
    print("Target distribution (train):", train_distribution)
    print("Target distribution (test):", test_distribution)

    return PipelineSummary(
        dataset_source=dataset_source,
        output_path=output_path,
        best_params=best_params,
        best_cv_f1=best_cv_f1,
        cv_accuracy_scores=cv_accuracy_scores,
        metrics=metrics,
        input_shape=df.shape,
        train_shape=result.X_train.shape,
        test_shape=result.X_test.shape,
        train_distribution=train_distribution,
        test_distribution=test_distribution,
    )


def main() -> None:
    args = parse_args()
    dataset_source = resolve_dataset_source(args.dataset, args.demo_data)
    output_path = resolve_output_path(args.output, dataset_source.using_demo)

    if dataset_source.fallback_reason:
        print(dataset_source.fallback_reason)

    summary = run_pipeline(dataset_source, output_path)

    if summary.dataset_source.using_demo:
        print(
            "\nDemo dataset note: these metrics come from a small synthetic sample that exists only to make the "
            "portfolio project runnable without a Kaggle download."
        )


if __name__ == "__main__":
    main()
