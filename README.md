# Telco Customer Churn Prediction

> Portfolio note: This repository contains the project code and documentation. The raw Kaggle dataset and generated CSV outputs are intentionally not redistributed.

## Overview

This project builds a supervised machine learning pipeline to predict customer churn from the Telco Customer Churn dataset. The model classifies whether a customer is likely to churn based on demographic, account, and service-related features.

## What It Demonstrates

- Data loading and validation for a real-world CSV dataset
- Cleaning semi-numeric fields such as `TotalCharges`
- Missing-value handling for numeric and categorical columns
- One-hot encoding for categorical features
- Standardization for numeric features
- Stratified train/test splitting to preserve churn balance
- Leakage-aware preprocessing fitted only on training data
- Logistic Regression hyperparameter tuning with `GridSearchCV`
- Evaluation with Accuracy, Precision, Recall, F1 score, and a classification report
- Local export of the fully preprocessed dataset for review

## Project Structure

```text
.
|-- data/
|   |-- demo_telco_churn.csv
|   `-- README.md
|-- outputs/
|   `-- README.md
|-- src/
|   `-- main.py
|-- tests/
|   `-- test_pipeline.py
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Dataset

The source dataset is [Telco Customer Churn by BlastChar on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Kaggle describes the dataset as customer churn data for customer retention modeling and lists the data license as "Data files copyright Original Authors."

Because the dataset license is not a standard permissive software license, the raw Kaggle CSV is not included in this repository. Download the dataset from Kaggle and place the file here when you want to reproduce the full dataset run:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

For quick portfolio review, the repository includes a bundled synthetic demo dataset at:

```text
data/demo_telco_churn.csv
```

## Requirements

- Python 3.10+
- pip
- pandas
- numpy
- scikit-learn 1.2+

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## How to Run

Quick portfolio run with the bundled demo dataset:

```bash
python src/main.py --demo-data
```

If you run `python src/main.py` without the Kaggle CSV present, the script automatically falls back to the demo dataset and prints a note so the project still works on a fresh machine.

Full dataset run with Kaggle data:

From the project root, confirm the Kaggle dataset file exists at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Then run:

```bash
python src/main.py
```

If your system uses `python3` instead of `python`:

```bash
python3 src/main.py
```

The script prints dataset source information, model tuning results, cross-validation accuracy, test-set evaluation metrics, target distributions, and the output file location.

Running the full Kaggle dataset script generates:

```text
outputs/preprocessed_telco_churn.csv
```

Running with the bundled demo dataset generates:

```text
outputs/preprocessed_telco_churn_demo.csv
```

Generated output files are intentionally kept out of version control.

## Test And Verify

Run the smoke tests from the repository root:

```bash
python -m unittest discover -s tests
```

Useful checks:

- Run `python src/main.py --demo-data` and confirm the full pipeline completes without extra setup.
- Download the Kaggle CSV, place it in `data/`, and run `python src/main.py` to reproduce the full project workflow.
- Confirm the exported CSV appears under `outputs/`.

## Verified Results

The project was verified locally with Python 3.14.3 using the Kaggle dataset. The tuned Logistic Regression model selected these parameters:

```text
{'C': 1.0, 'class_weight': 'balanced', 'solver': 'lbfgs'}
```

Holdout test-set metrics:

```text
Accuracy: 0.7381
Precision: 0.5043
Recall: 0.7834
F1 Score: 0.6136
```

The recall score indicates the model identifies a large share of customers who churn, while the lower precision shows that some customers predicted to churn are false positives. This tradeoff is reasonable for a churn-risk screening model where missed churners can be costly.

The bundled demo dataset exists only to make the portfolio project runnable without external downloads. Its metrics are not intended to replace or compare against the Kaggle results above.

## Modeling Workflow

The main application flow is implemented in `src/main.py`:

1. Load the Telco churn CSV.
2. Drop `customerID`, which is an identifier rather than a predictive feature.
3. Convert `TotalCharges` to numeric values and impute missing numeric values with the median.
4. Fill missing categorical values with the most frequent category.
5. Split the data into stratified training and test sets.
6. Fit preprocessing only on the training data to reduce leakage risk.
7. Tune Logistic Regression with a small grid over regularization strength and class weighting.
8. Evaluate the selected model on the holdout test set.
9. Export the combined preprocessed dataset to `outputs/preprocessed_telco_churn.csv`.

## Portfolio Notes

This project is suitable for a public portfolio repository because it does not redistribute the raw Kaggle dataset or generated data outputs. The repository also includes a small synthetic demo dataset so recruiters and reviewers can run the code immediately before deciding whether to download the full Kaggle data.

## Resume Bullet

Built an end-to-end customer churn prediction pipeline in Python using pandas and scikit-learn, including leakage-aware preprocessing, one-hot encoding, feature scaling, stratified train/test splitting, Logistic Regression, 5-fold cross-validation, and GridSearchCV hyperparameter tuning; achieved 0.78 recall for churn detection on the holdout test set.

## License

The project code is licensed under the MIT License. The dataset is provided separately by the original authors through Kaggle and is not covered by this repository's license.
