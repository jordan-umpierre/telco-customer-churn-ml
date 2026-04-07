# Telco Customer Churn Prediction

> Portfolio note: Before sharing this repository publicly, confirm you have the right to redistribute any included datasets and project materials.

## Overview

This project builds a supervised machine learning pipeline to predict customer churn from the Telco Customer Churn dataset. The model classifies whether a customer is likely to churn based on demographic, account, and service-related features.

The project demonstrates a practical end-to-end workflow:

- Load and validate the source dataset
- Clean semi-numeric fields such as `TotalCharges`
- Handle missing values
- Encode categorical features with one-hot encoding
- Scale numeric features with standardization
- Use a stratified train/test split to preserve churn balance
- Tune Logistic Regression hyperparameters with `GridSearchCV`
- Evaluate the final model with Accuracy, Precision, Recall, F1 score, and a classification report
- Export a fully preprocessed dataset for review

## Project Structure

```text
.
|-- data/
|   `-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- outputs/
|   `-- preprocessed_telco_churn.csv
|-- src/
|   `-- main.py
|-- README.md
`-- requirements.txt
```

## Requirements

- Python 3.10 or newer
- pip
- pandas
- numpy
- scikit-learn 1.2 or newer

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## How to Run

From the project root, confirm the dataset exists at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Then run:

```bash
python src/main.py
```

The script prints model tuning results, cross-validation accuracy, test-set evaluation metrics, target distributions, and output file location.

## Verified Results

The project was verified locally with Python 3.14.3. The tuned Logistic Regression model selected these parameters:

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

## Resume Bullet

Built an end-to-end customer churn prediction pipeline in Python using pandas and scikit-learn, including leakage-aware preprocessing, one-hot encoding, feature scaling, stratified train/test splitting, Logistic Regression, 5-fold cross-validation, and GridSearchCV hyperparameter tuning; achieved 0.78 recall for churn detection on the holdout test set.

## GitHub Readiness

This project is suitable for a portfolio repository after confirming the included dataset and materials can be redistributed.

Recommended cleanup before publishing:

- Do not commit local virtual environments, cache files, or editor workspace state.
- Add a short project description to the repository landing page.
