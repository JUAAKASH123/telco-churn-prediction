# Telco Customer Churn Prediction

A complete end-to-end machine learning solution to predict whether a telecom customer is likely to churn (leave the company). This project emphasizes clean data handling, clear reasoning, and production readiness—not just raw accuracy.

---

## Table of contents

- [Project overview](#project-overview)
- [Key features](#key-features)
- [Dataset](#dataset)
- [Data cleaning & preprocessing](#data-cleaning--preprocessing)
- [Modeling](#modeling)
- [Evaluation & results](#evaluation--results)
- [Why no feature scaling?](#why-no-feature-scaling)
- [Project structure](#project-structure)
- [Setup & usage](#setup--usage)
- [API](#api)
- [Contributing](#contributing)
- [License & contact](#license--contact)

---

## Project overview

This repository uses the IBM Telco Customer Churn dataset to build a churn prediction pipeline. The pipeline includes data cleaning, preprocessing (categorical handling, missing value handling), model training and evaluation, and a deployed prediction API (FastAPI) for serving the final model.

Goal: identify customers at risk of churn using demographics, service usage, and billing information so business teams can act.

---

## Key features

- Thorough data cleaning, including detection and handling of hidden missing values
- Proper handling of categorical variables (encoding)
- Comparison of two strong tree-based models: Random Forest and XGBoost
- Addressed class imbalance in training
- Deployed prediction API with FastAPI
- Modular, well-documented code and reproducible training flow

---

## Dataset

- 7,043 customer records
- 21 original features (customerID is dropped during preprocessing)
- Target: `Churn` (Yes / No → converted to 1 / 0)

Important data note:
- `TotalCharges` contained 11 records that were empty strings (not NaN). These required explicit conversion and handling during preprocessing.

---

## Data cleaning & preprocessing

- Convert billing columns to numeric (handle empty string -> NaN -> impute or drop as appropriate).
- Encode categorical features (one-hot encoding / target encoding / ordinal where applicable).
- Handle class imbalance:
  - Random Forest: `class_weight='balanced'`
  - XGBoost: tune `scale_pos_weight`
- Split dataset with stratification on the target to keep class distribution consistent between train/validation.

---

## Modeling

Two tree-based models were compared:

- Random Forest (sklearn)
- XGBoost (xgboost)

Both are well-suited for this tabular dataset and can handle categorical features after encoding.

---

## Evaluation & results

Model comparison (final reported metrics):

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|---------:|---------:|-------:|---------:|
| Random Forest  | 0.787    | 0.637    | 0.458  | 0.533    |
| XGBoost        | 0.755    | 0.535    | 0.567  | 0.551    |

Notes:
- Random Forest had slightly higher accuracy.
- XGBoost achieved better recall and F1-score, which can be more important for identifying customers at risk of churn (fewer false negatives).
- Final model chosen: XGBoost for its better balance on recall / F1 for the business objective.

---

## Why no feature scaling?

Both Random Forest and XGBoost are tree-based algorithms. They split on thresholds in individual features rather than relying on distances or gradient descent that is sensitive to feature scales. Scaling does not change tree-based model decision boundaries and would add unnecessary preprocessing complexity.

---

## Project structure

A high-level view of the repository:

- README.md
- requirements.txt
- data/
  - raw/ (original dataset, not tracked in repo)
  - processed/ (preprocessed artifacts)
- src/
  - train.py         — training pipeline and model export
  - app.py           — FastAPI app to serve predictions
  - preprocessing.py — cleaning & encoding utilities
  - model.py         — model wrappers & inference helpers
  - utils.py         — helper functions
- notebooks/         — exploratory analysis (optional)
- models/            — serialized model artifacts (.pkl, .joblib, or XGBoost binary)

Adjust paths above if your code layout differs.

---

## Setup & usage

1. Clone the repository
```bash
git clone https://github.com/JUAAKASH123/telco-churn-prediction.git
cd telco-churn-prediction
```

2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Train the model
```bash
python src/train.py
# resulting model artifact should be saved to models/ (see train.py for exact path)
```

5. Start the API
```bash
uvicorn src.app:app --reload
```

---

## API

After running the server, default FastAPI docs are available at:
- OpenAPI UI: http://127.0.0.1:8000/docs
- Redoc: http://127.0.0.1:8000/redoc

Example curl request (adjust fields to match your API schema):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature_1": value1, "feature_2": "categoryA", ... }'
```

Check `src/app.py` for the exact input schema.

---

## Reproducibility tips

- Fix random seeds in training (numpy, sklearn, xgboost) for reproducible runs.
- Save preprocessing pipeline (encoders/scalers) together with the model so inference uses identical transformations.
- Use a requirements.txt that pins package versions (e.g., scikit-learn, xgboost, pandas).

---

## Contributing

Contributions are welcome. Suggested workflow:
- Fork the repo
- Create a feature branch
- Add tests / update notebooks / update docs
- Open a PR with a clear description of changes

Please follow code style and add brief tests for any new logic.

---

## License & contact

Specify your project's license (e.g., MIT) or replace with your preferred license.

Author / Maintainer: JUAAKASH123  
Repository: https://github.com/JUAAKASH123/telco-churn-prediction

If you want, I can:
- update README further with exact command examples from the code,
- add badges (build / license), or
- generate a minimal example request body for the API based on src/app.py.
