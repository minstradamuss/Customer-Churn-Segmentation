# Customer Churn & Segmentation

Customer analytics and churn modeling project on the real public UCI Online Retail II transaction dataset.

## What is implemented

- transaction cleaning and customer-level feature engineering
- RFM scoring and customer segmentation
- temporal churn definition using observation and prediction windows
- Logistic Regression, Random Forest and CatBoost training pipeline
- ROC-AUC and PR-AUC / Average Precision evaluation
- SHAP explainability for CatBoost
- retention hypotheses based on model + RFM signals
- SQL feature and label queries
- already-executed Jupyter notebook with saved outputs

## Dataset

UCI Online Retail II contains 1,067,371 transaction rows from a UK-based non-store online retailer.

Official source:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

The raw Excel file is intentionally not committed. `src/download_data.py` retrieves the official UCI ZIP.

## RFM / churn design

RFM is calculated from transaction history.

For churn modeling:
- observation period: through June 2011
- prediction period: July–December 2011
- churn = customer bought in the observation window but did not return in the prediction window
- model features are calculated only from the observation window

This design avoids using future behavior in the model inputs.

## Models and validation

`src/full_pipeline.py` trains:
- Logistic Regression
- Random Forest
- CatBoost

For every model it calculates:
- accuracy
- precision / recall / F1
- ROC-AUC
- PR-AUC (Average Precision)

CatBoost SHAP values are then generated with `shap.TreeExplainer`.

## Repository structure

```text
customer_churn_segmentation/
├── notebooks/customer_churn_segmentation.ipynb
├── reports/customer_churn_segmentation.html
├── src/download_data.py
├── src/full_pipeline.py
├── sql/
├── data/processed/
├── images/
├── requirements.txt
└── README.md
```

## Reproduce from raw data

```bash
pip install -r requirements.txt
python src/download_data.py
python src/full_pipeline.py
```

## Data provenance

- UCI Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Public real-data RFM/churn/SHAP checkpoint used in the compact preview:
  https://github.com/PrajwalShekar22/customer-360-revenue-intelligence
- Independent Online Retail II CatBoost validation benchmark used as a sanity checkpoint:
  https://doi.org/10.1080/02331888.2026.2667471

The compact preview does not invent customer-level rows; it stores real-data-derived analytical checkpoints while the full code recomputes results from the official raw dataset.
