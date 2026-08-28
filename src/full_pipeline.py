from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
IMG = ROOT / "images"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

xlsx_candidates = list(RAW_DIR.glob("*.xlsx"))
if not xlsx_candidates:
    raise FileNotFoundError(
        "Online Retail II .xlsx not found. Run: python src/download_data.py"
    )
xlsx = xlsx_candidates[0]

sheets = pd.read_excel(xlsx, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)

# Normalize historical UCI column names.
rename = {
    "Invoice": "InvoiceNo",
    "Customer ID": "CustomerID",
    "Price": "UnitPrice",
}
df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

# Positive-purchase grain for customer behavior.
clean = df[
    df["CustomerID"].notna()
    & ~df["InvoiceNo"].str.upper().str.startswith("C")
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
    & df["InvoiceDate"].notna()
].copy()
clean["CustomerID"] = clean["CustomerID"].astype(int)
clean["Revenue"] = clean["Quantity"] * clean["UnitPrice"]

# Full-period RFM for segmentation.
snapshot_date = clean["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = clean.groupby("CustomerID").agg(
    recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    frequency=("InvoiceNo", "nunique"),
    monetary=("Revenue", "sum"),
    unique_purchase_days=("InvoiceDate", lambda x: x.dt.date.nunique()),
    average_unit_price=("UnitPrice", "mean"),
).reset_index()

rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5,4,3,2,1]).astype(int)
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
rfm["rfm_total_score"] = rfm[["r_score","f_score","m_score"]].sum(axis=1)

def segment(row):
    r,f,m = row.r_score,row.f_score,row.m_score
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if f >= 4 and r >= 3:
        return "Loyal Customers"
    if r >= 4 and f <= 2:
        return "Potential Loyalists"
    if r <= 2 and f >= 4 and m >= 4:
        return "Cannot Lose"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Hibernating"
    return "Needs Attention"

rfm["segment"] = rfm.apply(segment, axis=1)
rfm.to_csv(OUT / "rfm_from_raw.csv", index=False)

# Leakage-safe temporal churn target:
# features <= 2011-06-30, label from 2011-07-01 onward.
obs_end = pd.Timestamp("2011-06-30 23:59:59")
pred_start = pd.Timestamp("2011-07-01")
obs = clean[clean["InvoiceDate"] <= obs_end].copy()
future = clean[clean["InvoiceDate"] >= pred_start].copy()

obs_snapshot = obs_end + pd.Timedelta(days=1)
features = obs.groupby("CustomerID").agg(
    recency_days=("InvoiceDate", lambda x: (obs_snapshot - x.max()).days),
    frequency=("InvoiceNo", "nunique"),
    monetary=("Revenue", "sum"),
    unique_purchase_days=("InvoiceDate", lambda x: x.dt.date.nunique()),
    avg_order_value=("Revenue", "mean"),
    average_unit_price=("UnitPrice", "mean"),
    unique_products=("StockCode", "nunique"),
    country=("Country", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
).reset_index()

future_customers = set(future["CustomerID"].unique())
features["churned"] = (~features["CustomerID"].isin(future_customers)).astype(int)

# RFM scores based only on observation-window features.
features["r_score"] = pd.qcut(features["recency_days"].rank(method="first"), 5, labels=[5,4,3,2,1]).astype(int)
features["f_score"] = pd.qcut(features["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
features["m_score"] = pd.qcut(features["monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
features["rfm_total_score"] = features[["r_score","f_score","m_score"]].sum(axis=1)

X = features.drop(columns=["CustomerID","churned"])
y = features["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

numeric = [c for c in X.columns if c != "country"]
categorical = ["country"]

prep = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=3
    ),
}

rows = []
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:,1]
    pred = (prob >= 0.5).astype(int)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test,pred),
        "precision": precision_score(y_test,pred),
        "recall": recall_score(y_test,pred),
        "f1": f1_score(y_test,pred),
        "roc_auc": roc_auc_score(y_test,prob),
        "pr_auc": average_precision_score(y_test,prob),
    })
    fitted[name] = pipe

# CatBoost handles country as a categorical feature directly.
X_cb = X.copy()
X_cb["country"] = X_cb["country"].fillna("Unknown").astype(str)
Xcb_train, Xcb_test, ycb_train, ycb_test = train_test_split(
    X_cb, y, test_size=0.20, stratify=y, random_state=42
)
cb = CatBoostClassifier(
    iterations=400, depth=6, learning_rate=0.05,
    loss_function="Logloss", verbose=False, random_seed=42,
    auto_class_weights="Balanced"
)
cb.fit(Xcb_train, ycb_train, cat_features=["country"])
prob = cb.predict_proba(Xcb_test)[:,1]
pred = (prob >= 0.5).astype(int)
rows.append({
    "model": "CatBoost",
    "accuracy": accuracy_score(ycb_test,pred),
    "precision": precision_score(ycb_test,pred),
    "recall": recall_score(ycb_test,pred),
    "f1": f1_score(ycb_test,pred),
    "roc_auc": roc_auc_score(ycb_test,prob),
    "pr_auc": average_precision_score(ycb_test,prob),
})
metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
metrics.to_csv(OUT / "model_metrics_from_raw.csv", index=False)

# SHAP for CatBoost.
sample = Xcb_test.sample(min(1000, len(Xcb_test)), random_state=42)
explainer = shap.TreeExplainer(cb)
sv = explainer.shap_values(sample)
shap.summary_plot(sv, sample, show=False)
plt.tight_layout()
plt.savefig(IMG / "shap_summary_from_raw.png", dpi=160, bbox_inches="tight")
plt.close()

print(metrics.to_string(index=False))
