# 🏦 Loan Approval Predictor — Streamlit Deployment

Production-ready Streamlit web app for the `loan_approval_project`.
Reuses the existing `src/preprocessing.py` and `src/model.py` modules
exactly as used in the notebook.

---

## Project Structure

```
loan-prediction-app/
│
├── app.py                      ← Streamlit application  (main file)
├── train_model.py              ← one-time training script
├── loan_model.pkl              ← generated artifact (commit to git)
├── requirements.txt            ← Streamlit deployment dependencies
│
├── data/
│   └── loan-train.csv          ← 614-row training dataset
│
└── src/                        ← original project modules (unchanged)
    ├── __init__.py
    ├── preprocessing.py        ← preprocess(), FEATURES, TARGET
    ├── model.py                ← build_model(), evaluate(), cross_validate()
    └── visualize.py
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model *(generates `loan_model.pkl`)*
```bash
python train_model.py
```

Expected output:
```
5-Fold Stratified CV:
  accuracy    : 0.7932 ± 0.0131
  roc_auc     : 0.7642 ± 0.0290
  f1          : 0.8610 ± 0.0050

Hold-out Accuracy : 0.8537
Hold-out ROC-AUC  : 0.8328
```

### 3. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501**

---

## How It Works

### Training (`train_model.py`)
1. Calls `src/preprocessing.load_data()` and `preprocess()` — same as the notebook
2. Calls `src/model.build_model()` → `cross_validate()` → `evaluate()`
3. Retrains on **100% of data** for deployment (notebook only used 80%)
4. Saves a full artifact dict to `loan_model.pkl`:

```python
{
    "model":         final_model,      # RandomForestClassifier (300 trees)
    "features":      FEATURES,         # column order for predict()
    "enc_map":       enc_map,          # {col: {label→int}} per column
    "impute":        {...},            # fallback values for missing inputs
    "cv_accuracy":   0.7932,
    "cv_auc":        0.7642,
    "cv_f1":         0.8610,
    "feat_importance": fi_df,
}
```

### Inference (`app.py` → `build_row()`)
Replicates `preprocessing.preprocess()` for a single new applicant:
1. Compute derived features: `TotalIncomeLog`, `LoanAmountLog`, `EMI`, `BalanceIncome`
2. Encode categoricals using saved per-column label maps
3. Return a one-row DataFrame in the exact training column order
4. Call `rf.predict()` and `rf.predict_proba()`

---

## App Features

| Feature | Description |
|---|---|
| **Live KPI strip** | Total income, EMI, DTI, balance after EMI — updates as you type |
| **Prediction card** | Approved / Rejected + probability bars |
| **Engineered features panel** | Shows exact `TotalIncomeLog`, `EMI`, `BalanceIncome` values |
| **Feature importance chart** | Plotly — engineered features highlighted in amber |
| **Dataset Explorer tab** | Data preview, missing values, class distribution chart |
| **Model Pipeline tab** | Hyperparameters, engineering code, encoding map, CV metrics |
| **Batch Predict tab** | Upload CSV → predict all rows → download results |

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | `RandomForestClassifier` |
| `n_estimators` | 300 |
| `max_depth` | 8 |
| `max_features` | `'sqrt'` |
| `class_weight` | `'balanced'` |
| CV Accuracy | 0.7932 ± 0.0131 |
| CV ROC-AUC | 0.7642 ± 0.0290 |
| CV F1 | 0.8610 ± 0.0050 |

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this folder to a **public GitHub repository**
   - Include `loan_model.pkl` — the app cannot start without it
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, branch, set **Main file path** → `app.py`
4. Click **Deploy**

## Deploy to Render

```
Build Command : pip install -r requirements.txt
Start Command : streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## Recommended `.gitignore`

```
__pycache__/
*.pyc
.env
.DS_Store
notebooks/.ipynb_checkpoints/
```

> ✅ **Do commit** `loan_model.pkl` — it is required at runtime.
