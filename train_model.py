"""
train_model.py
--------------
Trains the Random Forest using the project's own src/ modules
and saves the full artifact to loan_model.pkl.

Run once before launching the Streamlit app:
    python train_model.py
"""
import pickle, sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import load_data, preprocess, FEATURES, TARGET
from src.model import build_model, cross_validate, evaluate, feature_importance_df

print("=" * 55)
print("  LOAN APPROVAL — MODEL TRAINING")
print("=" * 55)

# ── 1. Load & preprocess ──────────────────────────────────
train_df, _ = load_data("data/loan-train.csv")
df          = preprocess(train_df, is_train=True)

X = df[FEATURES]
y = df[TARGET]
print(f"\nDataset : {X.shape[0]} rows × {X.shape[1]} features")
print(f"Class   : Approved={int(y.sum())}  Rejected={int((y==0).sum())}")

# ── 2. Cross-validate (primary evaluation) ────────────────
model    = build_model()
cv_stats = cross_validate(model, X, y, n_splits=5)

print("\n5-Fold Stratified CV:")
for metric, vals in cv_stats.items():
    print(f"  {metric:<12}: {vals['mean']:.4f} ± {vals['std']:.4f}")

# ── 3. Hold-out evaluation for detailed metrics ───────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
results = evaluate(model, X_val, y_val)
print(f"\nHold-out Accuracy : {results['accuracy']:.4f}")
print(f"Hold-out ROC-AUC  : {results['roc_auc']:.4f}")
print(f"Hold-out F1       : {results['f1']:.4f}")
print("\nClassification Report:\n", results["report"])

# ── 4. Final model on 100% of data for deployment ─────────
print("Retraining on 100% of data for deployment…")
final_model = build_model()
final_model.fit(X, y)

# ── 5. Build per-column encoding maps ─────────────────────
# preprocessing.py reuses one LabelEncoder in a loop (overwrites state).
# We reconstruct explicit label→int dicts from the cleaned training data
# so app.py can encode new single-row inputs at inference time.
cleaned = preprocess(train_df, is_train=True)
enc_map = {}
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
    le = LabelEncoder()
    le.fit(cleaned[col].astype(str))
    enc_map[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}

# ── 6. Save full artifact ─────────────────────────────────
fi_df = feature_importance_df(final_model, FEATURES)

artifact = {
    "model":        final_model,
    "features":     FEATURES,
    "enc_map":      enc_map,
    # Imputation fallbacks (from raw training data)
    "impute": {
        "Gender":           train_df["Gender"].mode()[0],
        "Married":          train_df["Married"].mode()[0],
        "Self_Employed":    train_df["Self_Employed"].mode()[0],
        "LoanAmount":       float(train_df["LoanAmount"].median()),
        "Loan_Amount_Term": float(train_df["Loan_Amount_Term"].mode()[0]),
        "Credit_History":   float(train_df["Credit_History"].mode()[0]),
    },
    # Performance metrics
    "cv_accuracy":     round(cv_stats["accuracy"]["mean"], 4),
    "cv_auc":          round(cv_stats["roc_auc"]["mean"],  4),
    "cv_f1":           round(cv_stats["f1"]["mean"],       4),
    "val_accuracy":    round(results["accuracy"],           4),
    "val_auc":         round(results["roc_auc"],            4),
    "feat_importance": fi_df,
}

with open("loan_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nSaved → loan_model.pkl")
print("=" * 55)
