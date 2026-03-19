"""
preprocessing.py
----------------
Data cleaning and feature engineering for the Loan Approval dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(train_path: str, test_path: str = None):
    """Load train (and optionally test) CSV files."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path else None
    return train, test


def preprocess(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Drop Loan_ID
      2. Impute missing values
      3. Encode target (train only)
      4. Feature engineering
      5. Label-encode categoricals
    """
    df = df.copy()
    df = df.drop("Loan_ID", axis=1, errors="ignore")

    # --- Encode target ---
    if is_train and "Loan_Status" in df.columns:
        df["Loan_Status"] = (df["Loan_Status"] == "Y").astype(int)

    # --- Impute missing values ---
    for col in ["Gender", "Married", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["Dependents"] = (
        df["Dependents"]
        .fillna("0")
        .replace("3+", "3")
    )
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0).astype(int)

    df["LoanAmount"]       = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"]   = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    # --- Feature engineering ---
    df["TotalIncome"]    = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmountLog"]  = np.log1p(df["LoanAmount"])
    df["TotalIncomeLog"] = np.log1p(df["TotalIncome"])
    df["EMI"]            = df["LoanAmount"] / df["Loan_Amount_Term"]
    df["BalanceIncome"]  = df["TotalIncome"] - (df["EMI"] * 1000)

    # --- Label encode categoricals ---
    le = LabelEncoder()
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area",
    "TotalIncomeLog", "LoanAmountLog", "EMI", "BalanceIncome",
]
TARGET = "Loan_Status"
