"""
utils.py
========
Shared configuration, data/model loading, and preprocessing helpers used
across every page of the Predictive Sales Forecasting Streamlit app.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------
# CONFIGURATION -- update these paths if your folder structure changes
# --------------------------------------------------------------------------
DATA_PATH = "data/merged/merged_sales.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

TARGET_COL = "Weekly_Sales"

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


# --------------------------------------------------------------------------
# DATA / MODEL LOADING (cached)
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_data
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parses an uploaded CSV file (as bytes, for cache-hashing) into a DataFrame."""
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def get_dataset() -> pd.DataFrame:
    """Returns the active dataset for the app.

    If the user has uploaded a CSV on the home page (stored in
    st.session_state['uploaded_df']), that is used everywhere. Otherwise
    falls back to the default DATA_PATH on disk.
    """
    if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
        return st.session_state["uploaded_df"]
    return load_data(DATA_PATH)


def using_uploaded_data() -> bool:
    return st.session_state.get("uploaded_df") is not None


@st.cache_resource
def load_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_csv_report(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_all_models():
    """Returns (dict of individual models, best_model, scaler)."""
    models = {
        name: load_model(os.path.join(MODELS_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    best_model = load_model(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = load_model(os.path.join(MODELS_DIR, "scaler.pkl"))
    return models, best_model, scaler


# --------------------------------------------------------------------------
# PREPROCESSING (mirrors train_model.py)
# --------------------------------------------------------------------------
def preprocess_for_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the preprocessing done in train_model.py so we can recover
    the exact feature-column layout the models were trained on."""
    df = df.copy()
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    for col in markdown_cols:
        df[col] = df[col].fillna(0)

    for col in ["IsHoliday", "IsHoliday_feature"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0}
            ).fillna(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    numeric_fill_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]
    for col in numeric_fill_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])

    categorical_cols = [
        c for c in ["Type", "Store_Type", "Store_Size_Category"] if c in df.columns
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.fillna(0)
    return df


def predict_with_model(model, scaler, X: pd.DataFrame) -> np.ndarray:
    """Applies scaling only for models trained on scaled data
    (Linear Regression), matching the logic in train_model.py."""
    if isinstance(model, LinearRegression) and scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X
    return model.predict(X_input)


def check_data_path(path: str = DATA_PATH) -> bool:
    """Shows a Streamlit error and stops execution if no dataset is available
    -- either uploaded via the UI or found on disk at DATA_PATH."""
    if using_uploaded_data():
        return True
    if not os.path.exists(path):
        st.error(
            f"Dataset not found at:\n{path}\n\n"
            "Please check DATA_PATH in utils.py, or upload a CSV on the home page."
        )
        st.stop()
    return True
