"""
train_model.py
================
Trains and evaluates multiple regression models on the merged retail sales
dataset, then exports:

    models/
        linear_regression.pkl
        decision_tree.pkl
        random_forest.pkl
        xgboost.pkl
        best_model.pkl
        scaler.pkl

    reports/
        model_comparison.csv
        metrics.csv
        feature_importance.csv
        actual_vs_predicted.png
        feature_importance.png
        model_comparison.png

Usage:
    python train_model.py
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIGURATION -- update these paths if your folder structure changes
# --------------------------------------------------------------------------
DATA_PATH = r"C:\Users\amans\Downloads\Codtech_Projects\predictive_sales_forecast\data\merged\merged_sales.csv"
MODELS_DIR = r"C:\Users\amans\Downloads\Codtech_Projects\predictive_sales_forecast\models"
REPORTS_DIR = r"C:\Users\amans\Downloads\Codtech_Projects\predictive_sales_forecast\reports"

TARGET_COL = "Weekly_Sales"
RANDOM_STATE = 42
TEST_SIZE = 0.2

sns.set_style("whitegrid")


# --------------------------------------------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    print(f"[1/7] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"      Loaded shape: {df.shape}")
    return df


# --------------------------------------------------------------------------
# STEP 2: CLEAN + PREPROCESS
# --------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    print("[2/7] Preprocessing data...")
    df = df.copy()

    # Drop raw Date column if present -- Year/Month/Quarter/Week/Day/DayOfWeek
    # already capture the temporal information as numeric features.
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # MarkDown columns often contain NaNs when no promotion was active.
    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    for col in markdown_cols:
        df[col] = df[col].fillna(0)

    # Boolean-like holiday flags -> int
    for col in ["IsHoliday", "IsHoliday_feature"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0}
            ).fillna(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Numeric columns with occasional missing values -> median fill
    numeric_fill_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]
    for col in numeric_fill_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Drop any rows still missing the target
    df = df.dropna(subset=[TARGET_COL])

    # One-hot encode remaining categorical columns
    categorical_cols = [
        c for c in ["Type", "Store_Type", "Store_Size_Category"] if c in df.columns
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Fill any remaining NaNs in feature columns with 0 as a safe default
    df = df.fillna(0)

    print(f"      Shape after preprocessing: {df.shape}")
    return df


# --------------------------------------------------------------------------
# STEP 3: SPLIT + SCALE
# --------------------------------------------------------------------------
def split_and_scale(df: pd.DataFrame):
    print("[3/7] Splitting and scaling data...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"      Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


# --------------------------------------------------------------------------
# STEP 4: TRAIN MODELS
# --------------------------------------------------------------------------
def train_models(X_train, X_train_scaled, y_train):
    print("[4/7] Training models...")

    models = {}

    print("      -> Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models["Linear Regression"] = {"model": lr, "uses_scaled": True}

    print("      -> Decision Tree")
    dt = DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=15)
    dt.fit(X_train, y_train)
    models["Decision Tree"] = {"model": dt, "uses_scaled": False}

    print("      -> Random Forest")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = {"model": rf, "uses_scaled": False}

    print("      -> XGBoost")
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = {"model": xgb, "uses_scaled": False}

    return models


# --------------------------------------------------------------------------
# STEP 5: EVALUATE MODELS
# --------------------------------------------------------------------------
def evaluate_models(models, X_test, X_test_scaled, y_test):
    print("[5/7] Evaluating models...")
    results = []
    predictions = {}

    for name, info in models.items():
        model = info["model"]
        X_eval = X_test_scaled if info["uses_scaled"] else X_test
        y_pred = model.predict(X_eval)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2,
        })
        print(f"      {name:20s} MAE={mae:,.2f}  RMSE={rmse:,.2f}  R2={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False).reset_index(drop=True)
    return results_df, predictions


# --------------------------------------------------------------------------
# STEP 6: SAVE MODELS, SCALER, AND REPORTS
# --------------------------------------------------------------------------
def save_models(models, scaler, results_df, models_dir):
    print("[6/7] Saving models and scaler...")
    os.makedirs(models_dir, exist_ok=True)

    name_to_file = {
        "Linear Regression": "linear_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
    }

    for name, info in models.items():
        filename = name_to_file[name]
        path = os.path.join(models_dir, filename)
        joblib.dump(info["model"], path)
        print(f"      Saved: {path}")

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"      Saved: {scaler_path}")

    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]["model"]
    best_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_path)
    print(f"      Best model: {best_name} -> Saved: {best_path}")

    return best_name


def save_reports(results_df, models, predictions, y_test, feature_names, best_name, reports_dir):
    print("[7/7] Saving reports and charts...")
    os.makedirs(reports_dir, exist_ok=True)

    # --- model_comparison.csv & metrics.csv (same content, two names as requested) ---
    comparison_path = os.path.join(reports_dir, "model_comparison.csv")
    metrics_path = os.path.join(reports_dir, "metrics.csv")
    results_df.to_csv(comparison_path, index=False)
    results_df.to_csv(metrics_path, index=False)
    print(f"      Saved: {comparison_path}")
    print(f"      Saved: {metrics_path}")

    # --- feature_importance.csv + .png (from tree-based models) ---
    importance_source = None
    if "Random Forest" in models:
        importance_source = ("Random Forest", models["Random Forest"]["model"])
    elif "XGBoost" in models:
        importance_source = ("XGBoost", models["XGBoost"]["model"])

    if importance_source is not None:
        src_name, src_model = importance_source
        importances = src_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        fi_csv_path = os.path.join(reports_dir, "feature_importance.csv")
        fi_df.to_csv(fi_csv_path, index=False)
        print(f"      Saved: {fi_csv_path}")

        top_n = fi_df.head(15)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_n, x="Importance", y="Feature", color="steelblue")
        plt.title(f"Top 15 Feature Importances ({src_name})")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        fi_png_path = os.path.join(reports_dir, "feature_importance.png")
        plt.savefig(fi_png_path, dpi=150)
        plt.close()
        print(f"      Saved: {fi_png_path}")

    # --- actual_vs_predicted.png (best model) ---
    y_pred_best = predictions[best_name]
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_best, alpha=0.3, s=10, color="teal")
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal fit")
    plt.xlabel("Actual Weekly Sales")
    plt.ylabel("Predicted Weekly Sales")
    plt.title(f"Actual vs Predicted Weekly Sales ({best_name})")
    plt.legend()
    plt.tight_layout()
    avp_path = os.path.join(reports_dir, "actual_vs_predicted.png")
    plt.savefig(avp_path, dpi=150)
    plt.close()
    print(f"      Saved: {avp_path}")

    # --- model_comparison.png ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = ["MAE", "RMSE", "R2_Score"]
    for ax, metric in zip(axes, metrics_to_plot):
        sns.barplot(data=results_df, x="Model", y=metric, ax=ax, palette="viridis")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("Model Comparison")
    plt.tight_layout()
    comp_png_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(comp_png_path, dpi=150)
    plt.close()
    print(f"      Saved: {comp_png_path}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    df = preprocess(df)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = (
        split_and_scale(df)
    )
    models = train_models(X_train, X_train_scaled, y_train)
    results_df, predictions = evaluate_models(models, X_test, X_test_scaled, y_test)

    best_name = save_models(models, scaler, results_df, MODELS_DIR)
    save_reports(results_df, models, predictions, y_test, feature_names, best_name, REPORTS_DIR)

    print("\n Done! Summary of results:")
    print(results_df.to_string(index=False))
    print(f"\nBest model selected: {best_name}")


if __name__ == "__main__":
    main()
