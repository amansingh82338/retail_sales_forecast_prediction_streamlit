"""
app.py
======
Home page for the Predictive Sales Forecasting Streamlit app.

This app uses Streamlit's native multi-page structure. Navigation between
pages appears automatically in the sidebar (populated from the pages/
folder) -- this file only renders the landing/overview screen.

Pages:
    pages/1_Data_Explorer.py     - browse and filter the merged dataset
    pages/2_Sales_Trends.py      - visualize trends, seasonality, holiday impact
    pages/3_Model_Comparison.py  - compare the 4 trained models
    pages/4_Predict.py           - generate a live weekly sales prediction

Run with:
    streamlit run app.py
"""

import os

import streamlit as st

from utils import (
    DATA_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET_COL,
    load_all_models,
    load_data,
)

st.set_page_config(
    page_title="Predictive Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------------------------------
# HERO
# --------------------------------------------------------------------------
st.title("📈 Predictive Sales Forecasting")
st.markdown(
    """
    An end-to-end sales forecasting application that predicts future
    **weekly retail sales** by analyzing historical sales data, store
    information, and external factors such as holidays, temperature,
    fuel prices, CPI, and unemployment.

    👈 Use the sidebar to navigate between pages.
    """
)

st.divider()

# --------------------------------------------------------------------------
# STATUS CHECKS
# --------------------------------------------------------------------------
st.subheader("Project Status")

data_ok = os.path.exists(DATA_PATH)
models_ok = os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))
reports_ok = os.path.exists(os.path.join(REPORTS_DIR, "model_comparison.csv"))

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Dataset", "✅ Found" if data_ok else "❌ Missing")
status_col2.metric("Trained Models", "✅ Found" if models_ok else "❌ Missing")
status_col3.metric("Reports", "✅ Found" if reports_ok else "❌ Missing")

if not data_ok:
    st.error(
        f"Dataset not found at:\n\n`{DATA_PATH}`\n\n"
        "Update `DATA_PATH` in `utils.py` or make sure the merged dataset exists."
    )

if not models_ok or not reports_ok:
    st.warning(
        "Trained models and/or reports are missing. Run **train_model.py** first:\n\n"
        "```bash\npython train_model.py\n```"
    )

st.divider()

# --------------------------------------------------------------------------
# QUICK SNAPSHOT (only if data is available)
# --------------------------------------------------------------------------
if data_ok:
    df = load_data(DATA_PATH)
    _, best_model, _ = load_all_models()

    st.subheader("Dataset Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Stores", f"{df['Store'].nunique():,}" if "Store" in df.columns else "—")
    c3.metric("Departments", f"{df['Dept'].nunique():,}" if "Dept" in df.columns else "—")
    if TARGET_COL in df.columns:
        c4.metric("Avg Weekly Sales", f"${df[TARGET_COL].mean():,.2f}")
    else:
        c4.metric("Avg Weekly Sales", "—")

    if best_model is not None:
        st.info(f"Current best model loaded: **{type(best_model).__name__}**")

st.divider()

# --------------------------------------------------------------------------
# PAGE GUIDE
# --------------------------------------------------------------------------
st.subheader("What's on each page")

g1, g2 = st.columns(2)
with g1:
    st.markdown(
        """
        **🔍 Data Explorer**
        Filter by store, department, and holiday weeks; view summary
        statistics and download the filtered dataset.

        **📊 Sales Trends**
        Trend lines, seasonality by month/quarter/week, holiday impact,
        top stores/departments, and economic-factor correlations.
        """
    )
with g2:
    st.markdown(
        """
        **🏆 Model Comparison**
        Side-by-side MAE / RMSE / R² for all four models, feature
        importance, and the actual-vs-predicted plot for the best model.

        **🔮 Predict**
        Enter store and week details to get a live weekly sales
        prediction, compared across all four trained models.
        """
    )

st.divider()
st.caption(
    "Built with Python, Pandas, Scikit-learn, XGBoost, and Streamlit. "
    "See README.md for the full project workflow."
)

st.markdown(
    "<div style='text-align: center; padding-top: 10px;'>"
    "<b>Made by Aman Singh Chauhan</b><br>"
    "<span style='font-size: 0.85em; color: gray;'>"
    "© 2026 Aman Singh Chauhan. All rights reserved. Do not copy or redistribute without permission."
    "</span></div>",
    unsafe_allow_html=True,
)