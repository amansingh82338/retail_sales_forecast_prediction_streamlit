"""
pages/4_Predict.py
====================
Page 4 of the Predictive Sales Forecasting dashboard.
Lets users enter store/week details and generate a live Weekly Sales
prediction using the best trained model (with all-model comparison).
"""

import pandas as pd
import streamlit as st

from utils import (
    DATA_PATH,
    TARGET_COL,
    check_data_path,
    get_dataset,
    load_all_models,
    predict_with_model,
    preprocess_for_training_columns,
    using_uploaded_data,
)

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")

st.title("🔮 Generate a Sales Prediction")
st.caption("Enter store/week details below to predict Weekly Sales.")

check_data_path(DATA_PATH)
df = get_dataset()
models, best_model, scaler = load_all_models()

if using_uploaded_data():
    st.info("Using your uploaded dataset for default values below. Upload or reset it from the home page.")

if best_model is None:
    st.warning(
        "No trained model found in the models folder. "
        "Run train_model.py first to train and export models."
    )
    st.stop()

# --------------------------------------------------------------------------
# INPUT FORM
# --------------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Store & Department**")
        store = st.number_input(
            "Store", min_value=1,
            value=int(df["Store"].median()) if "Store" in df.columns else 1,
        )
        dept = st.number_input(
            "Dept", min_value=1,
            value=int(df["Dept"].median()) if "Dept" in df.columns else 1,
        )
        size = st.number_input(
            "Store Size", min_value=0,
            value=int(df["Size"].median()) if "Size" in df.columns else 100000,
        )
        store_type = st.selectbox(
            "Store Type",
            sorted(df["Type"].dropna().unique()) if "Type" in df.columns else ["A"],
        )
        is_holiday = st.selectbox("Is Holiday Week?", ["No", "Yes"])

    with col2:
        st.markdown("**Economic Factors**")
        temperature = st.number_input(
            "Temperature", value=float(df["Temperature"].median()) if "Temperature" in df.columns else 60.0,
        )
        fuel_price = st.number_input(
            "Fuel Price", value=float(df["Fuel_Price"].median()) if "Fuel_Price" in df.columns else 3.0,
        )
        cpi = st.number_input(
            "CPI", value=float(df["CPI"].median()) if "CPI" in df.columns else 200.0,
        )
        unemployment = st.number_input(
            "Unemployment", value=float(df["Unemployment"].median()) if "Unemployment" in df.columns else 7.5,
        )

    with col3:
        st.markdown("**Date Details**")
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
        quarter = st.number_input("Quarter", min_value=1, max_value=4, value=2)
        week = st.number_input("Week", min_value=1, max_value=53, value=24)
        day = st.number_input("Day", min_value=1, max_value=31, value=15)
        day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=2)

    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    markdown_values = {}
    if markdown_cols:
        st.markdown("**MarkDown Promotions**")
        md_cols = st.columns(len(markdown_cols))
        for i, col_name in enumerate(markdown_cols):
            with md_cols[i]:
                markdown_values[col_name] = st.number_input(col_name, min_value=0.0, value=0.0)

    submitted = st.form_submit_button("Predict Weekly Sales", use_container_width=True)

# --------------------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------------------
if submitted:
    input_row = {
        "Store": store,
        "Dept": dept,
        "Size": size,
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
        "CPI": cpi,
        "Unemployment": unemployment,
        "IsHoliday": 1 if is_holiday == "Yes" else 0,
        "Year": year,
        "Month": month,
        "Quarter": quarter,
        "Week": week,
        "Day": day,
        "DayOfWeek": day_of_week,
        "Type": store_type,
    }
    if "IsHoliday_feature" in df.columns:
        input_row["IsHoliday_feature"] = input_row["IsHoliday"]
    if "Store_Type" in df.columns:
        input_row["Store_Type"] = store_type
    if "Store_Size_Category" in df.columns:
        input_row["Store_Size_Category"] = df["Store_Size_Category"].mode()[0]
    input_row.update(markdown_values)

    input_df = pd.DataFrame([input_row])

    # Build the same feature layout the models were trained on
    training_cols_df = preprocess_for_training_columns(
        df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df
    )
    input_processed = preprocess_for_training_columns(input_df)
    input_aligned = input_processed.reindex(columns=training_cols_df.columns, fill_value=0)

    prediction = predict_with_model(best_model, scaler, input_aligned)[0]

    st.divider()
    st.subheader("Prediction Result")
    st.success(f"### Predicted Weekly Sales: **${prediction:,.2f}**")

    with st.expander("Compare predictions across all models", expanded=True):
        rows = []
        for name, model in models.items():
            if model is None:
                continue
            pred = predict_with_model(model, scaler, input_aligned)[0]
            rows.append({"Model": name, "Predicted Weekly Sales": pred})
        if rows:
            results_df = pd.DataFrame(rows).sort_values("Predicted Weekly Sales", ascending=False)
            st.dataframe(
                results_df.style.format({"Predicted Weekly Sales": "${:,.2f}"}),
                use_container_width=True,
            )

    with st.expander("View input features used for prediction"):
        st.dataframe(input_df, use_container_width=True)
