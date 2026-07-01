"""
pages/1_Data_Explorer.py
=========================
Page 1 of the Predictive Sales Forecasting dashboard.
Lets users browse, filter, and download the merged sales dataset.
"""

import streamlit as st

from utils import DATA_PATH, check_data_path, get_dataset, using_uploaded_data

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")

st.title("🔍 Data Explorer")
st.caption("Browse and filter the merged sales dataset.")

check_data_path(DATA_PATH)
df = get_dataset()

if using_uploaded_data():
    st.info("Using your uploaded dataset. Upload or reset it from the home page.")

# --------------------------------------------------------------------------
# FILTERS
# --------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    stores = sorted(df["Store"].dropna().unique()) if "Store" in df.columns else []
    selected_stores = st.multiselect("Store", stores, default=[])

with col2:
    depts = sorted(df["Dept"].dropna().unique()) if "Dept" in df.columns else []
    selected_depts = st.multiselect("Department", depts, default=[])

with col3:
    holiday_filter = st.selectbox("Holiday weeks only?", ["All", "Yes", "No"])

filtered_df = df.copy()

if selected_stores:
    filtered_df = filtered_df[filtered_df["Store"].isin(selected_stores)]

if selected_depts:
    filtered_df = filtered_df[filtered_df["Dept"].isin(selected_depts)]

if holiday_filter != "All" and "IsHoliday" in filtered_df.columns:
    want = True if holiday_filter == "Yes" else False
    filtered_df = filtered_df[
        filtered_df["IsHoliday"].astype(str).str.lower().isin(
            [str(want).lower(), "1" if want else "0"]
        )
    ]

# --------------------------------------------------------------------------
# KEY METRICS
# --------------------------------------------------------------------------
if "Weekly_Sales" in filtered_df.columns:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{len(filtered_df):,}")
    m2.metric("Total Sales", f"${filtered_df['Weekly_Sales'].sum():,.0f}")
    m3.metric("Avg Weekly Sales", f"${filtered_df['Weekly_Sales'].mean():,.2f}")
    m4.metric("Max Weekly Sales", f"${filtered_df['Weekly_Sales'].max():,.2f}")

# --------------------------------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------------------------------
st.subheader("Summary Statistics")
st.dataframe(filtered_df.describe(), use_container_width=True)

# --------------------------------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------------------------------
st.subheader(f"Dataset Preview ({len(filtered_df):,} rows)")
st.dataframe(filtered_df.head(500), use_container_width=True)

st.download_button(
    "Download filtered data as CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales_data.csv",
    mime="text/csv",
)
