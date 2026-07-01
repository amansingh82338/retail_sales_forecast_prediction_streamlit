"""
pages/2_Sales_Trends.py
=========================
Page 2 of the Predictive Sales Forecasting dashboard.
Visualizes sales trends, seasonality, holiday impact, and correlations.
"""

import numpy as np
import plotly.express as px
import streamlit as st

from utils import DATA_PATH, TARGET_COL, check_data_path, load_data

st.set_page_config(page_title="Sales Trends", page_icon="📊", layout="wide")

st.title("📊 Sales Trends & Seasonality")
st.caption("Explore how sales evolve over time and what drives them.")

check_data_path(DATA_PATH)
df = load_data(DATA_PATH)

# --------------------------------------------------------------------------
# FILTERS
# --------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    stores = sorted(df["Store"].dropna().unique()) if "Store" in df.columns else []
    selected_stores = st.multiselect("Filter by Store (optional)", stores, default=[])
with col2:
    depts = sorted(df["Dept"].dropna().unique()) if "Dept" in df.columns else []
    selected_depts = st.multiselect("Filter by Department (optional)", depts, default=[])

filtered_df = df.copy()
if selected_stores:
    filtered_df = filtered_df[filtered_df["Store"].isin(selected_stores)]
if selected_depts:
    filtered_df = filtered_df[filtered_df["Dept"].isin(selected_depts)]

st.divider()

# --------------------------------------------------------------------------
# OVERALL TREND
# --------------------------------------------------------------------------
if "Date" in filtered_df.columns and TARGET_COL in filtered_df.columns:
    st.subheader("Weekly Sales Over Time")
    trend_df = filtered_df.groupby("Date", as_index=False)[TARGET_COL].sum()
    fig = px.line(trend_df, x="Date", y=TARGET_COL, title="Total Weekly Sales Over Time")
    fig.update_traces(line_color="#2E86AB")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# SEASONALITY
# --------------------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    if "Month" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        st.subheader("Average Sales by Month")
        monthly = filtered_df.groupby("Month", as_index=False)[TARGET_COL].mean()
        fig = px.bar(monthly, x="Month", y=TARGET_COL, title="Avg Weekly Sales by Month")
        st.plotly_chart(fig, use_container_width=True)

with col4:
    if "Quarter" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        st.subheader("Average Sales by Quarter")
        quarterly = filtered_df.groupby("Quarter", as_index=False)[TARGET_COL].mean()
        fig = px.bar(quarterly, x="Quarter", y=TARGET_COL, title="Avg Weekly Sales by Quarter")
        st.plotly_chart(fig, use_container_width=True)

col5, col6 = st.columns(2)

with col5:
    if "DayOfWeek" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        st.subheader("Average Sales by Day of Week")
        dow = filtered_df.groupby("DayOfWeek", as_index=False)[TARGET_COL].mean()
        fig = px.bar(dow, x="DayOfWeek", y=TARGET_COL, title="Avg Weekly Sales by Day of Week")
        st.plotly_chart(fig, use_container_width=True)

with col6:
    if "Week" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        st.subheader("Average Sales by Week of Year")
        weekly = filtered_df.groupby("Week", as_index=False)[TARGET_COL].mean()
        fig = px.line(weekly, x="Week", y=TARGET_COL, title="Avg Weekly Sales by Week of Year")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# HOLIDAY IMPACT
# --------------------------------------------------------------------------
st.subheader("Holiday Impact")
col7, col8 = st.columns(2)

with col7:
    if "IsHoliday" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        holiday_avg = filtered_df.groupby("IsHoliday", as_index=False)[TARGET_COL].mean()
        holiday_avg["IsHoliday"] = holiday_avg["IsHoliday"].map(
            {1: "Holiday", 0: "Non-Holiday", True: "Holiday", False: "Non-Holiday"}
        ).fillna(holiday_avg["IsHoliday"])
        fig = px.bar(
            holiday_avg, x="IsHoliday", y=TARGET_COL, color="IsHoliday",
            title="Avg Weekly Sales: Holiday vs Non-Holiday",
        )
        st.plotly_chart(fig, use_container_width=True)

with col8:
    if "IsHoliday" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        fig = px.box(
            filtered_df, x="IsHoliday", y=TARGET_COL,
            title="Weekly Sales Distribution: Holiday vs Non-Holiday",
            points=False,
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# STORE / DEPARTMENT PERFORMANCE
# --------------------------------------------------------------------------
st.subheader("Store & Department Performance")
col9, col10 = st.columns(2)

with col9:
    if "Store" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        store_sales = (
            filtered_df.groupby("Store", as_index=False)[TARGET_COL].sum()
            .sort_values(TARGET_COL, ascending=False).head(10)
        )
        fig = px.bar(store_sales, x="Store", y=TARGET_COL, title="Top 10 Stores by Total Sales")
        st.plotly_chart(fig, use_container_width=True)

with col10:
    if "Dept" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        dept_sales = (
            filtered_df.groupby("Dept", as_index=False)[TARGET_COL].sum()
            .sort_values(TARGET_COL, ascending=False).head(10)
        )
        fig = px.bar(dept_sales, x="Dept", y=TARGET_COL, title="Top 10 Departments by Total Sales")
        st.plotly_chart(fig, use_container_width=True)

if "Type" in filtered_df.columns and TARGET_COL in filtered_df.columns:
    st.subheader("Average Sales by Store Type")
    type_avg = filtered_df.groupby("Type", as_index=False)[TARGET_COL].mean()
    fig = px.bar(type_avg, x="Type", y=TARGET_COL, color="Type", title="Avg Sales by Store Type")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# ECONOMIC FACTORS
# --------------------------------------------------------------------------
st.subheader("Economic Factors vs Sales")
col11, col12 = st.columns(2)

with col11:
    if "Temperature" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        fig = px.scatter(
            filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
            x="Temperature", y=TARGET_COL, opacity=0.4,
            title="Temperature vs Weekly Sales", trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

with col12:
    if "Fuel_Price" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        fig = px.scatter(
            filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
            x="Fuel_Price", y=TARGET_COL, opacity=0.4,
            title="Fuel Price vs Weekly Sales", trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

col13, col14 = st.columns(2)

with col13:
    if "CPI" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        fig = px.scatter(
            filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
            x="CPI", y=TARGET_COL, opacity=0.4,
            title="CPI vs Weekly Sales", trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

with col14:
    if "Unemployment" in filtered_df.columns and TARGET_COL in filtered_df.columns:
        fig = px.scatter(
            filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
            x="Unemployment", y=TARGET_COL, opacity=0.4,
            title="Unemployment vs Weekly Sales", trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------------------------------
st.subheader("Correlation Heatmap")
numeric_df = filtered_df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)