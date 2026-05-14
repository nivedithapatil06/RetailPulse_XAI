import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# PAGE SETTINGS
st.set_page_config(
    page_title="RetailPulse_XAI",
    layout="wide"
)

# SIDEBAR
st.sidebar.title("RetailPulse")

st.sidebar.info(
    "AI Powered Retail Analytics Dashboard"
)

# LOAD DATA
df = pd.read_csv("data/raw/Walmart.csv")

# LOAD SEGMENT DATA
segment_df = pd.read_csv(
    "data/processed/customer_segments.csv"
)

# CONVERT DATE
df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

# CREATE YEAR & MONTH
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# LOAD MODELS
model = joblib.load(
    "models/sales_forecast_model.pkl"
)

churn_model = joblib.load(
    "models/churn_model.pkl"
)

# STORE FILTER
store_list = df["Store"].unique()

selected_store = st.sidebar.selectbox(
    "Select Store",
    store_list
)

filtered_df = df[
    df["Store"] == selected_store
]

# MAIN TITLE
st.title("RetailPulse Dashboard")

# KPI CARDS
total_sales = filtered_df[
    "Weekly_Sales"
].sum()

avg_sales = filtered_df[
    "Weekly_Sales"
].mean()

max_sales = filtered_df[
    "Weekly_Sales"
].max()

total_records = filtered_df.shape[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Sales",
    f"${total_sales:,.0f}"
)

col2.metric(
    "Average Sales",
    f"${avg_sales:,.0f}"
)

col3.metric(
    "Maximum Sales",
    f"${max_sales:,.0f}"
)

col4.metric(
    "Total Records",
    total_records
)

# DATASET PREVIEW
st.subheader("Dataset Preview")

st.dataframe(
    filtered_df.head()
)

# DOWNLOAD CSV
csv = filtered_df.to_csv(
    index=False
).encode('utf-8')

st.download_button(
    label="Download Data as CSV",
    data=csv,
    file_name='retailpulse_report.csv',
    mime='text/csv',
)

# WEEKLY SALES TREND
st.subheader("Weekly Sales Trend")

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    filtered_df[
        "Weekly_Sales"
    ].values[:100]
)

ax.set_ylabel("Sales")

st.pyplot(fig)

# HEATMAP
st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(
    figsize=(10, 6)
)

sns.heatmap(
    filtered_df.corr(
        numeric_only=True
    ),
    annot=True,
    cmap="coolwarm",
    ax=ax2
)

st.pyplot(fig2)

# TOP PERFORMING STORES
st.subheader(
    "Top Performing Stores"
)

top_stores = df.groupby(
    "Store"
)["Weekly_Sales"].sum()

top_stores = top_stores.sort_values(
    ascending=False
).head(10)

fig3, ax3 = plt.subplots(
    figsize=(10, 5)
)

top_stores.plot(
    kind="bar",
    ax=ax3
)

ax3.set_ylabel("Sales")

st.pyplot(fig3)

# MONTHLY SALES TREND
st.subheader(
    "Monthly Sales Trend"
)

monthly_sales = filtered_df.groupby(
    "Month"
)["Weekly_Sales"].mean()

fig4, ax4 = plt.subplots(
    figsize=(10, 5)
)

monthly_sales.plot(
    kind="line",
    marker="o",
    ax=ax4
)

ax4.set_ylabel(
    "Average Sales"
)

st.pyplot(fig4)

# AI SALES PREDICTION
st.subheader(
    "AI Sales Prediction"
)

store = st.number_input(
    "Store",
    min_value=1,
    max_value=45,
    value=1
)

holiday = st.selectbox(
    "Holiday Flag",
    [0, 1]
)

temperature = st.number_input(
    "Temperature",
    value=25.0
)

fuel = st.number_input(
    "Fuel Price",
    value=2.5
)

cpi = st.number_input(
    "CPI",
    value=200.0
)

unemployment = st.number_input(
    "Unemployment",
    value=8.0
)

year = st.number_input(
    "Year",
    value=2012
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=1
)

if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "Store": [store],
        "Holiday_Flag": [holiday],
        "Temperature": [temperature],
        "Fuel_Price": [fuel],
        "CPI": [cpi],
        "Unemployment": [unemployment],
        "Year": [year],
        "Month": [month]
    })

    prediction = model.predict(
        input_data
    )

    st.success(
        f"Predicted Weekly Sales: ${prediction[0]:,.2f}"
    )

# CUSTOMER SEGMENTATION
st.subheader(
    "Customer Segmentation"
)

fig5, ax5 = plt.subplots(
    figsize=(10, 5)
)

ax5.scatter(
    segment_df["Store"],
    segment_df["TotalSales"],
    c=segment_df["Cluster"]
)

ax5.set_xlabel("Store")

ax5.set_ylabel("Total Sales")

ax5.set_title(
    "Customer Segments"
)

st.pyplot(fig5)

# CUSTOMER CHURN PREDICTION
st.subheader(
    "Customer Churn Prediction"
)

churn_store = st.number_input(
    "Churn Store",
    min_value=1,
    max_value=45,
    value=1
)

churn_holiday = st.selectbox(
    "Churn Holiday Flag",
    [0, 1]
)

churn_temperature = st.number_input(
    "Churn Temperature",
    value=25.0
)

churn_fuel = st.number_input(
    "Churn Fuel Price",
    value=2.5
)

churn_cpi = st.number_input(
    "Churn CPI",
    value=200.0
)

churn_unemployment = st.number_input(
    "Churn Unemployment",
    value=8.0
)

if st.button(
    "Predict Churn"
):

    churn_input = pd.DataFrame({
        "Store": [churn_store],
        "Holiday_Flag": [churn_holiday],
        "Temperature": [churn_temperature],
        "Fuel_Price": [churn_fuel],
        "CPI": [churn_cpi],
        "Unemployment": [churn_unemployment]
    })

    churn_prediction = churn_model.predict(
        churn_input
    )

    if churn_prediction[0] == 1:

        st.error(
            "High Churn Risk Detected"
        )

    else:

        st.success(
            "Low Churn Risk"
        )

# INVENTORY OPTIMIZATION
st.subheader(
    "Inventory Optimization"
)

current_stock = st.number_input(
    "Current Stock",
    min_value=0,
    value=500
)

predicted_demand = st.number_input(
    "Predicted Demand",
    min_value=0,
    value=700
)

safety_stock = st.number_input(
    "Safety Stock",
    min_value=0,
    value=100
)

recommended_order = (
    predicted_demand
    + safety_stock
    - current_stock
)

if recommended_order < 0:
    recommended_order = 0

st.info(
    f"Recommended Reorder Quantity: {recommended_order} units"
)

if current_stock < predicted_demand:

    st.warning(
        "Stock level is lower than expected demand."
    )

else:

    st.success(
        "Inventory level is sufficient."
    )

# BUSINESS INSIGHTS
st.subheader(
    "Business Insights"
)

top_store = top_stores.index[0]

st.success(
    f"Top Performing Store: {top_store}"
)

highest_sales = df[
    "Weekly_Sales"
].max()

st.info(
    f"Highest Weekly Sales Recorded: ${highest_sales:,.0f}"
)

holiday_sales = df.groupby(
    "Holiday_Flag"
)["Weekly_Sales"].mean()

if holiday_sales[1] > holiday_sales[0]:

    st.warning(
        "Holiday periods generate higher sales."
    )

else:

    st.warning(
        "Non-holiday periods generate higher sales."
    )

# EXPLAINABLE AI (XAI)
st.subheader("Explainable AI Insights")

try:

    sample_data = pd.DataFrame({
        "Store": [store],
        "Holiday_Flag": [holiday],
        "Temperature": [temperature],
        "Fuel_Price": [fuel],
        "CPI": [cpi],
        "Unemployment": [unemployment],
        "Year": [year],
        "Month": [month]
    })

    explainer = shap.Explainer(
        model.predict,
        sample_data
    )

    shap_values = explainer(
        sample_data
    )

    st.write(
        "Feature Impact on Prediction"
    )

    fig_shap, ax_shap = plt.subplots(
        figsize=(10, 5)
    )

    shap.plots.bar(
        shap_values,
        show=False
    )

    st.pyplot(fig_shap)

except Exception as e:

    st.warning(
        f"SHAP explanation unavailable: {e}"
    )

# MODEL PERFORMANCE METRICS

st.subheader(
    "Model Performance Metrics"
)

try:

    X_metrics = df[[
        "Store",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Year",
        "Month"
    ]]

    y_metrics = df[
        "Weekly_Sales"
    ]

    predictions_metrics = model.predict(
        X_metrics
    )

    mae = mean_absolute_error(
        y_metrics,
        predictions_metrics
    )

    rmse = mean_squared_error(
    y_metrics,
    predictions_metrics
    ) ** 0.5

    r2 = r2_score(
        y_metrics,
        predictions_metrics
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:,.2f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

    col3.metric(
        "R² Score",
        f"{r2:.2f}"
    )

except Exception as e:

    st.warning(
        f"Metrics unavailable: {e}"
    )
