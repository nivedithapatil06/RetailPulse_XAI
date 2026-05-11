import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

# LOAD MODEL
model = joblib.load(
    "models/sales_forecast_model.pkl"
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

# WEEKLY SALES TREND
st.subheader("Weekly Sales Trend")

fig, ax = plt.subplots(
    figsize=(10,5)
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
    figsize=(10,6)
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
    figsize=(10,5)
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

filtered_df["Month"] = (
    filtered_df["Date"].dt.month
)

monthly_sales = filtered_df.groupby(
    "Month"
)["Weekly_Sales"].mean()

fig4, ax4 = plt.subplots(
    figsize=(10,5)
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
    [0,1]
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
        "Store":[store],
        "Holiday_Flag":[holiday],
        "Temperature":[temperature],
        "Fuel_Price":[fuel],
        "CPI":[cpi],
        "Unemployment":[unemployment],
        "Year":[year],
        "Month":[month]
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
    figsize=(10,5)
)

scatter = ax5.scatter(
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