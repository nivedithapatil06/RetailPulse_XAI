import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# PAGE CONFIG
st.set_page_config(
    page_title="Sales Forecasting",
    layout="wide"
)

# TITLE
st.title("AI Sales Forecasting")

st.info(
    "Predict future retail sales using Machine Learning"
)

# LOAD DATA
df = pd.read_csv(
    "data/raw/Walmart.csv"
)

# LOAD MODEL
model = joblib.load(
    "models/sales_forecast_model.pkl"
)

# DATE PROCESSING
df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# SALES TREND
st.subheader(
    "Sales Trend Analysis"
)

fig = px.line(
    df.head(200),
    y="Weekly_Sales",
    title="Weekly Sales Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# FORECAST INPUTS
st.subheader(
    "Predict Sales"
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

# PREDICTION
if st.button(
    "Forecast Sales"
):

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