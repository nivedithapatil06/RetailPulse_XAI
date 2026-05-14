import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIG
st.set_page_config(
    page_title="Churn Prediction",
    layout="wide"
)

# TITLE
st.title("Customer Churn Prediction")

st.info(
    "Predict customers at risk of churn"
)

# LOAD MODEL
churn_model = joblib.load(
    "models/churn_model.pkl"
)

# INPUTS
st.subheader(
    "Customer Details"
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

# PREDICTION
if st.button(
    "Predict Churn"
):

    churn_input = pd.DataFrame({
        "Store": [store],
        "Holiday_Flag": [holiday],
        "Temperature": [temperature],
        "Fuel_Price": [fuel],
        "CPI": [cpi],
        "Unemployment": [unemployment]
    })

    prediction = churn_model.predict(
        churn_input
    )

    if prediction[0] == 1:

        st.error(
            "High Churn Risk Detected"
        )

        st.warning(
            "Customer retention strategies recommended."
        )

    else:

        st.success(
            "Low Churn Risk"
        )

        st.info(
            "Customer retention level is healthy."
        )