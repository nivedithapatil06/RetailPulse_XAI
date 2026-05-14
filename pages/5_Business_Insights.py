import streamlit as st
import pandas as pd
import plotly.express as px

# PAGE CONFIG
st.set_page_config(
    page_title="Business Insights",
    layout="wide"
)

# TITLE
st.title("Business Insights Dashboard")

st.info(
    "Retail business intelligence and analytics insights"
)

# LOAD DATA
df = pd.read_csv(
    "data/raw/Walmart.csv"
)

# KPI SECTION
total_sales = df[
    "Weekly_Sales"
].sum()

avg_sales = df[
    "Weekly_Sales"
].mean()

max_sales = df[
    "Weekly_Sales"
].max()

col1, col2, col3 = st.columns(3)

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

# TOP STORES
st.subheader(
    "Top Performing Stores"
)

top_stores = df.groupby(
    "Store"
)["Weekly_Sales"].sum()

top_stores = top_stores.sort_values(
    ascending=False
).head(10)

fig = px.bar(
    x=top_stores.index,
    y=top_stores.values,
    labels={
        "x": "Store",
        "y": "Sales"
    },
    title="Top Performing Stores"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# HOLIDAY INSIGHTS
st.subheader(
    "Holiday Sales Insights"
)

holiday_sales = df.groupby(
    "Holiday_Flag"
)["Weekly_Sales"].mean()

holiday_df = pd.DataFrame({
    "Holiday": [
        "Non-Holiday",
        "Holiday"
    ],
    "Sales": holiday_sales.values
})

fig2 = px.pie(
    holiday_df,
    names="Holiday",
    values="Sales",
    title="Holiday vs Non-Holiday Sales"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# BUSINESS RECOMMENDATIONS
st.subheader(
    "Business Recommendations"
)

top_store = top_stores.index[0]

st.success(
    f"Top Performing Store: {top_store}"
)

if holiday_sales[1] > holiday_sales[0]:

    st.info(
        "Holiday periods generate higher revenue."
    )

else:

    st.info(
        "Non-holiday periods generate higher revenue."
    )

st.warning(
    "Maintain optimized inventory during high-demand periods."
)