import streamlit as st
import pandas as pd
import plotly.express as px

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

# TITLE
st.title("Customer Segmentation")

st.info(
    "Customer clustering using Machine Learning"
)

# LOAD DATA
segment_df = pd.read_csv(
    "data/processed/customer_segments.csv"
)

# DATA PREVIEW
st.subheader(
    "Segment Data Preview"
)

st.dataframe(
    segment_df.head()
)

# SEGMENTATION CHART
st.subheader(
    "Customer Segment Visualization"
)

fig = px.scatter(
    segment_df,
    x="Store",
    y="TotalSales",
    color="Cluster",
    title="Customer Segments"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# CLUSTER INSIGHTS
st.subheader(
    "Cluster Insights"
)

total_clusters = segment_df[
    "Cluster"
].nunique()

st.success(
    f"Total Customer Segments Identified: {total_clusters}"
)

largest_cluster = segment_df[
    "Cluster"
].value_counts().idxmax()

st.info(
    f"Largest Cluster: {largest_cluster}"
)