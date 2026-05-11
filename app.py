import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# SIDEBAR
st.sidebar.title("RetailPulse")

st.sidebar.info(
    "AI Powered Retail Analytics Dashboard"
)

# TITLE
st.title("RetailPulse Dashboard")

# LOAD DATA
df = pd.read_csv("data/raw/Walmart.csv")

# SHOW DATA
st.subheader("Dataset Preview")
st.dataframe(df.head())

# SALES TREND
st.subheader("Weekly Sales Trend")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df["Weekly_Sales"][:100])

st.pyplot(fig)

# HEATMAP
st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            ax=ax2)

st.pyplot(fig2)