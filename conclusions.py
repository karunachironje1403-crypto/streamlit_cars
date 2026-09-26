
import streamlit as st
import numpy as np
import pandas as pd

from utils.data_loader import load_data

st.set_page_config(
    page_title="Conclusions",
    layout="wide"
)

st.title("Conclusions and Findings")

df = load_data()

st.write(
    """
    This page presents a summary of the car dataset
    based on numerical and categorical analysis.
    """
)

st.divider()

# Numerical summary
st.subheader("Numerical Findings")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if numeric_columns:

    summary = df[numeric_columns].describe().transpose()

    st.dataframe(
        summary,
        use_container_width=True
    )

# Most common fuel type
if "Fuel_Type" in df.columns:

    st.subheader("Most Common Fuel Type")

    fuel_counts = df["Fuel_Type"].value_counts()

    if not fuel_counts.empty:
        st.write(
            f"Most common fuel type: "
            f"**{fuel_counts.index[0]}**"
        )

# Transmission distribution
if "Transmission" in df.columns:

    st.subheader("Transmission Distribution")

    transmission_counts = df["Transmission"].value_counts()

    st.dataframe(
        transmission_counts.rename("Count"),
        use_container_width=True
    )

# Price analysis
if "Price" in df.columns:

    st.subheader("Price Analysis")

    average_price = df["Price"].mean()
    maximum_price = df["Price"].max()
    minimum_price = df["Price"].min()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Price",
            f"{average_price:.2f}"
        )

    with col2:
        st.metric(
            "Maximum Price",
            f"{maximum_price:.2f}"
        )

    with col3:
        st.metric(
            "Minimum Price",
            f"{minimum_price:.2f}"
        )

# Final conclusion
st.subheader("Final Conclusion")

st.write(
    """
    The exploratory data analysis helps us understand
    the distribution of car prices, fuel types,
    transmission types, engine characteristics,
    and relationships between numerical variables.

    The visualizations make it easier to identify
    patterns, trends, and possible outliers in the dataset.
    """
)

st.success(
    "The car dataset analysis has been completed."
)