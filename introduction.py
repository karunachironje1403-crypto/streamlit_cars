
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_data


st.set_page_config(
    page_title="Introduction",
    layout="wide"
)


st.title("Introduction and Data Information")


# Load dataset
df = load_data()


st.write(
    """
    The dataset contains information about cars,
    including their name, year, fuel type, transmission,
    engine, power, mileage, and price.
    """
)

st.divider()


# Dataset metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rows", df.shape[0])

with col2:
    st.metric("Total Columns", df.shape[1])

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )


st.divider()


# Preview of dataset
st.subheader("Preview of Dataset")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# Dataset information
st.subheader("Dataset Information")

info_df = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

st.dataframe(
    info_df,
    use_container_width=True
)


# Statistical summary
st.subheader("Statistical Summary")

st.dataframe(
    df.describe(include="all").transpose(),
    use_container_width=True
)


# Numerical columns
st.subheader("Numerical Columns")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

st.write(numeric_columns)


# Categorical columns
st.subheader("Categorical Columns")

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

st.write(categorical_columns)