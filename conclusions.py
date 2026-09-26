
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Conclusions",
    layout="wide"
)

st.title("Conclusions and Findings")


@st.cache_data
def load_data():

    file_path = (
        Path(__file__).resolve().parent.parent
        / "Household_clean_data.csv"
    )

    return pd.read_csv(file_path)


df = load_data()


st.write("""
This page summarizes the household electricity
consumption dataset.
""")


st.divider()


# Numerical summary
st.subheader("Numerical Summary")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

st.dataframe(
    df[numeric_columns].describe().transpose(),
    use_container_width=True
)


# Highest average active power by hour
if "Hours" in df.columns and "Global_active_power" in df.columns:

    st.subheader("Average Active Power by Hour")

    hourly_power = (
        df.groupby("Hours")["Global_active_power"]
        .mean()
        .sort_values(ascending=False)
    )

    if not hourly_power.empty:

        highest_hour = hourly_power.index[0]
        highest_value = hourly_power.iloc[0]

        st.write(
            f"Highest average active power hour: "
            f"**{highest_hour}**"
        )

        st.write(
            f"Average active power: **{highest_value:.2f}**"
        )


# Total metering analysis
if "Total_metering" in df.columns:

    st.subheader("Total Metering Analysis")

    average_metering = df["Total_metering"].mean()
    maximum_metering = df["Total_metering"].max()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Average Total Metering",
            f"{average_metering:.2f}"
        )

    with col2:
        st.metric(
            "Maximum Total Metering",
            f"{maximum_metering:.2f}"
        )


# Voltage analysis
if "Voltage" in df.columns:

    st.subheader("Voltage Analysis")

    st.write(
        f"Average voltage: "
        f"**{df['Voltage'].mean():.2f}**"
    )

    st.write(
        f"Minimum voltage: "
        f"**{df['Voltage'].min():.2f}**"
    )

    st.write(
        f"Maximum voltage: "
        f"**{df['Voltage'].max():.2f}**"
    )


# Conclusion
st.subheader("Final Conclusion")

st.write("""
The analysis helps us understand household electricity
consumption patterns.

The EDA visualizations can be used to examine:

1. Distributions of electricity measurements.
2. Changes in power consumption across hours.
3. Differences in consumption across days and months.
4. Relationships between electrical variables.
5. Correlations among numerical features.

These findings can help identify patterns and
possible variations in household power consumption.
""")

st.success(
    "Household power consumption analysis completed."
)