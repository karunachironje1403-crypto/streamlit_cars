
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Household Power Analysis",
    page_icon="🏠",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f4e79;
}

h2, h3 {
    color: #2874a6;
}

div.stButton > button {
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    file_path = (
        Path(__file__).resolve().parent
        / "Household_clean_data.csv"
    )

    if not file_path.exists():
        st.error(f"Dataset not found: {file_path}")
        st.stop()

    return pd.read_csv(file_path)


df = load_data()


# Convert date-related columns if available

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    if "Day_Name" not in df.columns:
        df["Day_Name"] = df["Date"].dt.day_name()

    if "Month_Name" not in df.columns:
        df["Month_Name"] = df["Date"].dt.month_name()

    if "Year" not in df.columns:
        df["Year"] = df["Date"].dt.year


if "Hours" in df.columns:

    df["Hours"] = pd.to_numeric(
        df["Hours"],
        errors="coerce"
    )


numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🏠 Household Analysis")

page = st.sidebar.radio(
    "Select Page",
    [
        "Introduction",
        "EDA Analysis",
        "Conclusions"
    ]
)


# =====================================================
# INTRODUCTION PAGE
# =====================================================

if page == "Introduction":

    st.title("🏠 Household Power Consumption Analysis")

    st.subheader("Introduction and Data Information")

    st.write("""
    This dataset contains household electricity
    consumption and electrical measurements.

    The data includes active power, reactive power,
    voltage, intensity, and sub-metering values.
    """)

    st.divider()

    st.subheader("📊 Dataset Overview")

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

    st.subheader("👀 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("📋 Dataset Information")

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

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        df.describe().transpose(),
        use_container_width=True
    )

    st.subheader("🔢 Numerical Columns")

    st.write(numeric_columns)

    st.subheader("🔤 Categorical Columns")

    st.write(categorical_columns)


# =====================================================
# EDA PAGE
# =====================================================

elif page == "EDA Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Explore the dataset using different visualizations."
    )

    analysis_type = st.radio(
        "Select Analysis Type",
        [
            "Univariate Analysis",
            "Bivariate Analysis",
            "Multivariate Analysis"
        ],
        horizontal=True
    )

    st.divider()


    # =================================================
    # UNIVARIATE ANALYSIS
    # =================================================

    if analysis_type == "Univariate Analysis":

        st.subheader("Univariate Analysis")

        st.info(
            "Analysis of one variable at a time."
        )

        plot_type = st.selectbox(
            "Select Plot",
            [
                "Histogram",
                "Box Plot",
                "Count Plot",
                "Value Counts"
            ]
        )

        if plot_type == "Histogram":

            column = st.selectbox(
                "Select Numerical Column",
                numeric_columns
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            sns.histplot(
                data=df,
                x=column,
                kde=True,
                ax=ax
            )

            ax.set_title(
                f"Distribution of {column}"
            )

            st.pyplot(fig)
            plt.close(fig)


        elif plot_type == "Box Plot":

            column = st.selectbox(
                "Select Numerical Column",
                numeric_columns
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            sns.boxplot(
                data=df,
                x=column,
                ax=ax
            )

            ax.set_title(
                f"Box Plot of {column}"
            )

            st.pyplot(fig)
            plt.close(fig)


        elif plot_type == "Count Plot":

            if categorical_columns:

                column = st.selectbox(
                    "Select Categorical Column",
                    categorical_columns
                )

                top_values = (
                    df[column]
                    .value_counts()
                    .head(15)
                    .index
                )

                filtered_df = df[
                    df[column].isin(top_values)
                ]

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                sns.countplot(
                    data=filtered_df,
                    y=column,
                    order=filtered_df[column]
                    .value_counts()
                    .index,
                    ax=ax
                )

                ax.set_title(
                    f"Count Plot of {column}"
                )

                st.pyplot(fig)
                plt.close(fig)

            else:

                st.warning(
                    "No categorical columns found."
                )


        elif plot_type == "Value Counts":

            column = st.selectbox(
                "Select Column",
                df.columns.tolist()
            )

            counts = (
                df[column]
                .value_counts()
                .head(20)
                .rename("Count")
            )

            st.dataframe(
                counts,
                use_container_width=True
            )


    # =================================================
    # BIVARIATE ANALYSIS
    # =================================================

    elif analysis_type == "Bivariate Analysis":

        st.subheader("Bivariate Analysis")

        st.info(
            "Analysis of the relationship between two variables."
        )

        plot_type = st.selectbox(
            "Select Plot",
            [
                "Scatter Plot",
                "Bar Plot",
                "Box Plot by Category"
            ]
        )


        if plot_type == "Scatter Plot":

            col1, col2 = st.columns(2)

            with col1:

                x_column = st.selectbox(
                    "Select X-Axis",
                    numeric_columns
                )

            with col2:

                y_column = st.selectbox(
                    "Select Y-Axis",
                    numeric_columns
                )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            sns.scatterplot(
                data=df,
                x=x_column,
                y=y_column,
                ax=ax
            )

            ax.set_title(
                f"{x_column} vs {y_column}"
            )

            st.pyplot(fig)
            plt.close(fig)


        elif plot_type == "Bar Plot":

            if categorical_columns:

                category_column = st.selectbox(
                    "Select Category Column",
                    categorical_columns
                )

                value_column = st.selectbox(
                    "Select Numerical Column",
                    numeric_columns
                )

                grouped_df = (
                    df.groupby(category_column)[value_column]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                )

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                grouped_df.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_title(
                    f"Average {value_column} "
                    f"by {category_column}"
                )

                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            else:

                st.warning(
                    "No categorical columns found."
                )


        elif plot_type == "Box Plot by Category":

            if categorical_columns:

                category_column = st.selectbox(
                    "Select Category Column",
                    categorical_columns
                )

                value_column = st.selectbox(
                    "Select Numerical Column",
                    numeric_columns
                )

                top_categories = (
                    df[category_column]
                    .value_counts()
                    .head(10)
                    .index
                )

                filtered_df = df[
                    df[category_column]
                    .isin(top_categories)
                ]

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                sns.boxplot(
                    data=filtered_df,
                    x=category_column,
                    y=value_column,
                    ax=ax
                )

                ax.set_title(
                    f"{value_column} by {category_column}"
                )

                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            else:

                st.warning(
                    "No categorical columns found."
                )


    # =================================================
    # MULTIVARIATE ANALYSIS
    # =================================================

    elif analysis_type == "Multivariate Analysis":

        st.subheader("Multivariate Analysis")

        st.info(
            "Analysis of three or more variables."
        )

        plot_type = st.selectbox(
            "Select Plot",
            [
                "Correlation Heatmap",
                "Pair Plot",
                "Grouped Bar Plot"
            ]
        )


        if plot_type == "Correlation Heatmap":

            correlation = df[numeric_columns].corr()

            fig, ax = plt.subplots(
                figsize=(12, 7)
            )

            sns.heatmap(
                correlation,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax
            )

            ax.set_title(
                "Correlation Heatmap"
            )

            st.pyplot(fig)
            plt.close(fig)


        elif plot_type == "Pair Plot":

            selected_columns = st.multiselect(
                "Select Numerical Columns",
                numeric_columns,
                default=numeric_columns[:3]
            )

            if len(selected_columns) >= 2:

                pair_df = (
                    df[selected_columns]
                    .dropna()
                    .sample(
                        min(500, len(df)),
                        random_state=42
                    )
                )

                pair_plot = sns.pairplot(
                    pair_df
                )

                st.pyplot(pair_plot.figure)
                plt.close(pair_plot.figure)

            else:

                st.warning(
                    "Select at least two numerical columns."
                )


        elif plot_type == "Grouped Bar Plot":

            if len(categorical_columns) >= 2:

                category_column = st.selectbox(
                    "Select First Category",
                    categorical_columns
                )

                remaining_categories = [
                    col for col in categorical_columns
                    if col != category_column
                ]

                second_category = st.selectbox(
                    "Select Second Category",
                    remaining_categories
                )

                value_column = st.selectbox(
                    "Select Numerical Column",
                    numeric_columns
                )

                grouped_df = (
                    df.groupby(
                        [
                            category_column,
                            second_category
                        ]
                    )[value_column]
                    .mean()
                    .reset_index()
                )

                top_categories = (
                    df[category_column]
                    .value_counts()
                    .head(10)
                    .index
                )

                grouped_df = grouped_df[
                    grouped_df[category_column]
                    .isin(top_categories)
                ]

                fig, ax = plt.subplots(
                    figsize=(12, 6)
                )

                sns.barplot(
                    data=grouped_df,
                    x=category_column,
                    y=value_column,
                    hue=second_category,
                    ax=ax
                )

                ax.set_title(
                    f"Average {value_column} by categories"
                )

                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            else:

                st.warning(
                    "At least two categorical columns "
                    "are required."
                )


# =====================================================
# CONCLUSIONS PAGE
# =====================================================

elif page == "Conclusions":

    st.title("📌 Conclusions and Findings")

    st.write("""
    This page summarizes the household electricity
    consumption dataset.
    """)

    st.divider()

    st.subheader("📈 Numerical Summary")

    st.dataframe(
        df[numeric_columns]
        .describe()
        .transpose(),
        use_container_width=True
    )


    # Average active power by hour

    if (
        "Hours" in df.columns
        and "Global_active_power" in df.columns
    ):

        st.subheader(
            "Average Active Power by Hour"
        )

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
                f"Average active power: "
                f"**{highest_value:.2f}**"
            )


    # Total metering analysis

    if "Total_metering" in df.columns:

        st.subheader(
            "⚡ Total Metering Analysis"
        )

        average_metering = (
            df["Total_metering"].mean()
        )

        maximum_metering = (
            df["Total_metering"].max()
        )

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

        st.subheader(
            "🔌 Voltage Analysis"
        )

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


    # Final conclusion

    st.subheader("📝 Final Conclusion")

    st.write("""
    The analysis helps us understand household
    electricity consumption patterns.

    The EDA visualizations can be used to examine:

    1. Distributions of electricity measurements.
    2. Changes in power consumption across hours.
    3. Differences in consumption across days and months.
    4. Relationships between electrical variables.
    5. Correlations among numerical features.

    These findings can help identify patterns
    and possible variations in household
    power consumption.
    """)

    st.success(
        "Household power consumption analysis completed."
    )