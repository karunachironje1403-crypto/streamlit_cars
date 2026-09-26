
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
    page_title="Cars Data Analysis",
    page_icon="",
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
    border: 1px solid #2874a6;
    background-color: #2874a6;
    color: white;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #1f4e79;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    file_path = Path(__file__).resolve().parent / "Cars.csv"

    if not file_path.exists():
        st.error(f"Cars.csv not found at: {file_path}")
        st.stop()

    df = pd.read_csv(file_path)

    # Convert numeric columns
    numeric_columns = [
        "Year",
        "Kilometers_Driven",
        "Price",
        "Seats",
        "No. of Doors"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Extract numeric values from text columns
    for column, new_column in [
        ("Mileage", "Mileage_value"),
        ("Engine", "Engine_value"),
        ("Power", "Power_value")
    ]:

        if column in df.columns:

            df[new_column] = pd.to_numeric(
                df[column]
                .astype(str)
                .str.extract(r"([-+]?\d*\.?\d+)")[0],
                errors="coerce"
            )

    return df


df = load_data()


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("Cars Analysis")

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

    st.title("Cars Data Analysis Dashboard")

    st.subheader("Introduction and Data Information")

    st.write("""
    This application analyzes a car dataset using
    Python, Pandas, NumPy, Seaborn, and Matplotlib.

    The dataset contains information about cars,
    including their name, year, fuel type, transmission,
    engine, power, mileage, and price.
    """)

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

    st.subheader("Preview of Dataset")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

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

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    st.subheader("Numerical Columns")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    st.write(numeric_columns)

    st.subheader("Categorical Columns")

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    st.write(categorical_columns)


# =====================================================
# EDA PAGE
# =====================================================

elif page == "EDA Analysis":

    st.title("Exploratory Data Analysis")

    st.write(
        "Choose an analysis type using the buttons below."
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if "analysis_type" not in st.session_state:
        st.session_state.analysis_type = "Univariate"

    # Analysis buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "Univariate Analysis",
            use_container_width=True
        ):
            st.session_state.analysis_type = "Univariate"

    with col2:
        if st.button(
            "Bivariate Analysis",
            use_container_width=True
        ):
            st.session_state.analysis_type = "Bivariate"

    with col3:
        if st.button(
            "Multivariate Analysis",
            use_container_width=True
        ):
            st.session_state.analysis_type = "Multivariate"

    st.divider()

    analysis_type = st.session_state.analysis_type

    st.subheader(f"{analysis_type} Analysis")


    # =================================================
    # UNIVARIATE ANALYSIS
    # =================================================

    if analysis_type == "Univariate":

        st.info(
            "Univariate analysis studies one variable."
        )

        plot_type = st.selectbox(
            "Select Plot",
            [
                "Histogram",
                "Box Plot",
                "Count Plot",
                "Value Counts"
            ],
            key="univariate_plot"
        )

        if plot_type == "Histogram":

            column = st.selectbox(
                "Select Numerical Column",
                numeric_columns,
                key="hist_column"
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.histplot(
                data=df,
                x=column,
                kde=True,
                ax=ax
            )

            ax.set_title(f"Distribution of {column}")

            st.pyplot(fig)
            plt.close(fig)


        elif plot_type == "Box Plot":

            column = st.selectbox(
                "Select Numerical Column",
                numeric_columns,
                key="box_column"
            )

            fig, ax = plt.subplots(figsize=(10, 4))

            sns.boxplot(
                data=df,
                x=column,
                ax=ax
            )

            ax.set_title(f"Box Plot of {column}")

            st.pyplot(fig)
            plt.close(fig)


        elif plot_type == "Count Plot":

            column = st.selectbox(
                "Select Categorical Column",
                categorical_columns,
                key="count_column"
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

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.countplot(
                data=filtered_df,
                y=column,
                order=filtered_df[column].value_counts().index,
                ax=ax
            )

            ax.set_title(f"Count Plot of {column}")

            st.pyplot(fig)
            plt.close(fig)


        elif plot_type == "Value Counts":

            column = st.selectbox(
                "Select Column",
                df.columns.tolist(),
                key="value_column"
            )

            value_counts = (
                df[column]
                .value_counts()
                .head(20)
            )

            st.dataframe(
                value_counts.rename("Count"),
                use_container_width=True
            )


    # =================================================
    # BIVARIATE ANALYSIS
    # =================================================

    elif analysis_type == "Bivariate":

        st.info(
            "Bivariate analysis studies two variables."
        )

        plot_type = st.selectbox(
            "Select Plot",
            [
                "Scatter Plot",
                "Bar Plot",
                "Box Plot by Category"
            ],
            key="bivariate_plot"
        )

        if plot_type == "Scatter Plot":

            col1, col2 = st.columns(2)

            with col1:
                x_column = st.selectbox(
                    "Select X-Axis",
                    numeric_columns,
                    key="x_column"
                )

            with col2:
                y_column = st.selectbox(
                    "Select Y-Axis",
                    numeric_columns,
                    key="y_column"
                )

            fig, ax = plt.subplots(figsize=(10, 5))

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

            category_column = st.selectbox(
                "Select Category Column",
                categorical_columns,
                key="bar_category"
            )

            value_column = st.selectbox(
                "Select Numerical Column",
                numeric_columns,
                key="bar_value"
            )

            grouped_df = (
                df.groupby(category_column)[value_column]
                .mean()
                .sort_values(ascending=False)
                .head(15)
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            grouped_df.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title(
                f"Average {value_column} by {category_column}"
            )

            ax.set_xlabel(category_column)
            ax.set_ylabel(f"Average {value_column}")

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)


        elif plot_type == "Box Plot by Category":

            category_column = st.selectbox(
                "Select Category Column",
                categorical_columns,
                key="category_box"
            )

            value_column = st.selectbox(
                "Select Numerical Column",
                numeric_columns,
                key="category_value"
            )

            top_categories = (
                df[category_column]
                .value_counts()
                .head(10)
                .index
            )

            filtered_df = df[
                df[category_column].isin(top_categories)
            ]

            fig, ax = plt.subplots(figsize=(10, 5))

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


    # =================================================
    # MULTIVARIATE ANALYSIS
    # =================================================

    elif analysis_type == "Multivariate":

        st.info(
            "Multivariate analysis studies three or more variables."
        )

        plot_type = st.selectbox(
            "Select Plot",
            [
                "Correlation Heatmap",
                "Pair Plot",
                "Grouped Bar Plot"
            ],
            key="multivariate_plot"
        )

        if plot_type == "Correlation Heatmap":

            correlation = df[numeric_columns].corr()

            fig, ax = plt.subplots(figsize=(12, 7))

            sns.heatmap(
                correlation,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax
            )

            ax.set_title("Correlation Heatmap")

            st.pyplot(fig)
            plt.close(fig)


        elif plot_type == "Pair Plot":

            selected_columns = st.multiselect(
                "Select Numerical Columns",
                numeric_columns,
                default=numeric_columns[:3],
                key="pair_columns"
            )

            if len(selected_columns) >= 2:

                pair_df = df[
                    selected_columns
                ].dropna()

                pair_df = pair_df.sample(
                    min(500, len(pair_df)),
                    random_state=42
                )

                pair_plot = sns.pairplot(pair_df)

                st.pyplot(pair_plot.figure)
                plt.close(pair_plot.figure)

            else:
                st.warning(
                    "Select at least two numerical columns."
                )


        elif plot_type == "Grouped Bar Plot":

            category_column = st.selectbox(
                "Select Category Column",
                categorical_columns,
                key="group_category"
            )

            value_column = st.selectbox(
                "Select Numerical Column",
                numeric_columns,
                key="group_value"
            )

            remaining_categories = [
                column for column in categorical_columns
                if column != category_column
            ]

            if remaining_categories:

                second_category = st.selectbox(
                    "Select Second Category",
                    remaining_categories,
                    key="second_category"
                )

                grouped_df = (
                    df.groupby(
                        [category_column, second_category]
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

                fig, ax = plt.subplots(figsize=(12, 6))

                sns.barplot(
                    data=grouped_df,
                    x=category_column,
                    y=value_column,
                    hue=second_category,
                    ax=ax
                )

                ax.set_title(
                    f"Average {value_column} by "
                    f"{category_column} and {second_category}"
                )

                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            else:
                st.warning(
                    "At least two categorical columns are required."
                )


# =====================================================
# CONCLUSIONS PAGE
# =====================================================

elif page == "Conclusions":

    st.title("Conclusions and Findings")

    st.write("""
    This page presents a summary of the car dataset
    based on numerical and categorical analysis.
    """)

    st.divider()

    # Numerical summary
    st.subheader("Numerical Findings")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_columns:

        summary = df[
            numeric_columns
        ].describe().transpose()

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

        transmission_counts = (
            df["Transmission"].value_counts()
        )

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

    st.write("""
    The exploratory data analysis helps us understand
    the distribution of car prices, fuel types,
    transmission types, engine characteristics,
    and relationships between numerical variables.

    The visualizations help identify patterns,
    trends, and possible outliers in the dataset.
    """)

    st.success(
        "The car dataset analysis has been completed."
    )