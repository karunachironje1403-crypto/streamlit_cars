
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_loader import load_data

st.set_page_config(
    page_title="EDA Analysis",
    layout="wide"
)

st.title("Exploratory Data Analysis")

df = load_data()

st.write(
    "Select an analysis type using the buttons below."
)

# Store selected analysis
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = "Univariate"

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Univariate Analysis", use_container_width=True):
        st.session_state.analysis_type = "Univariate"

with col2:
    if st.button("Bivariate Analysis", use_container_width=True):
        st.session_state.analysis_type = "Bivariate"

with col3:
    if st.button("Multivariate Analysis", use_container_width=True):
        st.session_state.analysis_type = "Multivariate"

st.divider()

analysis_type = st.session_state.analysis_type

st.subheader(f"{analysis_type} Analysis")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()


# =====================================================
# UNIVARIATE ANALYSIS
# =====================================================

if analysis_type == "Univariate":

    st.info("Univariate analysis studies one variable at a time.")

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

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.histplot(
            data=df,
            x=column,
            kde=True,
            ax=ax
        )

        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

    elif plot_type == "Box Plot":

        column = st.selectbox(
            "Select Numerical Column",
            numeric_columns
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
            categorical_columns
        )

        top_values = df[column].value_counts().head(15).index

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
            df.columns.tolist()
        )

        value_counts = df[column].value_counts().head(20)

        st.dataframe(
            value_counts.rename("Count"),
            use_container_width=True
        )


# =====================================================
# BIVARIATE ANALYSIS
# =====================================================

elif analysis_type == "Bivariate":

    st.info(
        "Bivariate analysis studies the relationship between two variables."
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
                numeric_columns,
                index=min(1, len(numeric_columns) - 1)
            )

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.scatterplot(
            data=df,
            x=x_column,
            y=y_column,
            ax=ax
        )

        ax.set_title(f"{x_column} vs {y_column}")

        st.pyplot(fig)
        plt.close(fig)

    elif plot_type == "Bar Plot":

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
            categorical_columns
        )

        value_column = st.selectbox(
            "Select Numerical Column",
            numeric_columns
        )

        top_categories = df[category_column].value_counts().head(10).index

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


# =====================================================
# MULTIVARIATE ANALYSIS
# =====================================================

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
        ]
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
            default=numeric_columns[:3]
        )

        if len(selected_columns) >= 2:

            pair_df = df[selected_columns].dropna()

            pair_df = pair_df.sample(
                min(500, len(pair_df)),
                random_state=42
            )

            fig = sns.pairplot(pair_df)

            st.pyplot(fig.figure)
            plt.close(fig.figure)

        else:
            st.warning(
                "Please select at least two numerical columns."
            )

    elif plot_type == "Grouped Bar Plot":

        category_column = st.selectbox(
            "Select Category Column",
            categorical_columns
        )

        value_column = st.selectbox(
            "Select Numerical Column",
            numeric_columns
        )

        second_category = st.selectbox(
            "Select Second Category",
            [
                column for column in categorical_columns
                if column != category_column
            ]
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
            grouped_df[category_column].isin(top_categories)
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