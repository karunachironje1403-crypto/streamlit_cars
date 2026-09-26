
import streamlit as st

st.set_page_config(
    page_title="Cars Data Analysis",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f4e79;
}

h2 {
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

st.title("Cars Data Analysis Dashboard")

st.subheader("Welcome to the Car Dataset Analysis Project")

st.write(
    """
    This application analyzes car data using Python,
    Pandas, NumPy, Seaborn, and Matplotlib.

    Use the sidebar to navigate through:
    - Introduction and data information
    - Exploratory Data Analysis (EDA)
    - Conclusions and findings
    """
)

st.info(
    "Select a page from the sidebar to explore the dataset."
)