
import streamlit as st

st.set_page_config(
    page_title="Household Power Analysis",
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

h2, h3 {
    color: #2874a6;
}

div.stButton > button {
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("Household Power Consumption Analysis")

st.subheader("Welcome to the Data Analysis Dashboard")

st.write("""
This application analyzes household electricity
consumption using Pandas, NumPy, Seaborn,
and Matplotlib.

Use the sidebar to explore:
- Introduction and data information
- Exploratory Data Analysis
- Conclusions and findings
""")

st.info("Select a page from the sidebar.")