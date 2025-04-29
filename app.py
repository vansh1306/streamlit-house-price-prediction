import streamlit as st

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 House Price Prediction using Machine Learning")

st.sidebar.success("Select a page from the navigation menu.")
st.write("""
Welcome to the **House Price Prediction App**.  
This app provides the following features:
1. **Exploratory Data Analysis (EDA)**
2. **Feature Engineering**
3. **Model Training**
4. **House Price Prediction**
5. **Summary & Insights**
""")
