import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🏡 House Price Prediction")

# Load Model
model = joblib.load("data/house_price_model.pkl")

# User Input
sqft_living = st.number_input("Enter square footage:", min_value=400, max_value=10000, step=50)
bedrooms = st.slider("Number of bedrooms:", 1, 10, 3)
bathrooms = st.slider("Number of bathrooms:", 1, 5, 2)

if st.button("Predict Price"):
    input_data = np.array([[sqft_living, bedrooms, bathrooms]])
    predicted_price = model.predict(input_data)
    st.success(f"🏡 Estimated House Price: **${predicted_price[0]:,.2f}**")
