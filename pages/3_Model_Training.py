import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

st.title("📊 Model Training")

df = pd.read_csv("data/processed_house_data.csv")

# Splitting Data
X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "data/house_price_model.pkl")

st.success("✅ Model trained and saved successfully.")
st.write(f"🔹 Model Score: **{model.score(X_test, y_test):.2f}**")
