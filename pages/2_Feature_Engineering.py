import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title("⚙️ Feature Engineering")

df = pd.read_csv("data/house_data.csv")

# Feature scaling
scaler = StandardScaler()
df[["sqft_living", "bedrooms", "bathrooms"]] = scaler.fit_transform(df[["sqft_living", "bedrooms", "bathrooms"]])

st.write("### 🔢 Transformed Data")
st.dataframe(df.head())

# Save processed data
df.to_csv("data/processed_house_data.csv", index=False)

st.success("✅ Data processed and saved successfully.")
