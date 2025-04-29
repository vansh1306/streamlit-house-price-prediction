import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🔍 Exploratory Data Analysis (EDA)")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/house_data.csv")

df = load_data()
st.write("### 📌 First 5 rows of the dataset")
st.dataframe(df.head())

st.write("### 📊 Summary Statistics")
st.write(df.describe())

# Plot
st.write("### 📈 Price Distribution")
fig, ax = plt.subplots()
sns.histplot(df["price"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.write("### 📉 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
