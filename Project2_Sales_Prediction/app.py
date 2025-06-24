import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")  
st.title("📈 Sales Predictor")

tv = st.slider("TV Budget", 0, 300)
radio = st.slider("Radio Budget", 0, 60)
news = st.slider("Newspaper Budget", 0, 100)

if st.button("Predict Sales"):
    sales = model.predict([[tv, radio, news]])[0]
    st.success(f"Estimated Sales: {sales:.2f} units")
