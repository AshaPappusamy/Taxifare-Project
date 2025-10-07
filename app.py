import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Load the saved model
model = joblib.load('best_gradient_boosting_model.pkl')

st.set_page_config(page_title="🚕 NYC Taxi Fare Predictor", layout="wide")

st.title("🚕 NYC Taxi Fare Prediction App")
st.markdown("Predict your total cab fare using our trained Gradient Boosting model 💰")

# Sidebar for user input
st.sidebar.header("Enter Trip Details")

pickup_longitude = st.sidebar.number_input("Pickup Longitude", value=-73.985)
pickup_latitude = st.sidebar.number_input("Pickup Latitude", value=40.758)
dropoff_longitude = st.sidebar.number_input("Dropoff Longitude", value=-73.985)
dropoff_latitude = st.sidebar.number_input("Dropoff Latitude", value=40.761)
passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)

pickup_datetime = st.sidebar.date_input("Pickup Date", datetime.now().date())
pickup_time = st.sidebar.time_input("Pickup Time", datetime.now().time())

# Feature engineering (if your model used time-based features)
pickup_hour = pickup_time.hour
pickup_day = pickup_datetime.day
pickup_month = pickup_datetime.month
pickup_weekday = pickup_datetime.weekday()

# Combine all features into a DataFrame
input_data = pd.DataFrame({
    'pickup_longitude': [pickup_longitude],
    'pickup_latitude': [pickup_latitude],
    'dropoff_longitude': [dropoff_longitude],
    'dropoff_latitude': [dropoff_latitude],
    'passenger_count': [passenger_count],
    'pickup_hour': [pickup_hour],
    'pickup_day': [pickup_day],
    'pickup_month': [pickup_month],
    'pickup_weekday': [pickup_weekday],
})

st.subheader("🧾 Input Summary")
st.write(input_data)

if st.button("💡 Predict Fare"):
    prediction_log = model.predict(input_data)[0]
    total_fare = np.exp(prediction_log)  # reverse log-transform
    st.success(f"💵 Estimated Total Fare: **${total_fare:.2f}**")
