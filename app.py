import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import math

# Load the model
model = joblib.load('best_gradient_boosting_model.pkl')

# Page config
st.set_page_config(page_title="🚖 NYC Taxi Fare Predictor", layout="centered")

# Light background
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1508717272800-9fff97da7e8f?auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {background-color: rgba(0,0,0,0);}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.85);
}
h1, h2, h3, p, label {
    color: #1a1a1a;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.title("🚖 NYC Taxi Fare Predictor")
st.markdown("### Estimate your total fare instantly 💰")

# Sidebar inputs
st.sidebar.header("📍 Trip Details")

pickup_zone = st.sidebar.selectbox("Pickup Zone", ["Other", "JFK Airport", "LaGuardia Airport"])
dropoff_zone = st.sidebar.selectbox("Dropoff Zone", ["Other", "JFK Airport", "LaGuardia Airport"])
trip_distance_km = st.sidebar.number_input("Trip Distance (km)", value=5.0, min_value=0.5)
trip_duration_hr = st.sidebar.number_input("Trip Duration (hours)", value=0.5, min_value=0.1)
pickup_date = st.sidebar.date_input("Pickup Date", datetime.now().date())
pickup_time = st.sidebar.time_input("Pickup Time", datetime.now().time())

# Derived features
pickup_hour = pickup_time.hour
is_night = 1 if pickup_hour >= 20 or pickup_hour < 5 else 0
is_rush_hour = 1 if pickup_hour in [7, 8, 9, 16, 17, 18] else 0

# Default assumptions
payment_type_2 = 1  # Credit Card
RatecodeID_2 = 1 if (pickup_zone == "JFK Airport" or dropoff_zone == "JFK Airport") else 0
RatecodeID_5 = 1 if (pickup_zone == "LaGuardia Airport" or dropoff_zone == "LaGuardia Airport") else 0

# Zone encoding
pickup_zone_LGA = 1 if pickup_zone == "LaGuardia Airport" else 0
dropoff_zone_LGA = 1 if dropoff_zone == "LaGuardia Airport" else 0
pickup_zone_JFK = 1 if pickup_zone == "JFK Airport" else 0
dropoff_zone_JFK = 1 if dropoff_zone == "JFK Airport" else 0

# Log transforms
trip_distance_km_log = math.log(trip_distance_km)
trip_duration_hr_log = math.log(trip_duration_hr)

# Input dataframe
input_data = pd.DataFrame([{
    'trip_duration_hr_log': trip_duration_hr_log,
    'trip_distance_km_log': trip_distance_km_log,
    'is_night': is_night,
    'is_rush_hour': is_rush_hour,
    'payment_type_2': payment_type_2,
    'RatecodeID_2': RatecodeID_2,
    'RatecodeID_5': RatecodeID_5,
    'pickup_zone_LaGuardia Airport': pickup_zone_LGA,
    'dropoff_zone_LaGuardia Airport': dropoff_zone_LGA,
    'pickup_zone_JFK Airport': pickup_zone_JFK,
    'dropoff_zone_JFK Airport': dropoff_zone_JFK
}])

st.markdown("#### 🧾 Trip Summary")
st.dataframe(input_data, use_container_width=True)

# Map (optional)
st.map(pd.DataFrame({
    'lat': [40.6413 if pickup_zone == "JFK Airport" else 40.7769 if pickup_zone == "LaGuardia Airport" else 40.7580,
            40.6413 if dropoff_zone == "JFK Airport" else 40.7769 if dropoff_zone == "LaGuardia Airport" else 40.7580],
    'lon': [-73.7781 if pickup_zone == "JFK Airport" else -73.8740 if pickup_zone == "LaGuardia Airport" else -73.9855,
            -73.7781 if dropoff_zone == "JFK Airport" else -73.8740 if dropoff_zone == "LaGuardia Airport" else -73.9855]
}), zoom=11)

# Predict
if st.button("💡 Predict Fare"):
    pred_log = model.predict(input_data)[0]
    total_fare = np.exp(pred_log)

    st.markdown(
        f"""
        <div style='background-color: rgba(255,255,255,0.85); padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 0 10px rgba(0,0,0,0.2);'>
            <h2 style='color: #000;'>🧾 Estimated Taxi Bill</h2>
            <p style='font-size: 20px;'>Pickup: <b>{pickup_zone}</b></p>
            <p style='font-size: 20px;'>Dropoff: <b>{dropoff_zone}</b></p>
            <p style='font-size: 22px; color: #00b300;'>💵 Total Fare: <b>${total_fare:.2f}</b></p>
            <p style='font-size: 14px; color: gray;'>Includes base fare, distance, duration, and timing factors.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.balloons()
