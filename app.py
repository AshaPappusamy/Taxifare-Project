import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import math

# -------------------------
# Load Zones and Model
# -------------------------
zones_df = pd.read_csv("Zones.csv")  # Make sure this file is in repo root
pickup_zones = zones_df['Zone'].str.strip().unique().tolist()
dropoff_zones = pickup_zones.copy()

model = joblib.load("best_gradient_boosting_model.pkl")

# -------------------------
# Page Config & Background
# -------------------------
st.set_page_config(page_title="🚖 NYC Taxi Fare Predictor", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1564709016983-56c1d12b0d61?auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {background-color: rgba(0,0,0,0);}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.95);
    color: black;
}
.stButton>button {
    background-color: #00b300;
    color: white;
    font-weight: bold;
    height: 50px;
    width: 100%;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover {
    background-color: #009900;
}
h1,h2,h3,p,label {
    color: #1a1a1a !important;
}
.main-container {
    background-color: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(0,0,0,0.3);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.markdown("""
<div style='background-color: rgba(255,255,255,0.85); padding: 20px; border-radius: 12px; text-align: center;'>
    <h1>🚖 NYC Taxi Fare Predictor</h1>
    <h3>Estimate your total fare instantly 💰</h3>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("📍 Trip Details")
pickup_zone = st.sidebar.selectbox("Pickup Zone", options=pickup_zones)
dropoff_zone = st.sidebar.selectbox("Dropoff Zone", options=dropoff_zones)
trip_distance_km = st.sidebar.number_input("Trip Distance (km)", value=5.0, min_value=0.5)
trip_duration_hr = st.sidebar.number_input("Trip Duration (hours)", value=0.5, min_value=0.1)
pickup_date = st.sidebar.date_input("Pickup Date", datetime.now().date())
pickup_time = st.sidebar.time_input("Pickup Time", datetime.now().time())

# -------------------------
# Feature Engineering
# -------------------------
pickup_hour = pickup_time.hour
is_night = 1 if pickup_hour >= 20 or pickup_hour < 5 else 0
is_rush_hour = 1 if pickup_hour in [7,8,9,16,17,18] else 0

payment_type_2 = 1
RatecodeID_2 = 1 if (pickup_zone == "JFK Airport" or dropoff_zone == "JFK Airport") else 0
RatecodeID_5 = 1 if (pickup_zone == "LaGuardia Airport" or dropoff_zone == "LaGuardia Airport") else 0

pickup_zone_LGA = 1 if pickup_zone == "LaGuardia Airport" else 0
dropoff_zone_LGA = 1 if dropoff_zone == "LaGuardia Airport" else 0
pickup_zone_JFK = 1 if pickup_zone == "JFK Airport" else 0
dropoff_zone_JFK = 1 if dropoff_zone == "JFK Airport" else 0

trip_distance_km_log = math.log(trip_distance_km)
trip_duration_hr_log = math.log(trip_duration_hr)

# -------------------------
# Prepare Input Data
# -------------------------
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

# -------------------------
# Layout: Small Map & Bill Side-by-Side
# -------------------------
col1, col2 = st.columns([1,1])

# Map (small placeholder just for visualization)
with col1:
    st.markdown("### 🗺️ Trip Map")
    st.map(pd.DataFrame({
        'lat': [40.7580, 40.7580],  # placeholder
        'lon': [-73.9855, -73.9855]  # placeholder
    }), zoom=12, use_container_width=True, height=200)

# Bill Summary
with col2:
    st.markdown("### 🧾 Trip Bill")
    st.dataframe(input_data, use_container_width=True)

# -------------------------
# Predict Button
# -------------------------
if st.button("💡 Predict Fare"):
    pred_log = model.predict(input_data)[0]
    total_fare = np.exp(pred_log)

    st.markdown(
        f"""
        <div style='background-color: rgba(255,255,255,0.95); padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 0 10px rgba(0,0,0,0.2); margin-top: 10px;'>
            <h2>🧾 Estimated Taxi Bill</h2>
            <p style='font-size: 20px;'>Pickup: <b>{pickup_zone}</b></p>
            <p style='font-size: 20px;'>Dropoff: <b>{dropoff_zone}</b></p>
            <p style='font-size: 24px; color: #00b300;'>💵 Total Fare: <b>${total_fare:.2f}</b></p>
            <p style='font-size: 14px; color: gray;'>Includes base fare, distance, duration, and timing factors.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.balloons()
