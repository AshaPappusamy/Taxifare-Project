import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from datetime import datetime

# -----------------------------
# Load data and model
# -----------------------------
zones_df = pd.read_csv("Zones.csv")
lookup_df = pd.read_csv("distance_duration_lookup.csv")
model = joblib.load("best_gradient_boosting_model.pkl")

# Expected features for the model
expected_features = model.feature_names_in_

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="🚕 NYC Taxi Fare Predictor",
    layout="wide"
)

# -----------------------------
# Background Image & Button Style
# -----------------------------
page_bg_img = """
<style>
body {
background-image: url("https://images.unsplash.com/photo-1505575967450-2b0b1d9f3de1?auto=format&fit=crop&w=1950&q=80");
background-size: cover;
background-attachment: fixed;
}
.stButton>button {
background-color: #28a745;
color: white;
font-size: 18px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🚕 NYC Taxi Fare Predictor")
st.markdown("Estimate your total fare instantly 💰")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Trip Details")

pickup_zones = zones_df['Zone'].str.strip().unique().tolist()
dropoff_zones = zones_df['Zone'].str.strip().unique().tolist()

pickup_zone = st.sidebar.selectbox("Pickup Zone", pickup_zones)
dropoff_zone = st.sidebar.selectbox("Dropoff Zone", dropoff_zones)
passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)
pickup_datetime = st.sidebar.date_input("Pickup Date", datetime.now().date())
pickup_time = st.sidebar.time_input("Pickup Time", datetime.now().time())

# -----------------------------
# Auto-fill Distance & Duration
# -----------------------------
trip_info = lookup_df[
    (lookup_df['pickup_zone'] == pickup_zone) &
    (lookup_df['dropoff_zone'] == dropoff_zone)
]

if not trip_info.empty:
    trip_distance_km = trip_info['avg_distance_km'].values[0]
    trip_duration_hr = trip_info['avg_duration_hr'].values[0]
else:
    trip_distance_km = 5.0
    trip_duration_hr = 0.5

# -----------------------------
# Feature Engineering
# -----------------------------
pickup_hour = pickup_time.hour
pickup_day = pickup_datetime.day
pickup_month = pickup_datetime.month
pickup_weekday = pickup_datetime.weekday()
is_night = 1 if pickup_hour >= 20 or pickup_hour <= 5 else 0
is_rush_hour = 1 if pickup_hour in [7,8,9,16,17,18] else 0

trip_distance_km_log = math.log(trip_distance_km)
trip_duration_hr_log = math.log(trip_duration_hr)

# -----------------------------
# Prepare Input for Model
# -----------------------------
input_data = pd.DataFrame({
    'trip_duration_hr_log':[trip_duration_hr_log],
    'trip_distance_km_log':[trip_distance_km_log],
    'is_night':[is_night],
    'is_rush_hour':[is_rush_hour],
    'passenger_count':[passenger_count],
    'pickup_hour':[pickup_hour],
    'pickup_day':[pickup_day],
    'pickup_month':[pickup_month],
    'pickup_weekday':[pickup_weekday]
})

# Add missing columns expected by model with 0
for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match model
input_data = input_data[expected_features]

# -----------------------------
# Layout: Map + Bill Summary
# -----------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("### 🗺️ Trip Map")
    
    # Get coordinates from Zones.csv for dynamic pin placement
    try:
        pickup_coords = zones_df[zones_df['Zone']==pickup_zone][['Latitude','Longitude']].values[0]
        dropoff_coords = zones_df[zones_df['Zone']==dropoff_zone][['Latitude','Longitude']].values[0]
    except:
        # fallback if Latitude/Longitude not present
        pickup_coords = [40.7580, -73.9855]
        dropoff_coords = [40.7580, -73.9855]
    
    st.map(pd.DataFrame({
        'lat':[pickup_coords[0], dropoff_coords[0]],
        'lon':[pickup_coords[1], dropoff_coords[1]]
    }), zoom=11, use_container_width=True, height=300)

with col2:
    st.markdown("### 🧾 Trip Summary")
    st.markdown(
        f"""
        <div style='background-color: rgba(255,255,255,0.8); color: black; padding: 15px; border-radius: 10px; font-size:16px;'>
        <b>Pickup Zone:</b> {pickup_zone} <br>
        <b>Dropoff Zone:</b> {dropoff_zone} <br>
        <b>Distance (km):</b> {trip_distance_km:.2f} <br>
        <b>Duration (hr):</b> {trip_duration_hr:.2f} <br>
        <b>Passenger Count:</b> {passenger_count} <br>
        </div>
        """, unsafe_allow_html=True
    )

# -----------------------------
# Predict Button
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("💚 Predict Fare", help="Click to predict the total fare"):
    prediction_log = model.predict(input_data)[0]
    total_fare = math.exp(prediction_log)
    st.success(f"💵 Estimated Total Fare: **${total_fare:.2f}**")
