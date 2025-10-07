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

# Light background image
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
.main-title {
    background-color: rgba(255, 255, 255, 0.8);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
}
h1, h2, h3, p, label {
    color: #1a1a1a;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Wrap your title in a visible box
st.markdown("""
<div class='main-title'>
    <h1>🚖 NYC Taxi Fare Predictor</h1>
    <h3>Estimate your total fare instantly 💰</h3>
</div>
""", unsafe_allow_html=True)
