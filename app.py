import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Food Delivery Prediction", page_icon="🍔", layout="centered")
st.title("🍔 Food Delivery Time & Status Prediction")

@st.cache_resource
def load_models():
    reg_model, class_model, scaler = None, None, None
    if os.path.exists('rf_regressor.pkl'):
        with open('rf_regressor.pkl', 'rb') as f: reg_model = pickle.load(f)
    if os.path.exists('rf_classifier.pkl'):
        with open('rf_classifier.pkl', 'rb') as f: class_model = pickle.load(f)
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    return reg_model, class_model, scaler

reg_model, class_model, scaler = load_models()

age = st.number_input("Delivery Person Age", min_value=15, max_value=70, value=30)
ratings = st.slider("Delivery Person Ratings", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
weather = st.selectbox("Weather Conditions", ["Sunny", "Stormy", "Sandstorms", "Windy", "Fog", "Cloudy"])
weather_map = {"Sunny": 0, "Stormy": 1, "Sandstorms": 2, "Windy": 3, "Fog": 4, "Cloudy": 5}
traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
vehicle_cond = st.selectbox("Vehicle Condition", [0, 1, 2, 3])
vehicle_type = st.selectbox("Type of Vehicle", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
vehicle_map = {"motorcycle": 0, "scooter": 1, "electric_scooter": 2, "bicycle": 3}
multiple_del = st.number_input("Multiple Deliveries", min_value=0, max_value=5, value=0)
festival = st.selectbox("Festival?", ["No", "Yes"])
festival_map = {"No": 0, "Yes": 1}
city = st.selectbox("City Type", ["Urban", "Semi-Urban", "Metropolitan"])
city_map = {"Urban": 0, "Semi-Urban": 1, "Metropolitan": 2}
distance = st.number_input("Distance (in meters)", min_value=100, max_value=50000, value=3000)
order_hour = st.slider("Order Hour (0-23)", min_value=0, max_value=23, value=19)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=4)
prep_time = st.number_input("Preparation Time (min)", min_value=5, max_value=60, value=15)

if st.button("Predict", type="primary"):
    if reg_model and class_model and scaler:
        features = np.array([[
            age, ratings, weather_map[weather], traffic_map[traffic], 
            vehicle_cond, vehicle_map[vehicle_type], multiple_del, 
            festival_map[festival], city_map[city], distance, 
            order_hour, day_of_week, prep_time
        ]])
        features_scaled = scaler.transform(features)
        pred_time = reg_model.predict(features_scaled)[0]
        pred_status = class_model.predict(features_scaled)[0]
        st.markdown("---")
        st.subheader(f"⏱️ Estimated Time: {pred_time:.2f} minutes")
        if pred_status == 1: st.success("✅ Prediction: On-Time Delivery")
        else: st.error("🚨 Prediction: Delayed Delivery")
    else:
        st.warning("⚠️ Model files (.pkl) missing in the folder!")