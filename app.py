import streamlit as st
import numpy as np
import pickle
import os
import gdown

st.set_page_config(
    page_title="Food Delivery Prediction",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food Delivery Time & Status Prediction")

# -----------------------------
# Download model files from Google Drive
# -----------------------------
FILES = {
    "rf_classifier.pkl": "1otEPEgitpI6cE-lZ-VBC6RFBg1gGGLQ-",
    "rf_regressor.pkl": "18WRFPr0PTTZLwkLaiVauY-TiXLJSxfYk",
    "scaler.pkl": "1CL6jhzYxCnudb_6_QGos7Z9W1L99oNAe"
}

def download_models():
    for filename, file_id in FILES.items():
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {filename}..."):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, filename, quiet=False)

download_models()

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    with open("rf_regressor.pkl", "rb") as f:
        reg_model = pickle.load(f)

    with open("rf_classifier.pkl", "rb") as f:
        class_model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return reg_model, class_model, scaler

reg_model, class_model, scaler = load_models()

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input(
    "Delivery Person Age",
    min_value=15,
    max_value=70,
    value=30
)

ratings = st.slider(
    "Delivery Person Ratings",
    min_value=1.0,
    max_value=5.0,
    value=4.5,
    step=0.1
)

weather = st.selectbox(
    "Weather Conditions",
    ["Sunny", "Stormy", "Sandstorms", "Windy", "Fog", "Cloudy"]
)

weather_map = {
    "Sunny": 0,
    "Stormy": 1,
    "Sandstorms": 2,
    "Windy": 3,
    "Fog": 4,
    "Cloudy": 5
}

traffic = st.selectbox(
    "Road Traffic Density",
    ["Low", "Medium", "High", "Jam"]
)

traffic_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Jam": 3
}

vehicle_cond = st.selectbox(
    "Vehicle Condition",
    [0, 1, 2, 3]
)

vehicle_type = st.selectbox(
    "Type of Vehicle",
    ["motorcycle", "scooter", "electric_scooter", "bicycle"]
)

vehicle_map = {
    "motorcycle": 0,
    "scooter": 1,
    "electric_scooter": 2,
    "bicycle": 3
}

multiple_del = st.number_input(
    "Multiple Deliveries",
    min_value=0,
    max_value=5,
    value=0
)

festival = st.selectbox(
    "Festival?",
    ["No", "Yes"]
)

festival_map = {
    "No": 0,
    "Yes": 1
}

city = st.selectbox(
    "City Type",
    ["Urban", "Semi-Urban", "Metropolitan"]
)

city_map = {
    "Urban": 0,
    "Semi-Urban": 1,
    "Metropolitan": 2
}

distance = st.number_input(
    "Distance (meters)",
    min_value=100,
    max_value=50000,
    value=3000
)

order_hour = st.slider(
    "Order Hour",
    0,
    23,
    19
)

day_of_week = st.slider(
    "Day of Week (0=Mon, 6=Sun)",
    0,
    6,
    4
)

prep_time = st.number_input(
    "Preparation Time (minutes)",
    min_value=5,
    max_value=60,
    value=15
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict", type="primary"):

    features = np.array([[
        age,
        ratings,
        weather_map[weather],
        traffic_map[traffic],
        vehicle_cond,
        vehicle_map[vehicle_type],
        multiple_del,
        festival_map[festival],
        city_map[city],
        distance,
        order_hour,
        day_of_week,
        prep_time
    ]])

    features_scaled = scaler.transform(features)

    pred_time = reg_model.predict(features_scaled)[0]
    pred_status = class_model.predict(features_scaled)[0]

    st.markdown("---")

    st.subheader(f"⏱️ Estimated Delivery Time: {pred_time:.2f} minutes")

    if pred_status == 1:
        st.success("✅ Prediction: On-Time Delivery")
    else:
        st.error("🚨 Prediction: Delayed Delivery")
