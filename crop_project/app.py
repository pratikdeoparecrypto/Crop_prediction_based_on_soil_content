# Import required libraries
import streamlit as st        # Used to create web app UI
import pickle                # Used to load saved model
import numpy as np           # Used for numerical operations

# -------------------------------
# Load trained model and encoder
# -------------------------------
# model.pkl → trained ML model
# encoder.pkl → converts numbers back to crop names
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌱"
)

# -------------------------------
# Title and description
# -------------------------------
st.title("🌱 Crop Recommendation System")

st.write("Enter soil and environmental details:")

# -------------------------------
# User Inputs (features)
# -------------------------------
# These values are taken from user and passed to model

N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)

temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Crop"):

    # Convert input values into array format (required for model)
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Predict crop (gives numeric output)
    prediction = model.predict(data)

    # Convert numeric output back to crop name
    crop = le.inverse_transform(prediction)

    # Display result
    st.success(f"🌾 Recommended Crop: {crop[0]}")