import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components  # <-- Added this import

# Configure the page layout
st.set_page_config(page_title="Smart Leak Detection Sim", layout="centered")

# Load the trained ML model
@st.cache_resource
def load_model():
    return joblib.load('water_leak_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'water_leak_model.pkl' not found. Please ensure the training script ran successfully.")
    st.stop()

st.title("💧 Smart Water Leakage Detection System")
st.markdown("### Edge ML Anomaly Simulator")
st.write("Adjust the sensor values below to test the live ML model's detection logic.")

st.divider()

# Create columns for the sliders
col1, col2, col3 = st.columns(3)

with col1:
    flow = st.slider("Water Flow Rate (L/min)", min_value=0, max_value=100, value=50)

with col2:
    pressure = st.slider("Internal Pressure (kPa)", min_value=0, max_value=500, value=300)

with col3:
    moisture = st.slider("Soil Moisture (%)", min_value=0, max_value=100, value=20)

st.divider()

# --- Machine Learning Prediction ---
input_data = pd.DataFrame({
    'Flow_Rate': [flow],
    'Pressure': [pressure],
    'Soil_Moisture': [moisture]
})

# Get prediction and the confidence probability from the ML model
prediction = model.predict(input_data)[0]  
probabilities = model.predict_proba(input_data)[0] 
leak_probability = int(probabilities[1] * 100) 

# --- Visual Pipe Simulation (HTML/CSS) ---
st.subheader("Live System Status")

# Using the dedicated components.html instead of st.markdown
if prediction == 1:
    # LEAKING STATE ANIMATION
    html_code = """
    <div style="text-align: center; padding-top: 10px;">
        <div style="width: 100%; height: 60px; background: linear-gradient(to bottom, #777, #999, #777); border-radius: 5px; position: relative; border: 2px solid #444; box-shadow: 0px 5px 15px rgba(0,0,0,0.3);">
            <div style="width: 100%; height: 40px; background-color: #3498db; margin-top: 8px; opacity: 0.6;"></div>
            
            <div style="position: absolute; top: 58px; left: 48%; width: 10px; height: 25px; background-color: #3498db; border-radius: 50%; animation: drop 0.6s infinite ease-in;"></div>
            <div style="position: absolute; top: 58px; left: 52%; width: 8px; height: 20px; background-color: #3498db; border-radius: 50%; animation: drop 0.8s infinite ease-in;"></div>
            <div style="position: absolute; top: 58px; left: 50%; width: 12px; height: 28px; background-color: #3498db; border-radius: 50%; animation: drop 0.7s infinite ease-in;"></div>
        </div>
        <style>
            @keyframes drop {
                0% { transform: translateY(0px) scaleY(1); opacity: 1; }
                80% { transform: translateY(60px) scaleY(1.5); opacity: 0.8; }
                100% { transform: translateY(80px) scaleY(0.5); opacity: 0; }
            }
        </style>
    </div>
    """
    components.html(html_code, height=150) # Isolates HTML so it won't bleed
else:
    # NORMAL STATE
    html_code = """
    <div style="text-align: center; padding-top: 10px;">
        <div style="width: 100%; height: 60px; background: linear-gradient(to bottom, #777, #999, #777); border-radius: 5px; position: relative; border: 2px solid #444; box-shadow: 0px 5px 15px rgba(0,0,0,0.3);">
            <div style="width: 100%; height: 40px; background-color: #3498db; margin-top: 8px; opacity: 0.9;"></div>
        </div>
    </div>
    """
    components.html(html_code, height=100) 


# --- Dashboard Metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Flow", value=f"{flow} L/min")
m2.metric(label="Pressure", value=f"{pressure} kPa")
m3.metric(label="Moisture", value=f"{moisture} %")
m4.metric(label="Leak Prob.", value=f"{leak_probability}%")

st.write("") # Spacer

# Dynamic Alert System based on ML Prediction
if prediction == 1:
    st.error(f"🚨 CRITICAL: LEAK DETECTED! (Probability: {leak_probability}%)")
    st.write("Action: ML model triggered anomaly alert. Shutting down localized valve.")
elif leak_probability >= 40:
    st.warning(f"⚠️ WARNING: System Anomaly. Inspect lines. (Probability: {leak_probability}%)")
else:
    st.success(f"✅ System Optimal. No anomalies detected. (Probability: {leak_probability}%)")