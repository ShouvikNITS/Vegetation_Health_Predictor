# app.py

import ee
import streamlit as st
from utils import quick_setup, train_and_predict

# Initialize Google Earth Engine
@st.cache_resource
def authenticate_gee():
    try:
        credentials = ee.ServiceAccountCredentials(
            st.secrets["GEE_SERVICE_ACCOUNT"],
            key_data=st.secrets["GEE_PRIVATE_KEY"]
        )
        ee.Initialize(credentials)
        st.sidebar.success("✅ GEE authenticated")
    except ee.EEException as e:
        st.sidebar.error(f"❌ GEE init failed: {str(e)}")
        raise

# Prediction wrapper
def predict_health(location):
    predictor = quick_setup(location)
    return train_and_predict(predictor, forecast_days=30)

# App UI
def main():
    # Page config
    st.set_page_config(page_title="Vegetation Health Prediction", page_icon="🛰️", layout="centered")

    # Title
    st.markdown("<h1 style='text-align: center; color: green;'>🛰️ Vegetation Health Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Satellite-powered forecasts of NDVI for any location 🌍</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Authenticate GEE
    authenticate_gee()

    # Sidebar inputs
    st.sidebar.header("📍 Location Input")
    location = st.sidebar.text_input("Enter your location", placeholder="e.g., Bengaluru")

    st.sidebar.markdown("---")
    st.sidebar.caption("ℹ️ Powered by Google Earth Engine + ML")

    # Prediction Button
    if st.sidebar.button("🔍 Predict Vegetation Health"):
        if not location:
            st.warning("⚠️ Please enter a location.")
        else:
            with st.spinner("🔄 Fetching NDVI & weather data..."):
                try:
                    prediction = predict_health([location])
                    st.success("✅ Prediction complete!")
                    st.markdown("### 🌿 Vegetation Forecast (Next 30 Days)")
                    if isinstance(prediction, (dict, list)):
                        st.json(prediction)
                    elif isinstance(prediction, str):
                        st.code(prediction, language='text')
                    else:
                        st.write("🔍 Output:")
                        st.write(prediction)
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")

    # Info section
    with st.expander("ℹ️ What does this app do?"):
        st.markdown("""
        This tool uses NDVI (Normalized Difference Vegetation Index) from **MODIS satellite** data and integrates **NASA POWER weather data**.
        
        It helps forecast vegetation health in a region for the next 30 days using a machine learning model trained on:
        - NDVI trends
        - Temperature, rainfall, humidity

        Ideal for farmers, researchers, and policymakers 👨‍🌾📊🌱
        """)

    st.markdown("---")
    st.caption("© 2025 GreenInnovators ")

if __name__ == '__main__':
    main()
