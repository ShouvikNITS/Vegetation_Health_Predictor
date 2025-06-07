# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 10:51:26 2025

@author: Shouvik
"""

import ee
import streamlit as st
from utils import quick_setup, train_and_predict

@st.cache_resource
def authenticate_gee():
    try:
        credentials = ee.ServiceAccountCredentials(
            st.secrets["GEE_SERVICE_ACCOUNT"],
            key_data=st.secrets["GEE_PRIVATE_KEY"]  # Path to JSON key file
        )
        ee.Initialize(credentials)
        st.write("GEE initialized successfully.")
    except ee.EEException as e:
        st.error(f"GEE initialization failed: {str(e)}")
        raise

def predict_health(location):
    
    predictor = quick_setup(location)
    
    return train_and_predict(predictor, forecast_days=30)
    

def main():
    #giving title
    st.title('Vegetation Health Prediction')
    
    #Getting the input data from user
    authenticate_gee()
    Location = st.text_input('Enter your location: ')
    
    #code for prediction
    Prediction = ''
    
    # creating a button
    if st.button('Predict Vegetation Health'):
        Prediction = predict_health([Location])
        
    st.success(Prediction)
    
if __name__ == '__main__':
    main()