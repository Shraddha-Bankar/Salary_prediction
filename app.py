import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
try:
    model = joblib.load('random_forest_regressor_model.joblib')
except FileNotFoundError:
    st.error("Model file 'random_forest_regressor_model.joblib' not found. Please ensure it's saved in the same directory.")
    st.stop()

st.title('Salary Prediction App')
st.write('Enter the feature values to predict the salary.')

# Input features (assuming they are already label-encoded for simplicity)
# In a real application, you would load and use the LabelEncoders for categorical inputs.
rating = st.number_input('Rating (Encoded)', min_value=0, value=26)
company_name = st.number_input('Company Name (Encoded)', min_value=0, value=8129)
jop_title = st.number_input('Job Title (Encoded)', min_value=0, value=29)
salaries_reported = st.number_input('Salaries Reported (Encoded)', min_value=0, value=35)
location = st.number_input('Location (Encoded)', min_value=0, value=0)
employment_status = st.number_input('Employment Status (Encoded)', min_value=0, value=1)
job_roles = st.number_input('Job Roles (Encoded)', min_value=0, value=10)

if st.button('Predict Salary'):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': company_name,
        'Job Title': jop_title,
        'Salaries Reported': salaries_reported,
        'Location': location,
        'Employment Status': employment_status,
        'Job Roles': job_roles
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: {prediction:,.2f} INR')
