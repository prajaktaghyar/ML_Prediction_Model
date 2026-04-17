import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import json

# --- Load Model and Data ---

# Load the best model
# Assuming 'random_forest_model.pkl' was saved correctly
random_forest_model = joblib.load('random_forest_model.pkl')

# To correctly encode 'Job Title', we need the original label_encoder
# and the unique Job Titles it was fitted on.
# In a real Streamlit app, you would save/load the label_encoder or its mappings.

# Re-create a label encoder and fit it to the original Job Titles for consistent encoding
# (This is important if the app runs independently of the notebook's prior execution)
original_df = pd.read_csv('/content/Salary Data (2).csv') # Reload original data for Job Title mapping
original_df = original_df.dropna(subset=['Job Title']) # Drop NaNs for fitting
job_title_encoder = LabelEncoder()
job_title_encoder.fit(original_df['Job Title'].astype(str)) # Fit on string type to avoid errors

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Salary Prediction App")

st.title("Salary Prediction")

st.sidebar.header("Input Features")

# Input widgets
age = st.sidebar.slider("Age", 18, 70, 30)
gender = st.sidebar.selectbox("Gender", options={'Male': 1, 'Female': 0}, format_func=lambda x: 'Male' if x == 1 else 'Female')
education_level = st.sidebar.selectbox("Education Level", options={
    "Bachelor's": 0,
    "Master's": 1,
    "PhD": 2
})
# Using a selectbox for Job Title, with all unique job titles from the training data
unique_job_titles = sorted(original_df['Job Title'].dropna().unique().tolist())
job_title_raw = st.sidebar.selectbox("Job Title", unique_job_titles)
years_experience = st.sidebar.slider("Years of Experience", 0.0, 50.0, 5.0, step=0.5)

# --- Prediction Logic ---
if st.sidebar.button("Predict Salary"):
    try:
        encoded_job_title = job_title_encoder.transform([job_title_raw])[0]
    except ValueError:
        st.error(f"Job Title '{job_title_raw}' not recognized. Please select from the list.")
        encoded_job_title = -1 # Indicate error or use a default

    if encoded_job_title != -1:
        prediction_input = pd.DataFrame([[
            age,
            gender,
            education_level,
            encoded_job_title,
            years_experience
        ]],
        columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
        )

        predicted_salary = random_forest_model.predict(prediction_input)[0]

        st.success(f"### Predicted Salary: ${predicted_salary:,.2f}")
